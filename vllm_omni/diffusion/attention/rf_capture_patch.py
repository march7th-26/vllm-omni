# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache MindIE-SD RainFusion geometry during eager warmup for NPUGraph.

The stock permutation uses boolean indexing, and the mask builder copies
protected block indices from the host. Both depend only on the packed video
layout, so they can be prepared once before capture.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_FAILED = False


def ensure_rf_capture_patch() -> bool:
    """Patch MindIE-SD's multi-video preprocessing. Idempotent.

    Returns True once the patch is in place, False when it could not be applied
    (in which case callers keep working -- capture simply stays unavailable for
    the multi-video path).
    """
    global _PATCHED, _FAILED
    if _PATCHED:
        return True
    if _FAILED:
        return False

    try:
        import mindiesd.layers.flash_attn.sparse_flash_attn as sparse_entry
        import mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 as rf_v2
        import mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 as rf_v3
    except Exception as exc:  # noqa: BLE001 - absent MindIE-SD must not break the caller
        _FAILED = True
        logger.warning("rf capture patch skipped: cannot import MindIE-SD sparse attention (%s)", exc)
        return False

    permutation = getattr(rf_v2, "_multi_span_permutation", None)
    mask = getattr(rf_v2, "get_multi_span_blockwise_mask", None)
    if (
        not callable(permutation)
        or not callable(mask)
        or not callable(getattr(rf_v2, "get_mask_index", None))
        or getattr(sparse_entry, "get_multi_span_blockwise_mask", None) is not mask
        or getattr(rf_v3, "get_multi_span_blockwise_mask", None) is not mask
    ):
        _FAILED = True
        logger.warning("rf capture patch skipped: MindIE-SD does not expose the expected RFv2/RFv3 helpers")
        return False

    # The two callers import the mask helper by name. The permutation is read
    # directly from the RFv2 module by the shared pooling helper.
    patched_mask = _make_span_blockwise_mask(rf_v2)
    rf_v2._multi_span_permutation = _cache_multi_span_permutation(permutation)
    rf_v2.get_multi_span_blockwise_mask = patched_mask
    sparse_entry.get_multi_span_blockwise_mask = patched_mask
    rf_v3.get_multi_span_blockwise_mask = patched_mask

    _PATCHED = True
    logger.info("rf capture patch applied: RFv2/RFv3 cache video permutations and protected block indices")
    return True


def _cache_multi_span_permutation(permutation):
    """Reuse the original permutation for the same layout and device."""
    cached = lru_cache(maxsize=32)(permutation)

    def _multi_span_permutation(spans, sequence_len, pool_size, input_layout, device):
        geometry = tuple((start, tuple(shape), length) for start, shape, length in spans)
        return cached(geometry, sequence_len, pool_size, input_layout, device)

    return _multi_span_permutation


def _make_span_blockwise_mask(rf_v2):
    """Same mask; the retained-block index tensor is built once, not per call."""
    import math

    get_mask_index = rf_v2.get_mask_index
    retained_cache: dict[tuple[int, ...], object] = {}

    def get_multi_span_blockwise_mask(
        qkv_pool, sparsity, scale, dense_blocks, first_frame_blocks, input_layout, return_binary=False
    ):
        query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
        if input_layout == "BSND":
            scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
        else:
            scores = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
        score_matrix = torch.nn.functional.softmax(scores, dim=-1)
        keep_len = max(1, math.ceil(score_matrix.shape[-1] * (1 - sparsity)))
        threshold = torch.topk(score_matrix, k=keep_len, dim=-1).values[..., -1:]
        mask = score_matrix >= threshold
        retained = sorted(dense_blocks | first_frame_blocks)
        if retained:
            cache_key = tuple(retained)
            retained_idx = retained_cache.get(cache_key)
            if retained_idx is None or retained_idx.device != mask.device:
                retained_idx = torch.tensor(retained, device=mask.device, dtype=torch.long)
                retained_cache[cache_key] = retained_idx
            mask[:, :, retained_idx, :] = True
            mask[:, :, :, retained_idx] = True
        if return_binary:
            return mask.to(torch.int8)
        select_idx = get_mask_index(mask)[0].transpose(0, 1)
        select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
        return select_idx, select_num_idx

    return get_multi_span_blockwise_mask
