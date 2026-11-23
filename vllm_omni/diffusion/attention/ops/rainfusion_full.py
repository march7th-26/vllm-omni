# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph-safe preparation for full-length, multi-span RainFusion mix on A5."""

from dataclasses import dataclass
from functools import lru_cache

import torch
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
    _multi_span_permutation,
    _normalize_image_spans,
    _normalize_video_spans,
    avgpool,
)

from vllm_omni.diffusion.attention.ops.rainfusion_rectangular import (
    apply_image_sparsity,
    native_mix,
    rank_blocks,
)


@dataclass(frozen=True)
class FullPlan:
    permutation: torch.Tensor
    inverse: torch.Tensor
    protected: torch.Tensor
    image_query_blocks: torch.Tensor
    image_groups: tuple[torch.Tensor, ...]


@lru_cache(maxsize=16)
def get_full_plan(
    rows: int,
    video_spans: tuple[tuple[int, tuple[int, ...]], ...],
    image_spans: tuple[tuple[int, tuple[int, ...]], ...],
    device: str,
) -> FullPlan:
    def descriptors(spans):
        return [{"start": start, "latent_shape": list(grid)} for start, grid in spans]

    permutation, inverse, dense, first, (images, eligible) = _multi_span_permutation(
        _normalize_video_spans(descriptors(video_spans), rows),
        rows,
        128,
        "BSND",
        "cpu",
        image_spans=_normalize_image_spans(descriptors(image_spans), rows) if image_spans else [],
        return_reference_plan=True,
    )
    # The official RF-v3 rules stay the same. Materialize only its
    # geometry-dependent indices before capture to avoid host transfers or
    # constructing new index tensors during replay.
    protected = dense | first
    return FullPlan(
        permutation.to(device),
        inverse.to(device),
        torch.tensor(sorted(protected), dtype=torch.long, device=device),
        torch.tensor(sorted(set(eligible) - protected), dtype=torch.long, device=device),
        tuple(torch.tensor(group, dtype=torch.long, device=device) for group in images if group),
    )


def run_full(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: FullPlan,
    sparsity: float,
    scale: float,
    image_sparsity: float,
) -> torch.Tensor:
    used = plan.permutation.numel()
    q = query[:, :used].index_select(1, plan.permutation)
    k = key[:, :used].index_select(1, plan.permutation)
    v = value[:, :used].index_select(1, plan.permutation)
    # RF-v3 mix pools all three packed operands together before ranking Q/K.
    query_pool, key_pool, _ = avgpool(torch.cat((q, k, v), dim=0), 128, "BSND").chunk(3, dim=0)
    scores, mask = rank_blocks(query_pool, key_pool, sparsity, scale)
    if plan.protected.numel():
        mask[:, :, plan.protected, :] = True
        mask[:, :, :, plan.protected] = True
    apply_image_sparsity(scores, mask, plan.image_query_blocks, plan.image_groups, image_sparsity)
    out = native_mix(q, k, v, mask.to(torch.int8), scale).index_select(1, plan.inverse)
    return torch.cat((out, query[:, used:]), dim=1) if query.shape[1] > used else out
