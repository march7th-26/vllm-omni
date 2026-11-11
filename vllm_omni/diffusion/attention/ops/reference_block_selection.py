# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Map per-image RBS ownership to compact RainFusion query blocks."""

from functools import lru_cache


def require_rbs() -> None:
    try:
        from mindiesd.layers.flash_attn.ref_sparse_mask import apply_ref_mask, build_ref_plan
    except ImportError as exc:
        raise ImportError(
            "ref_keep_ratio < 1 requires an RBS-capable MindIE-SD build. See recipes/h3_ref_optimizations/README.md."
        ) from exc
    if not callable(build_ref_plan) or not callable(apply_ref_mask):
        raise TypeError("MindIE-SD RBS helpers must be callable")


@lru_cache(maxsize=32)
def rectangular_plan(
    kv_rows: int,
    reference_rows: int,
    spans: tuple[tuple[int, tuple[int, ...]], ...],
    device: str,
    images: tuple[tuple[int, int], ...],
    image_positions: tuple[tuple[int, ...], ...],
    targets: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    require_rbs()
    from mindiesd.layers.flash_attn.ref_sparse_mask import build_ref_plan

    from vllm_omni.diffusion.attention.ops.rainfusion_rectangular import get_plan

    plan = get_plan(kv_rows, reference_rows, spans, device)
    groups, rows = build_ref_plan(
        plan.key_indices.cpu().tolist(),
        images,
        targets,
        128,
        set(plan.protected_keys.cpu().tolist()),
        image_positions,
    )
    lookup = {block: index for index, block in enumerate(plan.sparse_blocks)}
    return groups, tuple(lookup[block] for block in rows if block in lookup)
