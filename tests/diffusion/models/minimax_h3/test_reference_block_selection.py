# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of RBS configuration and model-owned token geometry.

Load the production definitions without importing vLLM or NPU libraries. These
checks do not execute or simulate the native attention kernels.
"""

import ast
import dataclasses
import importlib.util
import math
import sys
from bisect import bisect_left
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def definitions(name: str, relative: str, names: set[str], dependencies: dict) -> ModuleType:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ModuleType(name)
    module.__dict__.update(dependencies)
    sys.modules[name] = module
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), relative, "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture
def config():
    return definitions(
        "_rbs_attention_config",
        "vllm_omni/diffusion/data.py",
        {
            "parse_kv_cache_skip_selector",
            "_in_range",
            "RainFusionPrecision",
            "BlockSparseSpec",
            "SkipSoftmaxSpec",
            "AttnQuantSpec",
            "AttentionSpec",
            "AttentionConfig",
        },
        {
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "Enum": Enum,
            "Mapping": Mapping,
            "math": math,
            "BLOCK_SPARSE_BACKENDS": frozenset({"RAINFUSION_ATTN"}),
        },
    )


@pytest.fixture
def geometry():
    layouts = definitions(
        "_rbs_layout",
        "vllm_omni/diffusion/attention/backends/abstract.py",
        {"VideoTokenSpan", "VideoTokenLayout"},
        {"dataclass": dataclasses.dataclass},
    )
    return definitions(
        "_rbs_geometry",
        "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
        {"_reference_kv_video_layout"},
        {
            "VideoTokenSpan": layouts.VideoTokenSpan,
            "VideoTokenLayout": layouts.VideoTokenLayout,
            "math": math,
            "bisect_left": bisect_left,
        },
    )


@pytest.fixture
def backend():
    return definitions(
        "_rbs_backend",
        "vllm_omni/diffusion/attention/backends/rainfusion_attn.py",
        {"RainFusionConfig", "RainFusionPlan", "RainFusionAttentionImpl"},
        {
            "dataclass": dataclasses.dataclass,
            "AttentionImpl": object,
            "math": math,
            "logger": Mock(),
            "_MIN_VIDEO_BLOCKS": 32,
            "_BLOCK_SIZE": 128,
        },
    )


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8, 1.0])
def test_ratio_reaches_backend_from_standard_config(config, backend, ratio):
    parsed = config.AttentionConfig(
        default={
            "backend": "RAINFUSION_ATTN",
            "block_sparse": {"sparsity": 0.8, "precision": "mix", "ref_keep_ratio": ratio},
        }
    )
    kwargs = parsed.default.backend_kwargs()
    assert backend.RainFusionConfig.from_backend_kwargs(kwargs).ref_keep_ratio == ratio
    if ratio == 1.0:
        assert "ref_keep_ratio" not in kwargs


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_is_rejected(config, backend, ratio):
    for cls in (config.BlockSparseSpec, backend.RainFusionConfig):
        with pytest.raises(ValueError, match="ref_keep_ratio"):
            cls(sparsity=0.8, ref_keep_ratio=ratio)


def test_rbs_requires_sparse_attention(config, backend):
    for cls in (config.BlockSparseSpec, backend.RainFusionConfig):
        with pytest.raises(ValueError, match="positive block sparsity"):
            cls(sparsity=0, ref_keep_ratio=0.5)
    assert config.BlockSparseSpec(sparsity=0).ref_keep_ratio == 1.0


def test_reference_images_follow_cached_prefix_positions(geometry):
    layout = geometry.VideoTokenLayout(
        used_len=36,
        reference_image_spans=((2, 4), (8, 10)),
        video_spans=(
            geometry.VideoTokenSpan(start=16, latent_grid=(1, 2, 2), role="reference"),
            geometry.VideoTokenSpan(start=20, latent_grid=(4, 2, 2), role="target"),
        ),
    )
    # Other cached context makes the second image non-contiguous in the prefix.
    positions = torch.tensor([0, 2, 3, 6, 8, 9, 16, 17, 18, 19])
    compact = geometry.VideoTokenLayout(
        used_len=26, video_spans=(geometry.VideoTokenSpan(start=10, latent_grid=(4, 2, 2), role="target"),)
    )
    remapped = geometry._reference_kv_video_layout(layout, positions, compact)
    assert remapped.reference_image_positions == ((1, 2), (4, 5))
    assert [(span.start, span.role) for span in remapped.video_spans] == [(6, "reference"), (20, "target")]


def test_partially_cached_image_is_rejected(geometry):
    original = geometry.VideoTokenLayout(reference_image_spans=((2, 5),))
    compact = geometry.VideoTokenLayout(used_len=10)
    with pytest.raises(ValueError, match="not fully represented"):
        geometry._reference_kv_video_layout(original, torch.tensor([2, 4]), compact)


def test_full_query_plan_preserves_per_image_and_target_ownership(backend, geometry, monkeypatch):
    bridge = ModuleType("vllm_omni.diffusion.attention.ops.reference_block_selection")
    bridge.__dict__["require_rbs"] = Mock()
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    impl = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8, ref_keep_ratio=0.2)
    layout = geometry.VideoTokenLayout(
        used_len=8192,
        reference_image_spans=((0, 256), (384, 640)),
        video_spans=(geometry.VideoTokenSpan(start=1024, latent_grid=(4, 32, 56), role="target"),),
    )
    plan = impl._resolve_multi_span_plan(layout, 8192)
    assert plan.video_spans[0]["_ref_keep_ratio"] == 0.2
    assert plan.video_spans[0]["_ref_images"] == [(0, 256), (384, 640)]
    assert plan.video_spans[0]["_ref_image_positions"] == []
    assert plan.video_spans[0]["_ref_targets"] == [(1024, 8192)]
    bridge.require_rbs.assert_called_once_with()
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8)
    assert "_ref_keep_ratio" not in impl._resolve_multi_span_plan(layout, 8192).video_spans[0]
    bridge.require_rbs.assert_called_once_with()


@pytest.fixture
def selection():
    path = ROOT / "vllm_omni/diffusion/attention/ops/reference_block_selection.py"
    spec = importlib.util.spec_from_file_location("_rbs_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_mindiesd_rbs_helper_raises(selection, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.layers.flash_attn.ref_sparse_mask", None)
    with pytest.raises(ImportError, match="RBS-capable MindIE-SD"):
        selection.require_rbs()


def test_rectangular_query_blocks_are_remapped_and_plan_is_cached(selection, monkeypatch):
    geometry = definitions(
        "_rbs_rectangular_geometry",
        "vllm_omni/diffusion/attention/ops/research_rectangular_bf16.py",
        {"Plan"},
        {"dataclass": dataclasses.dataclass},
    )
    plan = geometry.Plan(
        key_indices=torch.arange(896),
        sparse_indices=torch.arange(384),
        dense_indices=torch.arange(128),
        protected_keys=torch.tensor([0, 1]),
        sparse_blocks=(2, 4, 6),
        sparse_rows=384,
        dense_rows=128,
        key_rows=896,
    )
    native_geometry = ModuleType("vllm_omni.diffusion.attention.ops.research_rectangular_bf16")
    native_geometry.__dict__["get_plan"] = Mock(return_value=plan)
    helper = ModuleType("mindiesd.layers.flash_attn.ref_sparse_mask")
    helper.__dict__["build_ref_plan"] = Mock(return_value=(((0, 1),), (4, 6)))
    helper.__dict__["apply_ref_mask"] = Mock()
    monkeypatch.setitem(sys.modules, native_geometry.__name__, native_geometry)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    args = (896, 384, ((384, (4, 8, 16)),), "cpu", (), ((0, 2, 4, 6),), ((384, 896),))
    assert selection.rectangular_plan(*args) == (((0, 1),), (1, 2))
    assert selection.rectangular_plan(*args) == (((0, 1),), (1, 2))
    helper.build_ref_plan.assert_called_once_with(list(range(896)), (), ((384, 896),), 128, {0, 1}, ((0, 2, 4, 6),))
    native_geometry.get_plan.assert_called_once_with(*args[:4])
