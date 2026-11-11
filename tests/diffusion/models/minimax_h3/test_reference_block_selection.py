# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of RBS configuration and model-owned token geometry.

Load the production definitions without importing vLLM or NPU libraries. These
checks do not execute or simulate the native attention kernels.
"""

import ast
import dataclasses
import functools
import importlib.util
import inspect
import math
import sys
from bisect import bisect_left
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import ModuleType

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
            "RainFusionSpec",
            "SkipSoftmaxSpec",
            "AttnQuantSpec",
            "SpargeAttnSpec",
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
        {"VideoTokenSpan", "VideoTokenLayout", "AttentionMetadata"},
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )
    return definitions(
        "_rbs_geometry",
        "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
        {"_reference_kv_video_layout"},
        {
            "VideoTokenSpan": layouts.VideoTokenSpan,
            "VideoTokenLayout": layouts.VideoTokenLayout,
            "AttentionMetadata": layouts.AttentionMetadata,
            "math": math,
            "bisect_left": bisect_left,
        },
    )


@pytest.fixture
def progress():
    @dataclasses.dataclass
    class Progress:
        denoise_step_idx: int | None = 0
        total_denoise_steps: int | None = 8

    return Progress()


@pytest.fixture
def backend(progress, mocker):
    return definitions(
        "_rbs_backend",
        "vllm_omni/diffusion/attention/backends/rainfusion_attn.py",
        {"RainFusionConfig", "RainFusionPlan", "RainFusionAttentionImpl", "_supports_video_spans"},
        {
            "dataclass": dataclasses.dataclass,
            "replace": dataclasses.replace,
            "AttentionImpl": object,
            "math": math,
            "torch": torch,
            "cast": lambda annotation, value: value,
            "AttentionMetadata": object,
            "functools": functools,
            "inspect": inspect,
            "get_forward_context": lambda: progress,
            "is_forward_context_available": lambda: True,
            "_mindiesd_supports_precision": lambda: True,
            "_INPUT_LAYOUT": "BSND",
            "_INNER_PRECISE": 0,
            "logger": mocker.Mock(),
            "_MIN_VIDEO_BLOCKS": 32,
            "_BLOCK_SIZE": 128,
        },
    )


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8, 1.0])
def test_ratio_reaches_backend_from_standard_config(config, backend, ratio):
    parsed = config.AttentionConfig(
        default={
            "backend": "RAINFUSION_ATTN",
            "block_sparse": {"sparsity": 0.8, "precision": "mix"},
            "rainfusion": {"ref_keep_ratio": ratio},
        }
    )
    kwargs = parsed.default.backend_kwargs()
    assert backend.RainFusionConfig.from_backend_kwargs(kwargs).ref_keep_ratio == ratio
    if ratio == 1.0:
        assert "ref_keep_ratio" not in kwargs


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_is_rejected(config, backend, ratio):
    for cls in (config.RainFusionSpec, backend.RainFusionConfig):
        with pytest.raises(ValueError, match="ref_keep_ratio"):
            cls(ref_keep_ratio=ratio)


def test_rbs_requires_sparse_attention(config, backend):
    with pytest.raises(ValueError, match="positive block sparsity"):
        config.AttentionSpec(
            backend="RAINFUSION_ATTN", block_sparse={"sparsity": 0}, rainfusion={"ref_keep_ratio": 0.5}
        )
    with pytest.raises(ValueError, match="positive block sparsity"):
        backend.RainFusionConfig(sparsity=0, ref_keep_ratio=0.5)
    assert config.RainFusionSpec().ref_keep_ratio == 1.0


def test_rainfusion_options_reject_unrelated_backend(config):
    with pytest.raises(ValueError, match="only supported"):
        config.AttentionSpec(backend="FLASH_ATTN", rainfusion={"ref_keep_ratio": 0.5})


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


def test_full_query_plan_preserves_per_image_and_target_ownership(backend, geometry, monkeypatch, mocker):
    bridge = ModuleType("vllm_omni.diffusion.attention.ops.reference_block_selection")
    bridge.__dict__["require_rbs"] = mocker.Mock()
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


def test_rectangular_query_blocks_are_remapped_and_plan_is_cached(selection, monkeypatch, mocker):
    geometry = definitions(
        "_rbs_rectangular_geometry",
        "vllm_omni/diffusion/attention/ops/rainfusion_rectangular.py",
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
    native_geometry = ModuleType("vllm_omni.diffusion.attention.ops.rainfusion_rectangular")
    native_geometry.__dict__["get_plan"] = mocker.Mock(return_value=plan)
    helper = ModuleType("mindiesd.layers.flash_attn.ref_sparse_mask")
    helper.__dict__["build_ref_plan"] = mocker.Mock(return_value=(((0, 1),), (4, 6)))
    helper.__dict__["apply_ref_mask"] = mocker.Mock()
    monkeypatch.setitem(sys.modules, native_geometry.__name__, native_geometry)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    args = (896, 384, ((384, (4, 8, 16)),), "cpu", (), ((0, 2, 4, 6),), ((384, 896),))
    assert selection.rectangular_plan(*args) == (((0, 1),), (1, 2))
    assert selection.rectangular_plan(*args) == (((0, 1),), (1, 2))
    helper.build_ref_plan.assert_called_once_with(list(range(896)), (), ((384, 896),), 128, {0, 1}, ((0, 2, 4, 6),))
    native_geometry.get_plan.assert_called_once_with(*args[:4])


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_default_compact_keeps_existing_path(backend, geometry, monkeypatch, mocker, precision):
    calls = []

    def sparse_attention(q, k, v, *, video_spans, **kwargs):
        calls.append((q.shape[1], k.shape[1]))
        return torch.zeros_like(q)

    native = ModuleType("mindiesd")
    native.__dict__["sparse_attention"] = sparse_attention
    platform = ModuleType("mindiesd.utils.get_platform")
    platform.__dict__["is_a5_device"] = lambda: True
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    instance = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    instance.rainfusion = backend.RainFusionConfig(precision=precision)
    instance.qkv_layout = "BSND"
    instance.softmax_scale = 0.25
    instance._forward_rectangular_npu = mocker.Mock(side_effect=AssertionError("default route changed"))
    instance._resolve_plan = lambda metadata: backend.RainFusionPlan(
        used_len=12, video_spans=[{"start": 4, "latent_shape": [1, 1, 8]}]
    )
    layout = geometry.VideoTokenLayout(
        used_len=8, video_spans=(geometry.VideoTokenSpan(start=0, latent_grid=(1, 1, 8), role="target"),)
    )
    metadata = geometry.AttentionMetadata(
        extra={"max_seqlen_q": 8, "max_seqlen_k": 12, "minimax_h3_compact_reference_rows": 4}, video_layout=layout
    )
    q = torch.ones(1, 8, 2, 16)
    kv = torch.ones(1, 12, 2, 16)
    assert instance.forward_npu(q, kv, kv, metadata).shape == q.shape
    assert calls == [(12, 12)]
    instance._forward_rectangular_npu.assert_not_called()


@pytest.mark.parametrize("override", [None, "float"])
@pytest.mark.parametrize("ratio", [0.1, 1.0])
def test_eight_steps_dispatch_full_then_rectangular(backend, config, geometry, progress, monkeypatch, override, ratio):
    calls = []

    @dataclasses.dataclass
    class RectangularPlan:
        dense_rows: int = 0

    def sparse_attention(q, k, v, *, precision, video_spans, sparsity, **kwargs):
        calls.append(("full", sparsity, video_spans[0].get("_ref_keep_ratio", 1.0), precision))
        return torch.zeros_like(q)

    def run(q, k, v, plan, sparsity, scale, *, ref_keep_ratio, **kwargs):
        calls.append(("rectangular", sparsity, ref_keep_ratio, kwargs["precision"]))
        return torch.zeros_like(q)

    native = ModuleType("mindiesd")
    native.__dict__["sparse_attention"] = sparse_attention
    platform = ModuleType("mindiesd.utils.get_platform")
    platform.__dict__["is_a5_device"] = lambda: True
    rectangular = ModuleType("vllm_omni.diffusion.attention.ops.rainfusion_rectangular")
    rectangular.__dict__["get_plan"] = lambda *args: RectangularPlan()
    rectangular.__dict__["run"] = run
    selection = ModuleType("vllm_omni.diffusion.attention.ops.reference_block_selection")
    selection.__dict__["rectangular_plan"] = lambda *args: (((0,),), (1,))
    for module in (native, platform, rectangular, selection):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    instance = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    instance.rainfusion = backend.RainFusionConfig.from_backend_kwargs(
        config.AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.8, "precision": "mix"},
            rainfusion={"ref_keep_ratio": ratio},
        ).backend_kwargs()
    )
    instance.qkv_layout = "BSND"
    instance.softmax_scale = 0.25
    instance._resolve_plan = lambda metadata: backend.RainFusionPlan(
        used_len=int(metadata.extra["max_seqlen_q"]),
        video_spans=[{"start": 4, "latent_shape": [1, 1, 8], "_ref_keep_ratio": ratio}],
    )
    q = torch.ones(1, 8, 2, 16)
    span = geometry.VideoTokenSpan(start=0, latent_grid=(1, 1, 8), role="target")
    layout = geometry.VideoTokenLayout(used_len=8, video_spans=(span,))
    kv_layout = dataclasses.replace(layout, used_len=12, reference_image_spans=((0, 4),))
    schedule = definitions(
        "_lq_main_schedule",
        "vllm_omni/diffusion/attention/schedule.py",
        {"AttentionScheduleRange", "_validate_step", "validate_attention_profile_name", "select_attention_profile"},
        {"dataclass": dataclasses.dataclass},
    )
    ranges = (schedule.AttentionScheduleRange(2, None, "late"),)
    for step in range(8):
        profile = schedule.select_attention_profile(ranges, step, total_steps=8)
        instance.rainfusion = dataclasses.replace(instance.rainfusion, sparsity=0.85 if profile == "late" else 0.8)
        progress.denoise_step_idx = step
        metadata = geometry.AttentionMetadata(extra={"max_seqlen_q": 8})
        kv = q
        if step:
            metadata = geometry.AttentionMetadata(
                extra={
                    "max_seqlen_q": 8,
                    "max_seqlen_k": 12,
                    "minimax_h3_compact_reference_rows": 4,
                    "minimax_h3_resident_reference_kv": True,
                    "minimax_h3_reference_kv_video_layout": kv_layout,
                },
                video_layout=layout,
            )
            kv = torch.ones(1, 12, 2, 16)
        if override is not None:
            metadata.extra["kv_cache_dtype"] = override
        assert instance.forward_npu(q, kv, kv, metadata).shape == q.shape
    precision = "bf16" if override == "float" else "mix"
    assert (
        calls
        == [("full", 0.8, ratio, precision), ("rectangular", 0.8, ratio, precision)]
        + [("rectangular", 0.85, ratio, precision)] * 6
    )
    # Token-refiner calls without sparse geometry must not need a step index.
    progress.denoise_step_idx = progress.total_denoise_steps = None
    instance._resolve_plan = lambda metadata: None
    instance.dense_fallback = torch.nn.Module()
    instance.dense_fallback.forward_npu = lambda q, k, v, metadata: q
    assert instance.forward_npu(q, q, q) is q
