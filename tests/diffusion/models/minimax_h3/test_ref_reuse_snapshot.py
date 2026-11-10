# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of D045 transport and opt-in config without installing vLLM/NPU."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def load_source(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_source("_h3_reuse_config", "vllm_omni/diffusion/kv_offload_config.py")
transport = load_source("_h3_reuse_transport", "vllm_omni/diffusion/attention/ops/research_sp_alignment.py")


@pytest.fixture
def cache_module() -> ModuleType:
    # Only import-time logging/platform dependencies are replaced. All cache
    # allocation, capture, reuse and cleanup below execute the published code
    # with real CPU tensors. No NPU/communication behavior is simulated.
    logger = ModuleType("vllm.logger")
    logger.__dict__["init_logger"] = logging.getLogger
    platform = ModuleType("vllm_omni.platforms")
    platform.__dict__["current_omni_platform"] = object()
    with patch.dict(
        sys.modules,
        {
            "vllm.logger": logger,
            "vllm_omni.platforms": platform,
            "vllm_omni.diffusion.kv_offload_config": config,
        },
    ):
        return load_source("_h3_reuse_cache", "vllm_omni/diffusion/models/minimax_h3/reference_kv_cache.py")


@torch.inference_mode()
def test_eight_step_reference_values_and_cleanup(cache_module: ModuleType) -> None:
    state = cache_module.MiniMaxH3ReferenceKVResidentState(
        num_layers=2, global_reference_rows=2, device=torch.device("cpu"), refresh_interval=8
    )
    state.set_global_reference_mask(torch.tensor([False, True, False, True, False]))
    original = torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2)
    for step in range(8):
        state.begin_step(step)
        for layer in range(2):
            full = original + 100 * layer + 1000 * step
            key = full if step == 0 else full[:, [0, 2, 4]]
            out_k, out_v = state.process_post_parallel_layer(
                layer, key, key + 50, compact=step > 0, parallel_strategy="none"
            )
            if step == 0:
                assert out_k is key
            else:
                reference = (original + 100 * layer)[:, [1, 3]]
                torch.testing.assert_close(out_k, torch.cat((reference, key), dim=1))
                torch.testing.assert_close(out_v, torch.cat((reference + 50, key + 50), dim=1))
        state.end_step()
    assert state.stats.refresh_steps == 1 and state.stats.resident_reuse_steps == 7
    assert state.stats.captured_layers == 2 and state.stats.substituted_layers == 14
    assert state.stats.h2d_bytes == state.stats.d2h_bytes == 0
    assert state.device_cache_bytes > 0 and state._device_ring is None
    state.close()
    assert state.device_cache_bytes == 0 and state._research_concat_scratch is None


@torch.inference_mode()
def test_partial_capture_fails_and_request_cleanup_releases_pool(cache_module: ModuleType) -> None:
    state = cache_module.MiniMaxH3ReferenceKVResidentState(
        num_layers=2, global_reference_rows=1, device=torch.device("cpu"), refresh_interval=8
    )
    state.set_global_reference_mask(torch.tensor([True, False]))
    key = torch.ones(1, 2, 1, 2)
    state.begin_step(0)
    state.process_post_parallel_layer(0, key, key, compact=False, parallel_strategy="none")
    with pytest.raises(RuntimeError, match="1/2 layers"):
        state.end_step()
    # The denoise loop closes this request-local state in finally. It never
    # resumes an aborted step with the partially populated cache object.
    state.close()
    assert state.device_cache_bytes == 0 and state._host_pool is None


@pytest.mark.parametrize("rows,world", [(13, 4), (16, 4), (2, 8), (17, 1), (89754, 4), (89754, 8)])
def test_transport_excludes_poisoned_padding(rows: int, world: int) -> None:
    # Values beyond the logical boundary must never reach attention, even if
    # communication padding contains arbitrary nonzero values.
    original = torch.arange(rows * 4, dtype=torch.float32).reshape(1, rows, 1, 4)
    shards = []
    coverage: list[int] = []
    for rank in range(world):
        start, valid, physical = transport.shard_bounds(rows, world, rank)
        coverage.extend(range(start, start + valid))
        shard = transport.append_rows(original[:, start : start + valid], physical - valid, 1)
        shard[:, valid:] = 999999
        shards.append(shard)
    assert coverage == list(range(rows))
    packed = torch.cat(shards, dim=1)
    padding = packed.shape[1] - rows
    q, k, v = transport.strip_qkv(packed, packed, packed, rows, padding)
    for tensor in (q, k, v):
        torch.testing.assert_close(tensor, original)
    restored = transport.append_rows(q, padding, 1)
    assert restored.shape == packed.shape
    assert torch.count_nonzero(restored[:, rows:]) == 0


@pytest.mark.parametrize("rows,world,rank", [(0, 4, 0), (2, 0, 0), (2, 4, 4), (2, 4, -1)])
def test_invalid_transport(rows: int, world: int, rank: int) -> None:
    with pytest.raises(ValueError):
        transport.shard_bounds(rows, world, rank)


def test_mismatched_qkv_fails() -> None:
    q = torch.zeros(1, 8, 1, 4)
    with pytest.raises(ValueError, match="shape"):
        transport.strip_qkv(q, q[:, :7], q, 5, 3)
    with pytest.raises(ValueError, match="layout"):
        transport.strip_qkv(q, q, q, 5, 2)


def test_reference_cache_is_opt_in() -> None:
    assert config.parse_reference_kv_config(None) is None
    assert config.parse_reference_kv_config({"mode": "on"}).kv_storage == "host"
    parsed = config.parse_reference_kv_config(
        {"mode": "on", "kv_storage": "npu", "kv_refresh_interval": 8, "kv_host_quantization": "none"}
    )
    assert parsed.kv_refresh_interval == 8 and parsed.kv_storage == "npu"


@pytest.mark.parametrize("quantization", ["fp8", "int8"])
def test_device_cache_rejects_quantized_storage(quantization: str) -> None:
    with pytest.raises(ValueError, match="unquantized"):
        config.parse_reference_kv_config({"mode": "on", "kv_storage": "npu", "kv_host_quantization": quantization})


@pytest.mark.parametrize(
    "value", [{"mode": "off"}, {"mode": "on", "unknown": 1}, {"mode": "on", "kv_refresh_interval": -1}]
)
def test_invalid_cache_config(value: dict) -> None:
    with pytest.raises(ValueError):
        config.parse_reference_kv_config(value)
