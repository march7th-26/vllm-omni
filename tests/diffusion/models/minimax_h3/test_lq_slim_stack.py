# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU math and head-placement checks; no native NPU/Graph claims."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def load(relative):
    spec = importlib.util.spec_from_file_location("_stack_math", ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audio = load("vllm_omni/diffusion/models/minimax_h3/audio_conv.py")
overlap = load("vllm_omni/diffusion/attention/ops/h3_head_overlap.py")
graphs = load("vllm_omni/diffusion/attention/ops/h3_sparse_graph.py")


@pytest.mark.parametrize(
    "stride,kernel,padding,output_padding", [(2, 4, 1, 0), (3, 7, 2, 1), (4, 8, 2, 0), (2, 3, 0, 1)]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("length", [1, 7, 19])
@torch.inference_mode()
def test_audio_polyphase_matches_native(stride, kernel, padding, output_padding, bias, length):
    torch.manual_seed(42)
    module = nn.ConvTranspose1d(3, 5, kernel, stride, padding, output_padding, bias=bias).double().eval()
    x = torch.randn(2, 3, length, dtype=torch.float64)
    expected = module(x)
    assert audio.install_audio_conv1d(module) == 1
    assert audio.install_audio_conv1d(module) == 0
    torch.testing.assert_close(module(x), expected, atol=1e-12, rtol=1e-12)
    # Updated weights are read on the next invocation, as with weight_norm.
    module.weight.mul_(2)
    torch.testing.assert_close(module(x), module._h3_native_forward(x), atol=1e-12, rtol=1e-12)


def test_audio_preserves_state_dict_and_unsupported_geometry():
    model = nn.Sequential(nn.ConvTranspose1d(4, 4, 4, 2), nn.ConvTranspose1d(4, 4, 4, 2, dilation=2))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    assert audio.install_audio_conv1d(model) == 1
    assert not hasattr(model[1], "_h3_native_forward")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    x = torch.randn(1, 4, 3)
    torch.testing.assert_close(
        model[0](x, output_size=[1, 4, 9]), model[0]._h3_native_forward(x, output_size=[1, 4, 9])
    )


@pytest.mark.parametrize("groups", [1, 2, 4])
@torch.inference_mode()
def test_audio_parametrized_grouped_weights(groups):
    module = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(4, 8, 5, 3, 1, groups=groups)).double()
    x = torch.randn(2, 4, 9, dtype=torch.float64)
    expected = module(x)
    audio.install_audio_conv1d(module)
    torch.testing.assert_close(module(x), expected, atol=1e-12, rtol=1e-12)
    module.parametrizations.weight.original0.mul_(1.5)
    torch.testing.assert_close(module(x), module._h3_native_forward(x), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ratio,channels,length", [(2, 1, 1), (2, 4, 17), (3, 2, 7), (4, 3, 19)])
@torch.inference_mode()
def test_alias_free_functional_upsampler(ratio, channels, length):
    class UpSample1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.ratio = self.stride = ratio
            kernel = 6 * ratio
            self.pad = kernel // ratio - 1
            self.pad_left = self.pad * ratio + (kernel - ratio) // 2
            self.pad_right = self.pad * ratio + (kernel - ratio + 1) // 2
            self.register_buffer("filter", torch.randn(1, 1, kernel, dtype=torch.float64))

        def forward(self, x):
            padded = nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
            result = self.ratio * nn.functional.conv_transpose1d(
                padded, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, groups=x.shape[1]
            )
            return result[..., self.pad_left : -self.pad_right]

    module = UpSample1d()
    x = torch.randn(2, channels, length, dtype=torch.float64)
    expected = module(x)
    assert audio.install_audio_conv1d(module) == 1
    assert audio.install_audio_conv1d(module) == 0
    torch.testing.assert_close(module(x), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("world,heads,chunk", [(2, 8, 1), (4, 56, 1), (4, 56, 3), (8, 56, 2)])
def test_chunk_ulysses_and_reverse_preserve_every_head(world, heads, chunk):
    # An independent global tensor is the oracle for both all-to-all layouts.
    global_tensor = torch.arange(world * 3 * heads * 2).reshape(1, world * 3, heads, 2)
    sends = list(global_tensor.chunk(world, dim=1))
    reconstructed = [torch.empty_like(x) for x in sends]
    local = heads // world
    for start in range(0, local, chunk):
        width = min(chunk, local - start)
        indices = overlap.head_indices(heads, world, start, width)
        packed = [x[:, :, indices] for x in sends]
        received = [torch.cat([x[:, :, rank * width : (rank + 1) * width] for x in packed], 1) for rank in range(world)]
        for rank, tensor in enumerate(received):
            torch.testing.assert_close(tensor, global_tensor[:, :, rank * local + start : rank * local + start + width])
        for rank in range(world):
            returned = torch.cat([x[:, rank * 3 : (rank + 1) * 3] for x in received], 2)
            reconstructed[rank][:, :, indices] = returned
    for actual, expected in zip(reconstructed, sends):
        torch.testing.assert_close(actual, expected)


def test_graph_updates_values_and_invalidates_geometry(monkeypatch):
    # Emulate graph recording/replay only. Real copies, static input pointers
    # and production graph-cache selection execute on CPU; no NPU emulation.
    current = SimpleNamespace(graph=None)

    class Graph:
        output: torch.Tensor
        inputs: tuple[torch.Tensor, torch.Tensor]

        def replay(self):
            self.output.copy_(self.inputs[0] + 2 * self.inputs[1])

    @contextmanager
    def capture(graph):
        current.graph = graph
        yield
        current.graph = None

    runtime = SimpleNamespace(
        NPUGraph=Graph, graph=capture, current_stream=lambda: SimpleNamespace(npu_stream=7, synchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=runtime))

    def kernel(x, y):
        output = x + 2 * y
        if current.graph is not None:
            current.graph.inputs = (x, y)
            current.graph.output = output
        return output

    cache = graphs.H3SparseGraph()
    for sparsity, offset in [(0.8, 1), (0.8, 10), (0.8, 100), (0.85, 1000), (0.85, 10000)]:
        inputs = (torch.arange(4.0) + offset, torch.ones(4) * offset)
        actual = cache.run(kernel, inputs, ("image-layout-a", sparsity))
        torch.testing.assert_close(actual, inputs[0] + 2 * inputs[1])
    assert cache.captures == 2 and cache.replays == 3
    # Same geometry, independent request values: no captured Ref-pool pointer.
    new_inputs = (torch.zeros(4), torch.ones(4))
    torch.testing.assert_close(cache.run(kernel, new_inputs, ("image-layout-a", 0.85)), torch.full((4,), 2.0))
    assert cache.captures == 2 and cache.replays == 4
    cache.run(kernel, (torch.ones(7), torch.ones(7)), ("image-layout-b", 0.85))
    assert cache.graph is None and cache.inputs is None
    shared = graphs.H3SparseGraphCache()
    for geometry in ["refresh", "early-reuse", "late-reuse", "new-media"]:
        shared.run(kernel, new_inputs, geometry)
    assert len(shared.entries) == 3
    assert all(key[0] != "refresh" for key in shared.entries)
    torch.testing.assert_close(shared.run(kernel, new_inputs, "late-reuse"), torch.full((4,), 2.0))
    runtime.current_stream = lambda: SimpleNamespace(npu_stream=8, synchronize=lambda: None)
    with pytest.raises(RuntimeError, match="compute stream"):
        shared.run(kernel, new_inputs, "late-reuse")
