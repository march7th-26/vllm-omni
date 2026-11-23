# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tensor and control-flow checks for the H3 overlap implementation."""

from contextlib import nullcontext
from functools import cache
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.minimax_h3.test_reference_block_selection import definitions
from tests.diffusion.models.minimax_h3.test_reference_kv_resident import transport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def overlap(mocker):
    graph_api = mocker.Mock()
    graph_api.is_supported.return_value = True
    graph_api._stream_is_capturing.return_value = False
    return definitions(
        "_h3_overlap",
        "vllm_omni/diffusion/attention/parallel/head_overlap.py",
        {"_group_heads", "_restore_heads", "_OverlapStreams", "_HeadOverlapRunner", "get_head_overlap_runner"},
        {
            "torch": torch,
            "dist": mocker.Mock(),
            "cache": cache,
            "logger": mocker.Mock(),
            "append_rows": transport.append_rows,
            "strip_qkv": transport.strip_qkv,
            "NPUExactGraphRunner": graph_api,
            "_MAX_GRAPHS": 4,
            "_CAPTURE_SAFETY_BYTES": 6 * 2**30,
            "_GRAPH_FAILURE": "Restart the worker",
        },
    )


@pytest.mark.parametrize("groups", [2, 7, 14])
def test_grouped_exchange_preserves_rank_heads_and_projection_order(overlap, groups):
    world = 4
    query = torch.arange(56).view(1, 1, 56, 1)
    original_by_rank = query.chunk(world, dim=2)
    packed = overlap._group_heads(query, world, groups)
    for rank in range(world):
        received = torch.cat([piece.chunk(world, dim=2)[rank] for piece in packed], dim=2)
        torch.testing.assert_close(received, original_by_rank[rank])
    torch.testing.assert_close(overlap._restore_heads(list(packed), world), query)


def test_overlap_strips_transport_padding_and_reads_reference_once(overlap, monkeypatch, mocker):
    world, groups, padding = 4, 2, 1
    comm = mocker.Mock()
    stream = mocker.Mock()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda **kwargs: comm,
            Event=mocker.Mock,
            current_stream=lambda device: stream,
            stream=lambda _: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    runner = overlap._HeadOverlapRunner(object(), world, groups)
    active = torch.ones(1, 2, 56, 1)
    heads = torch.arange(14, dtype=active.dtype)
    reference = heads.view(1, 14, 1).repeat(3, 1, 1)
    take_reference = mocker.Mock(return_value=(reference, reference + 100))

    class RankZero:
        def pre_attention(self, q, k, v, metadata):
            return (*(tensor.chunk(world, dim=2)[0].repeat(1, world, 1, 1) for tensor in (q, k, v)), metadata, None)

        def post_attention(self, output, ctx):
            # Take the final two transport rows and model only rank 0's heads.
            return torch.cat((output[:, -2:], torch.zeros(1, 2, 21, 1)), dim=2)

    def attention(q, k, v):
        assert q.shape == (1, 7, 7, 1)
        assert k.shape == v.shape == (1, 10, 7, 1)
        torch.testing.assert_close(k[:, 3:], torch.ones_like(q))
        torch.testing.assert_close(v[:, :3], k[:, :3] + 100)
        return q + k[:, :1]

    output = runner.run(RankZero(), (active, active, active), take_reference, 3, padding, (), attention, graph=False)
    take_reference.assert_called_once_with()
    expected = torch.zeros_like(active)
    expected[:, 0, :14, 0] = heads + 1
    torch.testing.assert_close(output, expected)


def test_peer_rejection_falls_back_before_reading_reference(overlap, monkeypatch, mocker):
    runner = overlap._HeadOverlapRunner(object(), 2, 2)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(mem_get_info=lambda _: (2**40, 2**41)), raising=False)

    def disagree(admissions, proposal, group):
        admissions[:] = [proposal, (proposal[0], False)]

    overlap.dist.all_gather_object.side_effect = disagree
    active = torch.zeros(1, 2, 8, 4)
    take_reference = mocker.Mock()
    attention = mocker.Mock()
    for _ in range(2):
        result = runner.run(object(), (active, active, active), take_reference, 3, 1, (), attention, graph=True)
        assert result is None
    take_reference.assert_not_called()
    attention.assert_not_called()
    overlap.dist.all_gather_object.assert_called_once()


def test_shared_graph_uses_each_layers_reference_and_keys_on_layout(overlap, monkeypatch, mocker):
    runner = overlap._HeadOverlapRunner(object(), 2, 2)
    monkeypatch.setattr(runner, "_admit_capture", lambda *args: True)

    def run(strategy, world, groups, padding, attention, q, k, v, ref_k, ref_v):
        return (q + ref_k[0, 0, 0] + ref_v[0, 0, 0],)

    monkeypatch.setattr(overlap, "_OverlapStreams", lambda *args: SimpleNamespace(run=run))
    runner._capture.capture.side_effect = lambda inputs, compute: SimpleNamespace(
        replay=lambda values: compute(*values)
    )
    q = torch.ones(1, 2, 8, 4)
    reference = torch.ones(3, 4, 4)
    take_reference = mocker.Mock()

    for layout, increment in [("A", 0), ("A", 2), ("B", 3)]:
        take_reference.return_value = (reference, reference + increment)
        out = runner.run(object(), (q, q, q), take_reference, 3, 1, (layout,), mocker.Mock(), graph=True)
        torch.testing.assert_close(out, q + 2 + increment)

    assert take_reference.call_count == 3
    assert runner._capture.capture.call_count == 2


def test_graph_capture_error_does_not_run_a_second_transport(overlap, monkeypatch, mocker):
    runner = overlap._HeadOverlapRunner(object(), 2, 2)
    monkeypatch.setattr(runner, "_admit_capture", lambda *args: True)
    active = torch.ones(1, 2, 8, 4)
    reference = torch.ones(3, 4, 4)
    take_reference = mocker.Mock(return_value=(reference, reference))
    grouped_forward = mocker.Mock(return_value=(active,))
    monkeypatch.setattr(overlap, "_OverlapStreams", lambda *args: SimpleNamespace(run=grouped_forward))
    runner._capture.capture.side_effect = RuntimeError("application-side capture failure")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Restart the worker"):
            runner.run(object(), (active, active, active), take_reference, 3, 1, (), mocker.Mock(), graph=True)

    take_reference.assert_called_once_with()
    grouped_forward.assert_called_once()
