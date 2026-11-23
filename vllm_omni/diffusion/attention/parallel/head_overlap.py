# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stream grouped Ulysses for MiniMax-H3 resident reference-KV attention."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.ops.reference_kv_alignment import append_rows, strip_qkv
from vllm_omni.platforms.npu.graph_tools import CapturedDeviceGraph, NPUExactGraphRunner

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

logger = init_logger(__name__)

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ReferenceKV = Callable[[], tuple[torch.Tensor, torch.Tensor]]
Inputs = tuple[torch.Tensor, ...]

_MAX_GRAPHS = 4
# The graph retains copies, communication buffers and the rectangular mask.
# Match the source experiment's conservative capture headroom.
_CAPTURE_SAFETY_BYTES = 6 * 2**30
_GRAPH_FAILURE = "H3 head overlap NPUGraph failed. Restart the worker without --diffusion-attn-head-overlap-graph."


def _group_heads(tensor: torch.Tensor, world: int, groups: int) -> tuple[torch.Tensor, ...]:
    """Give each group the same destination ranks as ordinary Ulysses."""
    by_rank = tensor.unflatten(2, (world, tensor.shape[2] // world))
    return tuple(piece.flatten(2, 3) for piece in by_rank.chunk(groups, dim=3))


def _restore_heads(pieces: list[torch.Tensor], world: int) -> torch.Tensor:
    by_rank = [piece.unflatten(2, (world, piece.shape[2] // world)) for piece in pieces]
    return torch.cat(by_rank, dim=3).flatten(2, 3)


class _OverlapStreams:
    def __init__(self, device: torch.device, groups: int) -> None:
        npu = torch.npu
        self.comm = npu.Stream(device=device)
        self.ready = [npu.Event() for _ in range(groups)]
        self.computed = [npu.Event() for _ in range(groups)]
        self.done = [npu.Event() for _ in range(groups)]

    def run(
        self,
        strategy: UlyssesParallelAttention,
        world: int,
        groups: int,
        padding: int,
        attention: AttentionFn,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        reference_k: torch.Tensor,
        reference_v: torch.Tensor | None = None,
    ) -> Inputs:
        npu = torch.npu
        compute = npu.current_stream(query.device)
        comm = self.comm
        comm.wait_stream(compute)
        prepared = []
        with npu.stream(comm):
            grouped = zip(
                _group_heads(query, world, groups),
                _group_heads(key, world, groups),
                _group_heads(value, world, groups),
                strict=True,
            )
            for index, (q, k, v) in enumerate(grouped):
                q, k, v, _, ctx = strategy.pre_attention(q, k, v, None)
                self.ready[index].record(comm)
                prepared.append((q, k, v, ctx))

        if reference_v is not None:
            ref_keys = reference_k.chunk(groups, dim=1)
            ref_values = reference_v.chunk(groups, dim=1)
        gathered_keys, gathered_values, pieces = [], [], []
        for index, (q, k, v, ctx) in enumerate(prepared):
            compute.wait_event(self.ready[index])
            if padding:
                q, k, v = strip_qkv(q, k, v, q.shape[1] - padding, padding)
            if reference_v is None:
                # The third graph mode input is the request's reference row
                # index. Return these rows in the original rank's head order.
                gathered_keys.append(k[0].index_select(0, reference_k))
                gathered_values.append(v[0].index_select(0, reference_k))
            else:
                k = torch.cat((ref_keys[index].unsqueeze(0), k), dim=1)
                v = torch.cat((ref_values[index].unsqueeze(0), v), dim=1)
            out = append_rows(attention(q, k, v), padding, 1)
            self.computed[index].record(compute)
            with npu.stream(comm):
                comm.wait_event(self.computed[index])
                # The output was allocated on the compute stream but HCCL
                # can still be reading it after this iteration releases it.
                out.record_stream(comm)
                pieces.append(strategy.post_attention(out, ctx))
                self.done[index].record(comm)

        for event in self.done:
            compute.wait_event(event)
        for piece in pieces:
            piece.record_stream(compute)
        output = _restore_heads(pieces, world)
        if reference_v is None:
            return output, torch.cat(gathered_keys, dim=1), torch.cat(gathered_values, dim=1)
        return (output,)


class _HeadOverlapRunner:
    def __init__(self, cpu_group: dist.ProcessGroup, world: int, groups: int) -> None:
        self.cpu_group = cpu_group
        self.world = world
        self.groups = groups
        self._capture_closed = False
        self._graph_failed = False
        self._eager_streams: _OverlapStreams | None = None
        self._capture = NPUExactGraphRunner(component_name="H3 head overlap")
        # Keep the callable alive alongside the graph: it owns the streams,
        # events and RainFusion plan tensors referenced by captured kernels.
        self._graphs: dict[tuple[object, ...], tuple[CapturedDeviceGraph, Callable[..., Inputs]]] = {}

    def _admit_capture(self, signature: tuple[object, ...], device: torch.device, input_bytes: int) -> bool:
        supported = NPUExactGraphRunner.is_supported()
        if supported:
            try:
                free, _ = torch.npu.mem_get_info(device)
                supported = free >= 8 * input_bytes + _CAPTURE_SAFETY_BYTES
            except (RuntimeError, TypeError):
                supported = False

        # No rank may launch an eager all-to-all while its peers are capturing.
        # Agreement is only needed for new signatures; all caches are populated
        # in collective order and each stores the same set of keys.
        admissions = [None] * self.world
        proposal = (signature, supported)
        dist.all_gather_object(admissions, proposal, group=self.cpu_group)
        if not supported or any(admission != proposal for admission in admissions):
            self._capture_closed = True
            logger.warning_once("H3 head overlap graph unavailable; using ordinary Ulysses")
            return False
        return True

    def _inputs(
        self,
        active_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        take_reference: ReferenceKV,
        reference_rows: int,
    ) -> Inputs:
        query = active_qkv[0]
        reference_k, reference_v = take_reference()
        expected = (reference_rows, query.shape[2] // self.world, query.shape[3])
        if reference_k.shape != expected or reference_v.shape != expected:
            raise ValueError("resident reference KV head layout does not match ordinary Ulysses")
        return (*active_qkv, reference_k, reference_v)

    def run(
        self,
        strategy: UlyssesParallelAttention,
        active_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        take_reference: ReferenceKV | None,
        reference_rows: int,
        padding: int,
        geometry: tuple[object, ...],
        attention: AttentionFn,
        *,
        graph: bool,
        reference_positions: torch.Tensor | None = None,
        write_reference: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor | None:
        query = active_qkv[0]
        phase = "compact" if reference_positions is None else "full"
        if phase == "full":
            assert reference_positions is not None and reference_positions.shape == (reference_rows,)
            assert write_reference is not None and take_reference is None and padding == 0
        else:
            assert take_reference is not None and write_reference is None

        def current_inputs() -> Inputs:
            if reference_positions is not None:
                return (*active_qkv, reference_positions)
            assert take_reference is not None
            return self._inputs(active_qkv, take_reference, reference_rows)

        def finish(outputs: Inputs) -> torch.Tensor:
            if write_reference is not None:
                # The request-local pool and layer accounting stay out of
                # capture. Warmup plus capture must register the layer once.
                write_reference(outputs[1], outputs[2])
            return outputs[0]

        if not graph:
            if self._eager_streams is None:
                self._eager_streams = _OverlapStreams(query.device, self.groups)
            logger.info_once("H3 head overlap eager phase=%s groups=%d", phase, self.groups, scope="process")
            return finish(
                self._eager_streams.run(strategy, self.world, self.groups, padding, attention, *current_inputs())
            )

        if self._graph_failed:
            raise RuntimeError(_GRAPH_FAILURE)
        signature = (phase, tuple(query.shape), str(query.dtype), reference_rows, padding, geometry)
        if NPUExactGraphRunner._stream_is_capturing():
            return None
        entry = self._graphs.get(signature)
        if entry is not None:
            captured, _ = entry
            inputs = current_inputs()
            try:
                outputs = captured.replay(inputs)
            except RuntimeError as exc:
                self._graph_failed = True
                raise RuntimeError(_GRAPH_FAILURE) from exc
            result = finish(outputs)
            logger.info_once(
                "H3 head overlap NPUGraph replay active phase=%s groups=%d", phase, self.groups, scope="process"
            )
            return result
        if self._capture_closed:
            return None
        if len(self._graphs) >= _MAX_GRAPHS:
            logger.info_once("H3 head overlap reached its graph limit; new layouts use ordinary Ulysses")
            return None
        # Account for the resident heads in either the compact graph inputs
        # or the full graph outputs.
        reference_elements = 2 * reference_rows * (query.shape[2] // self.world) * query.shape[3]
        input_bytes = (3 * query.numel() + reference_elements) * query.element_size()
        if not self._admit_capture(signature, query.device, input_bytes):
            return None

        streams = _OverlapStreams(query.device, self.groups)

        def compute(*inputs: torch.Tensor) -> Inputs:
            return streams.run(strategy, self.world, self.groups, padding, attention, *inputs)

        inputs = current_inputs()
        eager = compute(*inputs)  # Prime lazy operators before capture.
        try:
            captured = self._capture.capture(inputs, compute)
        except RuntimeError as exc:
            # Once HCCL capture starts, allocator/stream or collective state
            # may be invalid. Only preflight rejection can fall back safely.
            self._graph_failed = True
            raise RuntimeError(_GRAPH_FAILURE) from exc
        self._graphs[signature] = (captured, compute)
        logger.info("H3 head overlap captured NPUGraph phase=%s groups=%d", phase, self.groups)
        return finish(eager)


@cache
def get_head_overlap_runner(cpu_group: dist.ProcessGroup, world: int, groups: int) -> _HeadOverlapRunner:
    return _HeadOverlapRunner(cpu_group, world, groups)
