# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stream grouped Ulysses for MiniMax-H3 resident short-Q attention."""

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
_GRAPH_FAILURE = (
    "H3 short-Q head overlap NPUGraph failed. Restart the worker without --diffusion-attn-head-overlap-graph."
)


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
        reference_v: torch.Tensor,
    ) -> tuple[torch.Tensor]:
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

        ref_keys = reference_k.chunk(groups, dim=1)
        ref_values = reference_v.chunk(groups, dim=1)
        pieces = []
        for index, (q, k, v, ctx) in enumerate(prepared):
            compute.wait_event(self.ready[index])
            if padding:
                q, k, v = strip_qkv(q, k, v, q.shape[1] - padding, padding)
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
        return (_restore_heads(pieces, world),)


class _HeadOverlapRunner:
    def __init__(self, cpu_group: dist.ProcessGroup, world: int, groups: int) -> None:
        self.cpu_group = cpu_group
        self.world = world
        self.groups = groups
        self._capture_closed = False
        self._graph_failed = False
        self._eager_streams: _OverlapStreams | None = None
        self._capture = NPUExactGraphRunner(component_name="H3 short-Q head overlap")
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
            logger.warning_once("H3 short-Q head overlap graph unavailable; using ordinary Ulysses")
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
        take_reference: ReferenceKV,
        reference_rows: int,
        padding: int,
        geometry: tuple[object, ...],
        attention: AttentionFn,
        *,
        graph: bool,
    ) -> torch.Tensor | None:
        query = active_qkv[0]
        if not graph:
            if self._eager_streams is None:
                self._eager_streams = _OverlapStreams(query.device, self.groups)
                logger.info("H3 short-Q eager head overlap active with %d groups", self.groups)
            return self._eager_streams.run(
                strategy,
                self.world,
                self.groups,
                padding,
                attention,
                *self._inputs(active_qkv, take_reference, reference_rows),
            )[0]

        if self._graph_failed:
            raise RuntimeError(_GRAPH_FAILURE)
        signature = (tuple(query.shape), str(query.dtype), padding, geometry)
        if NPUExactGraphRunner._stream_is_capturing():
            return None
        entry = self._graphs.get(signature)
        if entry is not None:
            captured, _ = entry
            logger.info_once("H3 short-Q head overlap NPUGraph replay active with %d groups", self.groups)
            inputs = self._inputs(active_qkv, take_reference, reference_rows)
            try:
                return captured.replay(inputs)[0]
            except RuntimeError as exc:
                self._graph_failed = True
                raise RuntimeError(_GRAPH_FAILURE) from exc
        if self._capture_closed:
            return None
        if len(self._graphs) >= _MAX_GRAPHS:
            logger.info_once("H3 short-Q head overlap reached its graph limit; new layouts use ordinary Ulysses")
            return None
        reference_elements = 2 * reference_rows * (query.shape[2] // self.world) * query.shape[3]
        input_bytes = (3 * query.numel() + reference_elements) * query.element_size()
        if not self._admit_capture(signature, query.device, input_bytes):
            return None

        streams = _OverlapStreams(query.device, self.groups)

        def compute(*inputs: torch.Tensor) -> tuple[torch.Tensor]:
            q, k, v, ref_k, ref_v = inputs
            return streams.run(strategy, self.world, self.groups, padding, attention, q, k, v, ref_k, ref_v)

        inputs = self._inputs(active_qkv, take_reference, reference_rows)
        eager = compute(*inputs)[0]  # Prime lazy operators before capture.
        try:
            captured = self._capture.capture(inputs, compute)
        except RuntimeError as exc:
            # Once HCCL capture starts, allocator/stream or collective state
            # may be invalid. Only preflight rejection can fall back safely.
            self._graph_failed = True
            raise RuntimeError(_GRAPH_FAILURE) from exc
        self._graphs[signature] = (captured, compute)
        logger.info("H3 short-Q head overlap captured NPUGraph %d/%d", len(self._graphs), _MAX_GRAPHS)
        return eager


@cache
def get_head_overlap_runner(cpu_group: dist.ProcessGroup, world: int, groups: int) -> _HeadOverlapRunner:
    return _HeadOverlapRunner(cpu_group, world, groups)
