# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BF16 head-chunk transport for H3's public image-sparse consumer.

This deliberately does not use MindIE's equal-length, quantized-wire producer:
Ref KV is injected on the compute stream after each compact chunk arrives.
"""

import torch


def head_indices(heads: int, world: int, start: int, width: int) -> list[int]:
    if heads % world or not 0 <= start < heads // world or not 0 < width <= heads // world - start:
        raise ValueError("invalid strict Ulysses head chunk")
    local = heads // world
    return [rank * local + head for rank in range(world) for head in range(start, start + width)]


class H3HeadOverlap:
    """One invocation owns its buffers until the compute stream joins reverse A2A."""

    def __init__(self, group, chunk_size: int):
        import torch_npu

        self.group = group
        self.chunk_size = chunk_size
        self.platform = torch_npu.npu
        self.comm_stream = self.platform.Stream()

    def run(self, query, key, value, metadata, state, layer, compact, consumer):
        import torch.distributed as dist

        from vllm_omni.diffusion.attention.ops.research_sp_alignment import append_rows, strip_qkv
        from vllm_omni.diffusion.distributed.comm import all_to_all_4D

        if query.shape != key.shape or key.shape != value.shape or query.ndim != 4 or query.shape[0] != 1:
            raise ValueError("H3 head overlap requires matching [1,S_local,H,D] transport tensors")
        if query.dtype != torch.bfloat16 or query.device.type != "npu":
            raise ValueError("H3 head overlap requires BF16 NPU transport")
        world = dist.get_world_size(self.group)
        heads = query.shape[2]
        if heads % world:
            raise ValueError("H3 head overlap requires strict divisible heads")
        local_heads = heads // world
        alignment = metadata.extra.get("minimax_h3_transport_alignment")
        padding = alignment["transport_padding"] if alignment else 0
        compute = self.platform.current_stream()
        self.comm_stream.wait_stream(compute)
        pending = []
        output = torch.empty_like(query)
        # Queue forward transports before any compute-dependent reverse. All
        # ranks submit the same collective order, independent of stream timing.
        # Keep send/receive/output owners alive until the final join. A Python
        # wait_stream alone does not protect cross-stream allocator lifetimes.
        owners = [query, key, value, output]
        try:
            with self.platform.stream(self.comm_stream):
                for start in range(0, local_heads, self.chunk_size):
                    width = min(self.chunk_size, local_heads - start)
                    indices = torch.tensor(head_indices(heads, world, start, width), device=query.device)
                    sends = [x.index_select(2, indices) for x in (query, key, value)]
                    received = [all_to_all_4D(x, 2, 1, self.group) for x in sends]
                    ready = self.platform.Event()
                    ready.record(self.comm_stream)
                    pending.append((start, width, indices, sends, received, ready))
            for start, width, indices, sends, received, ready in pending:
                compute.wait_event(ready)
                q, k, v = received
                if padding:
                    q, k, v = strip_qkv(q, k, v, alignment["compact_rows"], padding)
                k, v = state.process_post_parallel_chunk(
                    layer, k, v, compact=compact, head_start=start, total_heads=local_heads
                )
                result = consumer(q, k, v, metadata)
                if padding:
                    result = append_rows(result, padding, 1)
                done = self.platform.Event()
                done.record(compute)
                with self.platform.stream(self.comm_stream):
                    self.comm_stream.wait_event(done)
                    returned = all_to_all_4D(result, 1, 2, self.group)
                    output.index_copy_(2, indices, returned)
                owners.extend([*sends, *received, q, k, v, result, returned, indices])
            compute.wait_stream(self.comm_stream)
            for tensor in owners:
                tensor.record_stream(compute)
                tensor.record_stream(self.comm_stream)
        except Exception:
            # Do not retry/fall back after submitting a collective. Drain our
            # streams before temporary storage or request-local Ref pools die.
            self.comm_stream.synchronize()
            compute.synchronize()
            raise
        if padding:
            state._research_sp_stripped_calls = getattr(state, "_research_sp_stripped_calls", 0) + 1
            state._research_sp_restored_calls = getattr(state, "_research_sp_restored_calls", 0) + 1
        return output
