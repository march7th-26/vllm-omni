# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded local sparse graphs, independent of layers and request Ref pools."""

from collections import OrderedDict
from functools import cache
from threading import get_ident

import torch


class H3SparseGraph:
    def __init__(self):
        self.key = None
        self.graph = None
        self.inputs = None
        self.output = None
        self.captures = 0
        self.replays = 0

    def run(self, function, tensors, geometry):
        import torch_npu

        runtime = torch_npu.npu
        key = (geometry, tuple((tuple(x.shape), x.dtype, x.device) for x in tensors))
        if self.key != key:
            # Warmup is an ordinary business invocation. Replacement is bounded
            # to one graph per cache entry, rather than one per head/layer.
            runtime.current_stream().synchronize()
            self.graph = self.inputs = self.output = None
            self.key = key
            return function(*tensors)
        if self.graph is None:
            self.inputs = tuple(torch.empty_like(x) for x in tensors)
            for destination, source in zip(self.inputs, tensors):
                destination.copy_(source)
            runtime.current_stream().synchronize()
            graph = runtime.NPUGraph()
            with runtime.graph(graph):
                self.output = function(*self.inputs)
            self.graph = graph
            self.captures += 1
        else:
            assert self.inputs is not None
            for destination, source in zip(self.inputs, tensors):
                destination.copy_(source)
        # Explicit replay also supplies the result of the capture invocation;
        # callers consume it before another head overwrites this static output.
        self.graph.replay()
        self.replays += 1
        return self.output


class H3SparseGraphCache:
    """Share identical pure sparse kernels across layers; at most three graphs.

    The default LQ-Slim recipe needs refresh, early reuse and late reuse keys.
    Different media geometries evict old entries rather than retaining every
    request's static Q/K/V buffers. The caller must own outputs before reuse.
    """

    def __init__(self, capacity=3):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.binding = None

    def run(self, function, tensors, geometry):
        import torch_npu

        stream = torch_npu.npu.current_stream()
        binding = (tensors[0].device, stream.npu_stream, get_ident())
        if self.binding is None:
            self.binding = binding
        elif binding != self.binding:
            raise RuntimeError("H3 sparse graphs require one worker device/thread/compute stream")
        key = (geometry, tuple((tuple(x.shape), x.dtype, x.device) for x in tensors))
        entry = self.entries.pop(key, None)
        if entry is None:
            if len(self.entries) == self.capacity:
                stream.synchronize()
                self.entries.popitem(last=False)
            entry = H3SparseGraph()
        self.entries[key] = entry
        return entry.run(function, tensors, geometry)


@cache
def get_shared_sparse_graph():
    return H3SparseGraphCache()
