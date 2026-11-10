# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Experiment-only transport padding. Original Q8 is retained separately.
import torch


def alignment(total: int, world: int) -> tuple[int, int]:
    if total <= 0 or world <= 0:
        raise ValueError("positive compact length and SP world required")
    padding = (-total) % world
    return total + padding, padding


def shard_bounds(total: int, world: int, rank: int) -> tuple[int, int, int]:
    if not 0 <= rank < world:
        raise ValueError("invalid SP rank")
    physical, _ = alignment(total, world)
    width = physical // world
    start = min(rank * width, total)
    end = min((rank + 1) * width, total)
    return start, end - start, width


def append_rows(tensor: torch.Tensor, count: int, dim: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("negative padding")
    if not count:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = count
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dim)


def strip_qkv(query, key, value, logical: int, padding: int):
    expected = logical + padding
    if query.shape != key.shape or key.shape != value.shape or query.shape[1] != expected:
        raise ValueError("transport QKV shape does not match compact layout")
    return tuple(t[:, :logical].contiguous() for t in (query, key, value))
