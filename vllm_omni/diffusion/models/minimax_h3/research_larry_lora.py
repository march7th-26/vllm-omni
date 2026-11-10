# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental immutable Larryvrh v4-600 EMA runtime adapter; TP=1 only."""

import hashlib
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from vllm.distributed import get_tensor_model_parallel_world_size

SHA = "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3"


def expected_targets():
    targets = {}
    for prefix, count in [("blocks", 50), ("token_refiner.blocks", 2)]:
        for i in range(count):
            for suffix, dims in {
                "attn.qkv_proj": (5376, 21504, 64),
                "attn.out_proj": (7168, 5376, 64),
                "mlp.fc1": (5376, 28672, 64),
                "mlp.fc2": (14336, 5376, 64),
            }.items():
                targets[f"{prefix}.{i}.{suffix}"] = dims
    for i in range(50):
        targets[f"blocks.{i}.adaln_proj.linear"] = (2688, 96768, 16)
    targets["final_layer.adaln_proj.linear"] = (2688, 10752, 16)
    return targets


def load_checked(path):
    with open(path, "rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == SHA
    state = load_file(path, device="cpu")
    expected = expected_targets()
    assert set(state) == {n + f".lora_{side}.weight" for n in expected for side in "AB"}
    pairs = {}
    for name, (ins, outs, rank) in expected.items():
        a, b = (state[name + f".lora_{side}.weight"] for side in "AB")
        assert a.dtype == b.dtype == torch.bfloat16
        assert tuple(a.shape) == (rank, ins) and tuple(b.shape) == (outs, rank)
        pairs[name] = (a, b)
    return pairs


def delta(x, a, b, scale=1.0):
    return F.linear(F.linear(x.to(torch.bfloat16), a), b) * scale


def install(model):
    path = os.environ.get("RESEARCH_LARRY_LORA_PATH")
    if not path or hasattr(model, "_larry_lora_state"):
        return
    assert get_tensor_model_parallel_world_size() == 1, "Experimental LoRA requires DiT TP1 (USP4 supported)"
    pairs = load_checked(path)
    state = {"pairs": pairs, "counts": dict.fromkeys(pairs, 0), "forward": 0, "prepared": False}
    model._larry_lora_state = state
    for name, (ins, outs, _) in expected_targets().items():
        module = model.get_submodule(name)
        assert module.input_size == ins and module.output_size == outs, (name, module)
        assert not getattr(module, "skip_bias_add", False), name

        def hook(module, args, output, name=name):
            a, b = state["pairs"][name]
            x = args[0]
            assert x.device == a.device == b.device
            assert isinstance(output, tuple) and len(output) == 2
            y, bias = output
            update = delta(x, a, b)
            assert y.shape == update.shape, (name, y.shape, update.shape)
            state["counts"][name] += 1
            return (y + update.to(y.dtype), bias)

        module.register_forward_hook(hook)

    def before(module, args, kwargs):
        if not state["prepared"]:
            device = next(p.device for p in module.parameters() if p.device.type not in ("cpu", "meta"))
            state["pairs"] = {n: (a.to(device), b.to(device)) for n, (a, b) in state["pairs"].items()}
            state["prepared"] = True
            # All timestep-cache values must be produced with this immutable adapter.
            module.adaln_cache.clear()
            print(
                "RESEARCH_LARRY_LOADED "
                + json.dumps(
                    {
                        "rank": torch.distributed.get_rank(),
                        "targets": len(pairs),
                        "sha256": SHA,
                        "scale": 1.0,
                        "tp": 1,
                        "bytes": sum(
                            a.numel() * a.element_size() + b.numel() * b.element_size() for a, b in pairs.values()
                        ),
                        "qkv_layout": "contiguous_q_k_v_no_reorder",
                        "device": str(device),
                    }
                ),
                flush=True,
            )
        state["forward"] += 1

    def after(module, args, output):
        # Full coverage once per 8-forward request; AdaLN values may hit the
        # model's existing exact timestep cache after the first request.
        if state["forward"] % 8 == 0:
            assert all(v > 0 for v in state["counts"].values()), "LoRA target never executed"
            print(
                "RESEARCH_LARRY_COVERAGE "
                + json.dumps(
                    {
                        "rank": torch.distributed.get_rank(),
                        "forward": state["forward"],
                        "targets_hit": sum(v > 0 for v in state["counts"].values()),
                        "counts": state["counts"],
                    }
                ),
                flush=True,
            )

    model.register_forward_pre_hook(before, with_kwargs=True)
    model.register_forward_hook(after)
