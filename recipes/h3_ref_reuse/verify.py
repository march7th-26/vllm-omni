# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the published source identity; optionally inspect installed NPU APIs."""

import argparse
import hashlib
import importlib
import inspect
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="also check installed Ascend dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    entries = (root / "recipes/h3_ref_reuse/source_published.sha256").read_text().splitlines()
    for entry in entries:
        digest, relative = entry.split("  ", 1)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Published source differs: {relative}")
    print(f"Published source verified: {len(entries)} files")
    if not args.runtime:
        return
    omni = importlib.import_module("vllm_omni")
    if Path(omni.__file__).resolve().parent != root / "vllm_omni":
        raise RuntimeError(f"Installed Omni resolves outside this checkout: {omni.__file__}")
    npu = importlib.import_module("torch_npu")
    torch = importlib.import_module("torch")
    rf2 = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2")
    rf3 = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3")
    sparse = importlib.import_module("mindiesd").sparse_attention
    for name in ("_normalize_video_spans", "_multi_span_permutation", "avgpool"):
        if not callable(getattr(rf2, name, None)):
            raise RuntimeError(f"MindIE-SD lacks rf_v2.{name}")
    for name in ("rain_fusion_attention_v3", "_eagle_qbsa_quant_qkv"):
        if not callable(getattr(rf3, name, None)):
            raise RuntimeError(f"MindIE-SD lacks rf_v3.{name}")
    if not {"precision", "video_spans"} <= set(inspect.signature(sparse).parameters):
        raise RuntimeError("MindIE-SD sparse_attention must explicitly accept precision and video_spans")
    platform = importlib.import_module("mindiesd.utils.get_platform")
    if not torch.npu.is_available() or not platform.is_a5_device():
        raise RuntimeError("This recipe requires an Ascend A5 NPU")
    if not hasattr(torch.ops.mindiesd, "eagle_quant_block_sparse_attention"):
        raise RuntimeError("MindIE-SD native Eagle operator is not registered")
    for module in (torch, npu, rf2, rf3):
        print(module.__name__, getattr(module, "__version__", ""), module.__file__)
    print("API presence only; communication, native numerics and model execution are separate checks.")


if __name__ == "__main__":
    main()
