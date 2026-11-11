# LQ-Slim

LQ-Slim combines eight-step MiniMax-H3 Ref2VA LoRA, NPU-resident reference KV,
rectangular mixed-precision attention and per-image reference block selection.
The target sparsity is 0.8 for steps 0–1 and 0.85 for steps 2–7; reference-image
candidate blocks use a 0.1 keep ratio. These are approximate optimizations;
output quality must be evaluated for the intended inputs.

For deployments that only need the two Ref extensions, see the independent
[reference optimization guide](../h3_ref_optimizations/README.md). They do not
require the fixed adapter or the eight-step preset below.

## Install

Use the project's normal [Ascend installation](../../docs/getting_started/installation/npu.md)
and a MindIE-SD build containing [reference block selection PR #842](https://gitcode.com/Ascend/MindIE-SD/pull/842/discuss).
The dependency revision used for this port is `98fd99717182623bbfa952dd676f6865f2729f9d`.
It includes the full-keep boundary correction; this PR is a separate dependency
and is not installed automatically with vLLM-Omni.

```bash
git clone --branch lq-slim-main --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

Prepare [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the
[Larry v4-step600 EMA adapter](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
The supported file is `minimax_h3_turbo_v4_step600_ema.safetensors`, SHA256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.
It has 259 BF16 A/B pairs, including rank-16 AdaLN and rank-64 attention/MLP
deltas. It uses contiguous Q/K/V layout and is a different format from the
FlashGen and LightX2V Turbo adapters.

## Serve

Select four Ascend A5 devices using the standard device environment settings,
then replace the model and adapter paths:

```bash
export MODEL=/path/to/MiniMax-H3
export LORA=/path/to/minimax_h3_turbo_v4_step600_ema.safetensors
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$MODEL" --omni --task-type ref2va \
    --host 127.0.0.1 --port 9098 --trust-remote-code \
    --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
    --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
    --lora-path "$LORA" --lora-backend distill --lora-scale 1 \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix"},"rainfusion":{"ref_keep_ratio":0.1}}}' \
    --diffusion-attention-schedule '{"profiles":{"late":{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.85,"precision":"mix"},"rainfusion":{"ref_keep_ratio":0.1}}}},"default":[{"start":2,"end":null,"profile":"late"}]}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

This uses the existing [attention schedule](../../docs/user_guide/diffusion/attention_backends/schedules.md)
interface. Profile options are complete configurations; include `rainfusion.ref_keep_ratio`
and `precision` in each profile. The immutable adapter is applied as a BF16
forward delta after each base projection, including quantized projections. It
requires DiT TP=1, resident DiT weights, eight denoising steps and scale 1;
Ulysses SP is supported. Per-request LoRA switching is disabled for this adapter.

For workloads that need more NPU memory, append the normal text-encoder offload
option to the command above:

```bash
--diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

## Request

Use the standard synchronous video endpoint:

```bash
curl --fail-with-body http://127.0.0.1:9098/v1/videos/sync \
    --form-string 'prompt=Use the object in <Picture 1> as the subject. A slow camera push-in in soft studio lighting, with quiet room ambience.' \
    --form-string 'short_edge=768' \
    --form-string 'aspect_ratio=9:16' \
    --form-string 'fps=24' \
    --form-string 'num_inference_steps=8' \
    --form-string 'flow_shift=12.0' \
    --form-string 'audio_flow_shift=3.0' \
    --form-string 'seed=42' \
    --form-string 'extra_params={"task":"ref2va","duration":4}' \
    --form 'input_references=@/path/to/reference.jpg;type=image/jpeg' \
    --output output.mp4
```

Repeat `input_references` in prompt order to supply several references.

## Controls and limits

| Control | Meaning |
| --- | --- |
| `rainfusion.ref_keep_ratio` | Per image, head and eligible target query block, keep `max(1, ceil(N * ratio))` of the image's N candidate key blocks. Default 1 preserves the original mask. |
| `sparsity` | Nominal fraction of pooled key blocks dropped. Protected blocks and reference selection change the realized mask. |
| `kv_storage` | `host` retains the existing layerwise host cache; `npu` keeps unquantized reference KV on the device and avoids host copies. Default `host`. |
| `kv_refresh_interval` | With eight-step inference, 8 refreshes at the first step and reuses references for seven steps. It is unrelated to INT8 cache storage. |
| `precision=mix` | Q/K are quantized to INT8 and V to FP8 inside attention. Inputs, cached reference KV and outputs remain BF16. |

Reference selection changes eligible attention connections. It does not shrink
the reference KV pool or remove reference keys from attention. Mixed/partial
image blocks, reference videos/audio, protected query rows and boundary blocks
retain their protection. Compact SP padding is removed before attention and
restored only for reverse communication. A5 rectangular execution supports
BF16 or mix with unquantized cached KV; ring parallelism is unsupported.

To compare with the earlier RBS working point, remove the attention schedule,
keep target sparsity at 0.8 and set `rainfusion.ref_keep_ratio=0.2`. Other settings should
match across both runs.

## Validation

This branch ports the complete scheme onto official main
`0b74da131f963ab2fd60f7eba813145672cf81e4`. Earlier results from the separate
`lq-slim` branch do not validate this revision. NPU end-to-end execution,
performance and output quality for this port remain unverified.

Focused CPU tests cover configuration, image ownership, step-profile dispatch,
resident/host KV values and cleanup, transport padding and LoRA
arithmetic/binding. Native attention calls in routing tests
are spies; no device operator or distributed execution is simulated.

For a minimal CPU environment, install PyTorch, safetensors, pytest,
pytest-mock and pytest-asyncio, then run:

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py -q
```

With the full repository test dependencies, the CI-aligned command is:

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py -m 'core_model and cpu' --run-level core_model -q
```
