# MiniMax-H3 reference optimizations

These opt-in extensions add NPU-resident reference KV and per-image reference
block selection to MiniMax-H3 Ref2VA. Keep your existing model, LoRA, weight
quantization, sampling and serving configuration. The extensions do not install
a LoRA adapter or require an eight-step schedule.

## Install

Use the normal [Ascend installation](../../docs/getting_started/installation/npu.md)
with this source branch:

```bash
git clone --branch h3-ref-opt-main --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

Reference block selection additionally needs a MindIE-SD build containing
[PR #842](https://gitcode.com/Ascend/MindIE-SD/pull/842/discuss), including its
full-keep boundary correction. The dependency revision used during preparation
is `98fd99717182623bbfa952dd676f6865f2729f9d`; it is installed separately from
vLLM-Omni. No new native attention operator is introduced by this branch.

## Add reference KV reuse

On Ascend, append the existing KV configuration flag to your normal
`vllm serve ... --omni --task-type ref2va` command:

```bash
--kv-offload-config '{"mode":"on","kv_storage":"npu","kv_refresh_interval":8,"kv_host_quantization":"none"}'
```

The first step captures reference KV on the device. Subsequent steps reuse it
until the refresh interval is reached. For example, interval 8 refreshes at
steps 0, 8, 16 and so on; a different denoise step count does not need a code
change. Choose the interval for your workload and evaluate output quality.

`kv_storage` defaults to `host`, which preserves the existing layerwise host
prefetch. `npu` avoids host copies and staging-ring transfers but increases
device memory use. It requires unquantized cache storage. This restriction is
about stored reference KV; existing DiT weight quantization stays independent.

The cache is local to one denoise request and is released when it closes.
Model weights and the active adapter must remain fixed during that request.
No SP and pure Ulysses SP are supported; Ring and paged KV are unsupported.
Compact SP transport padding is excluded from attention and cache capture.

## Add reference-image sparsity

For an existing RainFusion deployment, retain its `block_sparse` options and add
`rainfusion.ref_keep_ratio` in the same attention spec. For example:

```bash
--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix"},"rainfusion":{"ref_keep_ratio":0.1}}}'
```

This option can be used with or without reference KV reuse. It requires positive
block sparsity and exact reference-image ownership, which MiniMax-H3 supplies.
Other backends do not consume the option; selecting RainFusion changes the
attention backend if your deployment previously used another one.

For each image, head and eligible target query block, selection keeps
`max(1, ceil(N * ratio))` of that image's N candidate key blocks. It can restore
connections removed by global top-k, then drops the other candidates in that
image group. Mixed image blocks and protected rows/blocks remain protected.
Reference videos and audio do not use this image keep ratio.

The ratio must be in `(0, 1]`. Default 1 preserves the original reference mask.
Selection changes attention connections; it does not shrink the reference KV
pool or remove reference keys from the sequence.

When combined with compact reference KV on A5, the rectangular attention path
supports BF16 and mix. Mix uses the existing MindIE-SD Q/K INT8 and V FP8
quantization functions and operator; cached KV and outputs remain BF16. This
integration adapts the shorter Q sequence and its block mapping. It does not
introduce a new quantization algorithm. Other precision/backend combinations
require separate compatibility validation.

## Optional per-step profiles

Use the existing [attention schedule](../../docs/user_guide/diffusion/attention_backends/schedules.md)
if your workload benefits from different sparsity at different steps. The
following is an optional working point; neither extension requires it:

```bash
--diffusion-attention-schedule '{"profiles":{"late":{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.85,"precision":"mix"},"rainfusion":{"ref_keep_ratio":0.1}}}},"default":[{"start":2,"end":null,"profile":"late"}]}'
```

Each profile is a complete configuration. Keep the base attention configuration
from the preceding example for steps outside the schedule. Continue using the
normal request endpoint and your existing sampling and LoRA flags.

## Validation

The contribution is based on official main
`0b74da131f963ab2fd60f7eba813145672cf81e4`, which already includes host reference
KV reuse. This branch adds device storage, compact SP alignment and image
selection integration. Earlier experiments of a complete LoRA/quantization/
reference configuration do not validate this extracted revision or attribute
performance to the two extensions individually.

CPU tests cover typed configuration, ownership, default-path preservation,
step-profile dispatch, host/resident KV values, refresh intervals and cleanup,
and SP padding/metadata. Native calls in routing tests are recording doubles;
they do not execute or simulate device operators or distributed communication.
NPU end-to-end execution, performance and output quality remain unverified.

For a minimal CPU environment, install PyTorch, pytest, pytest-mock and
pytest-asyncio. Run:

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py -q
```

With the full repository test dependencies, the CI-aligned command is:

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py -m 'core_model and cpu' --run-level core_model -q
```
