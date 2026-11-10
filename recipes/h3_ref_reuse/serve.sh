#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the MiniMax-H3 model directory}"
NPU_COUNT=${NPU_COUNT:-8}
PORT=${PORT:-19136}
USE_LORA=${USE_LORA:-1}
USE_REF_CACHE=${USE_REF_CACHE:-1}
if [[ $NPU_COUNT != 4 && $NPU_COUNT != 8 ]]; then
    echo 'NPU_COUNT must be 4 or 8 (D045 used 8).' >&2
    exit 2
fi
if [[ $USE_LORA != 0 && $USE_LORA != 1 ]] || [[ $USE_REF_CACHE != 0 && $USE_REF_CACHE != 1 ]]; then
    echo 'USE_LORA and USE_REF_CACHE must be 0 or 1.' >&2
    exit 2
fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
# This enables scalar pipeline timings only. Never inherit a device trace session.
unset PROFILING_MODE PROFILING_OPTIONS MSPROF_CONFIG_PATH ASCEND_PROFILER_OUTPUT
python recipes/h3_ref_reuse/verify.py --runtime

export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_NPU_SOCKET_PORT_RANGE=auto
unset RESEARCH_LARRY_LORA_PATH
if [[ $USE_LORA == 1 ]]; then
    : "${LORA_PATH:?Set LORA_PATH to minimax_h3_turbo_v4_step600_ema.safetensors}"
    printf '%s  %s\n' '5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3' "$LORA_PATH" | sha256sum -c -
    export RESEARCH_LARRY_LORA_PATH=$LORA_PATH
fi
cache_args=()
if [[ $USE_REF_CACHE == 1 ]]; then
    cache_args=(--kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}')
fi
exec vllm serve "$MODEL_DIR" --omni --host 127.0.0.1 --port "$PORT" \
    --trust-remote-code --task-type ref2va --num-gpus "$NPU_COUNT" --usp "$NPU_COUNT" --ring 1 \
    --text-encoder-tp-size "$NPU_COUNT" --vae-parallel-mode tile --vae-use-tiling \
    --vae-patch-parallel-size "$NPU_COUNT" --enable-diffusion-pipeline-profiler \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0,"end_step":0}}}' \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' "${cache_args[@]}"
