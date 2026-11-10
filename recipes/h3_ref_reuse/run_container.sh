#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail
: "${IMAGE:?Set IMAGE to a compatible, locally available Ascend image; see README.md}"
: "${MODEL_DIR:?Set MODEL_DIR to an absolute host model directory}"
: "${INPUT_DIR:?Set INPUT_DIR to an absolute host reference-media directory}"
USE_LORA=${USE_LORA:-1}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
IFS=, read -r -a cards <<< "$DEVICES"
args=()
for card in "${cards[@]}"; do
    [[ $card =~ ^[0-9]+$ ]] || { echo 'DEVICES must be comma-separated NPU numbers.' >&2; exit 2; }
    args+=(--device "/dev/davinci$card")
done
if [[ $USE_LORA == 1 ]]; then
    : "${LORA_PATH:?Set LORA_PATH to the absolute host adapter path}"
    args+=(-v "$LORA_PATH:/lora/adapter.safetensors:ro" -e LORA_PATH=/lora/adapter.safetensors)
fi
# One fixed configuration per process/container. Run comparison variants separately.
exec docker run --rm --runtime ascend --network host --shm-size 32g \
    --device /dev/davinci_manager --device /dev/hisi_hdc "${args[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$ROOT:/vllm-workspace/vllm-omni:ro" -v "$MODEL_DIR:/weights/MiniMax-H3:ro" \
    -v "$INPUT_DIR:/inputs:ro" -w /vllm-workspace/vllm-omni \
    -e MODEL_DIR=/weights/MiniMax-H3 -e "NPU_COUNT=${#cards[@]}" \
    -e "ASCEND_RT_VISIBLE_DEVICES=$DEVICES" -e "PORT=${PORT:-19136}" \
    -e "USE_LORA=$USE_LORA" -e "USE_REF_CACHE=${USE_REF_CACHE:-1}" \
    --entrypoint /bin/bash "$IMAGE" recipes/h3_ref_reuse/serve.sh
