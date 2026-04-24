#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-outputs/sft_merged}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code
