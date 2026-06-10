#!/bin/bash

MODEL=$1
PORT=$2
SEED=$3
SWAP_SPACE=$4

# Read GPU IDX to use. Default is 0
GPU_IDX=${5:-0}
DTYPE="${VLLM_DTYPE:-}"

export VLLM_HF_FOLDER_CACHE_FILE=$HF_HOME/vllm_hf_folder_cache.json

if [ -z "$DTYPE" ]; then
    DTYPE="$(CUDA_VISIBLE_DEVICES=$GPU_IDX python3 - <<'PY'
import os
import torch

gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
major, _minor = torch.cuda.get_device_capability(gpu_idx)
print("bfloat16" if major >= 8 else "half")
PY
)"
fi

CUDA_VISIBLE_DEVICES=$GPU_IDX python3 -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--host 0.0.0.0 \
	--port "$PORT" \
	--seed "$SEED" \
	--swap-space "$SWAP_SPACE" \
	--dtype "$DTYPE"
