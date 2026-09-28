#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a COLMAP or Blender scene}"
MODEL_PATH="${MODEL_PATH:-./output}"

python train.py -s "$DATA_PATH" \
    --eval \
    --checkpoint_iterations 30000 \
    -m "$MODEL_PATH" \
    "$@"
