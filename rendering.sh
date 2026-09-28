#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the scene used for training}"
MODEL_PATH="${MODEL_PATH:-./output}"

python render.py -m "$MODEL_PATH" -s "$DATA_PATH" "$@"
