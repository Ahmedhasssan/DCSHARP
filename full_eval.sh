#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
M360_PATH="${M360_PATH:?Set M360_PATH to the Mip-NeRF 360 root}"
TAT_PATH="${TAT_PATH:?Set TAT_PATH to the Tanks and Temples root}"
DB_PATH="${DB_PATH:?Set DB_PATH to the Deep Blending root}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval}"

python full_eval.py \
    -m360 "$M360_PATH" \
    -tat "$TAT_PATH" \
    -db "$DB_PATH" \
    --output_path "$OUTPUT_PATH" \
    "$@"
