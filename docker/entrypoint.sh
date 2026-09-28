#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
DCSHARP container

  train     python train.py
  render    python render.py
  metrics   python metrics.py
  eval      python full_eval.py
  help      show this message

Any other command is executed as-is, for example:

  docker run --rm --gpus all dcsharp python -c "import torch; print(torch.__version__)"

Training one scene (mount the COLMAP or Blender scene on /data):

  docker run --rm --gpus all \
    -v "$DATA_DIR":/data \
    -v "$OUTPUT_DIR":/output \
    dcsharp train -s /data --eval -m /output --checkpoint_iterations 30000

Full benchmark train, render, and metrics:

  docker run --rm --gpus all \
    -v "$M360_PATH":/data/360 \
    -v "$TAT_PATH":/data/tandt \
    -v "$DB_PATH":/data/db \
    -v "$OUTPUT_DIR":/output \
    dcsharp eval -m360 /data/360 -tat /data/tandt -db /data/db --output_path /output
EOF
}

if [ "$#" -eq 0 ]; then
    set -- help
fi

case "$1" in
    help|-h|--help)
        usage
        ;;
    train)
        shift
        exec python train.py "$@"
        ;;
    render)
        shift
        exec python render.py "$@"
        ;;
    metrics)
        shift
        exec python metrics.py "$@"
        ;;
    eval|full-eval)
        shift
        exec python full_eval.py "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
