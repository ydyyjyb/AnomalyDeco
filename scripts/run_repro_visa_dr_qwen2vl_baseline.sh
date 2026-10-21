#!/usr/bin/env bash
set -euo pipefail

ADAR_ROOT="${ADAR_ROOT:-/home/vina04/ADAR}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
MODEL="${MODEL:-qwen2vl-7b}"
OUT_DIR="${OUT_DIR:-$ADAR_ROOT/Deco-main/runs/visa_dr_${MODEL}_paper_baseline_$(date +%Y%m%d_%H%M%S)}"

source "$ENV_PATH"
mkdir -p "$OUT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ADAR_ROOT/scripts:$ADAR_ROOT/Deco-main/transformers-4.57.6/src:${PYTHONPATH:-}"

python "$ADAR_ROOT/scripts/repro_visa_dr_qwen2vl_baseline.py" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --mode all \
  --local-files-only \
  "$@" 2>&1 | tee "$OUT_DIR/run.log"

echo "Results: $OUT_DIR"
