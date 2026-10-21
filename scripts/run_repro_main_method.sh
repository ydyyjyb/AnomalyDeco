#!/usr/bin/env bash
set -euo pipefail

ADAR_ROOT="${ADAR_ROOT:-$(pwd)}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
TRANSFORMERS_ROOT="${TRANSFORMERS_ROOT:-$ADAR_ROOT/Deco-main/transformers-4.57.6}"
MODEL_PATH="${MODEL_PATH:-$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct}"
DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/ODinW-13/data}"
OUT_DIR="${OUT_DIR:-$ADAR_ROOT/Deco-main/runs/repro_main_method_$(date +%Y%m%d_%H%M%S)}"

source "$ENV_PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CC="${CC:-/usr/bin/gcc}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vina04-triton-cache}"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR"

python "$ADAR_ROOT/scripts/repro_main_method.py" \
  --adar-root "$ADAR_ROOT" \
  --transformers-root "$TRANSFORMERS_ROOT" \
  --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --datasets visa mvtec aitex btad elpv mpdd brainmri headct br35h \
  --shots 0 1 2 3 4 \
  --local-files-only \
  --stream-jsonl \
  --resume \
  "$@" 2>&1 | tee "$OUT_DIR/repro_main_method.log"
