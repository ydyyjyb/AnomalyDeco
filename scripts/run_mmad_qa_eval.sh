#!/usr/bin/env bash
set -euo pipefail

ADAR_ROOT="${ADAR_ROOT:-$(pwd)}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
TRANSFORMERS_ROOT="${TRANSFORMERS_ROOT:-$ADAR_ROOT/Deco-main/transformers-4.57.6}"
MODEL_PATH="${MODEL_PATH:-$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct}"
DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/ODinW-13/data}"
MMAD_ROOT="${MMAD_ROOT:-$ADAR_ROOT/evaluation/MMAD}"
OUT_DIR="${OUT_DIR:-$ADAR_ROOT/Deco-main/runs/mmad_qa_eval_$(date +%Y%m%d_%H%M%S)}"

source "$ENV_PATH"

export PYTHONPATH="$TRANSFORMERS_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export CC="${CC:-/usr/bin/gcc}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vina04-triton-cache}"
mkdir -p "$TRITON_CACHE_DIR" "$OUT_DIR"

python "$ADAR_ROOT/scripts/repro_mmad_qa_eval.py" \
  --adar-root "$ADAR_ROOT" \
  --model-path "$MODEL_PATH" \
  --model-family auto \
  --data-root "$DATA_ROOT" \
  --mmad-root "$MMAD_ROOT" \
  --mmad-json "$MMAD_ROOT/mmad.json" \
  --download-mmad \
  --out-dir "$OUT_DIR" \
  --batch-size 2 \
  --image-size 512 \
  --grid-size 16 \
  --max-new-tokens 512 \
  --attn-implementation sdpa \
  --resume \
  "$@" 2>&1 | tee "$OUT_DIR/run.log"
