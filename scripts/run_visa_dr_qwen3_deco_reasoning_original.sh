#!/usr/bin/env bash
set -euo pipefail

export ADAR_ROOT="${ADAR_ROOT:-/home/vina04/ADAR}"
export ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
export TRANSFORMERS_ROOT="${TRANSFORMERS_ROOT:-$ADAR_ROOT/Deco-main/transformers-4.57.6}"
export MODEL_PATH="${MODEL_PATH:-$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct}"
export DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/VisA-DR-official}"
export VISA_DR_JSON="${VISA_DR_JSON:-$DATA_ROOT/VisA/visa_dr.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ADAR_ROOT/scripts:$TRANSFORMERS_ROOT/src:$TRANSFORMERS_ROOT/examples/deco:${PYTHONPATH:-}"

source "$ENV_PATH"
cd "$ADAR_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ADAR_ROOT/Deco-main/runs/visa_dr_qwen3_deco_reasoning_original_$STAMP}"

python "$ADAR_ROOT/scripts/repro_visa_dr_qwen3_deco_reasoning_original.py" \
  --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" \
  --visa-dr-json "$VISA_DR_JSON" \
  --out-dir "$OUT_DIR" \
  --variants qwen3_original qwen3_deco \
  --local-files-only \
  --max-image-side 1024 \
  --max-new-tokens 4096 \
  --log-every 20 \
  --deco-layers 20:29 \
  --deco-alpha 0.6 \
  --deco-threshold-top-p 0.9 \
  --deco-threshold-top-k 20

echo "All results: $OUT_DIR"
