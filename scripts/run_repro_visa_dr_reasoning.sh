#!/usr/bin/env bash
set -euo pipefail

ADAR_ROOT="${ADAR_ROOT:-/home/vina04/ADAR}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
MODEL_PATH="${MODEL_PATH:-$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct}"
DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/ODinW-13/data}"
VISA_DR_JSON="${VISA_DR_JSON:-$ADAR_ROOT/evaluation/VisA-DR/visa_dr.json}"
CASES_JSON="${CASES_JSON:-$ADAR_ROOT/Deco-main/runs/six_datasets_0727_rerun_20260806_081642/visa/up_cases.json}"
OUT_DIR="${OUT_DIR:-$ADAR_ROOT/Deco-main/runs/visa_dr_reasoning_$(date +%Y%m%d_%H%M%S)}"
GENERATION_PROFILE="${GENERATION_PROFILE:-official_fixed}"
USE_DECO_GENERATION="${USE_DECO_GENERATION:-0}"
DECO_ALPHA="${DECO_ALPHA:-0.6}"
DECO_THRESHOLD_TOP_P="${DECO_THRESHOLD_TOP_P:-0.9}"
DECO_THRESHOLD_TOP_K="${DECO_THRESHOLD_TOP_K:-20}"
DECO_LAYERS="${DECO_LAYERS:-20:29}"

source "$ENV_PATH"
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ADAR_ROOT/Deco-main/transformers-4.57.6/src:${PYTHONPATH:-}"

DECO_ARGS=()
if [[ "$USE_DECO_GENERATION" == "1" ]]; then
  DECO_ARGS=(
    --use-deco-generation
    --deco-alpha "$DECO_ALPHA"
    --deco-threshold-top-p "$DECO_THRESHOLD_TOP_P"
    --deco-threshold-top-k "$DECO_THRESHOLD_TOP_K"
    --deco-layers "$DECO_LAYERS"
  )
fi

python "$ADAR_ROOT/scripts/repro_visa_dr_reasoning.py" \
  --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" \
  --visa-dr-json "$VISA_DR_JSON" \
  --cases-json "$CASES_JSON" \
  --out-dir "$OUT_DIR" \
  --score-key hdec_score \
  --generation-profile "$GENERATION_PROFILE" \
  --generate \
  --local-files-only \
  --attn-implementation sdpa \
  "${DECO_ARGS[@]}" \
  "$@" 2>&1 | tee "$OUT_DIR/run.log"

echo "Results: $OUT_DIR"
