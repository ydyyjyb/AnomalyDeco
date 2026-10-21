#!/usr/bin/env bash
set -euo pipefail

# Add-on runner for visa_label_summary.py / probe_visa_layer_probs.py.
# This does not replace eval_visa_qwen3_vl_thinking.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/transformers-4.57.6}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/../evaluation/ODinW-13/data/VisA}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/qwen3_4b_vl_instruct}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/visa_layer_probe_$(date +%Y%m%d_%H%M%S)}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_vl}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-}"
if [[ "${MODEL_FAMILY}" == "llava" && -z "${VISION_TOWER_PATH}" ]]; then
  for cand in \
    "${SCRIPT_DIR}/clip-vit-large-patch14-336" \
    "${SCRIPT_DIR}/openai/clip-vit-large-patch14-336" \
    "${SCRIPT_DIR}/models/clip-vit-large-patch14-336"; do
    if [[ -f "${cand}/config.json" ]]; then
      VISION_TOWER_PATH="${cand}"
      break
    fi
  done
fi
if [[ -n "${VISION_TOWER_PATH}" ]]; then
  export VISION_TOWER_PATH
fi

# CUDA_VISIBLE_DEVICES should use comma-separated ids, not spaces.
GPU_ID="${GPU_ID:-5,6}"

ALL_CATEGORIES="${ALL_CATEGORIES:-candle,capsules,cashew,chewinggum,fryum,macaroni1,macaroni2,pipe_fryum,pcb1,pcb2,pcb3,pcb4}"

SHOT="${SHOT:-1}"
if [[ -z "${MAX_NEW_TOKENS+x}" ]]; then
  if [[ "${MODEL_FAMILY}" == "llava" ]]; then
    MAX_NEW_TOKENS="128"
  else
    MAX_NEW_TOKENS="128"
  fi
fi
ASSISTANT_PREFIX="${ASSISTANT_PREFIX:-{\"label\": \"}"

ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"

if [[ -z "${USE_DECO+x}" ]]; then
  if [[ "${MODEL_FAMILY}" == "llava" ]]; then
    USE_DECO="0"
  else
    USE_DECO="1"
  fi
fi
DECO_ALPHA="${DECO_ALPHA:-0.60}"
# 0.40 often keeps only one high-confidence token. Default to 0.90 here.
DECO_TOP_P="${DECO_TOP_P:-0.90}"
DECO_TOP_K="${DECO_TOP_K:-32}"
START_LAYER="${START_LAYER:-4}"
END_LAYER="${END_LAYER:-11}"
LAMBDA_NORM="${LAMBDA_NORM:-0.50}"
ETA_HEDGE="${ETA_HEDGE:-0.10}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-1}"
GAMMA="${GAMMA:-8.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-1}"

ANCHOR_SMOOTH="${ANCHOR_SMOOTH:-1}"
ANCHOR_REMOVE_OUTLIERS="${ANCHOR_REMOVE_OUTLIERS:-1}"
ANCHOR_HAMPEL_WINDOW="${ANCHOR_HAMPEL_WINDOW:-2}"
ANCHOR_HAMPEL_N_SIGMA="${ANCHOR_HAMPEL_N_SIGMA:-3.0}"
ANCHOR_SMOOTH_WINDOW="${ANCHOR_SMOOTH_WINDOW:-3}"
ANCHOR_SMOOTH_BETA="${ANCHOR_SMOOTH_BETA:-0.70}"

FIXED_SUPPORT_IMAGES="${FIXED_SUPPORT_IMAGES:-}"

EXTRA_ARGS=("$@")

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

FLAGS=(
  --dataset_root "${DATASET_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUT_DIR}"
  --model_family "${MODEL_FAMILY}"
  --categories "${ALL_CATEGORIES}"
  --shot "${SHOT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --assistant_prefix "${ASSISTANT_PREFIX}"
  --anchor_defect_words "${ANCHOR_DEFECT_WORDS}"
  --anchor_norm_words "${ANCHOR_NORM_WORDS}"
  --anchor_hedge_words "${ANCHOR_HEDGE_WORDS}"
  --allow_attn_fallback
)

if [[ -n "${VISION_TOWER_PATH}" ]]; then
  FLAGS+=(--vision_tower_path "${VISION_TOWER_PATH}")
fi

if [[ -n "${FIXED_SUPPORT_IMAGES}" ]]; then
  FLAGS+=(--fixed_support_images "${FIXED_SUPPORT_IMAGES}")
fi

if [[ "${USE_DECO}" == "1" ]]; then
  FLAGS+=(
    --use_deco
    --anchor_strategy defect_aware
    --alpha "${DECO_ALPHA}"
    --threshold_top_p "${DECO_TOP_P}"
    --threshold_top_k "${DECO_TOP_K}"
    --start_layer "${START_LAYER}"
    --end_layer "${END_LAYER}"
    --lambda_norm "${LAMBDA_NORM}"
    --eta_hedge "${ETA_HEDGE}"
    --gamma "${GAMMA}"
  )

  if [[ "${USE_LOGSUMEXP}" == "1" ]]; then
    FLAGS+=(--use_logsumexp)
  fi
  if [[ "${ENABLE_HEDGE}" == "1" ]]; then
    FLAGS+=(--enable_hedge)
  fi
  if [[ "${ANCHOR_SMOOTH}" == "1" ]]; then
    FLAGS+=(--anchor_smooth)
  fi
  if [[ "${ANCHOR_REMOVE_OUTLIERS}" == "1" ]]; then
    FLAGS+=(--anchor_remove_outliers)
  fi

  FLAGS+=(
    --anchor_hampel_window "${ANCHOR_HAMPEL_WINDOW}"
    --anchor_hampel_n_sigma "${ANCHOR_HAMPEL_N_SIGMA}"
    --anchor_smooth_window "${ANCHOR_SMOOTH_WINDOW}"
    --anchor_smooth_beta "${ANCHOR_SMOOTH_BETA}"
  )
fi

echo "===== Layer Probe Runtime Config ====="
echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "VISION_TOWER_PATH=${VISION_TOWER_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "GPU_ID=${GPU_ID}"
echo "ALL_CATEGORIES=${ALL_CATEGORIES}"
echo "SHOT=${SHOT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "ANCHOR_DEFECT_WORDS=${ANCHOR_DEFECT_WORDS}"
echo "ANCHOR_NORM_WORDS=${ANCHOR_NORM_WORDS}"
echo "ANCHOR_HEDGE_WORDS=${ANCHOR_HEDGE_WORDS}"
echo "USE_DECO=${USE_DECO}"
echo "DECO_TOP_P=${DECO_TOP_P}"
echo "DECO_TOP_K=${DECO_TOP_K}"
echo "LAYERS=[${START_LAYER},${END_LAYER})"
echo "FUSION_WEIGHT=alpha * unsmoothed_anchor_score_norm3"
echo "====================================="

PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python examples/deco/visa_label_summary.py \
    "${FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "[DONE] ${OUT_DIR}/visa_layer_probability_probe.json"
echo "[PLOTS] ${OUT_DIR}/plots"
