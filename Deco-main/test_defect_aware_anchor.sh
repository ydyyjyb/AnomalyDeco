#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DeCo defect-aware anchor selection test script
#
# Usage (lightweight checks only):
#   bash test_defect_aware_anchor.sh
#
# Usage (with end-to-end smoke test):
#   MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-4B-Instruct" \
#   DATASET_ROOT="/path/to/mvtec" \
#   CATEGORY="bottle" \
#   bash test_defect_aware_anchor.sh
#
# Optional env vars:
#   PYTHON_BIN            (default: python)
#   REPO_ROOT             (default auto-detect)
#   DECO_DIR              (default: $REPO_ROOT/examples/deco)
#   OUTPUT_DIR            (default: <script_dir>/runs/_tmp_defect_aware_test)
#   CUDA_DEVICE           (default: empty; e.g. 0 / 1 / 0,1. If set, exports CUDA_VISIBLE_DEVICES)
#   IMAGE_PATH            (if unset, auto-pick first image under CATEGORY/test/*)
#   SHOT                  (default: 3)
#   MAX_NEW_TOKENS        (default: 256 for eval; 8 for analyze)
#   LAMBDA_NORM           (default: 1.0)
#   ETA_HEDGE             (default: 0.15)
#   ENABLE_HEDGE          (default: 1 -> enable)
#   USE_LOGSUMEXP         (default: 0 -> max mode)
#   GAMMA                 (default: 8.0)
#   DECO_DEBUG            (default: 0)
#   ANCHOR_DEFECT_WORDS   (default built-in list below)
#   ANCHOR_NORM_WORDS     (default built-in list below)
#   ANCHOR_HEDGE_WORDS    (default built-in list below)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${SCRIPT_DIR}/transformers-4.57.6/src/transformers" ]]; then
  REPO_ROOT_DEFAULT="${SCRIPT_DIR}/transformers-4.57.6"
else
  echo "Failed to auto-detect transformers repo root from SCRIPT_DIR=${SCRIPT_DIR}"
  echo "Please set REPO_ROOT manually (e.g. REPO_ROOT=/path/to/transformers-4.57.6)."
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
DECO_DIR="${DECO_DIR:-${REPO_ROOT}/examples/deco}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/defect_aware_test_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_DEVICE:-}"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
if [[ -n "${CUDA_DEVICE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi

ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal,broken_small,broken_large,contamination}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"

LAMBDA_NORM="${LAMBDA_NORM:-1.0}"
ETA_HEDGE="${ETA_HEDGE:-0.15}"
ENABLE_HEDGE="${ENABLE_HEDGE:-1}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-1}"
GAMMA="${GAMMA:-8.0}"
DECO_DEBUG="${DECO_DEBUG:-0}"
SHOT="${SHOT:-0}"
MODEL_NAME_OR_PATH=/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl
DATASET_ROOT=/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec


echo "[1/3] Python syntax checks"
"${PYTHON_BIN}" -m py_compile \
  "${REPO_ROOT}/src/transformers/generation/utils.py" \
  "${DECO_DIR}/eval_mvtec_qwen3_vl_thinking.py" \
  "${DECO_DIR}/analyze_case_qwen3_vl_deco.py" \
  "${DECO_DIR}/selfcheck_defect_aware_anchor.py"

echo "[2/3] Defect-aware anchor unit self-check"
"${PYTHON_BIN}" "${DECO_DIR}/selfcheck_defect_aware_anchor.py"

if [[ -z "${MODEL_NAME_OR_PATH:-}" || -z "${DATASET_ROOT:-}" ]]; then
  cat <<'EOF'
[3/3] End-to-end smoke test skipped.
Reason: MODEL_NAME_OR_PATH or DATASET_ROOT is not set.

To run smoke test, set:
  MODEL_NAME_OR_PATH=<hf-model-or-local-path>
  DATASET_ROOT=<mvtec-root>
Optional:
  CATEGORY=<mvtec-category>   (default: bottle)
  IMAGE_PATH=<single-image>   (auto-picked if unset)
EOF
  exit 0
fi

CATEGORY="${CATEGORY:-bottle}"
EVAL_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS:-16000}"
EVAL_OUT="${OUTPUT_DIR}/eval_defect_aware_${CATEGORY}.json"
ANALYZE_OUT="${OUTPUT_DIR}/analyze_defect_aware_${CATEGORY}.json"

ENABLE_HEDGE_FLAG=""
if [[ "${ENABLE_HEDGE}" == "1" ]]; then
  ENABLE_HEDGE_FLAG="--enable_hedge"
fi

USE_LOGSUMEXP_FLAG=""
if [[ "${USE_LOGSUMEXP}" == "1" ]]; then
  USE_LOGSUMEXP_FLAG="--use_logsumexp"
fi

DECO_DEBUG_FLAG=""
if [[ "${DECO_DEBUG}" == "1" ]]; then
  DECO_DEBUG_FLAG="--deco_debug"
fi

if [[ -z "${IMAGE_PATH:-}" ]]; then
  SEARCH_DIR="${DATASET_ROOT}/${CATEGORY}/test"
  if [[ -d "${SEARCH_DIR}" ]]; then
    IMAGE_PATH="$(
      find "${SEARCH_DIR}" -type f \
        \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.webp" -o -iname "*.tif" -o -iname "*.tiff" \) \
        | head -n 1 || true
    )"
  else
    IMAGE_PATH=""
  fi
fi

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "No test image found under ${DATASET_ROOT}/${CATEGORY}/test. Please set IMAGE_PATH manually."
  exit 1
fi

echo "[3/3] End-to-end smoke test"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "CATEGORY=${CATEGORY}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DECO_DIR=${DECO_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set>}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

"${PYTHON_BIN}" "${DECO_DIR}/eval_mvtec_qwen3_vl_thinking.py" \
  --dataset_root "${DATASET_ROOT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_json "${EVAL_OUT}" \
  --shots 0 \
  --category "${CATEGORY}" \
  --max_samples_per_defect "${MAX_SAMPLES_PER_DEFECT:-1}" \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
  --attn_implementation flash_attention_2 \
  --allow_attn_fallback \
  --use_deco \
  --anchor_strategy defect_aware \
  --lambda_norm "${LAMBDA_NORM}" \
  --eta_hedge "${ETA_HEDGE}" \
  --gamma "${GAMMA}" \
  --anchor_defect_words "${ANCHOR_DEFECT_WORDS}" \
  --anchor_norm_words "${ANCHOR_NORM_WORDS}" \
  --anchor_hedge_words "${ANCHOR_HEDGE_WORDS}" \
  ${ENABLE_HEDGE_FLAG} \
  ${USE_LOGSUMEXP_FLAG} \
  ${DECO_DEBUG_FLAG}

"${PYTHON_BIN}" "${DECO_DIR}/analyze_case_qwen3_vl_deco.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_root "${DATASET_ROOT}" \
  --category "${CATEGORY}" \
  --image_path "${IMAGE_PATH}" \
  --output_json "${ANALYZE_OUT}" \
  --shot "${SHOT}" \
  --max_new_tokens "${ANALYZE_MAX_NEW_TOKENS}" \
  --attn_implementation flash_attention_2 \
  --allow_attn_fallback \
  --anchor_strategy defect_aware \
  --lambda_norm "${LAMBDA_NORM}" \
  --eta_hedge "${ETA_HEDGE}" \
  --gamma "${GAMMA}" \
  --anchor_defect_words "${ANCHOR_DEFECT_WORDS}" \
  --anchor_norm_words "${ANCHOR_NORM_WORDS}" \
  --anchor_hedge_words "${ANCHOR_HEDGE_WORDS}" \
  ${ENABLE_HEDGE_FLAG} \
  ${USE_LOGSUMEXP_FLAG} \
  ${DECO_DEBUG_FLAG}

echo "Done."
echo "Eval result:    ${EVAL_OUT}"
echo "Analyze result: ${ANALYZE_OUT}"
