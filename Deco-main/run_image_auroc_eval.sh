#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Image-level AUROC runner for Qwen3-VL (+/- DeCo)
#
# This script wraps:
#   examples/deco/eval_image_auroc_qwen3_vl_deco.py
#
# Supports:
# - MVTec and VisA
# - category subset evaluation
# - baseline / DeCo / both
#
# Required env vars:
#   MODEL_NAME_OR_PATH   HF model id or local model path
#   DATASET_ROOT         dataset root path
#
# Optional env vars (common):
#   PYTHON_BIN           default: python
#   REPO_ROOT            default: auto-detect from script location
#                      (supports script at same level as transformers-4.57.6,
#                       inside transformers-4.57.6, or under examples/deco)
#   DECO_DIR             default: $REPO_ROOT/examples/deco
#   STRICT_LOCAL_TRANSFORMERS 0|1 (default: 1)
#                      1 means fail fast if `import transformers` does not come
#                      from $REPO_ROOT/src/transformers.
#   CUDA_DEVICE         optional, e.g. 0 / 1 / 0,1
#                      if set, exports CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
#   DATASET_TYPE         auto|mvtec|visa   (default: auto)
#   OUTPUT_DIR           default: <script_dir>/runs/image_auroc
#   OUTPUT_BASENAME      default: image_auroc
#   CATEGORY             single category, optional
#   CATEGORIES           comma-separated categories, optional
#   MAX_SAMPLES_PER_SUBTYPE  optional integer
#   SHOT                 default: 3
#   SEED                 default: 42
#   SCORE_MODE           one_minus_good|max_defect|defect_minus_good|defect_over_sum
#                        default: one_minus_good
#   ANALYSIS_STEP        default: 0
#   MAX_NEW_TOKENS       default: 1
#   DO_SAMPLE            0|1  default: 0
#   TEMPERATURE          default: 0.7
#   NUM_BEAMS            default: 1
#   TOP_P                default: 0.95
#   TOP_K                default: 20
#   REPETITION_PENALTY   default: 1.0
#   MIN_PIXELS           default: 409600
#   MAX_PIXELS           default: 6291456
#   ATTN_IMPLEMENTATION  flash_attention_2|sdpa|eager (default: flash_attention_2)
#   ALLOW_ATTN_FALLBACK  0|1 (default: 1)
#   PRINT_IMAGE_NAME     0|1 (default: 0)
#
# Optional env vars (DeCo):
#   MODE                 baseline|deco|both (default: deco)
#   ALPHA                default: 0.6
#   THRESHOLD_TOP_P      default: 0.9
#   THRESHOLD_TOP_K      default: 20
#   START_LAYER          default: 20
#   END_LAYER            default: 29
#   ANCHOR_STRATEGY      default|defect_aware (default: default)
#   LAMBDA_NORM          default: 1.0
#   ETA_HEDGE            default: 0.0
#   ENABLE_HEDGE         0|1 (default: 0)
#   USE_LOGSUMEXP        0|1 (default: 0)
#   GAMMA                default: 8.0
#   ANCHOR_DEFECT_WORDS  optional csv
#   ANCHOR_NORM_WORDS    optional csv
#   ANCHOR_HEDGE_WORDS   optional csv
#   DECO_DEBUG           0|1 (default: 0)
#
# Example:
#   MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-4B-Instruct" \
#   DATASET_ROOT="/data/mvtec" \
#   DATASET_TYPE="mvtec" \
#   MODE="both" \
#   CATEGORY="bottle" \
#   bash run_image_auroc_eval.sh
# -----------------------------------------------------------------------------

# 默认值仅在没有通过环境变量传入时生效（便于 run_image_auroc_all_datasets.sh 等批量脚本传参）
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-/data/mvtec}"
DATASET_TYPE="${DATASET_TYPE:-auto}"
MODE="${MODE:-deco}"
CATEGORY="${CATEGORY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_repo_root() {
  local candidates=(
    "${SCRIPT_DIR}/transformers-4.57.6"
  )
  local cand=""
  for cand in "${candidates[@]}"; do
    if [[ -d "${cand}/src/transformers" && -f "${cand}/examples/deco/eval_image_auroc_qwen3_vl_deco.py" ]]; then
      (cd "${cand}" && pwd)
      return 0
    fi
  done
  return 1
}

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
else
  if ! REPO_ROOT="$(detect_repo_root)"; then
    echo "[ERROR] Failed to auto-detect REPO_ROOT from SCRIPT_DIR=${SCRIPT_DIR}"
    echo "Set REPO_ROOT manually, e.g.: REPO_ROOT=/path/to/transformers-4.57.6"
    exit 1
  fi
fi

DECO_DIR="${DECO_DIR:-${REPO_ROOT}/examples/deco}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STRICT_LOCAL_TRANSFORMERS="${STRICT_LOCAL_TRANSFORMERS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "${MODEL_NAME_OR_PATH}" || -z "${DATASET_ROOT}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH and DATASET_ROOT are required."
  exit 1
fi

DATASET_TYPE="${DATASET_TYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/image_auroc}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-image_auroc}"
CATEGORY="${CATEGORY:-}"
CATEGORIES="${CATEGORIES:-}"
MAX_SAMPLES_PER_SUBTYPE="${MAX_SAMPLES_PER_SUBTYPE:-}"

SHOT="${SHOT:-0}"
SEED="${SEED:-42}"
SCORE_MODE="${SCORE_MODE:-one_minus_good}"
ANALYSIS_STEP="${ANALYSIS_STEP:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"

DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_BEAMS="${NUM_BEAMS:-1}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MIN_PIXELS="${MIN_PIXELS:-409600}"
MAX_PIXELS="${MAX_PIXELS:-6291456}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ALLOW_ATTN_FALLBACK="${ALLOW_ATTN_FALLBACK:-1}"
PRINT_IMAGE_NAME="${PRINT_IMAGE_NAME:-0}"

MODE="${MODE:-deco}"  # baseline | deco | both
ALPHA="${ALPHA:-0.6}"
THRESHOLD_TOP_P="${THRESHOLD_TOP_P:-0.9}"
THRESHOLD_TOP_K="${THRESHOLD_TOP_K:-20}"
START_LAYER="${START_LAYER:-20}"
END_LAYER="${END_LAYER:-29}"
ANCHOR_STRATEGY="${ANCHOR_STRATEGY:-default}"
LAMBDA_NORM="${LAMBDA_NORM:-1.0}"
ETA_HEDGE="${ETA_HEDGE:-0.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-0}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-0}"
GAMMA="${GAMMA:-8.0}"
ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-}"
DECO_DEBUG="${DECO_DEBUG:-0}"

PY_SCRIPT="${DECO_DIR}/eval_image_auroc_qwen3_vl_deco.py"
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Missing script: ${PY_SCRIPT}"
  exit 1
fi

# Use only REPO_ROOT/src for transformers (avoid picking up Deco-main/transformers from CWD or env).
export PYTHONPATH="${REPO_ROOT}/src"
if [[ -n "${CUDA_DEVICE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi

verify_transformers_source() {
  local expected_prefix
  expected_prefix="$(cd "${REPO_ROOT}/src/transformers" && pwd)"
  local info
  # Run Python from REPO_ROOT with clean PYTHONPATH so CWD does not shadow transformers.
  info="$(
    cd "${REPO_ROOT}" && PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" - <<'PY'
import os
import transformers
print(os.path.abspath(transformers.__file__))
print(getattr(transformers, "__version__", "unknown"))
PY
  )"
  local file_path
  local version
  file_path="$(echo "${info}" | sed -n '1p')"
  version="$(echo "${info}" | sed -n '2p')"

  echo "transformers_file: ${file_path}"
  echo "transformers_version: ${version}"
  echo "expected_prefix: ${expected_prefix}"

  if [[ "${file_path}" != "${expected_prefix}"* ]]; then
    echo "[ERROR] transformers import is not from ${expected_prefix}"
    echo "        current import path: ${file_path}"
    echo "        Please check PYTHONPATH / environment."
    return 1
  fi
  return 0
}

if [[ "${STRICT_LOCAL_TRANSFORMERS}" == "1" ]]; then
  verify_transformers_source || exit 1
fi

mkdir -p "${OUTPUT_DIR}"
# Resolve to absolute paths so they work when Python is run from REPO_ROOT.
[[ -n "${DATASET_ROOT}" && "${DATASET_ROOT}" != /* ]] && DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
[[ -d "${OUTPUT_DIR}" ]] && OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

build_common_args() {
  local -n _out_arr=$1
  _out_arr=(
    "--dataset_root" "${DATASET_ROOT}"
    "--dataset_type" "${DATASET_TYPE}"
    "--model_name_or_path" "${MODEL_NAME_OR_PATH}"
    "--shot" "${SHOT}"
    "--seed" "${SEED}"
    "--score_mode" "${SCORE_MODE}"
    "--analysis_step" "${ANALYSIS_STEP}"
    "--max_new_tokens" "${MAX_NEW_TOKENS}"
    "--temperature" "${TEMPERATURE}"
    "--num_beams" "${NUM_BEAMS}"
    "--top_p" "${TOP_P}"
    "--top_k" "${TOP_K}"
    "--repetition_penalty" "${REPETITION_PENALTY}"
    "--min_pixels" "${MIN_PIXELS}"
    "--max_pixels" "${MAX_PIXELS}"
    "--attn_implementation" "${ATTN_IMPLEMENTATION}"
  )

  if [[ -n "${CATEGORY}" ]]; then
    _out_arr+=("--category" "${CATEGORY}")
  fi
  if [[ -n "${CATEGORIES}" ]]; then
    _out_arr+=("--categories" "${CATEGORIES}")
  fi
  if [[ -n "${MAX_SAMPLES_PER_SUBTYPE}" ]]; then
    _out_arr+=("--max_samples_per_subtype" "${MAX_SAMPLES_PER_SUBTYPE}")
  fi
  if [[ "${DO_SAMPLE}" == "1" ]]; then
    _out_arr+=("--do_sample")
  fi
  if [[ "${ALLOW_ATTN_FALLBACK}" == "1" ]]; then
    _out_arr+=("--allow_attn_fallback")
  fi
  if [[ "${PRINT_IMAGE_NAME}" == "1" ]]; then
    _out_arr+=("--print_image_name")
  fi
}

run_once() {
  local mode_name="$1"   # baseline or deco
  local output_json="$2"

  local args=()
  build_common_args args
  args+=("--output_json" "${output_json}")

  if [[ "${mode_name}" == "deco" ]]; then
    args+=(
      "--use_deco"
      "--alpha" "${ALPHA}"
      "--threshold_top_p" "${THRESHOLD_TOP_P}"
      "--threshold_top_k" "${THRESHOLD_TOP_K}"
      "--start_layer" "${START_LAYER}"
      "--end_layer" "${END_LAYER}"
      "--anchor_strategy" "${ANCHOR_STRATEGY}"
      "--lambda_norm" "${LAMBDA_NORM}"
      "--eta_hedge" "${ETA_HEDGE}"
      "--gamma" "${GAMMA}"
    )

    if [[ "${ENABLE_HEDGE}" == "1" ]]; then
      args+=("--enable_hedge")
    fi
    if [[ "${USE_LOGSUMEXP}" == "1" ]]; then
      args+=("--use_logsumexp")
    fi
    if [[ "${DECO_DEBUG}" == "1" ]]; then
      args+=("--deco_debug")
    fi
    if [[ -n "${ANCHOR_DEFECT_WORDS}" ]]; then
      args+=("--anchor_defect_words" "${ANCHOR_DEFECT_WORDS}")
    fi
    if [[ -n "${ANCHOR_NORM_WORDS}" ]]; then
      args+=("--anchor_norm_words" "${ANCHOR_NORM_WORDS}")
    fi
    if [[ -n "${ANCHOR_HEDGE_WORDS}" ]]; then
      args+=("--anchor_hedge_words" "${ANCHOR_HEDGE_WORDS}")
    fi
  fi

  echo "===== Running ${mode_name} ====="
  echo "output_json: ${output_json}"
  echo "dataset_root: ${DATASET_ROOT}"
  echo "dataset_type: ${DATASET_TYPE}"
  echo "model_name_or_path: ${MODEL_NAME_OR_PATH}"
  echo "repo_root: ${REPO_ROOT}"
  echo "deco_dir: ${DECO_DIR}"
  echo "python_script: ${PY_SCRIPT}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<not-set>}"
  echo "category: ${CATEGORY:-<unset>}"
  echo "categories: ${CATEGORIES:-<unset>}"
  echo "score_mode: ${SCORE_MODE}"
  echo "shot: ${SHOT}, seed: ${SEED}"

  # Run from REPO_ROOT with PYTHONPATH only REPO_ROOT/src so "import transformers" uses transformers-4.57.6.
  (cd "${REPO_ROOT}" && PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" examples/deco/eval_image_auroc_qwen3_vl_deco.py "${args[@]}")
}

case "${MODE}" in
  baseline)
    run_once "baseline" "${OUTPUT_DIR}/${OUTPUT_BASENAME}_baseline.json"
    ;;
  deco)
    run_once "deco" "${OUTPUT_DIR}/${OUTPUT_BASENAME}_deco.json"
    ;;
  both)
    run_once "baseline" "${OUTPUT_DIR}/${OUTPUT_BASENAME}_baseline.json"
    run_once "deco" "${OUTPUT_DIR}/${OUTPUT_BASENAME}_deco.json"
    ;;
  *)
    echo "[ERROR] Invalid MODE=${MODE}. Use baseline|deco|both"
    exit 1
    ;;
esac

echo "Done. Results saved to: ${OUTPUT_DIR}"
