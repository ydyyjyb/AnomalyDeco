#!/usr/bin/env bash
set -euo pipefail

# E2 prompt-construction ablation runner. Run from the PADA project root.
#
# Variants:
#   all              generic + category + learnable abnormal prompts
#   generic          generic hand-crafted abnormal prompts only
#   category         category-specific hand-crafted abnormal prompts only
#   learnable        learnable abnormal prompts only
#   generic_category generic + category hand-crafted abnormal prompts

DATASET="mvtec"
CLASSES=""
SHOTS="1"
SEEDS="111"
VARIANTS="all,generic,category,learnable,generic_category"
GPU_ID="0"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
EPOCHS="100"
CAL_PRO="False"
VIS="False"
SUPPORT_SAMPLING="seed_file"
SUPPORT_SEED=""
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e2_prompt_ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;;
    --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;;
    --vis) VIS="$2"; shift 2 ;;
    --support-sampling) SUPPORT_SAMPLING="$2"; shift 2 ;;
    --support-seed) SUPPORT_SEED="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

variant_flags() {
  case "$1" in
    all) echo "True True True" ;;
    generic) echo "True False False" ;;
    category) echo "False True False" ;;
    learnable) echo "False False True" ;;
    generic_category) echo "True True False" ;;
    generic_learnable) echo "True False True" ;;
    category_learnable) echo "False True True" ;;
    *) echo "Unknown prompt variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

for variant in "${VARIANT_LIST[@]}"; do
  read -r use_generic use_category use_learnable <<< "$(variant_flags "${variant}")"
  for seed in "${SEED_LIST[@]}"; do
    args=(
      --dataset "${DATASET}"
      --classes "${CLASSES}"
      --shots "${SHOTS}"
      --gpu-id "${GPU_ID}"
      --seed "${seed}"
      --support-sampling "${SUPPORT_SAMPLING}"
      --root-dir "${ROOT_DIR}"
      --epochs "${EPOCHS}"
      --cal-pro "${CAL_PRO}"
      --vis "${VIS}"
      --experiment-name "${EXPERIMENT_NAME}"
      --prompt-variant "${variant}"
      --use-generic-anomaly-prompt "${use_generic}"
      --use-category-anomaly-prompt "${use_category}"
      --use-learnable-anomaly-prompt "${use_learnable}"
    )
    if [[ -n "${MVTEC_ROOT}" ]]; then
      args+=(--mvtec-root "${MVTEC_ROOT}")
    fi
    if [[ -n "${VISA_ROOT}" ]]; then
      args+=(--visa-root "${VISA_ROOT}")
    fi
    if [[ -n "${SUPPORT_SEED}" ]]; then
      args+=(--support-seed "${SUPPORT_SEED}")
    fi
    if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
      args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
    fi

    echo "==> E2 variant=${variant} seed=${seed}"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done

python scripts/aggregate_results.py \
  --root-dir "${ROOT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}"
