#!/usr/bin/env bash
set -euo pipefail

# E6 DFA lambda sensitivity runner. Run from the PADA project root.

DATASET="mvtec"
CLASSES=""
SHOTS="1"
SEEDS="0"
LAMBDAS="0,0.01,0.05,0.10,0.20,0.50"
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
EXPERIMENT_NAME="e6_lambda_sensitivity"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --lambdas) LAMBDAS="$2"; shift 2 ;;
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

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a LAMBDA_LIST <<< "${LAMBDAS}"

lambda_variant() {
  echo "lambda_${1//./p}"
}

for lambda_dfa in "${LAMBDA_LIST[@]}"; do
  variant="$(lambda_variant "${lambda_dfa}")"
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
      --sensitivity-variant "${variant}"
      --use-dfa "True"
      --lambda-dfa "${lambda_dfa}"
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

    echo "==> E6 lambda=${lambda_dfa} seed=${seed}"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done

python scripts/aggregate_results.py \
  --root-dir "${ROOT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}"
