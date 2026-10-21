#!/usr/bin/env bash
set -euo pipefail

# E1 support-set stability runner. Run from the PADA project root.
# It samples normal support images with --support-sampling random and uses
# each seed to control both model initialization and support selection.

DATASET="mvtec"
CLASSES=""
SHOTS="1,2,4"
SEEDS="0,1,2,3,4"
GPU_ID="0"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
EPOCHS="100"
CAL_PRO="False"
VIS="False"
EXPERIMENT_NAME="e1_stability"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;;
    --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;;
    --vis) VIS="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"

for seed in "${SEED_LIST[@]}"; do
  echo "========== E1 ${DATASET} seed ${seed} =========="
  args=(
    --dataset "${DATASET}"
    --shots "${SHOTS}"
    --seed "${seed}"
    --support-sampling random
    --support-seed "${seed}"
    --gpu-id "${GPU_ID}"
    --root-dir "${ROOT_DIR}"
    --epochs "${EPOCHS}"
    --cal-pro "${CAL_PRO}"
    --vis "${VIS}"
    --experiment-name "${EXPERIMENT_NAME}"
  )

  if [[ -n "${CLASSES}" ]]; then
    args+=(--classes "${CLASSES}")
  fi
  if [[ "${DATASET}" == "mvtec" && -n "${MVTEC_ROOT}" ]]; then
    args+=(--mvtec-root "${MVTEC_ROOT}")
  fi
  if [[ "${DATASET}" == "visa" && -n "${VISA_ROOT}" ]]; then
    args+=(--visa-root "${VISA_ROOT}")
  fi

  bash scripts/run_e0_baseline.sh "${args[@]}"
done

python scripts/aggregate_results.py --root-dir "${ROOT_DIR}" --experiment-name "${EXPERIMENT_NAME}"
