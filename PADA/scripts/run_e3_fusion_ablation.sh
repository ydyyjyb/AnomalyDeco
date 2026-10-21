#!/usr/bin/env bash
set -euo pipefail

# E3 text/visual branch and fusion ablation runner. Run from the PADA project root.
#
# Variants:
#   f1_text       text branch only
#   f2_visual     visual memory branch only
#   f3_mean       text + visual, arithmetic mean
#   f4_max        text + visual, maximum
#   f5_harmonic   text + visual, harmonic mean
#   f6_weighted   text + visual, weighted mean

DATASET="mvtec"
CLASSES=""
SHOTS="1"
SEEDS="0,1,2"
VARIANTS="f1_text,f2_visual,f3_mean,f4_max,f5_harmonic,f6_weighted"
GPU_ID="0"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
EPOCHS="100"
CAL_PRO="False"
VIS="False"
SUPPORT_SAMPLING="seed_file"
SUPPORT_SEED=""
FUSION_ALPHA="0.5"
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e3_fusion_ablation"

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
    --fusion-alpha) FUSION_ALPHA="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

variant_flags() {
  case "$1" in
    f1_text) echo "True False none" ;;
    f2_visual) echo "False True none" ;;
    f3_mean) echo "True True mean" ;;
    f4_max) echo "True True max" ;;
    f5_harmonic) echo "True True harmonic" ;;
    f6_weighted) echo "True True weighted" ;;
    *) echo "Unknown fusion variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

for variant in "${VARIANT_LIST[@]}"; do
  read -r use_text use_visual fusion_type <<< "$(variant_flags "${variant}")"
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
      --fusion-variant "${variant}"
      --use-text-branch "${use_text}"
      --use-visual-memory "${use_visual}"
      --fusion-type "${fusion_type}"
      --fusion-alpha "${FUSION_ALPHA}"
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

    echo "==> E3 variant=${variant} seed=${seed}"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done

python scripts/aggregate_results.py \
  --root-dir "${ROOT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}"
