#!/usr/bin/env bash
set -euo pipefail

DATASET="visa"; CLASSES=""; SHOTS="1"; SEEDS="0,1,2"
VARIANTS="r0_none,r1_brightness_low,r2_brightness_high,r3_noise,r4_blur,r5_rotate"
GPU_ID="0"; MVTEC_ROOT=""; VISA_ROOT=""; ROOT_DIR="./result"; EPOCHS="100"; CAL_PRO="False"
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e10_support_robustness"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;; --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;; --seeds) SEEDS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;; --gpu-id) GPU_ID="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;; --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;; --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;; --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

perturb_for_variant() {
  case "$1" in
    r0_none) echo "none" ;; r1_brightness_low) echo "brightness_low" ;;
    r2_brightness_high) echo "brightness_high" ;; r3_noise) echo "noise" ;;
    r4_blur) echo "blur" ;; r5_rotate) echo "rotate" ;;
    *) echo "Unknown robustness variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
for variant in "${VARIANT_LIST[@]}"; do
  perturb="$(perturb_for_variant "$variant")"
  for seed in "${SEED_LIST[@]}"; do
    args=(--dataset "$DATASET" --classes "$CLASSES" --shots "$SHOTS" --gpu-id "$GPU_ID"
      --seed "$seed" --root-dir "$ROOT_DIR" --epochs "$EPOCHS" --cal-pro "$CAL_PRO"
      --experiment-name "$EXPERIMENT_NAME" --robustness-variant "$variant"
      --support-perturbation "$perturb")
    [[ -n "$MVTEC_ROOT" ]] && args+=(--mvtec-root "$MVTEC_ROOT")
    [[ -n "$VISA_ROOT" ]] && args+=(--visa-root "$VISA_ROOT")
    [[ -n "$MAX_TEST_SAMPLES" ]] && args+=(--max-test-samples "$MAX_TEST_SAMPLES")
    echo "==> E10 variant=$variant perturb=$perturb seed=$seed"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done
python scripts/aggregate_results.py --root-dir "$ROOT_DIR" --experiment-name "$EXPERIMENT_NAME"
