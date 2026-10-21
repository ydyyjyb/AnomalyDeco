#!/usr/bin/env bash
set -euo pipefail

DATASET="mvtec"; CLASSES=""; SHOTS="1"; SEEDS="0"
VARIANTS="m1_layer3,m2_layer8,m3_final,m4_layer3_layer8,m5_layer8_final,m6_all"
GPU_ID="0"; MVTEC_ROOT=""; VISA_ROOT=""; ROOT_DIR="./result"; EPOCHS="100"; CAL_PRO="False"
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e7_memory_layer_ablation"

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

layers_for_variant() {
  case "$1" in
    m1_layer3) echo "layer3" ;; m2_layer8) echo "layer8" ;; m3_final) echo "final" ;;
    m4_layer3_layer8) echo "layer3,layer8" ;; m5_layer8_final) echo "layer8,final" ;;
    m6_all) echo "layer3,layer8,final" ;; *) echo "Unknown memory variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
for variant in "${VARIANT_LIST[@]}"; do
  layers="$(layers_for_variant "$variant")"
  for seed in "${SEED_LIST[@]}"; do
    args=(--dataset "$DATASET" --classes "$CLASSES" --shots "$SHOTS" --gpu-id "$GPU_ID"
      --seed "$seed" --root-dir "$ROOT_DIR" --epochs "$EPOCHS" --cal-pro "$CAL_PRO"
      --experiment-name "$EXPERIMENT_NAME" --memory-variant "$variant" --memory-layers "$layers")
    [[ -n "$MVTEC_ROOT" ]] && args+=(--mvtec-root "$MVTEC_ROOT")
    [[ -n "$VISA_ROOT" ]] && args+=(--visa-root "$VISA_ROOT")
    [[ -n "$MAX_TEST_SAMPLES" ]] && args+=(--max-test-samples "$MAX_TEST_SAMPLES")
    echo "==> E7 variant=$variant layers=$layers seed=$seed"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done
python scripts/aggregate_results.py --root-dir "$ROOT_DIR" --experiment-name "$EXPERIMENT_NAME"
