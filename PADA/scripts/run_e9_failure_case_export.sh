#!/usr/bin/env bash
set -euo pipefail

DATASET="mvtec"; CLASSES="screw,transistor"; SHOTS="1"; SEED="111"; GPU_ID="0"
MVTEC_ROOT=""; VISA_ROOT=""; ROOT_DIR="./result"; EPOCHS="100"; CAL_PRO="False"
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e9_failure_cases"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;; --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;; --seed) SEED="$2"; shift 2 ;; --gpu-id) GPU_ID="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;; --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;; --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;; --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

args=(--dataset "$DATASET" --classes "$CLASSES" --shots "$SHOTS" --gpu-id "$GPU_ID"
  --seed "$SEED" --root-dir "$ROOT_DIR" --epochs "$EPOCHS" --cal-pro "$CAL_PRO"
  --experiment-name "$EXPERIMENT_NAME" --vis True --save-cases True)
[[ -n "$MVTEC_ROOT" ]] && args+=(--mvtec-root "$MVTEC_ROOT")
[[ -n "$VISA_ROOT" ]] && args+=(--visa-root "$VISA_ROOT")
[[ -n "$MAX_TEST_SAMPLES" ]] && args+=(--max-test-samples "$MAX_TEST_SAMPLES")

bash scripts/run_e0_baseline.sh "${args[@]}"
python scripts/aggregate_results.py --root-dir "$ROOT_DIR" --experiment-name "$EXPERIMENT_NAME"
find "$ROOT_DIR/$EXPERIMENT_NAME" -name "cases_*.csv" -o -name "*.png" | sort > "$ROOT_DIR/$EXPERIMENT_NAME/exported_case_files.txt"
echo "Wrote $ROOT_DIR/$EXPERIMENT_NAME/exported_case_files.txt"
