#!/usr/bin/env bash
set -euo pipefail

# E0 baseline runner. Run from the PADA project root.
#
# Examples:
#   bash scripts/run_e0_baseline.sh --dataset mvtec --classes tile --shots 1 --gpu-id 5 --mvtec-root /path/to/mvtec
#   bash scripts/run_e0_baseline.sh --dataset mvtec --shots 1,2,4 --gpu-id 5 --mvtec-root /path/to/mvtec

DATASET="mvtec"
CLASSES=""
SHOTS="1"
GPU_ID="0"
SEED="111"
SUPPORT_SAMPLING="seed_file"
SUPPORT_SEED=""
PROMPT_VARIANT=""
USE_GENERIC_ANOMALY_PROMPT="True"
USE_CATEGORY_ANOMALY_PROMPT="True"
USE_LEARNABLE_ANOMALY_PROMPT="True"
FUSION_VARIANT=""
USE_TEXT_BRANCH="True"
USE_VISUAL_MEMORY="True"
FUSION_TYPE="harmonic"
FUSION_ALPHA="0.5"
MODEL_VARIANT=""
PROMPT_MODULE="kan"
KAN_VARIANT="full"
DFA_VARIANT=""
USE_DFA="True"
SAVE_ANCHORS="False"
SENSITIVITY_VARIANT=""
LAMBDA_DFA="1.0"
MEMORY_VARIANT=""
MEMORY_LAYERS="layer3,layer8"
SAVE_CASES="False"
SUPPORT_PERTURBATION="none"
ROBUSTNESS_VARIANT=""
MAX_TEST_SAMPLES=""
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
EPOCHS="100"
CAL_PRO="False"
VIS="False"
EXPERIMENT_NAME="e0_baseline"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --support-sampling) SUPPORT_SAMPLING="$2"; shift 2 ;;
    --support-seed) SUPPORT_SEED="$2"; shift 2 ;;
    --prompt-variant) PROMPT_VARIANT="$2"; shift 2 ;;
    --use-generic-anomaly-prompt) USE_GENERIC_ANOMALY_PROMPT="$2"; shift 2 ;;
    --use-category-anomaly-prompt) USE_CATEGORY_ANOMALY_PROMPT="$2"; shift 2 ;;
    --use-learnable-anomaly-prompt) USE_LEARNABLE_ANOMALY_PROMPT="$2"; shift 2 ;;
    --fusion-variant) FUSION_VARIANT="$2"; shift 2 ;;
    --use-text-branch) USE_TEXT_BRANCH="$2"; shift 2 ;;
    --use-visual-memory) USE_VISUAL_MEMORY="$2"; shift 2 ;;
    --fusion-type) FUSION_TYPE="$2"; shift 2 ;;
    --fusion-alpha) FUSION_ALPHA="$2"; shift 2 ;;
    --model-variant) MODEL_VARIANT="$2"; shift 2 ;;
    --prompt-module) PROMPT_MODULE="$2"; shift 2 ;;
    --kan-variant) KAN_VARIANT="$2"; shift 2 ;;
    --dfa-variant) DFA_VARIANT="$2"; shift 2 ;;
    --use-dfa) USE_DFA="$2"; shift 2 ;;
    --save-anchors) SAVE_ANCHORS="$2"; shift 2 ;;
    --sensitivity-variant) SENSITIVITY_VARIANT="$2"; shift 2 ;;
    --lambda-dfa) LAMBDA_DFA="$2"; shift 2 ;;
    --memory-variant) MEMORY_VARIANT="$2"; shift 2 ;;
    --memory-layers) MEMORY_LAYERS="$2"; shift 2 ;;
    --save-cases) SAVE_CASES="$2"; shift 2 ;;
    --support-perturbation) SUPPORT_PERTURBATION="$2"; shift 2 ;;
    --robustness-variant) ROBUSTNESS_VARIANT="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
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

if [[ -z "${CLASSES}" ]]; then
  CLASSES="$(python - <<PY
from datasets import dataset_classes
print(",".join(dataset_classes["${DATASET}"]))
PY
)"
fi

IFS=',' read -r -a SHOT_LIST <<< "${SHOTS}"
IFS=',' read -r -a CLASS_LIST <<< "${CLASSES}"

for shot in "${SHOT_LIST[@]}"; do
  for class_name in "${CLASS_LIST[@]}"; do
    common_args=(
      --dataset "${DATASET}"
      --class_name "${class_name}"
      --k_shot "${shot}"
      --gpu-id "${GPU_ID}"
      --seed "${SEED}"
      --support-sampling "${SUPPORT_SAMPLING}"
      --use-generic-anomaly-prompt "${USE_GENERIC_ANOMALY_PROMPT}"
      --use-category-anomaly-prompt "${USE_CATEGORY_ANOMALY_PROMPT}"
      --use-learnable-anomaly-prompt "${USE_LEARNABLE_ANOMALY_PROMPT}"
      --use-text-branch "${USE_TEXT_BRANCH}"
      --use-visual-memory "${USE_VISUAL_MEMORY}"
      --fusion-type "${FUSION_TYPE}"
      --fusion-alpha "${FUSION_ALPHA}"
      --prompt-module "${PROMPT_MODULE}"
      --kan-variant "${KAN_VARIANT}"
      --use-dfa "${USE_DFA}"
      --save-anchors "${SAVE_ANCHORS}"
      --lambda-dfa "${LAMBDA_DFA}"
      --memory-layers "${MEMORY_LAYERS}"
      --save-cases "${SAVE_CASES}"
      --support-perturbation "${SUPPORT_PERTURBATION}"
      --root-dir "${ROOT_DIR}"
      --Epoch "${EPOCHS}"
      --experiment-name "${EXPERIMENT_NAME}"
      --vis "${VIS}"
    )

    if [[ "${DATASET}" == "mvtec" && -n "${MVTEC_ROOT}" ]]; then
      common_args+=(--mvtec-root "${MVTEC_ROOT}")
    fi
    if [[ "${DATASET}" == "visa" && -n "${VISA_ROOT}" ]]; then
      common_args+=(--visa-root "${VISA_ROOT}")
    fi
    if [[ -n "${SUPPORT_SEED}" ]]; then
      common_args+=(--support-seed "${SUPPORT_SEED}")
    fi
    if [[ -n "${PROMPT_VARIANT}" ]]; then
      common_args+=(--prompt-variant "${PROMPT_VARIANT}")
    fi
    if [[ -n "${FUSION_VARIANT}" ]]; then
      common_args+=(--fusion-variant "${FUSION_VARIANT}")
    fi
    if [[ -n "${MODEL_VARIANT}" ]]; then
      common_args+=(--model-variant "${MODEL_VARIANT}")
    fi
    if [[ -n "${DFA_VARIANT}" ]]; then
      common_args+=(--dfa-variant "${DFA_VARIANT}")
    fi
    if [[ -n "${SENSITIVITY_VARIANT}" ]]; then
      common_args+=(--sensitivity-variant "${SENSITIVITY_VARIANT}")
    fi
    if [[ -n "${MEMORY_VARIANT}" ]]; then
      common_args+=(--memory-variant "${MEMORY_VARIANT}")
    fi
    if [[ -n "${ROBUSTNESS_VARIANT}" ]]; then
      common_args+=(--robustness-variant "${ROBUSTNESS_VARIANT}")
    fi
    if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
      common_args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
    fi

    echo "==> CLS ${DATASET} ${class_name} ${shot}-shot"
    python MaPLe_cls.py "${common_args[@]}"

    echo "==> SEG ${DATASET} ${class_name} ${shot}-shot"
    python MaPLe_seg.py "${common_args[@]}" --cal-pro "${CAL_PRO}"
  done
done
