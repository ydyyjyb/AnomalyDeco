#!/usr/bin/env bash
set -euo pipefail

# E5 DFA effectiveness and anchor-distance diagnostics runner.
# Run from the PADA project root.
#
# Variants:
#   d1_kan_no_dfa          KAN-PAM, DFA off
#   d2_kan_dfa             KAN-PAM, DFA on
#   d3_transformer_no_dfa  Transformer prompt module, DFA off
#   d4_transformer_dfa     Transformer prompt module, DFA on
#   d5_mlp_no_dfa          MLP prompt module, DFA off
#   d6_mlp_dfa             MLP prompt module, DFA on

DATASET="mvtec"
CLASSES=""
SHOTS="1"
SEEDS="0,1,2"
VARIANTS="d1_kan_no_dfa,d2_kan_dfa,d3_transformer_no_dfa,d4_transformer_dfa,d5_mlp_no_dfa,d6_mlp_dfa"
GPU_ID="0"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
EPOCHS="100"
CAL_PRO="False"
VIS="False"
SUPPORT_SAMPLING="seed_file"
SUPPORT_SEED=""
SAVE_ANCHORS="True"
MAX_TEST_SAMPLES=""
EXPERIMENT_NAME="e5_dfa_analysis"

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
    --save-anchors) SAVE_ANCHORS="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

variant_flags() {
  case "$1" in
    d1_kan_no_dfa) echo "kan False" ;;
    d2_kan_dfa) echo "kan True" ;;
    d3_transformer_no_dfa) echo "transformer False" ;;
    d4_transformer_dfa) echo "transformer True" ;;
    d5_mlp_no_dfa) echo "mlp False" ;;
    d6_mlp_dfa) echo "mlp True" ;;
    *) echo "Unknown DFA variant: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

for variant in "${VARIANT_LIST[@]}"; do
  read -r prompt_module use_dfa <<< "$(variant_flags "${variant}")"
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
      --dfa-variant "${variant}"
      --model-variant "${prompt_module}"
      --prompt-module "${prompt_module}"
      --use-dfa "${use_dfa}"
      --save-anchors "${SAVE_ANCHORS}"
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

    echo "==> E5 variant=${variant} prompt_module=${prompt_module} use_dfa=${use_dfa} seed=${seed}"
    bash scripts/run_e0_baseline.sh "${args[@]}"
  done
done

python scripts/aggregate_results.py \
  --root-dir "${ROOT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}"
