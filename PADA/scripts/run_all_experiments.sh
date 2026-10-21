#!/usr/bin/env bash
set -euo pipefail

# Run E0-E10 in a reproducible order. Run from the PADA project root.
#
# Examples:
#   bash scripts/run_all_experiments.sh --mode smoke --gpu-id 5 --mvtec-root /path/mvtec --visa-root /path/VisA
#   bash scripts/run_all_experiments.sh --mode formal --gpu-id 5 --mvtec-root /path/mvtec --visa-root /path/VisA
#   bash scripts/run_all_experiments.sh --experiments e0,e2,e5 --mode formal ...

MODE="formal"
EXPERIMENTS="e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10"
GPU_ID="0"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
LOG_DIR="./logs"
EPOCHS="100"
SMOKE_EPOCHS="10"
CAL_PRO="False"
RUN_VISA="True"
RUN_MVTEC="True"
MVTEC_CLASSES=""
VISA_CLASSES=""
SHOTS_OVERRIDE=""
MAX_TEST_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;;
    --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --smoke-epochs) SMOKE_EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;;
    --run-visa) RUN_VISA="$2"; shift 2 ;;
    --run-mvtec) RUN_MVTEC="$2"; shift 2 ;;
    --mvtec-classes) MVTEC_CLASSES="$2"; shift 2 ;;
    --visa-classes) VISA_CLASSES="$2"; shift 2 ;;
    --shots) SHOTS_OVERRIDE="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"

contains_experiment() {
  local target="$1"
  [[ ",${EXPERIMENTS}," == *",${target},"* ]]
}

run_step() {
  local name="$1"
  shift
  echo "========== ${name} =========="
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

dataset_root_args() {
  local dataset="$1"
  if [[ "${dataset}" == "mvtec" ]]; then
    echo --mvtec-root "${MVTEC_ROOT}"
  else
    echo --visa-root "${VISA_ROOT}"
  fi
}

require_roots() {
  if [[ "${RUN_MVTEC}" == "True" && -z "${MVTEC_ROOT}" ]]; then
    echo "Warning: --mvtec-root is empty; MVTecAD experiments will be skipped." >&2
    RUN_MVTEC="False"
  fi
  if [[ "${RUN_VISA}" == "True" && -z "${VISA_ROOT}" ]]; then
    echo "Warning: --visa-root is empty; VisA experiments will be skipped." >&2
    RUN_VISA="False"
  fi
  if [[ "${RUN_MVTEC}" != "True" && "${RUN_VISA}" != "True" ]]; then
    echo "At least one dataset root is required: pass --mvtec-root and/or --visa-root." >&2
    exit 1
  fi
}

write_manifest() {
  local manifest="${ROOT_DIR}/all_experiments_manifest.txt"
  {
    echo "mode=${MODE}"
    echo "experiments=${EXPERIMENTS}"
    echo "generated_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo
    find "${ROOT_DIR}" -path "*/_aggregate/*.csv" -type f | sort
    find "${ROOT_DIR}" -path "*/efficiency.csv" -type f | sort
    find "${ROOT_DIR}" -name "exported_case_files.txt" -type f | sort
  } > "${manifest}"
  echo "Wrote ${manifest}"
}

require_roots

if [[ "${MODE}" == "smoke" ]]; then
  EPOCHS="${SMOKE_EPOCHS}"
  SHOTS_MAIN="1"
  SEEDS_MAIN="111"
  SEEDS_STABILITY="0,1"
  CLASSES_MVTEC="${MVTEC_CLASSES:-tile}"
  CLASSES_VISA="${VISA_CLASSES:-candle}"
  E2_VARIANTS="all,generic"
  E3_VARIANTS="f1_text,f2_visual,f5_harmonic"
  E4_VARIANTS="k1_none,k4_kan"
  E5_VARIANTS="d1_kan_no_dfa,d2_kan_dfa"
  E6_LAMBDAS="0,0.10"
  E6_ALPHAS="0,0.50,1.00"
  E7_VARIANTS="m1_layer3,m4_layer3_layer8,m6_all"
  E10_VARIANTS="r0_none,r1_brightness_low,r3_noise"
else
  SHOTS_MAIN="1,2,4"
  SEEDS_MAIN="111"
  SEEDS_STABILITY="0,1,2,3,4"
  CLASSES_MVTEC="${MVTEC_CLASSES}"
  CLASSES_VISA="${VISA_CLASSES}"
  E2_VARIANTS="all,generic,category,learnable,generic_category"
  E3_VARIANTS="f1_text,f2_visual,f3_mean,f4_max,f5_harmonic,f6_weighted"
  E4_VARIANTS="k1_none,k2_mlp,k3_transformer,k4_kan"
  E5_VARIANTS="d1_kan_no_dfa,d2_kan_dfa,d3_transformer_no_dfa,d4_transformer_dfa,d5_mlp_no_dfa,d6_mlp_dfa"
  E6_LAMBDAS="0,0.01,0.05,0.10,0.20,0.50"
  E6_ALPHAS="0,0.25,0.50,0.75,1.00"
  E7_VARIANTS="m1_layer3,m2_layer8,m3_final,m4_layer3_layer8,m5_layer8_final,m6_all"
  E10_VARIANTS="r0_none,r1_brightness_low,r2_brightness_high,r3_noise,r4_blur,r5_rotate"
fi

if [[ -n "${SHOTS_OVERRIDE}" ]]; then
  SHOTS_MAIN="${SHOTS_OVERRIDE}"
fi

run_dataset_pair() {
  local step="$1"
  shift
  if [[ "${RUN_MVTEC}" == "True" ]]; then
    "$@" mvtec "${MVTEC_ROOT}" "${CLASSES_MVTEC}"
  fi
  if [[ "${RUN_VISA}" == "True" ]]; then
    "$@" visa "${VISA_ROOT}" "${CLASSES_VISA}"
  fi
}

run_e0_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots "${SHOTS_MAIN}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e0_baseline_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e0_${dataset}" bash scripts/run_e0_baseline.sh "${args[@]}"
}

run_e1_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots "${SHOTS_MAIN}" --seeds "${SEEDS_STABILITY}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e1_stability_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e1_${dataset}" bash scripts/run_e1_stability.sh "${args[@]}"
}

run_e2_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds "${SEEDS_MAIN}" --variants "${E2_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e2_prompt_ablation_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e2_${dataset}" bash scripts/run_e2_prompt_ablation.sh "${args[@]}"
}

run_e3_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds "${SEEDS_MAIN}" --variants "${E3_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e3_fusion_ablation_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e3_${dataset}" bash scripts/run_e3_fusion_ablation.sh "${args[@]}"
}

run_e4_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds "${SEEDS_MAIN}" --variants "${E4_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e4_kan_ablation_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e4_${dataset}" bash scripts/run_e4_kan_ablation.sh "${args[@]}"
}

run_e5_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds "${SEEDS_MAIN}" --variants "${E5_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e5_dfa_analysis_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e5_${dataset}" bash scripts/run_e5_dfa_analysis.sh "${args[@]}"
}

run_e6_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args_l=(--dataset "${dataset}" --shots 1 --seeds 0 --lambdas "${E6_LAMBDAS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e6_lambda_sensitivity_${dataset}")
  local args_a=(--dataset "${dataset}" --shots 1 --seeds 0 --alphas "${E6_ALPHAS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e6_alpha_sensitivity_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args_l+=(--max-test-samples "${MAX_TEST_SAMPLES}") && args_a+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args_l+=(--classes "${classes}") && args_a+=(--classes "${classes}")
  if [[ "${dataset}" == "mvtec" ]]; then args_l+=(--mvtec-root "${root}"); args_a+=(--mvtec-root "${root}"); else args_l+=(--visa-root "${root}"); args_a+=(--visa-root "${root}"); fi
  run_step "e6_lambda_${dataset}" bash scripts/run_e6_lambda_sensitivity.sh "${args_l[@]}"
  run_step "e6_alpha_${dataset}" bash scripts/run_e6_alpha_sensitivity.sh "${args_a[@]}"
}

run_e7_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds 0 --variants "${E7_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e7_memory_layer_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e7_${dataset}" bash scripts/run_e7_memory_layer_ablation.sh "${args[@]}"
}

run_e8_once() {
  run_step "e8_efficiency" python scripts/run_e8_efficiency_test.py --gpu-id "${GPU_ID}" --variants none,mlp,transformer,kan --warmup 20 --iters 100 --output "${ROOT_DIR}/e8_efficiency/efficiency.csv"
}

run_e9() {
  if [[ "${RUN_MVTEC}" == "True" ]]; then
    local e9_mvtec_args=(--dataset mvtec --classes screw,transistor --shots 1 --seed 111 --gpu-id "${GPU_ID}" --mvtec-root "${MVTEC_ROOT}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name e9_failure_cases_mvtec)
    [[ -n "${MAX_TEST_SAMPLES}" ]] && e9_mvtec_args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
    run_step "e9_mvtec" bash scripts/run_e9_failure_case_export.sh "${e9_mvtec_args[@]}"
  fi
  if [[ "${RUN_VISA}" == "True" ]]; then
    local e9_visa_args=(--dataset visa --classes pcb2,pcb3,macaroni2 --shots 1 --seed 111 --gpu-id "${GPU_ID}" --visa-root "${VISA_ROOT}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name e9_failure_cases_visa)
    [[ -n "${MAX_TEST_SAMPLES}" ]] && e9_visa_args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
    run_step "e9_visa" bash scripts/run_e9_failure_case_export.sh "${e9_visa_args[@]}"
  fi
}

run_e10_dataset() {
  local dataset="$1"; local root="$2"; local classes="$3"
  local args=(--dataset "${dataset}" --shots 1 --seeds "${SEEDS_MAIN}" --variants "${E10_VARIANTS}" --gpu-id "${GPU_ID}" --root-dir "${ROOT_DIR}" --epochs "${EPOCHS}" --cal-pro "${CAL_PRO}" --experiment-name "e10_robustness_${dataset}")
  [[ -n "${MAX_TEST_SAMPLES}" ]] && args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  [[ -n "${classes}" ]] && args+=(--classes "${classes}")
  [[ "${dataset}" == "mvtec" ]] && args+=(--mvtec-root "${root}") || args+=(--visa-root "${root}")
  run_step "e10_${dataset}" bash scripts/run_e10_robustness.sh "${args[@]}"
}

contains_experiment e0 && run_dataset_pair e0 run_e0_dataset
contains_experiment e1 && run_dataset_pair e1 run_e1_dataset
contains_experiment e2 && run_dataset_pair e2 run_e2_dataset
contains_experiment e3 && run_dataset_pair e3 run_e3_dataset
contains_experiment e4 && run_dataset_pair e4 run_e4_dataset
contains_experiment e5 && run_dataset_pair e5 run_e5_dataset
contains_experiment e6 && run_dataset_pair e6 run_e6_dataset
contains_experiment e7 && run_dataset_pair e7 run_e7_dataset
contains_experiment e8 && run_e8_once
contains_experiment e9 && run_e9
contains_experiment e10 && run_dataset_pair e10 run_e10_dataset

write_manifest
