#!/usr/bin/env bash
set -euo pipefail

# Parallel experiment launcher for expensive formal runs.
# It does not reduce experiment volume; it splits independent category jobs.
#
# Examples:
#   bash scripts/run_parallel_batches.sh --experiment e1 --dataset mvtec --gpus 5 --jobs-per-gpu 3 --mvtec-root /path/mvtec
#   bash scripts/run_parallel_batches.sh --experiment e2 --dataset visa --gpus 5,6 --visa-root /path/VisA
#
# Supported experiments:
#   e1 e2 e3 e4 e5 e6 e6_lambda e6_alpha e7 e9 e10

EXPERIMENT="e1"
DATASET="mvtec"
CLASSES=""
SHOTS="1"
SEEDS="0,1,2"
GPUS="0"
JOBS_PER_GPU="1"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
LOG_DIR="./logs_parallel"
EPOCHS="100"
CAL_PRO="False"
MAX_TEST_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;;
    --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"

if [[ -z "${CLASSES}" ]]; then
  CLASSES="$(python - <<PY
from datasets import dataset_classes
print(",".join(dataset_classes["${DATASET}"]))
PY
)"
fi

IFS=',' read -r -a CLASS_LIST <<< "${CLASSES}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
MAX_JOBS=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
if [[ "${MAX_JOBS}" -lt 1 ]]; then
  MAX_JOBS=1
fi

active_jobs=0
job_index=0

wait_for_slot() {
  while [[ "${active_jobs}" -ge "${MAX_JOBS}" ]]; do
    wait -n
    active_jobs=$((active_jobs - 1))
  done
}

dataset_root_args=()
if [[ "${DATASET}" == "mvtec" ]]; then
  dataset_root_args=(--mvtec-root "${MVTEC_ROOT}")
else
  dataset_root_args=(--visa-root "${VISA_ROOT}")
fi

common_args=(
  --dataset "${DATASET}"
  --shots "${SHOTS}"
  --root-dir "${ROOT_DIR}"
  --epochs "${EPOCHS}"
  --cal-pro "${CAL_PRO}"
)
if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
  common_args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi

run_one_class() {
  local class_name="$1"
  local gpu_id="$2"
  local exp="$3"
  local log_file="${LOG_DIR}/${exp}_${DATASET}_${class_name}_gpu${gpu_id}.log"

  case "${exp}" in
    e1)
      bash scripts/run_e1_stability.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --experiment-name "e1_stability_${DATASET}"
      ;;
    e2)
      bash scripts/run_e2_prompt_ablation.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants all,generic,category,learnable,generic_category \
        --experiment-name "e2_prompt_ablation_${DATASET}"
      ;;
    e3)
      bash scripts/run_e3_fusion_ablation.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants f1_text,f2_visual,f3_mean,f4_max,f5_harmonic,f6_weighted \
        --experiment-name "e3_fusion_ablation_${DATASET}"
      ;;
    e4)
      bash scripts/run_e4_kan_ablation.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants k1_none,k2_mlp,k3_transformer,k4_kan \
        --experiment-name "e4_kan_ablation_${DATASET}"
      ;;
    e5)
      bash scripts/run_e5_dfa_analysis.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants d1_kan_no_dfa,d2_kan_dfa,d3_transformer_no_dfa,d4_transformer_dfa,d5_mlp_no_dfa,d6_mlp_dfa \
        --experiment-name "e5_dfa_analysis_${DATASET}"
      ;;
    e6|e6_lambda)
      bash scripts/run_e6_lambda_sensitivity.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds 0 --gpu-id "${gpu_id}" \
        --lambdas 0,0.01,0.05,0.10,0.20,0.50 \
        --experiment-name "e6_lambda_sensitivity_${DATASET}"
      if [[ "${exp}" == "e6" ]]; then
        bash scripts/run_e6_alpha_sensitivity.sh "${common_args[@]}" "${dataset_root_args[@]}" \
          --classes "${class_name}" --seeds 0 --gpu-id "${gpu_id}" \
          --alphas 0,0.25,0.50,0.75,1.00 \
          --experiment-name "e6_alpha_sensitivity_${DATASET}"
      fi
      ;;
    e6_alpha)
      bash scripts/run_e6_alpha_sensitivity.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds 0 --gpu-id "${gpu_id}" \
        --alphas 0,0.25,0.50,0.75,1.00 \
        --experiment-name "e6_alpha_sensitivity_${DATASET}"
      ;;
    e7)
      bash scripts/run_e7_memory_layer_ablation.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants m1_layer3,m2_layer8,m3_final,m4_layer3_layer8,m5_layer8_final,m6_all \
        --experiment-name "e7_memory_layer_${DATASET}"
      ;;
    e9)
      bash scripts/run_e9_failure_case_export.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seed 111 --gpu-id "${gpu_id}" \
        --experiment-name "e9_failure_cases_${DATASET}"
      ;;
    e10)
      bash scripts/run_e10_robustness.sh "${common_args[@]}" "${dataset_root_args[@]}" \
        --classes "${class_name}" --seeds "${SEEDS}" --gpu-id "${gpu_id}" \
        --variants r0_none,r1_brightness_low,r2_brightness_high,r3_noise,r4_blur,r5_rotate \
        --experiment-name "e10_robustness_${DATASET}"
      ;;
    *)
      echo "Unsupported parallel experiment: ${exp}" >&2
      return 1
      ;;
  esac
}

for class_name in "${CLASS_LIST[@]}"; do
  wait_for_slot
  gpu_id="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
  echo "Launching ${EXPERIMENT} ${DATASET}/${class_name} on GPU ${gpu_id}"
  (
    run_one_class "${class_name}" "${gpu_id}" "${EXPERIMENT}"
  ) > "${LOG_DIR}/${EXPERIMENT}_${DATASET}_${class_name}_gpu${gpu_id}.log" 2>&1 &
  active_jobs=$((active_jobs + 1))
  job_index=$((job_index + 1))
  sleep 2
done

wait

case "${EXPERIMENT}" in
  e1) EXP_NAME="e1_stability_${DATASET}" ;;
  e2) EXP_NAME="e2_prompt_ablation_${DATASET}" ;;
  e3) EXP_NAME="e3_fusion_ablation_${DATASET}" ;;
  e4) EXP_NAME="e4_kan_ablation_${DATASET}" ;;
  e5) EXP_NAME="e5_dfa_analysis_${DATASET}" ;;
  e6|e6_lambda) EXP_NAME="e6_lambda_sensitivity_${DATASET}" ;;
  e6_alpha) EXP_NAME="e6_alpha_sensitivity_${DATASET}" ;;
  e7) EXP_NAME="e7_memory_layer_${DATASET}" ;;
  e9) EXP_NAME="e9_failure_cases_${DATASET}" ;;
  e10) EXP_NAME="e10_robustness_${DATASET}" ;;
esac

python scripts/aggregate_results.py --root-dir "${ROOT_DIR}" --experiment-name "${EXP_NAME}"
if [[ "${EXPERIMENT}" == "e6" ]]; then
  python scripts/aggregate_results.py --root-dir "${ROOT_DIR}" --experiment-name "e6_alpha_sensitivity_${DATASET}"
fi
echo "Finished ${EXPERIMENT} ${DATASET}. Logs: ${LOG_DIR}"
