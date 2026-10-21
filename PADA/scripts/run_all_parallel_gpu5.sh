#!/usr/bin/env bash
set -euo pipefail

# Full parallel controller for E1-E10 on one or more GPUs.
# It keeps experiment definitions unchanged and only parallelizes independent categories.

GPUS="5"
JOBS_PER_GPU="3"
MVTEC_ROOT=""
VISA_ROOT=""
ROOT_DIR="./result"
LOG_DIR="./logs_parallel"
EPOCHS="100"
CAL_PRO="False"
EXPERIMENTS="e1,e2,e3,e4,e5,e6,e7,e8,e9,e10"
RUN_MVTEC="True"
RUN_VISA="True"
MAX_TEST_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-id) GPUS="$2"; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --mvtec-root) MVTEC_ROOT="$2"; shift 2 ;;
    --visa-root) VISA_ROOT="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --cal-pro) CAL_PRO="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --run-mvtec) RUN_MVTEC="$2"; shift 2 ;;
    --run-visa) RUN_VISA="$2"; shift 2 ;;
    --max-test-samples) MAX_TEST_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}" "${ROOT_DIR}"

contains_experiment() {
  local needle="$1"
  case ",${EXPERIMENTS}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}

parallel_common_args=(
  --gpus "${GPUS}"
  --jobs-per-gpu "${JOBS_PER_GPU}"
  --root-dir "${ROOT_DIR}"
  --epochs "${EPOCHS}"
  --cal-pro "${CAL_PRO}"
)
if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
  parallel_common_args+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi

run_parallel_dataset() {
  local exp="$1"
  local dataset="$2"
  local root_arg_name="$3"
  local root_value="$4"
  local log_subdir="${LOG_DIR}/${exp}_${dataset}"

  if [[ -z "${root_value}" ]]; then
    echo "Skip ${exp} ${dataset}: root is empty"
    return 0
  fi

  echo "========== ${exp}_${dataset} =========="
  bash scripts/run_parallel_batches.sh \
    --experiment "${exp}" \
    --dataset "${dataset}" \
    "${root_arg_name}" "${root_value}" \
    --log-dir "${log_subdir}" \
    "${parallel_common_args[@]}"
}

for exp in e1 e2 e3 e4 e5 e6 e7 e10; do
  if contains_experiment "${exp}"; then
    if [[ "${RUN_MVTEC}" == "True" ]]; then
      run_parallel_dataset "${exp}" "mvtec" "--mvtec-root" "${MVTEC_ROOT}"
    fi
    if [[ "${RUN_VISA}" == "True" ]]; then
      run_parallel_dataset "${exp}" "visa" "--visa-root" "${VISA_ROOT}"
    fi
  fi
done

if contains_experiment e8; then
  first_gpu="${GPUS%%,*}"
  echo "========== e8_efficiency =========="
  python scripts/run_e8_efficiency_test.py \
    --gpu-id "${first_gpu}" \
    --variants none,mlp,transformer,kan \
    --warmup 20 \
    --iters 100 \
    --output "${ROOT_DIR}/e8_efficiency/efficiency.csv" \
    2>&1 | tee "${LOG_DIR}/e8_efficiency.log"
fi

if contains_experiment e9; then
  if [[ "${RUN_MVTEC}" == "True" && -n "${MVTEC_ROOT}" ]]; then
    echo "========== e9_mvtec =========="
    bash scripts/run_parallel_batches.sh \
      --experiment e9 \
      --dataset mvtec \
      --classes screw,transistor \
      --gpus "${GPUS}" \
      --jobs-per-gpu "${JOBS_PER_GPU}" \
      --mvtec-root "${MVTEC_ROOT}" \
      --root-dir "${ROOT_DIR}" \
      --log-dir "${LOG_DIR}/e9_mvtec" \
      --epochs "${EPOCHS}" \
      --cal-pro "${CAL_PRO}"
  fi
  if [[ "${RUN_VISA}" == "True" && -n "${VISA_ROOT}" ]]; then
    echo "========== e9_visa =========="
    bash scripts/run_parallel_batches.sh \
      --experiment e9 \
      --dataset visa \
      --classes pcb2,pcb3,macaroni2 \
      --gpus "${GPUS}" \
      --jobs-per-gpu "${JOBS_PER_GPU}" \
      --visa-root "${VISA_ROOT}" \
      --root-dir "${ROOT_DIR}" \
      --log-dir "${LOG_DIR}/e9_visa" \
      --epochs "${EPOCHS}" \
      --cal-pro "${CAL_PRO}"
  fi
fi

find "${ROOT_DIR}" \
  \( -path "*/_aggregate/*.csv" -o -path "*/e8_efficiency/*.csv" -o -name "exported_case_files.txt" \) \
  | sort > "${ROOT_DIR}/parallel_experiments_manifest.txt"

echo "Finished requested experiments."
echo "Manifest: ${ROOT_DIR}/parallel_experiments_manifest.txt"
