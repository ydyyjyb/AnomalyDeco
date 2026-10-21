#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAR_ROOT="${ADAR_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -r -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
  printf '\n'
  export DEEPSEEK_API_KEY
fi

if [[ -z "${VIRTUAL_ENV:-}" && -f "${ADAR_ROOT}/uv-envs/qwen3-vl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ADAR_ROOT}/uv-envs/qwen3-vl/bin/activate"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ADAR_ROOT}/Deco-main/runs/visa_dr_quality_dimensions_${STAMP}/deepseek_api_full}"
mkdir -p "${OUT_DIR}"

python "${ADAR_ROOT}/scripts/evaluate_visa_dr_quality_dimensions.py" \
  --judge-backend deepseek-api \
  --api-model "${DEEPSEEK_API_MODEL:-deepseek-flash}" \
  --api-concurrency "${API_CONCURRENCY:-3}" \
  --data-root "${VISA_DR_DATA_ROOT:-${ADAR_ROOT}/evaluation/ODinW-13/data}" \
  --out-dir "${OUT_DIR}" \
  "$@"
