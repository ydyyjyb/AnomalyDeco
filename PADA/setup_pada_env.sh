#!/usr/bin/env bash
set -euo pipefail

# Server-side environment bootstrap for PADA.
# It keeps conda packages, temp files, and pip cache on the large disk.

BASE_DIR="${PADA_BASE_DIR:-/home/s414f/s414f1}"
ENV_NAME="${PADA_ENV_NAME:-pada-yd}"
PYTHON_VERSION="${PADA_PYTHON_VERSION:-3.8}"
CUDA_WHEEL="${PADA_CUDA_WHEEL:-cu118}"

ENV_DIR="${BASE_DIR}/conda_envs/${ENV_NAME}"
CONDA_PKGS_DIR="${BASE_DIR}/conda_pkgs"
TMP_DIR="${BASE_DIR}/tmp"
PIP_CACHE_DIR="${BASE_DIR}/pip_cache"

mkdir -p "${BASE_DIR}/conda_envs" "${CONDA_PKGS_DIR}" "${TMP_DIR}" "${PIP_CACHE_DIR}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export TMPDIR="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

touch "${HOME}/.bashrc"
grep -qxF "export CONDA_PKGS_DIRS=${CONDA_PKGS_DIR}" "${HOME}/.bashrc" || echo "export CONDA_PKGS_DIRS=${CONDA_PKGS_DIR}" >> "${HOME}/.bashrc"
grep -qxF "export TMPDIR=${TMP_DIR}" "${HOME}/.bashrc" || echo "export TMPDIR=${TMP_DIR}" >> "${HOME}/.bashrc"
grep -qxF "export PIP_CACHE_DIR=${PIP_CACHE_DIR}" "${HOME}/.bashrc" || echo "export PIP_CACHE_DIR=${PIP_CACHE_DIR}" >> "${HOME}/.bashrc"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please load Anaconda/Miniconda first, then rerun this script." >&2
  exit 1
fi

if [ ! -d "${ENV_DIR}" ]; then
  conda create -p "${ENV_DIR}" "python=${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

python -m pip install --upgrade pip

if [ "${CUDA_WHEEL}" = "cpu" ]; then
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "${SCRIPT_DIR}/requirements-pada.req"

python - <<'PY'
import torch
import cv2
import numpy
import scipy
import sklearn
import skimage
import pandas
import einops
import ftfy
import regex
import sympy
print("PADA environment check passed")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo "Environment ready."
echo "Activate it with:"
echo "  conda activate ${ENV_DIR}"
echo
echo "Run from the PADA project directory on the server, for example:"
echo "  cd /home/s414f/s414f1/ADAR/PADA"
echo "  python MaPLe_cls.py --dataset mvtec --class_name tile --k_shot 1 --gpu-id 0"
