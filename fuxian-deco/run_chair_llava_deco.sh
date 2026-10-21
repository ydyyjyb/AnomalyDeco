#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/Deco-main/llava-v1.5-7b}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-${PROJECT_DIR}/Deco-main/clip-vit-large-patch14-336}"
COCO_IMG="${COCO_IMG:-${PROJECT_DIR}/data/COCO_2014/val2014}"
COCO_ANN="${COCO_ANN:-${PROJECT_DIR}/data/COCO_2014/annotations_trainval2014/annotations}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/chair_llava_deco_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-5}"

MAX_SAMPLES="${MAX_SAMPLES:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_DECO="${USE_DECO:-1}"
ALPHA="${ALPHA:-0.6}"
THRESHOLD_TOP_P="${THRESHOLD_TOP_P:-0.9}"
THRESHOLD_TOP_K="${THRESHOLD_TOP_K:-20}"
START_LAYER="${START_LAYER:-20}"
END_LAYER="${END_LAYER:-29}"

mkdir -p "${OUT_DIR}"
cd "${SCRIPT_DIR}"

CAP_FILE="${OUT_DIR}/captions.jsonl"
CHAIR_FILE="${OUT_DIR}/chair_metrics.json"
CACHE_FILE="${OUT_DIR}/chair_cache.pkl"

echo "===== DeCo CHAIR Reproduction ====="
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VISION_TOWER_PATH=${VISION_TOWER_PATH}"
echo "COCO_IMG=${COCO_IMG}"
echo "COCO_ANN=${COCO_ANN}"
echo "OUT_DIR=${OUT_DIR}"
echo "GPU_ID=${GPU_ID}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "USE_DECO=${USE_DECO}"
echo "LAYERS=[${START_LAYER},${END_LAYER})"
echo "==================================="

gen_args=(
  --model-path "${MODEL_PATH}"
  --image-folder "${COCO_IMG}"
  --answers-file "${CAP_FILE}"
  --max-samples "${MAX_SAMPLES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --alpha "${ALPHA}"
  --threshold_top_p "${THRESHOLD_TOP_P}"
  --threshold_top_k "${THRESHOLD_TOP_K}"
  --start_layer "${START_LAYER}"
  --end_layer "${END_LAYER}"
)

if [[ "${USE_DECO}" == "1" ]]; then
  gen_args+=(--use_deco)
fi

PYTHONPATH="${SCRIPT_DIR}" CUDA_VISIBLE_DEVICES="${GPU_ID}" VISION_TOWER_PATH="${VISION_TOWER_PATH}" \
  python chair_llava.py "${gen_args[@]}"

python chair.py \
  --cap_file "${CAP_FILE}" \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path "${COCO_ANN}" \
  --cache "${CACHE_FILE}" \
  --save_path "${CHAIR_FILE}"

echo "[CAPTIONS] ${CAP_FILE}"
echo "[CHAIR] ${CHAIR_FILE}"
