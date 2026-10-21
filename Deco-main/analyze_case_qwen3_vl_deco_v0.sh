cd /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6
export PYTHONPATH="$(pwd)/src"
export CUDA_VISIBLE_DEVICES=3

MODEL=/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl
DATA=/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec
IMG=/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec/bottle/test/contamination/002.png
CATEGORY=bottle
SHOT=0

# 从路径自动提取信息
IMG_STEM="$(basename "$IMG" | sed 's/\.[^.]*$//')"
DEFECT_TYPE="$(basename "$(dirname "$IMG")")"
TS="$(date +%Y%m%d_%H%M%S)"

OUT_DIR=../case_analyze
mkdir -p "$OUT_DIR"

RUN_NAME="${CATEGORY}_${DEFECT_TYPE}_${IMG_STEM}_${SHOT}shot_nodeco"

python examples/deco/analyze_case_qwen3_vl_deco.py \
  --model_name_or_path "$MODEL" \
  --dataset_root "$DATA" \
  --category "$CATEGORY" \
  --image_path "$IMG" \
  --ground_truth_label "$DEFECT_TYPE" \
  --shot "$SHOT" \
  --assistant_prefix '{"label": "' \
  --start_layer 20 --end_layer 29 \
  --max_new_tokens 16000 \
  --output_json "${OUT_DIR}/${RUN_NAME}.json" \
  --output_plot "${OUT_DIR}/${RUN_NAME}.png" \
  --allow_attn_fallback
