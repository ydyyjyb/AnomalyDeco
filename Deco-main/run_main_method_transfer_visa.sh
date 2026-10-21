#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/vina04/ADAR/Deco-main
ROOT="$PROJECT/transformers-4.57.6"
DATA_ROOT=/home/vina04/ADAR/evaluation/ODinW-13/data/VisA
RUN_ROOT="${OUT_ROOT:-$PROJECT/runs/main_method_transfer_visa_20261019}"

source /home/vina04/ADAR/uv-envs/qwen3-vl/bin/activate
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/examples/deco"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/vina04-triton-cache
mkdir -p "$TRITON_CACHE_DIR" "$RUN_ROOT"

run_one() {
  local name="$1"
  local family="$2"
  local model_path="$3"
  local output_dir="$RUN_ROOT/$name"

  if [[ -f "$output_dir/mechanism_summary.json" ]]; then
    echo "[SKIP] Completed result exists: $output_dir"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[ERROR] Partial output exists; refusing to overwrite: $output_dir" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
  echo "[START] model=$name family=$family path=$model_path output=$output_dir"

  ADAR_MODEL_FAMILY="$family" python examples/deco/cross_arch_main_method.py \
    --visa-root "$DATA_ROOT" \
    --visa-split-set test \
    --model-path "$model_path" \
    --output-dir "$output_dir" \
    --run-all \
    --attn-implementation sdpa \
    --local-files-only \
    --skip-generation \
    --no-text-prior-baseline \
    --prompt-mode category_defects_guarded \
    --prompt-defect-limit 8 \
    --max-defect-labels 16 \
    --score-label-aggregation max \
    --score-deco-mode multi_top3 \
    --multi-deco-k 3 \
    --deco-alpha 0.6 \
    --trend-layers 0:34 \
    --deco-layers 20:29 \
    --metric-threshold-policy youden \
    --stream-jsonl \
    2>&1 | tee "$output_dir/run.log"

  echo "[DONE] model=$name output=$output_dir"
}

echo "[RUN_ROOT] $RUN_ROOT"
run_one qwen3_4b qwen3_vl "$PROJECT/qwen3_4b_vl_instruct"
run_one qwen2_2b qwen2_vl "$PROJECT/qwen2_vl_2b_instruct"
run_one qwen2_7b qwen2_vl "$PROJECT/qwen2_vl_7b_instruct"
run_one llava_ov_0_5b llava_onevision "$PROJECT/weights/visa_dr_open_models/llava_ov_0_5b"
run_one llava_ov_7b llava_onevision "$PROJECT/weights/visa_dr_open_models/llava_ov_7b"

echo "[COMPLETE] All model runs finished under $RUN_ROOT"
