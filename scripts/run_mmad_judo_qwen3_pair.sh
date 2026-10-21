#!/usr/bin/env bash
set -u

ADAR_ROOT=/home/vina04/ADAR
SCRIPT="$ADAR_ROOT/scripts/repro_mmad_qa_eval_judo_full.py"
MODEL="$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct"
DATA_ROOT="$ADAR_ROOT/evaluation/ODinW-13/data"
MMAD_ROOT="$ADAR_ROOT/evaluation/MMAD"
RUN_ROOT="$ADAR_ROOT/Deco-main/runs/mmad_judo_qwen3_pair_20261013"
LOG="$RUN_ROOT/driver.log"

mkdir -p "$RUN_ROOT"
exec >>"$LOG" 2>&1

echo "=== MMAD/JUDO Qwen3 paired evaluation ==="
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "model=$MODEL"
echo "run_root=$RUN_ROOT"

source "$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate"
export ADAR_ROOT
export PYTHONPATH="$ADAR_ROOT/Deco-main/transformers-4.57.6/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

COMMON_ARGS=(
  --adar-root "$ADAR_ROOT"
  --model-path "$MODEL"
  --model-family qwen3
  --data-root "$DATA_ROOT"
  --mmad-root "$MMAD_ROOT"
  --mmad-json "$MMAD_ROOT/mmad.json"
  --batch-size 4
  --image-size 512
  --grid-size 16
  --max-new-tokens 1024
  --attn-implementation sdpa
  --local-files-only
)

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" "${COMMON_ARGS[@]}" \
  --out-dir "$RUN_ROOT/native"
native_status=$?
echo "native_exit=$native_status finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" "${COMMON_ARGS[@]}" \
  --out-dir "$RUN_ROOT/deco" \
  --use-deco-generation \
  --deco-alpha 0.6 \
  --deco-threshold-top-p 0.9 \
  --deco-threshold-top-k 20 \
  --deco-layers 20:29
deco_status=$?
echo "deco_exit=$deco_status finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$native_status" -ne 0 ] || [ "$deco_status" -ne 0 ]; then
  exit 1
fi
echo "PAIR_COMPLETE"
