#!/usr/bin/env bash
set -euo pipefail

# Paper-ready VisA-D&R reasoning evaluation for Qwen3-VL and Qwen3-VL + DeCo.
#
# This protocol is designed for stable, table-ready text metrics:
#   - official VisA-D&R reasoning annotations;
#   - official-style multi-turn QA order;
#   - deterministic generation;
#   - concise plain-text answers;
#   - global repetition suppression;
#   - full quality control before reporting metrics.

ADAR_ROOT="${ADAR_ROOT:-/home/vina04/ADAR}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl/bin/activate}"
TRANSFORMERS_ROOT="${TRANSFORMERS_ROOT:-$ADAR_ROOT/Deco-main/transformers-4.57.6}"
MODEL_PATH="${MODEL_PATH:-$ADAR_ROOT/Deco-main/qwen3_4b_vl_instruct}"
DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/VisA-DR-official}"
VISA_DR_JSON="${VISA_DR_JSON:-$DATA_ROOT/VisA/visa_dr.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ADAR_ROOT/Deco-main/runs}"
RUN_NAME="${RUN_NAME:-visa_dr_reasoning_paper_ready_strict_full_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/$RUN_NAME}"
DECO_THRESHOLD_TOP_P="${DECO_THRESHOLD_TOP_P:-1.0}"
DECO_THRESHOLD_TOP_K="${DECO_THRESHOLD_TOP_K:-200}"

source "$ENV_PATH"

export PYTHONPATH="$ADAR_ROOT/scripts:$TRANSFORMERS_ROOT/src:$TRANSFORMERS_ROOT/examples/deco${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vina04-triton-cache}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
mkdir -p "$TRITON_CACHE_DIR" "$RUN_ROOT"

ANSWER_INSTRUCTION="Answer concisely in plain English in no more than 60 words. Do not use Markdown, URLs, bullet points, numbering, or repeated phrases. Stop after the answer."

python "$ADAR_ROOT/scripts/repro_visa_dr_qwen3_deco_reasoning_original.py" \
  --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" \
  --visa-dr-json "$VISA_DR_JSON" \
  --out-dir "$RUN_ROOT" \
  --variants qwen3_original qwen3_deco \
  --max-new-tokens 256 \
  --repetition-penalty 1.20 \
  --no-repeat-ngram-size 3 \
  --ban-markdown-url \
  --deco-threshold-top-p "$DECO_THRESHOLD_TOP_P" \
  --deco-threshold-top-k "$DECO_THRESHOLD_TOP_K" \
  --answer-instruction "$ANSWER_INSTRUCTION" \
  --local-files-only \
  --no-sbert

for variant in qwen3_original qwen3_deco; do
  python "$ADAR_ROOT/scripts/qc_visa_dr_reasoning_outputs.py" \
    --predictions "$RUN_ROOT/$variant/reasoning_predictions.jsonl" \
    --out-dir "$RUN_ROOT/$variant"
done

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python "$ADAR_ROOT/scripts/summarize_visa_dr_reasoning_outputs.py" \
  --root "$RUN_ROOT" \
  --variants qwen3_original qwen3_deco \
  --sbert-cpu

echo "$RUN_ROOT"
