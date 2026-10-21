#!/usr/bin/env bash
set -euo pipefail
BASE=/home/vina04/ADAR
PY="$BASE/uv-envs/qwen3-vl/bin/python"
export PYTHONPATH="$BASE/Deco-main/transformers-4.57.6/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUNS="$BASE/Deco-main/runs"
COMMON=(--config "$RUNS/visa_dr_reasoning_paper_ready_banbang_full_20260909_154349/run_config.json"
        --roi "$RUNS/qwen3_visa_roi_20260922" --per-category 10
        --exclude-manifest "$RUNS/qwen3_visa_roi_qa_pilot_20260922/protocol.json")
for MODE in independent questions_only; do
    FLAG=--independent-questions
    if [[ "$MODE" == questions_only ]]; then FLAG=--questions-only-history; fi
    "$PY" -u "$BASE/scripts/validate_visa_roi_reasoning.py" "${COMMON[@]}" "$FLAG" \
        --out "$RUNS/qwen3_visa_roi_qa_${MODE}_validation100_20260924"
done
"$PY" "$BASE/scripts/summarize_roi_history_validation.py" "$RUNS"
