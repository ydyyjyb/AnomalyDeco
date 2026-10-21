#!/usr/bin/env bash
set -euo pipefail

ADAR_ROOT="${ADAR_ROOT:-/home/vina04/ADAR}"
ENV_PATH="${ENV_PATH:-$ADAR_ROOT/uv-envs/qwen3-vl}"
MODEL_ROOT="${MODEL_ROOT:-$ADAR_ROOT/Deco-main/weights/visa_dr_open_models}"
DATA_ROOT="${DATA_ROOT:-$ADAR_ROOT/evaluation/ODinW-13/data}"
VISA_DR_JSON="${VISA_DR_JSON:-$ADAR_ROOT/evaluation/VisA-DR/visa_dr.json}"
RUN_ROOT="${RUN_ROOT:-$ADAR_ROOT/Deco-main/runs/visa_dr_open_models_$(date +%Y%m%d_%H%M%S)}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-$ADAR_ROOT/logs/visa_dr_open_model_downloads.log}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

source "$ENV_PATH/bin/activate"
cd "$ADAR_ROOT"
mkdir -p "$RUN_ROOT"

run_one() {
  local model="$1"
  local path="$2"
  shift 2
  local out="$RUN_ROOT/$model"
  mkdir -p "$out"

  echo "Waiting for pinned model files: $model"
  while ! grep -q "^DONE $model " "$DOWNLOAD_LOG"; do
    if grep -q "^FAILED $model:" "$DOWNLOAD_LOG"; then
      echo "Download failed for $model; see $DOWNLOAD_LOG" >&2
      return 1
    fi
    sleep 30
  done

  if [[ -f "$out/open_model_summary.json" ]] && \
     python - "$out/open_model_summary.json" <<'PY'
import json, sys
try:
    obj = json.load(open(sys.argv[1], encoding="utf-8"))
    raise SystemExit(0 if obj.get("reasoning_images") == 1000 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "Already complete; skipping $model"
    return 0
  fi

  if [[ "$model" == "internvl2_26b" ]]; then
    python -c 'import bitsandbytes' 2>/dev/null || \
      uv pip install --python "$ENV_PATH/bin/python" bitsandbytes
  fi

  if [[ ! -f "$out/reasoning_predictions.jsonl" ]]; then
    local smoke="$out/smoke_10_images"
    mkdir -p "$smoke"
    python -u "$ADAR_ROOT/scripts/repro_visa_dr_open_models.py" \
      --model "$model" --model-path "$path" \
      --data-root "$DATA_ROOT" --visa-dr-json "$VISA_DR_JSON" \
      --out-dir "$smoke" --max-image-side 1024 --max-new-tokens 4096 \
      --log-every 1 --limit-reasoning-per-category 1 \
      2>&1 | tee "$out/smoke.log"
  fi

  python -u "$ADAR_ROOT/scripts/repro_visa_dr_open_models.py" \
    --model "$model" \
    --model-path "$path" \
    --data-root "$DATA_ROOT" \
    --visa-dr-json "$VISA_DR_JSON" \
    --out-dir "$out" \
    --max-image-side 1024 \
    --max-new-tokens 4096 \
    --log-every 10 \
    "$@" 2>&1 | tee "$out/run.log"
}

# Do not compete with a currently active first-model full run if launched as a
# background continuation after a smoke test.
if [[ -n "$WAIT_FOR_PID" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 30; done
fi

# Open-weight base models only. GPT-4V/4o have no public downloadable weights.
run_one llava_ov_0_5b "$MODEL_ROOT/llava_ov_0_5b"
run_one internvl2_8b "$MODEL_ROOT/internvl2_8b"
run_one llava_ov_7b "$MODEL_ROOT/llava_ov_7b"
run_one ixc_2_5_7b "$MODEL_ROOT/ixc_2_5_7b"
run_one internvl2_26b "$MODEL_ROOT/internvl2_26b" --load-in-8bit

echo "All open-model runs completed: $RUN_ROOT"
