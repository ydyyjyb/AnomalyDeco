#!/usr/bin/env bash
set -euo pipefail

# ====== 可按需修改 ======
ROOT="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6"
MODEL="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl"
DATA="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec"
CATEGORY="bottle"
GPU_ID="5"

SAMPLES_PER_SUBTYPE=5
SHOT=0
EXCLUDE_GOOD=0
SAVE_CASE_ARTIFACTS=1

MAX_NEW_TOKENS=16000
SCORE_STEPS="0,1,2"
START_LAYER=20
END_LAYER=29

OUT_DIR="../case_analyze"
TS="$(date +%Y%m%d_%H%M%S)"
COMPARE_TOP_N=20
# ======================

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$OUT_DIR"

DECO_JSON="${OUT_DIR}/${CATEGORY}_batch_${SAMPLES_PER_SUBTYPE}shot_deco_${TS}.json"
NODECO_JSON="${OUT_DIR}/${CATEGORY}_batch_${SAMPLES_PER_SUBTYPE}shot_nodeco_${TS}.json"

EXTRA_ARGS=()
if [[ "$EXCLUDE_GOOD" == "1" ]]; then
  EXTRA_ARGS+=(--exclude_good)
fi
if [[ "$SAVE_CASE_ARTIFACTS" == "1" ]]; then
  EXTRA_ARGS+=(--save_case_artifacts)
fi

COMMON_ARGS=(
  --model_name_or_path "$MODEL"
  --dataset_root "$DATA"
  --category "$CATEGORY"
  --samples_per_subtype "$SAMPLES_PER_SUBTYPE"
  --shot "$SHOT"
  --assistant_prefix '{"label": "'
  --start_layer "$START_LAYER"
  --end_layer "$END_LAYER"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --score_analysis_steps "$SCORE_STEPS"
  --allow_attn_fallback
)

echo "===== Run DeCo ====="
python examples/deco/analyze_case_qwen3_vl_deco.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output_json "$DECO_JSON"

echo "===== Run No-DeCo ====="
python examples/deco/analyze_case_qwen3_vl_deco.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --disable_deco \
  --output_json "$NODECO_JSON"

echo "Done."
echo "DeCo JSON:   $DECO_JSON"
echo "No-DeCo JSON:$NODECO_JSON"

python - "$DECO_JSON" "$NODECO_JSON" <<'PY'
import json, sys
from pathlib import Path

deco = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
node = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

dmap = {x["image_path"]: x for x in deco["results"]}
nmap = {x["image_path"]: x for x in node["results"]}
keys = sorted(set(dmap) & set(nmap))

same_tokens = 0
max_abs_delta = 0.0

for k in keys:
    dj = json.loads(Path(dmap[k]["output_json"]).read_text(encoding="utf-8"))
    nj = json.loads(Path(nmap[k]["output_json"]).read_text(encoding="utf-8"))

    if dj["generation"]["generated_token_ids"] == nj["generation"]["generated_token_ids"]:
        same_tokens += 1

    drows = {r["step"]: r for r in dj["analysis"]["generation_score_analysis"].get("target_probs_by_step", [])}
    nrows = {r["step"]: r for r in nj["analysis"]["generation_score_analysis"].get("target_probs_by_step", [])}

    for s in set(drows) & set(nrows):
        for kk, dv in drows[s].items():
            if kk == "step" or kk not in nrows[s]:
                continue
            ndv = nrows[s][kk]
            max_abs_delta = max(max_abs_delta, abs(float(dv) - float(ndv)))

print("overlap_cases:", len(keys))
print("same_generated_token_ids:", same_tokens)
print("max_abs_target_prob_delta:", max_abs_delta)
PY

