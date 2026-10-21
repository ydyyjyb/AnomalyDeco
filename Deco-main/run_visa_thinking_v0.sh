#!/usr/bin/env bash
set -u

# --------------------------------------------
# 3-GPU VisA evaluation with DeCo enabled
# Balanced normal/anomaly experiment version
# --------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/transformers-4.57.6}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/../evaluation/ODinW-13/data/VisA}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/qwen3_4b_vl}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/visa_3gpu_deco_balanced_$(date +%Y%m%d_%H%M%S)}"

# Basic eval settings
SHOTS="${SHOTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_PIXELS="${MIN_PIXELS:-262144}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
GPU_IDS="${GPU_IDS:-3,5,6}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# 说明：
# 这里的等量 normal/anomaly 采样，不是在 shell 里做，
# 而是要求 eval_visa_qwen3_vl_thinking.py 内部 discover_visa() 已经实现：
#   good 最多 N 张，anomaly 最多 N 张
# 下面两个变量只是为了让实验记录更清楚。
BALANCED_GOOD_PER_CATEGORY="${BALANCED_GOOD_PER_CATEGORY:-100}"
BALANCED_ANOMALY_PER_CATEGORY="${BALANCED_ANOMALY_PER_CATEGORY:-100}"

# More conservative DeCo settings
DECO_ALPHA="${DECO_ALPHA:-0.45}"
DECO_TOP_P="${DECO_TOP_P:-0.88}"
DECO_TOP_K="${DECO_TOP_K:-16}"
START_LAYER="${START_LAYER:-21}"
END_LAYER="${END_LAYER:-30}"
LAMBDA_NORM="${LAMBDA_NORM:-1.00}"
ETA_HEDGE="${ETA_HEDGE:-0.20}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-1}"
GAMMA="${GAMMA:-8.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-1}"
USE_PROMPT_SUBTYPES="${USE_PROMPT_SUBTYPES:-0}"

ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"

GROUP0="${GROUP0:-candle,capsules,cashew,chewinggum}"
GROUP1="${GROUP1:-fryum,macaroni1,macaroni2,pipe_fryum}"
GROUP2="${GROUP2:-pcb1,pcb2,pcb3,pcb4}"

EXTRA_ARGS=("$@")

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"
cd "${ROOT_DIR}"

if [[ ! -f "examples/deco/eval_visa_qwen3_vl_thinking.py" ]]; then
  echo "[ERROR] Missing script: ${ROOT_DIR}/examples/deco/eval_visa_qwen3_vl_thinking.py"
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_ARRAY[@]} -ne 3 ]]; then
  echo "[ERROR] GPU_IDS must contain exactly 3 ids, got: ${GPU_IDS}"
  exit 1
fi

DECO_FLAGS=(
  --use_deco
  --anchor_strategy defect_aware
  --alpha "${DECO_ALPHA}"
  --threshold_top_p "${DECO_TOP_P}"
  --threshold_top_k "${DECO_TOP_K}"
  --start_layer "${START_LAYER}"
  --end_layer "${END_LAYER}"
  --lambda_norm "${LAMBDA_NORM}"
  --eta_hedge "${ETA_HEDGE}"
  --gamma "${GAMMA}"
  --anchor_defect_words "${ANCHOR_DEFECT_WORDS}"
  --anchor_norm_words "${ANCHOR_NORM_WORDS}"
  --anchor_hedge_words "${ANCHOR_HEDGE_WORDS}"
)

if [[ "${USE_LOGSUMEXP}" == "1" ]]; then
  DECO_FLAGS+=(--use_logsumexp)
fi
if [[ "${ENABLE_HEDGE}" == "1" ]]; then
  DECO_FLAGS+=(--enable_hedge)
fi
if [[ "${USE_PROMPT_SUBTYPES}" == "1" ]]; then
  DECO_FLAGS+=(--use_prompt_subtypes)
fi

echo "===== Runtime Config ====="
echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "SHOTS=${SHOTS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MIN_PIXELS=${MIN_PIXELS}"
echo "MAX_PIXELS=${MAX_PIXELS}"
echo "ATTN_IMPL=${ATTN_IMPL}"
echo "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
echo "GPU_IDS=${GPU_IDS}"
echo "GROUP0=${GROUP0}"
echo "GROUP1=${GROUP1}"
echo "GROUP2=${GROUP2}"
echo "BALANCED_GOOD_PER_CATEGORY=${BALANCED_GOOD_PER_CATEGORY}"
echo "BALANCED_ANOMALY_PER_CATEGORY=${BALANCED_ANOMALY_PER_CATEGORY}"
echo "DECO_ALPHA=${DECO_ALPHA}, DECO_TOP_P=${DECO_TOP_P}, DECO_TOP_K=${DECO_TOP_K}, LAYERS=[${START_LAYER},${END_LAYER})"
echo "Sampling policy: equal normal/anomaly per category is handled INSIDE python discover_visa()"
echo "=========================="

run_one_group() {
  local gpu_id="$1"
  local categories="$2"
  local out_json="$3"
  local log_file="$4"

  PYTHONPATH="$(pwd)/src" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
    python examples/deco/eval_visa_qwen3_vl_thinking.py \
      --dataset_root "${DATASET_ROOT}" \
      --model_name_or_path "${MODEL_PATH}" \
      --output_json "${out_json}" \
      --shots "${SHOTS}" \
      --categories "${categories}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --min_pixels "${MIN_PIXELS}" \
      --max_pixels "${MAX_PIXELS}" \
      --attn_implementation "${ATTN_IMPL}" \
      --allow_attn_fallback \
      "${DECO_FLAGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      > "${log_file}" 2>&1
}

echo "[LAUNCH] GPU ${GPU_ARRAY[0]} -> ${GROUP0}"
run_one_group "${GPU_ARRAY[0]}" "${GROUP0}" "${OUT_DIR}/res_gpu0.json" "${OUT_DIR}/logs/gpu0.log" & pid0=$!

echo "[LAUNCH] GPU ${GPU_ARRAY[1]} -> ${GROUP1}"
run_one_group "${GPU_ARRAY[1]}" "${GROUP1}" "${OUT_DIR}/res_gpu1.json" "${OUT_DIR}/logs/gpu1.log" & pid1=$!

echo "[LAUNCH] GPU ${GPU_ARRAY[2]} -> ${GROUP2}"
run_one_group "${GPU_ARRAY[2]}" "${GROUP2}" "${OUT_DIR}/res_gpu2.json" "${OUT_DIR}/logs/gpu2.log" & pid2=$!

status0=0
status1=0
status2=0

wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
wait "${pid2}" || status2=$?

echo "[STATUS] gpu0=${status0}, gpu1=${status1}, gpu2=${status2}"

python - "${OUT_DIR}" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
candidate_files = [os.path.join(out_dir, f"res_gpu{i}.json") for i in [0, 1, 2]]
files = [fp for fp in candidate_files if os.path.exists(fp)]

if len(files) == 0:
    raise SystemExit("[ERROR] No result files found. All workers failed.")

all_results = []
meta = None

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    m = d.get("meta", {})
    all_results.extend(d.get("results", []))
    if meta is None:
        meta = m

by_shot_cat = {}
for r in all_results:
    shot = str(r.get("shot"))
    cat = str(r.get("category"))
    gt = r.get("ground_truth", {}).get("label", "good")
    gt_binary = r.get("ground_truth", {}).get("binary_label", "good")
    pred = r.get("prediction", {}).get("label", "invalid")
    strict_ok = bool(r.get("prediction", {}).get("strict_parse_ok", False))
    extraction_ok = bool(r.get("prediction", {}).get("extraction_ok", False))

    key = (shot, cat)
    if key not in by_shot_cat:
        by_shot_cat[key] = {
            "total": 0,
            "binary_correct": 0,
            "anomaly_count": 0,
            "subtype_correct_on_anomaly": 0,
            "strict_parse_ok_count": 0,
            "extraction_ok_count": 0,
            "invalid_count": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    s = by_shot_cat[key]
    s["total"] += 1

    gt_anom = gt_binary != "good"
    pred_anom = pred not in {"good", "invalid"}

    s["binary_correct"] += int(gt_anom == pred_anom)

    if gt_anom and pred_anom:
        s["tp"] += 1
    elif (not gt_anom) and pred_anom:
        s["fp"] += 1
    elif (not gt_anom) and pred == "good":
        s["tn"] += 1
    else:
        s["fn"] += 1

    if gt_anom:
        s["anomaly_count"] += 1
        s["subtype_correct_on_anomaly"] += int(pred == gt)

    s["strict_parse_ok_count"] += int(strict_ok)
    s["extraction_ok_count"] += int(extraction_ok)
    s["invalid_count"] += int(pred == "invalid")

summary = {}
shots = sorted({k[0] for k in by_shot_cat}, key=lambda x: int(x))
for shot in shots:
    cats = sorted([k[1] for k in by_shot_cat if k[0] == shot])
    per_category = {}
    for c in cats:
        s = by_shot_cat[(shot, c)]
        total = max(1, s["total"])
        tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision is not None and recall is not None and (precision + recall) > 0)
            else None
        )

        per_category[c] = {
            "total": s["total"],
            "binary_acc": s["binary_correct"] / total,
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "subtype_acc_on_anomaly": (
                s["subtype_correct_on_anomaly"] / s["anomaly_count"] if s["anomaly_count"] > 0 else None
            ),
            "json_parse_success_rate": s["strict_parse_ok_count"] / total,
            "extraction_success_rate": s["extraction_ok_count"] / total,
            "invalid_rate": s["invalid_count"] / total,
        }

    binary_vals = [per_category[c]["binary_acc"] for c in cats]
    subtype_vals = [per_category[c]["subtype_acc_on_anomaly"] for c in cats if per_category[c]["subtype_acc_on_anomaly"] is not None]
    parse_vals = [per_category[c]["json_parse_success_rate"] for c in cats]
    extract_vals = [per_category[c]["extraction_success_rate"] for c in cats]
    invalid_vals = [per_category[c]["invalid_rate"] for c in cats]

    summary[shot] = {
        "num_categories": len(cats),
        "mean_binary_acc": sum(binary_vals) / len(binary_vals) if len(binary_vals) > 0 else None,
        "mean_subtype_acc_on_anomaly": sum(subtype_vals) / len(subtype_vals) if len(subtype_vals) > 0 else None,
        "mean_json_parse_success_rate": sum(parse_vals) / len(parse_vals) if len(parse_vals) > 0 else None,
        "mean_extraction_success_rate": sum(extract_vals) / len(extract_vals) if len(extract_vals) > 0 else None,
        "mean_invalid_rate": sum(invalid_vals) / len(invalid_vals) if len(invalid_vals) > 0 else None,
        "per_category": per_category,
    }

save_path = os.path.join(out_dir, "visa_results_3gpu_deco_balanced.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "summary": summary, "results": all_results}, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved merged result: {save_path}")
for shot in sorted(summary.keys(), key=lambda x: int(x)):
    v = summary[shot]
    print(
        f"[shot={shot}] "
        f"mean_binary_acc={v['mean_binary_acc']:.6f}, "
        f"mean_subtype_acc_on_anomaly={v['mean_subtype_acc_on_anomaly'] if v['mean_subtype_acc_on_anomaly'] is not None else 'None'}, "
        f"mean_json_parse_success_rate={v['mean_json_parse_success_rate']:.6f}, "
        f"mean_extraction_success_rate={v['mean_extraction_success_rate']:.6f}, "
        f"mean_invalid_rate={v['mean_invalid_rate']:.6f}"
    )
PY

echo "[DONE] ${OUT_DIR}/visa_results_3gpu_deco_balanced.json"
echo "[HINT] Logs are in: ${OUT_DIR}/logs"