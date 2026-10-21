#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# Multi-GPU VisA evaluation with DeCo enabled
# GPU selection is controlled ONLY by AVAILABLE_GPUS array below
# --------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/transformers-4.57.6}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/../evaluation/ODinW-13/data/VisA}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/qwen3_4b_vl}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/visa_multi_gpu_deco_$(date +%Y%m%d_%H%M%S)}"

# ===== 你只需要改这里 =====
AVAILABLE_GPUS=(5)

# 如果你想自动使用所有 GPU，可以把上面改成空数组：
# AVAILABLE_GPUS=()
  # candle
  # capsules
  # cashew
  # chewinggum
  # fryum
  # macaroni1
  # macaroni2
  # pipe_fryum
  # pcb1
  # pcb2
  # pcb3
  # pcb4
# ===== VisA 全部类别 =====
ALL_CATEGORIES=(
  pcb1
)

# ===== Basic eval settings =====
SHOTS="${SHOTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
VISA_SOURCE="${VISA_SOURCE:-per_category_csv}"

# ===== DeCo settings =====
DECO_ALPHA="${DECO_ALPHA:-0.60}"
DECO_TOP_P="${DECO_TOP_P:-0.92}"
DECO_TOP_K="${DECO_TOP_K:-32}"
START_LAYER="${START_LAYER:-18}"
END_LAYER="${END_LAYER:-32}"
LAMBDA_NORM="${LAMBDA_NORM:-0.50}"
ETA_HEDGE="${ETA_HEDGE:-0.10}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-1}"
GAMMA="${GAMMA:-8.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-1}"

# ===== Defect-aware anchor lexicons =====
ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"

EXTRA_ARGS=("$@")

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"
cd "${ROOT_DIR}"

if [[ ! -f "examples/deco/eval_visa_qwen3_vl_thinking.py" ]]; then
  echo "[ERROR] Missing script: ${ROOT_DIR}/examples/deco/eval_visa_qwen3_vl_thinking.py"
  exit 1
fi

# ===== 自动补全 GPU 列表 =====
if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi

NUM_GPU=${#AVAILABLE_GPUS[@]}
if [[ ${NUM_GPU} -lt 1 ]]; then
  echo "[ERROR] No GPU found."
  exit 1
fi

# ===== 自动按 GPU 数量分配类别 =====
CATEGORY_GROUPS=()
for ((i=0; i<NUM_GPU; i++)); do
  CATEGORY_GROUPS[i]=""
done

for ((i=0; i<${#ALL_CATEGORIES[@]}; i++)); do
  gpu_slot=$((i % NUM_GPU))
  if [[ -z "${CATEGORY_GROUPS[gpu_slot]}" ]]; then
    CATEGORY_GROUPS[gpu_slot]="${ALL_CATEGORIES[i]}"
  else
    CATEGORY_GROUPS[gpu_slot]+=",${ALL_CATEGORIES[i]}"
  fi
done

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

echo "===== Runtime Config ====="
echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "SHOTS=${SHOTS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "VISA_SOURCE=${VISA_SOURCE}"
echo "AVAILABLE_GPUS=${AVAILABLE_GPUS[*]}"
echo "NUM_GPU=${NUM_GPU}"
echo "ALL_CATEGORIES=${ALL_CATEGORIES[*]}"
echo "DECO_ALPHA=${DECO_ALPHA}, DECO_TOP_P=${DECO_TOP_P}, DECO_TOP_K=${DECO_TOP_K}, LAYERS=[${START_LAYER},${END_LAYER})"
echo "LAMBDA_NORM=${LAMBDA_NORM}, ETA_HEDGE=${ETA_HEDGE}, USE_LOGSUMEXP=${USE_LOGSUMEXP}, ENABLE_HEDGE=${ENABLE_HEDGE}, GAMMA=${GAMMA}"
for ((i=0; i<NUM_GPU; i++)); do
  echo "GPU ${AVAILABLE_GPUS[i]} -> ${CATEGORY_GROUPS[i]}"
done
echo "=========================="

run_one_group() {
  local gpu_id="$1"
  local categories="$2"
  local out_json="$3"
  local log_file="$4"

  PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python examples/deco/eval_visa_qwen3_vl_thinking.py \
      --dataset_root "${DATASET_ROOT}" \
      --model_name_or_path "${MODEL_PATH}" \
      --output_json "${out_json}" \
      --visa_source "${VISA_SOURCE}" \
      --shots "${SHOTS}" \
      --categories "${categories}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --allow_attn_fallback \
      "${DECO_FLAGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      > "${log_file}" 2>&1
}

PIDS=()

for ((i=0; i<NUM_GPU; i++)); do
  gpu_id="${AVAILABLE_GPUS[i]}"
  categories="${CATEGORY_GROUPS[i]}"

  if [[ -z "${categories}" ]]; then
    continue
  fi

  out_json="${OUT_DIR}/res_gpu${i}.json"
  log_file="${OUT_DIR}/logs/gpu${i}.log"

  echo "[LAUNCH] GPU ${gpu_id} -> ${categories}"
  run_one_group "${gpu_id}" "${categories}" "${out_json}" "${log_file}" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

python - "${OUT_DIR}" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]

files = sorted(
    [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("res_gpu") and f.endswith(".json")
    ]
)

if len(files) == 0:
    raise SystemExit("[ERROR] No result files found.")

all_results = []
meta = None

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    m = d.get("meta", {})
    if not m.get("use_deco", False):
        raise SystemExit(f"[ERROR] DeCo is not enabled in {os.path.basename(fp)} (meta.use_deco={m.get('use_deco')})")
    all_results.extend(d.get("results", []))
    if meta is None:
        meta = m

by_shot_cat = {}
for r in all_results:
    shot = str(r.get("shot"))
    cat = str(r.get("category"))
    gt = r.get("ground_truth", {}).get("label", "good")
    pred = r.get("prediction", {}).get("label", "good")
    strict_ok = bool(r.get("prediction", {}).get("strict_parse_ok", False))

    key = (shot, cat)
    if key not in by_shot_cat:
        by_shot_cat[key] = {
            "total": 0,
            "binary_correct": 0,
            "anomaly_count": 0,
            "subtype_correct_on_anomaly": 0,
            "strict_parse_ok_count": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    s = by_shot_cat[key]
    s["total"] += 1
    gt_anom = gt != "good"
    pred_anom = pred != "good"
    s["binary_correct"] += int(gt_anom == pred_anom)
    if gt_anom and pred_anom:
        s["tp"] += 1
    elif (not gt_anom) and pred_anom:
        s["fp"] += 1
    elif (not gt_anom) and (not pred_anom):
        s["tn"] += 1
    else:
        s["fn"] += 1
    if gt_anom:
        s["anomaly_count"] += 1
        s["subtype_correct_on_anomaly"] += int(pred == gt)
    s["strict_parse_ok_count"] += int(strict_ok)

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
        }

    binary_vals = [per_category[c]["binary_acc"] for c in cats]
    subtype_vals = [per_category[c]["subtype_acc_on_anomaly"] for c in cats if per_category[c]["subtype_acc_on_anomaly"] is not None]
    parse_vals = [per_category[c]["json_parse_success_rate"] for c in cats]
    summary[shot] = {
        "num_categories": len(cats),
        "mean_binary_acc": sum(binary_vals) / len(binary_vals) if len(binary_vals) > 0 else None,
        "mean_subtype_acc_on_anomaly": sum(subtype_vals) / len(subtype_vals) if len(subtype_vals) > 0 else None,
        "mean_json_parse_success_rate": sum(parse_vals) / len(parse_vals) if len(parse_vals) > 0 else None,
        "per_category": per_category,
    }

save_path = os.path.join(out_dir, "visa_results_multi_gpu_deco.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "summary": summary, "results": all_results}, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved merged result: {save_path}")
for shot in sorted(summary.keys(), key=lambda x: int(x)):
    v = summary[shot]
    print(
        f"[shot={shot}] mean_binary_acc={v['mean_binary_acc']:.6f}, "
        f"mean_subtype_acc_on_anomaly={v['mean_subtype_acc_on_anomaly'] if v['mean_subtype_acc_on_anomaly'] is not None else 'None'}, "
        f"mean_json_parse_success_rate={v['mean_json_parse_success_rate']:.6f}"
    )
PY

echo "[DONE] ${OUT_DIR}/visa_results_multi_gpu_deco.json"