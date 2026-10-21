#!/usr/bin/env bash
set -euo pipefail

# ===== 按你的实际环境配置 =====
ROOT_DIR="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6"
DATASET_ROOT="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/VisA"
MODEL_PATH="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl"
OUT_DIR="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/runs/VisA_eval_3gpu_$(date +%Y%m%d_%H%M%S)"

# 与你给的单卡命令保持一致
SHOTS="${SHOTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"

# 透传额外参数，例如: --use_deco --alpha 0.6
EXTRA_ARGS=("$@")

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

# 12 类 -> 3 组 * 4 类
GROUP0="candle,capsules,cashew"
GROUP1="chewinggum,fryum,macaroni1"
GROUP2="macaroni2,pcb1,pcb2"
GROUP3="pcb3,pcb4,pipe_fryum"

run_one_gpu() {
  local gpu_id="$1"
  local categories="$2"
  local out_json="$3"
  local log_file="$4"

  PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python examples/deco/eval_visa_qwen3_vl_thinking.py \
    --dataset_root "${DATASET_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_json "${out_json}" \
    --shots "${SHOTS}" \
    --use_deco \
    --alpha 0.45 \
    --threshold_top_p 0.88 \
    --threshold_top_k 16 \
    --start_layer 21 \
    --end_layer 30 \
    --categories "${categories}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_file}" 2>&1
}

run_one_gpu 0 "${GROUP0}" "${OUT_DIR}/res_gpu0.json" "${OUT_DIR}/gpu0.log" & pid0=$!
run_one_gpu 2 "${GROUP1}" "${OUT_DIR}/res_gpu1.json" "${OUT_DIR}/gpu1.log" & pid1=$!
run_one_gpu 3 "${GROUP2}" "${OUT_DIR}/res_gpu2.json" "${OUT_DIR}/gpu2.log" & pid2=$!
run_one_gpu 4 "${GROUP3}" "${OUT_DIR}/res_gpu3.json" "${OUT_DIR}/gpu3.log" & pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

python - "${OUT_DIR}" << 'PY'
import json, os, sys

out_dir = sys.argv[1]
files = [os.path.join(out_dir, f"res_gpu{i}.json") for i in [0,1,2,3]]
for f in files:
    if not os.path.exists(f):
        raise SystemExit(f"Missing result file: {f}")

all_results = []
for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    all_results.extend(d["results"])

# stats[(shot, category)]
stats = {}
for r in all_results:
    shot = str(r["shot"])
    cat = r["category"]
    gt = r["ground_truth"]["label"]
    pred = r["prediction"]["label"]
    strict_parse_ok = bool(r["prediction"].get("strict_parse_ok", False))

    k = (shot, cat)
    if k not in stats:
        stats[k] = {
            "total": 0,
            "binary_correct": 0,
            "anomaly_count": 0,
            "subtype_correct_on_anomaly": 0,
            "strict_parse_ok_count": 0,
        }

    s = stats[k]
    s["total"] += 1
    s["binary_correct"] += int((gt != "good") == (pred != "good"))
    s["strict_parse_ok_count"] += int(strict_parse_ok)

    if gt != "good":
        s["anomaly_count"] += 1
        s["subtype_correct_on_anomaly"] += int(pred == gt)

summary = {}
all_shots = sorted({k[0] for k in stats}, key=lambda x: int(x))
#类别数量
for shot in all_shots:
    cats = sorted([k[1] for k in stats if k[0] == shot])
    if len(cats) != 12:
        raise SystemExit(f"[shot={shot}] expected 12 categories, got {len(cats)}: {cats}")

    per_category = {}
    for c in cats:
        s = stats[(shot, c)]
        total = s["total"]
        anom = s["anomaly_count"]

        binary_acc = s["binary_correct"] / total
        subtype_acc = s["subtype_correct_on_anomaly"] / anom if anom > 0 else None
        json_rate = s["strict_parse_ok_count"] / total

        per_category[c] = {
            "binary_acc": binary_acc,
            "subtype_acc_on_anomaly": subtype_acc,
            "json_parse_success_rate": json_rate,
            "total": total,
            "anomaly_count": anom,
        }

    binary_vals = [per_category[c]["binary_acc"] for c in cats]
    subtype_vals = [per_category[c]["subtype_acc_on_anomaly"] for c in cats]
    json_vals = [per_category[c]["json_parse_success_rate"] for c in cats]

    if any(v is None for v in subtype_vals):
        raise SystemExit(f"[shot={shot}] found category with anomaly_count=0, cannot compute 12-class mean subtype_acc_on_anomaly.")

    summary[shot] = {
        "num_categories": 12,
        "mean_binary_acc_12cls": sum(binary_vals) / 12.0,
        "mean_subtype_acc_on_anomaly_12cls": sum(subtype_vals) / 12.0,
        "mean_json_parse_success_rate_12cls": sum(json_vals) / 12.0,
        "per_category": per_category,
    }

save_path = os.path.join(out_dir, "VisA_12cls_mean_metrics.json")
with open(save_path, "w", encoding="utf-8") as fp:
    json.dump(summary, fp, ensure_ascii=False, indent=2)

print("Saved:", save_path)
for shot, v in summary.items():
    print(
        f"[shot={shot}] "
        f"mean_binary_acc_12cls={v['mean_binary_acc_12cls']:.6f}, "
        f"mean_subtype_acc_on_anomaly_12cls={v['mean_subtype_acc_on_anomaly_12cls']:.6f}, "
        f"mean_json_parse_success_rate_12cls={v['mean_json_parse_success_rate_12cls']:.6f}"
    )
PY

echo "Done. See: ${OUT_DIR}/VisA_12cls_mean_metrics.json"
