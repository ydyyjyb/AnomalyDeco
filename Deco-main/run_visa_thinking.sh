#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/transformers-4.57.6}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/../evaluation/ODinW-13/data/VisA}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/qwen3_8b_vl_instruct}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/visa_multi_gpu_deco_$(date +%Y%m%d_%H%M%S)}"

# ===== 你只需要改这里 =====
read -r -a AVAILABLE_GPUS <<< "${AVAILABLE_GPUS:-5 6}"
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
  candle
  capsules
  cashew
  chewinggum
  fryum
  macaroni1
  macaroni2
  pipe_fryum
  pcb1
  pcb2
  pcb3
  pcb4
)

# ===== Basic eval settings =====
SHOTS="${SHOTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
VISA_SOURCE="${VISA_SOURCE:-per_category_csv}"
VISA_SPLIT_CSV="${VISA_SPLIT_CSV:-${DATASET_ROOT}/split_csv/1cls.csv}"
VISA_SPLIT_SET="${VISA_SPLIT_SET:-test}"
METRIC_THRESHOLD_POLICY="${METRIC_THRESHOLD_POLICY:-youden}"
METRIC_THRESHOLD="${METRIC_THRESHOLD:-0.0}"
SCORE_LABEL_AGGREGATION="${SCORE_LABEL_AGGREGATION:-max}"

# ===== 固定正常参考图，直接在脚本里传 =====
# 1-shot 示例:
FIXED_SUPPORT_IMAGES="${FIXED_SUPPORT_IMAGES:-pcb1=0717.JPG;pcb3=0032.JPG}"

# 如果以后跑 3-shot，可以这样写：
# FIXED_SUPPORT_IMAGES="${FIXED_SUPPORT_IMAGES:-pcb1=654.JPG,655.JPG,656.JPG;pcb3=101.JPG,102.JPG,103.JPG}"

# 如果以后跑全 VisA，就把 12 个类别都写上：

# FIXED_SUPPORT_IMAGES="${FIXED_SUPPORT_IMAGES:-candle=0120.JPG;capsules=0120.JPG;cashew=0120.JPG;chewinggum=0120.JPG;fryum=0120.JPG;macaroni1=0120.JPG;macaroni2=0120.JPG;pipe_fryum=0120.JPG;pcb1=0120.JPG;pcb2=0120.JPG;pcb3=0120.JPG;pcb4=0120.JPG}"
# ===== DeCo settings =====
DECO_ALPHA="${DECO_ALPHA:-0.60}"
DECO_TOP_P="${DECO_TOP_P:-0.90}"
DECO_TOP_K="${DECO_TOP_K:-32}"
START_LAYER="${START_LAYER:-30}"
END_LAYER="${END_LAYER:-35}"
LAMBDA_NORM="${LAMBDA_NORM:-0.50}"
ETA_HEDGE="${ETA_HEDGE:-0.10}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-1}"
GAMMA="${GAMMA:-8.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-1}"

# ===== Robust anchor smoothing settings =====
ANCHOR_SMOOTH="${ANCHOR_SMOOTH:-1}"
ANCHOR_REMOVE_OUTLIERS="${ANCHOR_REMOVE_OUTLIERS:-1}"
ANCHOR_HAMPEL_WINDOW="${ANCHOR_HAMPEL_WINDOW:-2}"
ANCHOR_HAMPEL_N_SIGMA="${ANCHOR_HAMPEL_N_SIGMA:-3.0}"
ANCHOR_SMOOTH_WINDOW="${ANCHOR_SMOOTH_WINDOW:-3}"
ANCHOR_SMOOTH_BETA="${ANCHOR_SMOOTH_BETA:-0.70}"

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

if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi

NUM_GPU=${#AVAILABLE_GPUS[@]}
if [[ ${NUM_GPU} -lt 1 ]]; then
  echo "[ERROR] No GPU found."
  exit 1
fi

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
USE_DECO="${USE_DECO:-1}"
DECO_FLAGS=()

if [[ "${USE_DECO}" == "1" ]]; then
  DECO_FLAGS+=(
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
  if [[ "${ANCHOR_SMOOTH}" == "1" ]]; then
    DECO_FLAGS+=(--anchor_smooth)
  fi
  if [[ "${ANCHOR_REMOVE_OUTLIERS}" == "1" ]]; then
    DECO_FLAGS+=(--anchor_remove_outliers)
  fi

  DECO_FLAGS+=(
    --anchor_hampel_window "${ANCHOR_HAMPEL_WINDOW}"
    --anchor_hampel_n_sigma "${ANCHOR_HAMPEL_N_SIGMA}"
    --anchor_smooth_window "${ANCHOR_SMOOTH_WINDOW}"
    --anchor_smooth_beta "${ANCHOR_SMOOTH_BETA}"
  )
fi

echo "===== Runtime Config ====="
echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "SHOTS=${SHOTS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "VISA_SOURCE=${VISA_SOURCE}"
echo "VISA_SPLIT_CSV=${VISA_SPLIT_CSV}"
echo "VISA_SPLIT_SET=${VISA_SPLIT_SET}"
echo "METRIC_THRESHOLD_POLICY=${METRIC_THRESHOLD_POLICY}"
echo "METRIC_THRESHOLD=${METRIC_THRESHOLD}"
echo "SCORE_LABEL_AGGREGATION=${SCORE_LABEL_AGGREGATION}"
echo "FIXED_SUPPORT_IMAGES=${FIXED_SUPPORT_IMAGES}"
echo "AVAILABLE_GPUS=${AVAILABLE_GPUS[*]}"
echo "NUM_GPU=${NUM_GPU}"
echo "ALL_CATEGORIES=${ALL_CATEGORIES[*]}"
echo "DECO_ALPHA=${DECO_ALPHA}, DECO_TOP_P=${DECO_TOP_P}, DECO_TOP_K=${DECO_TOP_K}, LAYERS=[${START_LAYER},${END_LAYER})"
echo "LAMBDA_NORM=${LAMBDA_NORM}, ETA_HEDGE=${ETA_HEDGE}, USE_LOGSUMEXP=${USE_LOGSUMEXP}, ENABLE_HEDGE=${ENABLE_HEDGE}, GAMMA=${GAMMA}"
echo "ANCHOR_SMOOTH=${ANCHOR_SMOOTH}, ANCHOR_REMOVE_OUTLIERS=${ANCHOR_REMOVE_OUTLIERS}, ANCHOR_HAMPEL_WINDOW=${ANCHOR_HAMPEL_WINDOW}, ANCHOR_HAMPEL_N_SIGMA=${ANCHOR_HAMPEL_N_SIGMA}, ANCHOR_SMOOTH_WINDOW=${ANCHOR_SMOOTH_WINDOW}, ANCHOR_SMOOTH_BETA=${ANCHOR_SMOOTH_BETA}"
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
      --visa_split_csv "${VISA_SPLIT_CSV}" \
      --visa_split_set "${VISA_SPLIT_SET}" \
      --shots "${SHOTS}" \
      --categories "${categories}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --metric_threshold_policy "${METRIC_THRESHOLD_POLICY}" \
      --metric_threshold "${METRIC_THRESHOLD}" \
      --score_label_aggregation "${SCORE_LABEL_AGGREGATION}" \
      --fixed_support_images "${FIXED_SUPPORT_IMAGES}" \
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
import csv

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
    # if not m.get("use_deco", False):
    #     raise SystemExit(f"[ERROR] DeCo is not enabled in {os.path.basename(fp)} (meta.use_deco={m.get('use_deco')})")
    all_results.extend(d.get("results", []))
    if meta is None:
        meta = m

threshold_policy = str((meta or {}).get("metric_threshold_policy") or "youden")
fixed_threshold = float((meta or {}).get("metric_threshold") or 0.0)

def binary_auroc(labels, scores):
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    sorted_pairs = sorted(enumerate(pairs), key=lambda x: x[1][1])
    ranks = [0.0] * len(sorted_pairs)
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][1][1] == sorted_pairs[i][1][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, (y, _) in zip(ranks, pairs) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def select_threshold(records, policy, fixed_threshold=0.0):
    if policy == "zero":
        return 0.0
    if policy == "fixed":
        return float(fixed_threshold)
    if not records:
        return float(fixed_threshold)
    scores = sorted({float(score) for _, score in records})
    candidates = [scores[0] - 1e-12] + scores + [scores[-1] + 1e-12]

    def stats_at(threshold):
        tp = sum(1 for y, s in records if y == 1 and s >= threshold)
        tn = sum(1 for y, s in records if y == 0 and s < threshold)
        fp = sum(1 for y, s in records if y == 0 and s >= threshold)
        fn = sum(1 for y, s in records if y == 1 and s < threshold)
        return tp, tn, fp, fn

    def div(num, den):
        return num / den if den > 0 else 0.0

    best_threshold = candidates[0]
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        tp, tn, fp, fn = stats_at(threshold)
        acc = div(tp + tn, len(records))
        recall_anomaly = div(tp, tp + fn)
        recall_normal = div(tn, tn + fp)
        precision_anomaly = div(tp, tp + fp)
        f1_anomaly = (
            2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
            if (precision_anomaly + recall_anomaly) > 0
            else 0.0
        )
        if policy == "best_f1":
            value = f1_anomaly
            tiebreak = acc
        elif policy == "best_acc":
            value = acc
            tiebreak = (recall_anomaly + recall_normal) / 2.0
        else:
            value = (recall_anomaly + recall_normal) / 2.0
            tiebreak = f1_anomaly
        if value > best_value or (value == best_value and tiebreak > best_tiebreak):
            best_value = value
            best_tiebreak = tiebreak
            best_threshold = threshold
    return float(best_threshold)


def compute_metrics(records):
    total = len(records)
    threshold = select_threshold(records, threshold_policy, fixed_threshold)
    tp = sum(1 for y, s in records if y == 1 and s >= threshold)
    tn = sum(1 for y, s in records if y == 0 and s < threshold)
    fp = sum(1 for y, s in records if y == 0 and s >= threshold)
    fn = sum(1 for y, s in records if y == 1 and s < threshold)

    def div(num, den):
        return num / den if den > 0 else None

    acc = div(tp + tn, total)
    precision_anomaly = div(tp, tp + fp)
    recall_anomaly = div(tp, tp + fn)
    f1_anomaly = (
        2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
        if precision_anomaly is not None and recall_anomaly is not None and (precision_anomaly + recall_anomaly) > 0
        else None
    )
    precision_normal = div(tn, tn + fn)
    recall_normal = div(tn, tn + fp)
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if precision_normal is not None and recall_normal is not None and (precision_normal + recall_normal) > 0
        else None
    )
    macro_precision_vals = [x for x in [precision_normal, precision_anomaly] if x is not None]
    macro_recall_vals = [x for x in [recall_normal, recall_anomaly] if x is not None]
    macro_f1_vals = [x for x in [f1_normal, f1_anomaly] if x is not None]
    return {
        "total": total,
        "acc": acc,
        "AUROC": binary_auroc([r[0] for r in records], [r[1] for r in records]),
        "auroc_score_source": "continuous_label_likelihood_score",
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": sum(macro_recall_vals) / len(macro_recall_vals) if macro_recall_vals else None,
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": sum(macro_precision_vals) / len(macro_precision_vals) if macro_precision_vals else None,
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": sum(macro_f1_vals) / len(macro_f1_vals) if macro_f1_vals else None,
        "recall_micro": acc,
        "precision_micro": acc,
        "F1_micro": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": tp + fn,
        "normal_total": tn + fp,
    }


records_by_shot = {}
records_by_shot_cat = {}
score_sources_by_shot = {}
score_sources_by_shot_cat = {}
for r in all_results:
    shot = str(r.get("shot"))
    cat = str(r.get("category"))
    gt = r.get("ground_truth", {}).get("label", "good")
    pred = r.get("prediction", {}).get("label", "good")
    y_true = 0 if gt == "good" else 1
    score = r.get("score", {}).get("anomaly_score")
    if score is None:
        score = 0.0 if pred == "good" else 1.0
    score_source = str(r.get("score", {}).get("score_source") or "continuous_label_likelihood_score")
    record = (y_true, float(score))
    records_by_shot.setdefault(shot, []).append(record)
    records_by_shot_cat.setdefault((shot, cat), []).append(record)
    score_sources_by_shot.setdefault(shot, set()).add(score_source)
    score_sources_by_shot_cat.setdefault((shot, cat), set()).add(score_source)

summary = {}
metric_rows = []
for shot in sorted(records_by_shot, key=lambda x: int(x)):
    overall = compute_metrics(records_by_shot[shot])
    overall_sources = score_sources_by_shot.get(shot, set())
    overall["auroc_score_source"] = next(iter(overall_sources)) if len(overall_sources) == 1 else "mixed"
    per_category = {
        cat: compute_metrics(records)
        for (shot_key, cat), records in sorted(records_by_shot_cat.items())
        if shot_key == shot
    }
    for cat, metrics in per_category.items():
        sources = score_sources_by_shot_cat.get((shot, cat), set())
        metrics["auroc_score_source"] = next(iter(sources)) if len(sources) == 1 else "mixed"
    summary[shot] = {
        "overall": overall,
        "per_category": per_category,
    }
    metric_rows.append({"shot": shot, "scope": "overall", "category": "ALL", **overall})
    for cat, metrics in per_category.items():
        metric_rows.append({"shot": shot, "scope": "category", "category": cat, **metrics})

save_path = os.path.join(out_dir, "visa_results_multi_gpu_deco.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "summary": summary, "results": all_results}, f, ensure_ascii=False, indent=2)

csv_path = os.path.join(out_dir, "visa_results_metrics_summary.csv")
fieldnames = [
    "shot", "scope", "category", "total", "acc", "AUROC",
    "threshold_policy", "threshold",
    "recall_normal", "recall_anomaly", "recall_all",
    "precision_normal", "precision_anomaly", "precision_all",
    "F1_normal", "F1_anomaly", "F1_all",
    "recall_micro", "precision_micro", "F1_micro",
    "tp", "tn", "fp", "fn", "pred_anomaly", "pred_normal",
    "anomaly_total", "normal_total", "auroc_score_source",
]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(metric_rows)

print(f"[OK] Saved merged result: {save_path}")
print(f"[OK] Saved merged metrics CSV: {csv_path}")
for shot in sorted(summary.keys(), key=lambda x: int(x)):
    v = summary[shot]
    overall = v["overall"]
    print(
        f"[shot={shot}] acc={overall['acc']:.6f}, "
        f"AUROC={overall['AUROC'] if overall['AUROC'] is not None else 'None'}, "
        f"recall_all={overall['recall_all'] if overall['recall_all'] is not None else 'None'}, "
        f"precision_all={overall['precision_all'] if overall['precision_all'] is not None else 'None'}, "
        f"F1_all={overall['F1_all'] if overall['F1_all'] is not None else 'None'}"
    )
PY

echo "[DONE] ${OUT_DIR}/visa_results_multi_gpu_deco.json"
