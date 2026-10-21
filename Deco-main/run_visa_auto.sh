#!/usr/bin/env bash
set -euo pipefail

# =========================
# ===== PATH CONFIG =======
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/transformers-4.57.6}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/../evaluation/ODinW-13/data/VisA}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/qwen3_4b_vl}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/visa_eval_$(date +%Y%m%d_%H%M%S)}"

# =========================
# ===== EXP CONFIG ========
# =========================
SHOTS="${SHOTS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
USE_DECO="${USE_DECO:-1}"

# DeCo 超参数
DECO_ALPHA="${DECO_ALPHA:-0.45}"
DECO_TOP_P="${DECO_TOP_P:-0.88}"
DECO_TOP_K="${DECO_TOP_K:-16}"
START_LAYER="${START_LAYER:-21}"
END_LAYER="${END_LAYER:-30}"

# =========================
# ===== DATASET ===========
# =========================
VISA_CLASSES=(
  candle capsules cashew chewinggum fryum macaroni1
  macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum
)

# =========================
# ===== RUNTIME ===========
# =========================
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cd "${ROOT_DIR}"

# ===== 手动指定物理GPU（核心）=====
PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1,2}"
IFS=',' read -ra GPU_LIST <<< "$PHYSICAL_GPUS"
MAX_PARALLEL=${#GPU_LIST[@]}

echo "Using physical GPUs: ${PHYSICAL_GPUS}"
echo "Max parallel jobs: ${MAX_PARALLEL}"

# =========================
# ===== RUN TASK ==========
# =========================
run_one_task() {
  local gpu_id="$1"
  local category="$2"
  local out_json="$3"
  local log_file="$4"

  echo "[START] ${category} on PHYSICAL GPU ${gpu_id}"

  local py_args=(
    --dataset_root "${DATASET_ROOT}"
    --model_name_or_path "${MODEL_PATH}"
    --output_json "${out_json}"
    --shots "${SHOTS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --allow_attn_fallback
    --categories "${category}"
  )

  if [[ "${USE_DECO}" == "1" ]]; then
    py_args+=(
      --use_deco
      --anchor_strategy defect_aware
      --alpha "${DECO_ALPHA}"
      --threshold_top_p "${DECO_TOP_P}"
      --threshold_top_k "${DECO_TOP_K}"
      --start_layer "${START_LAYER}"
      --end_layer "${END_LAYER}"
    )
  fi

  # ✅ 关键：每个任务单独绑定物理GPU
  PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python examples/deco/eval_visa_qwen3_vl_thinking.py \
    "${py_args[@]}" \
    > "${log_file}" 2>&1
}

# =========================
# ===== INIT TASK =========
# =========================
declare -a PENDING_TASKS=()
declare -A RUNNING_PIDS=()
declare -A RUNNING_GPU=()

for cls in "${VISA_CLASSES[@]}"; do
  if [[ -f "${OUT_DIR}/${cls}.json" ]]; then
    echo "[SKIP] ${cls} already done"
  else
    PENDING_TASKS+=("${cls}")
  fi
done

echo "Pending tasks: ${#PENDING_TASKS[@]}"

# =========================
# ===== SCHEDULER =========
# =========================
while [[ ${#PENDING_TASKS[@]} -gt 0 || ${#RUNNING_PIDS[@]} -gt 0 ]]; do

  # ---- 清理完成任务 ----
  for cls in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$cls]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[DONE] ${cls}"
      unset RUNNING_PIDS["$cls"]
      unset RUNNING_GPU["$cls"]
    fi
  done

  # ---- 并发控制 ----
  if [[ ${#RUNNING_PIDS[@]} -ge ${MAX_PARALLEL} ]]; then
    sleep 5
    continue
  fi

  # ---- 找空闲物理GPU ----
  USED_GPUS=("${RUNNING_GPU[@]:-}")
  gpu=""

  for i in "${GPU_LIST[@]}"; do
    skip=0
    for u in "${USED_GPUS[@]}"; do
      if [[ "$u" == "$i" ]]; then
        skip=1
        break
      fi
    done
    if [[ $skip -eq 0 ]]; then
      gpu=$i
      break
    fi
  done

  # ---- 分配任务 ----
  if [[ -n "${gpu}" && ${#PENDING_TASKS[@]} -gt 0 ]]; then
    cls="${PENDING_TASKS[0]}"
    PENDING_TASKS=("${PENDING_TASKS[@]:1}")

    out_json="${OUT_DIR}/${cls}.json"
    log_file="${LOG_DIR}/${cls}.log"

    run_one_task "${gpu}" "${cls}" "${out_json}" "${log_file}" &
    pid=$!

    RUNNING_PIDS["$cls"]=$pid
    RUNNING_GPU["$cls"]=$gpu

    echo "[LAUNCH] ${cls} → PHYSICAL GPU ${gpu} (PID ${pid})"
  fi

  sleep 5
done

# =========================
# ===== MERGE SUMMARY =====
# =========================
echo "Merging results..."

python - "${OUT_DIR}" << 'PY'
import json, os, sys

out_dir = sys.argv[1]
files = [f for f in os.listdir(out_dir) if f.endswith(".json") and f != "visa_summary.json"]

all_results = []

for f in files:
    with open(os.path.join(out_dir, f), "r", encoding="utf-8") as fp:
        d = json.load(fp)
    all_results.extend(d["results"])

by_shot = {}

for r in all_results:
    shot = int(r["shot"])
    gt = r["ground_truth"]["label"]
    pred = r["prediction"]["label"]

    if shot not in by_shot:
        by_shot[shot] = {"tp":0,"fp":0,"tn":0,"fn":0,"total":0}

    s = by_shot[shot]
    s["total"] += 1

    gt_a = gt != "good"
    pred_a = pred != "good"

    if gt_a and pred_a: s["tp"] += 1
    elif not gt_a and pred_a: s["fp"] += 1
    elif not gt_a and not pred_a: s["tn"] += 1
    else: s["fn"] += 1

summary = {}

for shot, s in by_shot.items():
    tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
    total = s["total"]

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None

    summary[str(shot)] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

save_path = os.path.join(out_dir, "visa_summary.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Saved summary:", save_path)
PY

echo "All done!"