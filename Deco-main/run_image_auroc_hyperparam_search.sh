#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# AUROC 超參數搜尋：調用 run_image_auroc_eval.sh，用 4 張 GPU 並行跑多組 DeCo 參數，
# 跑完後解析各輪 macro_image_auroc，選出最優超參數並輸出摘要。
#
# 用法：
#   bash run_image_auroc_hyperparam_search.sh              # 網格搜尋（默認）
#   SEARCH_MODE=optuna bash run_image_auroc_hyperparam_search.sh  # Optuna TPE 優化
#
# 可通過環境變量覆蓋（在腳本頂部「可修改配置」裡也有默認值）：
#   DATA_BASE, MODEL_NAME_OR_PATH, DATASET_NAME, DATASET_TYPE,
#   GPU_IDS, RUN_DIR,
#   ALPHAS, LAMBDA_NORMS, ... （網格）
#   SEARCH_MODE=grid|optuna, OPTUNA_N_TRIALS, OPTUNA_N_JOBS, OPTUNA_TIMEOUT （優化模式）
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_image_auroc_eval.sh"
OPTIMIZER_SCRIPT="${SCRIPT_DIR}/optimize_auroc_hyperparams.py"
SEARCH_MODE="${SEARCH_MODE:-grid}"

# ===================== 可修改配置 =====================
# 搜尋用的數據集（單一數據集，保證每輪條件一致）
# 這裡直接寫入你在訓練機上的實際路徑，避免每次手動輸入
DATA_BASE="${DATA_BASE:-/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl}"
DATASET_NAME="${DATASET_NAME:-mvtec}"
DATASET_TYPE="${DATASET_TYPE:-mvtec}"

# 4 張 GPU 並行
GPU_IDS=(${GPU_IDS:-0 1 2 3})

# 本輪搜尋結果目錄（每組參數一個 JSON，便於比對）
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/auroc_hp_search_$(date +%Y%m%d_%H%M%S)}"

# 與 run_image_auroc_eval.sh 一致的公共參數（不參與網格搜尋）
SHOT="${SHOT:-0}"
SEED="${SEED:-42}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
MIN_PIXELS="${MIN_PIXELS:-409600}"
export ALLOW_ATTN_FALLBACK="${ALLOW_ATTN_FALLBACK:-1}"

# 網格搜尋的超參數（數組，會做笛卡爾積，見下方 build_configs）
# 可通過環境變量覆蓋，例如:
#   ALPHAS="0.4 0.6" LAMBDA_NORMS="1.0" ENABLE_HEDGES_ARR="0 1" bash run_image_auroc_hyperparam_search.sh
ALPHAS=(${ALPHAS:-0.4 0.5 0.6 0.7})
LAMBDA_NORMS=(${LAMBDA_NORMS:-0.8 1.0 1.2})
THRESHOLD_TOP_PS=(${THRESHOLD_TOP_PS:-0.85 0.9})
THRESHOLD_TOP_KS=(${THRESHOLD_TOP_KS:-20})
ETA_HEDGES=(${ETA_HEDGES:-0.0})
ENABLE_HEDGES_ARR=(${ENABLE_HEDGES_ARR:-0})
USE_LOGSUMEXPS_ARR=(${USE_LOGSUMEXPS_ARR:-0})
GAMMAS=(${GAMMAS:-8.0})
# Optuna 優化模式專用（僅在 SEARCH_MODE=optuna 時生效）
OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
OPTUNA_N_JOBS="${OPTUNA_N_JOBS:-4}"
OPTUNA_TIMEOUT="${OPTUNA_TIMEOUT:-1200}"
OPTUNA_OUTPUT_DIR="${OPTUNA_OUTPUT_DIR:-}"
# =====================================================

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH is required (set it or edit the script)."
  exit 1
fi

DATASET_ROOT="${DATA_BASE}/${DATASET_NAME}"
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] Dataset directory not found: ${DATASET_ROOT}"
  exit 1
fi

# ---------- Optuna 優化模式：調用 Python 優化腳本後退出 ----------
if [[ "${SEARCH_MODE}" == "optuna" ]]; then
  if [[ ! -f "${OPTIMIZER_SCRIPT}" ]]; then
    echo "[ERROR] Optuna mode requires: ${OPTIMIZER_SCRIPT}"
    exit 1
  fi
  # 避免 GPU_IDS=(2) 传入时变成字符串 "(2)"，去掉括号并转为逗号分隔
  GPU_IDS_STR="${GPU_IDS[*]}"
  GPU_IDS_STR="${GPU_IDS_STR//[()]/}"
  GPU_IDS_STR="${GPU_IDS_STR// /,}"
  OPTUNA_ARGS=(
    --script_dir "${SCRIPT_DIR}"
    --data_base "${DATA_BASE}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --dataset_name "${DATASET_NAME}"
    --dataset_type "${DATASET_TYPE}"
    --n_trials "${OPTUNA_N_TRIALS}"
    --n_jobs "${OPTUNA_N_JOBS}"
    --gpu_ids "${GPU_IDS_STR}"
    --timeout "${OPTUNA_TIMEOUT}"
    --seed "${SEED}"
    --shot "${SHOT}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
  )
  [[ -n "${OPTUNA_OUTPUT_DIR}" ]] && OPTUNA_ARGS+=(--output_dir "${OPTUNA_OUTPUT_DIR}")
  echo "[INFO] Running Optuna optimizer (n_trials=${OPTUNA_N_TRIALS}, n_jobs=${OPTUNA_N_JOBS}, gpus=${GPU_IDS_STR})"
  exec python3 "${OPTIMIZER_SCRIPT}" "${OPTUNA_ARGS[@]}"
fi

# ---------- 以下為網格搜尋模式 ----------
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] Not found: ${RUN_SCRIPT}"
  exit 1
fi

mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
export DATA_BASE MODEL_NAME_OR_PATH DATASET_NAME DATASET_TYPE RUN_DIR
export SCRIPT_DIR RUN_SCRIPT

# 構建配置列表：每行為
#   run_id ALPHA LAMBDA_NORM THRESHOLD_TOP_P ETA_HEDGE ENABLE_HEDGE USE_LOGSUMEXP GAMMA THRESHOLD_TOP_K
# START_LAYER / END_LAYER 目前固定為 20 / 29，如需搜尋可以仿照下面擴展一層循環
build_configs() {
  local idx=0
  local a ln tp tk eh en ul g
  for a in "${ALPHAS[@]}"; do
    for ln in "${LAMBDA_NORMS[@]}"; do
      for tp in "${THRESHOLD_TOP_PS[@]}"; do
        for tk in "${THRESHOLD_TOP_KS[@]}"; do
          for eh in "${ETA_HEDGES[@]}"; do
            for en in "${ENABLE_HEDGES_ARR[@]}"; do
              for ul in "${USE_LOGSUMEXPS_ARR[@]}"; do
                for g in "${GAMMAS[@]}"; do
                  idx=$((idx + 1))
                  printf "run_%03d %s %s %s %s %s %s %s %s\n" \
                    "$idx" "$a" "$ln" "$tp" "$eh" "$en" "$ul" "$g" "$tk"
                done
              done
            done
          done
        done
      done
    done
  done
}

CONFIG_LIST=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && CONFIG_LIST+=("$line")
done < <(build_configs)

n_configs=${#CONFIG_LIST[@]}
n_gpu=${#GPU_IDS[@]}
echo "[INFO] Hyperparameter search: ${n_configs} configs, ${n_gpu} GPUs (parallel)."
echo "[INFO] Dataset: ${DATASET_ROOT} (${DATASET_TYPE})"
echo "[INFO] Run dir: ${RUN_DIR}"
echo ""

i=0
pids=()
for line in "${CONFIG_LIST[@]}"; do
  read -r run_id alpha lambda_norm th_top_p eta_hedge enable_hedge use_logsumexp gamma_val th_top_k <<< "$line"
  gpu="${GPU_IDS[$((i % n_gpu))]}"
  i=$((i + 1))
  log_file="${RUN_DIR}/${run_id}.log"
  params_file="${RUN_DIR}/${run_id}.params"

  # 寫入本輪參數，便於最後報告最優配置
  {
    echo "run_id=${run_id}"
    echo "ALPHA=${alpha}"
    echo "LAMBDA_NORM=${lambda_norm}"
    echo "THRESHOLD_TOP_P=${th_top_p}"
    echo "ETA_HEDGE=${eta_hedge}"
    echo "ENABLE_HEDGE=${enable_hedge}"
    echo "USE_LOGSUMEXP=${use_logsumexp}"
    echo "GAMMA=${gamma_val}"
    echo "THRESHOLD_TOP_K=${th_top_k}"
    echo "START_LAYER=20"
    echo "END_LAYER=29"
  } > "${params_file}"

  (
    export CUDA_DEVICE="${gpu}"
    export DATASET_ROOT="${DATASET_ROOT}"
    export DATASET_TYPE="${DATASET_TYPE}"
    export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
    export OUTPUT_DIR="${RUN_DIR}"
    export OUTPUT_BASENAME="${run_id}"
    export MODE="deco"
    export SHOT="${SHOT}"
    export SEED="${SEED}"
    export MAX_PIXELS="${MAX_PIXELS}"
    export MIN_PIXELS="${MIN_PIXELS}"
    export ALPHA="${alpha}"
    export LAMBDA_NORM="${lambda_norm}"
    export THRESHOLD_TOP_P="${th_top_p}"
    export THRESHOLD_TOP_K="${th_top_k}"
    export ETA_HEDGE="${eta_hedge}"
    export START_LAYER="${START_LAYER:-20}"
    export END_LAYER="${END_LAYER:-29}"
    export ANCHOR_STRATEGY="${ANCHOR_STRATEGY:-defect_aware}"
    export ENABLE_HEDGE="${enable_hedge}"
    export USE_LOGSUMEXP="${use_logsumexp}"
    export GAMMA="${gamma_val}"
    bash "${RUN_SCRIPT}" > "${log_file}" 2>&1
  ) &
  pids+=($!)
  echo "[GPU${gpu}] Started ${run_id} (alpha=${alpha} ln=${lambda_norm} tp=${th_top_p} eta=${eta_hedge}) -> ${log_file}"

  while [[ $(jobs -pr 2>/dev/null | wc -l | tr -d ' ') -ge "${n_gpu}" ]]; do
    wait -n 2>/dev/null || true
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo ""
echo "===== All runs finished. Parsing results. ====="

# 用 Python 解析所有 *_deco.json、按 macro_image_auroc 排序並輸出最優
summary_file="${RUN_DIR}/search_summary.txt"
best_info_file="${RUN_DIR}/.best_run"

(
python3 << PYEOF
import json
import os
run_dir = """${RUN_DIR}"""
rows = []
for f in sorted(os.listdir(run_dir)):
    if not f.endswith("_deco.json"):
        continue
    run_id = f.replace("_deco.json", "")
    json_path = os.path.join(run_dir, f)
    params_path = os.path.join(run_dir, f"{run_id}.params")
    auroc = None
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            auroc = json.load(fp).get("summary", {}).get("macro_image_auroc")
    except Exception:
        pass
    params_str = ""
    if os.path.isfile(params_path):
        with open(params_path, "r", encoding="utf-8") as fp:
            params_str = " ".join(l.strip() for l in fp if "=" in l)
    rows.append((run_id, auroc, params_str))

# 按 AUROC 降序（有效數值優先，null 放最後）
def key_fn(x):
    r, a, p = x
    if a is None or not isinstance(a, (int, float)):
        return (-1.0, r)
    return (-float(a), r)
rows.sort(key=key_fn)

for run_id, auroc, params_str in rows:
    a_str = f"{auroc:.6f}" if isinstance(auroc, (int, float)) else "(null)"
    print(f"{run_id} macro_image_auroc={a_str} {params_str}")

best_run_id = ""
best_auroc = ""
for run_id, auroc, _ in rows:
    if auroc is not None and isinstance(auroc, (int, float)):
        best_run_id = run_id
        best_auroc = f"{auroc:.6f}"
        break
with open(os.path.join(run_dir, ".best_run"), "w") as fp:
    fp.write(f"BEST_RUN_ID={best_run_id}\nBEST_AUROC={best_auroc}\n")
PYEOF
) | tee "${summary_file}"

if [[ -f "${best_info_file}" ]]; then
  source "${best_info_file}" 2>/dev/null || true
fi

echo ""
if [[ -n "${BEST_RUN_ID:-}" && -n "${BEST_AUROC:-}" ]]; then
  echo "===== Best config (macro_image_auroc=${BEST_AUROC}) ====="
  cat "${RUN_DIR}/${BEST_RUN_ID}.params"
  echo ""
  echo "Best result JSON: ${RUN_DIR}/${BEST_RUN_ID}_deco.json"
else
  echo "[WARN] No valid macro_image_auroc found; check logs in ${RUN_DIR}"
fi
echo "Done. Full summary: ${summary_file}"
