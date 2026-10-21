# #!/usr/bin/env bash
# set -euo pipefail

# # -----------------------------------------------------------------------------
# # 批量跑 Image-level AUROC（使用你的方法：DeCo + defect_aware 等），多卡并行
# # 参考 run_image_auroc_eval.sh / eval_mvtec_with_hyperparam.sh：顶部可改配置、GPU 分配、DeCo 参数
# #
# # 跑 AUROC 时与 run_image_auroc_eval.sh 一致：MODE=deco 即用 --use_deco + 下面 DeCo 参数，
# # 即你的改进 DeCo（defect_aware、lambda_norm、eta_hedge、anchor 词表等）。
# # -----------------------------------------------------------------------------

# # ===================== 需要修改 =====================
# # 与 eval_mvtec_with_hyperparam.sh 同风格；REPO_ROOT 由 run_image_auroc_eval.sh 按脚本位置自动检测
# DATA_BASE="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data"
# MODEL_NAME_OR_PATH="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl"

# # 四张卡并行：每个数据集绑到一张卡，轮询分配
# GPU_IDS=(0 1 2 3)

# # 输出目录（各数据集结果落在 RUN_DIR 下，如 RUN_DIR/AITEX_deco.json）
# RUN_DIR="${RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/image_auroc_batch_$(date +%Y%m%d_%H%M%S)}"

# # 与 run_image_auroc_eval.sh 一致的公共 / DeCo 参数（你的方法）
# # 降低 MAX_PIXELS 避免 Head-CT 等数据集 CUDA OOM（可 export MAX_PIXELS=6291456 恢复默认）
# MAX_PIXELS="${MAX_PIXELS:-2097152}"
# MIN_PIXELS="${MIN_PIXELS:-409600}"
# MODE="${MODE:-deco}"
# SHOT="${SHOT:-3}"
# SEED="${SEED:-42}"
# DATASET_TYPE_AUTO="${DATASET_TYPE_AUTO:-auto}"

# # DeCo（改进 DeCo）参数，与 run_image_auroc_eval.sh / eval_mvtec_with_hyperparam 对齐
# ALPHA="${ALPHA:-0.6}"
# THRESHOLD_TOP_P="${THRESHOLD_TOP_P:-0.9}"
# THRESHOLD_TOP_K="${THRESHOLD_TOP_K:-20}"
# START_LAYER="${START_LAYER:-20}"
# END_LAYER="${END_LAYER:-29}"
# ANCHOR_STRATEGY="${ANCHOR_STRATEGY:-defect_aware}"
# LAMBDA_NORM="${LAMBDA_NORM:-1.0}"
# ETA_HEDGE="${ETA_HEDGE:-0.0}"
# ENABLE_HEDGE="${ENABLE_HEDGE:-0}"
# USE_LOGSUMEXP="${USE_LOGSUMEXP:-0}"
# GAMMA="${GAMMA:-8.0}"
# ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal}"
# ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
# ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"
# # ===================================================

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RUN_SCRIPT="${SCRIPT_DIR}/run_image_auroc_eval.sh"

# # 数据集列表：(名称 dataset_type)，与 run_image_auroc_eval 的 DATASET_TYPE 一致
# DATASET_LIST=(
#   # "AITEX:flat_binary"
#   "Br35HDet:br35hdet"
#   # "BrainMRI:brainmri"
#   "Head-CT-hemorrhage-detection-master:headct"
#   # "MPDD:mvtec"
#   # "VisA:visa"
#   # "mvtec:mvtec"
# )

# mkdir -p "$RUN_DIR"
# export OUTPUT_DIR="$RUN_DIR"

# if [[ ! -f "${RUN_SCRIPT}" ]]; then
#   echo "[ERROR] Not found: ${RUN_SCRIPT}"
#   exit 1
# fi

# n_gpu=${#GPU_IDS[@]}
# i=0
# pids=()

# for entry in "${DATASET_LIST[@]}"; do
#   name="${entry%%:*}"
#   dtype="${entry##*:}"
#   dir="${DATA_BASE}/${name}"
#   if [[ ! -d "${dir}" ]]; then
#     echo "[SKIP] ${name}: 目录不存在 ${dir}"
#     continue
#   fi
#   gpu="${GPU_IDS[$((i % n_gpu))]}"
#   i=$((i + 1))
#   log_file="${RUN_DIR}/${name}.log"
#   (
#     export CUDA_DEVICE="$gpu"
#     export DATASET_ROOT="${dir}"
#     export DATASET_TYPE="${dtype}"
#     export OUTPUT_BASENAME="${name}"
#     export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
#     export MAX_PIXELS="${MAX_PIXELS}"
#     export MIN_PIXELS="${MIN_PIXELS}"
#     export MODE="${MODE}"
#     export SHOT="${SHOT}"
#     export SEED="${SEED}"
#     export ALPHA="${ALPHA}"
#     export THRESHOLD_TOP_P="${THRESHOLD_TOP_P}"
#     export THRESHOLD_TOP_K="${THRESHOLD_TOP_K}"
#     export START_LAYER="${START_LAYER}"
#     export END_LAYER="${END_LAYER}"
#     export ANCHOR_STRATEGY="${ANCHOR_STRATEGY}"
#     export LAMBDA_NORM="${LAMBDA_NORM}"
#     export ETA_HEDGE="${ETA_HEDGE}"
#     export ENABLE_HEDGE="${ENABLE_HEDGE}"
#     export USE_LOGSUMEXP="${USE_LOGSUMEXP}"
#     export GAMMA="${GAMMA}"
#     export ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS}"
#     export ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS}"
#     export ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS}"
#     bash "${RUN_SCRIPT}" > "${log_file}" 2>&1
#   ) &
#   pids+=($!)
#   echo "[GPU${gpu}] 启动 ${name} (dataset_type=${dtype}) -> ${log_file}"
#   while [[ $(jobs -pr | wc -l | tr -d ' ') -ge "$n_gpu" ]]; do
#     wait -n || true
#   done
# done

# for pid in "${pids[@]}"; do
#   wait "$pid" || true
# done

# echo ""
# echo "========== Batch AUROC 全部完成 =========="
# echo "结果目录: ${RUN_DIR}"
# echo "各数据集: ${RUN_DIR}/<名称>_deco.json，日志: ${RUN_DIR}/<名称>.log"

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 批量跑 Image-level AUROC（使用你的方法：DeCo + defect_aware 等），多卡并行
# 参考 run_image_auroc_eval.sh / eval_mvtec_with_hyperparam.sh：顶部可改配置、GPU 分配、DeCo 参数
#
# 跑 AUROC 时与 run_image_auroc_eval.sh 一致：MODE=deco 即用 --use_deco + 下面 DeCo 参数，
# 即你的改进 DeCo（defect_aware、lambda_norm、eta_hedge、anchor 词表等）。
# -----------------------------------------------------------------------------

# ===================== 需要修改 =====================
# 与 eval_mvtec_with_hyperparam.sh 同风格；REPO_ROOT 由 run_image_auroc_eval.sh 按脚本位置自动检测
DATA_BASE="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data"
MODEL_NAME_OR_PATH="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl"

# 四张卡并行：每个数据集绑到一张卡，轮询分配
GPU_IDS=(0 1 5 6)

# 输出目录（各数据集结果落在 RUN_DIR 下，如 RUN_DIR/AITEX_deco.json）
RUN_DIR="${RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs/image_auroc_batch_$(date +%Y%m%d_%H%M%S)}"

# 与 run_image_auroc_eval.sh 一致的公共 / DeCo 参数（你的方法）
# 降低 MAX_PIXELS 避免 Head-CT 等数据集 CUDA OOM（可 export MAX_PIXELS=6291456 恢复默认）
MAX_PIXELS="${MAX_PIXELS:-1048576}"
MIN_PIXELS="${MIN_PIXELS:-409600}"
MODE="${MODE:-deco}"
SHOT="${SHOT:-0}"
SEED="${SEED:-42}"
DATASET_TYPE_AUTO="${DATASET_TYPE_AUTO:-auto}"

# DeCo 参数
# 原始 DeCo（千问3+DeCo，不带你的改进）：设 DECO_VANILLA=1 再跑，则 anchor_strategy=default、不传自定义 anchor 词
# 改进 DeCo（默认）：DECO_VANILLA=0，使用 defect_aware + 自定义 anchor 词等
DECO_VANILLA="${DECO_VANILLA:-0}"
if [[ "${DECO_VANILLA}" == "1" ]]; then
  ANCHOR_STRATEGY="default"
  ANCHOR_DEFECT_WORDS=""
  ANCHOR_NORM_WORDS=""
  ANCHOR_HEDGE_WORDS=""
else
  ANCHOR_STRATEGY="${ANCHOR_STRATEGY:-defect_aware}"
  ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS:-defect,anomaly,abnormal}"
  ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
  ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"
fi
ALPHA="${ALPHA:-0.6}"
THRESHOLD_TOP_P="${THRESHOLD_TOP_P:-0.9}"
THRESHOLD_TOP_K="${THRESHOLD_TOP_K:-20}"
START_LAYER="${START_LAYER:-20}"
END_LAYER="${END_LAYER:-29}"
LAMBDA_NORM="${LAMBDA_NORM:-1.0}"
ETA_HEDGE="${ETA_HEDGE:-0.0}"
ENABLE_HEDGE="${ENABLE_HEDGE:-0}"
USE_LOGSUMEXP="${USE_LOGSUMEXP:-0}"
GAMMA="${GAMMA:-8.0}"
# ===================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_image_auroc_eval.sh"

# 数据集列表：(名称 dataset_type)，与 run_image_auroc_eval 的 DATASET_TYPE 一致
DATASET_LIST=(
  # "AITEX:flat_binary"
  # "Br35HDet:br35hdet"
  # "BrainMRI:brainmri"
  # "Head-CT-hemorrhage-detection-master:headct"
  # "MPDD:mvtec"
  # "VisA:visa"
  "mvtec:mvtec"
)

mkdir -p "$RUN_DIR"
export OUTPUT_DIR="$RUN_DIR"
[[ "${DECO_VANILLA}" == "1" ]] && echo "[INFO] 使用原始 DeCo（anchor_strategy=default，无自定义 anchor 词）"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] Not found: ${RUN_SCRIPT}"
  exit 1
fi

n_gpu=${#GPU_IDS[@]}
i=0
pids=()

for entry in "${DATASET_LIST[@]}"; do
  name="${entry%%:*}"
  dtype="${entry##*:}"
  dir="${DATA_BASE}/${name}"
  if [[ ! -d "${dir}" ]]; then
    echo "[SKIP] ${name}: 目录不存在 ${dir}"
    continue
  fi
  gpu="${GPU_IDS[$((i % n_gpu))]}"
  i=$((i + 1))
  log_file="${RUN_DIR}/${name}.log"
  (
    export CUDA_DEVICE="$gpu"
    export DATASET_ROOT="${dir}"
    export DATASET_TYPE="${dtype}"
    export OUTPUT_BASENAME="${name}"
    export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
    export MAX_PIXELS="${MAX_PIXELS}"
    export MIN_PIXELS="${MIN_PIXELS}"
    export MODE="${MODE}"
    export SHOT="${SHOT}"
    export SEED="${SEED}"
    export ALPHA="${ALPHA}"
    export THRESHOLD_TOP_P="${THRESHOLD_TOP_P}"
    export THRESHOLD_TOP_K="${THRESHOLD_TOP_K}"
    export START_LAYER="${START_LAYER}"
    export END_LAYER="${END_LAYER}"
    export ANCHOR_STRATEGY="${ANCHOR_STRATEGY}"
    export LAMBDA_NORM="${LAMBDA_NORM}"
    export ETA_HEDGE="${ETA_HEDGE}"
    export ENABLE_HEDGE="${ENABLE_HEDGE}"
    export USE_LOGSUMEXP="${USE_LOGSUMEXP}"
    export GAMMA="${GAMMA}"
    export ANCHOR_DEFECT_WORDS="${ANCHOR_DEFECT_WORDS}"
    export ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS}"
    export ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS}"
    bash "${RUN_SCRIPT}" > "${log_file}" 2>&1
  ) &
  pids+=($!)
  echo "[GPU${gpu}] 启动 ${name} (dataset_type=${dtype}) -> ${log_file}"
  while [[ $(jobs -pr | wc -l | tr -d ' ') -ge "$n_gpu" ]]; do
    wait -n || true
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo ""
echo "========== Batch AUROC 全部完成 =========="
echo "结果目录: ${RUN_DIR}"
echo "各数据集: ${RUN_DIR}/<名称>_deco.json，日志: ${RUN_DIR}/<名称>.log"
