#!/usr/bin/env python3
"""
DeCo AUROC 超参优化：使用 Optuna（TPE 采样）替代网格搜索，调用 run_image_auroc_eval.sh 单次评估。
目标：最大化 macro_image_auroc；支持多 GPU 并行 trial。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Optuna-based hyperparameter optimization for AUROC (DeCo).")
    p.add_argument("--script_dir", type=str, default=None,
                   help="Deco-main 目录，内含 run_image_auroc_eval.sh。默认：本脚本所在目录。")
    p.add_argument("--data_base", type=str, required=True,
                   help="数据集根目录，如 .../evaluation/ODinW-13/data")
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Qwen3-VL 模型路径")
    p.add_argument("--dataset_name", type=str, default="mvtec",
                   help="用于搜参的数据集子目录名，如 mvtec")
    p.add_argument("--dataset_type", type=str, default="mvtec",
                   help="传给 run_image_auroc_eval.sh 的 DATASET_TYPE")
    p.add_argument("--output_dir", type=str, default=None,
                   help="优化结果与各 trial 的 JSON 输出目录。默认：script_dir/runs/auroc_optuna_<timestamp>")
    p.add_argument("--n_trials", type=int, default=30,
                   help="Optuna 最大 trial 数")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="并行 trial 数（建议等于 GPU 数）")
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                   help="逗号分隔的 GPU ID，用于轮询分配给各 trial")
    p.add_argument("--timeout", type=float, default=1200.0,
                   help="单次评估超时秒数，超时则该 trial 判为失败")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shot", type=int, default=0)
    p.add_argument("--max_pixels", type=str, default="1048576")
    p.add_argument("--min_pixels", type=str, default="409600")
    p.add_argument("--study_name", type=str, default="auroc_deco",
                   help="Optuna study 名称，用于存储/恢复")
    p.add_argument("--resume", action="store_true",
                   help="从 output_dir 中已有 study 恢复并继续优化")
    return p.parse_args()


def run_single_eval(
    script_dir: Path,
    eval_script: Path,
    dataset_root: str,
    model_path: str,
    dataset_type: str,
    output_dir: Path,
    basename: str,
    gpu_id: str,
    params: dict,
    timeout: float,
    extra_env: dict,
) -> float | None:
    """执行一次 run_image_auroc_eval.sh，返回 macro_image_auroc；失败返回 None。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE": gpu_id,
        "DATASET_ROOT": dataset_root,
        "DATASET_TYPE": dataset_type,
        "MODEL_NAME_OR_PATH": model_path,
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_BASENAME": basename,
        "MODE": "deco",
        "ALLOW_ATTN_FALLBACK": "1",
        **extra_env,
        **{k: str(v) for k, v in params.items()},
    })
    cmd = ["bash", str(eval_script)]
    try:
        subprocess.run(
            cmd,
            cwd=str(script_dir),
            env=env,
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None

    json_path = output_dir / f"{basename}_deco.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        auroc = data.get("summary", {}).get("macro_image_auroc")
        if auroc is None:
            return None
        return float(auroc)
    except Exception:
        return None


def get_objective(
    script_dir: Path,
    eval_script: Path,
    dataset_root: str,
    model_path: str,
    dataset_type: str,
    output_dir: Path,
    gpu_ids: list[str],
    timeout: float,
    extra_env: dict,
):
    """返回供 Optuna 调用的 objective 闭包；用 trial.number 分配 GPU 与输出 basename。"""
    trial_output_dir = output_dir / "trials"

    def objective(trial):
        import optuna
        # 搜索空间（与 run_image_auroc_eval.sh 的 DeCo 参数对应）
        alpha = trial.suggest_float("alpha", 0.25, 0.85)
        lambda_norm = trial.suggest_float("lambda_norm", 0.5, 1.5)
        threshold_top_p = trial.suggest_float("threshold_top_p", 0.8, 0.98)
        threshold_top_k = trial.suggest_int("threshold_top_k", 10, 30)
        eta_hedge = trial.suggest_float("eta_hedge", 0.0, 0.5)
        start_layer = trial.suggest_int("start_layer", 18, 22)
        end_layer = trial.suggest_int("end_layer", 27, 30)
        enable_hedge = trial.suggest_categorical("enable_hedge", [0, 1])
        use_logsumexp = trial.suggest_categorical("use_logsumexp", [0, 1])
        gamma = trial.suggest_float("gamma", 4.0, 12.0)

        params = {
            "ALPHA": alpha,
            "LAMBDA_NORM": lambda_norm,
            "THRESHOLD_TOP_P": threshold_top_p,
            "THRESHOLD_TOP_K": threshold_top_k,
            "ETA_HEDGE": eta_hedge,
            "START_LAYER": start_layer,
            "END_LAYER": end_layer,
            "ENABLE_HEDGE": enable_hedge,
            "USE_LOGSUMEXP": use_logsumexp,
            "GAMMA": gamma,
            "ANCHOR_STRATEGY": "defect_aware",
        }
        gpu_id = gpu_ids[trial.number % len(gpu_ids)]
        basename = f"trial_{trial.number}"

        auroc = run_single_eval(
            script_dir=script_dir,
            eval_script=eval_script,
            dataset_root=dataset_root,
            model_path=model_path,
            dataset_type=dataset_type,
            output_dir=trial_output_dir,
            basename=basename,
            gpu_id=gpu_id,
            params=params,
            timeout=timeout,
            extra_env=extra_env,
        )
        if auroc is None:
            raise optuna.TrialFailed("Eval failed or timeout")
        # Optuna 默认最小化，所以返回负 AUROC
        return -float(auroc)

    return objective


def main():
    args = parse_args()
    script_dir = Path(args.script_dir or os.path.dirname(os.path.abspath(__file__)))
    eval_script = script_dir / "run_image_auroc_eval.sh"
    if not eval_script.exists():
        print(f"[ERROR] Not found: {eval_script}", file=sys.stderr)
        sys.exit(1)

    dataset_root = os.path.join(args.data_base, args.dataset_name)
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = script_dir / "runs" / f"auroc_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) == 0:
        gpu_ids = ["0"]

    extra_env = {
        "SEED": str(args.seed),
        "SHOT": str(args.shot),
        "MAX_PIXELS": args.max_pixels,
        "MIN_PIXELS": args.min_pixels,
    }

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[ERROR] Need optuna: pip install optuna", file=sys.stderr)
        sys.exit(1)

    storage = f"sqlite:///{output_dir / 'optuna_study.db'}"
    load_if_exists = args.resume
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=TPESampler(seed=args.seed, n_startup_trials=min(5, args.n_trials)),
        direction="minimize",
    )
    objective = get_objective(
        script_dir=script_dir,
        eval_script=eval_script,
        dataset_root=dataset_root,
        model_path=args.model_name_or_path,
        dataset_type=args.dataset_type,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        timeout=args.timeout,
        extra_env=extra_env,
    )

    print(f"Output dir: {output_dir}")
    print(f"Dataset: {dataset_root}, n_trials={args.n_trials}, n_jobs={args.n_jobs}, GPUs={gpu_ids}")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    best = study.best_trial
    best_auroc = -best.value
    print("\n===== Best trial =====")
    print(f"macro_image_auroc: {best_auroc:.6f}")
    print("Params:", best.params)

    best_path = output_dir / "best_params.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"macro_image_auroc": best_auroc, "params": best.params}, f, indent=2)
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
