#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean reproducible entry point for the ADAR/Deco main method.

This file is intentionally an orchestration layer.  The model/scoring
implementation remains in ``examples/deco/0727_UP.py``; this runner fixes the
formal method configuration, launches all requested dataset-shot combinations,
and writes a single main-method summary table.

Formal setting:

  prompt-mode: category_defects_guarded
  score source: E_raw_hdec
  threshold policy: category-wise Youden
  prediction: score >= threshold => ANOMALOUS

Reason audit, JUDO, T3, oracle source selection, source fusion and other
diagnostic branches are not part of this runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DATASETS = ["visa", "mvtec", "aitex", "btad", "elpv", "mpdd", "brainmri", "headct", "br35h"]
DATASET_DIRS = {
    "visa": "VisA",
    "mvtec": "MVTec",
    "aitex": "AITEX",
    "btad": "BTAD",
    "elpv": "ELPV",
    "mpdd": "MPDD",
    "brainmri": "BrainMRI",
    "headct": "HeadCT",
    "br35h": "Br35H",
}
DATASET_ROOT_FLAGS = {
    "visa": "--visa-root",
    "mvtec": "--mvtec-root",
    "aitex": "--aitex-root",
    "btad": "--btad-root",
    "elpv": "--elpv-root",
    "mpdd": "--mpdd-root",
    "brainmri": "--brainmri-root",
    "headct": "--headct-root",
    "br35h": "--br35h-root",
}

MAIN_EXPERIMENT = "E_raw_hdec"
MAIN_THRESHOLD_POLICY = "youden"
MAIN_METHOD_LABEL = "category_defects_guarded + E_raw_hdec + youden"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def command_to_string(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def dataset_args(args: argparse.Namespace, dataset: str) -> List[str]:
    data_root = Path(args.data_root)
    dataset_root = data_root / DATASET_DIRS[dataset]
    out = [DATASET_ROOT_FLAGS[dataset], str(dataset_root)]
    if dataset == "visa":
        split_csv = Path(args.visa_split_csv) if args.visa_split_csv else dataset_root / "split_csv" / "1cls.csv"
        if split_csv.exists():
            out.extend(["--visa-split-csv", str(split_csv)])
        out.extend(["--visa-split-set", args.visa_split_set])
        out.extend(["--support-split-set", args.visa_support_split_set])
    if dataset == "elpv":
        out.extend(["--elpv-label-policy", args.elpv_label_policy])
    return out


def combo_dir(args: argparse.Namespace, dataset: str, shot: int) -> Path:
    return Path(args.out_dir) / f"{shot}shot" / dataset


def build_command(args: argparse.Namespace, dataset: str, shot: int) -> Tuple[List[str], Path]:
    output_dir = combo_dir(args, dataset, shot)
    script_path = Path(args.transformers_root) / "examples" / "deco" / "0727_UP.py"
    cmd = [
        args.python_bin,
        str(script_path),
        "--dataset",
        dataset,
        *dataset_args(args, dataset),
        "--model-path",
        args.model_path,
        "--output-dir",
        str(output_dir),
        "--run-all",
        "--attn-implementation",
        args.attn_implementation,
        "--allow-attn-fallback",
        "--prompt-mode",
        "category_defects_guarded",
        "--trend-layers",
        args.trend_layers,
        "--deco-layers",
        args.deco_layers,
        "--deco-alpha",
        str(args.deco_alpha),
        "--score-deco-mode",
        args.score_deco_mode,
        "--main-method-only",
        "--main-method-source",
        "hdec",
        "--metric-threshold-policies",
        MAIN_THRESHOLD_POLICY,
        "--shots",
        str(shot),
        "--support-source",
        args.support_source,
        "--seed",
        str(args.seed),
        "--log-interval",
        str(args.log_interval),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.stream_jsonl:
        cmd.append("--stream-jsonl")
    if args.strict_fewshot_support:
        cmd.append("--strict-fewshot-support")
    return cmd, output_dir


def run_command(args: argparse.Namespace, cmd: Sequence[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path(args.transformers_root) / 'src'}:{Path(args.transformers_root) / 'examples' / 'deco'}:{env.get('PYTHONPATH', '')}"
    env["HF_HUB_OFFLINE"] = "1" if args.local_files_only else env.get("HF_HUB_OFFLINE", "0")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("CC", "/usr/bin/gcc")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/vina04-triton-cache")
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[COMMAND] {command_to_string(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            list(cmd),
            cwd=args.transformers_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def find_main_macro_row(summary_path: Path) -> Optional[Dict[str, str]]:
    for row in read_csv_rows(summary_path):
        if row.get("category") not in {"", "ALL"}:
            continue
        if row.get("experiment") != MAIN_EXPERIMENT:
            continue
        if row.get("threshold_policy") != MAIN_THRESHOLD_POLICY:
            continue
        return row
    return None


def collect_one(args: argparse.Namespace, dataset: str, shot: int) -> Optional[Dict[str, Any]]:
    out = combo_dir(args, dataset, shot)
    summary_path = out / "up_macro_summary.csv"
    row = find_main_macro_row(summary_path)
    if row is None:
        return None
    split_meta = read_json(out / "fewshot_split.json")
    exit_code = ""
    exit_path = out / "exit_code"
    if exit_path.exists():
        exit_code = exit_path.read_text(encoding="utf-8").strip()
    return {
        "shot": shot,
        "dataset": dataset,
        "method": MAIN_METHOD_LABEL,
        "experiment": MAIN_EXPERIMENT,
        "threshold_policy": MAIN_THRESHOLD_POLICY,
        "oracle": "no",
        "score_source": "hdec_score",
        "support_source": row.get("support_source", args.support_source if shot > 0 else ""),
        "support_total": split_meta.get("support_total", row.get("support_total", "")),
        "eval_total_before": split_meta.get("eval_total_before", row.get("eval_total_before", "")),
        "test_total_after": split_meta.get("test_total_after", row.get("test_total_after", "")),
        "ACC": row.get("acc", ""),
        "AUROC": row.get("AUROC", ""),
        "Recall_all": row.get("recall_all", ""),
        "Precision_all": row.get("precision_all", ""),
        "F1_all": row.get("F1_all", ""),
        "F1_anomaly": row.get("F1_anomaly", ""),
        "output_dir": str(out),
        "exit_code": exit_code,
    }


def collect_category_rows(args: argparse.Namespace, dataset: str, shot: int) -> List[Dict[str, Any]]:
    out = combo_dir(args, dataset, shot)
    rows: List[Dict[str, Any]] = []
    for row in read_csv_rows(out / "up_per_category_summary.csv"):
        if row.get("experiment") != MAIN_EXPERIMENT:
            continue
        if row.get("threshold_policy") != MAIN_THRESHOLD_POLICY:
            continue
        if row.get("category") in {"", "ALL"}:
            continue
        rows.append(
            {
                "shot": shot,
                "dataset": dataset,
                "category": row.get("category", ""),
                "method": MAIN_METHOD_LABEL,
                "ACC": row.get("acc", ""),
                "AUROC": row.get("AUROC", ""),
                "Recall_all": row.get("recall_all", ""),
                "Precision_all": row.get("precision_all", ""),
                "F1_all": row.get("F1_all", ""),
                "F1_anomaly": row.get("F1_anomaly", ""),
                "n": row.get("n", ""),
                "n_normal": row.get("n_normal", ""),
                "n_anomaly": row.get("n_anomaly", ""),
                "output_dir": str(out),
            }
        )
    return rows


def collect_all(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    main_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    for shot in args.shots:
        for dataset in args.datasets:
            row = collect_one(args, dataset, shot)
            if row is not None:
                main_rows.append(row)
            category_rows.extend(collect_category_rows(args, dataset, shot))
    out = Path(args.out_dir)
    write_csv(out / "main_method_summary.csv", main_rows)
    write_json(out / "main_method_summary.json", main_rows)
    write_csv(out / "main_method_by_category.csv", category_rows)
    write_json(out / "main_method_by_category.json", category_rows)
    return main_rows, category_rows


def should_skip(args: argparse.Namespace, dataset: str, shot: int) -> bool:
    if not args.resume:
        return False
    out = combo_dir(args, dataset, shot)
    return (out / "DONE").exists() and (out / "up_macro_summary.csv").exists()


def mark_status(output_dir: Path, return_code: int, elapsed: float) -> None:
    (output_dir / "exit_code").write_text(str(return_code), encoding="utf-8")
    (output_dir / "elapsed_seconds").write_text(f"{elapsed:.3f}", encoding="utf-8")
    done_path = output_dir / "DONE"
    failed_path = output_dir / "FAILED"
    if return_code == 0:
        failed_path.unlink(missing_ok=True)
        done_path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    else:
        done_path.unlink(missing_ok=True)
        failed_path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean reproducible runner for ADAR/Deco main method.")
    parser.add_argument("--adar-root", default=os.environ.get("ADAR_ROOT", str(Path.cwd())))
    parser.add_argument("--transformers-root", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--attn-implementation", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--trend-layers", default="0:34")
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--score-deco-mode", default="multi_top3", choices=["single", "multi_top3"])
    parser.add_argument("--support-source", default="auto", choices=["auto", "train", "eval_normal_holdout"])
    parser.add_argument("--strict-fewshot-support", action="store_true")
    parser.add_argument("--visa-split-csv", default="")
    parser.add_argument("--visa-split-set", default="test")
    parser.add_argument("--visa-support-split-set", default="train")
    parser.add_argument("--elpv-label-policy", default="nonzero_anomaly", choices=["strict", "threshold_0.5", "nonzero_anomaly"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.add_argument("--stream-jsonl", action="store_true", default=True)
    parser.add_argument("--no-stream-jsonl", dest="stream_jsonl", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    adar_root = Path(args.adar_root)
    if not args.transformers_root:
        args.transformers_root = str(adar_root / "Deco-main" / "transformers-4.57.6")
    if not args.model_path:
        args.model_path = str(adar_root / "Deco-main" / "qwen3_4b_vl_instruct")
    if not args.data_root:
        args.data_root = str(adar_root / "evaluation" / "ODinW-13" / "data")
    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(adar_root / "Deco-main" / "runs" / f"repro_main_method_{stamp}")
    return args


def main() -> None:
    args = normalize_args(parse_args())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", vars(args))

    commands_path = out / "commands.jsonl"
    if commands_path.exists() and not args.resume:
        commands_path.unlink()

    failures: List[Dict[str, Any]] = []
    for shot in args.shots:
        for dataset in args.datasets:
            cmd, output_dir = build_command(args, dataset, shot)
            command_row = {
                "shot": shot,
                "dataset": dataset,
                "output_dir": str(output_dir),
                "command": command_to_string(cmd),
            }
            append_jsonl(commands_path, command_row)
            if args.collect_only:
                continue
            if args.dry_run:
                print(command_row["command"])
                continue
            if should_skip(args, dataset, shot):
                print(f"[SKIP] {shot}shot {dataset}: DONE exists", flush=True)
                collect_all(args)
                continue

            print(f"[RUN] {shot}shot {dataset}", flush=True)
            start = time.time()
            return_code = run_command(args, cmd, output_dir)
            elapsed = time.time() - start
            mark_status(output_dir, return_code, elapsed)
            collect_all(args)
            if return_code != 0:
                failure = {"shot": shot, "dataset": dataset, "return_code": return_code, "output_dir": str(output_dir)}
                failures.append(failure)
                print(f"[FAILED] {shot}shot {dataset} rc={return_code}", flush=True)
                if args.fail_fast:
                    write_json(out / "failures.json", failures)
                    raise SystemExit(return_code)
            else:
                print(f"[DONE] {shot}shot {dataset} elapsed={elapsed:.1f}s", flush=True)

    main_rows, category_rows = collect_all(args)
    write_json(out / "failures.json", failures)
    print(f"[OK] output_dir={out}", flush=True)
    print(f"[OK] main rows={len(main_rows)} category rows={len(category_rows)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
