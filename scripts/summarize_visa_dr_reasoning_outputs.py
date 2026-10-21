#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize completed VisA-D&R reasoning prediction directories.

This script does not load the VLM and does not generate text.  It only reads
``reasoning_predictions.jsonl``, computes ROUGE-L / SBERT metrics, and attaches
quality-control counts.  Use it after generation so metric calculation cannot
block or corrupt the generation stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from qc_visa_dr_reasoning_outputs import qc_rows, read_jsonl, write_csv, write_json
from repro_visa_dr_reasoning import evaluate_reasoning


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def summarize_variant(args: argparse.Namespace, variant_dir: Path) -> Dict[str, Any]:
    predictions = variant_dir / "reasoning_predictions.jsonl"
    if not predictions.exists():
        raise FileNotFoundError(predictions)

    rows = read_jsonl(predictions)
    metrics = evaluate_reasoning(args, rows, variant_dir)
    qc_args = argparse.Namespace(
        predictions=predictions,
        ngram_size=args.qc_ngram_size,
        max_repeated_ngram=args.qc_max_repeated_ngram,
        max_repeated_line=args.qc_max_repeated_line,
        allow_markdown_url=args.qc_allow_markdown_url,
    )
    qc_summary, qc_failures = qc_rows(qc_args, rows)
    write_json(variant_dir / "quality_control.json", qc_summary)
    write_csv(variant_dir / "quality_control.csv", [qc_summary])
    write_csv(variant_dir / "quality_control_failures.csv", qc_failures)

    run_config_path = variant_dir / "run_config.json"
    run_config: Dict[str, Any] = {}
    if run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as f:
            run_config = json.load(f)

    summary = {
        "variant": variant_dir.name,
        "method": run_config.get("method", variant_dir.name),
        "prediction_file": str(predictions),
        "reasoning_rows": len(rows),
        "questions": qc_summary["questions"],
        "low_pairs": metrics.get("low_pairs", ""),
        "complex_pairs": metrics.get("complex_pairs", ""),
        "low_rouge_l": metrics.get("low_rouge_l", ""),
        "low_sbert": metrics.get("low_sbert", ""),
        "complex_sbert": metrics.get("complex_sbert", ""),
        "complex_rouge_l_aux": metrics.get("complex_rouge_l_aux", ""),
        "qc_pass": qc_summary["qc_pass"],
        "hit_max_new_tokens": qc_summary["hit_max_new_tokens"],
        "degenerate_outputs": qc_summary["degenerate_outputs"],
        "markdown_url_outputs": qc_summary["markdown_url_outputs"],
        "nonlexical_outputs": qc_summary["nonlexical_outputs"],
        "empty_outputs": qc_summary["empty_outputs"],
        "failure_records": qc_summary["failure_records"],
        "avg_output_tokens": qc_summary["avg_output_tokens"],
        "avg_chars": qc_summary["avg_chars"],
        "max_new_tokens": run_config.get("max_new_tokens", ""),
        "repetition_penalty": run_config.get("repetition_penalty", ""),
        "no_repeat_ngram_size": run_config.get("no_repeat_ngram_size", ""),
        "answer_instruction": run_config.get("answer_instruction", ""),
        "sbert_model": metrics.get("sbert_model", ""),
        "rouge_backend_counts": json.dumps(metrics.get("rouge_backend_counts", {}), ensure_ascii=False),
    }
    write_json(variant_dir / "paper_ready_summary.json", summary)
    write_csv(variant_dir / "paper_ready_summary.csv", [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed VisA-D&R reasoning outputs.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["qwen3_original", "qwen3_deco"])
    parser.add_argument("--summary-name", default="paper_ready_summary.csv")
    parser.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    parser.add_argument("--rouge-truncate-chars", type=int, default=512)
    parser.add_argument("--rouge-use-stemmer", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-cpu", action="store_true")
    parser.add_argument("--qc-ngram-size", type=int, default=8)
    parser.add_argument("--qc-max-repeated-ngram", type=int, default=4)
    parser.add_argument("--qc-max-repeated-line", type=int, default=3)
    parser.add_argument("--qc-allow-markdown-url", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: List[Dict[str, Any]] = []
    for variant in args.variants:
        summaries.append(summarize_variant(args, args.root / variant))
    write_csv(args.root / args.summary_name, summaries)
    write_json(args.root / args.summary_name.replace(".csv", ".json"), summaries)
    for row in summaries:
        print(json.dumps({k: fmt(v) for k, v in row.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
