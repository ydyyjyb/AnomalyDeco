#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry only empty/truncated LLaVA-OV VisA-D&R answers.

The original prediction file is never modified. Empty answers are retried with
the original generation limit; answers flagged as max-token truncated are
retried once with only ``max_new_tokens`` increased to the configured repair
limit. Later turns for an affected image are regenerated because the QA is
sequential and depends on earlier answers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repro_visa_dr_open_models import load_llava, llava_answer  # noqa: E402
from repro_visa_dr_qwen2vl_baseline import load_image_official, read_jsonl  # noqa: E402


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llava_ov_0_5b", "llava_ov_7b"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--predictions-in", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-image-side", type=int, default=1024)
    p.add_argument("--max-new-tokens-original", type=int, default=4096)
    p.add_argument("--max-new-tokens-repair", type=int, default=8192)
    p.add_argument("--empty-only", action="store_true",
                   help="Retry empty answers only; do not spend extra tokens on existing max-token loops.")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.predictions_in)
    rows = read_jsonl(source)
    shutil.copy2(source, out_dir / "reasoning_predictions.before_repair.jsonl")
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    affected = []
    for row_i, row in enumerate(rows):
        responses = row.get("responses") or []
        hits = row.get("hit_max_new_tokens") or []
        failures = [
            q for q, response in enumerate(responses)
            if (not args.empty_only and q < len(hits) and bool(hits[q]))
            or not str(response).strip()
        ]
        if failures:
            affected.append((row_i, failures))

    print(
        f"[DATA] rows={len(rows)} affected_images={len(affected)} "
        f"affected_answers={sum(len(qs) for _, qs in affected)}",
        flush=True,
    )
    if not affected:
        return

    backend = load_llava(args.model_path)
    repairs: List[Dict[str, Any]] = []
    repair_path = out_dir / "repair_log.jsonl"
    for n, (row_i, failures) in enumerate(affected, start=1):
        row = rows[row_i]
        first_q = min(failures)
        image_path = Path(args.data_root) / str(row["image"])
        image = load_image_official(str(image_path), args.max_image_side)
        original_hits = list(row.get("hit_max_new_tokens") or [])
        # Later turn answers must be regenerated after the first repaired turn.
        for q_idx in range(first_q, len(row["questions"])):
            was_truncated = q_idx < len(original_hits) and bool(original_hits[q_idx])
            limit = (
                args.max_new_tokens_repair if was_truncated
                else args.max_new_tokens_original
            )
            t0 = time.time()
            text, count, hit = llava_answer(
                backend,
                image,
                row["questions"],
                row["responses"],
                q_idx,
                limit,
            )
            # If a regenerated continuation hits the original limit, retry only
            # that continuation once with the larger limit as well.
            if hit and limit < args.max_new_tokens_repair and not args.empty_only:
                limit = args.max_new_tokens_repair
                text, count, hit = llava_answer(
                    backend,
                    image,
                    row["questions"],
                    row["responses"],
                    q_idx,
                    limit,
                )
            seconds = round(time.time() - t0, 3)
            row["responses"][q_idx] = text
            row.setdefault("raw_output_token_counts", [0] * len(row["questions"]))
            row.setdefault("hit_max_new_tokens", [False] * len(row["questions"]))
            row.setdefault("seconds_per_question", [0.0] * len(row["questions"]))
            row["raw_output_token_counts"][q_idx] = count
            row["hit_max_new_tokens"][q_idx] = hit
            row["seconds_per_question"][q_idx] = seconds
            repairs.append({
                "index": row.get("index"),
                "category": row.get("category"),
                "question_index": q_idx,
                "was_truncated": was_truncated,
                "max_new_tokens": limit,
                "output_tokens": count,
                "hit_max_new_tokens_after": hit,
                "empty_after": not bool(text.strip()),
                "seconds": seconds,
            })
            with repair_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(repairs[-1], ensure_ascii=False) + "\n")

        write_jsonl(out_dir / "reasoning_predictions.repaired.jsonl", rows)
        if n == 1 or n % args.log_every == 0 or n == len(affected):
            unresolved = sum(
                bool(row.get("hit_max_new_tokens", [])[q])
                or not str(row.get("responses", [""])[q]).strip()
                for q in range(len(row.get("responses", [])))
            )
            print(
                f"[REPAIR] {n}/{len(affected)} index={row.get('index')} "
                f"from_q={first_q} row_unresolved={unresolved}",
                flush=True,
            )

    unresolved = []
    for row in rows:
        for q_idx, response in enumerate(row.get("responses", [])):
            hits = row.get("hit_max_new_tokens") or []
            if (q_idx < len(hits) and bool(hits[q_idx])) or not str(response).strip():
                unresolved.append({"index": row.get("index"), "question_index": q_idx})
    summary = {
        "model": args.model,
        "input_rows": len(rows),
        "affected_images": len(affected),
        "retried_question_turns": len(repairs),
        "unresolved_answers": len(unresolved),
        "only_generation_changes": "max_new_tokens for truncations; same limit for empty outputs",
        "unresolved": unresolved,
    }
    (out_dir / "repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE] " + json.dumps(summary, ensure_ascii=False), flush=True)
    del backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
