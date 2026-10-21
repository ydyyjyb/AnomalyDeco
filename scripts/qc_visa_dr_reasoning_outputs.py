#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quality-control VisA-D&R reasoning generations.

The check is intentionally model-agnostic and sample-agnostic.  It does not
repair, filter, or change any answer.  It only reports whether a completed
generation file is clean enough to use as a paper table source.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def repeated_ngram_max(text: str, n: int) -> Tuple[int, str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return 0, ""
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return int(count), " ".join(gram)


def repeated_line_max(text: str) -> Tuple[int, str]:
    parts = []
    for raw in re.split(r"[\n\r]+|(?<=[.!?])\s+", text):
        norm = normalize_text(raw)
        norm = re.sub(r"^\s*\d+[\).\s-]+", "", norm)
        if len(norm) >= 20:
            parts.append(norm)
    if not parts:
        return 0, ""
    line, count = Counter(parts).most_common(1)[0]
    return int(count), line


def looks_degenerate(
    text: str,
    ngram_size: int,
    max_repeated_ngram: int,
    max_repeated_line: int,
) -> Tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return True, "empty_output"
    ngram_count, ngram = repeated_ngram_max(stripped, ngram_size)
    if ngram_count >= max_repeated_ngram:
        return True, f"repeated_{ngram_size}gram_x{ngram_count}: {ngram[:120]}"
    line_count, line = repeated_line_max(stripped)
    if line_count >= max_repeated_line:
        return True, f"repeated_sentence_x{line_count}: {line[:120]}"
    return False, ""


def looks_like_markdown_or_url(text: str) -> Tuple[bool, str]:
    checks = [
        (r"https?://", "url"),
        (r"\bwww\.", "url"),
        (r"!\s*\[", "markdown_image"),
        (r"\[[^\]]{0,120}\]\s*\(", "markdown_link"),
    ]
    for pattern, reason in checks:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True, reason
    return False, ""


def lacks_lexical_content(text: str) -> bool:
    return re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text or "") is None


def iter_answers(rows: Sequence[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], int, str]]:
    for row in rows:
        for i, text in enumerate(row.get("responses") or []):
            yield row, i, str(text)


def qc_rows(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    total_questions = 0
    hit_max_count = 0
    degenerate_count = 0
    markdown_url_count = 0
    nonlexical_count = 0
    empty_count = 0
    total_chars = 0
    total_tokens = 0
    token_observations = 0

    for row, q_idx, text in iter_answers(rows):
        total_questions += 1
        total_chars += len(text)

        token_counts = row.get("raw_output_token_counts") or []
        token_count = token_counts[q_idx] if q_idx < len(token_counts) else ""
        if isinstance(token_count, int):
            total_tokens += token_count
            token_observations += 1

        hits = row.get("hit_max_new_tokens") or []
        hit_max = bool(hits[q_idx]) if q_idx < len(hits) else False
        if hit_max:
            hit_max_count += 1
            failures.append(
                {
                    "index": row.get("index", ""),
                    "category": row.get("category", ""),
                    "image": row.get("image", ""),
                    "question_index": q_idx + 1,
                    "failure_type": "hit_max_new_tokens",
                    "token_count": token_count,
                    "chars": len(text),
                    "preview": text[:300],
                }
            )

        if not args.allow_markdown_url:
            bad_markup, markup_reason = looks_like_markdown_or_url(text)
            if bad_markup:
                markdown_url_count += 1
                failures.append(
                    {
                        "index": row.get("index", ""),
                        "category": row.get("category", ""),
                        "image": row.get("image", ""),
                        "question_index": q_idx + 1,
                        "failure_type": "markdown_or_url_output",
                        "failure_reason": markup_reason,
                        "token_count": token_count,
                        "chars": len(text),
                        "preview": text[:300],
                    }
            )

        if lacks_lexical_content(text):
            nonlexical_count += 1
            failures.append(
                {
                    "index": row.get("index", ""),
                    "category": row.get("category", ""),
                    "image": row.get("image", ""),
                    "question_index": q_idx + 1,
                    "failure_type": "nonlexical_output",
                    "failure_reason": "no_alphanumeric_or_cjk_content",
                    "token_count": token_count,
                    "chars": len(text),
                    "preview": text[:300],
                }
            )

        bad, reason = looks_degenerate(
            text,
            args.ngram_size,
            args.max_repeated_ngram,
            args.max_repeated_line,
        )
        if bad:
            degenerate_count += 1
            if reason == "empty_output":
                empty_count += 1
            failures.append(
                {
                    "index": row.get("index", ""),
                    "category": row.get("category", ""),
                    "image": row.get("image", ""),
                    "question_index": q_idx + 1,
                    "failure_type": "degenerate_output",
                    "failure_reason": reason,
                    "token_count": token_count,
                    "chars": len(text),
                    "preview": text[:300],
                }
            )

    summary = {
        "prediction_file": str(args.predictions),
        "rows": len(rows),
        "questions": total_questions,
        "hit_max_new_tokens": hit_max_count,
        "degenerate_outputs": degenerate_count,
        "markdown_url_outputs": markdown_url_count,
        "nonlexical_outputs": nonlexical_count,
        "empty_outputs": empty_count,
        "failure_records": len(failures),
        "qc_pass": int(
            hit_max_count == 0
            and degenerate_count == 0
            and markdown_url_count == 0
            and nonlexical_count == 0
            and total_questions > 0
        ),
        "avg_chars": round(total_chars / total_questions, 4) if total_questions else "",
        "avg_output_tokens": round(total_tokens / token_observations, 4) if token_observations else "",
        "ngram_size": args.ngram_size,
        "max_repeated_ngram": args.max_repeated_ngram,
        "max_repeated_line": args.max_repeated_line,
    }
    return summary, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC VisA-D&R reasoning outputs.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--max-repeated-ngram", type=int, default=4)
    parser.add_argument("--max-repeated-line", type=int, default=3)
    parser.add_argument(
        "--allow-markdown-url",
        action="store_true",
        help="Do not fail outputs containing Markdown links/images or URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    out_dir = args.out_dir or args.predictions.parent
    summary, failures = qc_rows(args, rows)
    write_json(out_dir / "quality_control.json", summary)
    write_csv(out_dir / "quality_control.csv", [summary])
    write_csv(out_dir / "quality_control_failures.csv", failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
