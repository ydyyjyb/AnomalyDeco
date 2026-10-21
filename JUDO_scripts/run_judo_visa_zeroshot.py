#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot VisA evaluation with the official JUDO checkpoint.

Definition used here:
  - No VisA training samples.
  - No normal template image.
  - Each test item is evaluated from the query image + object category text only.
  - Output is parsed from JUDO-style <seg>/<think>/<answer>A|B</answer>.

Label convention:
  A = normal
  B = anomalous
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def make_system_prompt(grid_size: int) -> str:
    return (
        "You are a helpful assistant for industrial visual quality inspection.\n"
        f"The query image is divided into a grid of {grid_size}×{grid_size} patches. "
        "Patch coordinates are (row, col), starting from 1 at the top-left.\n\n"
        "Task: decide whether the query image is normal or anomalous.\n"
        "Answer choices:\n"
        "A. Normal: no visible industrial anomaly or defect.\n"
        "B. Anomalous: a visible defect, damage, contamination, deformation, missing part, "
        "scratch, hole, color spot, or other abnormal region exists.\n\n"
        "Always respond in this exact order and format:\n"
        "1) <seg> Patch list of the anomalous region. If there is no anomaly, write None. </seg>\n"
        "2) <think> One concise sentence explaining the visual evidence. </think>\n"
        "3) <answer>A|B</answer>\n"
    )


def load_visa_test_cases(visa_root: Path, split_csv: str):
    csv_path = visa_root / "split_csv" / split_csv
    cases = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != "test":
                continue
            label = row["label"]
            gt = "A" if label == "normal" else "B"
            cases.append(
                {
                    "object": row["object"],
                    "label": label,
                    "image_rel": row["image"],
                    "mask_rel": row.get("mask") or "",
                    "gt_answer": gt,
                    "gt_binary": 0 if gt == "A" else 1,
                }
            )
    return cases


def parse_answer(text: str) -> str:
    m = re.search(r"<answer>\s*([AB])\s*</answer>", text, re.I)
    if m:
        return m.group(1).upper()
    # Prefer explicit "answer is A/B" patterns before any arbitrary letter.
    m = re.search(r"(?:answer|choice|correct answer)\s*(?:is|:)?\s*([AB])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", text)
    return m.group(1).upper() if m else "X"


def parse_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(records):
    tp = sum(1 for r in records if r["gt_binary"] == 1 and r["pred_binary"] == 1)
    tn = sum(1 for r in records if r["gt_binary"] == 0 and r["pred_binary"] == 0)
    fp = sum(1 for r in records if r["gt_binary"] == 0 and r["pred_binary"] == 1)
    fn = sum(1 for r in records if r["gt_binary"] == 1 and r["pred_binary"] == 0)
    invalid = sum(1 for r in records if r["prediction"] == "X")
    total = len(records)

    prec_anom = safe_div(tp, tp + fp)
    rec_anom = safe_div(tp, tp + fn)
    f1_anom = safe_div(2 * prec_anom * rec_anom, prec_anom + rec_anom)

    prec_norm = safe_div(tn, tn + fn)
    rec_norm = safe_div(tn, tn + fp)
    f1_norm = safe_div(2 * prec_norm * rec_norm, prec_norm + rec_norm)

    acc = safe_div(tp + tn, total)
    recall_all_macro = (rec_norm + rec_anom) / 2
    precision_all_macro = (prec_norm + prec_anom) / 2
    f1_all_macro = (f1_norm + f1_anom) / 2

    # With binary hard predictions as scores, AUROC reduces to balanced accuracy.
    auroc_binary_pred = recall_all_macro

    return {
        "total": total,
        "valid": total - invalid,
        "invalid": invalid,
        "acc": acc,
        "auroc_binary_pred": auroc_binary_pred,
        "precision_anomaly": prec_anom,
        "recall_anomaly": rec_anom,
        "f1_anomaly": f1_anom,
        "precision_normal": prec_norm,
        "recall_normal": rec_norm,
        "f1_normal": f1_norm,
        "precision_all_macro": precision_all_macro,
        "recall_all_macro": recall_all_macro,
        "f1_all_macro": f1_all_macro,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_normal": sum(1 for r in records if r["prediction"] == "A"),
        "pred_anomaly": sum(1 for r in records if r["prediction"] == "B"),
        "gt_normal": sum(1 for r in records if r["gt_binary"] == 0),
        "gt_anomaly": sum(1 for r in records if r["gt_binary"] == 1),
    }


def batch_iter(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vina04/ADAR/JUDO/models/JUDO")
    parser.add_argument("--visa-root", default="/home/vina04/ADAR/evaluation/ODinW-13/data/VisA")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split-csv", default="1cls.csv")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-fast-processor", action="store_true")
    args = parser.parse_args()

    visa_root = Path(args.visa_root)
    out_dir = Path(args.out_dir or f"/home/vina04/ADAR/JUDO/runs/visa_zeroshot_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_visa_test_cases(visa_root, args.split_csv)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    print(f"[INFO] cases={len(cases)} out_dir={out_dir}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=args.use_fast_processor,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    # Qwen2.5-VL is decoder-only. Batched generation must use left padding;
    # otherwise the model may condition on padding positions and produce
    # unreliable answers. The official JUDO eval script sets this as well.
    processor.tokenizer.padding_side = "left"
    tokenizer.padding_side = "left"

    system_prompt = make_system_prompt(args.grid_size)
    records = []
    jsonl_path = out_dir / "judo_visa_zeroshot_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for bidx, chunk in enumerate(batch_iter(cases, args.batch_size), 1):
            texts = []
            images = []
            for case in chunk:
                image_path = visa_root / case["image_rel"]
                image = Image.open(image_path).convert("RGB").resize((512, 512), Image.LANCZOS)
                question = (
                    f"Object category: {case['object']}.\n"
                    "Inspect the query image directly. Do not assume a normal reference image is available.\n"
                    "Question: Is the query image normal or anomalous?\n"
                    "Choices:\n"
                    "A. Normal.\n"
                    "B. Anomalous.\n"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                images.append(image)

            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda:0")
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            new_tokens = generated[:, inputs.input_ids.shape[1] :]
            outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for case, output in zip(chunk, outputs):
                output = output.strip()
                pred = parse_answer(output)
                pred_binary = 1 if pred == "B" else 0
                rec = {
                    **case,
                    "image_path": str(visa_root / case["image_rel"]),
                    "prediction": pred,
                    "pred_binary": pred_binary,
                    "correct": pred == case["gt_answer"],
                    "segmentation": parse_tag(output, "seg"),
                    "thinking": parse_tag(output, "think"),
                    "output": output,
                }
                records.append(rec)
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[DONE] {len(records)}/{len(cases)} {case['object']} "
                    f"label={case['label']} gt={case['gt_answer']} pred={pred} correct={rec['correct']}",
                    flush=True,
                )

            # Release per-batch tensors.
            del inputs, generated, new_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with (out_dir / "judo_visa_zeroshot_results.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary = compute_metrics(records)
    by_category = {}
    for obj in sorted({r["object"] for r in records}):
        by_category[obj] = compute_metrics([r for r in records if r["object"] == obj])
    summary["by_category"] = by_category

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Compact CSV for tables.
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "total",
            "acc",
            "auroc_binary_pred",
            "recall_all_macro",
            "precision_all_macro",
            "f1_all_macro",
            "precision_anomaly",
            "recall_anomaly",
            "f1_anomaly",
            "tp",
            "tn",
            "fp",
            "fn",
            "pred_normal",
            "pred_anomaly",
            "gt_normal",
            "gt_anomaly",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {k: summary.get(k) for k in fieldnames if k != "dataset"}
        row["dataset"] = "VisA"
        writer.writerow(row)
        for obj, m in by_category.items():
            row = {k: m.get(k) for k in fieldnames if k != "dataset"}
            row["dataset"] = f"VisA/{obj}"
            writer.writerow(row)

    print("[SUMMARY]", json.dumps({k: v for k, v in summary.items() if k != "by_category"}, ensure_ascii=False), flush=True)
    print("[OUT_DIR]", out_dir, flush=True)


if __name__ == "__main__":
    main()
