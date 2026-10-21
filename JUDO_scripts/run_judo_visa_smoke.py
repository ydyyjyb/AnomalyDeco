#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal JUDO smoke test on local VisA data.

This script intentionally does not modify the upstream JUDO repository. It loads
the official JUDO checkpoint, builds two-image QA inputs from VisA
(query image + same-category normal template), and asks the model to answer in
JUDO's <seg>/<think>/<answer> format.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def make_system_prompt(grid_size: int) -> str:
    return (
        "You are a helpful assistant for industrial quality control tasks. "
        "The user asks a question, and the Assistant solves it.\n"
        f"The query image is divided into a grid of {grid_size}×{grid_size} patches. "
        "Patch coordinates are (row, col) and both row and col start from 1 "
        "(top-left patch is (1,1)).\n\n"
        "Always respond in this exact order and format:\n"
        "1) <seg> The anomalous region patch list. Use (r,c) and "
        "(r,c_start)-(r,c_end) for contiguous columns. If there is no anomaly, "
        "respond 'None'. </seg>\n"
        "2) <think> Provide concise visual reasoning. </think>\n"
        "3) <answer>A|B</answer>\n"
    )


def read_visa_cases(visa_root: Path, split_csv: str, seed: int, n_anomaly: int, n_normal: int):
    csv_path = visa_root / "split_csv" / split_csv
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test":
                rows.append(row)

    train_normals = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "train" and row.get("label") == "normal":
                train_normals[row["object"]].append(row["image"])

    anomalies = [r for r in rows if r.get("label") != "normal"]
    normals = [r for r in rows if r.get("label") == "normal"]
    rnd = random.Random(seed)
    rnd.shuffle(anomalies)
    rnd.shuffle(normals)

    cases = []
    for row in anomalies[:n_anomaly] + normals[:n_normal]:
        obj = row["object"]
        if not train_normals[obj]:
            continue
        template = rnd.choice(train_normals[obj])
        cases.append(
            {
                "object": obj,
                "label": row["label"],
                "query_rel": row["image"],
                "template_rel": template,
                "gt_answer": "A" if row["label"] == "normal" else "B",
            }
        )
    return cases


def parse_answer(text: str) -> str:
    m = re.search(r"<answer>\s*([AB])\s*</answer>", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", text)
    return m.group(1).upper() if m else "X"


def parse_seg(text: str) -> str:
    m = re.search(r"<seg>(.*?)</seg>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def draw_patch_grid_preview(image: Image.Image, seg_text: str, grid_size: int, label: str) -> Image.Image:
    """Draw the raw query image with grid and model-reported patch cells highlighted.

    This is only a preview. It does not convert JUDO patch output into a precise
    pixel-level mask.
    """
    im = image.convert("RGB").resize((512, 512), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    w, h = im.size
    cw, ch = w / grid_size, h / grid_size

    for i in range(1, grid_size):
        x = int(i * cw)
        y = int(i * ch)
        draw.line([(x, 0), (x, h)], fill=(80, 80, 80), width=1)
        draw.line([(0, y), (w, y)], fill=(80, 80, 80), width=1)

    cells = set()
    for r, c in re.findall(r"\((\d+)\s*,\s*(\d+)\)", seg_text):
        rr, cc = int(r), int(c)
        if 1 <= rr <= grid_size and 1 <= cc <= grid_size:
            cells.add((rr, cc))
    for r, c1, c2 in re.findall(r"\((\d+)\s*,\s*(\d+)\s*\)\s*-\s*\(\s*\d+\s*,\s*(\d+)\s*\)", seg_text):
        rr, a, b = int(r), int(c1), int(c2)
        for cc in range(min(a, b), max(a, b) + 1):
            if 1 <= rr <= grid_size and 1 <= cc <= grid_size:
                cells.add((rr, cc))

    for rr, cc in cells:
        x0, y0 = int((cc - 1) * cw), int((rr - 1) * ch)
        x1, y1 = int(cc * cw), int(rr * ch)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)

    draw.rectangle([0, 0, w, 26], fill=(255, 255, 255))
    draw.text((4, 5), label[:90], fill=(0, 0, 0))
    return im


def make_contact_sheet(previews, out_path: Path, cols: int = 2):
    if not previews:
        return
    cell_w, cell_h = previews[0].size
    rows = (len(previews) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, im in enumerate(previews):
        sheet.paste(im, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    sheet.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vina04/ADAR/JUDO/models/JUDO")
    parser.add_argument("--visa-root", default="/home/vina04/ADAR/evaluation/ODinW-13/data/VisA")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split-csv", default="1cls.csv")
    parser.add_argument("--n-anomaly", type=int, default=4)
    parser.add_argument("--n-normal", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    visa_root = Path(args.visa_root)
    out_dir = Path(args.out_dir or f"/home/vina04/ADAR/JUDO/runs/visa_smoke_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = read_visa_cases(visa_root, args.split_csv, args.seed, args.n_anomaly, args.n_normal)
    print(f"[INFO] cases={len(cases)} out_dir={out_dir}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)

    system_prompt = make_system_prompt(args.grid_size)
    results = []
    previews = []
    for idx, case in enumerate(cases, 1):
        query_path = visa_root / case["query_rel"]
        template_path = visa_root / case["template_rel"]
        query = Image.open(query_path).convert("RGB")
        template = Image.open(template_path).convert("RGB")

        question = (
            "First image = QUERY. Second image = NORMAL template from the same object category.\n"
            f"Object category: {case['object']}.\n"
            "Question: Is there any visible industrial anomaly or defect in the QUERY image compared with the NORMAL template?\n"
            "Choices:\n"
            "A. No. The query image is normal.\n"
            "B. Yes. The query image is anomalous or defective.\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[query, template], padding=True, return_tensors="pt")
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
        output = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        pred = parse_answer(output)
        seg = parse_seg(output)
        ok = pred == case["gt_answer"]

        rec = {
            **case,
            "query_path": str(query_path),
            "template_path": str(template_path),
            "prediction": pred,
            "correct": ok,
            "segmentation": seg,
            "output": output,
        }
        results.append(rec)
        print(f"[DONE] {idx}/{len(cases)} {case['object']} label={case['label']} gt={case['gt_answer']} pred={pred} correct={ok}", flush=True)

        label = f"#{idx} {case['object']} {case['label']} GT={case['gt_answer']} PRED={pred}"
        previews.append(draw_patch_grid_preview(query, seg, args.grid_size, label))

    with (out_dir / "judo_visa_smoke_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (out_dir / "judo_visa_smoke_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else None,
        "answer_counts": {k: sum(1 for r in results if r["prediction"] == k) for k in ["A", "B", "X"]},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    make_contact_sheet(previews, out_dir / "judo_visa_smoke_contact_sheet.jpg")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False), flush=True)
    print("[OUT_DIR]", out_dir, flush=True)


if __name__ == "__main__":
    main()
