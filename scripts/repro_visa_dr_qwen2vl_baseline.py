#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen2-VL baseline reproduction on VisA-D&R.

This script follows the visible Anomaly-OneVision evaluation protocol for the
generalist MLLM rows in Table 4:

1. Detection:
   - Prompt: "Are there any defects for the object in the image? Please reply
     with 'Yes' or 'No'."
   - Prediction parser: anomalous if the generated text contains "Yes" or
     "yes", matching scripts/eval/calc_accuracy.py in the official repository.
   - Metrics: class-macro Accuracy / Precision / Recall / F1.

2. Reasoning:
   - Use VisA-D&R visa_dr.json.
   - Skip pcb2 and pcb4, matching scripts/eval/test_reasoning.py.
   - Ask Q1 with the image, then ask later questions with previous Q/A history.
   - Generation: do_sample=False, temperature=0, max_new_tokens=4096.

This is intentionally independent of ADAR HDEC/Youden/DeCo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from repro_visa_dr_reasoning import (
    evaluate_reasoning,
    fmt,
    norm_rel_image,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_ADAR_ROOT = "/home/vina04/ADAR"
DEFAULT_DATA_ROOT = f"{DEFAULT_ADAR_ROOT}/evaluation/ODinW-13/data"
DEFAULT_VISA_DR_JSON = f"{DEFAULT_ADAR_ROOT}/evaluation/VisA-DR/visa_dr.json"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_ADAR_ROOT}/Deco-main/runs"

DEFAULT_MODEL_PATHS = {
    "qwen2vl-2b": f"{DEFAULT_ADAR_ROOT}/Deco-main/qwen2_vl_2b_instruct",
    "qwen2vl-7b": f"{DEFAULT_ADAR_ROOT}/Deco-main/qwen2_vl_7b_instruct",
}

MODEL_DISPLAY_NAMES = {
    "qwen2vl-2b": "Qwen2-VL-2B-Instruct",
    "qwen2vl-7b": "Qwen2-VL-7B-Instruct",
}

PAPER_TABLE4 = {
    "Qwen2-VL-2B": {
        "acc": 0.65,
        "precision": 0.87,
        "recall": 0.55,
        "f1": 0.59,
        "low_rouge_l": 0.22,
        "low_sbert": 0.55,
        "low_gpt_score": 1.94,
        "complex_sbert": 0.74,
        "complex_gpt_score": 4.26,
    },
    "Qwen2-VL-7B": {
        "acc": 0.76,
        "precision": 0.91,
        "recall": 0.69,
        "f1": 0.75,
        "low_rouge_l": 0.25,
        "low_sbert": 0.61,
        "low_gpt_score": 3.09,
        "complex_sbert": 0.68,
        "complex_gpt_score": 4.62,
    },
    "Anomaly-OV-7B": {
        "acc": 0.79,
        "precision": 0.83,
        "recall": 0.86,
        "f1": 0.83,
        "low_rouge_l": 0.34,
        "low_sbert": 0.73,
        "low_gpt_score": 4.26,
        "complex_sbert": 0.84,
        "complex_gpt_score": 6.34,
    },
}

DETECTION_PROMPT = "Are there any defects for the object in the image? Please reply with 'Yes' or 'No'."
DEFAULT_REASONING_SKIP_CLASSES = ["pcb2", "pcb4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def category_from_rel(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[0] != "VisA":
        raise ValueError(f"cannot parse VisA category from path: {path}")
    return parts[1]


def load_image_official(path: str, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_side and max_side > 0 and max(image.size) > max_side:
        if image.width > image.height:
            new_width = max_side
            new_height = int(max_side * image.height / image.width)
        else:
            new_height = max_side
            new_width = int(max_side * image.width / image.height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def select_by_category(items: Sequence[Dict[str, Any]], limit: int = 0, limit_per_category: int = 0) -> List[Dict[str, Any]]:
    if limit_per_category and limit_per_category > 0:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            buckets[str(item["category"])].append(item)
        selected: List[Dict[str, Any]] = []
        for category in sorted(buckets):
            selected.extend(buckets[category][:limit_per_category])
        if limit and limit > 0:
            selected = selected[:limit]
        return selected
    if limit and limit > 0:
        return list(items[:limit])
    return list(items)


def select_detection_by_category_label(
    items: Sequence[Dict[str, Any]],
    limit: int = 0,
    limit_per_category: int = 0,
) -> List[Dict[str, Any]]:
    if limit_per_category and limit_per_category > 0:
        buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            buckets[(str(item["category"]), 1 if item["anomaly"] else 0)].append(item)
        selected: List[Dict[str, Any]] = []
        for category in sorted({str(item["category"]) for item in items}):
            selected.extend(buckets[(category, 0)][:limit_per_category])
            selected.extend(buckets[(category, 1)][:limit_per_category])
        if limit and limit > 0:
            selected = selected[:limit]
        return selected
    if limit and limit > 0:
        return list(items[:limit])
    return list(items)


def load_detection_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    json_path = Path(args.detection_json) if args.detection_json else None
    items: List[Dict[str, Any]] = []
    if json_path and json_path.exists():
        data = read_json(json_path)
        for i, raw in enumerate(data):
            rel = norm_rel_image(str(raw["image"]), args.data_root)
            category = category_from_rel(rel)
            if category in args.detection_skip_classes:
                continue
            items.append(
                {
                    "index": str(raw.get("index", f"{i:08d}")),
                    "image": rel,
                    "abs_image": str(Path(args.data_root) / rel),
                    "category": category,
                    "anomaly": bool(raw.get("anomaly")),
                }
            )
    else:
        visa_root = Path(args.data_root) / "VisA"
        idx = 0
        for category_dir in sorted(p for p in visa_root.iterdir() if p.is_dir()):
            category = category_dir.name
            if category in args.detection_skip_classes:
                continue
            image_root = category_dir / "Data" / "Images"
            for folder_name, is_anomaly in [("Normal", False), ("Anomaly", True)]:
                folder = image_root / folder_name
                if not folder.exists():
                    continue
                for image_path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
                    rel = image_path.relative_to(args.data_root).as_posix()
                    items.append(
                        {
                            "index": f"{idx:08d}",
                            "image": rel,
                            "abs_image": str(image_path),
                            "category": category,
                            "anomaly": is_anomaly,
                        }
                    )
                    idx += 1
    return select_detection_by_category_label(items, args.limit_detection, args.limit_detection_per_category)


def load_reasoning_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data = read_json(Path(args.visa_dr_json))
    skip = set(args.reasoning_skip_classes)
    items: List[Dict[str, Any]] = []
    for raw in data:
        rel = norm_rel_image(str(raw["image"]), args.data_root)
        category = category_from_rel(rel)
        if category in skip:
            continue
        row = dict(raw)
        row["index"] = str(row.get("index", len(items)))
        row["image"] = rel
        row["abs_image"] = str(Path(args.data_root) / rel)
        row["category"] = category
        items.append(row)
    return select_by_category(items, args.limit_reasoning, args.limit_reasoning_per_category)


def load_qwen2vl(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return processor, model


def qwen2_chat_text(processor: Any, messages: List[Dict[str, Any]]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    image: Image.Image,
    messages: List[Dict[str, Any]],
) -> Tuple[str, int, bool]:
    import torch

    text = qwen2_chat_text(processor, messages)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    text_outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    token_count = int((generated_ids != pad_token_id).sum().item()) if pad_token_id is not None else int(generated_ids.numel())
    return text_outputs[0].strip(), token_count, bool(token_count >= args.max_new_tokens)


def detection_pred_from_text(text: str) -> int:
    return 1 if "Yes" in text or "yes" in text else 0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def detection_class_macro(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["category"])].append(row)
    per_class: List[Dict[str, Any]] = []
    for category, group in sorted(buckets.items()):
        tp = sum(1 for r in group if int(r["gt"]) == 1 and int(r["pred"]) == 1)
        tn = sum(1 for r in group if int(r["gt"]) == 0 and int(r["pred"]) == 0)
        fp = sum(1 for r in group if int(r["gt"]) == 0 and int(r["pred"]) == 1)
        fn = sum(1 for r in group if int(r["gt"]) == 1 and int(r["pred"]) == 0)
        total = len(group)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "category": category,
                "n": total,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "acc": fmt(safe_div(tp + tn, total)),
                "precision": fmt(precision),
                "recall": fmt(recall),
                "f1": fmt(f1),
            }
        )
    macro: Dict[str, Any] = {
        "n": sum(int(r["n"]) for r in per_class),
        "num_classes": len(per_class),
    }
    for metric in ["acc", "precision", "recall", "f1"]:
        macro[metric] = fmt(sum(float(r[metric]) for r in per_class) / max(1, len(per_class)))
    return macro | {"per_class": per_class}


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_detection(args: argparse.Namespace, processor: Any, model: Any, out_dir: Path) -> List[Dict[str, Any]]:
    path = out_dir / "detection_predictions.jsonl"
    rows = read_jsonl(path) if args.resume and path.exists() else []
    done = {(str(r.get("index")), str(r.get("image"))) for r in rows}
    items = load_detection_items(args)
    todo = [item for item in items if (str(item["index"]), str(item["image"])) not in done]
    print(f"[DET] total={len(items)} done={len(rows)} todo={len(todo)}", flush=True)

    for i, item in enumerate(todo, start=1):
        if i == 1 or i % args.log_every == 0 or i == len(todo):
            print(f"[DET] {i}/{len(todo)} {item['category']} {item['image']}", flush=True)
        image = load_image_official(item["abs_image"], args.max_image_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ]
        start = time.time()
        response, token_count, hit_max = generate_text(args, processor, model, image, messages)
        row = {
            "index": str(item["index"]),
            "category": item["category"],
            "image": item["image"],
            "prompt": DETECTION_PROMPT,
            "response": response,
            "gt": 1 if item["anomaly"] else 0,
            "pred": detection_pred_from_text(response),
            "token_count": token_count,
            "hit_max_new_tokens": hit_max,
            "seconds": round(time.time() - start, 4),
        }
        rows.append(row)
        append_jsonl(path, row)

    metrics = detection_class_macro(rows)
    write_json(out_dir / "detection_summary.json", metrics)
    write_csv(out_dir / "detection_by_category.csv", metrics["per_class"])
    write_csv(out_dir / "detection_summary.csv", [{k: v for k, v in metrics.items() if k != "per_class"}])
    return rows


def build_reasoning_messages(item: Dict[str, Any], responses: Sequence[str], question_index: int) -> List[Dict[str, Any]]:
    questions = list(item["questions"])
    messages: List[Dict[str, Any]] = []
    for i in range(question_index):
        if i == 0:
            content = [{"type": "image"}, {"type": "text", "text": questions[i]}]
        else:
            content = [{"type": "text", "text": questions[i]}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": responses[i]})

    if question_index == 0:
        content = [{"type": "image"}, {"type": "text", "text": questions[question_index]}]
    else:
        content = [{"type": "text", "text": questions[question_index]}]
    messages.append({"role": "user", "content": content})
    return messages


def run_reasoning(args: argparse.Namespace, processor: Any, model: Any, out_dir: Path) -> List[Dict[str, Any]]:
    path = out_dir / "reasoning_predictions.jsonl"
    rows = read_jsonl(path) if args.resume and path.exists() else []
    done = {str(r.get("index")) for r in rows}
    items = load_reasoning_items(args)
    todo = [item for item in items if str(item["index"]) not in done]
    print(f"[RSN] total={len(items)} done={len(rows)} todo={len(todo)}", flush=True)

    for i, item in enumerate(todo, start=1):
        if i == 1 or i % args.log_every == 0 or i == len(todo):
            print(f"[RSN] image={i}/{len(todo)} {item['category']} {item['image']}", flush=True)
        image = load_image_official(item["abs_image"], args.max_image_side)
        responses: List[str] = []
        token_counts: List[int] = []
        hit_max_new_tokens: List[bool] = []
        seconds_per_question: List[float] = []
        for q_idx in range(len(item["questions"])):
            messages = build_reasoning_messages(item, responses, q_idx)
            start = time.time()
            response, token_count, hit_max = generate_text(args, processor, model, image, messages)
            responses.append(response)
            token_counts.append(token_count)
            hit_max_new_tokens.append(hit_max)
            seconds_per_question.append(round(time.time() - start, 4))
        row = {
            "index": str(item["index"]),
            "category": item["category"],
            "image": item["image"],
            "questions": item["questions"],
            "answers": item["answers"],
            "responses": responses,
            "raw_output_token_counts": token_counts,
            "hit_max_new_tokens": hit_max_new_tokens,
            "seconds_per_question": seconds_per_question,
        }
        rows.append(row)
        append_jsonl(path, row)

    write_json(out_dir / "reasoning_generation_checkpoint.json", rows)
    return rows


def write_table4_comparison(out_dir: Path, method_name: str, ours: Dict[str, Any]) -> None:
    metrics = [
        "acc",
        "precision",
        "recall",
        "f1",
        "low_rouge_l",
        "low_sbert",
        "low_gpt_score",
        "complex_sbert",
        "complex_gpt_score",
    ]
    rows = [{"method": method_name, **{m: ours.get(m, "") for m in metrics}}]
    for name, vals in PAPER_TABLE4.items():
        rows.append({"method": name, **{m: vals.get(m, "") for m in metrics}})
    short_name = "Qwen2-VL-2B" if "2B" in method_name else "Qwen2-VL-7B"
    if short_name in PAPER_TABLE4:
        delta = {"method": f"{method_name} - Paper {short_name}"}
        for m in metrics:
            if ours.get(m, "") != "" and PAPER_TABLE4[short_name].get(m, "") != "":
                delta[m] = fmt(float(ours[m]) - float(PAPER_TABLE4[short_name][m]))
            else:
                delta[m] = ""
        rows.append(delta)
    write_csv(out_dir / "table4_comparison.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce VisA-D&R Qwen2-VL baseline.")
    parser.add_argument("--model", choices=["qwen2vl-2b", "qwen2vl-7b"], required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--visa-dr-json", default=DEFAULT_VISA_DR_JSON)
    parser.add_argument("--detection-json", default="", help="Optional official VisA/test_data.json path if available.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mode", choices=["all", "detection", "reasoning", "evaluate"], default="all")
    parser.add_argument("--detection-skip-classes", nargs="*", default=[])
    parser.add_argument("--reasoning-skip-classes", nargs="*", default=DEFAULT_REASONING_SKIP_CLASSES)
    parser.add_argument("--limit-detection", type=int, default=0)
    parser.add_argument(
        "--limit-detection-per-category",
        type=int,
        default=0,
        help="For detection smoke tests, keep N normal and N anomaly images per category.",
    )
    parser.add_argument("--limit-reasoning", type=int, default=0)
    parser.add_argument("--limit-reasoning-per-category", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    parser.add_argument("--rouge-truncate-chars", type=int, default=512)
    parser.add_argument("--rouge-use-stemmer", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-cpu", action="store_true")
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = DEFAULT_MODEL_PATHS[args.model]
    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"{DEFAULT_OUTPUT_ROOT}/visa_dr_{args.model}_baseline_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))

    processor = model = None
    if args.mode in {"all", "detection", "reasoning"}:
        processor, model = load_qwen2vl(args)

    detection_metrics: Dict[str, Any] = {}
    if args.mode in {"all", "detection"}:
        det_rows = run_detection(args, processor, model, out_dir)
        detection_metrics = detection_class_macro(det_rows)
    elif (out_dir / "detection_predictions.jsonl").exists():
        detection_metrics = detection_class_macro(read_jsonl(out_dir / "detection_predictions.jsonl"))

    reasoning_metrics: Dict[str, Any] = {}
    if args.mode in {"all", "reasoning"}:
        rows = run_reasoning(args, processor, model, out_dir)
        if not args.no_sbert:
            import gc
            import torch

            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reasoning_metrics = evaluate_reasoning(args, rows, out_dir)
    elif (out_dir / "reasoning_predictions.jsonl").exists():
        rows = read_jsonl(out_dir / "reasoning_predictions.jsonl")
        reasoning_metrics = evaluate_reasoning(args, rows, out_dir)

    method_name = MODEL_DISPLAY_NAMES[args.model]
    ours = {
        "acc": detection_metrics.get("acc", ""),
        "precision": detection_metrics.get("precision", ""),
        "recall": detection_metrics.get("recall", ""),
        "f1": detection_metrics.get("f1", ""),
        **reasoning_metrics,
    }
    write_json(out_dir / "ours_summary.json", ours)
    write_csv(out_dir / "ours_summary.csv", [{"method": method_name, **ours}])
    write_table4_comparison(out_dir, method_name, ours)

    print("[DONE]", out_dir, flush=True)
    print(json.dumps({"method": method_name, "summary": ours}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
