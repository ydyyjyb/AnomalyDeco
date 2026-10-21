#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisA-D&R reasoning evaluation for ADAR/DeCo.

This script evaluates our score-based main method on the VisA-D&R protocol and
generates post-hoc benchmark answers for the VisA-D&R reasoning questions.

Protocol used here:

1. Detection set:
   - Use the 10 classes used by the official reasoning script, i.e. VisA-D&R
     categories excluding pcb2 and pcb4.
   - Reuse existing ADAR up_cases.json for VisA.
   - Reconstruct category-wise Youden fixed answers from E_raw_hdec/hdec_score.
   - Report class-macro ACC / Precision / Recall / F1 to align with the
     Anomaly-OV Table 4 detection calculation style.

2. Reasoning set:
   - Use visa_dr.json.
   - Exclude pcb2 and pcb4, matching the official Anomaly-OneVision
     scripts/eval/test_reasoning.py.
   - For each anomaly image, ask the benchmark questions Q1..Qn while
     conditioning generation on the already fixed score-based Answer.
   - Low-level reasoning metrics are averaged over Q1 and Q2.
   - Complex reasoning metrics are averaged over Q3 and later questions.

3. Metrics:
   - ROUGE-L uses the official repository's visible preprocessing style from
     llava/eval/evaluate_interleave.py: normalize whitespace, remove/space
     punctuation, lowercase, truncate predictions to 512 characters, then
     compute rouge-l F-measure.  The original VisA-D&R metric script is not
     released, so this is the closest visible repository-aligned setting.
   - SBERT is configurable.  The paper cites Sentence-BERT but does not release
     the exact checkpoint name.  The default is
     sentence-transformers/all-MiniLM-L6-v2.
   - GPT-Score is configurable through an OpenAI-compatible judge call on a
     0-10 scale.  The paper states "GPT-4 as judge" but does not release the
     judge prompt, so the prompt version is saved explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SKIP_CLASSES = ["pcb2", "pcb4"]
DEFAULT_MODEL_PATH = "/home/vina04/ADAR/Deco-main/qwen3_4b_vl_instruct"
DEFAULT_DATA_ROOT = "/home/vina04/ADAR/evaluation/ODinW-13/data"
DEFAULT_VISA_DR_JSON = "/home/vina04/ADAR/evaluation/VisA-DR/visa_dr.json"
DEFAULT_CASES_JSON = (
    "/home/vina04/ADAR/Deco-main/runs/"
    "six_datasets_0727_rerun_20260806_081642/visa/up_cases.json"
)


SOTA_TABLE4 = {
    "GPT-4V": {
        "acc": 0.68,
        "precision": 0.90,
        "recall": 0.49,
        "f1": 0.55,
        "low_rouge_l": 0.16,
        "low_sbert": 0.65,
        "low_gpt_score": 3.31,
        "complex_sbert": 0.77,
        "complex_gpt_score": 5.64,
    },
    "GPT-4o": {
        "acc": 0.70,
        "precision": 0.83,
        "recall": 0.71,
        "f1": 0.68,
        "low_rouge_l": 0.24,
        "low_sbert": 0.71,
        "low_gpt_score": 4.84,
        "complex_sbert": 0.81,
        "complex_gpt_score": 6.89,
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
    "Anomaly-OV-0.5B": {
        "acc": 0.79,
        "precision": 0.86,
        "recall": 0.83,
        "f1": 0.82,
        "low_rouge_l": 0.33,
        "low_sbert": 0.72,
        "low_gpt_score": 3.87,
        "complex_sbert": 0.83,
        "complex_gpt_score": 5.67,
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


FIXED_ANSWER_PREFIX = """A separate anomaly scoring module has fixed the binary decision for this image as {fixed_answer}.
Use this fixed decision as prior context and do not contradict it.
Do not output only the label {fixed_answer}; answer the benchmark question in natural language.
If the fixed decision is NORMAL, explain that no visible anomaly can be confidently identified.
If the fixed decision is ANOMALOUS, describe the visible anomaly and answer the question.
Keep the answer concise: no more than 60 words.
Stop after the answer. Do not repeat phrases or restate the instruction.
"""


GPT_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for industrial anomaly reasoning.

You will receive a benchmark question, a standard ground-truth answer, and a model prediction.
Score how well the prediction matches the ground truth for the given question.

Use a 0 to 10 scale:
10 = semantically equivalent to the ground truth and complete.
7-9 = mostly correct with minor missing details.
4-6 = partially correct but incomplete or imprecise.
1-3 = mostly wrong, contradictory, or hallucinated.
0 = irrelevant, empty, or no meaningful answer.

Return exactly one JSON object:
{"score": <number>, "reason": "<brief explanation>"}
"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.6f}"


def parse_int_range(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        return []
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def norm_rel_image(path: str, data_root: str) -> str:
    text = str(path).replace("\\", "/")
    marker = str(data_root).replace("\\", "/").rstrip("/") + "/"
    if text.startswith(marker):
        return text[len(marker) :]
    if "/ODinW-13/data/" in text:
        return text.split("/ODinW-13/data/", 1)[1]
    if text.startswith("VisA/"):
        return text
    idx = text.find("/VisA/")
    if idx >= 0:
        return text[idx + 1 :]
    return text


def category_from_rel(path: str) -> str:
    parts = norm_rel_image(path, "").split("/")
    if len(parts) < 2 or parts[0] != "VisA":
        raise ValueError(f"cannot parse VisA category from path: {path}")
    return parts[1]


def case_score(case: Dict[str, Any], score_key: str) -> float:
    probe = case.get("probe") or case.get("hallucination_probe") or {}
    if score_key in probe:
        return float(probe[score_key])
    if score_key in case:
        return float(case[score_key])
    raise KeyError(f"score key {score_key!r} not found in case")


def select_youden_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    records = [(int(y), float(s)) for y, s in zip(labels, scores)]
    unique_scores = sorted({s for _, s in records})
    if not unique_scores:
        return 0.0
    candidates = [unique_scores[0] - 1e-12, *unique_scores, unique_scores[-1] + 1e-12]
    best_threshold = candidates[0]
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        tp = sum(1 for y, s in records if y == 1 and s >= threshold)
        tn = sum(1 for y, s in records if y == 0 and s < threshold)
        fp = sum(1 for y, s in records if y == 0 and s >= threshold)
        fn = sum(1 for y, s in records if y == 1 and s < threshold)
        tpr = safe_div(tp, tp + fn) or 0.0
        fpr = safe_div(fp, fp + tn) or 0.0
        precision = safe_div(tp, tp + fp) or 0.0
        f1 = safe_div(2 * precision * tpr, precision + tpr) or 0.0
        value = tpr - fpr
        if value > best_value or (value == best_value and f1 > best_tiebreak):
            best_value = value
            best_tiebreak = f1
            best_threshold = threshold
    return float(best_threshold)


def load_detection_cases(args: argparse.Namespace, categories: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    raw_cases = read_json(Path(args.cases_json))
    category_set = set(categories)
    cases: List[Dict[str, Any]] = []
    for raw in raw_cases:
        image = str(raw.get("image") or raw.get("image_path"))
        rel = norm_rel_image(image, args.data_root)
        category = str(raw.get("category") or category_from_rel(rel))
        if category not in category_set:
            continue
        item = dict(raw)
        item["image"] = image
        item["rel_image"] = rel
        item["category"] = category
        item["label"] = int(item["label"])
        cases.append(item)

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case["category"])].append(case)

    thresholds: Dict[str, float] = {}
    for category, rows in sorted(by_category.items()):
        thresholds[category] = select_youden_threshold(
            [int(r["label"]) for r in rows],
            [case_score(r, args.score_key) for r in rows],
        )

    restored: List[Dict[str, Any]] = []
    for case in cases:
        score = case_score(case, args.score_key)
        threshold = thresholds[str(case["category"])]
        pred = 1 if score >= threshold else 0
        item = dict(case)
        item["score"] = score
        item["youden_threshold"] = threshold
        item["fixed_pred_binary"] = pred
        item["fixed_answer"] = "ANOMALOUS" if pred else "NORMAL"
        item["ground_truth"] = "ANOMALOUS" if int(item["label"]) else "NORMAL"
        restored.append(item)
    return restored, thresholds


def class_macro_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["category"])].append(row)

    per_class = []
    for category, group in sorted(buckets.items()):
        m = binary_metrics(group)
        m["category"] = category
        per_class.append(m)

    def avg(key: str) -> Optional[float]:
        values = [float(row[key]) for row in per_class if row.get(key) != ""]
        return sum(values) / len(values) if values else None

    return {
        "n": sum(int(row["n"]) for row in per_class),
        "num_classes": len(per_class),
        "acc": fmt(avg("acc")),
        "precision": fmt(avg("precision")),
        "recall": fmt(avg("recall")),
        "f1": fmt(avg("f1")),
    }


def binary_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in rows if int(r["label"]) == 1 and int(r["fixed_pred_binary"]) == 1)
    tn = sum(1 for r in rows if int(r["label"]) == 0 and int(r["fixed_pred_binary"]) == 0)
    fp = sum(1 for r in rows if int(r["label"]) == 0 and int(r["fixed_pred_binary"]) == 1)
    fn = sum(1 for r in rows if int(r["label"]) == 1 and int(r["fixed_pred_binary"]) == 0)
    total = tp + tn + fp + fn
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * (precision or 0.0) * (recall or 0.0), (precision or 0.0) + (recall or 0.0))
    return {
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


def load_visa_dr_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data = read_json(Path(args.visa_dr_json))
    skip = set(args.skip_classes)
    items: List[Dict[str, Any]] = []
    for item in data:
        rel = norm_rel_image(str(item["image"]), args.data_root)
        category = category_from_rel(rel)
        if category in skip:
            continue
        row = dict(item)
        row["rel_image"] = rel
        row["category"] = category
        row["abs_image"] = str(Path(args.data_root) / rel)
        items.append(row)
    return items


def select_items_for_run(args: argparse.Namespace, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply deterministic evaluation subsetting.

    --limit-per-category is preferred for quick comparisons because the raw
    VisA-D&R annotation is category-ordered; plain --limit would otherwise only
    sample the first categories.
    """

    if args.limit_per_category and args.limit_per_category > 0:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            buckets[str(item["category"])].append(item)
        selected: List[Dict[str, Any]] = []
        for category in sorted(buckets):
            selected.extend(buckets[category][: args.limit_per_category])
        if args.limit and args.limit > 0:
            selected = selected[: args.limit]
        return selected
    if args.limit and args.limit > 0:
        return list(items[: args.limit])
    return list(items)


def filter_rows_to_items(rows: Sequence[Dict[str, Any]], items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = {str(item["index"]) for item in items}
    return [row for row in rows if str(row.get("index")) in selected]


def map_fixed_answers(cases: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(c["rel_image"]): c for c in cases}


def fixed_question(question: str, fixed_answer: str, include_image: bool = False) -> str:
    prefix = FIXED_ANSWER_PREFIX.format(fixed_answer=fixed_answer).strip()
    if include_image:
        return f"{prefix}\n\n<image>\n{question}"
    return f"{prefix}\n\n{question}"


def build_messages(item: Dict[str, Any], question_index: int, fixed_answer: str, previous_responses: Sequence[str]) -> List[Dict[str, Any]]:
    questions = list(item["questions"])
    messages: List[Dict[str, Any]] = []
    for i in range(question_index):
        if i == 0:
            content = [{"type": "image"}, {"type": "text", "text": fixed_question(questions[i], fixed_answer)}]
        else:
            content = [{"type": "text", "text": fixed_question(questions[i], fixed_answer)}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(previous_responses[i])}]})

    if question_index == 0:
        content = [{"type": "image"}, {"type": "text", "text": fixed_question(questions[question_index], fixed_answer)}]
    else:
        content = [{"type": "text", "text": fixed_question(questions[question_index], fixed_answer)}]
    messages.append({"role": "user", "content": content})
    return messages


def load_image(path: str, image_size: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image_size and image_size > 0:
        image = image.resize((image_size, image_size), Image.LANCZOS)
    return image


def load_image_official(path: str, max_side: int):
    from PIL import Image

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


def batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def load_qwen(args: argparse.Namespace) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return processor, model


def generate_reasoning(args: argparse.Namespace, items: Sequence[Dict[str, Any]], fixed_by_rel: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    import torch

    predictions_path = out_dir / "reasoning_predictions.jsonl"
    if args.resume and predictions_path.exists():
        rows = read_jsonl(predictions_path)
        done = {str(row.get("index")) for row in rows}
        if len(done) >= len(items):
            print(f"[GEN] resume: all {len(rows)} predictions already exist", flush=True)
            return rows
    else:
        rows = []
        done = set()

    todo = [item for item in items if str(item.get("index")) not in done]
    print(f"[GEN] total={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return rows

    processor, model = load_qwen(args)
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    current: Dict[str, Dict[str, Any]] = {
        str(item["index"]): {
            "index": str(item["index"]),
            "category": item["category"],
            "image": item["rel_image"],
            "questions": item["questions"],
            "answers": item["answers"],
            "responses": [],
        }
        for item in todo
    }

    max_questions = max(len(item["questions"]) for item in todo)
    for q_idx in range(max_questions):
        active = [item for item in todo if q_idx < len(item["questions"])]
        if not active:
            continue
        print(f"[GEN] question={q_idx + 1} active={len(active)}", flush=True)
        num_batches = math.ceil(len(active) / max(1, args.batch_size))
        for batch_i, chunk in enumerate(batch_iter(active, args.batch_size), start=1):
            if batch_i == 1 or batch_i % args.log_every_batches == 0 or batch_i == num_batches:
                print(f"[GEN] question={q_idx + 1} batch={batch_i}/{num_batches}", flush=True)
            texts: List[str] = []
            images = []
            meta: List[Tuple[Dict[str, Any], str, str]] = []
            for item in chunk:
                rel = str(item["rel_image"])
                fixed_case = fixed_by_rel.get(rel)
                if fixed_case is None:
                    raise KeyError(f"missing fixed answer for {rel}")
                fixed_answer = str(fixed_case["fixed_answer"])
                previous = list(current[str(item["index"])]["responses"])
                messages = build_messages(item, q_idx, fixed_answer, previous)
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                if args.official_image_preprocess:
                    images.append(load_image_official(item["abs_image"], args.max_image_side))
                else:
                    images.append(load_image(item["abs_image"], args.image_size))
                meta.append((item, rel, fixed_answer))

            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            start = time.time()
            with torch.inference_mode():
                generate_kwargs: Dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "temperature": 0,
                }
                if args.repetition_penalty and args.repetition_penalty != 1.0:
                    generate_kwargs["repetition_penalty"] = args.repetition_penalty
                if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
                    generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
                if args.use_deco_generation:
                    generate_kwargs.update(
                        {
                            "use_deco": True,
                            "alpha": args.deco_alpha,
                            "threshold_top_p": args.deco_threshold_top_p,
                            "threshold_top_k": args.deco_threshold_top_k,
                            "early_exit_layers": parse_int_range(args.deco_layers),
                        }
                    )
                output_ids = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            generated_ids = output_ids[:, input_len:]
            outputs = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            seconds_per_item = (time.time() - start) / max(1, len(chunk))
            for (item, rel, fixed_answer), raw, gen_ids in zip(meta, outputs, generated_ids):
                token_count = int((gen_ids != pad_token_id).sum().item()) if pad_token_id is not None else int(gen_ids.numel())
                record = current[str(item["index"])]
                record["responses"].append(raw.strip())
                record["fixed_answer"] = fixed_answer
                record["fixed_pred_binary"] = 1 if fixed_answer == "ANOMALOUS" else 0
                record["ground_truth"] = "ANOMALOUS"
                record.setdefault("raw_output_token_counts", []).append(token_count)
                record.setdefault("hit_max_new_tokens", []).append(bool(token_count >= args.max_new_tokens))
                record.setdefault("seconds_per_question", []).append(round(seconds_per_item, 4))
                record["use_deco_generation"] = bool(args.use_deco_generation)
                record["deco_alpha"] = args.deco_alpha if args.use_deco_generation else ""
                record["deco_threshold_top_p"] = args.deco_threshold_top_p if args.use_deco_generation else ""
                record["deco_threshold_top_k"] = args.deco_threshold_top_k if args.use_deco_generation else ""
                record["deco_layers"] = args.deco_layers if args.use_deco_generation else ""

        checkpoint_rows = [current[str(item["index"])] for item in todo]
        write_json(out_dir / "reasoning_generation_checkpoint.json", checkpoint_rows)

    new_rows = [current[str(item["index"])] for item in todo]
    all_rows = rows + new_rows
    write_jsonl(predictions_path, all_rows)
    return all_rows


class OfficialLikeTextProcessor:
    """Text normalization copied from the visible Anomaly-OneVision eval helper."""

    def __init__(self) -> None:
        self.period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def process_punctuation(self, text: str) -> str:
        out = text
        for p in self.punct:
            if (p + " " in text or " " + p in text) or (re.search(self.comma_strip, text) is not None):
                out = out.replace(p, "")
            else:
                out = out.replace(p, " ")
        out = self.period_strip.sub("", out, re.UNICODE)
        return out

    def process(self, answer: str) -> str:
        answer = str(answer or "").replace("\n", " ").replace("\t", " ").strip()
        answer = self.process_punctuation(answer)
        answer = answer.strip("'").strip('"').strip(")").strip("(")
        return answer.strip().lower()


TEXT_PROCESSOR = OfficialLikeTextProcessor()
_TOKEN_RE = re.compile(r"\S+")


def tokenize_text(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text))


def rouge_l_f1_lcs(pred: str, ref: str) -> float:
    x = tokenize_text(pred)
    y = tokenize_text(ref)
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for token in x:
        curr = [0] * (len(y) + 1)
        for j, ref_token in enumerate(y, start=1):
            if token == ref_token:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(x)
    recall = lcs / len(y)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l_score(args: argparse.Namespace, pred: str, ref: str) -> Tuple[float, str]:
    gt_ans = TEXT_PROCESSOR.process(ref)
    pred_ans = TEXT_PROCESSOR.process(pred)
    if gt_ans == "":
        return 0.0, "empty_reference"
    if pred_ans == "":
        return 0.0, "empty_prediction"
    if len(pred_ans) > args.rouge_truncate_chars:
        pred_ans = pred_ans[: args.rouge_truncate_chars]

    if args.rouge_backend in {"auto", "rouge"}:
        try:
            from rouge import Rouge  # type: ignore

            score = Rouge().get_scores(pred_ans, gt_ans)[0]["rouge-l"]["f"]
            return float(score), "rouge"
        except Exception:
            if args.rouge_backend == "rouge":
                raise

    if args.rouge_backend in {"auto", "rouge-score"}:
        try:
            from rouge_score import rouge_scorer  # type: ignore

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=args.rouge_use_stemmer)
            score = scorer.score(gt_ans, pred_ans)["rougeL"].fmeasure
            return float(score), "rouge-score"
        except Exception:
            if args.rouge_backend == "rouge-score":
                raise

    return rouge_l_f1_lcs(pred_ans, gt_ans), "internal_lcs"


def collect_pairs(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    low_pairs: List[Dict[str, Any]] = []
    complex_pairs: List[Dict[str, Any]] = []
    for row in rows:
        responses = list(row.get("responses") or [])
        answers = list(row.get("answers") or [])
        questions = list(row.get("questions") or [])
        n = min(len(responses), len(answers), len(questions))
        for i in range(n):
            pair = {
                "index": row.get("index", ""),
                "category": row.get("category", ""),
                "image": row.get("image", ""),
                "question_index": i,
                "question": questions[i],
                "prediction": responses[i],
                "reference": answers[i],
            }
            if i <= 1:
                low_pairs.append(pair)
            else:
                complex_pairs.append(pair)
    return low_pairs, complex_pairs


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def sbert_scores(args: argparse.Namespace, pairs: Sequence[Dict[str, Any]]) -> Optional[List[float]]:
    if not pairs:
        return []
    if args.no_sbert:
        return None
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.sbert_model)
    model = AutoModel.from_pretrained(args.sbert_model)
    device = "cuda" if torch.cuda.is_available() and not args.sbert_cpu else "cpu"
    model.to(device)
    model.eval()

    def encode(texts: Sequence[str]):
        vectors = []
        for chunk in batch_iter(list(texts), args.sbert_batch_size):
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
        return torch.cat(vectors, dim=0)

    pred_vec = encode([p["prediction"] for p in pairs])
    ref_vec = encode([p["reference"] for p in pairs])
    return (pred_vec * ref_vec).sum(dim=1).tolist()


def evaluate_reasoning(args: argparse.Namespace, rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    low_pairs, complex_pairs = collect_pairs(rows)
    low_rouge_pairs = [rouge_l_score(args, p["prediction"], p["reference"]) for p in low_pairs]
    complex_rouge_pairs = [rouge_l_score(args, p["prediction"], p["reference"]) for p in complex_pairs]
    low_rouge = [score for score, _ in low_rouge_pairs]
    complex_rouge = [score for score, _ in complex_rouge_pairs]
    rouge_backends = Counter([backend for _, backend in low_rouge_pairs + complex_rouge_pairs])

    low_sbert = sbert_scores(args, low_pairs)
    complex_sbert = sbert_scores(args, complex_pairs)

    for pair, (score, backend) in zip(low_pairs, low_rouge_pairs):
        pair["rouge_l"] = score
        pair["rouge_backend"] = backend
    for pair, (score, backend) in zip(complex_pairs, complex_rouge_pairs):
        pair["rouge_l"] = score
        pair["rouge_backend"] = backend
    if low_sbert is not None:
        for pair, score in zip(low_pairs, low_sbert):
            pair["sbert"] = score
    if complex_sbert is not None:
        for pair, score in zip(complex_pairs, complex_sbert):
            pair["sbert"] = score

    write_jsonl(out_dir / "low_level_pairs.jsonl", low_pairs)
    write_jsonl(out_dir / "complex_pairs.jsonl", complex_pairs)

    return {
        "low_pairs": len(low_pairs),
        "complex_pairs": len(complex_pairs),
        "low_rouge_l": fmt(mean(low_rouge)),
        "rouge_backend_counts": dict(rouge_backends),
        "low_sbert": fmt(mean(low_sbert) if low_sbert is not None else None),
        "sbert_model": "" if args.no_sbert else args.sbert_model,
        "low_gpt_score": "",
        "complex_sbert": fmt(mean(complex_sbert) if complex_sbert is not None else None),
        "complex_gpt_score": "",
        "complex_rouge_l_aux": fmt(mean(complex_rouge)),
    }


def write_gpt_judge_pairs(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    low_pairs, complex_pairs = collect_pairs(rows)
    judge_rows = []
    for level, pairs in [("low_level", low_pairs), ("complex", complex_pairs)]:
        for pair in pairs:
            judge_rows.append(
                {
                    **pair,
                    "level": level,
                    "judge_instruction": (
                        "Score the prediction against the reference answer from 0 to 10. "
                        "Focus on semantic correctness for the benchmark question. "
                        "Return only a number."
                    ),
                }
            )
    write_jsonl(out_dir / "gpt_score_judge_pairs.jsonl", judge_rows)


def parse_gpt_score(text: str) -> Tuple[Optional[float], str]:
    raw = str(text or "").strip()
    try:
        obj = json.loads(raw)
        score = float(obj.get("score"))
        reason = str(obj.get("reason", ""))
        return max(0.0, min(10.0, score)), reason
    except Exception:
        pass
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return None, raw[:200]
    score = float(match.group(0))
    return max(0.0, min(10.0, score)), raw[:200]


def run_gpt_judge(args: argparse.Namespace, out_dir: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from openai import OpenAI  # type: ignore

    write_gpt_judge_pairs(out_dir, rows)
    pairs_path = out_dir / "gpt_score_judge_pairs.jsonl"
    results_path = out_dir / "gpt_score_results.jsonl"
    pairs = read_jsonl(pairs_path)
    done_rows = read_jsonl(results_path) if args.resume else []
    done = {
        (str(r.get("level")), str(r.get("index")), int(r.get("question_index", -1)))
        for r in done_rows
        if r.get("score") != ""
    }

    client = OpenAI()
    new_rows: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs, start=1):
        key = (str(pair.get("level")), str(pair.get("index")), int(pair.get("question_index", -1)))
        if key in done:
            continue
        if i == 1 or i % args.gpt_judge_log_every == 0 or i == len(pairs):
            print(f"[GPT-JUDGE] {i}/{len(pairs)}", flush=True)
        user_prompt = (
            f"Question:\n{pair['question']}\n\n"
            f"Ground truth answer:\n{pair['reference']}\n\n"
            f"Model prediction:\n{pair['prediction']}\n"
        )
        response = client.chat.completions.create(
            model=args.gpt_judge_model,
            messages=[
                {"role": "system", "content": GPT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=args.gpt_judge_max_tokens,
        )
        raw = response.choices[0].message.content or ""
        score, judge_reason = parse_gpt_score(raw)
        row = {
            **pair,
            "gpt_judge_model": args.gpt_judge_model,
            "gpt_judge_prompt_version": args.gpt_judge_prompt_version,
            "raw_judge_output": raw,
            "score": "" if score is None else score,
            "judge_reason": judge_reason,
        }
        new_rows.append(row)
        if len(new_rows) >= args.gpt_judge_flush_every:
            with results_path.open("a", encoding="utf-8") as f:
                for r in new_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            new_rows = []

    if new_rows:
        with results_path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_rows = read_jsonl(results_path)
    low_scores = [float(r["score"]) for r in all_rows if r.get("level") == "low_level" and r.get("score") != ""]
    complex_scores = [float(r["score"]) for r in all_rows if r.get("level") == "complex" and r.get("score") != ""]
    return {
        "low_gpt_score": fmt(mean(low_scores)),
        "complex_gpt_score": fmt(mean(complex_scores)),
        "gpt_judge_model": args.gpt_judge_model,
        "gpt_judge_prompt_version": args.gpt_judge_prompt_version,
        "gpt_judge_n": len(low_scores) + len(complex_scores),
    }


def comparison_rows(ours: Dict[str, Any], method_name: str = "Ours-HDEC+Qwen3Reason") -> List[Dict[str, Any]]:
    rows = []
    metric_keys = [
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
    rows.append({"method": method_name, **{k: ours.get(k, "") for k in metric_keys}})
    for name, vals in SOTA_TABLE4.items():
        row = {"method": name}
        row.update({k: vals.get(k, "") for k in metric_keys})
        rows.append(row)
    sota = SOTA_TABLE4["Anomaly-OV-7B"]
    delta = {"method": "Ours - Anomaly-OV-7B"}
    for k in metric_keys:
        try:
            delta[k] = fmt(float(ours[k]) - float(sota[k])) if ours.get(k) != "" and sota.get(k) != "" else ""
        except Exception:
            delta[k] = ""
    rows.append(delta)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ADAR on VisA-D&R detection/reasoning.")
    parser.add_argument(
        "--generation-profile",
        choices=["fast_fixed", "official_fixed"],
        default="fast_fixed",
        help=(
            "fast_fixed keeps the shorter ADAR practical generation settings. "
            "official_fixed uses fixed Answer plus the visible official test_reasoning.py settings: "
            "single-image sequential generation, max side 1024, max_new_tokens 4096, temperature 0."
        ),
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--visa-dr-json", default=DEFAULT_VISA_DR_JSON)
    parser.add_argument("--cases-json", default=DEFAULT_CASES_JSON)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--score-key", default="hdec_score")
    parser.add_argument("--skip-classes", nargs="+", default=DEFAULT_SKIP_CLASSES)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--official-image-preprocess", action="store_true")
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--use-deco-generation", action="store_true", help="Run Qwen3-VL generation with DeCo decoding.")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument("--deco-layers", default="20:29", help="Early-exit layers for DeCo generation; 20:29 means layers 20..28.")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=0,
        help="Deterministically keep the first N VisA-D&R anomaly images per category; useful for balanced smoke tests.",
    )
    parser.add_argument("--generate", action="store_true", help="Run Qwen3-VL generation if predictions do not exist.")
    parser.add_argument("--evaluate-only", action="store_true", help="Do not run generation; evaluate existing predictions.")
    parser.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    parser.add_argument("--rouge-truncate-chars", type=int, default=512)
    parser.add_argument("--rouge-use-stemmer", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-cpu", action="store_true")
    parser.add_argument("--write-gpt-judge-pairs", action="store_true", default=True)
    parser.add_argument("--gpt-score", action="store_true", help="Call an OpenAI-compatible GPT judge to compute GPT-Score.")
    parser.add_argument("--gpt-judge-model", default="gpt-4o")
    parser.add_argument("--gpt-judge-prompt-version", default="adar_v1")
    parser.add_argument("--gpt-judge-max-tokens", type=int, default=128)
    parser.add_argument("--gpt-judge-log-every", type=int, default=100)
    parser.add_argument("--gpt-judge-flush-every", type=int, default=50)
    args = parser.parse_args()
    if args.generation_profile == "official_fixed":
        args.official_image_preprocess = True
        args.image_size = 0
        args.max_image_side = 1024
        args.batch_size = 1
        args.max_new_tokens = 4096
        args.repetition_penalty = 1.0
        args.no_repeat_ngram_size = 0
        args.log_every_batches = 100
    return args


def main() -> None:
    args = parse_args()
    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"/home/vina04/ADAR/Deco-main/runs/visa_dr_reasoning_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = select_items_for_run(args, load_visa_dr_items(args))
    categories = sorted({item["category"] for item in items})
    detection_cases, thresholds = load_detection_cases(args, categories)
    fixed_by_rel = map_fixed_answers(detection_cases)

    missing = [item["rel_image"] for item in items if item["rel_image"] not in fixed_by_rel]
    if missing:
        raise RuntimeError(f"{len(missing)} VisA-D&R images are missing in cases_json; first={missing[:5]}")

    write_json(out_dir / "run_config.json", vars(args))
    write_json(out_dir / "youden_thresholds.json", thresholds)
    write_jsonl(out_dir / "detection_cases.jsonl", detection_cases)

    micro = binary_metrics(detection_cases)
    macro = class_macro_metrics(detection_cases)
    detection_summary = [
        {"scope": "class_macro_table4_aligned", **macro},
        {"scope": "micro_aux", **micro},
    ]
    write_csv(out_dir / "detection_summary.csv", detection_summary)

    predictions_path = out_dir / "reasoning_predictions.jsonl"
    if args.evaluate_only:
        rows = read_jsonl(predictions_path)
        if not rows:
            raise FileNotFoundError(f"missing predictions for --evaluate-only: {predictions_path}")
    elif args.generate or not predictions_path.exists():
        rows = generate_reasoning(args, items, fixed_by_rel, out_dir)
    else:
        rows = read_jsonl(predictions_path)
    rows = filter_rows_to_items(rows, items)

    reasoning = evaluate_reasoning(args, rows, out_dir)
    if args.gpt_score:
        gpt_metrics = run_gpt_judge(args, out_dir, rows)
        reasoning.update(gpt_metrics)
    method_name = "Ours-HDEC+Qwen3Reason-DeCo" if args.use_deco_generation else "Ours-HDEC+Qwen3Reason-Native"
    ours = {
        **{k: macro[k] for k in ["acc", "precision", "recall", "f1"]},
        **reasoning,
    }
    write_json(out_dir / "ours_summary.json", ours)
    write_csv(out_dir / "ours_summary.csv", [{"method": method_name, **ours}])
    comp = comparison_rows(ours, method_name)
    write_csv(out_dir / "table4_comparison.csv", comp)

    if args.write_gpt_judge_pairs and not args.gpt_score:
        write_gpt_judge_pairs(out_dir, rows)

    print("[DONE]", out_dir, flush=True)
    print(json.dumps({"detection_macro": macro, "reasoning": reasoning}, indent=2), flush=True)


if __name__ == "__main__":
    main()
