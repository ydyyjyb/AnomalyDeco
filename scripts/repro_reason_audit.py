#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproducible fixed-answer Reason audit for ADAR/Deco main-method results.

This script intentionally does not recompute the main anomaly score and does
not use Reason text to correct the main metrics.  It reads existing
``up_cases.json`` files, reconstructs the fixed Answer from ``E_raw_hdec`` scores
with category-wise Youden thresholds, asks Qwen3-VL to generate a structured
post-hoc Reason for that fixed Answer, and reports fixed-answer metrics plus
Answer-Reason consistency statistics.

Expected input layout by default:

  zero-shot industrial:
    <zero_industrial_root>/<dataset>/up_cases.json

  zero-shot medical:
    <zero_medical_root>/<dataset>/up_cases.json

  few-shot:
    <fewshot_root>/<shot>shot/<dataset>/up_cases.json

Positive class is ANOMALOUS. Prediction rule is:

  fixed_answer = ANOMALOUS if hdec_score >= category_youden_threshold else NORMAL
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = ["visa", "mvtec", "aitex", "btad", "elpv", "mpdd", "brainmri", "headct", "br35h"]
INDUSTRIAL_ZERO_DATASETS = {"visa", "mvtec", "aitex", "btad", "elpv", "mpdd"}
MEDICAL_DATASETS = {"brainmri", "headct", "br35h"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMAL_DIR_NAMES = {"good", "normal", "ok", "no_defect", "nodefect", "no-defect"}

MEDICAL_DEFECT_LABELS = {
    "brainmri": [
        "brain tumor",
        "intracranial mass",
        "abnormal brain lesion",
        "tumor tissue",
        "mass effect",
        "abnormal signal intensity",
    ],
    "headct": [
        "intracranial hemorrhage",
        "brain bleeding",
        "hyperdense hemorrhagic region",
        "epidural hemorrhage",
        "subdural hemorrhage",
        "subarachnoid hemorrhage",
    ],
    "br35h": [
        "brain tumor",
        "intracranial mass",
        "abnormal brain lesion",
        "tumor tissue",
        "mass effect",
    ],
}

GENERIC_DEFECT_LABELS = [
    "surface defect",
    "scratch",
    "dent",
    "contamination",
    "deformation",
    "damaged part",
    "missing part",
    "hole",
    "stain",
    "dark spot",
    "color anomaly",
    "texture anomaly",
]

AITEX_DEFECT_LABELS = [
    "fabric defect",
    "weaving defect",
    "thread defect",
    "hole",
    "stain",
    "broken yarn",
    "slub",
    "knot",
    "contamination",
    "texture anomaly",
]

ELPV_DEFECT_LABELS = [
    "solar cell defect",
    "crack",
    "micro crack",
    "finger interruption",
    "inactive area",
    "dark defect",
    "broken cell",
    "contamination",
    "solder defect",
    "edge defect",
]

PROMPT_TEMPLATE = """You are producing a post-hoc visual explanation for an anomaly scoring module.

The final Answer is already fixed by the scoring module:
{fixed_answer}

Do not re-decide the class. Do not change the fixed Answer.

Object/category: {category}
NORMAL means no visible defect-like abnormality is present in the query image.
ANOMALOUS means a visible defect-like abnormality is present in the query image.
{defect_line}{reference_line}

Your task:
1. Write a concise visual Reason for the fixed Answer using only image evidence.
2. If the visible evidence is weak or ambiguous, say so in the reason, but keep fixed_answer unchanged.
3. Return exactly one valid JSON object and no markdown.

Required JSON schema, using ASCII double quotes only:
{{
  "fixed_answer": "{fixed_answer}",
  "reason": "one or two concise sentences grounded in visible evidence",
  "reason_supports_fixed_answer": "YES or NO or UNCERTAIN",
  "visual_evidence_polarity": "NORMAL_EVIDENCE or ANOMALOUS_EVIDENCE or UNCERTAIN",
  "mentioned_opposite_evidence": false
}}

Allowed values:
- fixed_answer must be exactly "{fixed_answer}".
- reason_supports_fixed_answer must be one of: YES, NO, UNCERTAIN.
- visual_evidence_polarity must be one of: NORMAL_EVIDENCE, ANOMALOUS_EVIDENCE, UNCERTAIN.
- mentioned_opposite_evidence must be true or false.
"""


def safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den > 0 else None


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.6f}"


def select_youden_threshold(labels: Sequence[int], scores: Sequence[float], fallback: float = 0.0) -> float:
    """Select a category-level threshold by maximizing Youden's J.

    With positive class ANOMALOUS and prediction ``score >= threshold``, this is
    equivalent to maximizing TPR - FPR.  The implementation uses balanced
    accuracy ``(TPR + TNR) / 2`` as the objective, which has the same argmax.
    Ties are broken by anomaly-class F1 to keep the result deterministic.
    """

    records = [(int(y), float(s)) for y, s in zip(labels, scores)]
    if not records:
        return float(fallback)
    unique_scores = sorted({score for _, score in records})
    if not unique_scores:
        return float(fallback)

    candidates = [unique_scores[0] - 1e-12]
    candidates.extend(unique_scores)
    candidates.append(unique_scores[-1] + 1e-12)

    best_threshold = candidates[0]
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        tp = sum(1 for y, s in records if y == 1 and s >= threshold)
        tn = sum(1 for y, s in records if y == 0 and s < threshold)
        fp = sum(1 for y, s in records if y == 0 and s >= threshold)
        fn = sum(1 for y, s in records if y == 1 and s < threshold)
        tpr = safe_div(tp, tp + fn) or 0.0
        tnr = safe_div(tn, tn + fp) or 0.0
        precision = safe_div(tp, tp + fp) or 0.0
        f1 = safe_div(2.0 * precision * tpr, precision + tpr) or 0.0
        value = (tpr + tnr) / 2.0
        if value > best_value or (value == best_value and f1 > best_tiebreak):
            best_threshold = threshold
            best_value = value
            best_tiebreak = f1
    return float(best_threshold)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def case_score(case: Dict[str, Any], score_key: str) -> float:
    probe = case.get("probe") or case.get("hallucination_probe") or {}
    if score_key in probe:
        return float(probe[score_key])
    if score_key in case:
        return float(case[score_key])
    raise KeyError(f"score key {score_key!r} not found in case")


def source_cases_path(args: argparse.Namespace, dataset: str, shot: int) -> Path:
    if shot == 0:
        if dataset in INDUSTRIAL_ZERO_DATASETS:
            return Path(args.zero_industrial_root) / dataset / "up_cases.json"
        if dataset in MEDICAL_DATASETS:
            return Path(args.zero_medical_root) / dataset / "up_cases.json"
    return Path(args.fewshot_root) / f"{shot}shot" / dataset / "up_cases.json"


def load_cases_for_combo(args: argparse.Namespace, dataset: str, shot: int) -> List[Dict[str, Any]]:
    path = source_cases_path(args, dataset, shot)
    if not path.exists():
        raise FileNotFoundError(f"missing source cases: {path}")
    data = read_json(path)
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(data):
        item = dict(case)
        item["dataset"] = dataset
        item["shot"] = shot
        item["source_cases_path"] = str(path)
        item.setdefault("image_id", str(idx))
        if "image" not in item and "image_path" in item:
            item["image"] = item["image_path"]
        if "image" not in item:
            raise KeyError(f"case {idx} in {path} has no image path")
        rows.append(item)
    if args.limit_per_combo and args.limit_per_combo > 0:
        rows = rows[: args.limit_per_combo]
    return rows


def restore_fixed_answers(cases: List[Dict[str, Any]], score_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case["category"])].append(case)

    thresholds: Dict[str, float] = {}
    for category, rows in sorted(by_category.items()):
        labels = [int(r["label"]) for r in rows]
        scores = [case_score(r, score_key) for r in rows]
        thresholds[category] = select_youden_threshold(labels, scores)

    restored: List[Dict[str, Any]] = []
    for case in cases:
        item = dict(case)
        score = case_score(item, score_key)
        threshold = thresholds[str(item["category"])]
        pred_binary = 1 if score >= threshold else 0
        gt_binary = int(item["label"])
        item["fixed_score_source"] = score_key
        item["fixed_score"] = float(score)
        item["youden_threshold"] = float(threshold)
        item["score_margin"] = float(score) - float(threshold)
        item["fixed_pred_binary"] = pred_binary
        item["fixed_answer"] = "ANOMALOUS" if pred_binary else "NORMAL"
        item["ground_truth"] = "ANOMALOUS" if gt_binary else "NORMAL"
        item["answer_correct"] = pred_binary == gt_binary
        if pred_binary == 1 and gt_binary == 1:
            item["confusion"] = "TP"
        elif pred_binary == 0 and gt_binary == 0:
            item["confusion"] = "TN"
        elif pred_binary == 1 and gt_binary == 0:
            item["confusion"] = "FP"
        else:
            item["confusion"] = "FN"
        restored.append(item)
    return restored, thresholds


def get_dataset_root(data_root: Path, dataset: str) -> Path:
    names = {
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
    return data_root / names[dataset]


def defect_bank_for_case(args: argparse.Namespace, case: Dict[str, Any]) -> List[str]:
    dataset = str(case["dataset"])
    category = str(case["category"])
    root = get_dataset_root(Path(args.data_root), dataset)
    if dataset in MEDICAL_DEFECT_LABELS:
        return MEDICAL_DEFECT_LABELS[dataset]
    if dataset == "aitex":
        return AITEX_DEFECT_LABELS
    if dataset == "elpv":
        return ELPV_DEFECT_LABELS
    if dataset in {"mvtec", "btad", "mpdd"}:
        test_dir = root / category / "test"
        if test_dir.exists():
            labels = [
                p.name.replace("_", " ")
                for p in sorted(test_dir.iterdir())
                if p.is_dir() and p.name.lower() not in NORMAL_DIR_NAMES
            ]
            return labels[: args.defect_limit] if labels else GENERIC_DEFECT_LABELS
    return GENERIC_DEFECT_LABELS


def prompt_defect_text(defects: Sequence[str], limit: int) -> str:
    seen = set()
    out: List[str] = []
    for item in defects:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
        if len(out) >= limit:
            break
    return ", ".join(out)


def build_reason_messages(args: argparse.Namespace, case: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixed_answer = str(case["fixed_answer"])
    category = str(case["category"]).replace("_", " ")
    refs = list(case.get("reference_images") or [])
    defects = prompt_defect_text(defect_bank_for_case(args, case), args.defect_limit)
    defect_line = ""
    if defects:
        defect_line = f"Possible defect examples for this category include: {defects}. These are examples, not assumptions.\n"
    reference_line = ""
    if refs:
        reference_line = (
            f"\nYou are given {len(refs)} normal reference image(s) from the same category before the query image. "
            "Use them only as normal-appearance references."
        )

    prompt = PROMPT_TEMPLATE.format(
        fixed_answer=fixed_answer,
        category=category,
        defect_line=defect_line,
        reference_line=reference_line,
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i in range(len(refs)):
        content.append({"type": "text", "text": f"Normal reference image {i + 1}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Query image:"})
    content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def load_image(path: str, size: int) -> Image.Image:
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def batch_iter(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def normalize_json_text(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "：": ":",
        "，": ",",
        "。": ".",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def extract_first_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    normalized = normalize_json_text(text)
    normalized_changed = normalized != text
    decoder = json.JSONDecoder()
    for i, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(normalized[i:])
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(obj, dict):
            prefix = normalized[:i].strip()
            suffix = normalized[i + end :].strip()
            recovered = bool(prefix or suffix or normalized_changed)
            return obj, recovered, ""
    return None, False, locals().get("last_error", "no JSON object found")


def normalize_enum(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def is_degenerate_reason(text: str) -> bool:
    reason = str(text or "").strip()
    if len(reason) < 20:
        return True
    words = re.findall(r"[A-Za-z0-9]+", reason)
    if len(words) < 5:
        return True
    if re.fullmatch(r"([A-Za-z0-9.,;:!?\\-_/ ]{1,8})\\1{4,}", reason):
        return True
    compact = re.sub(r"\s+", "", reason)
    if compact and len(set(compact.lower())) <= 3 and len(compact) >= 20:
        return True
    return False


def validate_reason_output(case: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    expected_answer = str(case["fixed_answer"])
    obj, recovered, parse_error = extract_first_json_object(raw_output)
    if obj is None:
        return {
            "generated_fixed_answer": "",
            "reason": "",
            "reason_supports_fixed_answer": "",
            "visual_evidence_polarity": "",
            "mentioned_opposite_evidence": "",
            "json_parse_ok": False,
            "json_format_recovered": False,
            "validation_status": "json_parse_error",
            "consistency": "invalid_output",
            "validation_error": parse_error,
        }

    generated_answer = normalize_enum(obj.get("fixed_answer"))
    supports = normalize_enum(obj.get("reason_supports_fixed_answer"))
    polarity = normalize_enum(obj.get("visual_evidence_polarity"))
    reason = str(obj.get("reason") or "").strip()
    mentioned_opposite = obj.get("mentioned_opposite_evidence")
    if isinstance(mentioned_opposite, str):
        mentioned_opposite = mentioned_opposite.strip().lower() == "true"

    allowed_supports = {"YES", "NO", "UNCERTAIN"}
    allowed_polarity = {"NORMAL_EVIDENCE", "ANOMALOUS_EVIDENCE", "UNCERTAIN"}
    expected_polarity = "ANOMALOUS_EVIDENCE" if expected_answer == "ANOMALOUS" else "NORMAL_EVIDENCE"
    opposite_polarity = "NORMAL_EVIDENCE" if expected_answer == "ANOMALOUS" else "ANOMALOUS_EVIDENCE"

    status = "ok"
    consistency = "consistent"
    validation_error = ""
    if generated_answer != expected_answer:
        status = "answer_violation"
        consistency = "invalid_output"
        validation_error = f"fixed_answer must be {expected_answer}, got {generated_answer!r}"
    elif supports not in allowed_supports:
        status = "schema_violation"
        consistency = "invalid_output"
        validation_error = f"invalid reason_supports_fixed_answer: {supports!r}"
    elif polarity not in allowed_polarity:
        status = "schema_violation"
        consistency = "invalid_output"
        validation_error = f"invalid visual_evidence_polarity: {polarity!r}"
    elif not isinstance(mentioned_opposite, bool):
        status = "schema_violation"
        consistency = "invalid_output"
        validation_error = "mentioned_opposite_evidence must be boolean"
    elif is_degenerate_reason(reason):
        status = "degenerate_reason"
        consistency = "invalid_output"
        validation_error = "reason is empty, too short, or degenerate"
    elif supports == "YES" and polarity == expected_polarity and not mentioned_opposite:
        status = "ok"
        consistency = "consistent"
    elif supports == "NO" or polarity == opposite_polarity:
        status = "semantic_conflict"
        consistency = "contradiction"
    else:
        status = "uncertain"
        consistency = "uncertain"

    return {
        "generated_fixed_answer": generated_answer,
        "reason": reason,
        "reason_supports_fixed_answer": supports,
        "visual_evidence_polarity": polarity,
        "mentioned_opposite_evidence": mentioned_opposite,
        "json_parse_ok": True,
        "json_format_recovered": bool(recovered),
        "validation_status": status,
        "consistency": consistency,
        "validation_error": validation_error,
    }


def binary_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in rows if r["confusion"] == "TP")
    tn = sum(1 for r in rows if r["confusion"] == "TN")
    fp = sum(1 for r in rows if r["confusion"] == "FP")
    fn = sum(1 for r in rows if r["confusion"] == "FN")
    total = tp + tn + fp + fn
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2.0 * (precision or 0.0) * (recall or 0.0), (precision or 0.0) + (recall or 0.0))
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": format_float(safe_div(tp + tn, total)),
        "recall": format_float(recall),
        "precision": format_float(precision),
        "f1": format_float(f1),
    }


def reason_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)

    def count(key: str, value: Any) -> int:
        return sum(1 for r in rows if r.get(key) == value)

    consistent = count("consistency", "consistent")
    contradiction = count("consistency", "contradiction")
    uncertain = count("consistency", "uncertain")
    invalid = count("consistency", "invalid_output")
    parse_fail = sum(1 for r in rows if not r.get("json_parse_ok"))
    recovered = sum(1 for r in rows if r.get("json_format_recovered"))
    hit_max = sum(1 for r in rows if r.get("hit_max_new_tokens"))
    answer_violation = count("validation_status", "answer_violation")
    return {
        "reason_consistent": consistent,
        "reason_contradiction": contradiction,
        "reason_uncertain": uncertain,
        "reason_invalid": invalid,
        "json_parse_fail": parse_fail,
        "json_format_recovered": recovered,
        "answer_violation": answer_violation,
        "hit_max_new_tokens": hit_max,
        "reason_consistency_rate": format_float(safe_div(consistent, total)),
        "reason_contradiction_rate": format_float(safe_div(contradiction, total)),
        "reason_uncertain_rate": format_float(safe_div(uncertain, total)),
        "reason_invalid_rate": format_float(safe_div(invalid, total)),
        "json_parse_fail_rate": format_float(safe_div(parse_fail, total)),
        "answer_violation_rate": format_float(safe_div(answer_violation, total)),
    }


def summarize(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(k, "") for k in keys)].append(row)

    output: List[Dict[str, Any]] = []
    for values, group in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        base = {k: v for k, v in zip(keys, values)}
        base.update(binary_metrics(group))
        base.update(reason_metrics(group))
        output.append(base)
    return output


def load_done_rows(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    rows = read_jsonl(path)
    done = {
        (str(r.get("dataset")), int(r.get("shot")), str(r.get("image")))
        for r in rows
        if "dataset" in r and "shot" in r and "image" in r
    }
    return rows, done


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    return processor, model


def run_combo(
    args: argparse.Namespace,
    dataset: str,
    shot: int,
    processor: Any,
    model: Any,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    import torch

    combo = f"{dataset}_{shot}shot"
    raw_cases = load_cases_for_combo(args, dataset, shot)
    cases, thresholds = restore_fixed_answers(raw_cases, args.score_key)
    write_json(out_dir / "thresholds" / f"{combo}.json", thresholds)

    cases_path = out_dir / "cases" / f"{combo}.jsonl"
    done_rows, done_keys = load_done_rows(cases_path) if args.resume else ([], set())
    todo = [c for c in cases if (dataset, shot, c["image"]) not in done_keys]
    images_per_case = max(1, 1 + max((len(c.get("reference_images") or []) for c in cases), default=0))
    batch_size = max(1, args.base_batch_size // images_per_case)
    print(f"[COMBO] {combo} total={len(cases)} done={len(done_rows)} todo={len(todo)} batch={batch_size}", flush=True)

    all_rows = list(done_rows)
    for chunk in batch_iter(todo, batch_size):
        texts: List[str] = []
        images: List[Image.Image] = []
        for case in chunk:
            messages = build_reason_messages(args, case)
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            for ref_path in list(case.get("reference_images") or []):
                images.append(load_image(ref_path, args.image_size))
            images.append(load_image(case["image"], args.image_size))

        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        start = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=args._pad_token_id,
                eos_token_id=args._eos_token_id,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        seconds_per_case = (time.time() - start) / max(1, len(chunk))
        generated_ids = output_ids[:, input_len:]
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        new_rows: List[Dict[str, Any]] = []
        for case, raw, gen_ids in zip(chunk, outputs, generated_ids):
            raw = raw.strip()
            token_count = int((gen_ids != args._pad_token_id).sum().item()) if args._pad_token_id is not None else int(gen_ids.numel())
            audit = validate_reason_output(case, raw)
            row = {
                "dataset": dataset,
                "shot": shot,
                "category": case["category"],
                "image": case["image"],
                "image_id": case.get("image_id", ""),
                "label": int(case["label"]),
                "ground_truth": case["ground_truth"],
                "fixed_score_source": case["fixed_score_source"],
                "fixed_score": case["fixed_score"],
                "youden_threshold": case["youden_threshold"],
                "score_margin": case["score_margin"],
                "fixed_answer": case["fixed_answer"],
                "fixed_pred_binary": case["fixed_pred_binary"],
                "answer_correct": bool(case["answer_correct"]),
                "confusion": case["confusion"],
                "reference_image_count": len(case.get("reference_images") or []),
                "reference_images": case.get("reference_images") or [],
                "raw_output": raw,
                "raw_output_token_count": token_count,
                "hit_max_new_tokens": bool(token_count >= args.max_new_tokens),
                "seconds_per_case": round(seconds_per_case, 4),
            }
            row.update(audit)
            new_rows.append(row)
        append_jsonl(cases_path, new_rows)
        all_rows.extend(new_rows)

    write_csv(out_dir / "summaries" / f"{combo}_by_category.csv", summarize(all_rows, ["dataset", "shot", "category"]))
    return all_rows


def make_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    excluded = {"_pad_token_id", "_eos_token_id"}
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in excluded}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible fixed-answer Reason audit for ADAR/Deco results.")
    parser.add_argument("--model-path", default="/home/vina04/ADAR/Deco-main/qwen3_4b_vl_instruct")
    parser.add_argument("--data-root", default="/home/vina04/ADAR/evaluation/ODinW-13/data")
    parser.add_argument("--zero-industrial-root", default="/home/vina04/ADAR/Deco-main/runs/six_datasets_0727_rerun_20260806_081642")
    parser.add_argument("--zero-medical-root", default="/home/vina04/ADAR/Deco-main/runs/medical_datasets_0727_20260806_070731")
    parser.add_argument("--fewshot-root", default="/home/vina04/ADAR/Deco-main/runs/main_fewshot_refprompt_youden_20260817_090158")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--score-key", default="hdec_score", help="Score key used for fixed Answer reconstruction.")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--base-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--defect-limit", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--limit-per-combo", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"/home/vina04/ADAR/Deco-main/runs/repro_reason_audit_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    write_json(out_dir / "run_config.json", make_run_config(args))
    (out_dir / "prompt_template.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")

    processor, model = load_model_and_processor(args)
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    args._eos_token_id = eos_token_id
    args._pad_token_id = pad_token_id

    all_rows: List[Dict[str, Any]] = []
    for shot in args.shots:
        for dataset in args.datasets:
            rows = run_combo(args, dataset, shot, processor, model, out_dir)
            all_rows.extend(rows)
            write_csv(out_dir / "reason_summary_by_dataset_shot.csv", summarize(all_rows, ["shot", "dataset"]))

    write_csv(out_dir / "reason_summary_by_dataset_shot.csv", summarize(all_rows, ["shot", "dataset"]))
    write_csv(out_dir / "reason_summary_by_dataset.csv", summarize(all_rows, ["dataset"]))
    write_csv(out_dir / "reason_summary_by_shot.csv", summarize(all_rows, ["shot"]))
    write_csv(out_dir / "reason_summary_by_confusion.csv", summarize(all_rows, ["confusion"]))
    write_csv(out_dir / "reason_summary_by_status.csv", summarize(all_rows, ["validation_status"]))

    failure_rows = [r for r in all_rows if r.get("consistency") != "consistent"]
    failure_path = out_dir / "reason_failures.jsonl"
    if failure_path.exists():
        failure_path.unlink()
    append_jsonl(failure_path, failure_rows)

    print(f"[DONE] output_dir={out_dir}", flush=True)
    print(f"[DONE] cases={len(all_rows)} failures={len(failure_rows)}", flush=True)


if __name__ == "__main__":
    main()
