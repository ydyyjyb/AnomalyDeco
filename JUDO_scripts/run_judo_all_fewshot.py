#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the official JUDO checkpoint on ADAR datasets under 0-4 shot normal-reference settings.

This script is an evaluation adapter. It does not train or modify JUDO.

Label convention:
  A = normal
  B = anomalous

For k-shot, every query is prompted with:
  query image + k same-category normal reference images

For datasets without a train-normal split, k reference normals are held out from
the evaluation normals and removed from the test set, matching the strict
few-shot protocol used in the ADAR main-method runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMAL_DIR_NAMES = {
    "good",
    "normal",
    "ok",
    "no_defect",
    "nodefect",
    "no-defect",
    "nondefect",
    "non_defect",
    "defect_free",
}

DATASETS = ["visa", "mvtec", "aitex", "btad", "elpv", "mpdd", "brainmri", "headct", "br35h"]

MEDICAL_DATASET_CONFIG = {
    "brainmri": {"category": "brain MRI scan", "defect_type": "brain tumor"},
    "headct": {"category": "head CT scan", "defect_type": "intracranial hemorrhage"},
    "br35h": {"category": "brain MRI tumor scan", "defect_type": "brain tumor"},
}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def is_normal_name(name: str) -> bool:
    return name.strip().lower() in NORMAL_DIR_NAMES


def abs_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def make_sample(
    dataset: str,
    category: str,
    image: Path,
    root: Path,
    label: int,
    defect_type: str,
    split_name: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "category": category,
        "image": str(image),
        "image_id": image.stem,
        "label": int(label),
        "label_name": "normal" if int(label) == 0 else "anomaly",
        "defect_type": defect_type,
        "split_image": safe_rel(image, root),
        "split_name": split_name,
        "gt_answer": "A" if int(label) == 0 else "B",
        "gt_binary": int(label),
    }


def load_visa_eval_and_train(root: Path, split_csv: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    csv_path = root / "split_csv" / split_csv
    eval_samples: List[Dict[str, Any]] = []
    train_normals: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_text = row.get("label", "")
            label = 0 if label_text == "normal" else 1
            image = root / row["image"]
            sample = make_sample(
                "visa",
                row["object"],
                image,
                root,
                label,
                "good" if label == 0 else label_text,
                row.get("split", ""),
            )
            if row.get("split") == "test":
                eval_samples.append(sample)
            elif row.get("split") == "train" and label == 0:
                train_normals.append(sample)
    return eval_samples, train_normals


def load_mvtec_like_eval_and_train(root: Path, dataset: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_samples: List[Dict[str, Any]] = []
    train_normals: List[Dict[str, Any]] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name

        test_dir = category_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                label = 0 if is_normal_name(defect_type) else 1
                for image in list_images(defect_dir):
                    eval_samples.append(
                        make_sample(dataset, category, image, root, label, "good" if label == 0 else defect_type, "test")
                    )

        train_dir = category_dir / "train"
        if train_dir.exists():
            normal_dirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir() and is_normal_name(p.name)]
            image_paths: List[Path] = []
            if normal_dirs:
                for normal_dir in normal_dirs:
                    image_paths.extend(list_images(normal_dir))
            else:
                image_paths.extend(list_images(train_dir))
            for image in sorted(set(image_paths)):
                train_normals.append(make_sample(dataset, category, image, root, 0, "good", "train"))
    return eval_samples, train_normals


def load_aitex_eval_and_train(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_samples: List[Dict[str, Any]] = []
    category = "aitex_fabric"
    for image in list_images(root / "NODefect_images"):
        eval_samples.append(make_sample("aitex", category, image, root, 0, "good", "eval"))
    for image in list_images(root / "Defect_images"):
        eval_samples.append(make_sample("aitex", category, image, root, 1, "fabric_defect", "eval"))
    return eval_samples, []


def parse_elpv_rows(label_path: Path) -> List[Tuple[str, float, str]]:
    rows: List[Tuple[str, float, str]] = []
    with label_path.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if "," in raw:
                parts = [part.strip() for part in next(csv.reader([raw]))]
            elif ";" in raw:
                parts = [part.strip() for part in raw.split(";")]
            else:
                parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            cell_type = parts[2] if len(parts) > 2 else "cell"
            rows.append((parts[0], score, cell_type))
    return rows


def resolve_elpv_image(root: Path, rel_path: str) -> Optional[Path]:
    candidates = [
        root / rel_path,
        root / "images" / rel_path,
        root / "images" / Path(rel_path).name,
        root / "doc" / "images" / rel_path,
        root / "doc" / "images" / Path(rel_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_elpv_eval_and_train(root: Path, label_policy: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_samples: List[Dict[str, Any]] = []
    for rel_path, defect_score, cell_type in parse_elpv_rows(root / "labels.csv"):
        image = resolve_elpv_image(root, rel_path)
        if image is None or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if label_policy == "nonzero_anomaly":
            label = 1 if defect_score > 0.0 else 0
        elif label_policy == "threshold_0.5":
            label = 1 if defect_score >= 0.5 else 0
        elif label_policy == "strict":
            if defect_score <= 1e-8:
                label = 0
            elif defect_score >= 1.0 - 1e-8:
                label = 1
            else:
                continue
        else:
            raise ValueError(f"Unsupported ELPV label policy: {label_policy}")
        category = f"elpv_{cell_type}" if cell_type else "elpv"
        defect_type = "good" if label == 0 else "solar_cell_defect"
        eval_samples.append(make_sample("elpv", category, image, root, label, defect_type, "eval"))
    return eval_samples, []


def load_medical_eval_and_train(root: Path, dataset: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = MEDICAL_DATASET_CONFIG[dataset]
    category = cfg["category"]
    defect_type = cfg["defect_type"]
    eval_samples: List[Dict[str, Any]] = []
    train_normals: List[Dict[str, Any]] = []
    for label, folder_name in ((0, "good"), (1, "defect")):
        for image in list_images(root / "test" / folder_name):
            eval_samples.append(
                make_sample(dataset, category, image, root, label, "good" if label == 0 else defect_type, "test")
            )
    for candidate_dir in [root / "train" / "good", root / "train" / "normal", root / "train"]:
        if candidate_dir.exists():
            for image in list_images(candidate_dir):
                train_normals.append(make_sample(dataset, category, image, root, 0, "good", "train"))
            if train_normals:
                break
    return eval_samples, train_normals


def load_dataset(root_base: Path, dataset: str, visa_split_csv: str, elpv_label_policy: str):
    roots = {
        "visa": root_base / "VisA",
        "mvtec": root_base / "MVTec",
        "aitex": root_base / "AITEX",
        "btad": root_base / "BTAD",
        "elpv": root_base / "ELPV",
        "mpdd": root_base / "MPDD",
        "brainmri": root_base / "BrainMRI",
        "headct": root_base / "HeadCT",
        "br35h": root_base / "Br35H",
    }
    root = roots[dataset]
    if dataset == "visa":
        return root, *load_visa_eval_and_train(root, visa_split_csv)
    if dataset in {"mvtec", "btad", "mpdd"}:
        return root, *load_mvtec_like_eval_and_train(root, dataset)
    if dataset == "aitex":
        return root, *load_aitex_eval_and_train(root)
    if dataset == "elpv":
        return root, *load_elpv_eval_and_train(root, elpv_label_policy)
    if dataset in MEDICAL_DATASET_CONFIG:
        return root, *load_medical_eval_and_train(root, dataset)
    raise ValueError(dataset)


def select_support_and_test(
    eval_samples: List[Dict[str, Any]],
    train_normals: List[Dict[str, Any]],
    shots: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if shots <= 0:
        return [], list(eval_samples), {
            "shots": 0,
            "support_total": 0,
            "eval_total_before": len(eval_samples),
            "test_total_after": len(eval_samples),
            "removed_eval_support_overlap": 0,
            "per_category": {},
        }

    rng = random.Random(seed)
    categories = sorted({s["category"] for s in eval_samples})
    train_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    eval_norm_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in train_normals:
        train_by_cat[s["category"]].append(s)
    for s in eval_samples:
        if int(s["label"]) == 0:
            eval_norm_by_cat[s["category"]].append(s)

    support: List[Dict[str, Any]] = []
    per_category: Dict[str, Any] = {}
    for category in categories:
        train_bucket = sorted(train_by_cat.get(category, []), key=lambda x: abs_key(x["image"]))
        eval_bucket = sorted(eval_norm_by_cat.get(category, []), key=lambda x: abs_key(x["image"]))
        if len(train_bucket) >= shots:
            source, bucket = "train", train_bucket
        else:
            source, bucket = "eval_normal_holdout", eval_bucket
        if not bucket:
            chosen: List[Dict[str, Any]] = []
        else:
            chosen = rng.sample(bucket, min(shots, len(bucket)))
        for s in chosen:
            item = dict(s)
            item["support_source"] = source
            item["split_role"] = "support"
            support.append(item)
        per_category[category] = {
            "requested_shots": shots,
            "selected_shots": len(chosen),
            "selected_source": source,
            "available_train_normals": len(train_bucket),
            "available_eval_normals": len(eval_bucket),
            "support_images": [c["split_image"] for c in chosen],
        }

    support_keys = {abs_key(s["image"]) for s in support}
    test = [s for s in eval_samples if abs_key(s["image"]) not in support_keys]
    return support, test, {
        "shots": shots,
        "support_total": len(support),
        "eval_total_before": len(eval_samples),
        "test_total_after": len(test),
        "removed_eval_support_overlap": len(eval_samples) - len(test),
        "per_category": per_category,
    }


def attach_refs(test_samples: List[Dict[str, Any]], support: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in support:
        by_cat[s["category"]].append(s)
    out: List[Dict[str, Any]] = []
    for s in test_samples:
        refs = sorted(by_cat.get(s["category"], []), key=lambda x: abs_key(x["image"]))
        item = dict(s)
        item["reference_images"] = [r["image"] for r in refs]
        item["reference_image_count"] = len(refs)
        out.append(item)
    return out


def make_system_prompt(grid_size: int) -> str:
    return (
        "You are a helpful assistant for industrial quality control tasks. "
        "The user asks a question, and the Assistant solves it.\n"
        f"The query image is divided into a grid of {grid_size}×{grid_size} patches. "
        "Patch coordinates are (row, col) and both row and col start from 1 "
        "(top-left patch is (1,1)).\n\n"
        "Always respond in this exact order and format:\n"
        "1) <seg> The anomalous region patch list. Use (r,c) and "
        "(r,c_start)-(r,c_end) for contiguous columns. If there is no anomaly, respond 'None'. </seg>\n"
        "2) <think> Provide concise visual reasoning. </think>\n"
        "3) <answer>A|B</answer>\n"
    )


def make_question(case: Dict[str, Any], shots: int) -> str:
    category = case["category"]
    if shots <= 0:
        return (
            f"Object category: {category}.\n"
            "Only one image is provided: the QUERY image.\n"
            "Question: Is the QUERY image normal or anomalous?\n"
            "Choices:\n"
            "A. Normal: no visible anomaly or defect.\n"
            "B. Anomalous: visible defect, damage, contamination, deformation, missing part, scratch, hole, color spot, lesion, tumor, hemorrhage, or other abnormal region.\n"
        )
    return (
        f"Object category: {category}.\n"
        f"First image = QUERY. The next {shots} image(s) are NORMAL reference image(s) from the same category.\n"
        "Compare the QUERY image against the NORMAL reference image(s).\n"
        "Question: Is there any visible anomaly or defect in the QUERY image?\n"
        "Choices:\n"
        "A. No. The query image is normal.\n"
        "B. Yes. The query image is anomalous or defective.\n"
    )


def parse_answer(text: str) -> str:
    m = re.search(r"<answer>\s*([AB])\s*</answer>", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer|choice|correct answer)\s*(?:is|:)?\s*([AB])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", text)
    return m.group(1).upper() if m else "X"


def parse_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def answer_prefix_from_output(output: str) -> str:
    m = re.search(r"^(.*?<answer>\s*)[AB]\s*(?:</answer>)?", output, re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"^(.*?<answer>\s*)", output, re.S | re.I)
    if m:
        return m.group(1)
    return output.rstrip() + "\n<answer>"


def load_image(path: str | Path, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)


def batch_iter(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    sorted_pairs = sorted(enumerate(pairs), key=lambda x: x[1][1])
    ranks = [0.0] * len(sorted_pairs)
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][1][1] == sorted_pairs[i][1][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, (y, _) in zip(ranks, pairs) if y == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def div_optional(a: float, b: float) -> Optional[float]:
    return a / b if b > 0 else None


def select_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    policy: str = "youden",
    fixed_threshold: float = 0.0,
) -> float:
    if policy == "fixed":
        return float(fixed_threshold)
    if policy == "zero":
        return 0.0

    records = [(int(y), float(s)) for y, s in zip(labels, scores)]
    if not records:
        return float(fixed_threshold)
    unique_scores = sorted({score for _, score in records})
    if not unique_scores:
        return float(fixed_threshold)
    candidates = [unique_scores[0] - 1e-12]
    candidates.extend(unique_scores)
    candidates.append(unique_scores[-1] + 1e-12)

    def stats_at(threshold: float) -> Tuple[int, int, int, int]:
        tp = sum(1 for y, s in records if y == 1 and s >= threshold)
        tn = sum(1 for y, s in records if y == 0 and s < threshold)
        fp = sum(1 for y, s in records if y == 0 and s >= threshold)
        fn = sum(1 for y, s in records if y == 1 and s < threshold)
        return tp, tn, fp, fn

    best_threshold = candidates[0]
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        tp, tn, fp, fn = stats_at(threshold)
        acc = safe_div(tp + tn, len(records))
        recall_anomaly = safe_div(tp, tp + fn)
        recall_normal = safe_div(tn, tn + fp)
        precision_anomaly = safe_div(tp, tp + fp)
        f1_anomaly = (
            2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
            if (precision_anomaly + recall_anomaly) > 0
            else 0.0
        )
        if policy == "best_f1":
            value = f1_anomaly
            tiebreak = acc
        elif policy == "best_acc":
            value = acc
            tiebreak = (recall_anomaly + recall_normal) / 2.0
        else:
            value = (recall_anomaly + recall_normal) / 2.0
            tiebreak = f1_anomaly
        if value > best_value or (value == best_value and tiebreak > best_tiebreak):
            best_value = value
            best_tiebreak = tiebreak
            best_threshold = threshold
    return float(best_threshold)


def metrics_from_preds(labels: Sequence[int], scores: Sequence[float], preds: Sequence[int], threshold: Any) -> Dict[str, Any]:
    total = len(labels)
    tp = sum(1 for y, p in zip(labels, preds) if int(y) == 1 and int(p) == 1)
    tn = sum(1 for y, p in zip(labels, preds) if int(y) == 0 and int(p) == 0)
    fp = sum(1 for y, p in zip(labels, preds) if int(y) == 0 and int(p) == 1)
    fn = sum(1 for y, p in zip(labels, preds) if int(y) == 1 and int(p) == 0)

    precision_anomaly = div_optional(tp, tp + fp)
    recall_anomaly = div_optional(tp, tp + fn)
    f1_anomaly = (
        2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
        if precision_anomaly is not None and recall_anomaly is not None and (precision_anomaly + recall_anomaly) > 0
        else None
    )
    precision_normal = div_optional(tn, tn + fn)
    recall_normal = div_optional(tn, tn + fp)
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if precision_normal is not None and recall_normal is not None and (precision_normal + recall_normal) > 0
        else None
    )

    def avg_valid(values: Sequence[Optional[float]]) -> Optional[float]:
        valid = [float(x) for x in values if x is not None]
        return sum(valid) / len(valid) if valid else None

    return {
        "total": total,
        "acc": div_optional(tp + tn, total),
        "AUROC": binary_auroc(labels, scores),
        "threshold": threshold,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": avg_valid([recall_normal, recall_anomaly]),
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": avg_valid([precision_normal, precision_anomaly]),
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": avg_valid([f1_normal, f1_anomaly]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": tp + fn,
        "normal_total": tn + fp,
    }


def metrics_from_scores(records: List[Dict[str, Any]], score_key: str = "answer_score", policy: str = "youden") -> Dict[str, Any]:
    labels = [int(r["gt_binary"]) for r in records]
    scores = [float(r[score_key]) for r in records]
    threshold = select_threshold(labels, scores, policy)
    preds = [1 if s >= threshold else 0 for s in scores]
    return metrics_from_preds(labels, scores, preds, threshold)


def average_category_metrics(category_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "acc",
        "AUROC",
        "recall_normal",
        "recall_anomaly",
        "recall_all",
        "precision_normal",
        "precision_anomaly",
        "precision_all",
        "F1_normal",
        "F1_anomaly",
        "F1_all",
    ]
    out: Dict[str, Any] = {"total": len(category_metrics)}
    for key in keys:
        vals = [float(m[key]) for m in category_metrics if m.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def compute_score_metrics(records: List[Dict[str, Any]], score_key: str = "answer_score", policy: str = "youden") -> Dict[str, Any]:
    overall = metrics_from_scores(records, score_key=score_key, policy=policy)
    categories = sorted({r["category"] for r in records})
    category_metrics: List[Dict[str, Any]] = []
    thresholds: Dict[str, float] = {}
    for category in categories:
        subset = [r for r in records if r["category"] == category]
        metric = metrics_from_scores(subset, score_key=score_key, policy=policy)
        category_metrics.append(metric)
        thresholds[category] = float(metric["threshold"])
    macro = average_category_metrics(category_metrics)
    return {
        "score_source": score_key,
        "threshold_policy": policy,
        "threshold_by_category": json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
        "ACC": macro.get("acc"),
        "AUROC": macro.get("AUROC"),
        "Recall_all": macro.get("recall_all"),
        "Precision_all": macro.get("precision_all"),
        "F1_all": macro.get("F1_all"),
        "ACC_overall": overall.get("acc"),
        "AUROC_overall": overall.get("AUROC"),
        "Recall_all_overall": overall.get("recall_all"),
        "Precision_all_overall": overall.get("precision_all"),
        "F1_all_overall": overall.get("F1_all"),
        "macro_category_count": macro.get("total"),
    }


def compute_hard_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    invalid = sum(1 for r in records if r["prediction"] == "X")
    labels = [int(r["gt_binary"]) for r in records]
    preds = [int(r["pred_binary"]) for r in records]
    hard_scores = [1.0 if r["prediction"] == "B" else 0.0 if r["prediction"] == "A" else 0.5 for r in records]
    metrics = metrics_from_preds(labels, hard_scores, preds, threshold="hard_AB")
    return {
        "hard_valid": total - invalid,
        "hard_invalid": invalid,
        "ACC_hard": metrics.get("acc"),
        "AUROC_hard_binary_pred": metrics.get("AUROC"),
        "Recall_all_hard": metrics.get("recall_all"),
        "Precision_all_hard": metrics.get("precision_all"),
        "F1_all_hard": metrics.get("F1_all"),
        "tp_hard": metrics.get("tp"),
        "tn_hard": metrics.get("tn"),
        "fp_hard": metrics.get("fp"),
        "fn_hard": metrics.get("fn"),
        "gt_normal": sum(1 for r in records if r["gt_binary"] == 0),
        "gt_anomaly": sum(1 for r in records if r["gt_binary"] == 1),
        "pred_normal_hard": sum(1 for r in records if r["prediction"] == "A"),
        "pred_anomaly_hard": sum(1 for r in records if r["prediction"] == "B"),
    }


def format_float(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.4f}"
    return str(x)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, Any]], cols: Sequence[str]) -> None:
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_records(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    records: List[Dict[str, Any]] = []
    done: set[str] = set()
    if not path.exists():
        return records, done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            done.add(abs_key(rec.get("image_path", rec.get("image", ""))))
    return records, done


def run_combo(
    *,
    dataset: str,
    shots: int,
    samples: List[Dict[str, Any]],
    model,
    processor,
    tokenizer,
    system_prompt: str,
    answer_token_ids: Dict[str, int],
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    combo_name = f"{dataset}_{shots}shot"
    cases_path = out_dir / "cases" / f"{combo_name}.jsonl"
    summary_path = out_dir / "summaries" / f"{combo_name}.json"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing, done_keys = load_existing_records(cases_path)
    records = list(existing)
    todo = [s for s in samples if abs_key(s["image"]) not in done_keys]
    image_count = shots + 1
    eff_batch = max(1, args.base_batch_size // image_count)
    print(
        f"[COMBO] dataset={dataset} shots={shots} total={len(samples)} done={len(existing)} todo={len(todo)} batch={eff_batch}",
        flush=True,
    )

    with cases_path.open("a", encoding="utf-8") as f_jsonl:
        for chunk in batch_iter(todo, eff_batch):
            texts: List[str] = []
            images: List[Image.Image] = []
            case_image_paths: List[List[str]] = []
            for case in chunk:
                refs = list(case.get("reference_images") or [])
                case_images = [case["image"]] + refs
                case_image_paths.append(case_images)
                content = [{"type": "image"} for _ in case_images]
                content.append({"type": "text", "text": make_question(case, len(refs))})
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                for image_path in case_images:
                    images.append(load_image(image_path, args.image_size))

            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda:0")

            t0 = time.time()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            elapsed = (time.time() - t0) / max(1, len(chunk))
            new_tokens = generated[:, inputs.input_ids.shape[1] :]
            outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            score_texts: List[str] = []
            score_images: List[Image.Image] = []
            for prompt_text, output, image_paths in zip(texts, outputs, case_image_paths):
                score_texts.append(prompt_text + answer_prefix_from_output(output.strip()))
                for image_path in image_paths:
                    score_images.append(load_image(image_path, args.image_size))
            score_inputs = processor(text=score_texts, images=score_images, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                score_inputs = score_inputs.to("cuda:0")
            with torch.no_grad():
                score_outputs = model(**score_inputs, return_dict=True)
            pair_logits = score_outputs.logits[:, -1, [answer_token_ids["A"], answer_token_ids["B"]]].float()
            pair_probs = torch.softmax(pair_logits, dim=-1).detach().cpu().tolist()

            for case, output, probs in zip(chunk, outputs, pair_probs):
                output = output.strip()
                pred = parse_answer(output)
                pred_binary = 0 if pred == "A" else 1 if pred == "B" else -1
                normal_prob = float(probs[0])
                anomaly_prob = float(probs[1])
                rec = {
                    "dataset": dataset,
                    "shots": shots,
                    "category": case["category"],
                    "label": case["label_name"],
                    "gt_answer": case["gt_answer"],
                    "gt_binary": case["gt_binary"],
                    "prediction": pred,
                    "pred_binary": pred_binary,
                    "correct": pred == case["gt_answer"],
                    "answer_score": anomaly_prob,
                    "answer_normal_prob": normal_prob,
                    "answer_anomaly_prob": anomaly_prob,
                    "image": case["split_image"],
                    "image_path": case["image"],
                    "reference_image_count": len(case.get("reference_images") or []),
                    "reference_images": case.get("reference_images") or [],
                    "segmentation": parse_tag(output, "seg"),
                    "thinking": parse_tag(output, "think"),
                    "output": output,
                    "seconds_per_sample": elapsed,
                }
                records.append(rec)
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            print(
                f"[PROGRESS] {combo_name} {len(records)}/{len(samples)}",
                flush=True,
            )
            del inputs, generated, score_inputs, score_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    score_metrics = compute_score_metrics(records, score_key="answer_score", policy="youden")
    hard_metrics = compute_hard_metrics(records)
    summary = {
        "dataset": dataset,
        "shots": shots,
        "total": len(records),
        **score_metrics,
        **hard_metrics,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {combo_name} {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/vina04/ADAR/JUDO/models/JUDO")
    parser.add_argument("--data-root", default="/home/vina04/ADAR/evaluation/ODinW-13/data")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--visa-split-csv", default="1cls.csv")
    parser.add_argument("--elpv-label-policy", default="nonzero_anomaly", choices=["nonzero_anomaly", "threshold_0.5", "strict"])
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--use-fast-processor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir or f"/home/vina04/ADAR/JUDO/runs/judo_all_0to4shot_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_path": args.model_path,
        "data_root": args.data_root,
        "datasets": args.datasets,
        "shots": args.shots,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "image_size": args.image_size,
        "base_batch_size": args.base_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OUT_DIR] {out_dir}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=args.use_fast_processor,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    tokenizer.padding_side = "left"
    answer_a_ids = tokenizer.encode("A", add_special_tokens=False)
    answer_b_ids = tokenizer.encode("B", add_special_tokens=False)
    if len(answer_a_ids) != 1 or len(answer_b_ids) != 1:
        raise RuntimeError(f"Expected single-token A/B labels, got A={answer_a_ids}, B={answer_b_ids}")
    answer_token_ids = {"A": int(answer_a_ids[0]), "B": int(answer_b_ids[0])}
    print(f"[ANSWER_TOKEN_IDS] {answer_token_ids}", flush=True)

    data_root = Path(args.data_root)
    system_prompt = make_system_prompt(args.grid_size)
    all_summaries: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        root, eval_samples, train_normals = load_dataset(data_root, dataset, args.visa_split_csv, args.elpv_label_policy)
        if args.limit_per_dataset > 0:
            eval_samples = eval_samples[: args.limit_per_dataset]
        dataset_info = {
            "dataset": dataset,
            "root": str(root),
            "eval_samples": len(eval_samples),
            "train_normals": len(train_normals),
            "categories": sorted({s["category"] for s in eval_samples}),
            "gt_normal": sum(1 for s in eval_samples if int(s["label"]) == 0),
            "gt_anomaly": sum(1 for s in eval_samples if int(s["label"]) == 1),
        }
        (out_dir / "dataset_info").mkdir(exist_ok=True)
        (out_dir / "dataset_info" / f"{dataset}.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[DATASET] {json.dumps(dataset_info, ensure_ascii=False)}", flush=True)

        for shots in args.shots:
            support, test_samples, support_meta = select_support_and_test(eval_samples, train_normals, shots, args.seed)
            (out_dir / "support").mkdir(exist_ok=True)
            (out_dir / "support" / f"{dataset}_{shots}shot.json").write_text(
                json.dumps(support_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            samples = attach_refs(test_samples, support)
            summary = run_combo(
                dataset=dataset,
                shots=shots,
                samples=samples,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                answer_token_ids=answer_token_ids,
                out_dir=out_dir,
                args=args,
            )
            summary.update(
                {
                    "eval_total_before": support_meta["eval_total_before"],
                    "test_total_after": support_meta["test_total_after"],
                    "support_total": support_meta["support_total"],
                    "removed_eval_support_overlap": support_meta["removed_eval_support_overlap"],
                }
            )
            all_summaries.append(summary)
            write_csv(out_dir / "judo_0to4shot_summary.csv", all_summaries)
            write_markdown(
                out_dir / "judo_0to4shot_summary.md",
                all_summaries,
                [
                    "shots",
                    "dataset",
                    "ACC",
                    "AUROC",
                    "Recall_all",
                    "Precision_all",
                    "F1_all",
                    "ACC_hard",
                    "AUROC_hard_binary_pred",
                    "total",
                    "hard_invalid",
                ],
            )

    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
