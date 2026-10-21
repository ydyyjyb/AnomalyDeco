#!/usr/bin/env python3
"""Held-out, matched-set comparison for the cross-architecture VisA runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = ["qwen3_4b", "qwen2_2b", "qwen2_7b", "llava_ov_0_5b", "llava_ov_7b"]
SCORE_FIELDS = {"native": "final_score", "hdec": "hdec_score"}
SEED = 20261019
CALIBRATION_FRACTION = 0.20


def read_cases(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error") or not isinstance(row.get("probe"), dict):
                continue
            key = (str(row["category"]), str(row["image_id"]))
            rows[key] = row
    return rows


def assign_split(rows: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], str]:
    groups: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for key, row in rows.items():
        groups[(key[0], int(row["label"]))].append(key)
    split: dict[tuple[str, str], str] = {}
    for (category, label), keys in sorted(groups.items()):
        keys.sort(
            key=lambda key: hashlib.sha256(
                f"{SEED}|{category}|{label}|{key[1]}".encode("utf-8")
            ).hexdigest()
        )
        n_cal = 0 if len(keys) < 2 else max(1, round(len(keys) * CALIBRATION_FRACTION))
        n_cal = min(n_cal, len(keys) - 1)
        for i, key in enumerate(keys):
            split[key] = "calibration" if i < n_cal else "test"
    return split


def best_youden_threshold(rows: list[dict[str, Any]], score_field: str) -> float:
    labels = [int(row["label"]) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Youden calibration requires both normal and anomalous examples.")
    scores = sorted({float(row["probe"][score_field]) for row in rows})
    candidates = [float("-inf"), *scores, float("inf")]
    best_j = -float("inf")
    best_thresholds: list[float] = []
    for threshold in candidates:
        tp = sum(1 for row in rows if int(row["label"]) == 1 and float(row["probe"][score_field]) >= threshold)
        fp = sum(1 for row in rows if int(row["label"]) == 0 and float(row["probe"][score_field]) >= threshold)
        j = tp / positives - fp / negatives
        if j > best_j + 1e-12:
            best_j = j
            best_thresholds = [threshold]
        elif abs(j - best_j) <= 1e-12:
            best_thresholds.append(threshold)
    finite = [value for value in best_thresholds if math.isfinite(value)]
    return finite[len(finite) // 2] if finite else 0.0


def auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_metrics(rows: list[dict[str, Any]], score_field: str, threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        y = int(row["label"])
        score = float(row["probe"][score_field])
        pred = int(score >= threshold)
        labels.append(y)
        scores.append(score)
        if y and pred:
            tp += 1
        elif not y and pred:
            fp += 1
        elif not y and not pred:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows),
        "AUROC": auc(labels, scores),
        "ACC": (tp + tn) / len(rows) if rows else float("nan"),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "threshold": threshold,
    }


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    model_rows = {
        model: read_cases(args.run_root / model / "mechanism_cases.compact.jsonl")
        for model in MODELS
    }
    common_keys = set.intersection(*(set(rows) for rows in model_rows.values()))
    if not common_keys:
        raise RuntimeError("No matched valid samples across all model runs.")
    reference = model_rows[MODELS[0]]
    for model, rows in model_rows.items():
        for key in common_keys:
            if int(rows[key]["label"]) != int(reference[key]["label"]):
                raise RuntimeError(f"Ground-truth mismatch for {key} in {model}.")

    split = assign_split({key: reference[key] for key in common_keys})
    manifest = [
        {
            "category": key[0],
            "image_id": key[1],
            "label": int(reference[key]["label"]),
            "split": split[key],
        }
        for key in sorted(common_keys)
    ]
    csv_write(args.run_root / "heldout_split_manifest.csv", manifest)

    per_category_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    thresholds: dict[tuple[str, str, str], float] = {}
    test_by_model_method_category: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        cases = {key: model_rows[model][key] for key in common_keys}
        categories = sorted({key[0] for key in common_keys})
        for method, score_field in SCORE_FIELDS.items():
            metrics_by_category = []
            for category in categories:
                calibration = [
                    cases[key]
                    for key in common_keys
                    if key[0] == category and split[key] == "calibration"
                ]
                test = [
                    cases[key]
                    for key in common_keys
                    if key[0] == category and split[key] == "test"
                ]
                if not calibration or not test:
                    continue
                threshold = best_youden_threshold(calibration, score_field)
                thresholds[(model, method, category)] = threshold
                test_by_model_method_category[(model, method, category)] = test
                metrics = classification_metrics(test, score_field, threshold)
                metrics_by_category.append(metrics)
                per_category_rows.append(
                    {
                        "model": model,
                        "score": method,
                        "category": category,
                        "n_calibration": len(calibration),
                        **metrics,
                    }
                )
            macro = {
                metric: mean(row[metric] for row in metrics_by_category)
                for metric in ("AUROC", "ACC", "Precision", "Recall", "F1")
            }
            test_rows = [
                cases[key]
                for key in common_keys
                if split[key] == "test"
            ]
            # Micro classification uses the per-category calibration thresholds.
            micro_predictions = []
            tp = fp = tn = fn = 0
            micro_labels: list[int] = []
            micro_scores: list[float] = []
            for row in test_rows:
                category = str(row["category"])
                threshold = thresholds[(model, method, category)]
                y = int(row["label"])
                score = float(row["probe"][score_field])
                pred = int(score >= threshold)
                micro_labels.append(y)
                micro_scores.append(score)
                micro_predictions.append(pred)
                if y and pred:
                    tp += 1
                elif not y and pred:
                    fp += 1
                elif not y and not pred:
                    tn += 1
                else:
                    fn += 1
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            micro = {
                "n": len(test_rows),
                "AUROC_pooled": auc(micro_labels, micro_scores),
                "ACC": (tp + tn) / len(test_rows) if test_rows else float("nan"),
                "Precision": p,
                "Recall": r,
                "F1": 2 * p * r / (p + r) if p + r else 0.0,
            }
            summary_rows.append(
                {
                    "model": model,
                    "score": method,
                    "n_matched_total": len(common_keys),
                    "n_calibration": sum(split[key] == "calibration" for key in common_keys),
                    "n_test": len(test_rows),
                    **{f"macro_{k}": v for k, v in macro.items()},
                    **{f"micro_{k}": v for k, v in micro.items() if k != "n"},
                }
            )

    csv_write(args.run_root / "heldout_metrics_by_category.csv", per_category_rows)
    csv_write(args.run_root / "heldout_metrics.csv", summary_rows)

    # Paired, category-and-label-stratified bootstrap CIs for HDEC - native AUROC.
    rng = random.Random(SEED)
    bootstrap_rows: list[dict[str, Any]] = []
    for model in MODELS:
        strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for category in sorted({key[0] for key in common_keys}):
            rows = test_by_model_method_category[(model, "native", category)]
            for row in rows:
                strata[(category, int(row["label"]))].append(row)
        deltas_auc: list[float] = []
        deltas_f1: list[float] = []
        for _ in range(max(0, args.bootstrap)):
            sampled_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for (category, _label), rows in strata.items():
                sampled_by_category[category].extend(rng.choice(rows) for _ in rows)
            auc_diffs = []
            f1_diffs = []
            for category, sampled in sampled_by_category.items():
                native_t = thresholds[(model, "native", category)]
                hdec_t = thresholds[(model, "hdec", category)]
                native = classification_metrics(sampled, SCORE_FIELDS["native"], native_t)
                hdec = classification_metrics(sampled, SCORE_FIELDS["hdec"], hdec_t)
                if math.isfinite(native["AUROC"]) and math.isfinite(hdec["AUROC"]):
                    auc_diffs.append(hdec["AUROC"] - native["AUROC"])
                f1_diffs.append(hdec["F1"] - native["F1"])
            if auc_diffs:
                deltas_auc.append(mean(auc_diffs))
            if f1_diffs:
                deltas_f1.append(mean(f1_diffs))

        def ci(values: list[float]) -> tuple[float, float]:
            if not values:
                return (float("nan"), float("nan"))
            values.sort()
            return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]

        auc_lo, auc_hi = ci(deltas_auc)
        f1_lo, f1_hi = ci(deltas_f1)
        model_summary = {row["score"]: row for row in summary_rows if row["model"] == model}
        bootstrap_rows.append(
            {
                "model": model,
                "delta_macro_AUROC_hdec_minus_native": model_summary["hdec"]["macro_AUROC"]
                - model_summary["native"]["macro_AUROC"],
                "delta_macro_AUROC_CI95_low": auc_lo,
                "delta_macro_AUROC_CI95_high": auc_hi,
                "delta_macro_F1_hdec_minus_native": model_summary["hdec"]["macro_F1"]
                - model_summary["native"]["macro_F1"],
                "delta_macro_F1_CI95_low": f1_lo,
                "delta_macro_F1_CI95_high": f1_hi,
                "bootstrap_replicates": len(deltas_auc),
            }
        )
    csv_write(args.run_root / "heldout_paired_bootstrap.csv", bootstrap_rows)

    missing_rows = []
    union_keys = set.union(*(set(rows) for rows in model_rows.values()))
    for model in MODELS:
        for key in sorted(union_keys - set(model_rows[model])):
            missing_rows.append({"model": model, "category": key[0], "image_id": key[1]})
    csv_write(args.run_root / "failed_or_missing_cases.csv", missing_rows)

    (args.run_root / "heldout_metrics_readme.txt").write_text(
        "Held-out transfer evaluation. A deterministic, category x label stratified split assigns 20% of matched "
        "VisA test images to calibration and 80% to held-out testing. Youden thresholds are fitted on calibration "
        "only, separately per model, score source, and category. AUROC/ACC/Precision/Recall/F1 are then computed "
        "on held-out images. Macro metrics average categories; pooled micro AUROC can be affected by score-scale "
        "differences across categories and is reported only as a secondary statistic. Bootstrap CIs are paired and "
        "stratified by category and ground-truth label. The legacy mechanism_summary.csv reports Youden metrics on "
        "the full evaluated test set and should not be treated as the primary held-out result.\n",
        encoding="utf-8",
    )
    print(f"[OK] Matched images: {len(common_keys)}; held-out summary: {args.run_root / 'heldout_metrics.csv'}")


if __name__ == "__main__":
    main()
