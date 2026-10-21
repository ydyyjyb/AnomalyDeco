#!/usr/bin/env python3
"""0728_Fight: audit FP/FN cases from an existing 0727_UP run.

This script does not run Qwen3-VL again. It reads the saved 0727_UP outputs,
reconstructs predictions for one fixed experiment/policy, and writes only the
wrong cases for diagnosis.
"""

import argparse
import copy
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple


RAW_SOURCE_TO_PROBE_KEY = {
    "text_prior": "text_prior_score",
    "final": "final_score",
    "anchor": "anchor_score",
    "corrected": "corrected_score",
    "final_debiased": "final_debiased_score",
    "anchor_debiased": "anchor_debiased_score",
    "corrected_debiased": "corrected_debiased_score",
    "defect_aware_pool": "defect_aware_pool_score",
    "hdec": "hdec_score",
    "hdec_debiased": "hdec_debiased_score",
    "anchor_hdec": "anchor_hdec_score",
    "anchor_hdec_debiased": "anchor_hdec_debiased_score",
    "suppression_hdec": "suppression_hdec_score",
    "suppression_hdec_debiased": "suppression_hdec_debiased_score",
    "suppression_pool_hdec": "suppression_pool_hdec_score",
    "suppression_pool_hdec_debiased": "suppression_pool_hdec_debiased_score",
    "generic_defect_pool": "generic_defect_pool_score",
    "specific_defect_pool": "specific_defect_pool_score",
    "generic_hdec": "generic_hdec_score",
    "generic_hdec_debiased": "generic_hdec_debiased_score",
    "specific_hdec": "specific_hdec_score",
    "specific_hdec_debiased": "specific_hdec_debiased_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Create FP/FN audit CSVs from existing 0727_UP outputs without rerunning the model."
    )
    parser.add_argument(
        "--run-full-matrix",
        action="store_true",
        help=(
            "Run the full 0727-style score matrix twice: native Qwen3 scores and "
            "DeCo-corrected score remapping. Without this flag, the script audits "
            "an existing 0727_UP run."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="A prompt-mode output directory containing up_cases.json and up_matrix_summary.csv.",
    )
    parser.add_argument(
        "--visa-root",
        type=str,
        default="",
        help="VisA root. Used to read category/image_anno.csv for ground-truth anomaly types.",
    )
    parser.add_argument("--dataset", type=str, default="visa", choices=["visa", "mvtec", "aitex", "elpv", "btad", "mpdd"])
    parser.add_argument("--mvtec-root", type=str, default="")
    parser.add_argument("--aitex-root", type=str, default="")
    parser.add_argument("--elpv-root", type=str, default="")
    parser.add_argument("--btad-root", type=str, default="")
    parser.add_argument("--mpdd-root", type=str, default="")
    parser.add_argument(
        "--elpv-label-policy",
        type=str,
        default="strict",
        choices=["strict", "threshold_0.5", "nonzero_anomaly"],
        help=(
            "ELPV labels.csv policy. strict keeps only 0.0 as normal and 1.0 as anomaly; "
            "threshold_0.5 uses label>=0.5; nonzero_anomaly treats any positive score as anomaly."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="", help="Defaults to <run-dir>/fight_audit.")
    parser.add_argument("--experiment", type=str, default="E_raw_hdec")
    parser.add_argument("--threshold-policy", type=str, default="youden")
    parser.add_argument(
        "--threshold-scope",
        type=str,
        default="category",
        choices=["category", "overall"],
        help=(
            "category uses per-category thresholds from up_matrix_summary.csv and matches macro-summary "
            "classification; overall uses the single ALL threshold."
        ),
    )
    parser.add_argument(
        "--cases-file",
        type=str,
        default="",
        help="Override cases JSON path. Defaults to <run-dir>/up_cases.json.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="",
        help="Override summary CSV path. Defaults to <run-dir>/up_matrix_summary.csv.",
    )
    parser.add_argument(
        "--generate-raw-text",
        action="store_true",
        help="Load Qwen3-VL and generate raw_text only for FP/FN samples.",
    )
    parser.add_argument("--model-path", type=str, default="", help="Required when --generate-raw-text.")
    parser.add_argument("--visa-split-csv", type=str, default=None)
    parser.add_argument("--visa-split-set", type=str, default="test")
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-deco", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--trend-layers", type=str, default="0:34")
    parser.add_argument("--deco-layers", type=str, default="20:29")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument(
        "--score-deco-mode",
        type=str,
        default="multi_top3",
        choices=["single", "multi_top3"],
    )
    parser.add_argument("--multi-deco-k", type=int, default=3)
    parser.add_argument(
        "--anchor-selection",
        type=str,
        default="conflict",
        choices=["conflict", "confidence", "evidence"],
    )
    parser.add_argument(
        "--score-label-aggregation",
        type=str,
        default="max",
        choices=["max", "logsumexp"],
    )
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument(
        "--metric-threshold-policies",
        type=str,
        default="youden,zero,normal_q95",
        help="Comma-separated policies: youden, best_f1, best_acc, zero, normal_q90, normal_q95, normal_q99.",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.0)
    parser.add_argument("--no-text-prior-baseline", action="store_true")
    parser.add_argument("--normal-labels", type=str, default="good,normal,intact")
    parser.add_argument(
        "--generic-defect-labels",
        type=str,
        default=(
            "anomaly,defect,defective,abnormal,damage,damaged,crack,scratch,stain,hole,"
            "missing part,broken part,bent component,contamination,deformation"
        ),
    )
    parser.add_argument("--hedge-labels", type=str, default="maybe,possibly,likely,unclear,uncertain,appears,seems")
    parser.add_argument("--max-defect-labels", type=int, default=16)
    parser.add_argument(
        "--defect-label-json",
        type=str,
        default="",
        help=(
            "Optional category -> defect description JSON. If omitted, "
            "<dataset-root>/category_defect_labels.json is used when it exists."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="category_defects_guarded",
        choices=["generic", "category", "category_defects", "category_defects_guarded"],
        help="Prompt template forwarded to 0724_Mechanism_validation.py.",
    )
    parser.add_argument(
        "--prompt-defect-limit",
        type=int,
        default=8,
        help="Maximum category-level defect descriptions inserted into raw_text generation prompts.",
    )
    parser.add_argument("--defect-aware-pool-k", type=int, default=3)
    parser.add_argument("--defect-aware-lambda-norm", type=float, default=1.0)
    parser.add_argument("--defect-aware-eta-hedge", type=float, default=0.2)
    parser.add_argument("--defect-aware-enable-hedge", action="store_true", default=True)
    parser.add_argument("--defect-aware-use-logsumexp", action="store_true")
    parser.add_argument("--defect-aware-gamma", type=float, default=8.0)
    parser.add_argument("--hdec-gate-gamma", type=float, default=8.0)
    parser.add_argument("--hdec-gate-tau", type=float, default=0.0)
    parser.add_argument("--hdec-beta-prior", type=float, default=1.0)
    parser.add_argument("--suppression-gate-gamma", type=float, default=1.0)
    parser.add_argument("--suppression-gate-tau", type=float, default=0.0)
    parser.add_argument("--suppression-pool-gate-gamma", type=float, default=8.0)
    parser.add_argument("--suppression-pool-gate-tau", type=float, default=0.0)
    parser.add_argument(
        "--source-candidates",
        type=str,
        default=(
            "final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,"
            "suppression_pool_hdec_debiased,generic_defect_pool,specific_defect_pool,"
            "generic_hdec_debiased,specific_hdec_debiased"
        ),
    )
    parser.add_argument("--beta-grid", type=str, default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--robust-sources", type=str, default="corrected,hdec_debiased,corrected_debiased")
    parser.add_argument(
        "--zscore-sources",
        type=str,
        default=(
            "corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,"
            "suppression_pool_hdec_debiased,generic_defect_pool,specific_defect_pool,"
            "generic_hdec_debiased,specific_hdec_debiased"
        ),
    )
    parser.add_argument(
        "--quantile-center-sources",
        type=str,
        default=(
            "corrected_debiased,hdec_debiased,anchor_hdec,suppression_pool_hdec_debiased,"
            "generic_defect_pool,specific_defect_pool,generic_hdec_debiased,specific_hdec_debiased"
        ),
    )
    parser.add_argument("--normal-quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument(
        "--consensus-sources",
        type=str,
        default=(
            "final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,"
            "generic_defect_pool,specific_defect_pool,specific_hdec_debiased"
        ),
    )
    parser.add_argument(
        "--consensus-base-sources",
        type=str,
        default="corrected_debiased,hdec_debiased,specific_hdec_debiased",
    )
    parser.add_argument("--consensus-gamma-grid", type=str, default="0.25,0.5,1.0")
    parser.add_argument(
        "--fusion-sources",
        type=str,
        default=(
            "final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,"
            "generic_defect_pool,specific_defect_pool,specific_hdec_debiased"
        ),
    )
    parser.add_argument("--hard-categories", type=str, default="macaroni2,pcb1,pcb2,pcb3,capsules")
    parser.add_argument("--hard-category-source", type=str, default="specific_hdec_debiased")
    parser.add_argument("--easy-category-source", type=str, default="hdec_debiased")
    parser.add_argument("--disable-oracle-source-selection", action="store_true")
    parser.add_argument("--disable-default-source-map", action="store_true")
    parser.add_argument("--stream-jsonl", action="store_true")
    parser.add_argument("--include-generated-text", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--audit-experiment",
        type=str,
        default="E_raw_hdec",
        help="Experiment used for FP/FN raw-text audit in --run-full-matrix mode.",
    )
    parser.add_argument(
        "--audit-threshold-policy",
        type=str,
        default="youden",
        help="Threshold policy used for FP/FN raw-text audit in --run-full-matrix mode.",
    )
    parser.add_argument(
        "--raw-text-csv-name",
        type=str,
        default="error_audit_raw_text_minimal.csv",
        help="Minimal CSV with requested columns after optional raw_text generation.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def select_columns(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def normalize_rel(path_text: str) -> str:
    return str(path_text or "").replace("\\", "/").strip()


def load_image_anno_types(visa_root: Path) -> Dict[str, Dict[str, str]]:
    """Return category -> {relative image path or filename -> anomaly label text}."""
    if not visa_root or not visa_root.exists():
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for anno_path in sorted(visa_root.glob("*/image_anno.csv")):
        category = anno_path.parent.name
        category_map: Dict[str, str] = {}
        with open(anno_path, "r", encoding="utf-8-sig", newline="") as f:
            first_line = f.readline()
            f.seek(0)
            first_cells = next(csv.reader([first_line]), []) if first_line else []
            first_keys = {cell.strip().lower() for cell in first_cells}
            has_header = bool(
                first_keys
                & {
                    "image",
                    "img",
                    "path",
                    "image_path",
                    "mask",
                    "label",
                    "anomaly",
                    "anomaly_type",
                    "anomaly_types",
                    "prompt",
                }
            )
            if has_header:
                reader: Iterable[Dict[str, str]] = csv.DictReader(f)
                for row in reader:
                    image = normalize_rel(row.get("image") or row.get("img") or row.get("path") or "")
                    label = (
                        row.get("label")
                        or row.get("anomaly")
                        or row.get("anomaly_type")
                        or row.get("anomaly_types")
                        or row.get("prompt")
                        or ""
                    )
                    label = str(label).strip()
                    if image:
                        category_map[image] = label
                        category_map[Path(image).name] = label
            else:
                reader2 = csv.reader(f)
                for row in reader2:
                    if len(row) < 2:
                        continue
                    image = normalize_rel(row[0])
                    label = str(row[1]).strip()
                    if image:
                        category_map[image] = label
                        category_map[Path(image).name] = label
        out[category] = category_map
    return out


def get_case_rel_image(case: Dict[str, Any]) -> str:
    image = normalize_rel(case.get("image", ""))
    marker = "/VisA/"
    if marker in image:
        return image.split(marker, 1)[1]
    parts = image.split("/")
    for idx, part in enumerate(parts):
        if part == "VisA" and idx + 1 < len(parts):
            return "/".join(parts[idx + 1 :])
    category = str(case.get("category", "")).strip()
    if category and f"/{category}/" in image:
        return category + "/" + image.split(f"/{category}/", 1)[1]
    return image


def gt_anomaly_types_for_case(case: Dict[str, Any], anno_types: Dict[str, Dict[str, str]]) -> str:
    if int(case.get("label", 0)) == 0:
        return ""
    anomaly_types = case.get("anomaly_types")
    if isinstance(anomaly_types, list):
        return ",".join(str(item) for item in anomaly_types if str(item).strip())
    if isinstance(anomaly_types, str) and anomaly_types.strip():
        return anomaly_types
    defect_type = str(case.get("defect_type", "") or "").strip()
    if defect_type and defect_type != "good":
        return defect_type
    category = str(case.get("category", ""))
    category_map = anno_types.get(category, {})
    rel_image = get_case_rel_image(case)
    filename = Path(rel_image).name
    return category_map.get(rel_image, category_map.get(filename, ""))


def source_from_experiment(experiment: str) -> str:
    if experiment.startswith("E_raw_"):
        return experiment[len("E_raw_") :]
    raise ValueError(
        f"Only raw-source experiments are supported by this audit script for now: {experiment!r}. "
        "Use --experiment E_raw_hdec for the current main method."
    )


def score_for_case(case: Dict[str, Any], experiment: str) -> float:
    source = source_from_experiment(experiment)
    probe = case.get("probe") or case.get("hallucination_probe") or {}
    key = RAW_SOURCE_TO_PROBE_KEY.get(source)
    if not key:
        raise ValueError(f"Unknown raw source in experiment {experiment!r}: {source!r}")
    if key not in probe:
        raise KeyError(f"Case missing probe key {key!r} for experiment {experiment!r}")
    return float(probe[key])


def threshold_lookup(
    rows: List[Dict[str, str]],
    experiment: str,
    policy: str,
    scope: str,
) -> Tuple[Dict[str, float], Optional[float], Dict[str, Any]]:
    thresholds: Dict[str, float] = {}
    overall_threshold: Optional[float] = None
    selected_rows = [
        row
        for row in rows
        if row.get("experiment") == experiment and row.get("threshold_policy") == policy
    ]
    if not selected_rows:
        raise RuntimeError(f"No summary rows found for experiment={experiment}, policy={policy}")

    for row in selected_rows:
        row_scope = row.get("scope", "")
        category = row.get("category", "")
        threshold_text = row.get("threshold", "")
        if threshold_text == "":
            continue
        threshold = float(threshold_text)
        if row_scope == "overall" and category == "ALL":
            overall_threshold = threshold
        if row_scope == "category" and category:
            thresholds[category] = threshold

    if scope == "overall":
        if overall_threshold is None:
            raise RuntimeError("Overall threshold was not found.")
        return {}, overall_threshold, {"threshold_scope": scope, "overall_threshold": overall_threshold}

    if not thresholds:
        raise RuntimeError("Category thresholds were not found.")
    return thresholds, overall_threshold, {"threshold_scope": scope, "category_thresholds": thresholds}


def extract_raw_text(case: Dict[str, Any]) -> str:
    generated = case.get("generated")
    if isinstance(generated, dict):
        text = generated.get("text") or generated.get("raw_text") or ""
        if text:
            return str(text)
    for key in ("generated_text", "raw_text", "prediction_raw_text"):
        if case.get(key):
            return str(case.get(key))
    return ""


def compact_raw_text(text: str) -> str:
    return " ".join(str(text or "").split())


def load_defect_bank_by_category(run_dir: Path) -> Dict[str, List[str]]:
    candidates = [
        run_dir / "defect_bank_by_category.json",
        run_dir.parent / "defect_bank_by_category.json",
    ]
    for path in candidates:
        if path.exists():
            data = read_json(path)
            if isinstance(data, dict):
                return {
                    str(category): [str(item) for item in items]
                    for category, items in data.items()
                    if isinstance(items, list)
                }
    return {}


def prompt_defect_list(defect_bank: List[str], limit: int) -> List[str]:
    out = []
    seen = set()
    for defect in defect_bank:
        clean = str(defect).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
        if len(out) >= max(0, int(limit)):
            break
    return out


def build_raw_text_instruction(category: str, defect_bank: List[str], defect_limit: int) -> str:
    category_text = str(category or "object").replace("_", " ").strip() or "object"
    defects = prompt_defect_list(defect_bank, defect_limit)
    lines = [
        f"This is an image of {category_text}.",
    ]
    if defects:
        lines.append(f"This category may have defects such as: {', '.join(defects)}.")
        lines.append("These defects are only possible examples, not assumptions about this image.")
    lines.extend(
        [
            "Inspect the main subject for visual abnormalities.",
            "Use only visible evidence.",
            "Ignore viewpoint, lighting, background, image noise, pose, position, and orientation unless they create a clear physical defect.",
            "Briefly explain the visual evidence for a normal or anomalous decision.",
            "End with exactly one line:",
            "Answer: Normal",
            "or",
            "Answer: Anomaly",
        ]
    )
    return "\n".join(lines)


def load_qwen3_model(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    if not args.model_path:
        raise ValueError("--model-path is required when --generate-raw-text is used.")
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
        )
    except Exception:
        if not args.allow_attn_fallback or args.attn_implementation == "sdpa":
            raise
        print(f"[WARN] Failed to load with {args.attn_implementation}; fallback to sdpa.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model, torch


def generate_raw_text_for_errors(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    defect_bank_by_category: Dict[str, List[str]],
) -> None:
    if not rows:
        return
    processor, model, torch = load_qwen3_model(args)
    image_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}

    from PIL import Image

    for idx, row in enumerate(rows, start=1):
        if str(row.get("raw_text", "")).strip():
            continue
        category = str(row.get("category", ""))
        image_path = str(row.get("image_path", ""))
        defect_bank = defect_bank_by_category.get(category, [])
        instruction = build_raw_text_instruction(category, defect_bank, args.prompt_defect_limit)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": "Query image:"},
                    {"type": "image"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt", **image_kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output_ids[:, input_len:]
        raw_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        row["raw_text"] = compact_raw_text(raw_text)
        print(
            f"[RAW_TEXT] {idx}/{len(rows)} {row.get('error_type')} "
            f"{category}/{row.get('image_id', '')}"
        )


def build_error_rows(
    cases: List[Dict[str, Any]],
    experiment: str,
    thresholds_by_category: Dict[str, float],
    overall_threshold: Optional[float],
    threshold_scope: str,
    anno_types: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    error_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    fn_rows: List[Dict[str, Any]] = []

    for case in cases:
        if "error" in case and not (case.get("probe") or case.get("hallucination_probe")):
            continue
        label = int(case.get("label", 0))
        category = str(case.get("category", ""))
        threshold = (
            thresholds_by_category.get(category)
            if threshold_scope == "category"
            else overall_threshold
        )
        if threshold is None:
            continue
        score = score_for_case(case, experiment)
        pred = 1 if score >= float(threshold) else 0
        if pred == label:
            continue

        error_type = "FP" if label == 0 and pred == 1 else "FN"
        probe = case.get("probe") or case.get("hallucination_probe") or {}
        row = {
            "error_type": error_type,
            "category": category,
            "image_id": case.get("image_id", ""),
            "pred": "anomaly" if pred == 1 else "normal",
            "label": "anomaly" if label == 1 else "normal",
            "score": score,
            "threshold": float(threshold),
            "score_minus_threshold": score - float(threshold),
            "gt_anomaly_types": gt_anomaly_types_for_case(case, anno_types),
            "image_path": case.get("image", ""),
            "mask_path": case.get("mask", ""),
            "raw_text": compact_raw_text(extract_raw_text(case)),
            "hdec_gate": probe.get("hdec_gate", ""),
            "hdec_gate_arg": probe.get("hdec_gate_arg", ""),
            "anchor_score": probe.get("anchor_score", ""),
            "corrected_score": probe.get("corrected_score", ""),
            "hdec_score": probe.get("hdec_score", ""),
            "defect_aware_pool_score": probe.get("defect_aware_pool_score", ""),
            "text_prior_score": probe.get("text_prior_score", ""),
            "selected_layers": json.dumps(probe.get("selected_layers", []), ensure_ascii=False),
            "fp_type_manual": "",
            "notes": "",
        }
        error_rows.append(row)
        if error_type == "FP":
            fp_rows.append(row)
        else:
            fn_rows.append(row)

    return error_rows, fp_rows, fn_rows


def summarize_errors(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_category: Dict[str, Dict[str, int]] = {}
    for row in rows:
        category = str(row["category"])
        error_type = str(row["error_type"])
        by_category.setdefault(category, {"FP": 0, "FN": 0, "total": 0})
        by_category[category][error_type] += 1
        by_category[category]["total"] += 1
    return [
        {"category": category, **counts}
        for category, counts in sorted(by_category.items(), key=lambda item: item[0])
    ]


def load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_up_module() -> Any:
    return load_python_module(Path(__file__).with_name("0727_UP.py"), "fight_0727_up")


def prepare_full_args(args: argparse.Namespace) -> argparse.Namespace:
    root_arg_by_dataset = {
        "visa": "visa_root",
        "mvtec": "mvtec_root",
        "aitex": "aitex_root",
        "elpv": "elpv_root",
        "btad": "btad_root",
        "mpdd": "mpdd_root",
    }
    root_arg = root_arg_by_dataset.get(args.dataset)
    if root_arg is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if not str(getattr(args, root_arg, "") or "").strip():
        raise ValueError(f"--{root_arg.replace('_', '-')} is required with --dataset {args.dataset} --run-full-matrix.")
    if not args.model_path:
        raise ValueError("--model-path is required with --run-full-matrix.")
    if not args.output_dir:
        raise ValueError("--output-dir is required with --run-full-matrix.")
    args.skip_generation = not args.run_generation
    return args


DECO_SCORE_REMAP = {
    "final_score": "corrected_score",
    "final_pred": "corrected_pred",
    "final_correct": "corrected_correct",
    "final_debiased_score": "corrected_debiased_score",
    "final_debiased_pred": "corrected_debiased_pred",
    "final_debiased_correct": "corrected_debiased_correct",
    "text_prior_score": "corrected_prior_score",
}


def remap_case_to_deco_scores(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return a case where 0727's base final/text-prior sources are DeCo-corrected."""
    out = copy.deepcopy(case)
    probe = out.get("hallucination_probe") or out.get("probe")
    if not isinstance(probe, dict):
        return out
    for dst, src in DECO_SCORE_REMAP.items():
        if src in probe:
            probe[dst] = probe[src]
    probe["score_group"] = "deco_corrected"
    probe["score_group_note"] = (
        "final/text-prior fields are remapped to corrected/corrected-prior fields; "
        "0727 matrix logic is otherwise unchanged."
    )
    return out


def save_best_by_metric(up: Any, output_dir: Path, macro_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_by_metric: Dict[str, Any] = {}
    for key in ["acc", "AUROC", "recall_all", "precision_all", "F1_all", "F1_anomaly"]:
        candidates = [row for row in macro_rows if row.get(key) is not None and row.get(key) != ""]
        if candidates:
            best_by_metric[key] = max(candidates, key=lambda row: float(row[key]))
    up.save_json(output_dir / "up_best_by_metric.json", best_by_metric)
    return best_by_metric


def run_0727_matrix_for_group(
    up: Any,
    legacy: Any,
    args: argparse.Namespace,
    group_cases: List[Dict[str, Any]],
    output_dir: Path,
    score_group: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments, source_maps = up.build_experiments(legacy, args, group_cases)
    up.save_json(output_dir / "up_source_maps.json", source_maps)
    experiment_configs = [dict(up.experiment_config(experiment), score_group=score_group) for experiment in experiments]
    up.save_json(output_dir / "up_experiment_configs.json", experiment_configs)
    up.write_csv(output_dir / "up_experiment_configs.csv", experiment_configs)
    up.save_json(output_dir / "up_matrix_family_design.json", up.MATRIX_FAMILY_DESIGN)

    threshold_policies = up.parse_csv_words(args.metric_threshold_policies)
    all_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        scores = experiment["scores"]
        for policy in threshold_policies:
            rows, macro_metric = up.evaluate_experiment(
                legacy,
                group_cases,
                scores,
                experiment,
                policy,
                args.metric_threshold,
            )
            for row in rows:
                row["score_group"] = score_group
            all_rows.extend(rows)
            macro_row = {
                "score_group": score_group,
                "scope": "macro",
                "category": "ALL",
                "experiment": experiment["name"],
                "family": experiment.get("family", ""),
                "source": experiment.get("source", ""),
                "sources": experiment.get("sources", ""),
                "beta": experiment.get("beta", ""),
                "gamma": experiment.get("gamma", ""),
                "quantile": experiment.get("quantile", ""),
                "fusion_mode": experiment.get("fusion_mode", ""),
                "weights": experiment.get("weights", ""),
                "normalization": experiment.get("normalization", "none"),
                "source_selection": experiment.get("source_selection", ""),
                "route": experiment.get("route", ""),
                "threshold_policy": policy,
                "note": experiment.get("note", ""),
                **macro_metric,
                **up.target_gap(macro_metric),
            }
            macro_rows.append(macro_row)

    per_category_rows = [row for row in all_rows if row.get("scope") == "category"]
    up.write_csv(output_dir / "up_matrix_summary.csv", all_rows)
    up.write_csv(output_dir / "up_macro_summary.csv", macro_rows)
    up.write_csv(output_dir / "up_per_category_summary.csv", per_category_rows)
    best_by_metric = save_best_by_metric(up, output_dir, macro_rows)
    print(f"[OK][{score_group}] matrix: {output_dir / 'up_matrix_summary.csv'}")
    for key in ["acc", "AUROC", "recall_all", "precision_all", "F1_all"]:
        row = best_by_metric.get(key)
        if row:
            print(
                f"[BEST][{score_group}] {key}={row[key]} "
                f"experiment={row['experiment']} policy={row['threshold_policy']}"
            )
    return all_rows, macro_rows, per_category_rows


def extract_generated_text_pair(case: Dict[str, Any]) -> Tuple[str, str, str, str]:
    qwen_text = str(case.get("generated_text", "") or "")
    qwen_label = str(case.get("generated_label", "") or "")
    deco_text = str(case.get("generated_deco_text", "") or "")
    deco_label = str(case.get("generated_deco_label", "") or "")
    generated = case.get("generated")
    if isinstance(generated, dict):
        qwen_text = qwen_text or str(generated.get("text", "") or "")
        qwen_label = qwen_label or str(generated.get("label", "") or "")
    generated_deco = case.get("generation_deco")
    if isinstance(generated_deco, dict):
        deco_text = deco_text or str(generated_deco.get("text", "") or "")
        deco_label = deco_label or str(generated_deco.get("label", "") or "")
    return qwen_label, compact_raw_text(qwen_text), deco_label, compact_raw_text(deco_text)


def build_full_mode_error_rows(
    cases: List[Dict[str, Any]],
    group_cases: List[Dict[str, Any]],
    experiment: str,
    summary_rows: List[Dict[str, Any]],
    threshold_policy: str,
    anno_types: Dict[str, Dict[str, str]],
    score_group: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    thresholds_by_category, overall_threshold, _ = threshold_lookup(
        [{str(k): str(v) for k, v in row.items()} for row in summary_rows],
        experiment,
        threshold_policy,
        "category",
    )
    error_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    fn_rows: List[Dict[str, Any]] = []

    for original, scored_case in zip(cases, group_cases):
        label = int(scored_case.get("label", 0))
        category = str(scored_case.get("category", ""))
        threshold = thresholds_by_category.get(category, overall_threshold)
        if threshold is None:
            continue
        score = score_for_case(scored_case, experiment)
        pred = 1 if score >= float(threshold) else 0
        if pred == label:
            continue
        error_type = "FP" if label == 0 and pred == 1 else "FN"
        probe = scored_case.get("hallucination_probe") or scored_case.get("probe") or {}
        qwen_label, qwen_raw_text, deco_label, deco_raw_text = extract_generated_text_pair(original)
        row = {
            "score_group": score_group,
            "error_type": error_type,
            "category": category,
            "image_id": scored_case.get("image_id", ""),
            "pred": "anomaly" if pred == 1 else "normal",
            "label": "anomaly" if label == 1 else "normal",
            "score": score,
            "threshold": float(threshold),
            "score_minus_threshold": score - float(threshold),
            "gt_anomaly_types": gt_anomaly_types_for_case(scored_case, anno_types),
            "image_path": scored_case.get("image", ""),
            "mask_path": scored_case.get("mask", ""),
            "qwen_generated_label": qwen_label,
            "qwen_raw_text": qwen_raw_text,
            "deco_generated_label": deco_label,
            "deco_raw_text": deco_raw_text,
            "hdec_gate": probe.get("hdec_gate", ""),
            "hdec_gate_arg": probe.get("hdec_gate_arg", ""),
            "anchor_score": probe.get("anchor_score", ""),
            "corrected_score": probe.get("corrected_score", ""),
            "hdec_score": probe.get("hdec_score", ""),
            "defect_aware_pool_score": probe.get("defect_aware_pool_score", ""),
            "text_prior_score": probe.get("text_prior_score", ""),
            "selected_layers": json.dumps(probe.get("selected_layers", []), ensure_ascii=False),
            "notes": "",
        }
        error_rows.append(row)
        if error_type == "FP":
            fp_rows.append(row)
        else:
            fn_rows.append(row)
    return error_rows, fp_rows, fn_rows


def run_full_matrix(args: argparse.Namespace) -> None:
    args = prepare_full_args(args)
    up = load_up_module()
    legacy = up.load_legacy_module()

    import torch
    from tqdm import tqdm

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "0-config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "0728_Fight.py",
            "mode": "full_native_vs_deco_score_matrix",
            "training_free": True,
            "score_groups": {
                "native": "0727 matrix on original Qwen3 score probes.",
                "deco_corrected": "Same 0727 matrix after remapping final/text-prior base fields to DeCo-corrected scores.",
            },
            "args": vars(args),
        },
    )

    if args.dataset == "visa":
        samples = legacy.load_split_samples(args)
    else:
        samples = up.load_dataset_samples(args)
    if not samples:
        raise RuntimeError("No samples found.")
    print(f"[INFO] Loaded {len(samples)} samples.")

    normal_bank = up.parse_csv_words(args.normal_labels)
    hedge_bank = up.parse_csv_words(args.hedge_labels)
    generic_defect_labels = up.parse_csv_words(args.generic_defect_labels)
    categories = sorted({sample["category"] for sample in samples})
    dataset_root = up.get_dataset_root(args)
    defect_label_json = up.resolve_defect_label_json(args)
    custom_defect_labels = up.load_category_defect_label_json(defect_label_json)
    if defect_label_json:
        print(f"[INFO] Loaded category defect labels: {defect_label_json}")
    defect_bank_by_category = {
        category: up.build_focused_defect_bank(
            legacy,
            dataset_root,
            category,
            generic_defect_labels,
            args.max_defect_labels,
            custom_defect_labels,
            args.dataset,
        )
        for category in categories
    }
    write_json(output_dir / "defect_bank_by_category.json", defect_bank_by_category)
    write_json(output_dir / "custom_defect_labels_by_category.json", custom_defect_labels)
    write_json(output_dir / "hard_category_defect_hints.json", up.HARD_CATEGORY_DEFECT_HINTS)

    processor, model = legacy.load_model(args)
    cases: List[Dict[str, Any]] = []
    compact_cases: List[Dict[str, Any]] = []
    stream_jsonl = output_dir / "fight_cases.compact.jsonl"
    if stream_jsonl.exists():
        stream_jsonl.unlink()

    for idx, sample in enumerate(tqdm(samples, desc="0728_Fight probe", ncols=120), start=1):
        try:
            case = legacy.compute_probe(
                args,
                processor,
                model,
                sample,
                defect_bank_by_category[sample["category"]],
                normal_bank,
                hedge_bank,
            )
        except Exception as exc:
            case = {**sample, "error": str(exc)}
            print(f"[WARN] Failed sample {sample['category']}/{sample.get('image_id', '')}: {exc}")
        if "anomaly_types" in sample:
            case["anomaly_types"] = sample.get("anomaly_types", [])
        if "defect_type" in sample:
            case["defect_type"] = sample.get("defect_type", "")
        cases.append(case)
        compact = legacy.compact_case(case, include_generated_text=args.include_generated_text)
        if "anomaly_types" in case:
            compact["anomaly_types"] = case.get("anomaly_types", [])
        if "defect_type" in case:
            compact["defect_type"] = case.get("defect_type", "")
        compact_cases.append(compact)
        if args.stream_jsonl:
            with open(stream_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(compact, ensure_ascii=False) + "\n")
        if args.log_interval > 0 and idx % args.log_interval == 0:
            valid = sum(1 for item in cases if "hallucination_probe" in item)
            print(f"[INFO] processed={idx}/{len(samples)} valid={valid}")

    valid_cases = [case for case in cases if "hallucination_probe" in case]
    failed_cases = [case for case in cases if "hallucination_probe" not in case]
    write_json(output_dir / "fight_cases.json", compact_cases)
    write_json(output_dir / "failed_cases.json", failed_cases[:200])
    if not valid_cases:
        raise RuntimeError("No valid probe cases.")

    group_case_map = {
        "native": valid_cases,
        "deco_corrected": [remap_case_to_deco_scores(case) for case in valid_cases],
    }
    combined_all_rows: List[Dict[str, Any]] = []
    combined_macro_rows: List[Dict[str, Any]] = []
    combined_per_category_rows: List[Dict[str, Any]] = []
    audit_rows_all: List[Dict[str, Any]] = []
    anno_types = load_image_anno_types(Path(args.visa_root)) if args.dataset == "visa" else {}

    for score_group, group_cases in group_case_map.items():
        group_dir = output_dir / score_group
        all_rows, macro_rows, per_category_rows = run_0727_matrix_for_group(
            up,
            legacy,
            args,
            group_cases,
            group_dir,
            score_group,
        )
        combined_all_rows.extend(all_rows)
        combined_macro_rows.extend(macro_rows)
        combined_per_category_rows.extend(per_category_rows)

        try:
            error_rows, fp_rows, fn_rows = build_full_mode_error_rows(
                valid_cases,
                group_cases,
                args.audit_experiment,
                all_rows,
                args.audit_threshold_policy,
                anno_types,
                score_group,
            )
            up.write_csv(group_dir / "error_audit.csv", error_rows)
            up.write_csv(group_dir / "fp_audit.csv", fp_rows)
            up.write_csv(group_dir / "fn_audit.csv", fn_rows)
            up.write_csv(group_dir / "error_summary_by_category.csv", summarize_errors(error_rows))
            audit_rows_all.extend(error_rows)
            print(
                f"[OK][{score_group}] audit errors={len(error_rows)} "
                f"FP={len(fp_rows)} FN={len(fn_rows)}"
            )
        except Exception as exc:
            print(f"[WARN][{score_group}] Failed to build FP/FN audit: {exc}")

    up.write_csv(output_dir / "fight_matrix_summary.combined.csv", combined_all_rows)
    up.write_csv(output_dir / "fight_macro_summary.combined.csv", combined_macro_rows)
    up.write_csv(output_dir / "fight_per_category_summary.combined.csv", combined_per_category_rows)
    up.write_csv(output_dir / "fight_error_audit.combined.csv", audit_rows_all)
    best_by_group = {
        group: save_best_by_metric(
            up,
            output_dir / group,
            [row for row in combined_macro_rows if row.get("score_group") == group],
        )
        for group in group_case_map
    }
    write_json(output_dir / "fight_best_by_group.json", best_by_group)
    print(f"[OK] Combined macro summary: {output_dir / 'fight_macro_summary.combined.csv'}")
    print(f"[OK] Combined matrix summary: {output_dir / 'fight_matrix_summary.combined.csv'}")
    print(f"[OK] Combined FP/FN audit: {output_dir / 'fight_error_audit.combined.csv'}")


def main() -> None:
    args = parse_args()
    if args.run_full_matrix:
        run_full_matrix(args)
        return
    if not args.run_dir:
        raise ValueError("--run-dir is required unless --run-full-matrix is set.")
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "fight_audit"
    cases_path = Path(args.cases_file) if args.cases_file else run_dir / "up_cases.json"
    summary_path = Path(args.summary_file) if args.summary_file else run_dir / "up_matrix_summary.csv"

    cases = read_json(cases_path)
    summary_rows = read_csv_rows(summary_path)
    thresholds_by_category, overall_threshold, threshold_info = threshold_lookup(
        summary_rows,
        args.experiment,
        args.threshold_policy,
        args.threshold_scope,
    )
    anno_types = load_image_anno_types(Path(args.visa_root)) if args.visa_root else {}
    error_rows, fp_rows, fn_rows = build_error_rows(
        cases,
        args.experiment,
        thresholds_by_category,
        overall_threshold,
        args.threshold_scope,
        anno_types,
    )
    if args.generate_raw_text:
        defect_bank_by_category = load_defect_bank_by_category(run_dir)
        generate_raw_text_for_errors(args, error_rows, defect_bank_by_category)

    minimal_columns = ["error_type", "category", "pred", "score", "threshold", "image_path", "raw_text"]

    write_csv(output_dir / "error_audit.csv", error_rows)
    write_csv(output_dir / "fp_audit.csv", fp_rows)
    write_csv(output_dir / "fn_audit.csv", fn_rows)
    write_csv(output_dir / args.raw_text_csv_name, select_columns(error_rows, minimal_columns))
    write_csv(output_dir / "fp_audit_raw_text_minimal.csv", select_columns(fp_rows, minimal_columns))
    write_csv(output_dir / "fn_audit_raw_text_minimal.csv", select_columns(fn_rows, minimal_columns))
    write_csv(output_dir / "error_summary_by_category.csv", summarize_errors(error_rows))
    write_json(
        output_dir / "fight_audit_config.json",
        {
            "run_dir": str(run_dir),
            "cases_file": str(cases_path),
            "summary_file": str(summary_path),
            "visa_root": args.visa_root,
            "experiment": args.experiment,
            "threshold_policy": args.threshold_policy,
            **threshold_info,
            "num_cases": len(cases),
            "num_errors": len(error_rows),
            "num_fp": len(fp_rows),
            "num_fn": len(fn_rows),
            "generate_raw_text": bool(args.generate_raw_text),
            "model_path": args.model_path,
            "raw_text_csv_name": args.raw_text_csv_name,
            "note": "This audit is a post-processing diagnostic; Qwen3-VL is loaded only when --generate-raw-text is set.",
        },
    )

    print(f"[OK] errors: {len(error_rows)}")
    print(f"[OK] FP: {len(fp_rows)} -> {output_dir / 'fp_audit.csv'}")
    print(f"[OK] FN: {len(fn_rows)} -> {output_dir / 'fn_audit.csv'}")
    print(f"[OK] all errors: {output_dir / 'error_audit.csv'}")
    print(f"[OK] minimal raw-text CSV: {output_dir / args.raw_text_csv_name}")
    print(f"[OK] category summary: {output_dir / 'error_summary_by_category.csv'}")


if __name__ == "__main__":
    main()
