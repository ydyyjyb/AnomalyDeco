import argparse
import ast
import csv
import json
import random
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_ANCHOR_DEFECT_BASE_WORDS = ["defect", "anomaly", "abnormal"]
DEFAULT_PREDICTION = {
    "label": "good",
    "bbox": [],
    "reason": "",
    "root_cause": "",
    "improvement": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate Qwen3-VL-thinking on VisA with 0/1/3-shot prompts.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to VisA root directory.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save final JSON result.")
    parser.add_argument("--shots", type=str, default="0,1,3", help="Comma-separated shot settings, e.g. 0,1,3")
    parser.add_argument("--category", type=str, default=None, help="Single category name, e.g. pcb1.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category subset, optional.")
    parser.add_argument(
        "--visa_source",
        type=str,
        default="per_category_csv",
        choices=["per_category_csv", "split_csv"],
        help="VisA label source mode.",
    )
    parser.add_argument(
        "--visa_split_csv",
        type=str,
        default=None,
        help="Path to VisA split csv, e.g. split_csv/1cls.csv. Used when --visa_source split_csv.",
    )
    parser.add_argument(
        "--visa_split_set",
        type=str,
        default="test",
        help="Split name to evaluate in --visa_split_csv, e.g. test.",
    )
    parser.add_argument(
        "--max_samples_per_defect",
        type=int,
        default=None,
        help="Optional cap per (category, defect_type) for quick debugging.",
    )
    parser.add_argument(
        "--fixed_support_images",
        type=str,
        default=None,
        help="Fixed normal support images mapping, format: category=file1.jpg;category2=file2.jpg or category=file1.jpg,file2.jpg",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=16000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--metric_threshold_policy",
        type=str,
        default="youden",
        choices=["youden", "best_f1", "best_acc", "zero", "fixed"],
        help="Threshold policy for ACC/precision/recall/F1 from continuous anomaly scores.",
    )
    parser.add_argument(
        "--metric_threshold",
        type=float,
        default=0.0,
        help="Fixed threshold used when --metric_threshold_policy fixed.",
    )
    parser.add_argument(
        "--score_label_aggregation",
        type=str,
        default="max",
        choices=["max", "logsumexp"],
        help="How to aggregate anomaly candidate label token logprobs into one image anomaly score.",
    )
    parser.add_argument("--min_pixels", type=int, default=640 * 640)
    parser.add_argument("--max_pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation. Use flash_attention_2 for faster inference on supported GPUs.",
    )
    parser.add_argument(
        "--allow_attn_fallback",
        action="store_true",
        help="If set, fallback to sdpa when the requested attention implementation is unavailable.",
    )
    parser.add_argument(
        "--print_image_name",
        action="store_true",
        help="Print the current image file name being processed.",
    )
    parser.add_argument("--use_deco", action="store_true", help="Enable DeCo decoding strategy.")
    parser.add_argument("--alpha", type=float, default=0.6, help="DeCo alpha.")
    parser.add_argument("--threshold_top_p", type=float, default=0.9, help="DeCo top-p candidate cutoff.")
    parser.add_argument("--threshold_top_k", type=int, default=20, help="DeCo top-k candidate cutoff.")
    parser.add_argument(
        "--start_layer",
        type=int,
        default=20,
        help="DeCo early-exit layer range start (inclusive).",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=29,
        help="DeCo early-exit layer range end (exclusive).",
    )
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default="default",
        choices=["default", "defect_aware"],
        help="Anchor layer selection strategy inside DeCo.",
    )
    parser.add_argument("--lambda_norm", type=float, default=1.0, help="Weight for normal-token evidence penalty.")
    parser.add_argument("--eta_hedge", type=float, default=0.0, help="Weight for hedge-token evidence penalty.")
    parser.add_argument("--enable_hedge", action="store_true", help="Enable hedge-token penalty in defect-aware mode.")
    parser.add_argument("--use_logsumexp", action="store_true", help="Use logsumexp aggregation instead of max.")
    parser.add_argument("--gamma", type=float, default=8.0, help="Temperature for logsumexp aggregation.")
    parser.add_argument(
        "--anchor_defect_words",
        type=str,
        default=None,
        help="Comma-separated defect evidence lexicon. Optional; defaults are used when unset.",
    )
    parser.add_argument(
        "--anchor_norm_words",
        type=str,
        default=None,
        help="Comma-separated normal evidence lexicon. Optional; defaults are used when unset.",
    )
    parser.add_argument(
        "--anchor_hedge_words",
        type=str,
        default=None,
        help="Comma-separated hedge lexicon. Optional; defaults are used when unset.",
    )
    parser.add_argument("--deco_debug", action="store_true", help="Print per-layer defect-aware DeCo scores.")
    parser.add_argument("--anchor_smooth", action="store_true", help="Smooth defect-aware anchor scores before selecting anchor layer.")
    parser.add_argument("--anchor_remove_outliers", action="store_true", help="Remove isolated anchor-score outliers with Hampel filtering before smoothing.")
    parser.add_argument("--anchor_hampel_window", type=int, default=2, help="Local window radius for Hampel outlier detection.")
    parser.add_argument("--anchor_hampel_n_sigma", type=float, default=3.0, help="Hampel outlier threshold in scaled MAD units.")
    parser.add_argument("--anchor_smooth_window", type=int, default=3, help="Odd moving-average window for anchor-score smoothing.")
    parser.add_argument("--anchor_smooth_beta", type=float, default=0.7, help="Fusion weight for smoothed anchor scores.")
    return parser.parse_args()


def get_available_categories(dataset_root: Path) -> List[str]:
    return [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return files


def parse_shots(shots_text: str) -> List[int]:
    shots = []
    for chunk in shots_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        shot = int(chunk)
        if shot < 0:
            raise ValueError(f"Shot must be >= 0, got: {shot}")
        shots.append(shot)
    if len(shots) == 0:
        raise ValueError("No valid shots parsed from --shots.")
    return sorted(set(shots))


def parse_word_csv(words_text: Optional[str]) -> List[str]:
    if words_text is None:
        return []
    return [chunk.strip() for chunk in str(words_text).split(",") if chunk.strip()]


def parse_fixed_support_images(text: Optional[str]) -> Dict[str, List[str]]:
    """
    解析格式:
    pcb1=654.JPG;pcb3=101.JPG
    或
    pcb1=654.JPG,655.JPG;pcb3=101.JPG
    """
    if not text:
        return {}

    result: Dict[str, List[str]] = {}
    groups = [x.strip() for x in str(text).split(";") if x.strip()]
    for group in groups:
        if "=" not in group:
            continue
        category, refs = group.split("=", 1)
        category = category.strip()
        ref_list = [x.strip() for x in refs.split(",") if x.strip()]
        if category and ref_list:
            result[category] = ref_list
    return result


def merge_unique_words(*word_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for words in word_lists:
        for word in words:
            cleaned = str(word).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return merged


def load_visa_test_data(dataset_root: Path) -> Dict[str, List[str]]:
    test_json = dataset_root / "test_data.json"
    if not test_json.exists():
        raise FileNotFoundError(f"Missing test_data.json: {test_json}")

    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"test_data.json should be a list, got: {type(data)}")

    result: Dict[str, List[str]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        image_path = str(item.get("image", "")).replace("\\", "/").strip()
        if not image_path:
            continue

        parts = Path(image_path).parts
        category = None
        if len(parts) >= 2 and parts[0] == "VisA":
            category = parts[1]
        elif len(parts) >= 1:
            category = parts[1] if len(parts) > 1 else None

        if not category:
            continue

        result.setdefault(category, []).append(image_path)

    for k in list(result.keys()):
        uniq = []
        seen = set()
        for p in result[k]:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        result[k] = uniq

    print("Loaded test_data categories:", sorted(result.keys()))
    for k in sorted(result.keys()):
        print(f"[test_data] {k}: {len(result[k])} images")

    return result


def load_anomaly_prompt_map(category_dir: Path) -> Dict[str, List[str]]:
    json_path = category_dir / "anomaly_prompt.json"
    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {}

    prompt_map: Dict[str, List[str]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        image_key = None
        for k in ["image", "img", "img_path", "path", "file", "filename", "name"]:
            if k in item and str(item[k]).strip():
                image_key = str(item[k]).strip().replace("\\", "/")
                break

        prompt_text = str(item.get("prompt", "")).strip()
        if image_key is None or not prompt_text:
            continue

        labels = [x.strip() for x in prompt_text.split(",") if x.strip()]
        if len(labels) == 0:
            continue

        prompt_map[image_key] = labels
        prompt_map[Path(image_key).name] = labels

    return prompt_map


def load_visa_split_csv(
    dataset_root: Path,
    split_csv_path: Path,
    split_set: str,
    allowed_categories: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Missing VisA split csv: {split_csv_path}")

    allowed = set(allowed_categories or [])
    split_set_norm = str(split_set).strip().lower()
    rows = pd.read_csv(
        split_csv_path,
        header=None,
        names=["category", "split", "label", "image", "mask"],
        keep_default_na=False,
    )

    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}

    for _, row in rows.iterrows():
        category = str(row["category"]).strip()
        split = str(row["split"]).strip().lower()
        label = str(row["label"]).strip().lower()
        rel_image = str(row["image"]).replace("\\", "/").strip()
        rel_mask = str(row["mask"]).replace("\\", "/").strip()

        if not category or split != split_set_norm:
            continue
        if allowed and category not in allowed:
            continue
        if not rel_image:
            continue

        image_path = dataset_root / rel_image
        if not image_path.exists():
            print(f"[WARN] Missing split image: {image_path}")
            continue

        mask_path = dataset_root / rel_mask if rel_mask else None
        gt_label = "good" if label == "normal" else "anomaly"
        gt_labels = ["good"] if gt_label == "good" else ["anomaly"]

        if category not in category_meta:
            category_dir = dataset_root / category
            anomaly_prompt_map = load_anomaly_prompt_map(category_dir)
            defect_types = sorted(
                {
                    defect
                    for labels in anomaly_prompt_map.values()
                    for defect in labels
                    if str(defect).strip()
                }
            )
            if not defect_types:
                defect_types = ["anomaly"]

            support_good_paths = []
            image_anno_csv = category_dir / "image_anno.csv"
            if image_anno_csv.exists():
                df = pd.read_csv(image_anno_csv)
                if "label" in df.columns and "image" in df.columns:
                    df_train = df[df["label"] == "normal"]
                    support_good_paths = [
                        dataset_root / str(p)
                        for p in df_train["image"].tolist()
                        if (dataset_root / str(p)).exists()
                    ]

            category_meta[category] = {
                "defect_types": defect_types,
                "support_good_paths": support_good_paths,
            }

        samples.append(
            {
                "category": category,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path and mask_path.exists() else "",
                "ground_truth_label": gt_label,
                "ground_truth_labels": gt_labels,
                "split_label": label,
                "split": split,
            }
        )

    print(f"Loaded split csv samples: {len(samples)} from {split_csv_path} split={split_set}")
    return samples, category_meta


def discover_visa(
    dataset_root,
    allowed_categories=None,
    max_samples_per_defect=None,
    visa_source: str = "per_category_csv",
    visa_split_csv: Optional[str] = None,
    visa_split_set: str = "test",
):
    samples = []
    category_meta = {}

    dataset_root = Path(dataset_root)
    if visa_source == "split_csv":
        split_csv_path = Path(visa_split_csv) if visa_split_csv else dataset_root / "split_csv" / "1cls.csv"
        return load_visa_split_csv(
            dataset_root=dataset_root,
            split_csv_path=split_csv_path,
            split_set=visa_split_set,
            allowed_categories=allowed_categories,
        )

    test_data_map = load_visa_test_data(dataset_root)

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        if allowed_categories and category not in allowed_categories:
            continue

        category_test_images = test_data_map.get(category, [])
        if len(category_test_images) == 0:
            continue

        anomaly_prompt_map = load_anomaly_prompt_map(category_dir)

        defect_types = sorted(
            {
                label
                for labels in anomaly_prompt_map.values()
                for label in labels
            }
        )

        csv_path = category_dir / "image_anno.csv"
        support_good_paths = []
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "label" in df.columns and "image" in df.columns:
                df_train = df[df["label"] == "normal"]
                support_good_paths = [
                    dataset_root / str(p)
                    for p in df_train["image"].tolist()
                    if (dataset_root / str(p)).exists()
                ]

        category_meta[category] = {
            "defect_types": defect_types,
            "support_good_paths": support_good_paths,
        }

        category_samples = []
        for rel_path in category_test_images:
            rel_path = str(rel_path).replace("\\", "/").strip()

            rel_path_no_prefix = rel_path
            if rel_path_no_prefix.startswith("VisA/"):
                rel_path_no_prefix = rel_path_no_prefix[len("VisA/"):]

            candidates = [
                dataset_root / rel_path_no_prefix,
                dataset_root / rel_path,
                category_dir / "Data" / "Images" / "Normal" / Path(rel_path).name,
                category_dir / "Data" / "Images" / "Anomaly" / Path(rel_path).name,
            ]

            image_path = None
            for cand in candidates:
                if cand.exists():
                    image_path = cand
                    break

            if image_path is None:
                print(f"[WARN] Missing image for category={category}: {rel_path}")
                continue

            image_name = image_path.name
            rel_norm = rel_path.replace("\\", "/")

            labels = anomaly_prompt_map.get(rel_norm)
            if labels is None:
                labels = anomaly_prompt_map.get(image_name)

            if labels is None or len(labels) == 0:
                gt_label = "good"
                gt_labels = ["good"]
            else:
                gt_labels = labels
                gt_label = labels[0]

            category_samples.append(
                {
                    "category": category,
                    "image_path": str(image_path),
                    "ground_truth_label": gt_label,
                    "ground_truth_labels": gt_labels,
                }
            )

        if max_samples_per_defect is not None:
            reduced = []
            buckets = {}
            for item in category_samples:
                key = item["ground_truth_label"]
                buckets.setdefault(key, []).append(item)

            for key, items in buckets.items():
                if key == "good":
                    reduced.extend(items)
                else:
                    if len(items) > max_samples_per_defect:
                        items = random.sample(items, max_samples_per_defect)
                    reduced.extend(items)
            category_samples = reduced

        samples.extend(category_samples)

    print("Loaded samples:", len(samples))
    return samples, category_meta


def expand_defect_words_with_morphology(defect_types: List[str]) -> List[str]:
    base_words = ["defect", "anomaly", "abnormal"]

    morphology_map = {
        "excessive solder": [
            "solder", "excessive", "extra", "overflow", "residue", "blob", "metal residue"
        ],
        "missing component": [
            "missing", "absent", "lost", "not present", "component missing"
        ],
        "bent component": [
            "bent", "deformed", "misaligned", "crooked", "tilted"
        ],
        "scratch": [
            "scratch", "scratched", "mark", "surface damage", "damaged", "line", "streak"
        ],
        "crack": [
            "crack", "cracked", "fracture", "split", "broken line"
        ],
        "dent": [
            "dent", "dented", "depression", "pressed"
        ],
        "stain": [
            "stain", "spot", "smudge", "dirty", "contamination"
        ],
    }

    expanded = []
    seen = set()

    def add_word(w: str):
        k = str(w).strip().lower()
        if not k or k in seen:
            return
        seen.add(k)
        expanded.append(str(w).strip())

    for w in base_words:
        add_word(w)

    for defect in defect_types:
        defect = str(defect).strip()
        add_word(defect)

        key = defect.lower()
        if key in morphology_map:
            for w in morphology_map[key]:
                add_word(w)
        else:
            for token in key.replace("-", " ").replace("_", " ").split():
                add_word(token)

    return expanded


def build_messages(category: str, defect_types: List[str], shot: int) -> List[Dict[str, Any]]:
    candidate_labels = ["good"] + defect_types
    candidate_text = ", ".join(candidate_labels)

    reference_text = ""
    if shot > 0:
        reference_text = f"""
{shot} normal reference image(s) from the same category will be provided first.
All reference images are normal samples.
You may compare the query image with these normal reference images when helpful.
"""

    instruction = f"""
You are the chief expert in image anomaly detection.
You are highly sensitive to subtle abnormalities in images and can accurately identify abnormal visual evidence.
Anomaly refers to defects that affect normal functionality. Do not be distracted by insignificant issues.

Current category: {category}
Candidate labels: [{candidate_text}]
{reference_text}
This is a local anomaly detection task, not ordinary image classification.

Decision rules:
1. Focus on local abnormal visual evidence rather than only global appearance.
2. Abnormalities may be subtle, local, small, or low-contrast.
3. You must choose exactly ONE label from the candidate list.
4. Do not discuss multiple possibilities.
5. Do not repeat defect checking.
6. Do not think step by step in the output.
7. Do not output your reasoning process.
8. Do not self-correct in the output.
9. Return the final answer immediately.
10. Use "good" ONLY if no suspicious local abnormality is visible anywhere in the image.

Output rules:
- You have a limited number of output tokens, so you must return one complete JSON object immediately and avoid any extra text. Set max output tokens to 1024.
- Output JSON only.
- No extra text before or after JSON.
- Do not output Markdown code fences.
- Make sure the JSON object is complete and properly closed.
- Keep the output short.
- If label is "good", "reason", "root_cause", and "improvement" must be empty strings.
- If label is anomalous, provide one short and direct reason.

Output format:
{{
  "label": "",
  "reason": "",
  "root_cause": "",
  "improvement": ""
}}
"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    for i in range(shot):
        content.append({"type": "text", "text": f"Normal reference image {i + 1} (this is a normal sample):"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Query image:"})
    content.append({"type": "image"})
    return [{"role": "user", "content": content}]

# def build_messages(category: str, defect_types: List[str], shot: int) -> List[Dict[str, Any]]:
#     candidate_labels = ["good"] + defect_types
#     candidate_text = ", ".join(candidate_labels)

#     schema_text = (
#         "{\n"
#         '  "label": "",\n'
#         '  "reason": "",\n'
#         '  "root_cause": "",\n'
#         '  "improvement": ""\n'
#         "}"
#     )

#     instruction = (
#         f"Classify the query image into exactly one label from this list: [{candidate_text}].\n"
#         f"Current category: {category}.\n\n"
#         "Rules:\n"
#         "1. Output one JSON object only.\n"
#         "2. The first character of your answer must be '{'.\n"
#         "3. Do not output analysis, comparison, reasoning, explanation, or step-by-step thinking.\n"
#         "4. Do not use words such as: let's, wait, maybe, seems, appears, likely, because.\n"
#         "5. Do not output <think> or </think>.\n"
#         "6. The label must be exactly one of the candidate labels.\n"
#         "7. If the image is normal, label must be good and the other fields must be empty strings.\n"
#         "8. If the image is abnormal, fill reason/root_cause/improvement briefly, each within 12 words.\n\n"
#         f"JSON format:\n{schema_text}"
#     )

#     content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
#     for i in range(shot):
#         content.append({"type": "text", "text": f"Normal reference image {i + 1}:"})
#         content.append({"type": "image"})
#     content.append({"type": "text", "text": "Query image:"})
#     content.append({"type": "image"})
#     content.append({"type": "text", "text": "Return JSON now. No reasoning."})
#     return [{"role": "user", "content": content}]


def clean_generation_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    fence_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if len(fence_blocks) > 0:
        text = fence_blocks[-1].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def extract_json_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    stack: List[str] = []
    start_idx: Optional[int] = None
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    reverse_pairs = {"}": "{", "]": "["}

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in pairs:
            if len(stack) == 0:
                start_idx = idx
            stack.append(ch)
            continue

        if ch in reverse_pairs and len(stack) > 0 and stack[-1] == reverse_pairs[ch]:
            stack.pop()
            if len(stack) == 0 and start_idx is not None:
                blocks.append(text[start_idx : idx + 1])
                start_idx = None
    return blocks


def normalize_parsed_object(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        out = dict(obj)
        if "bbox_2d" in out and "bbox" not in out:
            out["bbox"] = out["bbox_2d"]
        nested = out.get("result")
        if isinstance(nested, dict):
            for key, value in nested.items():
                out.setdefault(key, value)
        return out

    if isinstance(obj, list) and len(obj) > 0:
        if len(obj) == 1 and isinstance(obj[0], dict):
            return normalize_parsed_object(obj[0])
        if all(isinstance(item, dict) for item in obj):
            for item in obj:
                if any(k in item for k in ("label", "defect_type", "class", "prediction", "bbox", "bbox_2d")):
                    return normalize_parsed_object(item)
            return normalize_parsed_object(obj[0])
    return None


def try_parse_json_candidate(candidate: str) -> Tuple[Optional[Dict[str, Any]], str]:
    s = candidate.strip()
    if not s:
        return None, "none"

    parse_attempts: List[Tuple[str, str]] = [("json", s)]
    s_no_trailing_comma = re.sub(r",\s*([}\]])", r"\1", s)
    if s_no_trailing_comma != s:
        parse_attempts.append(("json_no_trailing_comma", s_no_trailing_comma))
    parse_attempts.append(("ast_literal", s))
    if s_no_trailing_comma != s:
        parse_attempts.append(("ast_literal_no_trailing_comma", s_no_trailing_comma))

    for name, payload in parse_attempts:
        try:
            if name.startswith("json"):
                parsed = json.loads(payload)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
            else:
                parsed = ast.literal_eval(payload)
            normalized = normalize_parsed_object(parsed)
            if normalized is not None:
                return normalized, name
        except Exception:
            continue
    return None, "none"


def recover_prediction_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    text = clean_generation_text(raw_text)
    candidates = [text]
    candidates.extend(reversed(extract_json_blocks(text)))
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = cand.strip()
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(key)

    for cand in unique_candidates:
        parsed, _ = try_parse_json_candidate(cand)
        if parsed is not None:
            return parsed

    label_match = re.search(
        r"(?is)(?:label|defect_type|class|prediction|result)\s*[:=]\s*[\"']?([^\"'\n,\}\]]+)",
        text,
    )
    bbox_match = re.search(
        r"(?is)(?:bbox|box|rect|bbox_2d)\s*[:=]\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
        text,
    )
    reason_match = re.search(r"(?is)(?:reason|why|evidence)\s*[:=]\s*[\"']?([^\"'\n\}]+)", text)
    root_match = re.search(r"(?is)(?:root_cause|cause)\s*[:=]\s*[\"']?([^\"'\n\}]+)", text)
    improve_match = re.search(r"(?is)(?:improvement|action)\s*[:=]\s*[\"']?([^\"'\n\}]+)", text)

    if not any([label_match, bbox_match, reason_match, root_match, improve_match]):
        return None

    recovered = dict(DEFAULT_PREDICTION)
    if label_match:
        recovered["label"] = label_match.group(1).strip()
    if bbox_match:
        recovered["bbox"] = [float(bbox_match.group(i)) for i in range(1, 5)]
    if reason_match:
        recovered["reason"] = reason_match.group(1).strip()
    if root_match:
        recovered["root_cause"] = root_match.group(1).strip()
    if improve_match:
        recovered["improvement"] = improve_match.group(1).strip()
    return recovered


def parse_model_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    text = clean_generation_text(raw_text)
    candidates = [text]
    candidates.extend(reversed(extract_json_blocks(text)))

    seen = set()
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        parsed, parse_source = try_parse_json_candidate(key)
        if parsed is not None:
            return parsed, parse_source
    return None, "none"


def get_first_existing_key(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def normalize_bbox(raw_bbox: Any, width: int, height: int) -> List[int]:
    if raw_bbox is None:
        return []

    if isinstance(raw_bbox, dict):
        x1 = get_first_existing_key(raw_bbox, ["x1", "xmin", "left"], None)
        y1 = get_first_existing_key(raw_bbox, ["y1", "ymin", "top"], None)
        x2 = get_first_existing_key(raw_bbox, ["x2", "xmax", "right"], None)
        y2 = get_first_existing_key(raw_bbox, ["y2", "ymax", "bottom"], None)
        if None in (x1, y1, x2, y2):
            return []
        raw_bbox = [x1, y1, x2, y2]

    if not (isinstance(raw_bbox, list) or isinstance(raw_bbox, tuple)) or len(raw_bbox) != 4:
        return []

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return []

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def map_label(raw_label: Any, defect_types: List[str]) -> str:
    if not isinstance(raw_label, str):
        return "good"
    s = raw_label.strip()
    s_lower = s.lower()
    if s_lower in {"good", "normal", "ok", "no defect", "non-defective"}:
        return "good"

    lower_map = {d.lower(): d for d in defect_types}
    if s_lower in lower_map:
        return lower_map[s_lower]
    for defect in defect_types:
        d_lower = defect.lower()
        if d_lower in s_lower or s_lower in d_lower:
            return defect
    return "good"



def normalize_text_for_label_match(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_string_field_from_text(text: str, field_names: List[str]) -> str:
    if not text:
        return ""
    for field in field_names:
        pattern = rf'(?is)["\']?{re.escape(field)}["\']?\s*[:=]\s*["\']([^"\']*)["\']'
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def reason_indicates_anomaly(reason: str) -> bool:
    """
    Binary fallback copied from the visa_label_summary-style logic:
    if JSON/label parsing would otherwise return good, inspect the generated reason.
    """
    r = normalize_text_for_label_match(reason)
    if not r:
        return False

    normal_only_patterns = [
        "no anomaly",
        "no defect",
        "no abnormal",
        "no visible defect",
        "no visible anomaly",
        "appears normal",
        "looks normal",
        "is normal",
        "normal sample",
        "good sample",
        "uniform appearance",
        "nothing abnormal",
        "without defect",
    ]
    if any(p in r for p in normal_only_patterns):
        return False

    anomaly_keywords = [
        "visible", "spot", "spots", "hole", "holes", "scratch", "scratches",
        "missing", "damaged", "damage", "broken", "breakage", "burnt", "stuck",
        "different", "colour", "color", "dark", "mark", "defect", "anomaly",
        "abnormal", "irregular", "foreign", "crack", "dent", "contamination",
        "consistent with", "surface", "corner", "edge", "excessive", "extra",
        "residue", "blob", "deformed", "misaligned", "crooked", "tilted",
    ]
    if any(k in r for k in anomaly_keywords):
        return True

    # In the current prompt, good samples should have empty reason/root_cause/improvement.
    # Therefore a non-empty reason that is not explicitly normal is treated as anomaly evidence.
    return len(r.split()) >= 3


def find_known_label_in_text(text: str, defect_types: List[str]) -> str:
    raw = str(text or "").strip()
    norm = normalize_text_for_label_match(raw)

    labels = ["good"] + [str(x).strip() for x in defect_types if str(x).strip()]
    labels = sorted(set(labels), key=lambda x: len(x), reverse=True)

    # Direct field extraction, e.g. "label": "small holes"
    label_value = extract_string_field_from_text(raw, ["label", "defect_type", "class", "prediction", "result"])
    if label_value:
        label_value_norm = normalize_text_for_label_match(label_value)
        for label in labels:
            if label_value_norm == normalize_text_for_label_match(label):
                return label

    # Raw text may start with the label value.
    for label in labels:
        label_norm = normalize_text_for_label_match(label)
        if not label_norm:
            continue
        if norm == label_norm:
            return label
        if norm.startswith(label_norm + '"') or norm.startswith(label_norm + "'"):
            return label
        if norm.startswith(label_norm + ",") or norm.startswith(label_norm + "}") or norm.startswith(label_norm + "]"):
            return label
        if norm.startswith(label_norm + " "):
            return label

    # Conservative substring fallback for complete multi-word labels.
    for label in labels:
        label_norm = normalize_text_for_label_match(label)
        if " " in label_norm and label_norm in norm:
            return label

    return ""


def select_label_from_reason(reason: str, defect_types: List[str]) -> str:
    reason_norm = normalize_text_for_label_match(reason)
    if not reason_norm:
        return ""

    for defect in sorted(defect_types, key=lambda x: len(str(x)), reverse=True):
        d_norm = normalize_text_for_label_match(defect)
        if d_norm and d_norm in reason_norm:
            return defect

    best_label = ""
    best_overlap = 0
    for defect in defect_types:
        d_norm = normalize_text_for_label_match(defect)
        words = [w for w in d_norm.split() if len(w) >= 4]
        if not words:
            continue
        overlap = sum(1 for w in words if w in reason_norm)
        if overlap > best_overlap and overlap >= max(1, min(2, len(words))):
            best_overlap = overlap
            best_label = defect

    return best_label



def normalize_prediction(
    parsed: Optional[Dict[str, Any]],
    parse_source: str,
    raw_text: str,
    defect_types: List[str],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    width, height = image_size

    strict_parse_ok = parse_source.startswith("json")
    extraction_ok = parsed is not None

    if parsed is None:
        parsed = recover_prediction_from_text(raw_text)
        if parsed is not None:
            parse_source = "recovered"
            extraction_ok = True
        else:
            parsed = dict(DEFAULT_PREDICTION)
            parse_source = "default"
            extraction_ok = False

    raw_label = get_first_existing_key(parsed, ["label", "defect_type", "class", "prediction", "result"], "good")
    label = map_label(raw_label, defect_types)

    bbox = normalize_bbox(get_first_existing_key(parsed, ["bbox", "box", "rect"], None), width, height)
    reason = str(get_first_existing_key(parsed, ["reason", "why", "evidence"], "") or "")
    root_cause = str(get_first_existing_key(parsed, ["root_cause", "cause"], "") or "")
    improvement = str(get_first_existing_key(parsed, ["improvement", "action"], "") or "")

    # visa_label_summary-style binary fallback:
    # If label parsing maps the output to good, but the generated reason clearly describes
    # abnormal evidence, keep the prediction as anomaly instead of falling back to good.
    reason_flag = reason_indicates_anomaly(reason)

    if label == "good":
        # First try to recover a known defect type from the model text/reason.
        recovered_label = ""
        if reason_flag:
            recovered_label = select_label_from_reason(reason, defect_types)
            if not recovered_label:
                recovered_label = find_known_label_in_text(raw_text, defect_types)

        if recovered_label and recovered_label != "good":
            label = recovered_label
            raw_label = recovered_label
            parse_source = "reason_indicates_anomaly"
            extraction_ok = True
        elif reason_flag:
            # Binary anomaly fallback. Subtype is unknown/wrong, but binary prediction should not be good.
            label = "anomaly"
            raw_label = "anomaly"
            parse_source = "reason_indicates_anomaly"
            extraction_ok = True
        else:
            bbox = []
            reason = ""
            root_cause = ""
            improvement = ""

    return {
        "label": label,
        "bbox": bbox,
        "reason": reason,
        "root_cause": root_cause,
        "improvement": improvement,
        "raw_label": str(raw_label),
        "parse_ok": True,
        "strict_parse_ok": strict_parse_ok,
        "extraction_ok": extraction_ok,
        "parse_source": parse_source,
        "reason_indicates_anomaly": reason_flag,
        "raw_text": raw_text,
    }


def first_token_ids_for_labels(tokenizer: Any, labels: List[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for label in labels:
        clean = str(label).strip()
        if not clean:
            continue
        encoded = tokenizer(clean, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            result[clean] = int(ids[0])
    return result


def get_nested_attr(obj: Any, dotted_path: str) -> Any:
    current = obj
    for name in dotted_path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def get_final_norm_module(model: Any) -> Any:
    for path in (
        "model.norm",
        "language_model.model.norm",
        "model.language_model.norm",
        "model.language_model.model.norm",
        "transformer.norm",
        "base_model.norm",
    ):
        module = get_nested_attr(model, path)
        if module is not None:
            return module
    return None


def candidate_logits_from_hidden(model: Any, hidden: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("model.get_output_embeddings() returned None")
    token_tensor = torch.tensor(token_ids, device=hidden.device, dtype=torch.long)
    weight = output_embeddings.weight.index_select(0, token_tensor).to(hidden.dtype)
    logits = torch.matmul(weight, hidden)
    bias = getattr(output_embeddings, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, token_tensor).to(logits.dtype)
    return logits.float()


def aggregate_values(values: torch.Tensor, mode: str) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(float("-inf"), device=values.device)
    if mode == "logsumexp":
        return torch.logsumexp(values.float(), dim=0)
    return torch.max(values.float(), dim=0).values


def build_binary_label_logits(
    token_logits: torch.Tensor,
    token_ids: List[int],
    good_ids: List[int],
    anomaly_ids: List[int],
    aggregation: str,
) -> torch.Tensor:
    positions = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
    good_values = token_logits[[positions[token_id] for token_id in good_ids if token_id in positions]]
    anomaly_values = token_logits[[positions[token_id] for token_id in anomaly_ids if token_id in positions]]
    return torch.stack(
        [
            aggregate_values(good_values, aggregation),
            aggregate_values(anomaly_values, aggregation),
        ]
    )


def compute_label_likelihood_score(
    model: Any,
    processor: Any,
    text: str,
    model_images: List[Image.Image],
    image_resolution_kwargs: Dict[str, int],
    defect_types: List[str],
    aggregation: str = "max",
    use_deco: bool = False,
    deco_alpha: float = 0.6,
    deco_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return {
            "anomaly_score": 0.0,
            "score_error": "processor has no tokenizer",
            "score_source": "continuous_label_likelihood_score",
        }

    anomaly_labels = [
        str(label).strip()
        for label in defect_types
        if str(label).strip() and str(label).strip().lower() != "good"
    ]
    if not anomaly_labels:
        anomaly_labels = ["anomaly"]

    good_token_ids = first_token_ids_for_labels(tokenizer, ["good"])
    anomaly_token_ids = first_token_ids_for_labels(tokenizer, anomaly_labels)
    if not good_token_ids or not anomaly_token_ids:
        return {
            "anomaly_score": 0.0,
            "score_error": "failed to tokenize scoring labels",
            "score_source": "continuous_label_likelihood_score",
        }

    score_prefix = text + '{\n  "label": "'
    score_inputs = processor(
        text=[score_prefix],
        images=model_images,
        return_tensors="pt",
        **image_resolution_kwargs,
    )
    score_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in score_inputs.items()}

    with torch.no_grad():
        outputs = model(
            **score_inputs,
            output_hidden_states=use_deco,
            return_dict=True,
        )

    final_logits = outputs.logits[0, -1].float()
    log_probs = torch.log_softmax(final_logits, dim=-1)
    good_items = [
        {"label": label, "token_id": token_id, "logprob": float(log_probs[token_id].detach().cpu())}
        for label, token_id in good_token_ids.items()
    ]
    anomaly_items = [
        {"label": label, "token_id": token_id, "logprob": float(log_probs[token_id].detach().cpu())}
        for label, token_id in anomaly_token_ids.items()
    ]
    good_logprob = max(item["logprob"] for item in good_items)

    if aggregation == "logsumexp":
        vals = torch.tensor([item["logprob"] for item in anomaly_items], dtype=torch.float32)
        anomaly_logprob = float(torch.logsumexp(vals, dim=0).item())
        best_item = max(anomaly_items, key=lambda x: x["logprob"])
    else:
        best_item = max(anomaly_items, key=lambda x: x["logprob"])
        anomaly_logprob = float(best_item["logprob"])

    anomaly_score = anomaly_logprob - good_logprob
    score_data = {
        "anomaly_score": float(anomaly_score),
        "base_anomaly_score": float(anomaly_score),
        "anomaly_probability_sigmoid": float(torch.sigmoid(torch.tensor(anomaly_score)).item()),
        "good_logprob": float(good_logprob),
        "anomaly_logprob": float(anomaly_logprob),
        "best_anomaly_label": best_item["label"],
        "best_anomaly_token_id": best_item["token_id"],
        "score_source": "continuous_label_likelihood_score",
        "score_prefix": score_prefix,
        "score_label_aggregation": aggregation,
        "good_label_items": good_items,
        "top_anomaly_label_items": sorted(anomaly_items, key=lambda x: x["logprob"], reverse=True)[:10],
    }
    if not use_deco:
        return score_data

    try:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("model output has no hidden_states")
        final_norm = get_final_norm_module(model)
        if final_norm is None:
            raise RuntimeError("could not locate final norm module for intermediate layer logits")

        good_ids = [int(x) for x in good_token_ids.values()]
        anomaly_ids = [int(x) for x in anomaly_token_ids.values()]
        candidate_ids = sorted(set(good_ids + anomaly_ids))
        final_candidate_logits = final_logits[candidate_ids]
        final_binary_logits = build_binary_label_logits(
            token_logits=final_candidate_logits,
            token_ids=candidate_ids,
            good_ids=good_ids,
            anomaly_ids=anomaly_ids,
            aggregation=aggregation,
        )

        layer_rows = []
        best_row = None
        for layer_idx in deco_layers or []:
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                continue
            hidden = hidden_states[layer_idx][0, -1]
            hidden = final_norm(hidden)
            layer_candidate_logits = candidate_logits_from_hidden(model, hidden, candidate_ids)
            layer_binary_logits = build_binary_label_logits(
                token_logits=layer_candidate_logits,
                token_ids=candidate_ids,
                good_ids=good_ids,
                anomaly_ids=anomaly_ids,
                aggregation=aggregation,
            )
            layer_probs = torch.softmax(layer_binary_logits.float(), dim=-1)
            confidence = float(torch.max(layer_probs).detach().cpu())
            row = {
                "layer": int(layer_idx),
                "good_logit": float(layer_binary_logits[0].detach().cpu()),
                "anomaly_logit": float(layer_binary_logits[1].detach().cpu()),
                "good_prob": float(layer_probs[0].detach().cpu()),
                "anomaly_prob": float(layer_probs[1].detach().cpu()),
                "confidence": confidence,
            }
            layer_rows.append(row)
            if best_row is None or confidence > best_row["confidence"]:
                best_row = row

        if best_row is None:
            raise RuntimeError("no valid DeCo score layers")

        selected_binary_logits = torch.tensor(
            [best_row["good_logit"], best_row["anomaly_logit"]],
            device=final_binary_logits.device,
            dtype=final_binary_logits.dtype,
        )
        calibrated_logits = final_binary_logits + float(deco_alpha) * float(best_row["confidence"]) * selected_binary_logits
        deco_score = float((calibrated_logits[1] - calibrated_logits[0]).detach().cpu())
        score_data.update(
            {
                "anomaly_score": deco_score,
                "deco_anomaly_score": deco_score,
                "anomaly_probability_sigmoid": float(torch.sigmoid(torch.tensor(deco_score)).item()),
                "deco_anomaly_probability_sigmoid": float(torch.sigmoid(torch.tensor(deco_score)).item()),
                "deco_enabled_for_score": True,
                "deco_alpha": float(deco_alpha),
                "deco_score_layer_range": [int(x) for x in (deco_layers or [])],
                "deco_selected_layer": int(best_row["layer"]),
                "deco_selected_confidence": float(best_row["confidence"]),
                "deco_final_good_logit": float(final_binary_logits[0].detach().cpu()),
                "deco_final_anomaly_logit": float(final_binary_logits[1].detach().cpu()),
                "deco_calibrated_good_logit": float(calibrated_logits[0].detach().cpu()),
                "deco_calibrated_anomaly_logit": float(calibrated_logits[1].detach().cpu()),
                "deco_layer_scores": layer_rows,
                "score_source": "deco_continuous_label_likelihood_score",
            }
        )
    except Exception as exc:
        score_data.update(
            {
                "deco_enabled_for_score": False,
                "deco_score_error": str(exc),
                "score_source": "continuous_label_likelihood_score",
            }
        )

    return score_data


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def binary_auroc(labels: List[int], scores: List[float]) -> Optional[float]:
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


def select_threshold(records: List[Tuple[int, float]], policy: str, fixed_threshold: float = 0.0) -> float:
    if policy == "zero":
        return 0.0
    if policy == "fixed":
        return float(fixed_threshold)
    if not records:
        return float(fixed_threshold)

    scores = sorted({float(score) for _, score in records})
    candidates = [scores[0] - 1e-12]
    candidates.extend(scores)
    candidates.append(scores[-1] + 1e-12)

    def stats_at(threshold: float) -> Tuple[int, int, int, int]:
        tp = sum(1 for y, s in records if y == 1 and s >= threshold)
        tn = sum(1 for y, s in records if y == 0 and s < threshold)
        fp = sum(1 for y, s in records if y == 0 and s >= threshold)
        fn = sum(1 for y, s in records if y == 1 and s < threshold)
        return tp, tn, fp, fn

    def safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

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


def compute_binary_metrics(
    records: List[Tuple[int, float]],
    threshold_policy: str = "youden",
    fixed_threshold: float = 0.0,
    score_source: str = "continuous_label_likelihood_score",
) -> Dict[str, Any]:
    total = len(records)
    threshold = select_threshold(records, threshold_policy, fixed_threshold)
    tp = sum(1 for y, s in records if y == 1 and s >= threshold)
    tn = sum(1 for y, s in records if y == 0 and s < threshold)
    fp = sum(1 for y, s in records if y == 0 and s >= threshold)
    fn = sum(1 for y, s in records if y == 1 and s < threshold)

    def safe_div(num: float, den: float) -> Optional[float]:
        return num / den if den > 0 else None

    precision_anomaly = safe_div(tp, tp + fp)
    recall_anomaly = safe_div(tp, tp + fn)
    f1_anomaly = (
        2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
        if precision_anomaly is not None and recall_anomaly is not None and (precision_anomaly + recall_anomaly) > 0
        else None
    )

    precision_normal = safe_div(tn, tn + fn)
    recall_normal = safe_div(tn, tn + fp)
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if precision_normal is not None and recall_normal is not None and (precision_normal + recall_normal) > 0
        else None
    )

    macro_precision_vals = [x for x in [precision_normal, precision_anomaly] if x is not None]
    macro_recall_vals = [x for x in [recall_normal, recall_anomaly] if x is not None]
    macro_f1_vals = [x for x in [f1_normal, f1_anomaly] if x is not None]
    anomaly_total = tp + fn
    normal_total = tn + fp
    acc = safe_div(tp + tn, total)
    labels = [y for y, _ in records]
    scores = [s for _, s in records]

    return {
        "total": total,
        "acc": acc,
        "AUROC": binary_auroc(labels, scores),
        "auroc_score_source": score_source,
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": sum(macro_recall_vals) / len(macro_recall_vals) if macro_recall_vals else None,
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": sum(macro_precision_vals) / len(macro_precision_vals) if macro_precision_vals else None,
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": sum(macro_f1_vals) / len(macro_f1_vals) if macro_f1_vals else None,
        "recall_micro": acc,
        "precision_micro": acc,
        "F1_micro": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": anomaly_total,
        "normal_total": normal_total,
    }


def compute_summary(
    results: List[Dict[str, Any]],
    threshold_policy: str = "youden",
    fixed_threshold: float = 0.0,
) -> Dict[str, Any]:
    by_shot: Dict[int, Dict[str, float]] = {}
    binary_records_by_shot: Dict[int, List[Tuple[int, float]]] = {}
    binary_records_by_shot_category: Dict[Tuple[int, str], List[Tuple[int, float]]] = {}
    score_sources_by_shot: Dict[int, set] = {}
    score_sources_by_shot_category: Dict[Tuple[int, str], set] = {}
    for item in results:
        shot = int(item["shot"])
        gt = item["ground_truth"]["label"]
        gt_labels = item["ground_truth"].get("labels", [gt])
        pred = item["prediction"]["label"]
        category = str(item.get("category", ""))

        if shot not in by_shot:
            by_shot[shot] = {
                "total": 0,
                "binary_correct": 0,
                "subtype_correct_on_anomaly": 0,
                "anomaly_count": 0,
                "parse_ok_count": 0,
                "strict_parse_ok_count": 0,
                "extraction_ok_count": 0,
            }
        by_shot[shot]["total"] += 1
        by_shot[shot]["parse_ok_count"] += 1 if item["prediction"]["parse_ok"] else 0
        by_shot[shot]["strict_parse_ok_count"] += 1 if item["prediction"].get("strict_parse_ok", False) else 0
        by_shot[shot]["extraction_ok_count"] += 1 if item["prediction"].get("extraction_ok", False) else 0
        gt_is_anomaly = gt != "good"
        pred_is_anomaly = pred != "good"
        y_true = 1 if gt_is_anomaly else 0
        score_obj = item.get("score", {})
        score = score_obj.get("anomaly_score", None)
        if score is None:
            score = 1.0 if pred_is_anomaly else 0.0
        score_source = str(score_obj.get("score_source", "continuous_label_likelihood_score"))
        binary_records_by_shot.setdefault(shot, []).append((y_true, float(score)))
        binary_records_by_shot_category.setdefault((shot, category), []).append((y_true, float(score)))
        score_sources_by_shot.setdefault(shot, set()).add(score_source)
        score_sources_by_shot_category.setdefault((shot, category), set()).add(score_source)
        if gt_is_anomaly == pred_is_anomaly:
            by_shot[shot]["binary_correct"] += 1
        if gt_is_anomaly:
            by_shot[shot]["anomaly_count"] += 1
            if pred in gt_labels:
                by_shot[shot]["subtype_correct_on_anomaly"] += 1

    summary: Dict[str, Any] = {}
    for shot, stats in sorted(by_shot.items()):
        total = max(1, int(stats["total"]))
        anomaly_count = int(stats["anomaly_count"])
        summary[str(shot)] = {
            "total": int(stats["total"]),
            "binary_acc": stats["binary_correct"] / total,
            "binary_metrics": compute_binary_metrics(
                binary_records_by_shot.get(shot, []),
                threshold_policy=threshold_policy,
                fixed_threshold=fixed_threshold,
                score_source=(
                    next(iter(score_sources_by_shot.get(shot, set())))
                    if len(score_sources_by_shot.get(shot, set())) == 1
                    else "mixed"
                ),
            ),
            "per_category_binary_metrics": {
                category: compute_binary_metrics(
                    records,
                    threshold_policy=threshold_policy,
                    fixed_threshold=fixed_threshold,
                    score_source=(
                        next(iter(score_sources_by_shot_category.get((shot_key, category), set())))
                        if len(score_sources_by_shot_category.get((shot_key, category), set())) == 1
                        else "mixed"
                    ),
                )
                for (shot_key, category), records in sorted(binary_records_by_shot_category.items())
                if shot_key == shot
            },
            "subtype_acc_on_anomaly": (
                stats["subtype_correct_on_anomaly"] / anomaly_count if anomaly_count > 0 else None
            ),
            "json_parse_success_rate": stats["strict_parse_ok_count"] / total,
            "extraction_success_rate": stats["extraction_ok_count"] / total,
            "normalized_output_rate": stats["parse_ok_count"] / total,
        }
    return summary


def flatten_metric_rows(output_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = output_data.get("summary", {})
    for shot, shot_summary in sorted(summary.items(), key=lambda kv: int(kv[0])):
        overall_metrics = shot_summary.get("binary_metrics", {})
        rows.append({"shot": shot, "scope": "overall", "category": "ALL", **overall_metrics})
        per_category = shot_summary.get("per_category_binary_metrics", {})
        for category, metrics in sorted(per_category.items()):
            rows.append({"shot": shot, "scope": "category", "category": category, **metrics})
    return rows


def save_metrics_csv(output_json_path: Path, output_data: Dict[str, Any]) -> None:
    rows = flatten_metric_rows(output_data)
    if not rows:
        return
    csv_path = output_json_path.with_name(output_json_path.stem + "_metrics_summary.csv")
    fieldnames = [
        "shot", "scope", "category", "total", "acc", "AUROC",
        "threshold_policy", "threshold",
        "recall_normal", "recall_anomaly", "recall_all",
        "precision_normal", "precision_anomaly", "precision_all",
        "F1_normal", "F1_anomaly", "F1_all",
        "recall_micro", "precision_micro", "F1_micro",
        "tp", "tn", "fp", "fn", "pred_anomaly", "pred_normal",
        "anomaly_total", "normal_total", "auroc_score_source",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics summary CSV to: {csv_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    available_categories = get_available_categories(dataset_root)
    selected_categories: List[str] = []
    if args.category:
        selected_categories.append(args.category.strip())
    if args.categories:
        selected_categories.extend([x.strip() for x in args.categories.split(",") if x.strip()])

    selected_categories = list(dict.fromkeys([c for c in selected_categories if c]))
    allowed_categories = selected_categories if len(selected_categories) > 0 else None
    if allowed_categories is not None:
        invalid_categories = [c for c in allowed_categories if c not in available_categories]
        if len(invalid_categories) > 0:
            raise ValueError(
                f"Invalid category names: {invalid_categories}. "
                f"Available categories: {available_categories}"
            )

    categories_for_run = allowed_categories if allowed_categories is not None else available_categories
    print(f"Running categories: {categories_for_run}")
    shots = parse_shots(args.shots)
    anchor_defect_base_words = (
        parse_word_csv(args.anchor_defect_words)
        if args.anchor_defect_words
        else list(DEFAULT_ANCHOR_DEFECT_BASE_WORDS)
    )
    early_exit_layers = list(range(args.start_layer, args.end_layer))
    if args.use_deco and len(early_exit_layers) == 0:
        raise ValueError("When `--use_deco` is set, `--start_layer` must be smaller than `--end_layer`.")

    print("===== Key Runtime Config =====")
    print(f"dataset_root: {dataset_root}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"visa_source: {args.visa_source}")
    print(f"shots: {shots}")
    print(f"categories_count: {len(categories_for_run)}")
    print(f"do_sample: {args.do_sample}")
    print(f"temperature: {args.temperature if args.do_sample else 'N/A (do_sample=False)'}")
    print(f"num_beams: {args.num_beams}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"metric_threshold_policy: {args.metric_threshold_policy}")
    print(f"metric_threshold: {args.metric_threshold}")
    print(f"score_label_aggregation: {args.score_label_aggregation}")
    print(f"min_pixels: {args.min_pixels}")
    print(f"max_pixels: {args.max_pixels}")
    print(f"seed: {args.seed}")
    print(f"attn_implementation: {args.attn_implementation}")
    print(f"allow_attn_fallback: {args.allow_attn_fallback}")
    print(f"print_image_name: {args.print_image_name}")
    print(f"use_deco: {args.use_deco}")
    print(f"fixed_support_images: {args.fixed_support_images}")
    if args.use_deco:
        print(f"deco_alpha: {args.alpha}")
        print(f"deco_threshold_top_p: {args.threshold_top_p}")
        print(f"deco_threshold_top_k: {args.threshold_top_k}")
        print(f"deco_early_exit_layers: {early_exit_layers}")
        print(f"deco_anchor_strategy: {args.anchor_strategy}")
        if args.anchor_strategy == "defect_aware":
            print(f"deco_lambda_norm: {args.lambda_norm}")
            print(f"deco_eta_hedge: {args.eta_hedge}")
            print(f"deco_enable_hedge: {args.enable_hedge}")
            print(f"deco_use_logsumexp: {args.use_logsumexp}")
            print(f"deco_gamma: {args.gamma}")
            print(f"deco_anchor_defect_words_base: {','.join(anchor_defect_base_words)}")
            print("deco_anchor_defect_words_policy: base + current_category_defect_types")
            print(f"deco_anchor_norm_words: {args.anchor_norm_words}")
            print(f"deco_anchor_hedge_words: {args.anchor_hedge_words}")
            print(f"deco_debug: {args.deco_debug}")
            print(f"anchor_smooth: {args.anchor_smooth}")
            print(f"anchor_remove_outliers: {args.anchor_remove_outliers}")
            print(f"anchor_hampel_window: {args.anchor_hampel_window}")
            print(f"anchor_hampel_n_sigma: {args.anchor_hampel_n_sigma}")
            print(f"anchor_smooth_window: {args.anchor_smooth_window}")
            print(f"anchor_smooth_beta: {args.anchor_smooth_beta}")
    print("==============================")

    samples, category_meta = discover_visa(
        dataset_root=dataset_root,
        allowed_categories=allowed_categories,
        max_samples_per_defect=args.max_samples_per_defect,
        visa_source=args.visa_source,
        visa_split_csv=args.visa_split_csv,
        visa_split_set=args.visa_split_set,
    )
    if len(samples) == 0:
        raise RuntimeError("No samples found. Please check VisA directory structure and image_anno.csv files.")
    print(f"Total images to process: {len(samples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    attn_impl = args.attn_implementation
    if device != "cuda" and attn_impl == "flash_attention_2":
        message = "flash_attention_2 requires CUDA device."
        if args.allow_attn_fallback:
            print(f"[WARN] {message} Fallback to sdpa.")
            attn_impl = "sdpa"
        else:
            raise RuntimeError(f"{message} Re-run with --allow_attn_fallback or --attn_implementation sdpa.")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else None,
        "attn_implementation": attn_impl,
        "trust_remote_code": True
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            **load_kwargs,
        )
    except Exception as e:
        if args.allow_attn_fallback and attn_impl == "flash_attention_2":
            print(f"[WARN] Failed to load with flash_attention_2: {e}")
            print("[WARN] Fallback to sdpa.")
            load_kwargs["attn_implementation"] = "sdpa"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                **load_kwargs,
            )
        else:
            raise
    if device != "cuda":
        model.to(device)
    model.eval()

    if not args.do_sample:
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    fixed_support_map = parse_fixed_support_images(args.fixed_support_images)

    support_cache: Dict[Tuple[str, int], List[str]] = {}
    for category, meta in category_meta.items():
        support_pool = list(meta["support_good_paths"])
        support_pool_str = [str(p) for p in support_pool]
        support_name_to_path = {Path(p).name: str(p) for p in support_pool}

        for shot in shots:
            if shot == 0:
                support_cache[(category, shot)] = []
                continue

            if category in fixed_support_map:
                wanted_refs = fixed_support_map[category]

                if len(wanted_refs) < shot:
                    raise ValueError(
                        f"Category '{category}' requires shot={shot}, but only {len(wanted_refs)} fixed support image(s) were provided."
                    )

                resolved_refs: List[str] = []
                for ref in wanted_refs[:shot]:
                    if ref in support_name_to_path:
                        resolved_refs.append(support_name_to_path[ref])
                        continue

                    candidate = dataset_root / ref
                    if candidate.exists():
                        resolved_refs.append(str(candidate))
                        continue

                    if Path(ref).exists():
                        resolved_refs.append(str(Path(ref)))
                        continue

                    raise FileNotFoundError(
                        f"Fixed support image '{ref}' for category '{category}' not found in support_good_paths or filesystem."
                    )

                support_cache[(category, shot)] = resolved_refs
                continue

            if len(support_pool_str) == 0:
                support_cache[(category, shot)] = []
                continue

            if len(support_pool_str) >= shot:
                support_cache[(category, shot)] = [str(p) for p in random.sample(support_pool_str, shot)]
            else:
                support_cache[(category, shot)] = [str(p) for p in random.choices(support_pool_str, k=shot)]

    results: List[Dict[str, Any]] = []
    eval_jobs = [(shot, sample) for shot in shots for sample in samples]
    total_eval = len(eval_jobs)
    print(f"Total inference calls (images x shots): {total_eval}")
    category_sample_counts: Dict[str, int] = {}
    for sample in samples:
        category_key = str(sample["category"])
        category_sample_counts[category_key] = category_sample_counts.get(category_key, 0) + 1

    progress = tqdm(
        eval_jobs,
        total=total_eval,
        desc="Evaluating VisA",
        ncols=120,
        leave=True,
        mininterval=0.2,
    )
    printed_support_keys = set()
    printed_category_runtime_config = set()

    for idx, (shot, sample) in enumerate(progress, start=1):
        category = sample["category"]
        image_path = sample["image_path"]
        image_name = Path(image_path).name
        gt_label = sample["ground_truth_label"]
        defect_types = category_meta[category]["defect_types"]
        support_paths = support_cache[(category, shot)]

        if category not in printed_category_runtime_config:
            progress.write("===== Category Runtime Config =====")
            progress.write(f"category: {category}")
            progress.write(f"category_samples: {category_sample_counts.get(category, 0)}")
            progress.write(
                f"category_inference_calls(all_shots): {category_sample_counts.get(category, 0) * len(shots)}"
            )
            progress.write(f"shots: {shots}")
            progress.write(f"defect_types: {defect_types}")
            progress.write(f"fixed_support_for_category: {fixed_support_map.get(category, None)}")
            if args.use_deco:
                progress.write("use_deco: True")
                progress.write(f"deco_alpha: {args.alpha}")
                progress.write(f"deco_threshold_top_p: {args.threshold_top_p}")
                progress.write(f"deco_threshold_top_k: {args.threshold_top_k}")
                progress.write(f"deco_early_exit_layers: {early_exit_layers}")
                progress.write(f"deco_anchor_strategy: {args.anchor_strategy}")
                if args.anchor_strategy == "defect_aware":
                    dynamic_anchor_words_preview = merge_unique_words(anchor_defect_base_words, defect_types)
                    progress.write(f"deco_lambda_norm: {args.lambda_norm}")
                    progress.write(f"deco_eta_hedge: {args.eta_hedge}")
                    progress.write(f"deco_enable_hedge: {args.enable_hedge}")
                    progress.write(f"deco_use_logsumexp: {args.use_logsumexp}")
                    progress.write(f"deco_gamma: {args.gamma}")
                    progress.write(f"deco_anchor_defect_words_runtime: {','.join(dynamic_anchor_words_preview)}")
                    progress.write(f"deco_anchor_norm_words: {args.anchor_norm_words}")
                    progress.write(f"deco_anchor_hedge_words: {args.anchor_hedge_words}")
                    progress.write(f"deco_debug: {args.deco_debug}")
                    progress.write(f"anchor_smooth: {args.anchor_smooth}")
                    progress.write(f"anchor_remove_outliers: {args.anchor_remove_outliers}")
                    progress.write(f"anchor_hampel_window: {args.anchor_hampel_window}")
                    progress.write(f"anchor_hampel_n_sigma: {args.anchor_hampel_n_sigma}")
                    progress.write(f"anchor_smooth_window: {args.anchor_smooth_window}")
                    progress.write(f"anchor_smooth_beta: {args.anchor_smooth_beta}")
            progress.write("==================================")
            printed_category_runtime_config.add(category)

        if args.print_image_name:
            progress.write(f"[{idx}/{total_eval}] Processing image: {image_name}")
        support_key = (category, shot)
        if shot > 0 and support_key not in printed_support_keys:
            support_names = [Path(p).name for p in support_paths]
            progress.write(
                f"[{idx}/{total_eval}] Category={category}, {shot}-shot normal reference image(s): {support_names}"
            )
            printed_support_keys.add(support_key)

        support_images = [Image.open(p).convert("RGB") for p in support_paths]
        query_image = Image.open(image_path).convert("RGB")
        image_size = query_image.size

        messages = build_messages(category=category, defect_types=defect_types, shot=shot)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_images = support_images + [query_image]
        image_resolution_kwargs = {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels
        }

        inputs = processor(
            text=[text],
            images=model_images,
            return_tensors="pt",
            **image_resolution_kwargs
        )

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_token_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "num_beams": args.num_beams,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty
        }
        if args.use_deco:
            dynamic_anchor_defect_words = args.anchor_defect_words
            if args.anchor_strategy == "defect_aware":
                expanded_defect_words = expand_defect_words_with_morphology(defect_types)
                dynamic_anchor_defect_words = ",".join(expanded_defect_words) if expanded_defect_words else None
            generation_kwargs.update(
                {
                    "use_deco": True,
                    "alpha": args.alpha,
                    "threshold_top_p": args.threshold_top_p,
                    "threshold_top_k": args.threshold_top_k,
                    "early_exit_layers": early_exit_layers,
                    "anchor_strategy": args.anchor_strategy,
                    "lambda_norm": args.lambda_norm,
                    "eta_hedge": args.eta_hedge,
                    "use_logsumexp": args.use_logsumexp,
                    "gamma": args.gamma,
                    "enable_hedge": args.enable_hedge,
                    "anchor_defect_words": dynamic_anchor_defect_words,
                    "anchor_norm_words": args.anchor_norm_words,
                    "anchor_hedge_words": args.anchor_hedge_words,
                    "deco_debug": args.deco_debug,
                    "tokenizer": getattr(processor, "tokenizer", None),
                    "anchor_smooth": args.anchor_smooth,
                    "anchor_remove_outliers": args.anchor_remove_outliers,
                    "anchor_hampel_window": args.anchor_hampel_window,
                    "anchor_hampel_n_sigma": args.anchor_hampel_n_sigma,
                    "anchor_smooth_window": args.anchor_smooth_window,
                    "anchor_smooth_beta": args.anchor_smooth_beta,
                }
            )
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        gen_ids = output_ids[:, input_token_len:]
        raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        parsed, parse_source = parse_model_json(raw_text)
        prediction = normalize_prediction(
            parsed=parsed,
            parse_source=parse_source,
            raw_text=raw_text,
            defect_types=defect_types,
            image_size=image_size,
        )
        score = compute_label_likelihood_score(
            model=model,
            processor=processor,
            text=text,
            model_images=model_images,
            image_resolution_kwargs=image_resolution_kwargs,
            defect_types=defect_types,
            aggregation=args.score_label_aggregation,
            use_deco=args.use_deco,
            deco_alpha=args.alpha,
            deco_layers=early_exit_layers,
        )

        results.append(
            {
                "shot": shot,
                "category": category,
                "image_path": image_path,
                "ground_truth": {
                    "label": gt_label,
                    "labels": sample.get("ground_truth_labels", [gt_label]),
                    "binary_label": "good" if gt_label == "good" else "anomaly",
                    "is_anomaly": gt_label != "good",
                },
                "prediction": prediction,
                "score": score,
                "parsed_json": parsed,
                "support_image_paths": support_paths,
            }
        )
        short_image_name = image_name if len(image_name) <= 32 else f"...{image_name[-29:]}"
        progress.set_postfix({"shot": shot, "category": category, "image": short_image_name}, refresh=False)

    output_data = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": str(dataset_root),
            "model_name_or_path": args.model_name_or_path,
            "visa_source": args.visa_source,
            "visa_split_csv": args.visa_split_csv,
            "visa_split_set": args.visa_split_set if args.visa_source == "split_csv" else None,
            "shots": shots,
            "selected_categories": categories_for_run,
            "shot_definition": "k-shot means k normal reference images from the same VisA category.",
            "total_samples": len(samples),
            "total_inference_calls": total_eval,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "attn_implementation": load_kwargs["attn_implementation"],
            "allow_attn_fallback": args.allow_attn_fallback,
            "print_image_name": args.print_image_name,
            "use_deco": args.use_deco,
            "fixed_support_images": args.fixed_support_images,
            "deco_alpha": args.alpha if args.use_deco else None,
            "deco_threshold_top_p": args.threshold_top_p if args.use_deco else None,
            "deco_threshold_top_k": args.threshold_top_k if args.use_deco else None,
            "deco_early_exit_layers": early_exit_layers if args.use_deco else None,
            "deco_anchor_strategy": args.anchor_strategy if args.use_deco else None,
            "deco_lambda_norm": args.lambda_norm if args.use_deco else None,
            "deco_eta_hedge": args.eta_hedge if args.use_deco else None,
            "deco_use_logsumexp": args.use_logsumexp if args.use_deco else None,
            "deco_gamma": args.gamma if args.use_deco else None,
            "deco_enable_hedge": args.enable_hedge if args.use_deco else None,
            "deco_anchor_defect_words": args.anchor_defect_words if args.use_deco else None,
            "deco_anchor_defect_words_base": ",".join(anchor_defect_base_words) if args.use_deco else None,
            "deco_anchor_defect_words_policy": (
                "base_plus_category_defect_types" if args.use_deco and args.anchor_strategy == "defect_aware" else None
            ),
            "deco_anchor_norm_words": args.anchor_norm_words if args.use_deco else None,
            "deco_anchor_hedge_words": args.anchor_hedge_words if args.use_deco else None,
            "deco_debug": args.deco_debug if args.use_deco else None,
        },
        "summary": compute_summary(
            results,
            threshold_policy=args.metric_threshold_policy,
            fixed_threshold=args.metric_threshold,
        ),
        "results": results,
    }
    output_path = Path(args.output_json)
    save_json(output_path, output_data)
    print(f"Saved JSON result to: {output_path}")
    save_metrics_csv(output_path, output_data)


if __name__ == "__main__":
    main()
