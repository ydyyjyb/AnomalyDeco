import argparse
import ast
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
        choices=["per_category_csv"],
        help="VisA label source mode.",
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


def discover_visa(dataset_root, allowed_categories=None, max_samples_per_defect=None):
    samples = []
    category_meta = {}

    dataset_root = Path(dataset_root)
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

    if label == "good":
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
        "raw_text": raw_text,
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_shot: Dict[int, Dict[str, float]] = {}
    for item in results:
        shot = int(item["shot"])
        gt = item["ground_truth"]["label"]
        gt_labels = item["ground_truth"].get("labels", [gt])
        pred = item["prediction"]["label"]

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
            "subtype_acc_on_anomaly": (
                stats["subtype_correct_on_anomaly"] / anomaly_count if anomaly_count > 0 else None
            ),
            "json_parse_success_rate": stats["strict_parse_ok_count"] / total,
            "extraction_success_rate": stats["extraction_ok_count"] / total,
            "normalized_output_rate": stats["parse_ok_count"] / total,
        }
    return summary


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
        "summary": compute_summary(results),
        "results": results,
    }
    output_path = Path(args.output_json)
    save_json(output_path, output_data)
    print(f"Saved JSON result to: {output_path}")


if __name__ == "__main__":
    main()