import argparse
import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_DEFECT_BANK = [
    "anomaly",
    "defect",
    "defective",
    "abnormal",
    "damage",
    "damaged",
    "crack",
    "scratch",
    "stain",
    "hole",
    "missing part",
    "broken part",
    "bent component",
    "contamination",
    "deformation",
]
DEFAULT_NORMAL_BANK = ["good", "normal", "intact"]
DEFAULT_HEDGE_BANK = [
    "maybe",
    "possibly",
    "likely",
    "unclear",
    "uncertain",
    "appears",
    "seems",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Mechanism validation for Qwen3-VL industrial anomaly detection layer conflict."
    )
    parser.add_argument("--visa-root", type=str, required=True)
    parser.add_argument("--visa-split-csv", type=str, default=None)
    parser.add_argument("--visa-split-set", type=str, default="test")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category subset.")
    parser.add_argument("--run-all", action="store_true", help="Evaluate all matching split samples.")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--trend-layers", type=str, default="0:34")
    parser.add_argument("--deco-layers", type=str, default="20:29")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument(
        "--generation-deco",
        action="store_true",
        help="Also run official generative DeCo through model.generate(..., use_deco=True).",
    )
    parser.add_argument(
        "--score-deco-mode",
        type=str,
        default="single",
        choices=["single", "multi_top3"],
        help="Use one selected anchor layer or weighted top-3 selected layers for score-side DeCo correction.",
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
        "--metric-threshold-policy",
        type=str,
        default="youden",
        choices=["youden", "best_f1", "best_acc", "zero", "fixed"],
        help=(
            "Threshold policy for summary ACC/precision/recall/F1 from continuous scores. "
            "--decision-threshold is still used for per-case mechanism labels and plots."
        ),
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.0,
        help="Fixed threshold used when --metric-threshold-policy fixed.",
    )
    parser.add_argument(
        "--no-text-prior-baseline",
        action="store_true",
        help="Disable text-only prior baseline and visual-minus-text-prior debiased metrics.",
    )
    parser.add_argument("--normal-labels", type=str, default=",".join(DEFAULT_NORMAL_BANK))
    parser.add_argument("--generic-defect-labels", type=str, default=",".join(DEFAULT_DEFECT_BANK))
    parser.add_argument("--hedge-labels", type=str, default=",".join(DEFAULT_HEDGE_BANK))
    parser.add_argument(
        "--defect-aware-pool-k",
        type=int,
        default=3,
        help="Top-k DeCo-range layers used by the defect-aware evidence pool.",
    )
    parser.add_argument(
        "--defect-aware-lambda-norm",
        type=float,
        default=1.0,
        help="Penalty weight for normal evidence in defect-aware evidence pool.",
    )
    parser.add_argument(
        "--defect-aware-eta-hedge",
        type=float,
        default=0.0,
        help="Penalty weight for hedge/uncertainty evidence in defect-aware evidence pool.",
    )
    parser.add_argument(
        "--defect-aware-enable-hedge",
        action="store_true",
        help="Enable hedge-token penalty in defect-aware evidence pool.",
    )
    parser.add_argument(
        "--defect-aware-use-logsumexp",
        action="store_true",
        help="Use gamma-scaled logsumexp aggregation for evidence-token groups instead of max.",
    )
    parser.add_argument(
        "--defect-aware-gamma",
        type=float,
        default=8.0,
        help="Gamma for defect-aware logsumexp evidence aggregation.",
    )
    parser.add_argument(
        "--hdec-gate-gamma",
        type=float,
        default=8.0,
        help="Sigmoid sharpness for HDEC dynamic gate from defect-aware evidence pool.",
    )
    parser.add_argument(
        "--hdec-gate-tau",
        type=float,
        default=0.0,
        help="Evidence threshold tau for HDEC gate: sigmoid(gamma * (pool_score - tau)).",
    )
    parser.add_argument(
        "--hdec-beta-prior",
        type=float,
        default=1.0,
        help="Text-only prior subtraction weight beta for HDEC debiased score.",
    )
    parser.add_argument(
        "--suppression-gate-gamma",
        type=float,
        default=1.0,
        help="Sigmoid sharpness for suppression gate from anchor_score - final_score.",
    )
    parser.add_argument(
        "--suppression-gate-tau",
        type=float,
        default=0.0,
        help="Threshold tau for suppression gate: sigmoid(gamma * ((anchor_score - final_score) - tau)).",
    )
    parser.add_argument(
        "--suppression-pool-gate-gamma",
        type=float,
        default=8.0,
        help="Sigmoid sharpness for the pool confirmation gate in suppression_pool_hdec.",
    )
    parser.add_argument(
        "--suppression-pool-gate-tau",
        type=float,
        default=0.0,
        help="Pool evidence threshold tau for suppression_pool_hdec.",
    )
    parser.add_argument(
        "--max-defect-labels",
        type=int,
        default=16,
        help="Maximum category-level anomaly labels used for scoring and prompt.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="generic",
        choices=["generic", "category", "category_defects", "category_defects_guarded"],
        help=(
            "Prompt template. generic keeps the old category-free prompt; category adds the object category; "
            "category_defects also lists possible category-level defects; category_defects_guarded adds a guard "
            "that the listed defects are only possibilities."
        ),
    )
    parser.add_argument(
        "--prompt-defect-limit",
        type=int,
        default=8,
        help="Maximum number of category-level defect descriptions inserted into the prompt.",
    )
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-limit", type=int, default=80)
    parser.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="Write compact per-case JSONL while running. Useful for long jobs and crash recovery.",
    )
    parser.add_argument(
        "--save-full-cases",
        action="store_true",
        help="Also save verbose full cases with prompts, tokenization maps, and raw logits.",
    )
    parser.add_argument(
        "--include-generated-text",
        action="store_true",
        help="Keep raw generated text in compact case JSON. Disabled by default to keep files small.",
    )
    parser.add_argument("--target-acc", type=float, default=0.79)
    parser.add_argument("--target-auroc", type=float, default=0.911)
    parser.add_argument("--target-recall", type=float, default=0.86)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--target-f1", type=float, default=0.83)
    return parser.parse_args()


def parse_range(text: str) -> List[int]:
    text = str(text).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_words(text: str) -> List[str]:
    seen = set()
    words = []
    for chunk in str(text or "").split(","):
        word = chunk.strip()
        key = word.lower()
        if word and key not in seen:
            seen.add(key)
            words.append(word)
    return words


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_anomaly_prompt_map(category_dir: Path) -> Dict[str, List[str]]:
    path = category_dir / "anomaly_prompt.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        image = str(item.get("image", "")).replace("\\", "/").strip()
        prompt = str(item.get("prompt", "")).strip()
        labels = [x.strip() for x in prompt.split(",") if x.strip()]
        if image and labels:
            out[image] = labels
            out[Path(image).name] = labels
    return out


def build_category_defect_bank(visa_root: Path, category: str, generic_labels: List[str], max_labels: int) -> List[str]:
    category_dir = visa_root / category
    prompt_map = load_anomaly_prompt_map(category_dir)
    labels: List[str] = []
    seen = set()
    for label_list in prompt_map.values():
        for label in label_list:
            clean = str(label).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                labels.append(clean)
    for label in generic_labels:
        clean = str(label).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            labels.append(clean)
    return labels[:max_labels]


def load_split_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    visa_root = Path(args.visa_root)
    split_csv = Path(args.visa_split_csv) if args.visa_split_csv else visa_root / "split_csv" / "1cls.csv"
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    rows = pd.read_csv(
        split_csv,
        header=None,
        names=["category", "split", "label", "image", "mask"],
        keep_default_na=False,
    )
    split_name = str(args.visa_split_set).strip().lower()
    samples = []
    for _, row in rows.iterrows():
        category = str(row["category"]).strip()
        split = str(row["split"]).strip().lower()
        label_text = str(row["label"]).strip().lower()
        rel_image = str(row["image"]).replace("\\", "/").strip()
        rel_mask = str(row["mask"]).replace("\\", "/").strip()
        if split != split_name:
            continue
        if selected and category not in selected:
            continue
        image_path = visa_root / rel_image
        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue
        label = 0 if label_text == "normal" else 1
        samples.append(
            {
                "category": category,
                "image": str(image_path),
                "image_id": Path(rel_image).stem,
                "label": label,
                "label_name": "anomaly" if label == 1 else "normal",
                "mask": str(visa_root / rel_mask) if rel_mask and (visa_root / rel_mask).exists() else "",
                "split_image": rel_image,
                "split_mask": rel_mask,
            }
        )
    if args.run_all:
        return samples

    rng = random.Random(args.seed)
    selected_samples = []
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for sample in samples:
        buckets.setdefault((sample["category"], sample["label"]), []).append(sample)
    for key in sorted(buckets):
        bucket = buckets[key]
        k = args.sample_anomaly if key[1] == 1 else args.sample_normal
        if k is None or k < 0 or len(bucket) <= k:
            selected_samples.extend(bucket)
        else:
            selected_samples.extend(rng.sample(bucket, k))
    return selected_samples


def display_category(category: str) -> str:
    return str(category or "object").replace("_", " ").strip() or "object"


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


def build_instruction(category: str = "", defect_bank: Optional[List[str]] = None, args: Optional[argparse.Namespace] = None) -> str:
    mode = getattr(args, "prompt_mode", "generic")
    defect_limit = int(getattr(args, "prompt_defect_limit", 8))
    category_text = display_category(category)
    defect_items = prompt_defect_list(defect_bank or [], defect_limit)

    lines = []
    if mode in {"category", "category_defects", "category_defects_guarded"}:
        lines.append(f"This is an image of {category_text}.")
    else:
        lines.append("Examine the image and describe the visible condition of the main subject.")

    if mode in {"category_defects", "category_defects_guarded"} and defect_items:
        lines.append(f"This category may have defects such as: {', '.join(defect_items)}.")
        if mode == "category_defects_guarded":
            lines.append("These defects are only possible examples, not assumptions about this image.")

    lines.extend(
        [
            "Inspect the main subject for visual abnormalities.",
            "Use only visual evidence, and ignore viewpoint, lighting, background, and image noise.",
            "Ignore pose, position, and orientation unless they create a clear physical defect.",
            "Describe any clear localized abnormality if present; otherwise state that it appears normal.",
            "",
            "End with exactly one line:",
            "Answer: Normal",
            "or",
            "Answer: Anomaly",
        ]
    )
    return "\n".join(lines)


def build_messages(
    category: str,
    defect_bank: List[str],
    args: Optional[argparse.Namespace] = None,
    reference_count: int = 0,
) -> List[Dict[str, Any]]:
    instruction = build_instruction(category, defect_bank, args)
    content = [{"type": "text", "text": instruction}]
    if reference_count > 0:
        content.append(
            {
                "type": "text",
                "text": (
                    f"You are given {reference_count} normal reference image(s) from the same category before the query image. "
                    "Each reference image is defect-free and shows acceptable normal appearance. "
                    "Use them as visual references, then judge only whether the final query image contains a defect "
                    "or abnormality relative to those normal references."
                ),
            }
        )
        for index in range(reference_count):
            content.extend(
                [
                    {"type": "text", "text": f"Normal reference image {index + 1}:"},
                    {"type": "image"},
                ]
            )
    content.extend(
        [
            {"type": "text", "text": "Query image:"},
            {"type": "image"},
        ]
    )
    return [{"role": "user", "content": content}]


def build_text_only_messages(category: str, defect_bank: List[str], args: Optional[argparse.Namespace] = None) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": build_instruction(category, defect_bank, args)}]}]


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


def unique_token_ids_from_label_map(label_map: Dict[str, int]) -> List[int]:
    return sorted({int(token_id) for token_id in label_map.values()})


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


def aggregate_log_mass(values: torch.Tensor, use_logsumexp: bool, gamma: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(float("-inf"), device=values.device, dtype=torch.float32)
    values = values.float()
    if use_logsumexp:
        g = max(float(gamma), 1e-6)
        return torch.logsumexp(g * values, dim=0) / g
    return torch.max(values, dim=0).values


def evidence_log_mass(
    token_logits: torch.Tensor,
    token_ids: List[int],
    group_ids: List[int],
    use_logsumexp: bool,
    gamma: float,
) -> torch.Tensor:
    positions = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
    values = token_logits[[positions[token_id] for token_id in group_ids if token_id in positions]]
    return aggregate_log_mass(values, use_logsumexp, gamma)


def binary_logits_from_token_logits(
    token_logits: torch.Tensor,
    token_ids: List[int],
    normal_ids: List[int],
    anomaly_ids: List[int],
    aggregation: str,
) -> torch.Tensor:
    positions = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
    normal_values = token_logits[[positions[token_id] for token_id in normal_ids if token_id in positions]]
    anomaly_values = token_logits[[positions[token_id] for token_id in anomaly_ids if token_id in positions]]
    return torch.stack(
        [
            aggregate_values(normal_values, aggregation),
            aggregate_values(anomaly_values, aggregation),
        ]
    )


def parse_generated_label(raw_text: str, defect_bank: List[str]) -> str:
    text = str(raw_text or "")
    answer_matches = re.findall(
        r"answer\s*:\s*(normal|anomaly|yes|no|good|abnormal|defective|defect)",
        text,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        answer = answer_matches[-1].strip().lower()
        if answer in {"normal", "no", "good"}:
            return "good"
        return "anomaly"

    match = re.search(r'"label"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if not match:
        return "unknown"

    value = match.group(1).strip().lower()
    if value in {"good", "normal", "ok", "no defect", "non-defective", "intact"}:
        return "good"
    generic_anomaly = {"anomaly", "abnormal", "defect", "defective", "damaged", "damage", "bad", "flawed", "ng", "reject"}
    if value in generic_anomaly:
        return "anomaly"
    for defect in defect_bank:
        defect_norm = defect.lower()
        if defect_norm and (defect_norm in value or value in defect_norm):
            return defect
    return "unknown"


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    except Exception as exc:
        if not args.allow_attn_fallback or args.attn_implementation == "sdpa":
            raise
        print(f"[WARN] Failed to load with {args.attn_implementation}: {exc}")
        print("[WARN] Fallback to sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model


def anchor_selection_score(row: Dict[str, Any], final_score: float, mode: str) -> float:
    score = float(row["score"])
    confidence = float(row["confidence"])
    if mode == "confidence":
        return confidence
    if mode == "evidence":
        return abs(score) * confidence
    return abs(score - final_score) * abs(score) * confidence


def select_anchors(layer_rows: List[Dict[str, Any]], final_score: float, mode: str, k: int = 1) -> List[Dict[str, Any]]:
    if not layer_rows:
        raise RuntimeError("No layer scores available.")
    scored_rows = []
    for row in layer_rows:
        copied = dict(row)
        copied["selection_score"] = anchor_selection_score(row, final_score, mode)
        scored_rows.append(copied)
    scored_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    return scored_rows[: max(1, int(k))]


def combine_anchor_logits(
    selected_anchors: List[Dict[str, Any]],
    final_binary_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    anchor_logits = torch.tensor(
        [[row["normal_logit"], row["anomaly_logit"]] for row in selected_anchors],
        device=final_binary_logits.device,
        dtype=final_binary_logits.dtype,
    )
    confidences = torch.tensor(
        [max(float(row["confidence"]), 1e-6) for row in selected_anchors],
        device=final_binary_logits.device,
        dtype=final_binary_logits.dtype,
    )
    weights = confidences / torch.clamp(confidences.sum(), min=1e-6)
    combined_logits = torch.sum(anchor_logits * weights[:, None], dim=0)
    combined_confidence = torch.sum(confidences * weights)
    selected_info = []
    for row, weight in zip(selected_anchors, weights.detach().float().cpu().tolist()):
        selected_info.append(
            {
                "layer": int(row["layer"]),
                "score": float(row["score"]),
                "normal_logit": float(row["normal_logit"]),
                "anomaly_logit": float(row["anomaly_logit"]),
                "normal_prob": float(row["normal_prob"]),
                "anomaly_prob": float(row["anomaly_prob"]),
                "confidence": float(row["confidence"]),
                "selection_score": float(row.get("selection_score", 0.0)),
                "fusion_weight": float(weight),
            }
        )
    return combined_logits, combined_confidence, selected_info


def classify(score: float, threshold: float) -> int:
    return 1 if float(score) >= float(threshold) else 0


def conflict_type(label: int, final_pred: int, anchor_pred: int) -> str:
    if final_pred == anchor_pred:
        return "consistent_anomaly" if final_pred == 1 else "consistent_normal"
    if label == 1 and anchor_pred == 1 and final_pred == 0:
        return "defect_omission"
    if label == 0 and anchor_pred == 0 and final_pred == 1:
        return "defect_hallucination"
    return "other_conflict"


def compute_binary_layer_probe(
    args: argparse.Namespace,
    model: Any,
    outputs: Any,
    candidate_ids: List[int],
    normal_ids: List[int],
    anomaly_ids: List[int],
    trend_layers: List[int],
    deco_layers: set,
) -> Dict[str, Any]:
    final_logits = outputs.logits[0, -1].float()
    final_candidate_logits = final_logits[candidate_ids]
    final_binary_logits = binary_logits_from_token_logits(
        final_candidate_logits,
        candidate_ids,
        normal_ids,
        anomaly_ids,
        args.score_label_aggregation,
    )
    final_probs = torch.softmax(final_binary_logits.float(), dim=-1)
    final_score = float((final_binary_logits[1] - final_binary_logits[0]).detach().cpu())

    hidden_states = getattr(outputs, "hidden_states", None)
    final_norm = get_final_norm_module(model)
    if hidden_states is None:
        raise RuntimeError("Model output has no hidden_states.")
    norm_module_found = final_norm is not None
    if final_norm is None:
        final_norm = torch.nn.Identity().to(model.device)

    layer_rows = []
    for layer_idx in trend_layers:
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            continue
        hidden = final_norm(hidden_states[layer_idx][0, -1])
        candidate_logits = candidate_logits_from_hidden(model, hidden, candidate_ids)
        binary_logits = binary_logits_from_token_logits(
            candidate_logits,
            candidate_ids,
            normal_ids,
            anomaly_ids,
            args.score_label_aggregation,
        )
        probs = torch.softmax(binary_logits.float(), dim=-1)
        score = float((binary_logits[1] - binary_logits[0]).detach().cpu())
        row = {
            "layer": int(layer_idx),
            "normal_logit": float(binary_logits[0].detach().cpu()),
            "anomaly_logit": float(binary_logits[1].detach().cpu()),
            "score": score,
            "normal_prob": float(probs[0].detach().cpu()),
            "anomaly_prob": float(probs[1].detach().cpu()),
            "confidence": float(torch.max(probs).detach().cpu()),
            "in_deco_range": layer_idx in deco_layers,
        }
        layer_rows.append(row)

    return {
        "final_binary_logits": final_binary_logits,
        "final_probs": final_probs,
        "final_score": final_score,
        "layer_rows": layer_rows,
        "norm_module_found": bool(norm_module_found),
    }


def compute_defect_aware_pool(
    args: argparse.Namespace,
    model: Any,
    outputs: Any,
    defect_ids: List[int],
    normal_ids: List[int],
    hedge_ids: List[int],
    deco_layers: set,
) -> Dict[str, Any]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output has no hidden_states.")

    evidence_ids = sorted(set(defect_ids + normal_ids + (hedge_ids if args.defect_aware_enable_hedge else [])))
    if not defect_ids or not normal_ids or not evidence_ids:
        return {
            "score": 0.0,
            "pred": 0,
            "selected_layers": [],
            "selected_details": [],
            "layer_scores": [],
            "error": "empty defect/normal evidence token ids",
        }

    final_norm = get_final_norm_module(model)
    if final_norm is None:
        final_norm = torch.nn.Identity().to(model.device)

    layer_rows = []
    for layer_idx in sorted(deco_layers):
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            continue
        hidden = final_norm(hidden_states[layer_idx][0, -1])
        token_logits = candidate_logits_from_hidden(model, hidden, evidence_ids)
        defect_log_mass = evidence_log_mass(
            token_logits,
            evidence_ids,
            defect_ids,
            args.defect_aware_use_logsumexp,
            args.defect_aware_gamma,
        )
        normal_log_mass = evidence_log_mass(
            token_logits,
            evidence_ids,
            normal_ids,
            args.defect_aware_use_logsumexp,
            args.defect_aware_gamma,
        )
        if args.defect_aware_enable_hedge:
            hedge_log_mass = evidence_log_mass(
                token_logits,
                evidence_ids,
                hedge_ids,
                args.defect_aware_use_logsumexp,
                args.defect_aware_gamma,
            )
        else:
            hedge_log_mass = torch.tensor(float("-inf"), device=token_logits.device, dtype=torch.float32)

        norm3 = torch.softmax(torch.stack([defect_log_mass, normal_log_mass, hedge_log_mass]), dim=0)
        defect_norm3 = norm3[0]
        normal_norm3 = norm3[1]
        hedge_norm3 = norm3[2]
        score = (
            defect_norm3
            - float(args.defect_aware_lambda_norm) * normal_norm3
            - float(args.defect_aware_eta_hedge) * hedge_norm3
        )
        confidence = torch.max(norm3)
        selection_score = score * confidence
        layer_rows.append(
            {
                "layer": int(layer_idx),
                "score": float(score.detach().cpu()),
                "selection_score": float(selection_score.detach().cpu()),
                "confidence": float(confidence.detach().cpu()),
                "defect_norm3": float(defect_norm3.detach().cpu()),
                "normal_norm3": float(normal_norm3.detach().cpu()),
                "hedge_norm3": float(hedge_norm3.detach().cpu()),
                "defect_log_mass": float(defect_log_mass.detach().cpu()),
                "normal_log_mass": float(normal_log_mass.detach().cpu()),
                "hedge_log_mass": None
                if torch.isneginf(hedge_log_mass).item()
                else float(hedge_log_mass.detach().cpu()),
            }
        )

    if not layer_rows:
        return {
            "score": 0.0,
            "pred": 0,
            "selected_layers": [],
            "selected_details": [],
            "layer_scores": [],
            "error": "no valid DeCo-range layers",
        }

    top_k = max(1, int(args.defect_aware_pool_k))
    selected = sorted(layer_rows, key=lambda row: float(row["selection_score"]), reverse=True)[:top_k]
    weights = [max(float(row["confidence"]), 1e-6) for row in selected]
    weight_sum = sum(weights)
    pooled_score = sum(float(row["score"]) * weight for row, weight in zip(selected, weights)) / max(weight_sum, 1e-6)
    selected_details = []
    for row, weight in zip(selected, weights):
        copied = dict(row)
        copied["pool_weight"] = weight / max(weight_sum, 1e-6)
        selected_details.append(copied)
    return {
        "score": float(pooled_score),
        "pred": classify(pooled_score, args.decision_threshold),
        "selected_layers": [int(row["layer"]) for row in selected_details],
        "selected_details": selected_details,
        "layer_scores": layer_rows,
        "error": "",
    }


def compute_probe(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    sample: Dict[str, Any],
    defect_bank: List[str],
    normal_bank: List[str],
    hedge_bank: List[str],
) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor has no tokenizer")

    reference_paths = [
        str(path)
        for path in sample.get("reference_images", [])
        if str(path).strip()
    ]
    reference_images = [Image.open(path).convert("RGB") for path in reference_paths]
    image = Image.open(sample["image"]).convert("RGB")
    images = reference_images + [image]
    messages = build_messages(sample["category"], defect_bank, args, reference_count=len(reference_images))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    inputs = processor(text=[text], images=images, return_tensors="pt", **image_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    deco_layer_list = parse_range(args.deco_layers)

    generation_text = ""
    generated_label = "unknown"
    generation_deco_text = ""
    generation_deco_label = "unknown"
    if not args.skip_generation:
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
        }
        if args.do_sample:
            gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[:, input_len:]
        generation_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generated_label = parse_generated_label(generation_text, defect_bank)
        if args.generation_deco:
            deco_gen_kwargs = {
                **gen_kwargs,
                "use_deco": True,
                "alpha": args.deco_alpha,
                "threshold_top_p": args.deco_threshold_top_p,
                "threshold_top_k": args.deco_threshold_top_k,
                "early_exit_layers": deco_layer_list,
            }
            with torch.no_grad():
                deco_output_ids = model.generate(**inputs, **deco_gen_kwargs)
            deco_gen_ids = deco_output_ids[:, input_len:]
            generation_deco_text = processor.batch_decode(
                deco_gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            generation_deco_label = parse_generated_label(generation_deco_text, defect_bank)

    score_prefix = text + "Answer: "
    score_inputs = processor(text=[score_prefix], images=images, return_tensors="pt", **image_kwargs)
    score_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in score_inputs.items()}

    score_normal_bank = ["Normal", "normal"]
    score_anomaly_bank = ["Anomaly", "anomaly"]
    normal_ids_map = first_token_ids_for_labels(tokenizer, score_normal_bank)
    anomaly_ids_map = first_token_ids_for_labels(tokenizer, score_anomaly_bank)
    normal_ids = sorted(set(normal_ids_map.values()))
    anomaly_ids = sorted(set(anomaly_ids_map.values()))
    candidate_ids = sorted(set(normal_ids + anomaly_ids))
    if not normal_ids or not anomaly_ids:
        raise RuntimeError("Failed to tokenize normal/anomaly candidate labels.")

    trend_layers = parse_range(args.trend_layers)
    deco_layers = set(deco_layer_list)

    with torch.no_grad():
        outputs = model(**score_inputs, output_hidden_states=True, return_dict=True)
    image_probe = compute_binary_layer_probe(
        args=args,
        model=model,
        outputs=outputs,
        candidate_ids=candidate_ids,
        normal_ids=normal_ids,
        anomaly_ids=anomaly_ids,
        trend_layers=trend_layers,
        deco_layers=deco_layers,
    )

    final_binary_logits = image_probe["final_binary_logits"]
    final_probs = image_probe["final_probs"]
    final_score = float(image_probe["final_score"])
    final_pred = classify(final_score, args.decision_threshold)
    norm_module_found = bool(image_probe["norm_module_found"])
    layer_rows = image_probe["layer_rows"]

    evidence_defect_ids_map = first_token_ids_for_labels(tokenizer, defect_bank)
    evidence_normal_ids_map = first_token_ids_for_labels(tokenizer, normal_bank)
    evidence_hedge_ids_map = first_token_ids_for_labels(tokenizer, hedge_bank)

    generic_defect_bank = parse_csv_words(getattr(args, "generic_defect_labels", ""))
    generic_defect_keys = {label.lower() for label in generic_defect_bank}
    specific_defect_bank = [
        label for label in defect_bank if str(label).strip().lower() not in generic_defect_keys
    ]
    if not specific_defect_bank:
        specific_defect_bank = list(defect_bank)
    generic_defect_ids_map = first_token_ids_for_labels(tokenizer, generic_defect_bank)
    specific_defect_ids_map = first_token_ids_for_labels(tokenizer, specific_defect_bank)

    defect_aware_pool = compute_defect_aware_pool(
        args=args,
        model=model,
        outputs=outputs,
        defect_ids=unique_token_ids_from_label_map(evidence_defect_ids_map),
        normal_ids=unique_token_ids_from_label_map(evidence_normal_ids_map),
        hedge_ids=unique_token_ids_from_label_map(evidence_hedge_ids_map),
        deco_layers=deco_layers,
    )
    defect_aware_pool_score = float(defect_aware_pool["score"])
    defect_aware_pool_pred = int(defect_aware_pool["pred"])
    generic_defect_pool = compute_defect_aware_pool(
        args=args,
        model=model,
        outputs=outputs,
        defect_ids=unique_token_ids_from_label_map(generic_defect_ids_map),
        normal_ids=unique_token_ids_from_label_map(evidence_normal_ids_map),
        hedge_ids=unique_token_ids_from_label_map(evidence_hedge_ids_map),
        deco_layers=deco_layers,
    )
    generic_defect_pool_score = float(generic_defect_pool["score"])
    generic_defect_pool_pred = int(generic_defect_pool["pred"])
    specific_defect_pool = compute_defect_aware_pool(
        args=args,
        model=model,
        outputs=outputs,
        defect_ids=unique_token_ids_from_label_map(specific_defect_ids_map),
        normal_ids=unique_token_ids_from_label_map(evidence_normal_ids_map),
        hedge_ids=unique_token_ids_from_label_map(evidence_hedge_ids_map),
        deco_layers=deco_layers,
    )
    specific_defect_pool_score = float(specific_defect_pool["score"])
    specific_defect_pool_pred = int(specific_defect_pool["pred"])

    text_prior_enabled = not args.no_text_prior_baseline
    text_prior_error = ""
    text_prior_prefix = ""
    text_prior_score = 0.0
    text_prior_probs = torch.tensor([0.5, 0.5], device=final_binary_logits.device, dtype=torch.float32)
    text_prior_layer_rows: List[Dict[str, Any]] = []
    if text_prior_enabled:
        try:
            text_only_messages = build_text_only_messages(sample["category"], defect_bank, args)
            text_only_text = processor.apply_chat_template(
                text_only_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_prior_prefix = text_only_text + "Answer: "
            text_prior_inputs = processor(text=[text_prior_prefix], return_tensors="pt")
            text_prior_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in text_prior_inputs.items()}
            with torch.no_grad():
                text_prior_outputs = model(**text_prior_inputs, output_hidden_states=True, return_dict=True)
            text_prior_probe = compute_binary_layer_probe(
                args=args,
                model=model,
                outputs=text_prior_outputs,
                candidate_ids=candidate_ids,
                normal_ids=normal_ids,
                anomaly_ids=anomaly_ids,
                trend_layers=trend_layers,
                deco_layers=deco_layers,
            )
            text_prior_score = float(text_prior_probe["final_score"])
            text_prior_probs = text_prior_probe["final_probs"]
            text_prior_layer_rows = text_prior_probe["layer_rows"]
        except Exception as exc:
            text_prior_error = str(exc)
            text_prior_enabled = False

    text_prior_layer_by_id = {int(row["layer"]): row for row in text_prior_layer_rows}
    final_debiased_score = final_score - text_prior_score
    final_debiased_pred = classify(final_debiased_score, args.decision_threshold)
    debiased_layer_rows = []
    for row in layer_rows:
        prior_row = text_prior_layer_by_id.get(int(row["layer"]))
        prior_score = float(prior_row["score"]) if prior_row is not None else text_prior_score
        copied = dict(row)
        copied["raw_image_score"] = float(row["score"])
        copied["text_prior_score"] = prior_score
        copied["debiased_score"] = float(row["score"]) - prior_score
        if text_prior_enabled:
            copied["score"] = copied["debiased_score"]
        debiased_layer_rows.append(copied)

    anchor_candidates = [row for row in layer_rows if row["in_deco_range"]]
    anchor_k = args.multi_deco_k if args.score_deco_mode == "multi_top3" else 1
    selected_anchors = select_anchors(anchor_candidates, final_score, args.anchor_selection, anchor_k)
    primary_anchor = selected_anchors[0]
    selected_binary_logits, selected_confidence, selected_anchor_info = combine_anchor_logits(
        selected_anchors,
        final_binary_logits,
    )
    selected_probs = torch.softmax(selected_binary_logits.float(), dim=-1)
    anchor_score = float((selected_binary_logits[1] - selected_binary_logits[0]).detach().cpu())
    anchor_pred = classify(anchor_score, args.decision_threshold)
    corrected_logits = final_binary_logits + args.deco_alpha * selected_confidence * selected_binary_logits
    corrected_probs = torch.softmax(corrected_logits.float(), dim=-1)
    corrected_score = float((corrected_logits[1] - corrected_logits[0]).detach().cpu())
    corrected_pred = classify(corrected_score, args.decision_threshold)

    anchor_prior_score = 0.0
    for item in selected_anchor_info:
        prior_row = text_prior_layer_by_id.get(int(item["layer"]))
        prior_score = float(prior_row["score"]) if prior_row is not None else text_prior_score
        item["text_prior_score"] = prior_score
        item["debiased_score"] = float(item["score"]) - prior_score
        anchor_prior_score += float(item["fusion_weight"]) * prior_score
    anchor_debiased_score = anchor_score - anchor_prior_score
    anchor_debiased_pred = classify(anchor_debiased_score, args.decision_threshold)
    selected_confidence_float = float(selected_confidence.detach().cpu())
    corrected_prior_score = text_prior_score + args.deco_alpha * selected_confidence_float * anchor_prior_score
    corrected_debiased_score = corrected_score - corrected_prior_score
    corrected_debiased_pred = classify(corrected_debiased_score, args.decision_threshold)
    text_prior_pred = classify(text_prior_score, args.decision_threshold)

    hdec_gate_arg = float(args.hdec_gate_gamma) * (defect_aware_pool_score - float(args.hdec_gate_tau))
    hdec_gate = 1.0 / (1.0 + math.exp(-max(min(hdec_gate_arg, 60.0), -60.0)))
    hdec_score = final_score + hdec_gate * (corrected_score - final_score)
    hdec_prior_score = text_prior_score + hdec_gate * (corrected_prior_score - text_prior_score)
    hdec_debiased_score = hdec_score - float(args.hdec_beta_prior) * hdec_prior_score
    hdec_pred = classify(hdec_score, args.decision_threshold)
    hdec_debiased_pred = classify(hdec_debiased_score, args.decision_threshold)

    generic_hdec_gate_arg = float(args.hdec_gate_gamma) * (
        generic_defect_pool_score - float(args.hdec_gate_tau)
    )
    generic_hdec_gate = 1.0 / (1.0 + math.exp(-max(min(generic_hdec_gate_arg, 60.0), -60.0)))
    generic_hdec_score = final_score + generic_hdec_gate * (corrected_score - final_score)
    generic_hdec_prior_score = text_prior_score + generic_hdec_gate * (
        corrected_prior_score - text_prior_score
    )
    generic_hdec_debiased_score = (
        generic_hdec_score - float(args.hdec_beta_prior) * generic_hdec_prior_score
    )
    generic_hdec_pred = classify(generic_hdec_score, args.decision_threshold)
    generic_hdec_debiased_pred = classify(generic_hdec_debiased_score, args.decision_threshold)

    specific_hdec_gate_arg = float(args.hdec_gate_gamma) * (
        specific_defect_pool_score - float(args.hdec_gate_tau)
    )
    specific_hdec_gate = 1.0 / (1.0 + math.exp(-max(min(specific_hdec_gate_arg, 60.0), -60.0)))
    specific_hdec_score = final_score + specific_hdec_gate * (corrected_score - final_score)
    specific_hdec_prior_score = text_prior_score + specific_hdec_gate * (
        corrected_prior_score - text_prior_score
    )
    specific_hdec_debiased_score = (
        specific_hdec_score - float(args.hdec_beta_prior) * specific_hdec_prior_score
    )
    specific_hdec_pred = classify(specific_hdec_score, args.decision_threshold)
    specific_hdec_debiased_pred = classify(specific_hdec_debiased_score, args.decision_threshold)

    anchor_hdec_score = final_score + hdec_gate * (anchor_score - final_score)
    anchor_hdec_prior_score = text_prior_score + hdec_gate * (anchor_prior_score - text_prior_score)
    anchor_hdec_debiased_score = (
        anchor_hdec_score - float(args.hdec_beta_prior) * anchor_hdec_prior_score
    )
    anchor_hdec_pred = classify(anchor_hdec_score, args.decision_threshold)
    anchor_hdec_debiased_pred = classify(anchor_hdec_debiased_score, args.decision_threshold)
    suppression_delta = anchor_score - final_score
    suppression_gate_arg = float(args.suppression_gate_gamma) * (
        suppression_delta - float(args.suppression_gate_tau)
    )
    suppression_gate = 1.0 / (1.0 + math.exp(-max(min(suppression_gate_arg, 60.0), -60.0)))
    suppression_pool_gate_arg = float(args.suppression_pool_gate_gamma) * (
        defect_aware_pool_score - float(args.suppression_pool_gate_tau)
    )
    suppression_pool_gate = 1.0 / (
        1.0 + math.exp(-max(min(suppression_pool_gate_arg, 60.0), -60.0))
    )
    suppression_pool_combined_gate = suppression_gate * suppression_pool_gate
    suppression_hdec_score = final_score + suppression_gate * (anchor_score - final_score)
    suppression_hdec_prior_score = text_prior_score + suppression_gate * (
        anchor_prior_score - text_prior_score
    )
    suppression_hdec_debiased_score = (
        suppression_hdec_score - float(args.hdec_beta_prior) * suppression_hdec_prior_score
    )
    suppression_pool_hdec_score = final_score + suppression_pool_combined_gate * (
        anchor_score - final_score
    )
    suppression_pool_hdec_prior_score = text_prior_score + suppression_pool_combined_gate * (
        anchor_prior_score - text_prior_score
    )
    suppression_pool_hdec_debiased_score = (
        suppression_pool_hdec_score
        - float(args.hdec_beta_prior) * suppression_pool_hdec_prior_score
    )
    suppression_hdec_pred = classify(suppression_hdec_score, args.decision_threshold)
    suppression_hdec_debiased_pred = classify(
        suppression_hdec_debiased_score,
        args.decision_threshold,
    )
    suppression_pool_hdec_pred = classify(suppression_pool_hdec_score, args.decision_threshold)
    suppression_pool_hdec_debiased_pred = classify(
        suppression_pool_hdec_debiased_score,
        args.decision_threshold,
    )

    label = int(sample["label"])
    final_correct = int(final_pred == label)
    corrected_correct = int(corrected_pred == label)
    anchor_correct = int(anchor_pred == label)
    final_debiased_correct = int(final_debiased_pred == label)
    anchor_debiased_correct = int(anchor_debiased_pred == label)
    corrected_debiased_correct = int(corrected_debiased_pred == label)
    defect_aware_pool_correct = int(defect_aware_pool_pred == label)
    generic_defect_pool_correct = int(generic_defect_pool_pred == label)
    specific_defect_pool_correct = int(specific_defect_pool_pred == label)
    hdec_correct = int(hdec_pred == label)
    hdec_debiased_correct = int(hdec_debiased_pred == label)
    generic_hdec_correct = int(generic_hdec_pred == label)
    generic_hdec_debiased_correct = int(generic_hdec_debiased_pred == label)
    specific_hdec_correct = int(specific_hdec_pred == label)
    specific_hdec_debiased_correct = int(specific_hdec_debiased_pred == label)
    anchor_hdec_correct = int(anchor_hdec_pred == label)
    anchor_hdec_debiased_correct = int(anchor_hdec_debiased_pred == label)
    suppression_hdec_correct = int(suppression_hdec_pred == label)
    suppression_hdec_debiased_correct = int(suppression_hdec_debiased_pred == label)
    suppression_pool_hdec_correct = int(suppression_pool_hdec_pred == label)
    suppression_pool_hdec_debiased_correct = int(suppression_pool_hdec_debiased_pred == label)

    return {
        "image": sample["image"],
        "image_id": sample["image_id"],
        "category": sample["category"],
        "label": label,
        "label_name": sample["label_name"],
        "mask": sample.get("mask", ""),
        "prompt_mode": getattr(args, "prompt_mode", "generic"),
        "prompt_defect_limit": int(getattr(args, "prompt_defect_limit", 8)),
        "reference_image_count": len(reference_images),
        "reference_images": reference_paths,
        "prompt_text": text,
        "score_prefix": score_prefix,
        "generated_text": generation_text,
        "generated_label": generated_label,
        "generated_pred": 0 if generated_label == "good" else (1 if generated_label != "unknown" else None),
        "generation_deco_enabled": bool(args.generation_deco),
        "generated_deco_text": generation_deco_text,
        "generated_deco_label": generation_deco_label,
        "generated_deco_pred": 0
        if generation_deco_label == "good"
        else (1 if generation_deco_label != "unknown" else None),
        "score_normal_bank": score_normal_bank,
        "score_anomaly_bank": score_anomaly_bank,
        "text_prior_enabled": bool(text_prior_enabled),
        "text_prior_error": text_prior_error,
        "text_prior_prefix": text_prior_prefix,
        "normal_bank": normal_bank,
        "defect_bank": defect_bank,
        "generic_defect_bank": generic_defect_bank,
        "specific_defect_bank": specific_defect_bank,
        "hedge_bank": hedge_bank,
        "normal_tokenization": normal_ids_map,
        "anomaly_tokenization": anomaly_ids_map,
        "evidence_defect_tokenization": evidence_defect_ids_map,
        "generic_defect_tokenization": generic_defect_ids_map,
        "specific_defect_tokenization": specific_defect_ids_map,
        "evidence_normal_tokenization": evidence_normal_ids_map,
        "evidence_hedge_tokenization": evidence_hedge_ids_map,
        "hallucination_probe": {
            "text_prior_score": text_prior_score,
            "text_prior_pred": "anomaly" if text_prior_pred else "good",
            "text_prior_normal_prob": float(text_prior_probs[0].detach().cpu()),
            "text_prior_anomaly_prob": float(text_prior_probs[1].detach().cpu()),
            "final_score": final_score,
            "anchor_score": anchor_score,
            "corrected_score": corrected_score,
            "final_debiased_score": final_debiased_score,
            "anchor_debiased_score": anchor_debiased_score,
            "corrected_debiased_score": corrected_debiased_score,
            "defect_aware_pool_score": defect_aware_pool_score,
            "generic_defect_pool_score": generic_defect_pool_score,
            "specific_defect_pool_score": specific_defect_pool_score,
            "hdec_score": hdec_score,
            "hdec_debiased_score": hdec_debiased_score,
            "generic_hdec_score": generic_hdec_score,
            "generic_hdec_debiased_score": generic_hdec_debiased_score,
            "specific_hdec_score": specific_hdec_score,
            "specific_hdec_debiased_score": specific_hdec_debiased_score,
            "anchor_hdec_score": anchor_hdec_score,
            "anchor_hdec_debiased_score": anchor_hdec_debiased_score,
            "suppression_hdec_score": suppression_hdec_score,
            "suppression_hdec_debiased_score": suppression_hdec_debiased_score,
            "suppression_pool_hdec_score": suppression_pool_hdec_score,
            "suppression_pool_hdec_debiased_score": suppression_pool_hdec_debiased_score,
            "hdec_gate": hdec_gate,
            "hdec_gate_arg": hdec_gate_arg,
            "hdec_gate_gamma": args.hdec_gate_gamma,
            "hdec_gate_tau": args.hdec_gate_tau,
            "hdec_beta_prior": args.hdec_beta_prior,
            "hdec_prior_score": hdec_prior_score,
            "generic_hdec_gate": generic_hdec_gate,
            "generic_hdec_gate_arg": generic_hdec_gate_arg,
            "generic_hdec_prior_score": generic_hdec_prior_score,
            "specific_hdec_gate": specific_hdec_gate,
            "specific_hdec_gate_arg": specific_hdec_gate_arg,
            "specific_hdec_prior_score": specific_hdec_prior_score,
            "anchor_hdec_prior_score": anchor_hdec_prior_score,
            "suppression_delta": suppression_delta,
            "suppression_gate": suppression_gate,
            "suppression_gate_arg": suppression_gate_arg,
            "suppression_gate_gamma": args.suppression_gate_gamma,
            "suppression_gate_tau": args.suppression_gate_tau,
            "suppression_pool_gate": suppression_pool_gate,
            "suppression_pool_gate_arg": suppression_pool_gate_arg,
            "suppression_pool_gate_gamma": args.suppression_pool_gate_gamma,
            "suppression_pool_gate_tau": args.suppression_pool_gate_tau,
            "suppression_pool_combined_gate": suppression_pool_combined_gate,
            "suppression_hdec_prior_score": suppression_hdec_prior_score,
            "suppression_pool_hdec_prior_score": suppression_pool_hdec_prior_score,
            "anchor_prior_score": anchor_prior_score,
            "corrected_prior_score": corrected_prior_score,
            "selected_layer": int(primary_anchor["layer"]),
            "selected_layers": [int(row["layer"]) for row in selected_anchors],
            "selected_anchor_count": len(selected_anchors),
            "selected_anchor_details": selected_anchor_info,
            "score_deco_mode": args.score_deco_mode,
            "anchor_selection": args.anchor_selection,
            "decision_threshold": args.decision_threshold,
            "conflict": bool(final_pred != anchor_pred),
            "conflict_type": conflict_type(label, final_pred, anchor_pred),
            "final_pred": "anomaly" if final_pred else "good",
            "anchor_pred": "anomaly" if anchor_pred else "good",
            "corrected_pred": "anomaly" if corrected_pred else "good",
            "final_debiased_pred": "anomaly" if final_debiased_pred else "good",
            "anchor_debiased_pred": "anomaly" if anchor_debiased_pred else "good",
            "corrected_debiased_pred": "anomaly" if corrected_debiased_pred else "good",
            "defect_aware_pool_pred": "anomaly" if defect_aware_pool_pred else "good",
            "generic_defect_pool_pred": "anomaly" if generic_defect_pool_pred else "good",
            "specific_defect_pool_pred": "anomaly" if specific_defect_pool_pred else "good",
            "hdec_pred": "anomaly" if hdec_pred else "good",
            "hdec_debiased_pred": "anomaly" if hdec_debiased_pred else "good",
            "generic_hdec_pred": "anomaly" if generic_hdec_pred else "good",
            "generic_hdec_debiased_pred": "anomaly" if generic_hdec_debiased_pred else "good",
            "specific_hdec_pred": "anomaly" if specific_hdec_pred else "good",
            "specific_hdec_debiased_pred": "anomaly" if specific_hdec_debiased_pred else "good",
            "anchor_hdec_pred": "anomaly" if anchor_hdec_pred else "good",
            "anchor_hdec_debiased_pred": "anomaly" if anchor_hdec_debiased_pred else "good",
            "suppression_hdec_pred": "anomaly" if suppression_hdec_pred else "good",
            "suppression_hdec_debiased_pred": "anomaly" if suppression_hdec_debiased_pred else "good",
            "suppression_pool_hdec_pred": "anomaly" if suppression_pool_hdec_pred else "good",
            "suppression_pool_hdec_debiased_pred": "anomaly"
            if suppression_pool_hdec_debiased_pred
            else "good",
            "final_correct": bool(final_correct),
            "anchor_correct": bool(anchor_correct),
            "corrected_correct": bool(corrected_correct),
            "final_debiased_correct": bool(final_debiased_correct),
            "anchor_debiased_correct": bool(anchor_debiased_correct),
            "corrected_debiased_correct": bool(corrected_debiased_correct),
            "defect_aware_pool_correct": bool(defect_aware_pool_correct),
            "generic_defect_pool_correct": bool(generic_defect_pool_correct),
            "specific_defect_pool_correct": bool(specific_defect_pool_correct),
            "hdec_correct": bool(hdec_correct),
            "hdec_debiased_correct": bool(hdec_debiased_correct),
            "generic_hdec_correct": bool(generic_hdec_correct),
            "generic_hdec_debiased_correct": bool(generic_hdec_debiased_correct),
            "specific_hdec_correct": bool(specific_hdec_correct),
            "specific_hdec_debiased_correct": bool(specific_hdec_debiased_correct),
            "anchor_hdec_correct": bool(anchor_hdec_correct),
            "anchor_hdec_debiased_correct": bool(anchor_hdec_debiased_correct),
            "suppression_hdec_correct": bool(suppression_hdec_correct),
            "suppression_hdec_debiased_correct": bool(suppression_hdec_debiased_correct),
            "suppression_pool_hdec_correct": bool(suppression_pool_hdec_correct),
            "suppression_pool_hdec_debiased_correct": bool(suppression_pool_hdec_debiased_correct),
            "correction_success": bool((not final_correct) and corrected_correct),
            "debiased_correction_success": bool((not final_debiased_correct) and corrected_debiased_correct),
            "hdec_success": bool((not final_correct) and hdec_correct),
            "hdec_debiased_success": bool((not final_debiased_correct) and hdec_debiased_correct),
            "anchor_hdec_success": bool((not final_correct) and anchor_hdec_correct),
            "anchor_hdec_debiased_success": bool(
                (not final_debiased_correct) and anchor_hdec_debiased_correct
            ),
            "suppression_hdec_success": bool((not final_correct) and suppression_hdec_correct),
            "suppression_hdec_debiased_success": bool(
                (not final_debiased_correct) and suppression_hdec_debiased_correct
            ),
            "suppression_pool_hdec_success": bool(
                (not final_correct) and suppression_pool_hdec_correct
            ),
            "suppression_pool_hdec_debiased_success": bool(
                (not final_debiased_correct) and suppression_pool_hdec_debiased_correct
            ),
            "defect_aware_pool_selected_layers": defect_aware_pool.get("selected_layers", []),
            "defect_aware_pool_selected_details": defect_aware_pool.get("selected_details", []),
            "defect_aware_pool_layer_scores": defect_aware_pool.get("layer_scores", []),
            "defect_aware_pool_error": defect_aware_pool.get("error", ""),
            "generic_defect_pool_selected_layers": generic_defect_pool.get("selected_layers", []),
            "generic_defect_pool_selected_details": generic_defect_pool.get("selected_details", []),
            "generic_defect_pool_layer_scores": generic_defect_pool.get("layer_scores", []),
            "generic_defect_pool_error": generic_defect_pool.get("error", ""),
            "specific_defect_pool_selected_layers": specific_defect_pool.get("selected_layers", []),
            "specific_defect_pool_selected_details": specific_defect_pool.get("selected_details", []),
            "specific_defect_pool_layer_scores": specific_defect_pool.get("layer_scores", []),
            "specific_defect_pool_error": specific_defect_pool.get("error", ""),
            "defect_aware_pool_k": args.defect_aware_pool_k,
            "defect_aware_lambda_norm": args.defect_aware_lambda_norm,
            "defect_aware_eta_hedge": args.defect_aware_eta_hedge,
            "defect_aware_enable_hedge": args.defect_aware_enable_hedge,
            "defect_aware_use_logsumexp": args.defect_aware_use_logsumexp,
            "defect_aware_gamma": args.defect_aware_gamma,
            "final_normal_prob": float(final_probs[0].detach().cpu()),
            "final_anomaly_prob": float(final_probs[1].detach().cpu()),
            "anchor_normal_prob": float(selected_probs[0].detach().cpu()),
            "anchor_anomaly_prob": float(selected_probs[1].detach().cpu()),
            "selected_anchor_confidence": float(selected_confidence.detach().cpu()),
            "primary_anchor_normal_prob": float(primary_anchor["normal_prob"]),
            "primary_anchor_anomaly_prob": float(primary_anchor["anomaly_prob"]),
            "primary_anchor_score": float(primary_anchor["score"]),
            "corrected_normal_prob": float(corrected_probs[0].detach().cpu()),
            "corrected_anomaly_prob": float(corrected_probs[1].detach().cpu()),
            "deco_alpha": args.deco_alpha,
            "deco_threshold_top_p": args.deco_threshold_top_p,
            "deco_threshold_top_k": args.deco_threshold_top_k,
            "norm_module_found": bool(norm_module_found),
            "trend_layers": trend_layers,
            "deco_layers": sorted(deco_layers),
            "layer_scores": layer_rows,
            "text_prior_layer_scores": text_prior_layer_rows,
            "debiased_layer_scores": debiased_layer_rows,
        },
    }


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


def compact_case(case: Dict[str, Any], include_generated_text: bool = False) -> Dict[str, Any]:
    if "hallucination_probe" not in case:
        return {
            "image": case.get("image", ""),
            "image_id": case.get("image_id", ""),
            "category": case.get("category", ""),
            "label": case.get("label"),
            "label_name": case.get("label_name", ""),
            "error": case.get("error", ""),
        }

    probe = case["hallucination_probe"]
    selected_anchor_details = [
        {
            "layer": item["layer"],
            "score": item["score"],
            "normal_prob": item["normal_prob"],
            "anomaly_prob": item["anomaly_prob"],
            "confidence": item["confidence"],
            "selection_score": item["selection_score"],
            "fusion_weight": item["fusion_weight"],
            "text_prior_score": item.get("text_prior_score", 0.0),
            "debiased_score": item.get("debiased_score", item["score"]),
        }
        for item in probe.get("selected_anchor_details", [])
    ]
    layer_scores = [
        {
            "layer": row["layer"],
            "score": row["score"],
            "normal_prob": row["normal_prob"],
            "anomaly_prob": row["anomaly_prob"],
            "confidence": row["confidence"],
            "in_deco_range": row["in_deco_range"],
        }
        for row in probe.get("layer_scores", [])
    ]
    debiased_layer_scores = [
        {
            "layer": row["layer"],
            "raw_image_score": row.get("raw_image_score", row["score"]),
            "text_prior_score": row.get("text_prior_score", 0.0),
            "debiased_score": row.get("debiased_score", row["score"]),
            "normal_prob": row["normal_prob"],
            "anomaly_prob": row["anomaly_prob"],
            "confidence": row["confidence"],
            "in_deco_range": row["in_deco_range"],
        }
        for row in probe.get("debiased_layer_scores", [])
    ]
    defect_aware_pool_layer_scores = [
        {
            "layer": row["layer"],
            "score": row["score"],
            "selection_score": row["selection_score"],
            "confidence": row["confidence"],
            "defect_norm3": row["defect_norm3"],
            "normal_norm3": row["normal_norm3"],
            "hedge_norm3": row["hedge_norm3"],
        }
        for row in probe.get("defect_aware_pool_layer_scores", [])
    ]
    compact = {
        "image": case["image"],
        "image_id": case["image_id"],
        "category": case["category"],
        "label": case["label"],
        "label_name": case["label_name"],
        "generated": {
            "label": case.get("generated_label", "unknown"),
            "pred": case.get("generated_pred"),
        },
        "generation_deco": {
            "enabled": case.get("generation_deco_enabled", False),
            "label": case.get("generated_deco_label", "unknown"),
            "pred": case.get("generated_deco_pred"),
        },
        "probe": {
            "score_deco_mode": probe["score_deco_mode"],
            "anchor_selection": probe["anchor_selection"],
            "decision_threshold": probe["decision_threshold"],
            "text_prior_score": probe["text_prior_score"],
            "text_prior_pred": probe["text_prior_pred"],
            "text_prior_probs": {
                "normal": probe["text_prior_normal_prob"],
                "anomaly": probe["text_prior_anomaly_prob"],
            },
            "final_score": probe["final_score"],
            "anchor_score": probe["anchor_score"],
            "corrected_score": probe["corrected_score"],
            "final_debiased_score": probe["final_debiased_score"],
            "anchor_debiased_score": probe["anchor_debiased_score"],
            "corrected_debiased_score": probe["corrected_debiased_score"],
            "defect_aware_pool_score": probe["defect_aware_pool_score"],
            "generic_defect_pool_score": probe.get("generic_defect_pool_score"),
            "specific_defect_pool_score": probe.get("specific_defect_pool_score"),
            "hdec_score": probe["hdec_score"],
            "hdec_debiased_score": probe["hdec_debiased_score"],
            "generic_hdec_score": probe.get("generic_hdec_score"),
            "generic_hdec_debiased_score": probe.get("generic_hdec_debiased_score"),
            "specific_hdec_score": probe.get("specific_hdec_score"),
            "specific_hdec_debiased_score": probe.get("specific_hdec_debiased_score"),
            "anchor_hdec_score": probe["anchor_hdec_score"],
            "anchor_hdec_debiased_score": probe["anchor_hdec_debiased_score"],
            "suppression_hdec_score": probe["suppression_hdec_score"],
            "suppression_hdec_debiased_score": probe["suppression_hdec_debiased_score"],
            "suppression_pool_hdec_score": probe["suppression_pool_hdec_score"],
            "suppression_pool_hdec_debiased_score": probe["suppression_pool_hdec_debiased_score"],
            "hdec_gate": probe["hdec_gate"],
            "hdec_gate_arg": probe["hdec_gate_arg"],
            "hdec_gate_gamma": probe["hdec_gate_gamma"],
            "hdec_gate_tau": probe["hdec_gate_tau"],
            "hdec_beta_prior": probe["hdec_beta_prior"],
            "hdec_prior_score": probe["hdec_prior_score"],
            "generic_hdec_gate": probe.get("generic_hdec_gate"),
            "specific_hdec_gate": probe.get("specific_hdec_gate"),
            "anchor_hdec_prior_score": probe["anchor_hdec_prior_score"],
            "suppression_delta": probe["suppression_delta"],
            "suppression_gate": probe["suppression_gate"],
            "suppression_gate_arg": probe["suppression_gate_arg"],
            "suppression_gate_gamma": probe["suppression_gate_gamma"],
            "suppression_gate_tau": probe["suppression_gate_tau"],
            "suppression_pool_gate": probe["suppression_pool_gate"],
            "suppression_pool_gate_arg": probe["suppression_pool_gate_arg"],
            "suppression_pool_gate_gamma": probe["suppression_pool_gate_gamma"],
            "suppression_pool_gate_tau": probe["suppression_pool_gate_tau"],
            "suppression_pool_combined_gate": probe["suppression_pool_combined_gate"],
            "suppression_hdec_prior_score": probe["suppression_hdec_prior_score"],
            "suppression_pool_hdec_prior_score": probe["suppression_pool_hdec_prior_score"],
            "final_pred": probe["final_pred"],
            "anchor_pred": probe["anchor_pred"],
            "corrected_pred": probe["corrected_pred"],
            "final_debiased_pred": probe["final_debiased_pred"],
            "anchor_debiased_pred": probe["anchor_debiased_pred"],
            "corrected_debiased_pred": probe["corrected_debiased_pred"],
            "defect_aware_pool_pred": probe["defect_aware_pool_pred"],
            "generic_defect_pool_pred": probe.get("generic_defect_pool_pred"),
            "specific_defect_pool_pred": probe.get("specific_defect_pool_pred"),
            "hdec_pred": probe["hdec_pred"],
            "hdec_debiased_pred": probe["hdec_debiased_pred"],
            "generic_hdec_pred": probe.get("generic_hdec_pred"),
            "generic_hdec_debiased_pred": probe.get("generic_hdec_debiased_pred"),
            "specific_hdec_pred": probe.get("specific_hdec_pred"),
            "specific_hdec_debiased_pred": probe.get("specific_hdec_debiased_pred"),
            "anchor_hdec_pred": probe["anchor_hdec_pred"],
            "anchor_hdec_debiased_pred": probe["anchor_hdec_debiased_pred"],
            "suppression_hdec_pred": probe["suppression_hdec_pred"],
            "suppression_hdec_debiased_pred": probe["suppression_hdec_debiased_pred"],
            "suppression_pool_hdec_pred": probe["suppression_pool_hdec_pred"],
            "suppression_pool_hdec_debiased_pred": probe["suppression_pool_hdec_debiased_pred"],
            "final_correct": probe["final_correct"],
            "anchor_correct": probe["anchor_correct"],
            "corrected_correct": probe["corrected_correct"],
            "final_debiased_correct": probe["final_debiased_correct"],
            "anchor_debiased_correct": probe["anchor_debiased_correct"],
            "corrected_debiased_correct": probe["corrected_debiased_correct"],
            "defect_aware_pool_correct": probe["defect_aware_pool_correct"],
            "generic_defect_pool_correct": probe.get("generic_defect_pool_correct"),
            "specific_defect_pool_correct": probe.get("specific_defect_pool_correct"),
            "hdec_correct": probe["hdec_correct"],
            "hdec_debiased_correct": probe["hdec_debiased_correct"],
            "generic_hdec_correct": probe.get("generic_hdec_correct"),
            "generic_hdec_debiased_correct": probe.get("generic_hdec_debiased_correct"),
            "specific_hdec_correct": probe.get("specific_hdec_correct"),
            "specific_hdec_debiased_correct": probe.get("specific_hdec_debiased_correct"),
            "anchor_hdec_correct": probe["anchor_hdec_correct"],
            "anchor_hdec_debiased_correct": probe["anchor_hdec_debiased_correct"],
            "suppression_hdec_correct": probe["suppression_hdec_correct"],
            "suppression_hdec_debiased_correct": probe["suppression_hdec_debiased_correct"],
            "suppression_pool_hdec_correct": probe["suppression_pool_hdec_correct"],
            "suppression_pool_hdec_debiased_correct": probe["suppression_pool_hdec_debiased_correct"],
            "conflict": probe["conflict"],
            "conflict_type": probe["conflict_type"],
            "correction_success": probe["correction_success"],
            "debiased_correction_success": probe["debiased_correction_success"],
            "hdec_success": probe["hdec_success"],
            "hdec_debiased_success": probe["hdec_debiased_success"],
            "anchor_hdec_success": probe["anchor_hdec_success"],
            "anchor_hdec_debiased_success": probe["anchor_hdec_debiased_success"],
            "suppression_hdec_success": probe["suppression_hdec_success"],
            "suppression_hdec_debiased_success": probe["suppression_hdec_debiased_success"],
            "suppression_pool_hdec_success": probe["suppression_pool_hdec_success"],
            "suppression_pool_hdec_debiased_success": probe["suppression_pool_hdec_debiased_success"],
            "final_probs": {
                "normal": probe["final_normal_prob"],
                "anomaly": probe["final_anomaly_prob"],
            },
            "anchor_probs": {
                "normal": probe["anchor_normal_prob"],
                "anomaly": probe["anchor_anomaly_prob"],
            },
            "corrected_probs": {
                "normal": probe["corrected_normal_prob"],
                "anomaly": probe["corrected_anomaly_prob"],
            },
            "selected_layer": probe["selected_layer"],
            "selected_layers": probe.get("selected_layers", [probe["selected_layer"]]),
            "selected_anchor_details": selected_anchor_details,
            "defect_aware_pool_selected_layers": probe["defect_aware_pool_selected_layers"],
            "defect_aware_pool_selected_details": probe["defect_aware_pool_selected_details"],
            "defect_aware_pool_error": probe["defect_aware_pool_error"],
            "generic_defect_pool_selected_layers": probe.get("generic_defect_pool_selected_layers", []),
            "specific_defect_pool_selected_layers": probe.get("specific_defect_pool_selected_layers", []),
            "generic_defect_pool_error": probe.get("generic_defect_pool_error", ""),
            "specific_defect_pool_error": probe.get("specific_defect_pool_error", ""),
            "defect_aware_pool_k": probe["defect_aware_pool_k"],
            "defect_aware_lambda_norm": probe["defect_aware_lambda_norm"],
            "defect_aware_eta_hedge": probe["defect_aware_eta_hedge"],
            "defect_aware_enable_hedge": probe["defect_aware_enable_hedge"],
            "deco_layers": probe["deco_layers"],
            "deco_alpha": probe["deco_alpha"],
            "layer_scores": layer_scores,
            "debiased_layer_scores": debiased_layer_scores,
            "defect_aware_pool_layer_scores": defect_aware_pool_layer_scores,
        },
    }
    if include_generated_text:
        compact["generated"]["text"] = case.get("generated_text", "")
        compact["generation_deco"]["text"] = case.get("generated_deco_text", "")
    return compact


def select_threshold(
    labels: List[int],
    scores: List[float],
    policy: str,
    fixed_threshold: float = 0.0,
) -> float:
    if policy == "zero":
        return 0.0
    if policy == "fixed":
        return float(fixed_threshold)

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

    def div0(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    best_threshold = candidates[0]
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        tp, tn, fp, fn = stats_at(threshold)
        acc = div0(tp + tn, len(records))
        recall_anomaly = div0(tp, tp + fn)
        recall_normal = div0(tn, tn + fp)
        precision_anomaly = div0(tp, tp + fp)
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


def binary_metrics(
    labels: List[int],
    scores: List[float],
    threshold_policy: str = "youden",
    fixed_threshold: float = 0.0,
    score_source: str = "",
) -> Dict[str, Any]:
    threshold = select_threshold(labels, scores, threshold_policy, fixed_threshold)
    preds = [1 if s >= threshold else 0 for s in scores]
    total = len(labels)
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    def div(num: float, den: float) -> Optional[float]:
        return num / den if den > 0 else None

    precision_anomaly = div(tp, tp + fp)
    recall_anomaly = div(tp, tp + fn)
    f1_anomaly = (
        2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
        if precision_anomaly is not None and recall_anomaly is not None and (precision_anomaly + recall_anomaly) > 0
        else None
    )
    precision_normal = div(tn, tn + fn)
    recall_normal = div(tn, tn + fp)
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if precision_normal is not None and recall_normal is not None and (precision_normal + recall_normal) > 0
        else None
    )
    return {
        "total": total,
        "acc": div(tp + tn, total),
        "AUROC": binary_auroc(labels, scores),
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "score_source": score_source,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": sum(x for x in [recall_normal, recall_anomaly] if x is not None)
        / max(1, len([x for x in [recall_normal, recall_anomaly] if x is not None])),
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": sum(x for x in [precision_normal, precision_anomaly] if x is not None)
        / max(1, len([x for x in [precision_normal, precision_anomaly] if x is not None])),
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": sum(x for x in [f1_normal, f1_anomaly] if x is not None)
        / max(1, len([x for x in [f1_normal, f1_anomaly] if x is not None])),
        "recall_micro": div(tp + tn, total),
        "precision_micro": div(tp + tn, total),
        "F1_micro": div(tp + tn, total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": tp + fn,
        "normal_total": tn + fp,
    }


def _metric_value(metric: Dict[str, Any], key: str) -> float:
    value = metric.get(key)
    return float(value) if value is not None else -1.0


def _slim_metric(metric: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metric:
        return None
    keys = [
        "threshold",
        "acc",
        "AUROC",
        "recall_normal",
        "recall_anomaly",
        "precision_normal",
        "precision_anomaly",
        "F1_normal",
        "F1_anomaly",
        "tp",
        "tn",
        "fp",
        "fn",
        "pred_anomaly",
        "pred_normal",
        "anomaly_total",
        "normal_total",
    ]
    return {key: metric.get(key) for key in keys}


def threshold_target_diagnostics(
    labels: List[int],
    scores: List[float],
    score_source: str,
    target_acc: float,
    target_auroc: float,
    target_recall: float,
    target_precision: float,
    target_f1: float,
) -> Dict[str, Any]:
    records = [(int(y), float(s)) for y, s in zip(labels, scores)]
    if not records:
        return {"score_source": score_source, "total": 0}

    unique_scores = sorted({score for _, score in records})
    thresholds = [unique_scores[0] - 1e-12] + unique_scores + [unique_scores[-1] + 1e-12]
    candidates = [
        binary_metrics(
            [y for y, _ in records],
            [s for _, s in records],
            threshold_policy="fixed",
            fixed_threshold=threshold,
            score_source=score_source,
        )
        for threshold in thresholds
    ]

    def best(items: List[Dict[str, Any]], primary: str, *tie_keys: str) -> Optional[Dict[str, Any]]:
        if not items:
            return None
        return max(items, key=lambda item: tuple(_metric_value(item, key) for key in (primary,) + tie_keys))

    precision_ok = [
        item for item in candidates if _metric_value(item, "precision_anomaly") >= target_precision
    ]
    recall_ok = [item for item in candidates if _metric_value(item, "recall_anomaly") >= target_recall]
    all_classification_targets_ok = [
        item
        for item in candidates
        if _metric_value(item, "acc") >= target_acc
        and _metric_value(item, "recall_anomaly") >= target_recall
        and _metric_value(item, "precision_anomaly") >= target_precision
        and _metric_value(item, "F1_anomaly") >= target_f1
    ]

    auroc = binary_auroc([y for y, _ in records], [s for _, s in records])
    best_acc = best(candidates, "acc", "F1_anomaly", "recall_anomaly", "precision_anomaly")
    best_f1 = best(candidates, "F1_anomaly", "acc", "recall_anomaly", "precision_anomaly")
    best_precision_at_recall = best(recall_ok, "precision_anomaly", "F1_anomaly", "acc")
    best_recall_at_precision = best(precision_ok, "recall_anomaly", "F1_anomaly", "acc")
    best_f1_at_precision = best(precision_ok, "F1_anomaly", "recall_anomaly", "acc")
    best_all_targets = best(all_classification_targets_ok, "F1_anomaly", "acc", "recall_anomaly")

    return {
        "score_source": score_source,
        "total": len(records),
        "AUROC": auroc,
        "target_acc": target_acc,
        "target_AUROC": target_auroc,
        "target_recall_anomaly": target_recall,
        "target_precision_anomaly": target_precision,
        "target_F1_anomaly": target_f1,
        "beats_AUROC_target": bool(auroc is not None and auroc >= target_auroc),
        "can_meet_precision_target": bool(precision_ok),
        "can_meet_recall_target": bool(recall_ok),
        "can_meet_all_classification_targets": bool(all_classification_targets_ok),
        "can_exceed_all_targets": bool(
            auroc is not None and auroc >= target_auroc and all_classification_targets_ok
        ),
        "best_acc": _slim_metric(best_acc),
        "best_F1": _slim_metric(best_f1),
        "best_precision_at_target_recall": _slim_metric(best_precision_at_recall),
        "best_recall_at_target_precision": _slim_metric(best_recall_at_precision),
        "best_F1_at_target_precision": _slim_metric(best_f1_at_precision),
        "best_all_classification_targets": _slim_metric(best_all_targets),
    }


def score_separation_diagnostics(labels: List[int], values: List[float], score_source: str) -> Dict[str, Any]:
    normal_values = [float(v) for y, v in zip(labels, values) if int(y) == 0]
    anomaly_values = [float(v) for y, v in zip(labels, values) if int(y) == 1]

    def mean(items: List[float]) -> Optional[float]:
        return sum(items) / len(items) if items else None

    mean_normal = mean(normal_values)
    mean_anomaly = mean(anomaly_values)
    return {
        "score_source": score_source,
        "mean_normal": mean_normal,
        "mean_anomaly": mean_anomaly,
        "mean_gap_anomaly_minus_normal": (
            mean_anomaly - mean_normal if mean_anomaly is not None and mean_normal is not None else None
        ),
        "AUROC": binary_auroc(labels, values),
        "normal_count": len(normal_values),
        "anomaly_count": len(anomaly_values),
    }


def source_reliability_diagnostics(labels: List[int], score_map: Dict[str, List[float]]) -> Dict[str, Any]:
    aurocs = {name: binary_auroc(labels, scores) for name, scores in score_map.items()}
    final_auroc = aurocs.get("final")
    deltas = {
        name: (value - final_auroc if value is not None and final_auroc is not None else None)
        for name, value in aurocs.items()
    }
    best_source = None
    valid_aurocs = {name: value for name, value in aurocs.items() if value is not None}
    if valid_aurocs:
        best_source = max(valid_aurocs.items(), key=lambda item: item[1])[0]
    return {
        "AUROC_by_source": aurocs,
        "AUROC_delta_vs_final": deltas,
        "best_source_by_AUROC": best_source,
    }


def hard_prediction_metrics(cases: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    labels = []
    scores = []
    total = len(cases)
    for case in cases:
        pred = case.get(pred_key)
        if pred is None:
            continue
        labels.append(int(case["label"]))
        scores.append(float(pred))
    if not labels:
        return {
            "total": total,
            "valid": 0,
            "coverage": 0.0,
            "acc": None,
            "AUROC": None,
            "threshold_policy": "fixed",
            "threshold": 0.5,
            "score_source": pred_key,
        }
    metric = binary_metrics(
        labels,
        scores,
        threshold_policy="fixed",
        fixed_threshold=0.5,
        score_source=pred_key,
    )
    metric["valid"] = len(labels)
    metric["total"] = total
    metric["coverage"] = len(labels) / max(1, total)
    return metric


def summarize_cases(
    cases: List[Dict[str, Any]],
    sign_threshold: float,
    metric_threshold_policy: str,
    metric_threshold: float,
    target_acc: float,
    target_auroc: float,
    target_recall: float,
    target_precision: float,
    target_f1: float,
) -> Dict[str, Any]:
    labels = [int(case["label"]) for case in cases]
    text_prior_scores = [float(case["hallucination_probe"]["text_prior_score"]) for case in cases]
    final_scores = [float(case["hallucination_probe"]["final_score"]) for case in cases]
    anchor_scores = [float(case["hallucination_probe"]["anchor_score"]) for case in cases]
    corrected_scores = [float(case["hallucination_probe"]["corrected_score"]) for case in cases]
    final_debiased_scores = [float(case["hallucination_probe"]["final_debiased_score"]) for case in cases]
    anchor_debiased_scores = [float(case["hallucination_probe"]["anchor_debiased_score"]) for case in cases]
    corrected_debiased_scores = [float(case["hallucination_probe"]["corrected_debiased_score"]) for case in cases]
    defect_aware_pool_scores = [float(case["hallucination_probe"]["defect_aware_pool_score"]) for case in cases]
    hdec_scores = [float(case["hallucination_probe"]["hdec_score"]) for case in cases]
    hdec_debiased_scores = [float(case["hallucination_probe"]["hdec_debiased_score"]) for case in cases]
    anchor_hdec_scores = [float(case["hallucination_probe"]["anchor_hdec_score"]) for case in cases]
    anchor_hdec_debiased_scores = [
        float(case["hallucination_probe"]["anchor_hdec_debiased_score"]) for case in cases
    ]
    suppression_hdec_scores = [
        float(case["hallucination_probe"]["suppression_hdec_score"]) for case in cases
    ]
    suppression_hdec_debiased_scores = [
        float(case["hallucination_probe"]["suppression_hdec_debiased_score"]) for case in cases
    ]
    suppression_pool_hdec_scores = [
        float(case["hallucination_probe"]["suppression_pool_hdec_score"]) for case in cases
    ]
    suppression_pool_hdec_debiased_scores = [
        float(case["hallucination_probe"]["suppression_pool_hdec_debiased_score"]) for case in cases
    ]
    continuous_score_map = {
        "text_prior": text_prior_scores,
        "final": final_scores,
        "anchor": anchor_scores,
        "corrected": corrected_scores,
        "final_debiased": final_debiased_scores,
        "anchor_debiased": anchor_debiased_scores,
        "corrected_debiased": corrected_debiased_scores,
        "defect_aware_pool": defect_aware_pool_scores,
        "hdec": hdec_scores,
        "hdec_debiased": hdec_debiased_scores,
        "anchor_hdec": anchor_hdec_scores,
        "anchor_hdec_debiased": anchor_hdec_debiased_scores,
        "suppression_hdec": suppression_hdec_scores,
        "suppression_hdec_debiased": suppression_hdec_debiased_scores,
        "suppression_pool_hdec": suppression_pool_hdec_scores,
        "suppression_pool_hdec_debiased": suppression_pool_hdec_debiased_scores,
    }

    normal_cases = [case for case in cases if int(case["label"]) == 0]
    anomaly_cases = [case for case in cases if int(case["label"]) == 1]
    final_wrong = [case for case in cases if not case["hallucination_probe"]["final_correct"]]
    final_debiased_wrong = [case for case in cases if not case["hallucination_probe"]["final_debiased_correct"]]

    lcr = sum(1 for case in cases if case["hallucination_probe"]["conflict"]) / max(1, len(cases))
    dhr = sum(1 for case in normal_cases if case["hallucination_probe"]["conflict_type"] == "defect_hallucination") / max(
        1, len(normal_cases)
    )
    dor = sum(1 for case in anomaly_cases if case["hallucination_probe"]["conflict_type"] == "defect_omission") / max(
        1, len(anomaly_cases)
    )
    csr = sum(1 for case in final_wrong if case["hallucination_probe"]["correction_success"]) / max(1, len(final_wrong))
    debiased_csr = sum(
        1 for case in final_debiased_wrong if case["hallucination_probe"]["debiased_correction_success"]
    ) / max(1, len(final_debiased_wrong))
    hdec_csr = sum(1 for case in final_wrong if case["hallucination_probe"]["hdec_success"]) / max(1, len(final_wrong))
    hdec_debiased_csr = sum(
        1 for case in final_debiased_wrong if case["hallucination_probe"]["hdec_debiased_success"]
    ) / max(1, len(final_debiased_wrong))
    anchor_hdec_csr = sum(
        1 for case in final_wrong if case["hallucination_probe"]["anchor_hdec_success"]
    ) / max(1, len(final_wrong))
    anchor_hdec_debiased_csr = sum(
        1 for case in final_debiased_wrong if case["hallucination_probe"]["anchor_hdec_debiased_success"]
    ) / max(1, len(final_debiased_wrong))
    suppression_hdec_csr = sum(
        1 for case in final_wrong if case["hallucination_probe"]["suppression_hdec_success"]
    ) / max(1, len(final_wrong))
    suppression_hdec_debiased_csr = sum(
        1 for case in final_debiased_wrong if case["hallucination_probe"]["suppression_hdec_debiased_success"]
    ) / max(1, len(final_debiased_wrong))
    suppression_pool_hdec_csr = sum(
        1 for case in final_wrong if case["hallucination_probe"]["suppression_pool_hdec_success"]
    ) / max(1, len(final_wrong))
    suppression_pool_hdec_debiased_csr = sum(
        1
        for case in final_debiased_wrong
        if case["hallucination_probe"]["suppression_pool_hdec_debiased_success"]
    ) / max(1, len(final_debiased_wrong))
    text_prior_pred_anomaly_rate = sum(1 for score in text_prior_scores if score >= sign_threshold) / max(
        1, len(text_prior_scores)
    )
    text_prior_mean_score = sum(text_prior_scores) / max(1, len(text_prior_scores))
    hdec_mean_gate = sum(float(case["hallucination_probe"]["hdec_gate"]) for case in cases) / max(1, len(cases))
    suppression_mean_gate = sum(
        float(case["hallucination_probe"]["suppression_gate"]) for case in cases
    ) / max(1, len(cases))
    suppression_pool_mean_gate = sum(
        float(case["hallucination_probe"]["suppression_pool_combined_gate"]) for case in cases
    ) / max(1, len(cases))

    summary = {
        "num_images": len(cases),
        "threshold_for_sign_metrics": sign_threshold,
        "metric_threshold_policy": metric_threshold_policy,
        "metric_threshold": metric_threshold,
        "target_metrics": {
            "ACC": target_acc,
            "I-AUROC": target_auroc,
            "recall_anomaly": target_recall,
            "precision_anomaly": target_precision,
            "F1_anomaly": target_f1,
        },
        "generated_metrics": hard_prediction_metrics(cases, "generated_pred"),
        "generation_deco_metrics": hard_prediction_metrics(cases, "generated_deco_pred"),
        "text_prior_metrics": binary_metrics(
            labels,
            text_prior_scores,
            metric_threshold_policy,
            metric_threshold,
            "text_prior_score",
        ),
        "final_metrics": binary_metrics(
            labels,
            final_scores,
            metric_threshold_policy,
            metric_threshold,
            "final_score",
        ),
        "anchor_metrics": binary_metrics(
            labels,
            anchor_scores,
            metric_threshold_policy,
            metric_threshold,
            "anchor_score",
        ),
        "corrected_metrics": binary_metrics(
            labels,
            corrected_scores,
            metric_threshold_policy,
            metric_threshold,
            "corrected_score",
        ),
        "final_debiased_metrics": binary_metrics(
            labels,
            final_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "final_debiased_score",
        ),
        "anchor_debiased_metrics": binary_metrics(
            labels,
            anchor_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "anchor_debiased_score",
        ),
        "corrected_debiased_metrics": binary_metrics(
            labels,
            corrected_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "corrected_debiased_score",
        ),
        "defect_aware_pool_metrics": binary_metrics(
            labels,
            defect_aware_pool_scores,
            metric_threshold_policy,
            metric_threshold,
            "defect_aware_pool_score",
        ),
        "hdec_metrics": binary_metrics(
            labels,
            hdec_scores,
            metric_threshold_policy,
            metric_threshold,
            "hdec_score",
        ),
        "hdec_debiased_metrics": binary_metrics(
            labels,
            hdec_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "hdec_debiased_score",
        ),
        "anchor_hdec_metrics": binary_metrics(
            labels,
            anchor_hdec_scores,
            metric_threshold_policy,
            metric_threshold,
            "anchor_hdec_score",
        ),
        "anchor_hdec_debiased_metrics": binary_metrics(
            labels,
            anchor_hdec_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "anchor_hdec_debiased_score",
        ),
        "suppression_hdec_metrics": binary_metrics(
            labels,
            suppression_hdec_scores,
            metric_threshold_policy,
            metric_threshold,
            "suppression_hdec_score",
        ),
        "suppression_hdec_debiased_metrics": binary_metrics(
            labels,
            suppression_hdec_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "suppression_hdec_debiased_score",
        ),
        "suppression_pool_hdec_metrics": binary_metrics(
            labels,
            suppression_pool_hdec_scores,
            metric_threshold_policy,
            metric_threshold,
            "suppression_pool_hdec_score",
        ),
        "suppression_pool_hdec_debiased_metrics": binary_metrics(
            labels,
            suppression_pool_hdec_debiased_scores,
            metric_threshold_policy,
            metric_threshold,
            "suppression_pool_hdec_debiased_score",
        ),
        "source_reliability": source_reliability_diagnostics(labels, continuous_score_map),
        "gate_diagnostics": {
            "hdec_gate": score_separation_diagnostics(
                labels,
                [float(case["hallucination_probe"]["hdec_gate"]) for case in cases],
                "hdec_gate",
            ),
            "suppression_gate": score_separation_diagnostics(
                labels,
                [float(case["hallucination_probe"]["suppression_gate"]) for case in cases],
                "suppression_gate",
            ),
            "suppression_pool_gate": score_separation_diagnostics(
                labels,
                [float(case["hallucination_probe"]["suppression_pool_gate"]) for case in cases],
                "suppression_pool_gate",
            ),
            "suppression_pool_combined_gate": score_separation_diagnostics(
                labels,
                [float(case["hallucination_probe"]["suppression_pool_combined_gate"]) for case in cases],
                "suppression_pool_combined_gate",
            ),
            "suppression_delta_anchor_minus_final": score_separation_diagnostics(
                labels,
                [float(case["hallucination_probe"]["suppression_delta"]) for case in cases],
                "suppression_delta_anchor_minus_final",
            ),
            "defect_aware_pool_score": score_separation_diagnostics(
                labels,
                defect_aware_pool_scores,
                "defect_aware_pool_score",
            ),
        },
        "target_diagnostics": {
            name: threshold_target_diagnostics(
                labels,
                scores,
                f"{name}_score",
                target_acc,
                target_auroc,
                target_recall,
                target_precision,
                target_f1,
            )
            for name, scores in continuous_score_map.items()
        },
        "mechanism_metrics": {
            "LCR_layer_conflict_rate": lcr,
            "DHR_defect_hallucination_rate_on_normal": dhr,
            "DOR_defect_omission_rate_on_anomaly": dor,
            "CSR_correction_success_rate": csr,
            "debiased_CSR_correction_success_rate": debiased_csr,
            "HDEC_CSR_correction_success_rate": hdec_csr,
            "HDEC_debiased_CSR_correction_success_rate": hdec_debiased_csr,
            "Anchor_HDEC_CSR_correction_success_rate": anchor_hdec_csr,
            "Anchor_HDEC_debiased_CSR_correction_success_rate": anchor_hdec_debiased_csr,
            "Suppression_HDEC_CSR_correction_success_rate": suppression_hdec_csr,
            "Suppression_HDEC_debiased_CSR_correction_success_rate": suppression_hdec_debiased_csr,
            "Suppression_pool_HDEC_CSR_correction_success_rate": suppression_pool_hdec_csr,
            "Suppression_pool_HDEC_debiased_CSR_correction_success_rate": suppression_pool_hdec_debiased_csr,
            "HDEC_mean_gate": hdec_mean_gate,
            "Suppression_HDEC_mean_gate": suppression_mean_gate,
            "Suppression_pool_HDEC_mean_gate": suppression_pool_mean_gate,
            "text_prior_pred_anomaly_rate": text_prior_pred_anomaly_rate,
            "text_prior_mean_score": text_prior_mean_score,
            "final_wrong_count": len(final_wrong),
            "final_debiased_wrong_count": len(final_debiased_wrong),
            "defect_aware_pool_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["defect_aware_pool_correct"]
            ),
            "hdec_wrong_count": sum(1 for case in cases if not case["hallucination_probe"]["hdec_correct"]),
            "hdec_debiased_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["hdec_debiased_correct"]
            ),
            "anchor_hdec_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["anchor_hdec_correct"]
            ),
            "anchor_hdec_debiased_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["anchor_hdec_debiased_correct"]
            ),
            "suppression_hdec_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["suppression_hdec_correct"]
            ),
            "suppression_hdec_debiased_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["suppression_hdec_debiased_correct"]
            ),
            "suppression_pool_hdec_wrong_count": sum(
                1 for case in cases if not case["hallucination_probe"]["suppression_pool_hdec_correct"]
            ),
            "suppression_pool_hdec_debiased_wrong_count": sum(
                1
                for case in cases
                if not case["hallucination_probe"]["suppression_pool_hdec_debiased_correct"]
            ),
            "normal_count": len(normal_cases),
            "anomaly_count": len(anomaly_cases),
        },
    }
    return summary


def save_curve_plot(case: Dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plot: {exc}")
        return

    probe = case["hallucination_probe"]
    layers = [row["layer"] for row in probe["layer_scores"]]
    scores = [row["score"] for row in probe["layer_scores"]]
    debiased_rows = probe.get("debiased_layer_scores", [])
    debiased_by_layer = {int(row["layer"]): row for row in debiased_rows}
    debiased_scores = [
        float(debiased_by_layer.get(int(layer), {}).get("debiased_score", score))
        for layer, score in zip(layers, scores)
    ]
    selected_layer = int(probe["selected_layer"])
    selected_layers = [int(x) for x in probe.get("selected_layers", [selected_layer])]
    pool_selected_layers = [int(x) for x in probe.get("defect_aware_pool_selected_layers", [])]
    final_score = float(probe["final_score"])
    corrected_score = float(probe["corrected_score"])
    text_prior_score = float(probe.get("text_prior_score", 0.0))
    final_debiased_score = float(probe.get("final_debiased_score", final_score))
    corrected_debiased_score = float(probe.get("corrected_debiased_score", corrected_score))
    hdec_score = float(probe.get("hdec_score", final_score))
    hdec_debiased_score = float(probe.get("hdec_debiased_score", hdec_score))
    anchor_hdec_score = float(probe.get("anchor_hdec_score", final_score))
    anchor_hdec_debiased_score = float(probe.get("anchor_hdec_debiased_score", anchor_hdec_score))
    suppression_hdec_score = float(probe.get("suppression_hdec_score", final_score))
    suppression_pool_hdec_score = float(probe.get("suppression_pool_hdec_score", final_score))
    threshold = float(probe["decision_threshold"])
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    plt.plot(layers, scores, marker="o", linewidth=1.8, label="layer anomaly score")
    if debiased_rows:
        plt.plot(
            layers,
            debiased_scores,
            marker="s",
            linewidth=1.4,
            alpha=0.85,
            label="layer score - text prior",
        )
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="decision threshold")
    plt.axhline(final_score, color="tab:blue", linestyle=":", linewidth=1.4, label="final score")
    plt.axhline(corrected_score, color="tab:green", linestyle=":", linewidth=1.4, label="corrected score")
    plt.axhline(hdec_score, color="tab:cyan", linestyle="--", linewidth=1.2, label="HDEC gated score")
    plt.axhline(
        anchor_hdec_score,
        color="tab:gray",
        linestyle="--",
        linewidth=1.2,
        label="Anchor-HDEC gated score",
    )
    plt.axhline(
        suppression_hdec_score,
        color="tab:red",
        linestyle="-.",
        linewidth=1.0,
        label="Suppression-HDEC score",
    )
    plt.axhline(
        suppression_pool_hdec_score,
        color="tab:blue",
        linestyle="-.",
        linewidth=1.0,
        label="Suppression-pool-HDEC score",
    )
    plt.axhline(text_prior_score, color="tab:orange", linestyle="-.", linewidth=1.1, label="text-only prior")
    plt.axhline(final_debiased_score, color="tab:purple", linestyle=":", linewidth=1.1, label="final - text prior")
    plt.axhline(
        corrected_debiased_score,
        color="tab:brown",
        linestyle=":",
        linewidth=1.1,
        label="corrected - text prior",
    )
    plt.axhline(
        hdec_debiased_score,
        color="tab:pink",
        linestyle=":",
        linewidth=1.1,
        label="HDEC - gated prior",
    )
    plt.axhline(
        anchor_hdec_debiased_score,
        color="tab:olive",
        linestyle=":",
        linewidth=1.1,
        label="Anchor-HDEC - gated prior",
    )
    for layer in selected_layers:
        label = f"primary anchor {layer}" if layer == selected_layer else f"top anchor {layer}"
        plt.axvline(layer, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.85, label=label)
    for layer in pool_selected_layers:
        plt.axvline(
            layer,
            color="tab:olive",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.75,
            label=f"defect-aware pool layer {layer}",
        )
    title = f"{case['category']} {case['image_id']} gt={case['label_name']} conflict={probe['conflict_type']}"
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Anomaly score = anomaly logit - normal logit")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "0-config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
        },
    )

    samples = load_split_samples(args)
    if not samples:
        raise RuntimeError("No samples found. Check --visa-root, --visa-split-csv, --visa-split-set, and --categories.")
    print(f"[INFO] Loaded {len(samples)} samples for mechanism validation.")

    normal_bank = parse_csv_words(args.normal_labels)
    hedge_bank = parse_csv_words(args.hedge_labels)
    generic_defect_labels = parse_csv_words(args.generic_defect_labels)
    categories = sorted({sample["category"] for sample in samples})
    defect_bank_by_category = {
        category: build_category_defect_bank(Path(args.visa_root), category, generic_defect_labels, args.max_defect_labels)
        for category in categories
    }

    processor, model = load_model(args)
    cases = []
    compact_cases = []
    results_json = output_dir / "mechanism_cases.json"
    stream_jsonl = output_dir / "mechanism_cases.compact.jsonl"
    full_jsonl = output_dir / "mechanism_cases.full.jsonl"
    for stale_path in (stream_jsonl, full_jsonl):
        if stale_path.exists():
            stale_path.unlink()

    for idx, sample in enumerate(tqdm(samples, desc="Mechanism validation", ncols=120), start=1):
        defect_bank = defect_bank_by_category[sample["category"]]
        try:
            case = compute_probe(args, processor, model, sample, defect_bank, normal_bank, hedge_bank)
        except Exception as exc:
            case = {
                **sample,
                "error": str(exc),
            }
            print(f"[WARN] Failed sample {sample['category']}/{sample['image_id']}: {exc}")
        cases.append(case)
        compact = compact_case(case, include_generated_text=args.include_generated_text)
        compact_cases.append(compact)
        if args.stream_jsonl:
            with open(stream_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(compact, ensure_ascii=False) + "\n")
        if args.save_full_cases:
            with open(full_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        if args.save_plots and "hallucination_probe" in case and idx <= args.plot_limit:
            plot_name = f"{idx:04d}-{case['category']}-{case['label_name']}-{case['image_id']}-layer_curve.png"
            save_curve_plot(case, output_dir / "plots" / plot_name)

    valid_cases = [case for case in cases if "hallucination_probe" in case]
    failed_cases = [case for case in cases if "hallucination_probe" not in case]
    summary = {
        "overall": summarize_cases(
            valid_cases,
            args.decision_threshold,
            args.metric_threshold_policy,
            args.metric_threshold,
            args.target_acc,
            args.target_auroc,
            args.target_recall,
            args.target_precision,
            args.target_f1,
        )
        if valid_cases
        else {},
        "per_category": {},
        "num_failed": len(failed_cases),
        "failed_cases": failed_cases[:50],
    }
    for category in sorted({case["category"] for case in valid_cases}):
        category_cases = [case for case in valid_cases if case["category"] == category]
        summary["per_category"][category] = summarize_cases(
            category_cases,
            args.decision_threshold,
            args.metric_threshold_policy,
            args.metric_threshold,
            args.target_acc,
            args.target_auroc,
            args.target_recall,
            args.target_precision,
            args.target_f1,
        )

    save_json(output_dir / "mechanism_summary.json", summary)
    save_json(results_json, compact_cases)
    save_json(output_dir / "defect_bank_by_category.json", defect_bank_by_category)

    csv_path = output_dir / "mechanism_summary.csv"
    rows = []
    for scope, category, data in [("overall", "ALL", summary["overall"])] + [
        ("category", category, data) for category, data in summary["per_category"].items()
    ]:
        if not data:
            continue
        mech = data["mechanism_metrics"]
        for method in [
            "generated_metrics",
            "generation_deco_metrics",
            "text_prior_metrics",
            "final_metrics",
            "anchor_metrics",
            "corrected_metrics",
            "final_debiased_metrics",
            "anchor_debiased_metrics",
            "corrected_debiased_metrics",
            "defect_aware_pool_metrics",
            "hdec_metrics",
            "hdec_debiased_metrics",
            "anchor_hdec_metrics",
            "anchor_hdec_debiased_metrics",
            "suppression_hdec_metrics",
            "suppression_hdec_debiased_metrics",
            "suppression_pool_hdec_metrics",
            "suppression_pool_hdec_debiased_metrics",
        ]:
            metric = data[method]
            rows.append(
                {
                    "scope": scope,
                    "category": category,
                    "method": method.replace("_metrics", ""),
                    **metric,
                    **mech,
                }
            )
    if rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    target_diag_path = output_dir / "mechanism_target_diagnostics.csv"
    target_rows = []
    for scope, category, data in [("overall", "ALL", summary["overall"])] + [
        ("category", category, data) for category, data in summary["per_category"].items()
    ]:
        if not data:
            continue
        for method, diag in data.get("target_diagnostics", {}).items():
            row = {
                "scope": scope,
                "category": category,
                "method": method,
                "total": diag.get("total"),
                "AUROC": diag.get("AUROC"),
                "target_acc": diag.get("target_acc"),
                "target_AUROC": diag.get("target_AUROC"),
                "target_recall_anomaly": diag.get("target_recall_anomaly"),
                "target_precision_anomaly": diag.get("target_precision_anomaly"),
                "target_F1_anomaly": diag.get("target_F1_anomaly"),
                "beats_AUROC_target": diag.get("beats_AUROC_target"),
                "can_meet_precision_target": diag.get("can_meet_precision_target"),
                "can_meet_recall_target": diag.get("can_meet_recall_target"),
                "can_meet_all_classification_targets": diag.get("can_meet_all_classification_targets"),
                "can_exceed_all_targets": diag.get("can_exceed_all_targets"),
            }
            for prefix in [
                "best_acc",
                "best_F1",
                "best_precision_at_target_recall",
                "best_recall_at_target_precision",
                "best_F1_at_target_precision",
                "best_all_classification_targets",
            ]:
                slim = diag.get(prefix) or {}
                for key, value in slim.items():
                    row[f"{prefix}_{key}"] = value
            target_rows.append(row)
    if target_rows:
        fieldnames = []
        for row in target_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(target_diag_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(target_rows)

    gate_diag_path = output_dir / "mechanism_gate_diagnostics.csv"
    gate_rows = []
    for scope, category, data in [("overall", "ALL", summary["overall"])] + [
        ("category", category, data) for category, data in summary["per_category"].items()
    ]:
        if not data:
            continue
        for gate_name, diag in data.get("gate_diagnostics", {}).items():
            gate_rows.append(
                {
                    "scope": scope,
                    "category": category,
                    "gate": gate_name,
                    **diag,
                }
            )
    if gate_rows:
        fieldnames = []
        for row in gate_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(gate_diag_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(gate_rows)

    source_diag_path = output_dir / "mechanism_source_reliability.csv"
    source_rows = []
    for scope, category, data in [("overall", "ALL", summary["overall"])] + [
        ("category", category, data) for category, data in summary["per_category"].items()
    ]:
        if not data:
            continue
        reliability = data.get("source_reliability", {})
        aurocs = reliability.get("AUROC_by_source", {})
        deltas = reliability.get("AUROC_delta_vs_final", {})
        for source_name, auroc in aurocs.items():
            source_rows.append(
                {
                    "scope": scope,
                    "category": category,
                    "source": source_name,
                    "AUROC": auroc,
                    "AUROC_delta_vs_final": deltas.get(source_name),
                    "best_source_by_AUROC": reliability.get("best_source_by_AUROC"),
                }
            )
    if source_rows:
        fieldnames = []
        for row in source_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(source_diag_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(source_rows)

    print(f"[OK] Saved compact cases: {results_json}")
    if args.stream_jsonl:
        print(f"[OK] Saved compact stream JSONL: {stream_jsonl}")
    if args.save_full_cases:
        print(f"[OK] Saved full cases JSONL: {full_jsonl}")
    print(f"[OK] Saved summary: {output_dir / 'mechanism_summary.json'}")
    print(f"[OK] Saved summary CSV: {csv_path}")
    if target_rows:
        print(f"[OK] Saved target diagnostics CSV: {target_diag_path}")
    if gate_rows:
        print(f"[OK] Saved gate diagnostics CSV: {gate_diag_path}")
    if source_rows:
        print(f"[OK] Saved source reliability CSV: {source_diag_path}")
    if summary["overall"]:
        overall = summary["overall"]
        print(
            "[OVERALL] "
            f"metric_policy={overall.get('metric_threshold_policy')} "
            f"final_threshold={overall['final_metrics'].get('threshold')} "
            f"corrected_threshold={overall['corrected_metrics'].get('threshold')} "
            f"corrected_debiased_threshold={overall['corrected_debiased_metrics'].get('threshold')} "
            f"defect_aware_pool_threshold={overall['defect_aware_pool_metrics'].get('threshold')} "
            f"hdec_threshold={overall['hdec_metrics'].get('threshold')} "
            f"hdec_debiased_threshold={overall['hdec_debiased_metrics'].get('threshold')} "
            f"anchor_hdec_threshold={overall['anchor_hdec_metrics'].get('threshold')} "
            f"anchor_hdec_debiased_threshold={overall['anchor_hdec_debiased_metrics'].get('threshold')} "
            f"suppression_hdec_threshold={overall['suppression_hdec_metrics'].get('threshold')} "
            f"suppression_pool_hdec_threshold={overall['suppression_pool_hdec_metrics'].get('threshold')} "
            f"generated_acc={overall['generated_metrics']['acc']} "
            f"generation_deco_acc={overall['generation_deco_metrics']['acc']} "
            f"defect_aware_pool_acc={overall['defect_aware_pool_metrics']['acc']} "
            f"hdec_acc={overall['hdec_metrics']['acc']} "
            f"hdec_debiased_acc={overall['hdec_debiased_metrics']['acc']} "
            f"anchor_hdec_acc={overall['anchor_hdec_metrics']['acc']} "
            f"anchor_hdec_debiased_acc={overall['anchor_hdec_debiased_metrics']['acc']} "
            f"suppression_hdec_acc={overall['suppression_hdec_metrics']['acc']} "
            f"suppression_pool_hdec_acc={overall['suppression_pool_hdec_metrics']['acc']} "
            f"text_prior_AUROC={overall['text_prior_metrics']['AUROC']} "
            f"final_AUROC={overall['final_metrics']['AUROC']} "
            f"corrected_AUROC={overall['corrected_metrics']['AUROC']} "
            f"final_debiased_AUROC={overall['final_debiased_metrics']['AUROC']} "
            f"corrected_debiased_AUROC={overall['corrected_debiased_metrics']['AUROC']} "
            f"defect_aware_pool_AUROC={overall['defect_aware_pool_metrics']['AUROC']} "
            f"hdec_AUROC={overall['hdec_metrics']['AUROC']} "
            f"hdec_debiased_AUROC={overall['hdec_debiased_metrics']['AUROC']} "
            f"anchor_hdec_AUROC={overall['anchor_hdec_metrics']['AUROC']} "
            f"anchor_hdec_debiased_AUROC={overall['anchor_hdec_debiased_metrics']['AUROC']} "
            f"suppression_hdec_AUROC={overall['suppression_hdec_metrics']['AUROC']} "
            f"suppression_pool_hdec_AUROC={overall['suppression_pool_hdec_metrics']['AUROC']} "
            f"LCR={overall['mechanism_metrics']['LCR_layer_conflict_rate']} "
            f"DOR={overall['mechanism_metrics']['DOR_defect_omission_rate_on_anomaly']} "
            f"DHR={overall['mechanism_metrics']['DHR_defect_hallucination_rate_on_normal']} "
            f"CSR={overall['mechanism_metrics']['CSR_correction_success_rate']} "
            f"debiased_CSR={overall['mechanism_metrics']['debiased_CSR_correction_success_rate']} "
            f"HDEC_CSR={overall['mechanism_metrics']['HDEC_CSR_correction_success_rate']} "
            f"HDEC_debiased_CSR={overall['mechanism_metrics']['HDEC_debiased_CSR_correction_success_rate']} "
            f"Anchor_HDEC_CSR={overall['mechanism_metrics']['Anchor_HDEC_CSR_correction_success_rate']} "
            f"Anchor_HDEC_debiased_CSR={overall['mechanism_metrics']['Anchor_HDEC_debiased_CSR_correction_success_rate']} "
            f"Suppression_HDEC_CSR={overall['mechanism_metrics']['Suppression_HDEC_CSR_correction_success_rate']} "
            f"Suppression_pool_HDEC_CSR={overall['mechanism_metrics']['Suppression_pool_HDEC_CSR_correction_success_rate']} "
            f"HDEC_mean_gate={overall['mechanism_metrics']['HDEC_mean_gate']} "
            f"Suppression_HDEC_mean_gate={overall['mechanism_metrics']['Suppression_HDEC_mean_gate']} "
            f"Suppression_pool_HDEC_mean_gate={overall['mechanism_metrics']['Suppression_pool_HDEC_mean_gate']}"
        )


if __name__ == "__main__":
    main()
