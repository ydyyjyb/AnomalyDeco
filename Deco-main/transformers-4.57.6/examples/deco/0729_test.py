#!/usr/bin/env python3
"""Standalone Qwen3-VL DeCo-style evidence calibration for VisA.

This script intentionally does not import 0727_UP.py or 0728_Fight.py.

Main idea:
  image evidence score =
      defect_evidence - lambda * normal_evidence - eta * nuisance_evidence

  calibrated score =
      image evidence score - beta * text-only prior score

  HDEC score =
      gate * preceding-layer(anchor) score + (1 - gate) * final-layer score
      - beta * matched text-only prior

It can optionally calibrate thresholds with train-normal images, which is a
training-free normal distribution calibration step often used in industrial
anomaly detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None


GENERIC_DEFECT_LABELS = [
    "defect",
    "anomaly",
    "abnormal",
    "defective",
    "damage",
    "damaged",
    "scratch",
    "crack",
    "stain",
    "hole",
    "missing",
    "broken",
    "melt",
    "burnt",
    "contamination",
    "bubble",
    "leak",
    "discoloration",
    "deformation",
    "bent",
]

NORMAL_LABELS = [
    "normal",
    "good",
    "intact",
    "complete",
    "clean",
    "regular",
    "undamaged",
    "acceptable",
]

NUISANCE_LABELS = [
    "background",
    "shadow",
    "reflection",
    "lighting",
    "glare",
    "texture",
    "fabric",
    "viewpoint",
    "orientation",
    "position",
    "pose",
    "noise",
    "blur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone DeCo/HDEC evidence calibration for Qwen3-VL on VisA."
    )
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--visa-split-csv", default="")
    parser.add_argument("--visa-split-set", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--categories", default="")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=409600)
    parser.add_argument("--max-pixels", type=int, default=12582912)

    parser.add_argument("--defect-label-json", default="")
    parser.add_argument("--prompt-defect-limit", type=int, default=8)
    parser.add_argument("--max-defect-labels", type=int, default=24)
    parser.add_argument("--score-suffix", default="The visual condition is ")
    parser.add_argument("--trend-layers", default="0:34")
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--anchor-top-k", type=int, default=3)

    parser.add_argument("--lambda-normal", type=float, default=1.0)
    parser.add_argument("--eta-nuisance", type=float, default=0.35)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--group-aggregation", default="max", choices=["max", "mean", "logsumexp"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hdec-gamma", type=float, default=8.0)
    parser.add_argument("--hdec-tau", type=float, default=0.0)

    parser.add_argument(
        "--threshold-policies",
        default="zero,normal_q95,youden",
        help="Comma-separated: zero, fixed, normal_q90, normal_q95, normal_q97, normal_q99, youden.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.0)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument(
        "--calibration-normal-limit",
        type=int,
        default=100,
        help="Normal calibration images per category. Use -1 for all train normal images.",
    )

    parser.add_argument("--generate-explanations", action="store_true")
    parser.add_argument("--explain-errors-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--explanation-method", default="hdec_calibrated")
    parser.add_argument("--explanation-policy", default="normal_q95")
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save full prompt text in each JSONL case. Disabled by default for full-dataset runs.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip train-normal calibration even if normal_q policies are requested.",
    )
    parser.add_argument(
        "--partial-csv-interval",
        type=int,
        default=200,
        help="Write 0729_case_scores.partial.csv every N test samples. Use 0 to disable.",
    )

    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def parse_csv_words(text: str) -> List[str]:
    seen = set()
    out = []
    for chunk in str(text or "").split(","):
        item = chunk.strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def parse_range(text: str) -> List[int]:
    text = str(text).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def display_category(category: str) -> str:
    return str(category or "object").replace("_", " ").strip() or "object"


def normalize_label_items(items: Iterable[Any]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            nested = normalize_label_items(item)
        else:
            nested = re.split(r"[,;/|]", str(item))
        for label in nested:
            clean = str(label).strip().strip('"').strip("'")
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                out.append(clean)
    return out


def load_custom_defect_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for category, labels in data.items():
            out[str(category).strip()] = normalize_label_items(labels if isinstance(labels, list) else [labels])
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or item.get("name") or "").strip()
            labels = item.get("labels") or item.get("defects") or item.get("defect_labels") or []
            if category:
                out[category] = normalize_label_items(labels if isinstance(labels, list) else [labels])
    return {k: v for k, v in out.items() if v}


def load_image_anno_types(visa_root: Path) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    by_image: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    by_category: Dict[str, List[str]] = defaultdict(list)
    seen_by_category: Dict[str, set] = defaultdict(set)
    for anno_path in sorted(visa_root.glob("*/image_anno.csv")):
        category = anno_path.parent.name
        with anno_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                rel_image = row[0].strip().replace("\\", "/")
                labels = normalize_label_items([row[1]])
                by_image[category][rel_image] = labels
                for label in labels:
                    key = label.lower()
                    if key not in seen_by_category[category]:
                        seen_by_category[category].add(key)
                        by_category[category].append(label)
    return dict(by_image), dict(by_category)


def load_split_samples(
    visa_root: Path,
    split_csv: Path,
    split_set: str,
    categories: Optional[set],
    run_all: bool,
    sample_normal: int,
    sample_anomaly: int,
    seed: int,
    anno_by_image: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            category = row[0].strip()
            split = row[1].strip().lower()
            label_text = row[2].strip().lower()
            rel_image = row[3].strip().replace("\\", "/")
            rel_mask = row[4].strip().replace("\\", "/") if len(row) > 4 else ""
            if split != split_set.lower():
                continue
            if categories and category not in categories:
                continue
            image_path = visa_root / rel_image
            if not image_path.exists():
                print(f"[WARN] missing image: {image_path}")
                continue
            label = 0 if label_text == "normal" else 1
            mask_path = visa_root / rel_mask if rel_mask else Path("")
            samples.append(
                {
                    "category": category,
                    "split": split,
                    "image": str(image_path),
                    "image_id": Path(rel_image).stem,
                    "split_image": rel_image,
                    "label": label,
                    "label_name": "anomaly" if label else "normal",
                    "mask": str(mask_path) if rel_mask and mask_path.exists() else "",
                    "anomaly_types": anno_by_image.get(category, {}).get(rel_image, []),
                }
            )
    if run_all:
        return samples

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[(sample["category"], sample["label"])].append(sample)
    selected: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        k = sample_anomaly if key[1] == 1 else sample_normal
        if k is None or k < 0 or len(bucket) <= k:
            selected.extend(bucket)
        else:
            selected.extend(rng.sample(bucket, k))
    return selected


def build_defect_bank(
    category: str,
    custom_labels: Dict[str, List[str]],
    anno_labels: Dict[str, List[str]],
    max_labels: int,
) -> List[str]:
    labels: List[str] = []
    if category in custom_labels:
        labels.extend(custom_labels[category])
    else:
        labels.extend(anno_labels.get(category, []))
    labels.extend(GENERIC_DEFECT_LABELS)
    return normalize_label_items(labels)[:max_labels]


def build_instruction(category: str, defect_bank: Sequence[str], prompt_defect_limit: int) -> str:
    category_text = display_category(category)
    defects = list(defect_bank)[: max(0, prompt_defect_limit)]
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
            "Ignore viewpoint, pose, position, orientation, lighting, background texture, shadows, glare, and image noise unless they clearly cause a physical defect.",
            "A minor mark is anomalous only if it is a clear physical defect on the main subject.",
            "End with exactly one line: Answer: Normal or Answer: Anomaly.",
        ]
    )
    return "\n".join(lines)


def build_messages(category: str, defect_bank: Sequence[str], prompt_defect_limit: int, image: bool) -> List[Dict[str, Any]]:
    instruction = build_instruction(category, defect_bank, prompt_defect_limit)
    content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    if image:
        content.extend([{"type": "text", "text": "Query image:"}, {"type": "image"}])
    return [{"role": "user", "content": content}]


def nested_attr(obj: Any, dotted: str) -> Any:
    current = obj
    for part in dotted.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
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
        module = nested_attr(model, path)
        if module is not None:
            return module
    return torch.nn.Identity().to(model.device)


def first_token_ids(tokenizer: Any, labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label in labels:
        clean = str(label).strip()
        if not clean:
            continue
        variants = [clean]
        if not clean.startswith(" "):
            variants.append(" " + clean)
        for variant in variants:
            encoded = tokenizer(variant, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if ids:
                out.setdefault(clean, int(ids[0]))
    return out


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


def aggregate(values: torch.Tensor, mode: str) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=torch.float32)
    values = values.float()
    if mode == "mean":
        return values.mean()
    if mode == "logsumexp":
        return torch.logsumexp(values, dim=0)
    return torch.max(values, dim=0).values


def group_value(
    token_values: torch.Tensor,
    token_ids: List[int],
    group_ids: List[int],
    mode: str,
) -> torch.Tensor:
    pos = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
    idxs = [pos[int(token_id)] for token_id in group_ids if int(token_id) in pos]
    if not idxs:
        return torch.tensor(0.0, device=token_values.device, dtype=torch.float32)
    return aggregate(token_values[idxs], mode)


def evidence_score_from_logits(
    token_logits: torch.Tensor,
    token_ids: List[int],
    defect_ids: List[int],
    normal_ids: List[int],
    nuisance_ids: List[int],
    aggregation: str,
    temperature: float,
    lambda_normal: float,
    eta_nuisance: float,
) -> Dict[str, float]:
    temp = max(float(temperature), 1e-6)
    probs = torch.softmax(token_logits.float() / temp, dim=0)
    defect = group_value(probs, token_ids, defect_ids, aggregation)
    normal = group_value(probs, token_ids, normal_ids, aggregation)
    nuisance = group_value(probs, token_ids, nuisance_ids, aggregation)
    score = defect - float(lambda_normal) * normal - float(eta_nuisance) * nuisance
    return {
        "score": float(score.detach().cpu()),
        "defect": float(defect.detach().cpu()),
        "normal": float(normal.detach().cpu()),
        "nuisance": float(nuisance.detach().cpu()),
    }


def prepare_category_token_bank(
    tokenizer: Any,
    category: str,
    defect_bank: Sequence[str],
) -> Dict[str, Any]:
    defect_map = first_token_ids(tokenizer, defect_bank)
    normal_map = first_token_ids(tokenizer, NORMAL_LABELS)
    nuisance_map = first_token_ids(tokenizer, NUISANCE_LABELS)
    defect_ids = sorted(set(defect_map.values()))
    normal_ids = sorted(set(normal_map.values()))
    nuisance_ids = sorted(set(nuisance_map.values()))
    token_ids = sorted(set(defect_ids + normal_ids + nuisance_ids))
    if not defect_ids or not normal_ids or not token_ids:
        raise RuntimeError(f"Failed to build token bank for category={category}")
    return {
        "defect_labels": list(defect_bank),
        "normal_labels": list(NORMAL_LABELS),
        "nuisance_labels": list(NUISANCE_LABELS),
        "defect_token_map": defect_map,
        "normal_token_map": normal_map,
        "nuisance_token_map": nuisance_map,
        "defect_ids": defect_ids,
        "normal_ids": normal_ids,
        "nuisance_ids": nuisance_ids,
        "token_ids": token_ids,
    }


def score_model_outputs(
    model: Any,
    outputs: Any,
    token_bank: Dict[str, Any],
    trend_layers: List[int],
    deco_layers: List[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    token_ids = token_bank["token_ids"]
    defect_ids = token_bank["defect_ids"]
    normal_ids = token_bank["normal_ids"]
    nuisance_ids = token_bank["nuisance_ids"]

    final_logits = outputs.logits[0, -1].float()[token_ids]
    final_parts = evidence_score_from_logits(
        final_logits,
        token_ids,
        defect_ids,
        normal_ids,
        nuisance_ids,
        args.group_aggregation,
        args.temperature,
        args.lambda_normal,
        args.eta_nuisance,
    )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output has no hidden_states.")
    norm = get_final_norm_module(model)

    layer_rows: List[Dict[str, Any]] = []
    for layer_idx in trend_layers:
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            continue
        hidden = norm(hidden_states[layer_idx][0, -1])
        layer_logits = candidate_logits_from_hidden(model, hidden, token_ids)
        parts = evidence_score_from_logits(
            layer_logits,
            token_ids,
            defect_ids,
            normal_ids,
            nuisance_ids,
            args.group_aggregation,
            args.temperature,
            args.lambda_normal,
            args.eta_nuisance,
        )
        parts["layer"] = int(layer_idx)
        layer_rows.append(parts)

    deco_rows = [row for row in layer_rows if int(row["layer"]) in set(deco_layers)]
    selected = sorted(deco_rows, key=lambda row: float(row["score"]), reverse=True)[: max(1, args.anchor_top_k)]
    if selected:
        anchor_raw = sum(float(row["score"]) for row in selected) / len(selected)
        anchor_defect = sum(float(row["defect"]) for row in selected) / len(selected)
        anchor_normal = sum(float(row["normal"]) for row in selected) / len(selected)
        anchor_nuisance = sum(float(row["nuisance"]) for row in selected) / len(selected)
        selected_layers = [int(row["layer"]) for row in selected]
    else:
        anchor_raw = final_parts["score"]
        anchor_defect = final_parts["defect"]
        anchor_normal = final_parts["normal"]
        anchor_nuisance = final_parts["nuisance"]
        selected_layers = []

    gate_arg = float(args.hdec_gamma) * (anchor_raw - float(args.hdec_tau))
    gate = 1.0 / (1.0 + math.exp(-max(min(gate_arg, 60.0), -60.0)))
    hdec_raw = gate * anchor_raw + (1.0 - gate) * float(final_parts["score"])

    return {
        "final_raw": float(final_parts["score"]),
        "final_defect": float(final_parts["defect"]),
        "final_normal": float(final_parts["normal"]),
        "final_nuisance": float(final_parts["nuisance"]),
        "anchor_raw": float(anchor_raw),
        "anchor_defect": float(anchor_defect),
        "anchor_normal": float(anchor_normal),
        "anchor_nuisance": float(anchor_nuisance),
        "hdec_raw": float(hdec_raw),
        "hdec_gate": float(gate),
        "hdec_gate_arg": float(gate_arg),
        "selected_layers": selected_layers,
        "layer_rows": layer_rows,
    }


def load_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
        raise ImportError(
            "transformers with Qwen3VLForConditionalGeneration is not available. "
            "Run with PYTHONPATH pointing to transformers-4.57.6/src in the qwen3-vl environment."
        )
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    except Exception as exc:
        if not args.allow_attn_fallback or args.attn_implementation == "sdpa":
            raise
        print(f"[WARN] Failed to load with {args.attn_implementation}: {exc}")
        print("[WARN] Fallback to sdpa.")
        kwargs["attn_implementation"] = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model


def make_prompt_text(
    processor: Any,
    category: str,
    defect_bank: Sequence[str],
    prompt_defect_limit: int,
    image: bool,
) -> str:
    messages = build_messages(category, defect_bank, prompt_defect_limit, image=image)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def forward_score(
    processor: Any,
    model: Any,
    sample: Optional[Dict[str, Any]],
    category: str,
    defect_bank: Sequence[str],
    token_bank: Dict[str, Any],
    trend_layers: List[int],
    deco_layers: List[int],
    args: argparse.Namespace,
    image: bool,
) -> Dict[str, Any]:
    prompt_text = make_prompt_text(processor, category, defect_bank, args.prompt_defect_limit, image=image)
    score_prefix = prompt_text + args.score_suffix
    if image:
        assert sample is not None
        pil_image = Image.open(sample["image"]).convert("RGB")
        inputs = processor(
            text=[score_prefix],
            images=[pil_image],
            return_tensors="pt",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    else:
        inputs = processor(text=[score_prefix], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    scores = score_model_outputs(model, outputs, token_bank, trend_layers, deco_layers, args)
    scores["prompt_text"] = prompt_text
    scores["score_prefix"] = score_prefix
    return scores


def build_case_scores(
    raw_scores: Dict[str, Any],
    prior_scores: Dict[str, Any],
    beta_prior: float,
) -> Dict[str, float]:
    final = raw_scores["final_raw"] - beta_prior * prior_scores["final_raw"]
    anchor = raw_scores["anchor_raw"] - beta_prior * prior_scores["anchor_raw"]
    hdec_prior = raw_scores["hdec_gate"] * prior_scores["anchor_raw"] + (1.0 - raw_scores["hdec_gate"]) * prior_scores["final_raw"]
    hdec = raw_scores["hdec_raw"] - beta_prior * hdec_prior
    return {
        "final_calibrated": float(final),
        "anchor_top3_calibrated": float(anchor),
        "hdec_calibrated": float(hdec),
        "hdec_raw_no_prior": float(raw_scores["hdec_raw"]),
    }


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def confusion_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float, source: str) -> Dict[str, Any]:
    preds = [1 if float(s) >= float(threshold) else 0 for s in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = len(labels)

    def div(a: float, b: float) -> float:
        return a / b if b else 0.0

    recall_anomaly = div(tp, tp + fn)
    recall_normal = div(tn, tn + fp)
    precision_anomaly = div(tp, tp + fp)
    precision_normal = div(tn, tn + fn)
    f1_anomaly = div(2 * precision_anomaly * recall_anomaly, precision_anomaly + recall_anomaly)
    f1_normal = div(2 * precision_normal * recall_normal, precision_normal + recall_normal)
    return {
        "total": total,
        "acc": div(tp + tn, total),
        "AUROC": auc_score(labels, scores),
        "threshold": threshold,
        "score_source": source,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": (recall_normal + recall_anomaly) / 2.0,
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": (precision_normal + precision_anomaly) / 2.0,
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": (f1_normal + f1_anomaly) / 2.0,
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


def youden_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    unique_scores = sorted({float(s) for s in scores if math.isfinite(float(s))})
    if not unique_scores:
        return 0.0
    candidates = unique_scores[:]
    best_t = candidates[0]
    best_j = -1e9
    for t in candidates:
        m = confusion_metrics(labels, scores, t, "tmp")
        j = float(m["recall_anomaly"]) + float(m["recall_normal"]) - 1.0
        if j > best_j:
            best_j = j
            best_t = t
    return best_t


def thresholds_for_policy(
    policy: str,
    method: str,
    cases: List[Dict[str, Any]],
    scores: List[float],
    calibration_scores: Dict[str, Dict[str, List[float]]],
    fixed_threshold: float,
) -> Dict[str, float]:
    if policy == "zero":
        return {"__global__": 0.0}
    if policy == "fixed":
        return {"__global__": float(fixed_threshold)}
    if policy == "youden":
        labels = [int(case["label"]) for case in cases]
        return {"__global__": youden_threshold(labels, scores)}
    if policy.startswith("normal_q"):
        q = float(policy.replace("normal_q", "")) / 100.0
        thresholds: Dict[str, float] = {}
        global_normals: List[float] = []
        for category, by_method in calibration_scores.items():
            vals = by_method.get(method, [])
            if vals:
                thresholds[category] = percentile(vals, q)
                global_normals.extend(vals)
        thresholds["__global__"] = percentile(global_normals, q) if global_normals else 0.0
        return thresholds
    raise ValueError(f"Unknown threshold policy: {policy}")


def evaluate_policy(
    cases: List[Dict[str, Any]],
    method: str,
    scores: List[float],
    policy: str,
    thresholds: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labels = [int(case["label"]) for case in cases]
    global_threshold = thresholds.get("__global__", 0.0)
    rows: List[Dict[str, Any]] = []

    if policy.startswith("normal_q"):
        preds = []
        for case, score in zip(cases, scores):
            t = thresholds.get(case["category"], global_threshold)
            preds.append(1 if score >= t else 0)
        # Reconstruct equivalent metrics for category-specific thresholds.
        tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
        tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
        fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

        def div(a: float, b: float) -> float:
            return a / b if b else 0.0

        ra = div(tp, tp + fn)
        rn = div(tn, tn + fp)
        pa = div(tp, tp + fp)
        pn = div(tn, tn + fn)
        f1a = div(2 * pa * ra, pa + ra)
        f1n = div(2 * pn * rn, pn + rn)
        overall = {
            "total": len(labels),
            "acc": div(tp + tn, len(labels)),
            "AUROC": auc_score(labels, scores),
            "threshold": global_threshold,
            "score_source": method,
            "recall_normal": rn,
            "recall_anomaly": ra,
            "recall_all": (rn + ra) / 2.0,
            "precision_normal": pn,
            "precision_anomaly": pa,
            "precision_all": (pn + pa) / 2.0,
            "F1_normal": f1n,
            "F1_anomaly": f1a,
            "F1_all": (f1n + f1a) / 2.0,
            "recall_micro": div(tp + tn, len(labels)),
            "precision_micro": div(tp + tn, len(labels)),
            "F1_micro": div(tp + tn, len(labels)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pred_anomaly": tp + fp,
            "pred_normal": tn + fn,
            "anomaly_total": tp + fn,
            "normal_total": tn + fp,
        }
    else:
        overall = confusion_metrics(labels, scores, global_threshold, method)
    rows.append({"scope": "overall", "category": "ALL", "method": method, "threshold_policy": policy, **overall})

    category_metrics = []
    for category in sorted({case["category"] for case in cases}):
        idxs = [i for i, case in enumerate(cases) if case["category"] == category]
        category_labels = [labels[i] for i in idxs]
        category_scores = [scores[i] for i in idxs]
        threshold = thresholds.get(category, global_threshold)
        metric = confusion_metrics(category_labels, category_scores, threshold, method)
        category_metrics.append(metric)
        rows.append({"scope": "category", "category": category, "method": method, "threshold_policy": policy, **metric})

    macro = {"total": len(category_metrics)}
    for key in [
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
    ]:
        vals = [m[key] for m in category_metrics if m.get(key) is not None]
        macro[key] = sum(vals) / len(vals) if vals else None
    rows.append({"scope": "macro", "category": "ALL", "method": method, "threshold_policy": policy, **macro})
    return rows, macro


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_score_driven_explanation(
    processor: Any,
    model: Any,
    sample: Dict[str, Any],
    pred: int,
    score: float,
    threshold: float,
    defect_bank: Sequence[str],
    args: argparse.Namespace,
) -> str:
    label_text = "Anomaly" if pred == 1 else "Normal"
    category_text = display_category(sample["category"])
    prompt = (
        f"The scoring module predicts: {label_text}.\n"
        f"Category: {category_text}.\n"
        f"Score: {score:.6f}; threshold: {threshold:.6f}.\n"
        f"Candidate defect terms: {', '.join(list(defect_bank)[:args.prompt_defect_limit])}.\n"
        "Explain this decision using only visible evidence. Do not change the predicted label.\n"
        "If the visual evidence is weak, say it is weak. Output one short paragraph."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(sample["image"]).convert("RGB")
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
    gen_ids = output_ids[:, input_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "0729_cases.jsonl",
        "0729_case_scores.csv",
        "0729_case_scores.partial.csv",
        "0729_summary.csv",
        "0729_macro_summary.csv",
        "0729_per_category_summary.csv",
        "0729_best_by_metric.json",
        "0729_thresholds.json",
        "0729_error_audit.csv",
        "0729_score_driven_explanations.csv",
    ):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    visa_root = Path(args.visa_root)
    split_csv = Path(args.visa_split_csv) if args.visa_split_csv else visa_root / "split_csv" / "1cls.csv"
    categories = set(parse_csv_words(args.categories)) if args.categories else None
    trend_layers = parse_range(args.trend_layers)
    deco_layers = parse_range(args.deco_layers)
    threshold_policies = parse_csv_words(args.threshold_policies)
    needs_calibration = (
        not args.skip_calibration
        and any(policy.startswith("normal_q") for policy in threshold_policies)
    )

    anno_by_image, anno_by_category = load_image_anno_types(visa_root)
    defect_json_path = Path(args.defect_label_json) if args.defect_label_json else visa_root / "category_defect_labels.json"
    custom_labels = load_custom_defect_json(defect_json_path)
    if custom_labels:
        print(f"[INFO] Loaded custom category defect labels: {defect_json_path}")
    else:
        print(f"[INFO] No custom defect label json loaded: {defect_json_path}")

    test_samples = load_split_samples(
        visa_root,
        split_csv,
        args.visa_split_set,
        categories,
        args.run_all,
        args.sample_normal,
        args.sample_anomaly,
        args.seed,
        anno_by_image,
    )
    if not test_samples:
        raise RuntimeError("No test samples loaded.")

    calibration_samples: List[Dict[str, Any]] = []
    if needs_calibration:
        calibration_samples = load_split_samples(
            visa_root,
            split_csv,
            args.calibration_split,
            categories,
            True,
            -1,
            -1,
            args.seed,
            anno_by_image,
        )
        calibration_samples = [s for s in calibration_samples if int(s["label"]) == 0]
        if args.calibration_normal_limit >= 0:
            rng = random.Random(args.seed)
            buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for sample in calibration_samples:
                buckets[sample["category"]].append(sample)
            limited = []
            for category, bucket in sorted(buckets.items()):
                if len(bucket) <= args.calibration_normal_limit:
                    limited.extend(bucket)
                else:
                    limited.extend(rng.sample(bucket, args.calibration_normal_limit))
            calibration_samples = limited

    print(f"[INFO] Test samples: {len(test_samples)}")
    if needs_calibration:
        print(f"[INFO] Train-normal calibration samples: {len(calibration_samples)}")
    else:
        print("[INFO] Train-normal calibration skipped.")
    print(f"[INFO] Methods: final_calibrated, anchor_top3_calibrated, hdec_calibrated, hdec_raw_no_prior")

    processor, model = load_model(args)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor has no tokenizer")

    all_categories = sorted({s["category"] for s in test_samples + calibration_samples})
    defect_banks = {
        category: build_defect_bank(category, custom_labels, anno_by_category, args.max_defect_labels)
        for category in all_categories
    }
    token_banks = {
        category: prepare_category_token_bank(tokenizer, category, defect_banks[category])
        for category in all_categories
    }

    (out_dir / "defect_bank_by_category.json").write_text(
        json.dumps(defect_banks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_samples": len(test_samples),
                "calibration_samples": len(calibration_samples),
                "needs_calibration": needs_calibration,
                "categories": all_categories,
                "methods": [
                    "final_calibrated",
                    "anchor_top3_calibrated",
                    "hdec_calibrated",
                    "hdec_raw_no_prior",
                ],
                "threshold_policies": threshold_policies,
                "trend_layers": trend_layers,
                "deco_layers": deco_layers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_prior_cache: Dict[str, Dict[str, Any]] = {}

    def get_text_prior(category: str) -> Dict[str, Any]:
        if category not in text_prior_cache:
            text_prior_cache[category] = forward_score(
                processor,
                model,
                None,
                category,
                defect_banks[category],
                token_banks[category],
                trend_layers,
                deco_layers,
                args,
                image=False,
            )
        return text_prior_cache[category]

    methods = ["final_calibrated", "anchor_top3_calibrated", "hdec_calibrated", "hdec_raw_no_prior"]
    calibration_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, sample in enumerate(calibration_samples, start=1):
        category = sample["category"]
        prior = get_text_prior(category)
        raw = forward_score(
            processor,
            model,
            sample,
            category,
            defect_banks[category],
            token_banks[category],
            trend_layers,
            deco_layers,
            args,
            image=True,
        )
        scores = build_case_scores(raw, prior, args.beta_prior)
        for method in methods:
            calibration_scores[category][method].append(float(scores[method]))
        if args.log_interval and idx % args.log_interval == 0:
            print(f"[CAL] {idx}/{len(calibration_samples)} latest={category}/{sample['image_id']}")

    case_rows: List[Dict[str, Any]] = []
    compact_cases: List[Dict[str, Any]] = []
    for idx, sample in enumerate(test_samples, start=1):
        category = sample["category"]
        prior = get_text_prior(category)
        raw = forward_score(
            processor,
            model,
            sample,
            category,
            defect_banks[category],
            token_banks[category],
            trend_layers,
            deco_layers,
            args,
            image=True,
        )
        scores = build_case_scores(raw, prior, args.beta_prior)
        case = {
            "category": category,
            "image_id": sample["image_id"],
            "image": sample["image"],
            "split_image": sample["split_image"],
            "label": int(sample["label"]),
            "label_name": sample["label_name"],
            "mask": sample["mask"],
            "anomaly_types": sample["anomaly_types"],
            "scores": scores,
            "raw_parts": {
                "final_raw": raw["final_raw"],
                "anchor_raw": raw["anchor_raw"],
                "hdec_raw": raw["hdec_raw"],
                "text_prior_final": prior["final_raw"],
                "text_prior_anchor": prior["anchor_raw"],
                "hdec_gate": raw["hdec_gate"],
                "hdec_gate_arg": raw["hdec_gate_arg"],
                "selected_layers": raw["selected_layers"],
                "final_defect": raw["final_defect"],
                "final_normal": raw["final_normal"],
                "final_nuisance": raw["final_nuisance"],
                "anchor_defect": raw["anchor_defect"],
                "anchor_normal": raw["anchor_normal"],
                "anchor_nuisance": raw["anchor_nuisance"],
            },
        }
        if args.save_prompts:
            case["prompt_text"] = raw["prompt_text"]
            case["score_prefix"] = raw["score_prefix"]
        compact_cases.append(case)
        case_rows.append(
            {
                "category": category,
                "image_id": sample["image_id"],
                "label": sample["label_name"],
                "anomaly_types": ",".join(sample["anomaly_types"]),
                "image": sample["image"],
                **{method: scores[method] for method in methods},
                "hdec_gate": raw["hdec_gate"],
                "selected_layers": ",".join(str(x) for x in raw["selected_layers"]),
            }
        )
        with (out_dir / "0729_cases.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
        if args.partial_csv_interval and idx % args.partial_csv_interval == 0:
            write_csv(out_dir / "0729_case_scores.partial.csv", case_rows)
        if args.log_interval and idx % args.log_interval == 0:
            print(f"[TEST] {idx}/{len(test_samples)} latest={category}/{sample['label_name']}/{sample['image_id']}")

    write_csv(out_dir / "0729_case_scores.csv", case_rows)

    summary_rows: List[Dict[str, Any]] = []
    thresholds_out: Dict[str, Any] = {}
    for method in methods:
        scores = [float(case["scores"][method]) for case in compact_cases]
        for policy in threshold_policies:
            thresholds = thresholds_for_policy(
                policy,
                method,
                compact_cases,
                scores,
                calibration_scores,
                args.fixed_threshold,
            )
            thresholds_out[f"{method}/{policy}"] = thresholds
            rows, _ = evaluate_policy(compact_cases, method, scores, policy, thresholds)
            summary_rows.extend(rows)
    write_csv(out_dir / "0729_summary.csv", summary_rows)
    macro_summary_rows = [row for row in summary_rows if row.get("scope") == "macro"]
    per_category_rows = [row for row in summary_rows if row.get("scope") == "category"]
    write_csv(out_dir / "0729_macro_summary.csv", macro_summary_rows)
    write_csv(out_dir / "0729_per_category_summary.csv", per_category_rows)
    best_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric in ("acc", "AUROC", "recall_all", "precision_all", "F1_all"):
        candidates = [
            row for row in macro_summary_rows
            if row.get(metric) is not None and math.isfinite(float(row.get(metric) or 0.0))
        ]
        if candidates:
            best_by_metric[metric] = max(candidates, key=lambda row: float(row[metric]))
    (out_dir / "0729_best_by_metric.json").write_text(
        json.dumps(best_by_metric, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "0729_thresholds.json").write_text(
        json.dumps(thresholds_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    error_rows: List[Dict[str, Any]] = []
    for method in methods:
        scores = [float(case["scores"][method]) for case in compact_cases]
        for policy in threshold_policies:
            thresholds = thresholds_out[f"{method}/{policy}"]
            global_t = thresholds.get("__global__", 0.0)
            for case, score in zip(compact_cases, scores):
                threshold = thresholds.get(case["category"], global_t)
                pred = 1 if score >= threshold else 0
                label = int(case["label"])
                if pred == label:
                    continue
                error_rows.append(
                    {
                        "method": method,
                        "threshold_policy": policy,
                        "error_type": "FP" if pred == 1 else "FN",
                        "category": case["category"],
                        "image_id": case["image_id"],
                        "pred": "anomaly" if pred else "normal",
                        "label": case["label_name"],
                        "score": score,
                        "threshold": threshold,
                        "score_minus_threshold": score - threshold,
                        "anomaly_types": ",".join(case["anomaly_types"]),
                        "image": case["image"],
                        "selected_layers": ",".join(str(x) for x in case["raw_parts"]["selected_layers"]),
                        "hdec_gate": case["raw_parts"]["hdec_gate"],
                    }
                )
    write_csv(out_dir / "0729_error_audit.csv", error_rows)

    if args.generate_explanations:
        method = args.explanation_method
        policy = args.explanation_policy
        if f"{method}/{policy}" not in thresholds_out:
            raise RuntimeError(f"Explanation method/policy not evaluated: {method}/{policy}")
        thresholds = thresholds_out[f"{method}/{policy}"]
        global_t = thresholds.get("__global__", 0.0)
        explain_rows = []
        for case in compact_cases:
            threshold = thresholds.get(case["category"], global_t)
            score = float(case["scores"][method])
            pred = 1 if score >= threshold else 0
            if args.explain_errors_only and pred == int(case["label"]):
                continue
            text = generate_score_driven_explanation(
                processor,
                model,
                case,
                pred,
                score,
                threshold,
                defect_banks[case["category"]],
                args,
            )
            explain_rows.append(
                {
                    "category": case["category"],
                    "image_id": case["image_id"],
                    "label": case["label_name"],
                    "pred": "anomaly" if pred else "normal",
                    "score": score,
                    "threshold": threshold,
                    "image": case["image"],
                    "anomaly_types": ",".join(case["anomaly_types"]),
                    "explanation": text,
                }
            )
            print(f"[EXPLAIN] {len(explain_rows)} {case['category']}/{case['image_id']}")
        write_csv(out_dir / "0729_score_driven_explanations.csv", explain_rows)

    # Print a compact macro view for quick terminal inspection.
    macro_rows = [r for r in summary_rows if r.get("scope") == "macro"]
    print("\n===== Macro Summary =====")
    for row in macro_rows:
        print(
            f"{row['method']:24s} {row['threshold_policy']:10s} "
            f"ACC={float(row['acc']):.4f} AUROC={float(row['AUROC'] or 0):.4f} "
            f"R={float(row['recall_all']):.4f} P={float(row['precision_all']):.4f} F1={float(row['F1_all']):.4f}"
        )
    print(f"[OK] Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
