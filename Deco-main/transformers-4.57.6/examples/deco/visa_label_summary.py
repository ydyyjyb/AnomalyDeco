#!/usr/bin/env python
# coding: utf-8
"""
VisA layer-wise probability + DeCo decision trace plugin.

This is an ADD-ON script, not a replacement for eval_visa_qwen3_vl_thinking.py.

What it does:
1) Imports your existing eval_visa_qwen3_vl_thinking.py.
2) Reuses its build_messages() unchanged.
3) For each selected category, samples exactly:
   - one normal image
   - one anomaly image
4) For each image, saves:
   - two layer-wise probability plots: raw and anchor-normalized normal / anomaly / hedge
   - one detailed JSON record containing:
     a) selected category/image/GT
     b) anchor words and token ids
     c) layer-wise normal/anomaly/hedge probabilities
     d) DeCo candidate tokens
     e) DeCo selected anchor layer
     f) DeCo weighting formula and effective weight
     g) generated output and normalized prediction

Put this file at:
  Deco-main/transformers-4.57.6/examples/deco/probe_visa_layer_probs.py
"""

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


DEFAULT_ANCHOR_DEFECT_WORDS = "defect,anomaly,abnormal"
DEFAULT_ANCHOR_NORM_WORDS = "normal,uniform,undamaged,no defect,good"
DEFAULT_ANCHOR_HEDGE_WORDS = "maybe,seems,likely,appears,possibly,could,might"


def parse_csv(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def unique_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def safe_name(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "na"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VisA layer-wise probability and DeCo decision trace plugin.")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen3_vl", "llava"])
    parser.add_argument("--model_base", type=str, default=None, help="Optional base model path for LLaVA LoRA/projector loading.")
    parser.add_argument("--vision_tower_path", type=str, default=None, help="Optional local CLIP vision tower path for offline LLaVA loading.")
    parser.add_argument("--conv_mode", type=str, default=None, help="LLaVA conversation template override, e.g. llava_v1.")

    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--fixed_support_images", type=str, default=None)

    # This does not modify build_messages(); it only appends after apply_chat_template.
    parser.add_argument("--assistant_prefix", type=str, default='{"label": "')

    parser.add_argument("--analysis_layer_start", type=int, default=1)
    parser.add_argument("--analysis_layer_end", type=int, default=None)
    parser.add_argument("--prob_aggregate", type=str, default="sum", choices=["sum", "max"])

    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--min_pixels", type=int, default=640 * 640)
    parser.add_argument("--max_pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow_attn_fallback", action="store_true")

    # Anchor lexicons.
    parser.add_argument("--anchor_defect_words", type=str, default=DEFAULT_ANCHOR_DEFECT_WORDS)
    parser.add_argument("--anchor_norm_words", type=str, default=DEFAULT_ANCHOR_NORM_WORDS)
    parser.add_argument("--anchor_hedge_words", type=str, default=DEFAULT_ANCHOR_HEDGE_WORDS)
    parser.add_argument("--expand_defect_with_category_labels", action="store_true", default=True)
    parser.add_argument("--no_expand_defect_with_category_labels", action="store_false", dest="expand_defect_with_category_labels")

    # DeCo.
    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.4)
    parser.add_argument("--threshold_top_k", type=int, default=32)
    parser.add_argument("--start_layer", type=int, default=4)
    parser.add_argument("--end_layer", type=int, default=11)
    parser.add_argument("--anchor_strategy", type=str, default="defect_aware", choices=["default", "defect_aware"])
    parser.add_argument("--lambda_norm", type=float, default=0.5)
    parser.add_argument("--eta_hedge", type=float, default=0.1)
    parser.add_argument("--enable_hedge", action="store_true")
    parser.add_argument("--use_logsumexp", action="store_true")
    parser.add_argument("--gamma", type=float, default=8.0)
    parser.add_argument("--deco_debug", action="store_true")

    # Robust anchor smoothing. Used to reconstruct/report DeCo selection if enabled in generation.
    parser.add_argument("--anchor_smooth", action="store_true")
    parser.add_argument("--anchor_remove_outliers", action="store_true")
    parser.add_argument("--anchor_hampel_window", type=int, default=2)
    parser.add_argument("--anchor_hampel_n_sigma", type=float, default=3.0)
    parser.add_argument("--anchor_smooth_window", type=int, default=3)
    parser.add_argument("--anchor_smooth_beta", type=float, default=0.7)

    return parser.parse_args()


def infer_model_family(args: argparse.Namespace) -> str:
    if args.model_family != "auto":
        return args.model_family
    model_name = str(args.model_name_or_path).lower()
    if "llava" in model_name:
        return "llava"
    return "qwen3_vl"


def ensure_llava_on_path() -> None:
    deco_main = Path(__file__).resolve().parents[3]
    if str(deco_main) not in sys.path:
        sys.path.insert(0, str(deco_main))


def load_model_bundle(args: argparse.Namespace, device: str, attn_impl: str) -> Dict[str, Any]:
    family = infer_model_family(args)

    if family == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("Qwen3VLForConditionalGeneration is not available from transformers.")
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        tokenizer = processor.tokenizer
        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto" if device == "cuda" else None,
            "attn_implementation": attn_impl,
            "trust_remote_code": True,
        }
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
        except Exception as e:
            if args.allow_attn_fallback and attn_impl == "flash_attention_2":
                print(f"[WARN] Failed to load with flash_attention_2: {e}")
                print("[WARN] Fallback to sdpa.")
                load_kwargs["attn_implementation"] = "sdpa"
                model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
            else:
                raise
        if device != "cuda":
            model.to(device)
        model.eval()
        return {
            "family": family,
            "processor": processor,
            "tokenizer": tokenizer,
            "model": model,
            "image_processor": None,
            "llava_modules": None,
        }

    ensure_llava_on_path()
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    llava_device = "cuda" if device == "cuda" else "cpu"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device=llava_device,
        vision_tower_path=args.vision_tower_path,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return {
        "family": family,
        "processor": None,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "llava_modules": {
            "model_name": model_name,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
        },
    }


def build_llava_prompt(model, model_name: str, conv_mode: Optional[str], messages: List[Dict[str, Any]], assistant_prefix: str) -> Tuple[str, str]:
    ensure_llava_on_path()
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.conversation import SeparatorStyle, conv_templates

    image_token_count = 0
    text_parts: List[str] = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif item.get("type") == "image":
                image_token_count += 1
                if getattr(model.config, "mm_use_im_start_end", False):
                    text_parts.append(DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
                else:
                    text_parts.append(DEFAULT_IMAGE_TOKEN)

    if image_token_count != 1:
        raise ValueError(
            f"LLaVA-v1.5 probing expects exactly one image token; got {image_token_count}. "
            "Please use SHOT=0 for the LLaVA probability probe."
        )

    qs = "\n".join(x for x in text_parts if x.strip())
    if conv_mode is None:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], assistant_prefix if assistant_prefix else None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def prepare_model_inputs(
    *,
    family: str,
    bundle: Dict[str, Any],
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    all_images: List[Image.Image],
) -> Tuple[Dict[str, Any], int, str]:
    model = bundle["model"]
    if family == "qwen3_vl":
        processor = bundle["processor"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if args.assistant_prefix:
            text = text + args.assistant_prefix
        inputs = processor(
            text=[text],
            images=all_images,
            return_tensors="pt",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, int(inputs["input_ids"].shape[-1]), ""

    ensure_llava_on_path()
    from llava.constants import IMAGE_TOKEN_INDEX

    llava_modules = bundle["llava_modules"]
    tokenizer = bundle["tokenizer"]
    prompt, stop_str = build_llava_prompt(
        model,
        llava_modules["model_name"],
        args.conv_mode,
        messages,
        args.assistant_prefix,
    )
    input_ids = llava_modules["tokenizer_image_token"](
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    images_tensor = llava_modules["process_images"](
        all_images,
        bundle["image_processor"],
        model.config,
    ).to(model.device, dtype=torch.float16)
    return {"input_ids": input_ids, "images": images_tensor}, int(input_ids.shape[-1]), stop_str


def load_eval_module() -> Any:
    eval_path = Path(__file__).with_name("eval_visa_qwen3_vl_thinking.py")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval script beside this plugin: {eval_path}")
    spec = importlib.util.spec_from_file_location("eval_visa_qwen3_vl_thinking", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load eval script: {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_selected_categories(args: argparse.Namespace, available: List[str]) -> List[str]:
    selected = []
    if args.category:
        selected.append(args.category)
    if args.categories:
        selected.extend(parse_csv(args.categories))
    selected = unique_keep_order(selected)
    if not selected:
        return available
    missing = [c for c in selected if c not in set(available)]
    if missing:
        raise ValueError(f"Requested categories not found: {missing}. Available: {available}")
    return selected


def choose_two_cases_per_category(samples: List[Dict[str, Any]], categories: List[str], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_cat: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for item in samples:
        cat = str(item["category"])
        if cat not in categories:
            continue
        gt = str(item.get("ground_truth_label", "good"))
        bucket = "good" if gt == "good" else "anomaly"
        by_cat.setdefault(cat, {"good": [], "anomaly": []})[bucket].append(item)

    selected: List[Dict[str, Any]] = []
    for cat in categories:
        good_items = sorted(by_cat.get(cat, {}).get("good", []), key=lambda x: str(x["image_path"]))
        anomaly_items = sorted(by_cat.get(cat, {}).get("anomaly", []), key=lambda x: str(x["image_path"]))

        if good_items:
            selected.append(dict(rng.choice(good_items), probe_kind="normal_probe"))
        else:
            print(f"[WARN] category={cat}: no normal image found.")

        if anomaly_items:
            selected.append(dict(rng.choice(anomaly_items), probe_kind="anomaly_probe"))
        else:
            print(f"[WARN] category={cat}: no anomaly image found.")

    return selected


def resolve_support_paths(
    *,
    eval_module: Any,
    dataset_root: Path,
    category: str,
    shot: int,
    category_meta: Dict[str, Any],
    fixed_support_map: Dict[str, List[str]],
    seed: int,
) -> List[str]:
    if shot <= 0:
        return []

    support_pool = [str(p) for p in category_meta.get(category, {}).get("support_good_paths", [])]
    support_name_to_path = {Path(p).name: str(p) for p in support_pool}

    if category in fixed_support_map:
        wanted_refs = fixed_support_map[category]
        if len(wanted_refs) < shot:
            raise ValueError(f"Category '{category}' requires shot={shot}, but only {len(wanted_refs)} fixed support image(s) provided.")
        resolved = []
        for ref in wanted_refs[:shot]:
            if ref in support_name_to_path:
                resolved.append(support_name_to_path[ref])
                continue
            p1 = dataset_root / ref
            if p1.exists():
                resolved.append(str(p1))
                continue
            p2 = Path(ref)
            if p2.exists():
                resolved.append(str(p2))
                continue
            raise FileNotFoundError(f"Fixed support image '{ref}' for category '{category}' not found.")
        return resolved

    rng = random.Random(seed + abs(hash(category)) % 100000)
    support_pool = [p for p in support_pool if Path(p).exists()]
    if not support_pool:
        return []
    if len(support_pool) >= shot:
        return rng.sample(support_pool, shot)
    return [rng.choice(support_pool) for _ in range(shot)]


def token_ids_for_words(tokenizer, words: List[str]) -> Tuple[List[int], List[Dict[str, Any]]]:
    ids: List[int] = []
    specs: List[Dict[str, Any]] = []

    for word in words:
        word = str(word).strip()
        if not word:
            continue

        variants = [word]
        if not word.startswith(" "):
            variants.append(" " + word)

        collected = []
        for v in variants:
            try:
                encoded = tokenizer.encode(v, add_special_tokens=False)
            except Exception:
                encoded = []
            if encoded:
                collected.append(int(encoded[0]))

        collected = sorted(set(collected))
        if not collected:
            continue

        ids.extend(collected)
        specs.append(
            {
                "word": word,
                "first_piece_token_ids": collected,
                "first_piece_tokens": [tokenizer.convert_ids_to_tokens(i) for i in collected],
            }
        )

    return sorted(set(ids)), specs


def aggregate_prob_tensor(probs: torch.Tensor, token_ids: List[int], mode: str) -> torch.Tensor:
    if not token_ids:
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
    idx = torch.tensor(token_ids, dtype=torch.long, device=probs.device)
    vals = probs[idx]
    if vals.numel() == 0:
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
    return vals.max() if mode == "max" else vals.sum()


def aggregate_prob_float(probs: torch.Tensor, token_ids: List[int], mode: str) -> float:
    return float(aggregate_prob_tensor(probs, token_ids, mode).item())


def _normalize_text_for_label_match(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_string_field_from_text(text: str, field_names: List[str]) -> str:
    """
    Extract simple JSON-like string fields from either a full JSON object or a JSON fragment.
    Works for fragments such as:
      small holes", "reason": "..."
    after the caller prepends {"label": ".
    """
    if not text:
        return ""
    for field in field_names:
        pattern = rf'(?is)["\']?{re.escape(field)}["\']?\s*[:=]\s*["\']([^"\']*)["\']'
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def _reason_indicates_anomaly(reason: str) -> bool:
    """
    Decide binary anomaly from the generated reason text.

    Conservative rule:
    - empty reason => not anomaly
    - normal/no-defect wording only => not anomaly
    - visible/evidence/defect wording, or any non-trivial reason after an abnormal label fragment => anomaly

    This is only used when JSON/label parsing would otherwise fall back to DEFAULT_PREDICTION=good.
    """
    r = _normalize_text_for_label_match(reason)
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
        "consistent with", "surface", "corner", "edge",
    ]
    if any(k in r for k in anomaly_keywords):
        return True

    # If the model bothered to provide a non-empty reason and it is not a normal-only statement,
    # treat it as anomalous for binary probing.
    return len(r.split()) >= 3


def _select_label_from_reason(reason: str, defect_types: List[str]) -> str:
    """
    Try to map reason content to one of the known defect labels.
    This is intentionally conservative and only returns a known defect type.
    """
    reason_norm = _normalize_text_for_label_match(reason)
    if not reason_norm:
        return ""

    # Direct multi-word defect mention.
    for defect in sorted(defect_types, key=lambda x: len(str(x)), reverse=True):
        d_norm = _normalize_text_for_label_match(defect)
        if d_norm and d_norm in reason_norm:
            return defect

    # Token-overlap fallback for labels like "different colour spot", "small holes".
    best_label = ""
    best_overlap = 0
    for defect in defect_types:
        d_norm = _normalize_text_for_label_match(defect)
        words = [w for w in d_norm.split() if len(w) >= 4]
        if not words:
            continue
        overlap = sum(1 for w in words if w in reason_norm)
        if overlap > best_overlap and overlap >= max(1, min(2, len(words))):
            best_overlap = overlap
            best_label = defect
    return best_label


def _find_known_label_in_text(text: str, defect_types: List[str]) -> str:
    """
    Match generated text against the known candidate labels.

    Why this exists:
    The probe often appends assistant_prefix='{"label": "' so raw_text can be a JSON fragment:
      small holes", "reason": "..."}
    eval_visa_qwen3_vl_thinking.py's strict parser expects complete JSON, so fragment outputs can
    incorrectly fall back to DEFAULT_PREDICTION=good. This function recovers known labels directly.
    """
    raw = str(text or "").strip()
    norm = _normalize_text_for_label_match(raw)

    labels = ["good"] + [str(x).strip() for x in defect_types if str(x).strip()]
    labels = sorted(set(labels), key=lambda x: len(x), reverse=True)

    for label in labels:
        label_norm = _normalize_text_for_label_match(label)
        if not label_norm:
            continue

        # Most important case: raw_text begins with the label value.
        # Example: small holes", "reason": ...
        if norm == label_norm or norm.startswith(label_norm + '"') or norm.startswith(label_norm + "'"):
            return label
        if norm.startswith(label_norm + ",") or norm.startswith(label_norm + "}") or norm.startswith(label_norm + "]"):
            return label
        if norm.startswith(label_norm + " "):
            # Avoid treating a label prefix as a hit when the next part is another label word.
            return label

    # Full/reconstructed JSON-like text may contain "label": "xxx".
    label_value = _extract_string_field_from_text(raw, ["label", "defect_type", "class", "prediction", "result"])
    if label_value:
        label_value_norm = _normalize_text_for_label_match(label_value)
        for label in labels:
            if label_value_norm == _normalize_text_for_label_match(label):
                return label

    # Conservative substring fallback for complete multi-word labels.
    # Do not use this for single ambiguous words unless they are exact labels.
    for label in labels:
        label_norm = _normalize_text_for_label_match(label)
        if " " in label_norm and label_norm in norm:
            return label

    return ""


def robust_parse_prediction_with_assistant_prefix(
    *,
    eval_module: Any,
    raw_text: str,
    assistant_prefix: str,
    defect_types: List[str],
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str, str, str]:
    """
    Return:
      prediction, parsed, parse_source, parse_input_source, raw_text_for_parse

    It first tries the original parser on reconstructed text.
    If that still falls back to default, it recovers known labels from the generated fragment itself.
    """
    parse_candidates: List[Tuple[str, str]] = []
    if assistant_prefix:
        parse_candidates.append(("assistant_prefix_plus_raw_text", assistant_prefix + raw_text))
    parse_candidates.append(("raw_text", raw_text))

    parsed = None
    parse_source = "none"
    parse_input_source = "none"
    raw_text_for_parse = raw_text

    for candidate_source, candidate_text in parse_candidates:
        candidate_parsed, candidate_parse_source = eval_module.parse_model_json(candidate_text)
        if candidate_parsed is not None:
            parsed = candidate_parsed
            parse_source = candidate_parse_source
            parse_input_source = candidate_source
            raw_text_for_parse = candidate_text
            break

    if parsed is None and assistant_prefix:
        parse_input_source = "assistant_prefix_plus_raw_text"
        raw_text_for_parse = assistant_prefix + raw_text

    prediction = eval_module.normalize_prediction(
        parsed=parsed,
        parse_source=parse_source,
        raw_text=raw_text_for_parse,
        defect_types=defect_types,
        image_size=image_size,
    )

    # If original normalize_prediction still fell back to DEFAULT_PREDICTION, recover a known label.
    if prediction.get("parse_source") == "default" or (
        prediction.get("label") == "good"
        and str(prediction.get("raw_label", "")).lower() == "good"
        and _find_known_label_in_text(raw_text, defect_types)
    ):
        recovered_label = ""
        recovered_from = "none"

        # Prefer raw fragment, because it starts exactly at label value when assistant_prefix is used.
        recovered_label = _find_known_label_in_text(raw_text, defect_types)
        if recovered_label:
            recovered_from = "generated_fragment_label_match"
        elif assistant_prefix:
            recovered_label = _find_known_label_in_text(assistant_prefix + raw_text, defect_types)
            if recovered_label:
                recovered_from = "reconstructed_text_label_match"

        if recovered_label:
            reconstructed = assistant_prefix + raw_text if assistant_prefix else raw_text
            recovered_parsed = {
                "label": recovered_label,
                "reason": _extract_string_field_from_text(reconstructed, ["reason", "why", "evidence"]),
                "root_cause": _extract_string_field_from_text(reconstructed, ["root_cause", "cause"]),
                "improvement": _extract_string_field_from_text(reconstructed, ["improvement", "action"]),
            }
            parsed = recovered_parsed
            parse_source = recovered_from
            parse_input_source = recovered_from
            raw_text_for_parse = reconstructed

            prediction = eval_module.normalize_prediction(
                parsed=recovered_parsed,
                parse_source=recovered_from,
                raw_text=raw_text_for_parse,
                defect_types=defect_types,
                image_size=image_size,
            )

    # Final binary fallback: if label parsing still falls back to good, inspect reason.
    # Example:
    #   raw_text = small holes", "reason": "Two small dark spots are visible ..."
    # If reason is non-empty and abnormal, treat binary prediction as anomaly.
    if prediction.get("label") == "good" and prediction.get("parse_source") == "default":
        reconstructed = assistant_prefix + raw_text if assistant_prefix else raw_text
        reason_text = _extract_string_field_from_text(reconstructed, ["reason", "why", "evidence"])
        if _reason_indicates_anomaly(reason_text):
            reason_label = _select_label_from_reason(reason_text, defect_types)
            if not reason_label:
                reason_label = _find_known_label_in_text(raw_text, defect_types)
            if not reason_label:
                reason_label = "anomaly"

            parsed = {
                "label": reason_label,
                "reason": reason_text,
                "root_cause": _extract_string_field_from_text(reconstructed, ["root_cause", "cause"]),
                "improvement": _extract_string_field_from_text(reconstructed, ["improvement", "action"]),
            }
            parse_source = "reason_indicates_anomaly"
            parse_input_source = "reason_field_binary_fallback"
            raw_text_for_parse = reconstructed

            if reason_label in defect_types:
                prediction = eval_module.normalize_prediction(
                    parsed=parsed,
                    parse_source=parse_source,
                    raw_text=raw_text_for_parse,
                    defect_types=defect_types,
                    image_size=image_size,
                )
            else:
                # Keep binary anomaly even when subtype cannot be recovered as a known defect type.
                prediction = {
                    "label": "anomaly",
                    "bbox": [],
                    "reason": reason_text,
                    "root_cause": parsed["root_cause"],
                    "improvement": parsed["improvement"],
                    "raw_label": "anomaly",
                    "parse_ok": True,
                    "strict_parse_ok": False,
                    "extraction_ok": True,
                    "parse_source": parse_source,
                    "raw_text": raw_text_for_parse,
                }

    prediction["reason_indicates_anomaly"] = _reason_indicates_anomaly(
        _extract_string_field_from_text((assistant_prefix + raw_text) if assistant_prefix else raw_text, ["reason", "why", "evidence"])
    )
    prediction["generated_fragment"] = raw_text
    prediction["parse_input_source"] = parse_input_source
    prediction["raw_text_for_parse"] = raw_text_for_parse
    return prediction, parsed, parse_source, parse_input_source, raw_text_for_parse


def get_norm_and_lm_head(model):
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Cannot find lm_head / output embeddings on model.")

    norm = None
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        norm = getattr(model.model.language_model, "norm", None)
    if norm is None and hasattr(model, "model"):
        norm = getattr(model.model, "norm", None)
    if norm is None and hasattr(model, "language_model"):
        norm = getattr(model.language_model, "norm", None)
    if norm is None:
        raise RuntimeError("Cannot find final norm layer. Expected model.model.language_model.norm or model.model.norm.")
    return norm, lm_head


def resolve_layer_indices(total_layers: int, start: int, end: Optional[int]) -> List[int]:
    s = max(0, int(start))
    e = total_layers if end is None else min(total_layers, int(end))
    if s >= e:
        raise ValueError(f"Invalid layer range: start={s}, end={e}, total_layers={total_layers}")
    return list(range(s, e))


def analyze_hidden_states_by_anchor_groups(
    *,
    model,
    hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    normal_ids: List[int],
    anomaly_ids: List[int],
    hedge_ids: List[int],
    aggregate: str,
) -> List[Dict[str, Any]]:
    norm, lm_head = get_norm_and_lm_head(model)
    rows: List[Dict[str, Any]] = []
    eps = 1e-12

    with torch.no_grad():
        for layer_idx in layer_indices:
            h = hidden_states[layer_idx][:, -1, :]
            logits = lm_head(norm(h))
            probs = F.softmax(logits.float(), dim=-1)[0]

            p_normal = aggregate_prob_float(probs, normal_ids, aggregate)
            p_anomaly = aggregate_prob_float(probs, anomaly_ids, aggregate)
            p_hedge = aggregate_prob_float(probs, hedge_ids, aggregate)

            anchor_mass = p_normal + p_anomaly + p_hedge
            if anchor_mass > eps:
                p_normal_norm3 = p_normal / anchor_mass
                p_anomaly_norm3 = p_anomaly / anchor_mass
                p_hedge_norm3 = p_hedge / anchor_mass
            else:
                p_normal_norm3 = 0.0
                p_anomaly_norm3 = 0.0
                p_hedge_norm3 = 0.0

            dominant_anchor = max(
                [
                    ("normal", p_normal_norm3),
                    ("anomaly", p_anomaly_norm3),
                    ("hedge", p_hedge_norm3),
                ],
                key=lambda x: x[1],
            )[0]

            rows.append(
                {
                    "layer_index": int(layer_idx),

                    # Raw probabilities under the full vocabulary softmax.
                    "normal_probability": float(p_normal),
                    "anomaly_probability": float(p_anomaly),
                    "hedge_probability": float(p_hedge),

                    # Sum of the three anchor groups in the full-vocabulary distribution.
                    "anchor_probability_mass": float(anchor_mass),

                    # Normalized only among the three anchor groups.
                    # These values show relative preference, not absolute generation probability.
                    "normal_probability_norm3": float(p_normal_norm3),
                    "anomaly_probability_norm3": float(p_anomaly_norm3),
                    "hedge_probability_norm3": float(p_hedge_norm3),
                    "dominant_anchor_norm3": dominant_anchor,
                }
            )

    return rows


def hampel_interpolate_1d(scores: torch.Tensor, window: int = 2, n_sigma: float = 3.0, eps: float = 1e-6):
    if scores.numel() <= 2:
        return scores, torch.zeros_like(scores, dtype=torch.bool)

    x = scores.float().clone()
    L = x.numel()
    outlier = torch.zeros(L, dtype=torch.bool, device=x.device)

    for i in range(L):
        left = max(0, i - int(window))
        right = min(L, i + int(window) + 1)
        local = x[left:right]
        med = torch.median(local)
        mad = torch.median(torch.abs(local - med))
        threshold = float(n_sigma) * 1.4826 * mad.clamp_min(eps)
        if torch.abs(x[i] - med) > threshold:
            outlier[i] = True

    repaired = x.clone()
    valid_idx = torch.where(~outlier)[0]
    if valid_idx.numel() == 0:
        return x, outlier

    for i in torch.where(outlier)[0].tolist():
        left_candidates = valid_idx[valid_idx < i]
        right_candidates = valid_idx[valid_idx > i]
        if left_candidates.numel() > 0 and right_candidates.numel() > 0:
            l = int(left_candidates[-1].item())
            r = int(right_candidates[0].item())
            ratio = float(i - l) / float(r - l)
            repaired[i] = repaired[l] * (1.0 - ratio) + repaired[r] * ratio
        elif left_candidates.numel() > 0:
            repaired[i] = repaired[int(left_candidates[-1].item())]
        elif right_candidates.numel() > 0:
            repaired[i] = repaired[int(right_candidates[0].item())]

    return repaired, outlier


def smooth_1d_moving_average(scores: torch.Tensor, window: int = 3):
    if window <= 1 or scores.numel() <= 2:
        return scores
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x = scores.float().view(1, 1, -1)
    x_pad = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    weight = torch.ones(1, 1, window, device=scores.device, dtype=torch.float32) / float(window)
    return torch.nn.functional.conv1d(x_pad, weight).view(-1)


def smooth_anchor_scores(raw_scores: torch.Tensor, args: argparse.Namespace):
    if (not args.anchor_smooth) or raw_scores.numel() <= 2:
        return raw_scores, {
            "enabled": False,
            "outlier_local_indices": [],
            "raw_scores": raw_scores.detach().float().cpu().tolist(),
            "final_scores": raw_scores.detach().float().cpu().tolist(),
        }

    x = raw_scores.float()
    if args.anchor_remove_outliers:
        repaired, outlier_mask = hampel_interpolate_1d(
            x,
            window=args.anchor_hampel_window,
            n_sigma=args.anchor_hampel_n_sigma,
        )
    else:
        repaired = x
        outlier_mask = torch.zeros_like(x, dtype=torch.bool)

    smoothed = smooth_1d_moving_average(repaired, window=args.anchor_smooth_window)
    beta = max(0.0, min(1.0, float(args.anchor_smooth_beta)))
    final = (1.0 - beta) * x + beta * smoothed

    return final, {
        "enabled": True,
        "outlier_local_indices": torch.where(outlier_mask)[0].detach().cpu().tolist(),
        "raw_scores": x.detach().cpu().tolist(),
        "repaired_scores": repaired.detach().cpu().tolist(),
        "smoothed_scores": smoothed.detach().cpu().tolist(),
        "final_scores": final.detach().cpu().tolist(),
        "hampel_window": args.anchor_hampel_window,
        "hampel_n_sigma": args.anchor_hampel_n_sigma,
        "smooth_window": args.anchor_smooth_window,
        "smooth_beta": beta,
    }


def select_top_p_from_top_k(probs: torch.Tensor, top_k: int, top_p: float):
    vocab_size = probs.shape[-1]
    k = max(1, min(int(top_k), int(vocab_size)))
    candidate_probs, candidate_ids = torch.topk(probs, dim=-1, k=k)
    cumulative_probs = candidate_probs.cumsum(dim=-1)
    cutoff_idx = torch.searchsorted(
        cumulative_probs,
        torch.tensor(float(top_p), device=probs.device, dtype=probs.dtype),
        right=False,
    )
    cutoff = min(int(cutoff_idx.item()) + 1, k)
    return candidate_ids[:cutoff], candidate_probs[:cutoff]


def compute_deco_trace(
    *,
    model,
    model_outputs,
    tokenizer,
    args: argparse.Namespace,
    early_exit_layers: List[int],
    normal_ids: List[int],
    anomaly_ids: List[int],
    hedge_ids: List[int],
    aggregate: str,
) -> Dict[str, Any]:
    if not args.use_deco:
        return {"enabled": False, "reason": "use_deco is false"}

    early_exit_logits = getattr(model_outputs, "early_exit_logits", None)
    if not isinstance(early_exit_logits, dict):
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if hidden_states is not None:
            try:
                norm, lm_head = get_norm_and_lm_head(model)
                early_exit_logits = {}
                with torch.no_grad():
                    for layer_idx in early_exit_layers:
                        if 0 <= int(layer_idx) < len(hidden_states):
                            h = hidden_states[int(layer_idx)][:, -1, :]
                            early_exit_logits[int(layer_idx)] = lm_head(norm(h))
            except Exception as e:
                return {
                    "enabled": False,
                    "error": f"model forward did not return early_exit_logits and hidden-state fallback failed: {e}",
                }
        if not isinstance(early_exit_logits, dict) or not early_exit_logits:
            return {
                "enabled": False,
                "error": "model forward did not return early_exit_logits and hidden-state fallback could not build them",
            }

    last_layer_logits = model_outputs.logits[:, -1, :].float()
    seq_next_token_logits = last_layer_logits[0]
    final_probs = F.softmax(seq_next_token_logits, dim=-1)

    candidate_ids, candidate_probs = select_top_p_from_top_k(
        final_probs,
        top_k=args.threshold_top_k,
        top_p=args.threshold_top_p,
    )

    candidate_records = []
    for rank, (tid, prob) in enumerate(zip(candidate_ids.tolist(), candidate_probs.tolist()), start=1):
        candidate_records.append(
            {
                "rank": rank,
                "token_id": int(tid),
                "token": tokenizer.convert_ids_to_tokens(int(tid)),
                "decoded": tokenizer.decode([int(tid)], skip_special_tokens=False),
                "last_layer_probability": float(prob),
            }
        )

    stacked_early_probs = torch.stack(
        [
            F.softmax(early_exit_logits[layer_idx][0, -1, :].float(), dim=-1)
            for layer_idx in early_exit_layers
        ],
        dim=0,
    )
    candidate_early_probs = stacked_early_probs[:, candidate_ids]

    # The original DeCo "premature_max_prob" is still used as the scaling factor.
    max_candidate_idx = torch.argmax(candidate_early_probs)
    default_target_layer_local = int((max_candidate_idx // candidate_early_probs.size(1)).item())
    default_target_token_local = int((max_candidate_idx % candidate_early_probs.size(1)).item())
    default_selected_layer = int(early_exit_layers[default_target_layer_local])
    premature_max_prob = candidate_early_probs.max().to(dtype=seq_next_token_logits.dtype)

    layer_candidate_table = []
    for local_layer, layer_idx in enumerate(early_exit_layers):
        layer_row = {"layer_index": int(layer_idx), "candidate_probs": []}
        for local_token, token_id in enumerate(candidate_ids.tolist()):
            layer_row["candidate_probs"].append(
                {
                    "token_id": int(token_id),
                    "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                    "probability": float(candidate_early_probs[local_layer, local_token].item()),
                }
            )
        layer_candidate_table.append(layer_row)

    selected_anchor_score_norm3_for_weight = None
    effective_weight_source = "alpha * premature_max_probability"

    if args.anchor_strategy == "defect_aware":
        score_rows = []
        raw_scores = []
        for layer_idx in early_exit_layers:
            probs = F.softmax(early_exit_logits[layer_idx][0, -1, :].float(), dim=-1)
            defect_score = aggregate_prob_tensor(probs, anomaly_ids, aggregate)
            norm_score = aggregate_prob_tensor(probs, normal_ids, aggregate)
            hedge_score = aggregate_prob_tensor(probs, hedge_ids, aggregate) if args.enable_hedge else torch.tensor(0.0, device=probs.device)

            # Match the actual DeCo logic in utils.py: anchor selection uses norm3 score.
            # This trace does not affect generation; it only records the same computation for JSON.
            anchor_mass = (defect_score + norm_score + hedge_score).clamp_min(1e-12)
            defect_score_norm3 = defect_score / anchor_mass
            norm_score_norm3 = norm_score / anchor_mass
            hedge_score_norm3 = hedge_score / anchor_mass

            anchor_score = (
                defect_score_norm3
                - float(args.lambda_norm) * norm_score_norm3
                - float(args.eta_hedge) * hedge_score_norm3
            )
            raw_scores.append(anchor_score)

            score_rows.append(
                {
                    "layer_index": int(layer_idx),
                    "score_type": "norm3",
                    "score_formula": "defect_norm3 - lambda_norm * normal_norm3 - eta_hedge * hedge_norm3",

                    # Raw anchor-group probabilities are kept only for inspection.
                    "defect_score_raw": float(defect_score.item()),
                    "normal_score_raw": float(norm_score.item()),
                    "hedge_score_raw": float(hedge_score.item()),
                    "anchor_probability_mass": float(anchor_mass.item()),

                    # These are the actual quantities used by DeCo anchor selection now.
                    "defect_score_norm3": float(defect_score_norm3.item()),
                    "normal_score_norm3": float(norm_score_norm3.item()),
                    "hedge_score_norm3": float(hedge_score_norm3.item()),
                    "anchor_score_norm3": float(anchor_score.item()),
                }
            )

        raw_scores_t = torch.stack(raw_scores)
        final_scores_t, smooth_info = smooth_anchor_scores(raw_scores_t, args)
        best_local = int(torch.argmax(final_scores_t).item())
        selected_layer = int(early_exit_layers[best_local])
        # Match utils.py: layer selection may use smoothed final_scores_t, but the fusion
        # weight uses the unsmoothed anchor_score_norm3 at the selected layer.
        selected_anchor_score_norm3_for_weight = raw_scores_t[best_local]
        effective_weight_source = "alpha * unsmoothed_anchor_score_norm3"

        for i, row in enumerate(score_rows):
            row["final_anchor_score"] = float(final_scores_t[i].item())
            row["is_selected_anchor_layer"] = bool(i == best_local)
            row["used_for_fusion_weight"] = bool(i == best_local)

        anchor_selection = {
            "strategy": "defect_aware",
            "selected_layer": selected_layer,
            "selected_local_index": best_local,
            "score_formula": "score_l = defect_norm3_l - lambda_norm * normal_norm3_l - eta_hedge * hedge_norm3_l",
            "score_type": "norm3",
            "lambda_norm": float(args.lambda_norm),
            "eta_hedge": float(args.eta_hedge),
            "enable_hedge": bool(args.enable_hedge),
            "aggregate": aggregate,
            "smoothing": smooth_info,
            "per_layer_scores": score_rows,
        }
    else:
        selected_layer = default_selected_layer
        anchor_selection = {
            "strategy": "default",
            "selected_layer": selected_layer,
            "selected_local_index": default_target_layer_local,
            "selected_by": "maximum early-exit probability among candidate tokens",
            "selected_candidate_token_local_index": default_target_token_local,
            "selected_candidate_token_id": int(candidate_ids[default_target_token_local].item()),
            "selected_candidate_token": tokenizer.convert_ids_to_tokens(int(candidate_ids[default_target_token_local].item())),
        }

    selected_layer_logits = early_exit_logits[selected_layer][0, -1, :].float()
    if args.anchor_strategy == "defect_aware" and selected_anchor_score_norm3_for_weight is not None:
        # Match utils.py: use unsmoothed anchor_score_norm3, not premature_max_probability.
        effective_weight = float(args.alpha) * float(selected_anchor_score_norm3_for_weight.item())
        effective_weight_source = "alpha * unsmoothed_anchor_score_norm3"
    else:
        effective_weight = float(args.alpha) * float(premature_max_prob.item())
        effective_weight_source = "alpha * premature_max_probability"
    calibrated_logits = seq_next_token_logits + effective_weight * selected_layer_logits

    keep_mask = torch.zeros_like(calibrated_logits, dtype=torch.bool)
    keep_mask[candidate_ids] = True
    calibrated_logits = calibrated_logits.masked_fill(~keep_mask, -float("inf"))
    calibrated_probs = F.softmax(calibrated_logits, dim=-1)

    top_vals, top_ids = torch.topk(calibrated_probs, k=min(10, calibrated_probs.numel()))
    calibrated_top_tokens = []
    for prob, tid in zip(top_vals.tolist(), top_ids.tolist()):
        calibrated_top_tokens.append(
            {
                "token_id": int(tid),
                "token": tokenizer.convert_ids_to_tokens(int(tid)),
                "decoded": tokenizer.decode([int(tid)], skip_special_tokens=False),
                "probability_after_deco": float(prob),
            }
        )

    return {
        "enabled": True,
        "early_exit_layers": [int(x) for x in early_exit_layers],
        "candidate_selection": {
            "source": "last-layer next-token distribution",
            "threshold_top_k": int(args.threshold_top_k),
            "threshold_top_p": float(args.threshold_top_p),
            "candidates": candidate_records,
        },
        "default_deco_candidate_layer_table": layer_candidate_table,
        "premature_max_probability": float(premature_max_prob.item()),
        "alpha": float(args.alpha),
        "effective_anchor_weight": effective_weight,
        "effective_anchor_weight_source": effective_weight_source,
        "selected_anchor_score_norm3_for_weight": (
            float(selected_anchor_score_norm3_for_weight.item())
            if selected_anchor_score_norm3_for_weight is not None
            else None
        ),
        "logit_fusion_formula": (
            "final_logits = last_layer_logits + alpha * unsmoothed_anchor_score_norm3 * selected_anchor_layer_logits"
            if effective_weight_source == "alpha * unsmoothed_anchor_score_norm3"
            else "final_logits = last_layer_logits + alpha * premature_max_probability * selected_anchor_layer_logits"
        ),
        "candidate_mask_formula": "tokens outside the top-k/top-p candidate set are set to -inf after DeCo fusion",
        "anchor_selection": anchor_selection,
        "selected_anchor_layer": int(selected_layer),
        "calibrated_top_tokens": calibrated_top_tokens,
    }


def save_prob_plot(rows: List[Dict[str, Any]], output_path: Path, title: str, mode: str = "raw") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "norm3":
        normal_key = "normal_probability_norm3"
        anomaly_key = "anomaly_probability_norm3"
        hedge_key = "hedge_probability_norm3"
        ylabel = "Anchor-normalized probability"
        ylim = (0.0, 1.0)
        normal_label = "normal norm3"
        anomaly_label = "anomaly norm3"
        hedge_label = "maybe norm3"
    else:
        normal_key = "normal_probability"
        anomaly_key = "anomaly_probability"
        hedge_key = "hedge_probability"
        ylabel = "Raw full-vocabulary probability"
        ylim = None
        normal_label = "normal raw"
        anomaly_label = "anomaly raw"
        hedge_label = "maybe raw"

    xs = [r["layer_index"] for r in rows]
    plt.figure(figsize=(11, 5.5))
    plt.plot(xs, [r[normal_key] for r in rows], marker="o", linewidth=1.8, markersize=4, label=normal_label)
    plt.plot(xs, [r[anomaly_key] for r in rows], marker="s", linewidth=1.8, markersize=4, label=anomaly_label)
    plt.plot(xs, [r[hedge_key] for r in rows], marker="^", linewidth=1.8, markersize=4, label=hedge_label)
    plt.xlabel("Layer Index")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1")
    if args.shot < 0:
        raise ValueError("--shot must be >= 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    eval_module = load_eval_module()

    available_categories = eval_module.get_available_categories(dataset_root)
    selected_categories = resolve_selected_categories(args, available_categories)

    samples, category_meta = eval_module.discover_visa(
        dataset_root=dataset_root,
        allowed_categories=selected_categories,
        max_samples_per_defect=None,
    )
    cases = choose_two_cases_per_category(samples, selected_categories, args.seed)
    if not cases:
        raise RuntimeError("No probe cases selected. Check dataset labels and category filters.")

    fixed_support_map = (
        eval_module.parse_fixed_support_images(args.fixed_support_images)
        if hasattr(eval_module, "parse_fixed_support_images")
        else {}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_family = infer_model_family(args)

    attn_impl = args.attn_implementation
    if device != "cuda" and attn_impl == "flash_attention_2":
        if args.allow_attn_fallback:
            print("[WARN] flash_attention_2 requires CUDA. Fallback to sdpa.")
            attn_impl = "sdpa"
        else:
            raise RuntimeError("flash_attention_2 requires CUDA. Use --allow_attn_fallback or --attn_implementation sdpa.")

    bundle = load_model_bundle(args, device=device, attn_impl=attn_impl)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    if not args.do_sample:
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    print("===== VisA Layer Probability + DeCo Trace Probe =====")
    print(f"dataset_root: {dataset_root}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"model_family: {model_family}")
    print(f"output_dir: {output_dir}")
    print(f"categories: {selected_categories}")
    print(f"num_probe_cases: {len(cases)}")
    print(f"shot: {args.shot}")
    print(f"use_deco: {args.use_deco}")
    print("====================================================")

    results: List[Dict[str, Any]] = []

    for case_idx, sample in enumerate(tqdm(cases, desc="Probing layers", ncols=120), start=1):
        category = str(sample["category"])
        image_path = Path(sample["image_path"])
        image_name = image_path.name
        gt_label = str(sample.get("ground_truth_label", "good"))
        gt_labels = sample.get("ground_truth_labels", [gt_label])
        probe_kind = str(sample.get("probe_kind", "probe"))

        defect_types = [str(x).strip() for x in category_meta[category].get("defect_types", []) if str(x).strip()]

        anomaly_words = parse_csv(args.anchor_defect_words)
        if args.expand_defect_with_category_labels:
            anomaly_words = unique_keep_order(anomaly_words + defect_types)
        normal_words = parse_csv(args.anchor_norm_words)
        hedge_words = parse_csv(args.anchor_hedge_words)

        anomaly_ids, anomaly_specs = token_ids_for_words(tokenizer, anomaly_words)
        normal_ids, normal_specs = token_ids_for_words(tokenizer, normal_words)
        hedge_ids, hedge_specs = token_ids_for_words(tokenizer, hedge_words)

        support_paths = resolve_support_paths(
            eval_module=eval_module,
            dataset_root=dataset_root,
            category=category,
            shot=args.shot,
            category_meta=category_meta,
            fixed_support_map=fixed_support_map,
            seed=args.seed,
        )

        support_images = [Image.open(p).convert("RGB") for p in support_paths]
        query_image = Image.open(image_path).convert("RGB")
        all_images = support_images + [query_image]

        # Reuse current eval script's build_messages() exactly.
        messages = eval_module.build_messages(category=category, defect_types=defect_types, shot=args.shot)
        inputs, input_token_len, stop_str = prepare_model_inputs(
            family=model_family,
            bundle=bundle,
            args=args,
            messages=messages,
            all_images=all_images,
        )

        early_exit_layers = list(range(args.start_layer, args.end_layer)) if args.use_deco else []
        if args.use_deco and not early_exit_layers:
            raise ValueError("--start_layer must be smaller than --end_layer when --use_deco is set.")

        # Separate forward pass for transparent intermediate states and DeCo reconstruction.
        forward_kwargs = {
            **inputs,
            "return_dict": True,
            "output_hidden_states": True,
        }
        if args.use_deco and model_family == "qwen3_vl":
            forward_kwargs["early_exit_layers"] = early_exit_layers

        with torch.no_grad():
            forward_outputs = model(**forward_kwargs)

        hidden_states = getattr(forward_outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("No hidden_states returned by forward().")

        total_layers = len(hidden_states)
        layer_indices = resolve_layer_indices(total_layers, args.analysis_layer_start, args.analysis_layer_end)
        layer_rows = analyze_hidden_states_by_anchor_groups(
            model=model,
            hidden_states=hidden_states,
            layer_indices=layer_indices,
            normal_ids=normal_ids,
            anomaly_ids=anomaly_ids,
            hedge_ids=hedge_ids,
            aggregate=args.prob_aggregate,
        )

        deco_trace = compute_deco_trace(
            model=model,
            model_outputs=forward_outputs,
            tokenizer=tokenizer,
            args=args,
            early_exit_layers=early_exit_layers,
            normal_ids=normal_ids,
            anomaly_ids=anomaly_ids,
            hedge_ids=hedge_ids,
            aggregate=args.prob_aggregate,
        )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "num_beams": args.num_beams,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "repetition_penalty": args.repetition_penalty,
        }

        if args.use_deco:
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
                    "anchor_defect_words": ",".join(anomaly_words),
                    "anchor_norm_words": args.anchor_norm_words,
                    "anchor_hedge_words": args.anchor_hedge_words,
                    "deco_debug": args.deco_debug,
                    "tokenizer": tokenizer,
                    "anchor_smooth": args.anchor_smooth,
                    "anchor_remove_outliers": args.anchor_remove_outliers,
                    "anchor_hampel_window": args.anchor_hampel_window,
                    "anchor_hampel_n_sigma": args.anchor_hampel_n_sigma,
                    "anchor_smooth_window": args.anchor_smooth_window,
                    "anchor_smooth_beta": args.anchor_smooth_beta,
                }
            )
        else:
            generation_kwargs["use_deco"] = False

        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        gen_ids = output_ids[:, input_token_len:]
        if model_family == "qwen3_vl":
            raw_text = bundle["processor"].batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            raw_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            if stop_str and raw_text.endswith(stop_str):
                raw_text = raw_text[:-len(stop_str)]
            raw_text = raw_text.strip()

        prediction, parsed, parse_source, parse_input_source, raw_text_for_parse = robust_parse_prediction_with_assistant_prefix(
            eval_module=eval_module,
            raw_text=raw_text,
            assistant_prefix=args.assistant_prefix,
            defect_types=defect_types,
            image_size=query_image.size,
        )

        base_plot_name = (
            f"{case_idx:03d}_{safe_name(category)}_{safe_name(probe_kind)}_"
            f"GT-{safe_name(gt_label)}_{safe_name(image_path.stem)}"
        )
        raw_plot_path = plot_dir / f"{base_plot_name}_raw.png"
        norm3_plot_path = plot_dir / f"{base_plot_name}_norm3.png"
        title_base = (
            f"{category} | {probe_kind} | {image_name} | GT={gt_label} | "
            f"pred={prediction.get('label')} | raw_label={prediction.get('raw_label')} | parse={prediction.get('parse_source')}"
        )

        save_prob_plot(
            rows=layer_rows,
            output_path=raw_plot_path,
            title=title_base + " | raw full-vocabulary probability",
            mode="raw",
        )
        save_prob_plot(
            rows=layer_rows,
            output_path=norm3_plot_path,
            title=title_base + " | anchor-normalized probability",
            mode="norm3",
        )

        result = {
            "index": case_idx,
            "category": category,
            "probe_kind": probe_kind,
            "image_path": str(image_path),
            "image_name": image_name,
            "ground_truth": {
                "label": gt_label,
                "labels": gt_labels,
                "is_anomaly": gt_label != "good",
            },
            "prediction": prediction,
            "support_image_paths": support_paths,
            "defect_types": defect_types,
            "anchor_words": {
                "normal": normal_words,
                "anomaly": anomaly_words,
                "hedge": hedge_words,
            },
            "anchor_token_ids": {
                "normal": normal_ids,
                "anomaly": anomaly_ids,
                "hedge": hedge_ids,
            },
            "anchor_token_specs": {
                "normal": normal_specs,
                "anomaly": anomaly_specs,
                "hedge": hedge_specs,
            },
            "layer_probability_analysis": {
                "prob_aggregate": args.prob_aggregate,
                "layer_indices": layer_indices,
                "layer_probabilities": layer_rows,
            },
            "deco_decision_trace": deco_trace,
            "generation": {
                "raw_text": raw_text,
                "raw_text_for_parse": raw_text_for_parse,
                "parse_input_source": parse_input_source,
                "parsed_json": parsed,
                "parse_source": parse_source,
                "generated_token_ids": [int(x) for x in gen_ids[0].tolist()],
                "final_label_source": "normalized_prediction.label",
            },
            "artifacts": {
                "raw_plot_path": str(raw_plot_path),
                "norm3_plot_path": str(norm3_plot_path),
            },
        }
        results.append(result)

    output_json = output_dir / "visa_layer_probability_probe.json"
    output_data = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": str(dataset_root),
            "model_name_or_path": args.model_name_or_path,
            "model_family": model_family,
            "selected_categories": selected_categories,
            "num_probe_cases": len(results),
            "case_policy": "one normal image and one anomaly image per category",
            "shot": args.shot,
            "assistant_prefix": args.assistant_prefix,
            "prob_aggregate": args.prob_aggregate,
            "anchor_defect_words_base": args.anchor_defect_words,
            "anchor_norm_words": args.anchor_norm_words,
            "anchor_hedge_words": args.anchor_hedge_words,
            "expand_defect_with_category_labels": args.expand_defect_with_category_labels,
            "use_deco": args.use_deco,
            "deco_config": {
                "alpha": args.alpha,
                "threshold_top_p": args.threshold_top_p,
                "threshold_top_k": args.threshold_top_k,
                "start_layer": args.start_layer,
                "end_layer": args.end_layer,
                "anchor_strategy": args.anchor_strategy,
                "lambda_norm": args.lambda_norm,
                "eta_hedge": args.eta_hedge,
                "enable_hedge": args.enable_hedge,
                "use_logsumexp": args.use_logsumexp,
                "gamma": args.gamma,
                "anchor_smooth": args.anchor_smooth,
                "anchor_remove_outliers": args.anchor_remove_outliers,
                "anchor_hampel_window": args.anchor_hampel_window,
                "anchor_hampel_n_sigma": args.anchor_hampel_n_sigma,
                "anchor_smooth_window": args.anchor_smooth_window,
                "anchor_smooth_beta": args.anchor_smooth_beta,
            } if args.use_deco else None,
        },
        "results": results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved JSON: {output_json}")
    print(f"[OK] Saved plots: {plot_dir}")


if __name__ == "__main__":
    main()
