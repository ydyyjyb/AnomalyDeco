import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from probe_qwen3_deco_layer_curve import Qwen3VLForConditionalGeneration, build_messages, parse_int_range
from eval_qwen3_visa_patch_phrase_auroc import (
    accuracy,
    binary_auc,
    build_patch_question,
    build_question,
    collect_visa_samples,
    load_mask,
    make_regions,
    patch_map_to_pixel,
    resize_score_values,
    score_region,
    smooth_patch_scores,
    topk_count,
    write_csv,
)


STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "because",
    "based",
    "can",
    "condition",
    "does",
    "evidence",
    "for",
    "from",
    "has",
    "have",
    "image",
    "indicates",
    "is",
    "it",
    "main",
    "normal",
    "not",
    "object",
    "of",
    "on",
    "or",
    "overall",
    "patch",
    "reasoning",
    "region",
    "seems",
    "shows",
    "surface",
    "the",
    "there",
    "this",
    "visible",
    "with",
    "yes",
    "no",
    "answer",
    "final",
}

DESCRIPTIVE_STOPWORDS = {
    "black",
    "blue",
    "brown",
    "center",
    "circular",
    "clear",
    "color",
    "colored",
    "dark",
    "front",
    "gray",
    "green",
    "large",
    "left",
    "light",
    "middle",
    "orange",
    "part",
    "piece",
    "red",
    "right",
    "round",
    "shape",
    "side",
    "small",
    "top",
    "white",
    "yellow",
}

DEFECT_LEXICON = {
    "abnormal",
    "anomalous",
    "anomaly",
    "bent",
    "broken",
    "bubble",
    "burn",
    "contamination",
    "crack",
    "cracked",
    "damage",
    "damaged",
    "defect",
    "defective",
    "deformation",
    "dent",
    "dirty",
    "discoloration",
    "excessive",
    "fault",
    "faulty",
    "foreign",
    "hole",
    "missing",
    "particle",
    "particles",
    "scratch",
    "scratched",
    "solder",
    "spot",
    "stain",
    "tear",
    "wear",
}

GENERIC_OBJECT_WORDS = {
    "board",
    "boards",
    "candle",
    "candles",
    "capsule",
    "capsules",
    "cashew",
    "cashews",
    "chewinggum",
    "fryum",
    "macaroni",
    "pcb",
    "pipe",
    "product",
    "products",
    "object",
    "objects",
    "wick",
    "wicks",
    "packaging",
    "surface",
    "surfaces",
    "component",
    "components",
}

NEGATION_WORDS = {"no", "not", "without", "none", "neither", "nor", "absence", "absent", "free"}
INSPECTION_WORDS = {"check", "inspect", "look", "looking", "verify", "examine"}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def display_category(category):
    return str(category or "object").replace("_", " ").replace("-", " ")


def clean_token_text(text):
    text = str(text).replace("\u0120", " ").replace("\u2581", " ").strip()
    text = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text)
    return text.lower()


def token_to_word(tokenizer, token_id):
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return clean_token_text(token)


def is_candidate_word(word):
    if not word or len(word) < 3:
        return False
    if word in STOPWORDS or word in DESCRIPTIVE_STOPWORDS:
        return False
    if not re.fullmatch(r"[a-z][a-z\-]*", word):
        return False
    return True


def category_defect_description(category_bank, max_items=8):
    phrases = []
    for phrase in category_bank.get("defect_bank", []):
        phrase = str(phrase).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= int(max_items):
            break
    if not phrases:
        phrases = ["visible surface abnormality", "damage", "contamination", "missing parts", "deformation"]
    return ", ".join(phrases)


def build_real_question(record, category_bank):
    anomaly_prompt = category_defect_description(category_bank)
    return (
        "Inspect the image for visual anomaly detection and localization. "
        f"The object category is {display_category(record['category'])}. "
        f"Known possible defect descriptions for this category include: {anomaly_prompt}. "
        "Ignore differences caused only by object placement, position, rotation, orientation, or overall arrangement. "
        "Do not treat these layout differences as defects unless they are accompanied by visible structural damage, "
        "missing parts, contamination, deformation, or another category-specific defect. "
        "Reason step by step from visual evidence. Mention suspicious visual cues and likely locations. "
        "Finish with exactly one final answer line: Answer: Yes if anomalous, otherwise Answer: No."
    )


def processor_inputs(processor, model, image, image_path, question, min_pixels, max_pixels):
    messages = build_messages(image_path, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}, text


def generate_reasoning(processor, model, tokenizer, image, image_path, question, args):
    inputs, prompt_text = processor_inputs(
        processor, model, image, image_path, question, args.min_pixels, args.max_pixels
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=False,
        )
    generated_ids = output_ids[:, prompt_len:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return inputs, output_ids, generated_ids[0].detach().cpu().tolist(), generated_text, prompt_text


def get_image_token_positions(tokenizer, input_ids):
    image_token_ids = []
    for token in ["<|image_pad|>", "<image>", "<|vision_pad|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid is not None and int(tid) >= 0:
                image_token_ids.append(int(tid))
        except Exception:
            pass
    ids = input_ids[0].detach().cpu().tolist()
    positions = [idx for idx, tid in enumerate(ids) if int(tid) in image_token_ids]
    return positions


def infer_visual_grid(num_tokens, image_grid_thw):
    if image_grid_thw is not None:
        vals = image_grid_thw.detach().cpu().reshape(-1).tolist()
        if len(vals) >= 3:
            h, w = int(vals[-2]), int(vals[-1])
            if h > 0 and w > 0 and h * w == num_tokens:
                return h, w
    side = int(round(math.sqrt(max(1, num_tokens))))
    if side * side == num_tokens:
        return side, side
    h = side
    w = int(math.ceil(num_tokens / max(1, h)))
    return h, w


def vector_to_grid(vec, h, w):
    vec = torch.as_tensor(vec, dtype=torch.float32)
    target = h * w
    if vec.numel() < target:
        pad = torch.zeros(target - vec.numel(), dtype=torch.float32)
        vec = torch.cat([vec, pad], dim=0)
    if vec.numel() > target:
        vec = vec[:target]
    return vec.reshape(h, w)


def spatial_concentration(att_map, top_frac=0.05):
    vals = att_map.flatten().float().clamp_min(0)
    total = vals.sum().clamp_min(1e-12)
    probs = vals / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    entropy_norm = entropy / math.log(max(2, vals.numel()))
    k = max(1, int(math.ceil(vals.numel() * top_frac)))
    top_conc = torch.topk(probs, k=k).values.sum()
    return float((0.5 * (1.0 - entropy_norm) + 0.5 * top_conc).clamp(0, 1).item())


def lexical_st(word, anomaly_words):
    if not word:
        return 0.0
    if word in GENERIC_OBJECT_WORDS:
        return 0.0
    if word in DEFECT_LEXICON:
        return 1.0
    if any(word == x or word in x or x in word for x in anomaly_words):
        return 0.75
    if any(root in word for root in ["defect", "damage", "scratch", "crack", "stain", "hole", "missing", "broken"]):
        return 0.75
    return 0.0


def extract_anomaly_words(record, category_bank):
    text = " ".join(category_bank.get("defect_bank", []))
    words = set()
    category_words = set(re.findall(r"[a-z][a-z\-]{2,}", str(record.get("category", "")).lower()))
    for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()):
        if word in category_words or word in GENERIC_OBJECT_WORDS:
            continue
        if word in DEFECT_LEXICON or word in {"foreign", "particles", "different", "spot", "weird", "extra", "excessive", "missing", "melded"}:
            words.add(word)
    return words


def is_negated_or_inspection_context(words, idx, window=6):
    left = words[max(0, idx - window) : idx]
    left_set = set(left)
    if left_set & NEGATION_WORDS:
        return True
    if left_set & INSPECTION_WORDS:
        return True
    phrase = " ".join(left)
    if any(p in phrase for p in ["no visible", "no sign", "no signs", "free of", "absence of", "look for", "check for"]):
        return True
    return False


def real_lite_attention_map(processor, model, tokenizer, image, record, category_bank, args):
    question = build_real_question(record, category_bank)
    prompt_inputs, full_sequence, generated_ids, generated_text, prompt_text = generate_reasoning(
        processor, model, tokenizer, image, record["image"], question, args
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    generated_tokens_total = len(generated_ids)
    attention_cap = int(getattr(args, "attention_max_generated_tokens", 256))
    if attention_cap > 0 and generated_tokens_total > attention_cap:
        generated_ids_for_attention = generated_ids[:attention_cap]
        full_sequence_for_attention = full_sequence[:, : prompt_len + attention_cap]
        attention_truncated = True
    else:
        generated_ids_for_attention = generated_ids
        full_sequence_for_attention = full_sequence
        attention_truncated = False
    image_positions = get_image_token_positions(tokenizer, prompt_inputs["input_ids"])
    if not image_positions:
        return {
            "score": 0.0,
            "map": torch.zeros((1, 1), dtype=torch.float32),
            "pixel_map": torch.zeros((image.size[1], image.size[0]), dtype=torch.float32),
            "tokens": [],
            "generated_text": generated_text,
            "prompt_text": prompt_text,
            "error": "no image token positions found",
            "generated_tokens_total": generated_tokens_total,
            "attention_tokens_used": len(generated_ids_for_attention),
            "attention_truncated": attention_truncated,
        }

    forward_inputs = dict(prompt_inputs)
    forward_inputs["input_ids"] = full_sequence_for_attention.to(model.device)
    forward_inputs["attention_mask"] = torch.ones_like(forward_inputs["input_ids"], device=model.device)
    with torch.inference_mode():
        outputs = model(**forward_inputs, return_dict=True, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise RuntimeError("No attentions returned. Use --attn-implementation eager for ReAL-lite.")

    att = outputs.attentions[-1][0].float().mean(dim=0).detach().cpu()
    h, w = infer_visual_grid(len(image_positions), prompt_inputs.get("image_grid_thw"))
    anomaly_words = extract_anomaly_words(record, category_bank)

    selected = []
    weighted_maps = []
    generated_token_positions = list(range(prompt_len, int(full_sequence_for_attention.shape[1])))
    generated_words = [token_to_word(tokenizer, tid) for tid in generated_ids_for_attention]
    for local_idx, seq_pos in enumerate(generated_token_positions):
        if local_idx >= len(generated_ids_for_attention):
            continue
        token_id = generated_ids_for_attention[local_idx]
        word = token_to_word(tokenizer, token_id)
        if not is_candidate_word(word):
            continue
        if word in GENERIC_OBJECT_WORDS:
            continue
        if is_negated_or_inspection_context(generated_words, local_idx):
            continue
        token_att_vec = att[seq_pos, image_positions]
        token_map = vector_to_grid(token_att_vec, h, w)
        token_map = token_map / token_map.sum().clamp_min(1e-12)
        st = lexical_st(word, anomaly_words)
        si = spatial_concentration(token_map, args.si_top_frac)
        if st < args.st_threshold or si < args.si_threshold:
            continue
        weight = float(args.alpha) * st + float(args.beta) * si
        selected.append(
            {
                "token_index": local_idx,
                "token_id": int(token_id),
                "word": word,
                "ST": st,
                "SI": si,
                "weight": weight,
            }
        )
        weighted_maps.append(token_map * weight)

    if weighted_maps:
        anomaly_map = torch.stack(weighted_maps, dim=0).sum(dim=0)
        anomaly_map = anomaly_map / anomaly_map.max().clamp_min(1e-12)
    else:
        anomaly_map = torch.zeros((h, w), dtype=torch.float32)
    pixel_map = F.interpolate(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        size=(image.size[1], image.size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    score = top_percent_mean(pixel_map, args.real_top_percent)
    return {
        "score": score,
        "map": anomaly_map,
        "pixel_map": pixel_map,
        "tokens": selected,
        "generated_text": generated_text,
        "prompt_text": prompt_text,
        "error": "",
        "generated_tokens_total": generated_tokens_total,
        "attention_tokens_used": len(generated_ids_for_attention),
        "attention_truncated": attention_truncated,
    }


def top_percent_mean(pixel_map, percent):
    vals = torch.as_tensor(pixel_map, dtype=torch.float32).flatten()
    k = max(1, int(math.ceil(vals.numel() * float(percent) / 100.0)))
    return float(torch.topk(vals, k=k).values.mean().item())


def patch_phrase_score(processor, model, image, record, category_bank, layers, args):
    defect_phrases = category_bank["defect_bank"][: args.max_defect_phrases]
    normal_phrases = category_bank["normal_bank"][: args.max_normal_phrases]
    nuisance_phrases = category_bank["nuisance_bank"][: args.max_nuisance_phrases]
    question = build_question(record, args.patch_question_template)
    full_score, _, _, _, _, _ = score_region(
        processor,
        model,
        image,
        record["image"],
        question,
        args.patch_prefix,
        defect_phrases,
        normal_phrases,
        nuisance_phrases,
        layers,
        args,
    )
    patch_rows = []
    for region in make_regions(image, args.grid):
        region_score, _, _, _, _, _ = score_region(
            processor,
            model,
            region["image"],
            record["image"],
            build_patch_question(question, region, args.grid),
            args.patch_prefix,
            defect_phrases,
            normal_phrases,
            nuisance_phrases,
            layers,
            args,
        )
        if args.patch_full_contrast:
            region_score -= full_score
        x0, y0, x1, y1 = region["box"]
        patch_rows.append(
            {
                "row": region["row"],
                "col": region["col"],
                "patch_index": region["patch_index"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "patch_score_raw": region_score,
                "patch_score": region_score,
            }
        )
    _, smoothed = smooth_patch_scores(patch_rows, args.grid)
    k = topk_count(args.topk_patches, len(patch_rows))
    ranked = sorted([float(x["patch_score"]) for x in patch_rows], reverse=True)
    patch_score = sum(ranked[:k]) / k
    image_score = patch_score + float(args.full_score_weight) * full_score
    pixel_map = torch.as_tensor(patch_map_to_pixel(image.size, patch_rows, args.grid), dtype=torch.float32)
    return {
        "score": float(image_score),
        "full_score": float(full_score),
        "patch_score": float(patch_score),
        "pixel_map": pixel_map,
        "patch_map": smoothed,
    }


def resize_score_values_from_tensor(pixel_map, size):
    if size is None:
        return [float(x) for x in pixel_map.reshape(-1).tolist()]
    tensor = pixel_map.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(int(size[1]), int(size[0])), mode="bilinear", align_corners=False)
    return [float(x) for x in resized.squeeze().reshape(-1).tolist()]


def save_heatmap_overlay(image, pixel_map, output_png, title="anomaly map"):
    plt.figure(figsize=(7, 6))
    plt.imshow(image.convert("RGB"))
    plt.imshow(pixel_map.detach().cpu().numpy(), cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def save_map_only(pixel_map, output_png, title):
    plt.figure(figsize=(6, 5.5))
    im = plt.imshow(pixel_map.detach().cpu().numpy(), cmap="jet")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_block(rows, key, pixel_key=None):
    labels = [int(r["label"]) for r in rows]
    scores = [float(r[key]) for r in rows]
    return {
        "acc": accuracy(labels, scores, 0.0),
        "I-AUROC": binary_auc(labels, scores),
    }


def best_accuracy(labels, scores):
    if not labels:
        return None, None
    candidates = sorted(set(float(s) for s in scores if math.isfinite(float(s))))
    if not candidates:
        return None, None
    thresholds = [candidates[0] - 1e-6] + candidates + [candidates[-1] + 1e-6]
    best_acc = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        acc = accuracy(labels, scores, thr)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_acc, best_thr


def make_metric(rows, score_key, pixel_labels, pixel_scores, threshold):
    labels = [int(r["label"]) for r in rows]
    scores = [float(r[score_key]) for r in rows]
    best_acc, best_thr = best_accuracy(labels, scores)
    return {
        "acc": accuracy(labels, scores, threshold),
        "threshold": threshold,
        "best_acc": best_acc,
        "best_threshold": best_thr,
        "I-AUROC": binary_auc(labels, scores),
        "p-AUROC": binary_auc(pixel_labels, pixel_scores) if pixel_labels else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--categories", default="")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--layers", default="20:29")
    parser.add_argument("--topk-patches", default="top5%")
    parser.add_argument("--normal-lambda", type=float, default=1.0)
    parser.add_argument("--nuisance-lambda", type=float, default=0.5)
    parser.add_argument("--patch-full-contrast", action="store_true")
    parser.add_argument("--full-score-weight", type=float, default=0.3)
    parser.add_argument("--fusion-patch-weight", type=float, default=0.5)
    parser.add_argument("--fusion-real-weight", type=float, default=0.5)
    parser.add_argument("--patch-threshold", type=float, default=0.0)
    parser.add_argument("--real-threshold", type=float, default=0.5)
    parser.add_argument("--fusion-threshold", type=float, default=0.0)
    parser.add_argument("--max-defect-phrases", type=int, default=8)
    parser.add_argument("--max-normal-phrases", type=int, default=6)
    parser.add_argument("--max-nuisance-phrases", type=int, default=6)
    parser.add_argument("--phrase-batch-size", type=int, default=1)
    parser.add_argument("--phrase-bank-agg", default="max", choices=["max", "mean"])
    parser.add_argument("--phrase-reduction", default="mean", choices=["mean", "sum"])
    parser.add_argument("--score-mode", default="prob_margin", choices=["prob_margin", "logprob_margin"])
    parser.add_argument("--patch-question-template", default=(
        "Decide whether the main object in this image is normal or anomalous. "
        "The object category is {category}. "
        "Known category-specific defect descriptions include: {anomaly_prompt}."
    ))
    parser.add_argument("--patch-prefix", default="The visual condition is")
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--real-top-percent", type=float, default=5.0)
    parser.add_argument("--st-threshold", type=float, default=0.5)
    parser.add_argument("--si-threshold", type=float, default=0.12)
    parser.add_argument("--si-top-frac", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--pixel-auroc-size", type=int, default=256)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--skip-patch-baseline", action="store_true")
    args = parser.parse_args()

    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration is not available from transformers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_range(args.layers)
    samples, banks = collect_visa_samples(
        args.visa_root,
        args.categories,
        None,
        args.sample_normal,
        args.sample_anomaly,
        args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    pixel_labels = {"patch": [], "real": [], "fusion": []}
    pixel_scores = {"patch": [], "real": [], "fusion": []}
    eval_size = None if int(args.pixel_auroc_size) <= 0 else (int(args.pixel_auroc_size), int(args.pixel_auroc_size))
    for idx, record in enumerate(samples, start=1):
        image = Image.open(record["image"]).convert("RGB")
        sample_dir = out_dir / f"{idx:03d}-{record['category']}-{record['label_name']}-{record['image_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        bank = banks[record["category"]]

        if args.skip_patch_baseline:
            patch_result = {
                "score": 0.0,
                "full_score": 0.0,
                "patch_score": 0.0,
                "pixel_map": torch.zeros((image.size[1], image.size[0]), dtype=torch.float32),
                "patch_map": [],
            }
        else:
            patch_result = patch_phrase_score(processor, model, image, record, bank, layers, args)

        real_result = real_lite_attention_map(processor, model, tokenizer, image, record, bank, args)
        real_score = float(real_result["score"]) + float(args.full_score_weight) * float(patch_result["full_score"])
        fusion_score = float(args.fusion_patch_weight) * float(patch_result["score"]) + float(args.fusion_real_weight) * real_score
        fusion_pixel_map = (
            float(args.fusion_patch_weight) * patch_result["pixel_map"].float()
            + float(args.fusion_real_weight) * real_result["pixel_map"].float()
        )
        row = {
            **record,
            "patch_score": float(patch_result["score"]),
            "real_score": real_score,
            "fusion_score": fusion_score,
            "full_score": float(patch_result["full_score"]),
            "real_raw_score": float(real_result["score"]),
            "num_real_tokens": len(real_result["tokens"]),
            "real_error": real_result["error"],
        }
        rows.append(row)

        mask_labels = load_mask(record["mask"], eval_size or image.size)
        if mask_labels is not None:
            for name, pmap in [
                ("patch", patch_result["pixel_map"]),
                ("real", real_result["pixel_map"]),
                ("fusion", fusion_pixel_map),
            ]:
                pixel_labels[name].extend(mask_labels)
                pixel_scores[name].extend(resize_score_values_from_tensor(pmap, eval_size))

        write_json(
            sample_dir / "summary.json",
            {
                **row,
                "generated_reasoning": real_result["generated_text"],
                "prompt_text": real_result["prompt_text"],
                "selected_real_tokens": real_result["tokens"],
                "defect_bank": bank["defect_bank"],
            },
        )
        if args.save_visualizations:
            save_heatmap_overlay(image, real_result["pixel_map"], sample_dir / "real_lite_anomaly_overlay.png", "ReAL-lite anomaly map")
            save_map_only(real_result["map"], sample_dir / "real_lite_visual_token_map.png", "ReAL-lite visual-token map")
            save_heatmap_overlay(image, patch_result["pixel_map"], sample_dir / "patch_phrase_overlay.png", "Patch phrase score")
            save_heatmap_overlay(image, fusion_pixel_map, sample_dir / "fusion_overlay.png", "Fusion score")
        print(
            f"[{idx}/{len(samples)}] {record['category']} {record['label_name']} {record['image_id']} "
            f"patch={row['patch_score']:.4f} real={row['real_score']:.4f} fusion={row['fusion_score']:.4f} tokens={row['num_real_tokens']}"
        )

    metrics = {
        "patch_phrase_only": make_metric(rows, "patch_score", pixel_labels["patch"], pixel_scores["patch"], args.patch_threshold),
        "ReAL_lite_only": make_metric(rows, "real_score", pixel_labels["real"], pixel_scores["real"], args.real_threshold),
        "fusion": make_metric(rows, "fusion_score", pixel_labels["fusion"], pixel_scores["fusion"], args.fusion_threshold),
    }

    write_csv(out_dir / "image_scores.csv", rows)
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "0-manifest.json",
        {
            "model_path": args.model_path,
            "visa_root": args.visa_root,
            "num_samples": len(samples),
            "layers": layers,
            "real_logic": "Generate reasoning, collect generated-token to image-token attentions, filter tokens by ST/SI, aggregate weighted maps.",
            "args": vars(args),
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
