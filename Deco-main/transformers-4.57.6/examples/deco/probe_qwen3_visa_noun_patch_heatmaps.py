import argparse
import csv
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor

from probe_qwen3_deco_layer_curve import (
    Qwen3VLForConditionalGeneration,
    build_messages,
    logits_by_layer_from_hidden_states,
    parse_int_range,
    token_label,
)
from probe_qwen3_deco_layer_curve_norm import parse_csv_words


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "around",
    "because",
    "being",
    "between",
    "board",
    "could",
    "defect",
    "defective",
    "defects",
    "image",
    "images",
    "into",
    "itself",
    "larger",
    "local",
    "main",
    "normal",
    "object",
    "objects",
    "product",
    "products",
    "should",
    "surface",
    "there",
    "these",
    "thing",
    "this",
    "those",
    "visible",
    "where",
    "which",
    "with",
    "without",
}


def display_category(category):
    return str(category or "object").replace("_", " ").replace("-", " ")


def flatten_prompt_strings(obj):
    strings = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        if isinstance(obj.get("prompt"), str):
            strings.append(obj["prompt"])
            return strings
        for value in obj.values():
            strings.extend(flatten_prompt_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            strings.extend(flatten_prompt_strings(value))
    return strings


def load_category_anomaly_prompt(category_dir):
    path = Path(category_dir) / "anomaly_prompt.json"
    if not path.exists():
        return {}, [], str(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, [], str(path)
    return data, flatten_prompt_strings(data), str(path)


def normalized_rel_path(path):
    return str(path).replace("\\", "/").lstrip("./")


def image_relative_candidates(image_path, visa_root, category_dir):
    candidates = [normalized_rel_path(image_path)]
    for base in [visa_root, category_dir]:
        try:
            candidates.append(normalized_rel_path(Path(image_path).relative_to(base)))
        except ValueError:
            pass
    return list(dict.fromkeys(candidates))


def iter_prompt_entries(prompt_data):
    if isinstance(prompt_data, dict):
        if "prompt" in prompt_data and "image" in prompt_data:
            yield prompt_data
        for value in prompt_data.values():
            yield from iter_prompt_entries(value)
    elif isinstance(prompt_data, list):
        for value in prompt_data:
            yield from iter_prompt_entries(value)


def prompt_for_image(prompt_data, image_path, visa_root, category_dir):
    image_candidates = image_relative_candidates(image_path, visa_root, category_dir)
    for entry in iter_prompt_entries(prompt_data):
        entry_image = normalized_rel_path(entry.get("image", ""))
        if not entry_image:
            continue
        for candidate in image_candidates:
            if entry_image == candidate or candidate.endswith(entry_image) or entry_image.endswith(candidate):
                return str(entry.get("prompt", "")).strip(), entry
    return "", {}


def prompt_for_type(prompt_data, anomaly_type):
    if isinstance(prompt_data, dict):
        candidates = [
            anomaly_type,
            anomaly_type.lower(),
            anomaly_type.upper(),
            anomaly_type.replace("_", " "),
            anomaly_type.replace(" ", "_"),
        ]
        for key in candidates:
            if key in prompt_data:
                return " ".join(flatten_prompt_strings(prompt_data[key]))
        for key, value in prompt_data.items():
            if str(key).lower() in {candidate.lower() for candidate in candidates}:
                return " ".join(flatten_prompt_strings(value))
        for value in prompt_data.values():
            matched = prompt_for_type(value, anomaly_type)
            if matched:
                return matched
    return ""


def candidate_words_from_prompts(strings, limit):
    counts = {}
    for text in strings:
        for word in re.findall(r"[A-Za-z][A-Za-z_\-]{2,}", text.lower()):
            word = word.replace("_", " ").replace("-", " ").strip()
            if " " in word:
                for part in word.split():
                    if len(part) >= 3 and part not in STOPWORDS:
                        counts[part] = counts.get(part, 0) + 1
                continue
            if word not in STOPWORDS:
                counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, _ in ranked[:limit]]


def collect_visa_anomalies(visa_root):
    root = Path(visa_root)
    records = []
    for image_path in root.glob("*/Data/Images/*/*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        anomaly_type = image_path.parent.name
        if anomaly_type.lower() == "normal":
            continue
        try:
            category = image_path.parents[3].name
            category_dir = image_path.parents[3]
        except IndexError:
            category = "unknown"
            category_dir = image_path.parent
        prompt_data, prompt_strings, prompt_path = load_category_anomaly_prompt(category_dir)
        matched_prompt, matched_entry = prompt_for_image(prompt_data, image_path, root, category_dir)
        if not matched_prompt:
            matched_prompt = prompt_for_type(prompt_data, anomaly_type)
        records.append(
            {
                "image": str(image_path),
                "image_id": image_path.stem,
                "category": category,
                "anomaly_type": anomaly_type,
                "anomaly_prompt_path": prompt_path,
                "anomaly_prompt_text": matched_prompt or " ".join(prompt_strings),
                "anomaly_prompt_index": matched_entry.get("index", ""),
                "anomaly_prompt_image": matched_entry.get("image", ""),
                "anomaly_prompt_mask": matched_entry.get("mask", ""),
                "category_anomaly_prompts": prompt_strings,
            }
        )
    return sorted(records, key=lambda x: x["image"])


def sample_records(records, n, seed):
    rng = random.Random(seed)
    if len(records) < n:
        raise RuntimeError(f"Not enough anomaly images: need {n}, found {len(records)}")
    return rng.sample(records, n)


def candidate_token_ids(tokenizer, words):
    detail = {}
    all_ids = []
    for word in words:
        variants = [word, " " + word] if not word.startswith(" ") else [word]
        ids = []
        for variant in variants:
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                enc = []
            if enc:
                ids.append(int(enc[0]))
        ids = sorted(set(ids))
        detail[word] = ids
        all_ids.extend(ids)
    return sorted(set(all_ids)), detail


def prepare_inputs(processor, model, image, source_image_path, question, prefix, min_pixels, max_pixels):
    messages = build_messages(source_image_path, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefix:
        text += " " + prefix.strip()
    inputs = processor(
        text=[text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}, text


def build_record_question(record, question_template):
    category = display_category(record.get("category", "object"))
    anomaly_prompt = str(record.get("anomaly_prompt_text", "")).strip()
    if not anomaly_prompt:
        anomaly_prompt = "Use visible evidence of abnormal appearance or damage."
    return question_template.format(
        category=category,
        anomaly_type=record.get("anomaly_type", "anomaly"),
        anomaly_prompt=anomaly_prompt,
    )


def make_regions(image, grid, overlap):
    width, height = image.size
    regions = [
        {
            "region": "full",
            "kind": "full",
            "patch_index": 0,
            "patch_total": grid * grid,
            "row": -1,
            "col": -1,
            "box": (0, 0, width, height),
            "image": image,
        }
    ]
    step_x = width / grid
    step_y = height / grid
    expand_x = step_x * overlap / 2.0
    expand_y = step_y * overlap / 2.0
    for row in range(grid):
        for col in range(grid):
            x0 = int(max(0, round(col * step_x - expand_x)))
            y0 = int(max(0, round(row * step_y - expand_y)))
            x1 = int(min(width, round((col + 1) * step_x + expand_x)))
            y1 = int(min(height, round((row + 1) * step_y + expand_y)))
            regions.append(
                {
                    "region": f"patch_r{row}_c{col}",
                    "kind": "patch",
                    "patch_index": row * grid + col + 1,
                    "patch_total": grid * grid,
                    "row": row,
                    "col": col,
                    "box": (x0, y0, x1, y1),
                    "image": image.crop((x0, y0, x1, y1)),
                }
            )
    return regions


def build_patch_question(base_question, region, grid, patch_context):
    if region["kind"] == "full" or not patch_context:
        return base_question
    return (
        f"{base_question}\n"
        f"You are seeing a local crop from a larger image of the main object. This is patch "
        f"{region['patch_index']} of {region['patch_total']} in a {grid}x{grid} grid "
        f"(row {region['row'] + 1}, column {region['col'] + 1}). "
        "Answer based on local visual evidence in this crop, while remembering that normal local texture, "
        "edges, holes, shadows, or product structure are not necessarily defects."
    )


def word_probabilities(logits, candidate_ids, word_to_ids):
    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    denom_ids = torch.tensor(candidate_ids, dtype=torch.long, device=probs.device)
    denom = probs[denom_ids].sum().clamp_min(1e-12)
    out = {}
    for word, ids in word_to_ids.items():
        if not ids:
            raw = 0.0
        else:
            token_ids = torch.tensor(ids, dtype=torch.long, device=probs.device)
            raw = float(probs[token_ids].sum().item())
        out[word] = {"raw_prob": raw, "candidate_norm_prob": raw / float(denom.item())}
    return out


def clean_token_label(text):
    text = text.replace("\u0120", " ").replace("\u2581", " ").strip()
    text = text.replace("</s>", "").replace("<|endoftext|>", "").strip()
    text = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text)
    return text.lower()


def is_candidate_token(text):
    if not text:
        return False
    if len(text) < 3:
        return False
    if text in STOPWORDS:
        return False
    if not re.fullmatch(r"[a-z][a-z\-]*", text):
        return False
    return True


def full_image_global_top_candidates(processor, model, tokenizer, record, image, question, prefix, args):
    inputs, prompt_text = prepare_inputs(
        processor,
        model,
        image,
        record["image"],
        question,
        prefix,
        args.min_pixels,
        args.max_pixels,
    )
    with torch.inference_mode():
        logits = model(**inputs, return_dict=True, use_cache=False).logits[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    topk = min(int(args.global_candidate_top_pool), probs.numel())
    values, indices = torch.topk(probs, k=topk)
    words = []
    word_to_ids = {}
    scan_rows = []
    input_ids = set(int(x) for x in inputs["input_ids"].detach().cpu().flatten().tolist())
    for rank, (value, token_id_tensor) in enumerate(zip(values.detach().cpu(), indices.detach().cpu()), start=1):
        token_id = int(token_id_tensor.item())
        raw_label = token_label(tokenizer, token_id)
        candidate = clean_token_label(raw_label)
        accepted = is_candidate_token(candidate) and token_id not in input_ids and candidate not in word_to_ids
        scan_rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": raw_label,
                "candidate": candidate,
                "prob": float(value.item()),
                "accepted": int(accepted),
            }
        )
        if accepted:
            words.append(candidate)
            word_to_ids[candidate] = [token_id]
        if len(words) >= args.num_candidate_nouns:
            break
    candidate_ids = [ids[0] for ids in word_to_ids.values()]
    return words, candidate_ids, word_to_ids, scan_rows, prompt_text


def region_rows(processor, model, record, region, question, prefix, layers, candidate_ids, word_to_ids, args):
    region_question = build_patch_question(question, region, args.grid, args.patch_context)
    inputs, prompt_text = prepare_inputs(
        processor,
        model,
        region["image"],
        record["image"],
        region_question,
        prefix,
        args.min_pixels,
        args.max_pixels,
    )
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    logits_by_layer = logits_by_layer_from_hidden_states(model, outputs.hidden_states, layers)

    rows = []
    x0, y0, x1, y1 = region["box"]
    for layer in layers:
        if layer not in logits_by_layer:
            continue
        probs = word_probabilities(logits_by_layer[layer], candidate_ids, word_to_ids)
        for word, vals in probs.items():
            rows.append(
                {
                    "image": record["image"],
                    "image_id": record["image_id"],
                    "category": record["category"],
                    "anomaly_type": record["anomaly_type"],
                    "region": region["region"],
                    "kind": region["kind"],
                    "patch_index": region["patch_index"],
                    "patch_total": region["patch_total"],
                    "row": region["row"],
                    "col": region["col"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "layer": int(layer),
                    "candidate_noun": word,
                    "raw_prob": vals["raw_prob"],
                    "candidate_norm_prob": vals["candidate_norm_prob"],
                    "region_question": region_question,
                }
            )

    final_probs = word_probabilities(outputs.logits[:, -1, :], candidate_ids, word_to_ids)
    final_rows = []
    for word, vals in final_probs.items():
        final_rows.append(
            {
                "image": record["image"],
                "image_id": record["image_id"],
                "category": record["category"],
                "anomaly_type": record["anomaly_type"],
                "region": region["region"],
                "kind": region["kind"],
                "patch_index": region["patch_index"],
                "patch_total": region["patch_total"],
                "candidate_noun": word,
                "raw_prob": vals["raw_prob"],
                "candidate_norm_prob": vals["candidate_norm_prob"],
            }
        )
    return rows, final_rows, prompt_text


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_grid_image(image, regions, output_path):
    grid_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(grid_image, "RGBA")
    for r in regions:
        if r["kind"] != "patch":
            continue
        x0, y0, x1, y1 = r["box"]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 210), width=2)
        label = f"{r['row']},{r['col']}"
        draw.rectangle((x0, y0, x0 + 44, y0 + 18), fill=(0, 0, 0, 150))
        draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255, 255))
    grid_image.save(output_path)


def plot_full_curves(rows, output_png, prob_key):
    full_rows = [r for r in rows if r["kind"] == "full"]
    by_word = {}
    for row in full_rows:
        by_word.setdefault(row["candidate_noun"], []).append(row)
    plt.figure(figsize=(10, 5.8))
    for word, word_rows in sorted(by_word.items()):
        word_rows = sorted(word_rows, key=lambda r: r["layer"])
        plt.plot(
            [r["layer"] for r in word_rows],
            [r[prob_key] for r in word_rows],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=word,
        )
    plt.xlabel("Layer")
    plt.ylabel(prob_key)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_patch_heatmap(rows, word, layer, grid, output_png, prob_key):
    vals = [[float("nan") for _ in range(grid)] for _ in range(grid)]
    for row in rows:
        if row["kind"] != "patch":
            continue
        if row["candidate_noun"] != word or int(row["layer"]) != int(layer):
            continue
        vals[int(row["row"])][int(row["col"])] = float(row[prob_key])
    plt.figure(figsize=(6.4, 5.6))
    im = plt.imshow(vals, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=prob_key)
    plt.title(f"{word} probability, layer {layer}")
    plt.xticks(range(grid), [f"c{c}" for c in range(grid)], fontsize=7)
    plt.yticks(range(grid), [f"r{r}" for r in range(grid)], fontsize=7)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_layer_contact_sheet(rows, words, layer, grid, output_png, prob_key):
    cols = 5
    rows_n = (len(words) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.0, rows_n * 2.8), squeeze=False)
    for idx, word in enumerate(words):
        ax = axes[idx // cols][idx % cols]
        vals = [[float("nan") for _ in range(grid)] for _ in range(grid)]
        for row in rows:
            if row["kind"] == "patch" and row["candidate_noun"] == word and int(row["layer"]) == int(layer):
                vals[int(row["row"])][int(row["col"])] = float(row[prob_key])
        ax.imshow(vals, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(word, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(len(words), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"Candidate noun patch probabilities, layer {layer}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--question-template",
        default=(
            "Decide whether the main object in this image is normal or anomalous. "
            "The object category is {category}. "
            "Category-specific anomaly description for this image: {anomaly_prompt}"
        ),
    )
    parser.add_argument("--prefix", default="The condition label is")
    parser.add_argument(
        "--candidate-nouns",
        default="",
        help="Optional fixed comma-separated nouns. If omitted, nouns are extracted per category from anomaly_prompt.json.",
    )
    parser.add_argument(
        "--candidate-source",
        default="global_top",
        choices=["global_top", "prompt_json", "fixed", "merged"],
        help="global_top first selects candidate nouns from the full-image next-token distribution.",
    )
    parser.add_argument("--num-candidate-nouns", type=int, default=10)
    parser.add_argument("--global-candidate-top-pool", type=int, default=500)
    parser.add_argument("--layers", default="1:33")
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--overlap", type=float, default=0.05)
    parser.add_argument("--patch-context", action="store_true")
    parser.add_argument("--prob-key", default="candidate_norm_prob", choices=["candidate_norm_prob", "raw_prob"])
    parser.add_argument("--heatmap-layers", default="", help="Default: same as --layers.")
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow-attn-fallback", action="store_true")
    args = parser.parse_args()

    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration is not available from transformers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_range(args.layers)
    heatmap_layers = parse_int_range(args.heatmap_layers) if args.heatmap_layers else layers
    fixed_candidate_words = parse_csv_words(args.candidate_nouns)

    anomaly_records = collect_visa_anomalies(args.visa_root)
    sampled = sample_records(anomaly_records, args.num_images, args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    category_prompt_cache = {}

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": True,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    except Exception as exc:
        if not args.allow_attn_fallback or args.attn_implementation != "flash_attention_2":
            raise
        print(f"[WARN] flash_attention_2 load failed: {exc}")
        print("[WARN] Falling back to sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    manifest = {
        "visa_root": args.visa_root,
        "model_path": args.model_path,
        "question_template": args.question_template,
        "prefix": args.prefix,
        "layers": layers,
        "heatmap_layers": heatmap_layers,
        "grid": args.grid,
        "overlap": args.overlap,
        "candidate_source": args.candidate_source,
        "fixed_candidate_words": fixed_candidate_words,
        "num_candidate_nouns": args.num_candidate_nouns,
        "global_candidate_top_pool": args.global_candidate_top_pool,
        "sampled_images": sampled,
    }
    (out_dir / "0-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "0-sampled_anomaly_images.csv", sampled)

    all_patch_rows = []
    all_final_rows = []
    for idx, record in enumerate(sampled, start=1):
        image = Image.open(record["image"]).convert("RGB")
        image_dir = out_dir / f"{idx:02d}-{record['category']}-{record['anomaly_type']}-{record['image_id']}"
        image_dir.mkdir(parents=True, exist_ok=True)
        record_question = build_record_question(record, args.question_template)
        global_scan_rows = []
        global_prompt_text = ""
        if args.candidate_source == "global_top":
            candidate_words, candidate_ids, word_to_ids, global_scan_rows, global_prompt_text = full_image_global_top_candidates(
                processor,
                model,
                tokenizer,
                record,
                image,
                record_question,
                args.prefix,
                args,
            )
        else:
            prompt_source_texts = [record.get("anomaly_prompt_text", "")]
            if not prompt_source_texts[0]:
                prompt_source_texts = record.get("category_anomaly_prompts", [])
            prompt_words = candidate_words_from_prompts(prompt_source_texts, args.num_candidate_nouns)
            if args.candidate_source == "fixed":
                candidate_words = fixed_candidate_words
            elif args.candidate_source == "merged":
                candidate_words = []
                for word in fixed_candidate_words + prompt_words:
                    if word not in candidate_words:
                        candidate_words.append(word)
                candidate_words = candidate_words[: args.num_candidate_nouns]
            else:
                candidate_words = prompt_words
            candidate_words = candidate_words[: args.num_candidate_nouns]
            candidate_ids, word_to_ids = candidate_token_ids(tokenizer, candidate_words)
        if not candidate_words:
            candidate_words = fixed_candidate_words or [
                "scratch",
                "crack",
                "damage",
                "stain",
                "hole",
                "missing",
                "broken",
                "contamination",
                "deformation",
                "defect",
            ]
            candidate_words = candidate_words[: args.num_candidate_nouns]
            candidate_ids, word_to_ids = candidate_token_ids(tokenizer, candidate_words)
        category_prompt_cache[record["category"]] = {
            "anomaly_prompt_path": record.get("anomaly_prompt_path", ""),
            "category_anomaly_prompts": record.get("category_anomaly_prompts", []),
            "candidate_words": candidate_words,
            "candidate_tokenization": {
                word: [{"token_id": tid, "token": token_label(tokenizer, tid)} for tid in ids]
                for word, ids in word_to_ids.items()
            },
        }
        image.save(image_dir / "1-original_full_image.jpg")
        regions = make_regions(image, args.grid, args.overlap)
        save_grid_image(image, regions, image_dir / "2-patch_grid_indices.jpg")
        if global_scan_rows:
            write_csv(image_dir / "2-global_next_token_candidate_scan.csv", global_scan_rows)

        image_rows = []
        image_final_rows = []
        prompt_text_example = None
        for region in regions:
            rows, final_rows, prompt_text = region_rows(
                processor,
                model,
                record,
                region,
                record_question,
                args.prefix,
                layers,
                candidate_ids,
                word_to_ids,
                args,
            )
            if prompt_text_example is None:
                prompt_text_example = prompt_text
            image_rows.extend(rows)
            image_final_rows.extend(final_rows)
        write_csv(image_dir / "3-layer_patch_candidate_noun_probs.csv", image_rows)
        write_csv(image_dir / "4-final_candidate_noun_probs.csv", image_final_rows)
        plot_full_curves(image_rows, image_dir / "5-full_image_candidate_noun_probs_by_layer.png", args.prob_key)

        heatmap_root = image_dir / "6-token_patch_heatmaps_by_layer"
        heatmap_root.mkdir(parents=True, exist_ok=True)
        for layer in heatmap_layers:
            plot_layer_contact_sheet(
                image_rows,
                candidate_words,
                layer,
                args.grid,
                heatmap_root / f"layer_{layer:02d}_all_candidates.png",
                args.prob_key,
            )
            for word in candidate_words:
                word_dir = heatmap_root / word
                word_dir.mkdir(parents=True, exist_ok=True)
                plot_patch_heatmap(
                    image_rows,
                    word,
                    layer,
                    args.grid,
                    word_dir / f"layer_{layer:02d}.png",
                    args.prob_key,
                )

        per_image_summary = {
            **record,
            "question": record_question,
            "candidate_words": candidate_words,
            "candidate_tokenization": {
                word: [{"token_id": tid, "token": token_label(tokenizer, tid)} for tid in ids]
                for word, ids in word_to_ids.items()
            },
            "prompt_text_example": prompt_text_example,
            "global_candidate_prompt_text": global_prompt_text,
            "global_candidate_scan": str(image_dir / "2-global_next_token_candidate_scan.csv") if global_scan_rows else "",
            "files": {
                "layer_probs": str(image_dir / "3-layer_patch_candidate_noun_probs.csv"),
                "final_probs": str(image_dir / "4-final_candidate_noun_probs.csv"),
                "full_curves": str(image_dir / "5-full_image_candidate_noun_probs_by_layer.png"),
                "heatmaps": str(heatmap_root),
            },
        }
        (image_dir / "7-summary.json").write_text(json.dumps(per_image_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        all_patch_rows.extend(image_rows)
        all_final_rows.extend(image_final_rows)
        print(f"[{idx}/{len(sampled)}] saved {image_dir}")

    manifest["category_prompt_cache"] = category_prompt_cache
    (out_dir / "0-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "all-layer_patch_candidate_noun_probs.csv", all_patch_rows)
    write_csv(out_dir / "all-final_candidate_noun_probs.csv", all_final_rows)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
