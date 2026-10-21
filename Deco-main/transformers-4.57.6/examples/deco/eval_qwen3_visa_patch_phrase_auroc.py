import argparse
import csv
import json
import math
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
    get_norm_and_lm_head,
    parse_int_range,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
GENERIC_DEFECT_PHRASES = [
    "visible defect",
    "surface defect",
    "abnormal damage",
    "scratch",
    "crack",
    "stain",
    "hole",
    "broken part",
    "missing part",
    "contamination",
]
GENERIC_NORMAL_PHRASES = [
    "normal appearance",
    "intact object",
    "good condition",
    "clean surface",
    "regular appearance",
    "undamaged",
]
GENERIC_NUISANCE_PHRASES = [
    "edge",
    "corner",
    "shadow",
    "highlight",
    "reflection",
    "texture",
    "background",
    "crease",
    "fold",
    "normal texture",
    "lighting variation",
    "viewpoint",
]

# Category-level VisA taxonomy. These phrases are shared by every image in a
# category, so they provide useful domain context without leaking per-image
# anomaly labels or requiring the non-standard anomaly_prompt.json files.
VISA_CATEGORY_DEFECT_PHRASES = {
    "candle": [
        "missing wax",
        "foreign particles",
        "different color spot",
        "damaged packaging",
        "abnormal wick",
        "extra wax",
        "melted wax",
    ],
    "capsules": ["bubble", "discoloration", "scratch", "leak", "misshapen capsule"],
    "cashew": ["burnt area", "edge breakage", "middle breakage", "color spot", "small hole", "scratch", "stuck together"],
    "chewinggum": ["missing gum", "missing corner", "scratch", "color spot", "small crack"],
    "fryum": ["burnt area", "edge breakage", "middle breakage", "color spot", "stuck together", "scratch"],
    "macaroni1": ["chipped edge", "chipped corner", "color spot", "small crack", "middle breakage", "scratch"],
    "macaroni2": ["middle breakage", "color spot", "chipped edge", "small crack", "scratch"],
    "pipe_fryum": ["burnt area", "edge breakage", "color spot", "scratch", "stuck together", "small crack"],
    "pcb1": ["bent component", "melted component", "missing component", "scratch"],
    "pcb2": ["bent component", "melted component", "missing component", "scratch"],
    "pcb3": ["bent component", "melted component", "missing component", "scratch"],
    "pcb4": ["burnt component", "scratch", "dirt", "damage", "extra component", "missing component", "misplaced component"],
}


def display_category(category):
    return str(category or "object").replace("_", " ").replace("-", " ")


def clean_phrase(phrase):
    phrase = re.sub(r"\s+", " ", str(phrase).strip().lower())
    phrase = phrase.replace("particals", "particles")
    return phrase


def is_bad_phrase(phrase):
    phrase = clean_phrase(phrase)
    if not phrase:
        return True
    banned = {
        "normal defect",
        "normal defects",
        "intact defect",
        "good defect",
        "clean defect",
        "undamaged defect",
    }
    if phrase in banned:
        return True
    if re.search(r"\b(normal|intact|good|clean|regular|undamaged)\b.*\bdefect\b", phrase):
        return True
    return False


def build_phrase_bank(phrases, max_phrases):
    out = []
    for phrase in phrases:
        phrase = clean_phrase(phrase)
        if is_bad_phrase(phrase):
            continue
        if phrase and phrase not in out:
            out.append(phrase)
        if len(out) >= max_phrases:
            return out
    return out


def resolve_mask_path(category_dir, image_path, label):
    if label == 0:
        return ""
    candidates = []
    image_path = Path(image_path)
    try:
        rel = image_path.relative_to(category_dir / "Data" / "Images")
        candidates.append(category_dir / "Data" / "Masks" / rel)
        candidates.append((category_dir / "Data" / "Masks" / rel).with_suffix(".png"))
        candidates.append((category_dir / "Data" / "Masks" / rel).with_suffix(".JPG"))
    except ValueError:
        pass
    for cand in candidates:
        if cand and cand.exists():
            return str(cand)
    return ""


def build_category_bank(category_dir):
    category = category_dir.name
    cat = display_category(category)
    category_phrases = VISA_CATEGORY_DEFECT_PHRASES.get(category, [])
    defect_bank = build_phrase_bank(category_phrases + GENERIC_DEFECT_PHRASES, max_phrases=32)
    normal_bank = build_phrase_bank(
        [f"normal {cat}", f"intact {cat}", f"undamaged {cat}"] + GENERIC_NORMAL_PHRASES,
        max_phrases=16,
    )
    nuisance_bank = build_phrase_bank(GENERIC_NUISANCE_PHRASES, max_phrases=16)
    return None, [], {
        "source": "builtin_visa_category_taxonomy",
        "category": category,
        "anomaly_prompt_path": "",
        "defect_bank": defect_bank,
        "normal_bank": normal_bank,
        "nuisance_bank": nuisance_bank,
    }


def collect_visa_samples(
    visa_root,
    categories=None,
    max_images_per_category=None,
    sample_normal=None,
    sample_anomaly=None,
    seed=42,
    split_csv=None,
    split_set="test",
):
    root = Path(visa_root)
    allowed = set([c.strip() for c in categories.split(",") if c.strip()]) if categories else None
    samples = []
    category_banks = {}
    if split_csv:
        split_path = Path(split_csv)
        if not split_path.exists():
            raise FileNotFoundError(f"Missing VisA split CSV: {split_path}")
        category_cache = {}
        wanted_splits = set([x.strip() for x in str(split_set).split(",") if x.strip()])
        with split_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = (row.get("object") or row.get("category") or "").strip()
                if not category:
                    continue
                if allowed and category not in allowed:
                    continue
                row_split = str(row.get("split", "")).strip()
                if wanted_splits and row_split not in wanted_splits:
                    continue
                category_dir = root / category
                if not category_dir.exists():
                    continue
                if category not in category_cache:
                    _, _, bank = build_category_bank(category_dir)
                    category_cache[category] = bank
                    category_banks[category] = bank
                image_rel = str(row.get("image", "")).strip()
                if not image_rel:
                    continue
                image_path = root / image_rel
                if not image_path.exists():
                    continue
                label_text = str(row.get("label", "")).strip().lower()
                label = 0 if label_text == "normal" else 1
                mask_rel = str(row.get("mask", "")).strip()
                mask_path = str(root / mask_rel) if mask_rel and (root / mask_rel).exists() else ""
                if not mask_path:
                    mask_path = resolve_mask_path(category_dir, image_path, label)
                samples.append(
                    {
                        "image": str(image_path),
                        "image_id": image_path.stem,
                        "category": category,
                        "label": int(label),
                        "label_name": "anomaly" if label else "normal",
                        "anomaly_prompt": "",
                        "anomaly_prompt_index": "",
                        "specific_anomaly_type": "normal" if label == 0 else "anomaly_unspecified",
                        "defect_bank_source": "builtin_visa_category_taxonomy",
                        "mask": mask_path,
                        "csv_split": row_split,
                        "csv_label": label_text,
                        "csv_image": image_rel,
                        "csv_mask": mask_rel,
                    }
                )
        if sample_normal is not None or sample_anomaly is not None:
            rng = random.Random(int(seed))
            normals = [x for x in samples if x["label"] == 0]
            anomalies = [x for x in samples if x["label"] == 1]
            if sample_normal is not None:
                normals = rng.sample(normals, min(int(sample_normal), len(normals)))
            if sample_anomaly is not None:
                anomalies = rng.sample(anomalies, min(int(sample_anomaly), len(anomalies)))
            samples = normals + anomalies
            rng.shuffle(samples)
        return samples, category_banks

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if allowed and category not in allowed:
            continue
        _, _, category_banks[category] = build_category_bank(category_dir)

        category_samples = []
        for split_name, label in [("Normal", 0), ("Anomaly", 1)]:
            image_root = category_dir / "Data" / "Images" / split_name
            if not image_root.exists():
                continue
            for image_path in sorted(image_root.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                mask_path = resolve_mask_path(category_dir, image_path, label)
                category_samples.append(
                    {
                        "image": str(image_path),
                        "image_id": image_path.stem,
                        "category": category,
                        "label": int(label),
                        "label_name": "anomaly" if label else "normal",
                        "anomaly_prompt": "",
                        "anomaly_prompt_index": "",
                        "specific_anomaly_type": "normal" if label == 0 else "anomaly_unspecified",
                        "defect_bank_source": "builtin_visa_category_taxonomy",
                        "mask": mask_path,
                    }
                )
        if max_images_per_category:
            normals = [x for x in category_samples if x["label"] == 0][: int(max_images_per_category)]
            anomalies = [x for x in category_samples if x["label"] == 1][: int(max_images_per_category)]
            category_samples = normals + anomalies
        samples.extend(category_samples)
    if sample_normal is not None or sample_anomaly is not None:
        rng = random.Random(int(seed))
        normals = [x for x in samples if x["label"] == 0]
        anomalies = [x for x in samples if x["label"] == 1]
        if sample_normal is not None:
            normals = rng.sample(normals, min(int(sample_normal), len(normals)))
        if sample_anomaly is not None:
            anomalies = rng.sample(anomalies, min(int(sample_anomaly), len(anomalies)))
        samples = normals + anomalies
        rng.shuffle(samples)
    return samples, category_banks


def make_regions(image, grid):
    width, height = image.size
    regions = []
    step_x = width / grid
    step_y = height / grid
    for row in range(grid):
        for col in range(grid):
            x0 = int(round(col * step_x))
            y0 = int(round(row * step_y))
            x1 = int(round((col + 1) * step_x))
            y1 = int(round((row + 1) * step_y))
            regions.append(
                {
                    "region": f"patch_r{row}_c{col}",
                    "patch_index": row * grid + col + 1,
                    "patch_total": grid * grid,
                    "row": row,
                    "col": col,
                    "box": (x0, y0, x1, y1),
                    "image": image.crop((x0, y0, x1, y1)),
                }
            )
    return regions


def build_question(record, template):
    category_phrases = VISA_CATEGORY_DEFECT_PHRASES.get(record["category"], [])
    anomaly_prompt = ", ".join(category_phrases[:8]) or "visible surface abnormality or damage"
    return template.format(
        category=display_category(record["category"]),
        anomaly_prompt=anomaly_prompt,
        label_name=record["label_name"],
    )


def build_patch_question(base_question, region, grid):
    return (
        f"{base_question}\n"
        f"You are seeing patch {region['patch_index']} of {region['patch_total']} from a {grid}x{grid} grid. "
        "Judge this local crop for visual defect evidence."
    )


def prepare_inputs(processor, model, image, source_image_path, question, prefix, phrase, min_pixels, max_pixels):
    messages = build_messages(source_image_path, question)
    base_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefix:
        base_text += " " + prefix.strip()
    phrase_text = " " + phrase.strip()
    full_text = base_text + phrase_text
    base_inputs = processor(
        text=[base_text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    full_inputs = processor(
        text=[full_text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    base_len = int(base_inputs["input_ids"].shape[1])
    full_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
    phrase_ids = full_inputs["input_ids"][0, base_len:].detach().cpu().tolist()
    return full_inputs, base_len, [int(x) for x in phrase_ids], base_text


def phrase_logprobs_by_layer_batched(processor, model, image, source_image_path, question, prefix, phrases, layers, args):
    if not phrases:
        return {}, ""
    _, base_len, _, base_text = prepare_inputs(
        processor,
        model,
        image,
        source_image_path,
        question,
        prefix,
        phrases[0],
        args.min_pixels,
        args.max_pixels,
    )
    messages = build_messages(source_image_path, question)
    raw_base_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefix:
        raw_base_text += " " + prefix.strip()
    texts = [raw_base_text + " " + phrase.strip() for phrase in phrases]
    images = [image.convert("RGB") for _ in phrases]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    norm, lm_head = get_norm_and_lm_head(model)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    per_phrase = {phrase: {} for phrase in phrases}
    for layer in layers:
        if layer < 0 or layer >= len(outputs.hidden_states):
            continue
        hidden = outputs.hidden_states[layer]
        for bidx, phrase in enumerate(phrases):
            if attention_mask is not None:
                nonpad = torch.where(attention_mask[bidx] > 0)[0]
                if nonpad.numel() == 0:
                    per_phrase[phrase][int(layer)] = -float("inf")
                    continue
                start = int(nonpad[0].item())
                actual_len = int(nonpad.numel())
            else:
                start = 0
                actual_len = int(input_ids.shape[1])
            phrase_len = max(0, actual_len - base_len)
            total = 0.0
            count = 0
            for offset in range(phrase_len):
                token_pos = start + base_len + offset
                pred_pos = token_pos - 1
                if pred_pos < 0 or token_pos >= input_ids.shape[1]:
                    continue
                token_id = int(input_ids[bidx, token_pos].item())
                h = hidden[bidx : bidx + 1, pred_pos, :]
                logits = lm_head(norm(h)).float()
                logp = F.log_softmax(logits, dim=-1)[0, token_id]
                total += float(logp.item())
                count += 1
            if count == 0:
                per_phrase[phrase][int(layer)] = -float("inf")
            elif args.phrase_reduction == "sum":
                per_phrase[phrase][int(layer)] = total
            else:
                per_phrase[phrase][int(layer)] = total / count
    return per_phrase, base_text


def aggregate_phrase_bank(processor, model, image, source_image_path, question, prefix, phrases, layers, args):
    per_phrase = {}
    prompt_text = ""
    batch_size = max(1, int(args.phrase_batch_size))
    for start in range(0, len(phrases), batch_size):
        chunk = phrases[start : start + batch_size]
        chunk_scores, prompt_text = phrase_logprobs_by_layer_batched(
            processor, model, image, source_image_path, question, prefix, chunk, layers, args
        )
        per_phrase.update(chunk_scores)
    by_layer = {}
    for layer in layers:
        vals = [per_phrase[p].get(layer, -float("inf")) for p in phrases]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            by_layer[layer] = -float("inf")
        elif args.phrase_bank_agg == "mean":
            by_layer[layer] = sum(finite) / len(finite)
        else:
            by_layer[layer] = max(finite)
    return by_layer, per_phrase, prompt_text


def mean_top_n(values, n):
    finite = sorted([float(v) for v in values if math.isfinite(float(v))], reverse=True)
    if not finite:
        return -float("inf")
    n = min(int(n), len(finite))
    return sum(finite[:n]) / n


def score_region(processor, model, image, source_image_path, question, prefix, defect_phrases, normal_phrases, nuisance_phrases, layers, args):
    defect_by_layer, defect_detail, prompt_text = aggregate_phrase_bank(
        processor, model, image, source_image_path, question, prefix, defect_phrases, layers, args
    )
    normal_by_layer, normal_detail, _ = aggregate_phrase_bank(
        processor, model, image, source_image_path, question, prefix, normal_phrases, layers, args
    )
    nuisance_by_layer, nuisance_detail, _ = aggregate_phrase_bank(
        processor, model, image, source_image_path, question, prefix, nuisance_phrases, layers, args
    )
    layer_rows = []
    layer_scores = []
    for layer in layers:
        defect_score = defect_by_layer.get(layer, -float("inf"))
        normal_score = normal_by_layer.get(layer, -float("inf"))
        nuisance_score = nuisance_by_layer.get(layer, -float("inf"))
        if getattr(args, "score_mode", "prob_margin") == "logprob_margin":
            score = defect_score - float(args.normal_lambda) * normal_score - float(args.nuisance_lambda) * nuisance_score
            defect_prob = normal_prob = nuisance_prob = None
        else:
            logits = torch.tensor([defect_score, normal_score, nuisance_score], dtype=torch.float32)
            probs = F.softmax(logits, dim=0)
            defect_prob = float(probs[0].item())
            normal_prob = float(probs[1].item())
            nuisance_prob = float(probs[2].item())
            score = defect_prob - float(args.normal_lambda) * normal_prob - float(args.nuisance_lambda) * nuisance_prob
        layer_scores.append(score)
        layer_rows.append(
            {
                "layer": int(layer),
                "defect_score": defect_score,
                "normal_score": normal_score,
                "nuisance_score": nuisance_score,
                "defect_prob": defect_prob,
                "normal_prob": normal_prob,
                "nuisance_prob": nuisance_prob,
                "patch_score_before_contrast": score,
            }
        )
    region_score = mean_top_n(layer_scores, 3)
    return region_score, layer_rows, defect_detail, normal_detail, nuisance_detail, prompt_text


def topk_count(mode, n):
    mode = str(mode).lower()
    if mode.startswith("top") and mode.endswith("%"):
        pct = float(mode[3:-1]) / 100.0
        return max(1, int(math.ceil(n * pct)))
    return max(1, min(int(mode), n))


def patch_map_to_pixel(image_size, patch_rows, grid, key="patch_score"):
    width, height = image_size
    score = torch.zeros((height, width), dtype=torch.float32)
    count = torch.zeros((height, width), dtype=torch.float32)
    for row in patch_rows:
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        score[y0:y1, x0:x1] += float(row[key])
        count[y0:y1, x0:x1] += 1
    count = torch.clamp(count, min=1)
    return (score / count).numpy()


def smooth_patch_scores(patch_rows, grid, source_key="patch_score_raw", target_key="patch_score"):
    vals = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for row in patch_rows:
        vals[int(row["row"])][int(row["col"])] = float(row[source_key])
    smoothed = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for r in range(grid):
        for c in range(grid):
            local = []
            for rr in range(max(0, r - 1), min(grid, r + 2)):
                for cc in range(max(0, c - 1), min(grid, c + 2)):
                    local.append(vals[rr][cc])
            smoothed[r][c] = sum(local) / len(local)
    for row in patch_rows:
        row[target_key] = smoothed[int(row["row"])][int(row["col"])]
    return vals, smoothed


def load_mask(mask_path, size):
    if not mask_path or not Path(mask_path).exists():
        return None
    mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)
    return [(1 if p > 0 else 0) for p in mask.getdata()]


def resize_score_values(pixel_map, size):
    if size is None:
        return [float(x) for x in pixel_map.reshape(-1).tolist()]
    tensor = torch.tensor(pixel_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(int(size[1]), int(size[0])), mode="bilinear", align_corners=False)
    return [float(x) for x in resized.squeeze(0).squeeze(0).reshape(-1).tolist()]


def binary_auc(labels, scores):
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    pos = sum(y == 1 for _, y in pairs)
    neg = sum(y == 0 for _, y in pairs)
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def accuracy(labels, scores, threshold):
    if not labels:
        return None
    correct = 0
    for y, s in zip(labels, scores):
        pred = 1 if float(s) >= float(threshold) else 0
        correct += int(pred == int(y))
    return correct / len(labels)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_patch_score_map(patch_rows, grid, output_png):
    vals = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for row in patch_rows:
        vals[int(row["row"])][int(row["col"])] = float(row["patch_score"])
    plt.figure(figsize=(7, 6))
    im = plt.imshow(vals, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="patch_score")
    plt.xticks(range(grid), [f"c{x}" for x in range(grid)], fontsize=7)
    plt.yticks(range(grid), [f"r{x}" for x in range(grid)], fontsize=7)
    plt.title("14x14 patch score map")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def grid_from_patch_rows(patch_rows, grid, key):
    vals = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for row in patch_rows:
        vals[int(row["row"])][int(row["col"])] = float(row.get(key, 0.0))
    return vals


def save_map_contact_sheet(maps, output_png, title, cmap="coolwarm"):
    if not maps:
        return
    cols = min(4, len(maps))
    rows = int(math.ceil(len(maps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0), squeeze=False)
    finite_vals = []
    for _, vals in maps:
        for row in vals:
            for val in row:
                if math.isfinite(float(val)):
                    finite_vals.append(float(val))
    if finite_vals:
        vmax = max(abs(min(finite_vals)), abs(max(finite_vals)))
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0
    for idx, (name, vals) in enumerate(maps):
        ax = axes[idx // cols][idx % cols]
        im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(len(maps), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def save_phrase_layer_contact_sheet(phrase_maps, phrases, layer, grid, output_png, title):
    selected = [(phrase, phrase_maps.get((phrase, int(layer)))) for phrase in phrases]
    selected = [(name, vals) for name, vals in selected if vals is not None]
    if not selected:
        return
    cols = min(5, len(selected))
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.8), squeeze=False)
    finite_vals = []
    for _, vals in selected:
        for row in vals:
            finite_vals.extend([float(x) for x in row if math.isfinite(float(x))])
    vmin = min(finite_vals) if finite_vals else -20.0
    vmax = max(finite_vals) if finite_vals else 0.0
    for idx, (phrase, vals) in enumerate(selected):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(vals, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(phrase[:32], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(len(selected), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"{title}, layer {layer}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def save_full_layer_plot(layer_rows, output_png):
    if not layer_rows:
        return
    layer_rows = sorted(layer_rows, key=lambda x: int(x["layer"]))
    plt.figure(figsize=(8, 4.8))
    for key, label in [
        ("defect_score", "defect"),
        ("normal_score", "normal"),
        ("nuisance_score", "nuisance"),
        ("patch_score_before_contrast", "combined"),
    ]:
        plt.plot([r["layer"] for r in layer_rows], [r[key] for r in layer_rows], marker="o", label=label)
    plt.xlabel("Layer")
    plt.ylabel("mean phrase logprob score")
    plt.title("Full-image layer scores")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def patch_rows_to_grid(patch_rows, grid, key="patch_score"):
    vals = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for row in patch_rows:
        vals[int(row["row"])][int(row["col"])] = float(row[key])
    return vals


def save_pixel_heatmap(image, pixel_map, output_png):
    plt.figure(figsize=(7, 6))
    plt.imshow(image.convert("RGB"))
    plt.imshow(pixel_map, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-images-per-category", type=int, default=None)
    parser.add_argument("--sample-normal", type=int, default=None)
    parser.add_argument("--sample-anomaly", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--layers", default="20:29")
    parser.add_argument("--topk-patches", default="top5%", help="1/3/5/10/top3%/top5%")
    parser.add_argument("--normal-lambda", type=float, default=1.0)
    parser.add_argument("--nuisance-lambda", type=float, default=0.5)
    parser.add_argument("--patch-full-contrast", action="store_true")
    parser.add_argument("--full-score-weight", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-defect-phrases", type=int, default=12)
    parser.add_argument("--max-normal-phrases", type=int, default=8)
    parser.add_argument("--max-nuisance-phrases", type=int, default=8)
    parser.add_argument("--phrase-batch-size", type=int, default=4, help="Number of phrases evaluated in one model forward. Lower this if CUDA/flash-attn is unstable.")
    parser.add_argument("--phrase-bank-agg", default="max", choices=["max", "mean"])
    parser.add_argument("--phrase-reduction", default="mean", choices=["mean", "sum"])
    parser.add_argument("--score-mode", default="prob_margin", choices=["prob_margin", "logprob_margin"])
    parser.add_argument("--pixel-auroc-size", type=int, default=256, help="Resize masks/scores to this square size for p-AUROC. Use 0 for full resolution.")
    parser.add_argument("--heatmap-layers", default="", help="Layers for phrase contact-sheet heatmaps. Default: last evaluated layer.")
    parser.add_argument("--save-csv-details", action="store_true", help="Save per-patch per-layer CSV files. Disabled by default because it is slow.")
    parser.add_argument(
        "--question-template",
        default=(
            "Decide whether the main object in this image is normal or anomalous. "
            "The object category is {category}. "
            "Known category-specific defect descriptions include: {anomaly_prompt}."
        ),
    )
    parser.add_argument("--prefix", default="The visual condition is")
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    args = parser.parse_args()

    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration is not available from transformers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_int_range(args.layers)
    heatmap_layers = parse_int_range(args.heatmap_layers) if args.heatmap_layers else [layers[-1]]
    samples, category_banks = collect_visa_samples(
        args.visa_root,
        args.categories,
        args.max_images_per_category,
        args.sample_normal,
        args.sample_anomaly,
        args.seed,
    )
    if not samples:
        raise RuntimeError("No VisA samples found.")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
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
        "layers": layers,
        "heatmap_layers": heatmap_layers,
        "grid": args.grid,
        "topk_patches": args.topk_patches,
        "normal_lambda": args.normal_lambda,
        "nuisance_lambda": args.nuisance_lambda,
        "patch_full_contrast": args.patch_full_contrast,
        "full_score_weight": args.full_score_weight,
        "patch_score_smoothing": "3x3_mean",
        "aggregation": "mean_top3_layer",
        "category_banks": category_banks,
        "num_samples": len(samples),
    }
    save_json(out_dir / "0-manifest.json", manifest)

    image_rows = []
    all_pixel_labels = []
    all_pixel_scores = []
    pixel_by_category = {}
    patch_rows_all = []

    for idx, record in enumerate(samples, start=1):
        category_bank = category_banks[record["category"]]
        defect_phrases = category_bank["defect_bank"][: args.max_defect_phrases]
        normal_phrases = category_bank["normal_bank"][: args.max_normal_phrases]
        nuisance_phrases = category_bank["nuisance_bank"][: args.max_nuisance_phrases]
        image = Image.open(record["image"]).convert("RGB")
        question = build_question(record, args.question_template)
        sample_dir = out_dir / f"{idx:05d}-{record['category']}-{record['label_name']}-{record['image_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        full_score, full_layer_rows, _, _, _, prompt_text = score_region(
            processor,
            model,
            image,
            record["image"],
            question,
            args.prefix,
            defect_phrases,
            normal_phrases,
            nuisance_phrases,
            layers,
            args,
        )

        patch_rows = []
        patch_layer_debug_rows = []
        defect_phrase_maps = {}
        normal_phrase_maps = {}
        nuisance_phrase_maps = {}
        for region in make_regions(image, args.grid):
            patch_question = build_patch_question(question, region, args.grid)
            region_score, layer_rows, defect_detail, normal_detail, nuisance_detail, _ = score_region(
                processor,
                model,
                region["image"],
                record["image"],
                patch_question,
                args.prefix,
                defect_phrases,
                normal_phrases,
                nuisance_phrases,
                layers,
                args,
            )
            score_before_contrast = region_score
            if args.patch_full_contrast:
                region_score = region_score - full_score
            defect_mean_top3 = mean_top_n([r["defect_score"] for r in layer_rows], 3)
            normal_mean_top3 = mean_top_n([r["normal_score"] for r in layer_rows], 3)
            nuisance_mean_top3 = mean_top_n([r["nuisance_score"] for r in layer_rows], 3)
            x0, y0, x1, y1 = region["box"]
            patch_row = {
                **record,
                "row": region["row"],
                "col": region["col"],
                "patch_index": region["patch_index"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "patch_score_raw": region_score,
                "patch_score": region_score,
                "patch_score_before_contrast": score_before_contrast,
                "full_image_score": full_score,
                "defect_mean_top3_layer": defect_mean_top3,
                "normal_mean_top3_layer": normal_mean_top3,
                "nuisance_mean_top3_layer": nuisance_mean_top3,
            }
            patch_rows.append(patch_row)
            patch_rows_all.append(patch_row)
            for lr in layer_rows:
                lr.update(
                    {
                        "image": record["image"],
                        "image_id": record["image_id"],
                        "category": record["category"],
                        "label": record["label"],
                        "row": region["row"],
                        "col": region["col"],
                        "patch_index": region["patch_index"],
                        "patch_score": region_score,
                    }
                )
                patch_layer_debug_rows.append(lr)
            if args.save_csv_details:
                write_csv(sample_dir / f"layer_scores_patch_r{region['row']}_c{region['col']}.csv", layer_rows)
            if args.save_visualizations:
                for layer in heatmap_layers:
                    for phrase in defect_phrases:
                        defect_phrase_maps.setdefault((phrase, int(layer)), [[0.0 for _ in range(args.grid)] for _ in range(args.grid)])
                        defect_phrase_maps[(phrase, int(layer))][region["row"]][region["col"]] = defect_detail.get(phrase, {}).get(int(layer), -float("inf"))
                    for phrase in normal_phrases:
                        normal_phrase_maps.setdefault((phrase, int(layer)), [[0.0 for _ in range(args.grid)] for _ in range(args.grid)])
                        normal_phrase_maps[(phrase, int(layer))][region["row"]][region["col"]] = normal_detail.get(phrase, {}).get(int(layer), -float("inf"))
                    for phrase in nuisance_phrases:
                        nuisance_phrase_maps.setdefault((phrase, int(layer)), [[0.0 for _ in range(args.grid)] for _ in range(args.grid)])
                        nuisance_phrase_maps[(phrase, int(layer))][region["row"]][region["col"]] = nuisance_detail.get(phrase, {}).get(int(layer), -float("inf"))

        raw_patch_score_map, smoothed_patch_score_map = smooth_patch_scores(patch_rows, args.grid)
        k = topk_count(args.topk_patches, len(patch_rows))
        ranked_scores = sorted([float(x["patch_score"]) for x in patch_rows], reverse=True)
        patch_image_score = sum(ranked_scores[:k]) / k
        image_score = patch_image_score + float(args.full_score_weight) * full_score
        image_pred = 1 if image_score >= args.threshold else 0
        pixel_map = patch_map_to_pixel(image.size, patch_rows, args.grid)
        pixel_map_path = sample_dir / "pixel_score_map.npy"
        try:
            import numpy as np

            np.save(pixel_map_path, pixel_map)
        except Exception:
            pixel_map_path = ""

        eval_size = None if int(args.pixel_auroc_size) <= 0 else (int(args.pixel_auroc_size), int(args.pixel_auroc_size))
        mask_labels = load_mask(record["mask"], eval_size or image.size)
        if mask_labels is not None:
            all_pixel_labels.extend(mask_labels)
            pixel_scores = resize_score_values(pixel_map, eval_size)
            all_pixel_scores.extend(pixel_scores)
            bucket = pixel_by_category.setdefault(record["category"], {"labels": [], "scores": []})
            bucket["labels"].extend(mask_labels)
            bucket["scores"].extend(pixel_scores)

        write_csv(sample_dir / "patch_scores.csv", patch_rows)
        save_json(sample_dir / "patch_score_map_raw.json", raw_patch_score_map)
        save_json(sample_dir / "patch_score_map.json", smoothed_patch_score_map)
        save_json(
            sample_dir / "summary.json",
            {
                **record,
                "image_score": image_score,
                "image_pred": image_pred,
                "full_image_score": full_score,
                "patch_image_score": patch_image_score,
                "image_score_formula": "topk_mean(smoothed_patch_score) + full_score_weight * full_image_score",
                "topk_patch_count": k,
                "question": question,
                "prompt_text": prompt_text,
                "defect_phrases": defect_phrases,
                "normal_phrases": normal_phrases,
                "nuisance_phrases": nuisance_phrases,
                "patch_score_map_raw": str(sample_dir / "patch_score_map_raw.json"),
                "patch_score_map": str(sample_dir / "patch_score_map.json"),
                "pixel_score_map": str(pixel_map_path),
            },
        )
        if args.save_visualizations:
            save_full_layer_plot(full_layer_rows, sample_dir / "1-full_image_layer_scores.png")
            save_map_contact_sheet(
                [
                    ("defect mean_top3", grid_from_patch_rows(patch_rows, args.grid, "defect_mean_top3_layer")),
                    ("normal mean_top3", grid_from_patch_rows(patch_rows, args.grid, "normal_mean_top3_layer")),
                    ("nuisance mean_top3", grid_from_patch_rows(patch_rows, args.grid, "nuisance_mean_top3_layer")),
                    ("raw patch score", raw_patch_score_map),
                    ("smoothed patch score", smoothed_patch_score_map),
                ],
                sample_dir / "2-process_score_maps.png",
                "Intermediate patch scoring maps",
            )
            save_patch_score_map(patch_rows, args.grid, sample_dir / "3-final_patch_score_map.png")
            save_pixel_heatmap(image, pixel_map, sample_dir / "4-pixel_score_heatmap_overlay.png")
            for layer in heatmap_layers:
                save_phrase_layer_contact_sheet(
                    defect_phrase_maps,
                    defect_phrases,
                    layer,
                    args.grid,
                    sample_dir / f"5-defect_phrase_heatmaps_layer_{int(layer):02d}.png",
                    "Defect phrase patch logprob",
                )
                save_phrase_layer_contact_sheet(
                    normal_phrase_maps,
                    normal_phrases,
                    layer,
                    args.grid,
                    sample_dir / f"6-normal_phrase_heatmaps_layer_{int(layer):02d}.png",
                    "Normal phrase patch logprob",
                )
                save_phrase_layer_contact_sheet(
                    nuisance_phrase_maps,
                    nuisance_phrases,
                    layer,
                    args.grid,
                    sample_dir / f"7-nuisance_phrase_heatmaps_layer_{int(layer):02d}.png",
                    "Nuisance phrase patch logprob",
                )
            if args.save_csv_details:
                write_csv(sample_dir / "patch_layer_debug_scores.csv", patch_layer_debug_rows)

        image_rows.append(
            {
                **record,
                "image_score": image_score,
                "image_pred": image_pred,
                "correct": int(image_pred == int(record["label"])),
                "full_image_score": full_score,
                "patch_image_score": patch_image_score,
                "topk_patch_count": k,
                "patch_score_map": str(sample_dir / "patch_score_map.json"),
                "patch_score_map_raw": str(sample_dir / "patch_score_map_raw.json"),
                "patch_scores_csv": str(sample_dir / "patch_scores.csv"),
                "pixel_score_map": str(pixel_map_path),
            }
        )
        print(f"[{idx}/{len(samples)}] {record['category']} {record['label_name']} {record['image_id']} score={image_score:.4f}")

    labels = [int(r["label"]) for r in image_rows]
    scores = [float(r["image_score"]) for r in image_rows]
    overall = {
        "binary_acc": accuracy(labels, scores, args.threshold),
        "I-AUROC": binary_auc(labels, scores),
        "p-AUROC": binary_auc(all_pixel_labels, all_pixel_scores) if all_pixel_labels else None,
        "num_images": len(image_rows),
        "num_pixel_labels": len(all_pixel_labels),
        "threshold": args.threshold,
    }

    per_category = {}
    for category in sorted(set(r["category"] for r in image_rows)):
        rows = [r for r in image_rows if r["category"] == category]
        c_labels = [int(r["label"]) for r in rows]
        c_scores = [float(r["image_score"]) for r in rows]
        per_category[category] = {
            "binary_acc": accuracy(c_labels, c_scores, args.threshold),
            "I-AUROC": binary_auc(c_labels, c_scores),
            "p-AUROC": binary_auc(
                pixel_by_category.get(category, {}).get("labels", []),
                pixel_by_category.get(category, {}).get("scores", []),
            )
            if category in pixel_by_category
            else None,
            "num_images": len(rows),
            "num_normal": sum(1 for x in c_labels if x == 0),
            "num_anomaly": sum(1 for x in c_labels if x == 1),
        }

    write_csv(out_dir / "image_scores.csv", image_rows)
    write_csv(out_dir / "patch_scores_all.csv", patch_rows_all)
    save_json(out_dir / "metrics.json", {"overall": overall, "per_category": per_category})
    print(json.dumps({"overall": overall, "per_category": per_category}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
