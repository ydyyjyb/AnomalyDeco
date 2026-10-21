import argparse
import csv
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


def load_legacy_module() -> Any:
    path = Path(__file__).with_name("0724_Mechanism_validation.py")
    spec = importlib.util.spec_from_file_location("mechanism_0724", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_words(text: str) -> List[str]:
    out = []
    seen = set()
    for part in str(text or "").split(","):
        word = part.strip()
        key = word.lower()
        if word and key not in seen:
            seen.add(key)
            out.append(word)
    return out


def parse_range(text: str) -> List[int]:
    text = str(text).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_paths(args: argparse.Namespace) -> None:
    visa_root = Path(args.visa_root)
    split_csv = Path(args.visa_split_csv) if args.visa_split_csv else visa_root / "split_csv" / "1cls.csv"
    model_path = Path(args.model_path)
    problems = []
    if not str(args.visa_root).strip():
        problems.append("--visa-root is empty. Did you forget to export VISA_ROOT?")
    if not visa_root.exists():
        problems.append(f"VisA root does not exist: {visa_root}")
    if not split_csv.exists():
        problems.append(f"split csv does not exist: {split_csv}")
    if not str(args.model_path).strip():
        problems.append("--model-path is empty. Did you forget to export MODEL?")
    if not model_path.exists():
        problems.append(f"model path does not exist: {model_path}")
    if problems:
        hint = (
            "\n\nSet paths explicitly, for example:\n"
            "  export VISA_ROOT=/home/vina04/ADAR/evaluation/ODinW-13/data/VisA\n"
            "  export MODEL=/home/vina04/ADAR/Deco-main/qwen3_4b_vl_instruct\n"
            "  export ROOT=/home/vina04/ADAR/Deco-main/transformers-4.57.6\n"
        )
        raise FileNotFoundError("\n".join(problems) + hint)


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


def sample_from_split(args: argparse.Namespace) -> List[Dict[str, Any]]:
    visa_root = Path(args.visa_root)
    split_csv = Path(args.visa_split_csv) if args.visa_split_csv else visa_root / "split_csv" / "1cls.csv"
    categories = set(parse_csv_words(args.categories))
    rows = []
    prompt_maps: Dict[str, Dict[str, List[str]]] = {}
    with open(split_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            category, split, label_text, rel_image = [x.strip() for x in row[:4]]
            rel_mask = row[4].strip() if len(row) > 4 else ""
            if category not in categories:
                continue
            if split.lower() != str(args.visa_split_set).lower():
                continue
            image_path = visa_root / rel_image
            if not image_path.exists():
                continue
            label = 0 if label_text.lower() == "normal" else 1
            if category not in prompt_maps:
                prompt_maps[category] = load_anomaly_prompt_map(visa_root / category)
            defect_labels: List[str]
            if label == 0:
                defect_labels = ["normal"]
            else:
                prompt_map = prompt_maps[category]
                defect_labels = (
                    prompt_map.get(rel_image.replace("\\", "/"))
                    or prompt_map.get(Path(rel_image).name)
                    or [args.fallback_defect_type]
                )
            rows.append(
                {
                    "category": category,
                    "image": str(image_path),
                    "image_id": Path(rel_image).stem,
                    "label": label,
                    "label_name": "anomaly" if label else "normal",
                    "defect_labels": defect_labels,
                    "split_image": rel_image,
                    "mask": str(visa_root / rel_mask) if rel_mask and (visa_root / rel_mask).exists() else "",
                }
            )

    rng = random.Random(args.seed)
    selected = []
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for sample in rows:
        buckets.setdefault((sample["category"], sample["label"]), []).append(sample)
    for category in sorted(categories):
        for label in [0, 1]:
            bucket = buckets.get((category, label), [])
            k = args.sample_anomaly if label == 1 else args.sample_normal
            if k < 0 or len(bucket) <= k:
                selected.extend(bucket)
            else:
                selected.extend(rng.sample(bucket, k))
    return selected


def build_instruction(category: str) -> str:
    return (
        "Inspect the image. Decide whether the main subject is normal or anomalous. "
        f"The object category is {category}. "
        "Ignore viewpoint, orientation, object placement, lighting, background, and image noise. "
        "Use only visible local evidence."
    )


def build_score_prefix(processor: Any, category: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_instruction(category)},
                {"type": "text", "text": "Query image:"},
                {"type": "image"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "Answer: "


def first_token_ids(tokenizer: Any, labels: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label in labels:
        label = str(label).strip()
        if not label:
            continue
        encoded = tokenizer(label, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            out[label] = int(ids[0])
    return out


def logit_group(logits: torch.Tensor, token_ids: List[int], mode: str) -> torch.Tensor:
    values = logits[torch.tensor(token_ids, device=logits.device, dtype=torch.long)].float()
    if mode == "logsumexp":
        return torch.logsumexp(values, dim=0)
    return torch.max(values, dim=0).values


def get_image_token_positions(processor: Any, model: Any, input_ids: torch.Tensor) -> List[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    candidates = []
    for attr in ("image_token_id", "image_token_index"):
        value = getattr(getattr(model, "config", None), attr, None)
        if isinstance(value, int):
            candidates.append(value)
    if tokenizer is not None:
        for token in ("<|image_pad|>", "<image>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0:
                candidates.append(token_id)
    ids = input_ids.detach().cpu().tolist()
    for token_id in candidates:
        positions = [idx for idx, value in enumerate(ids) if int(value) == int(token_id)]
        if positions:
            return positions
    raise RuntimeError("Could not locate image visual token positions in input_ids.")


def visual_grid_shape(inputs: Dict[str, Any], model: Any, token_count: int, image_size: Tuple[int, int]) -> Tuple[int, int]:
    grid = inputs.get("image_grid_thw")
    merge = getattr(getattr(getattr(model, "config", None), "vision_config", None), "spatial_merge_size", 2)
    if grid is not None:
        values = grid.detach().cpu().tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        if len(values) >= 3:
            _, gh, gw = [int(x) for x in values[:3]]
            candidates = [
                (max(1, gh // max(1, int(merge))), max(1, gw // max(1, int(merge)))),
                (max(1, gh), max(1, gw)),
            ]
            for h, w in candidates:
                if h * w == token_count:
                    return h, w
    width, height = image_size
    aspect = max(width / max(height, 1), 1e-6)
    h = max(1, int(round(math.sqrt(token_count / aspect))))
    w = max(1, int(math.ceil(token_count / h)))
    while h * w < token_count:
        h += 1
    return h, w


def normalize_values(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def make_heatmap_image(values: List[float], shape: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
    h, w = shape
    padded = list(values) + [0.0] * max(0, h * w - len(values))
    pixels = [int(max(0.0, min(1.0, v)) * 255) for v in padded[: h * w]]
    gray = Image.new("L", (w, h))
    gray.putdata(pixels)
    gray = gray.resize(size, Image.Resampling.BICUBIC)
    return ImageOps.colorize(gray, black="#001252", white="#ff2400")


def otsu_threshold(gray: Image.Image) -> int:
    hist = gray.histogram()
    total = sum(hist)
    if total <= 0:
        return 127
    sum_total = sum(i * count for i, count in enumerate(hist))
    sum_b = 0.0
    weight_b = 0
    best_threshold = 127
    best_var = -1.0
    for threshold, count in enumerate(hist):
        weight_b += count
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += threshold * count
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_threshold = threshold
    return int(best_threshold)


def border_mean(gray: Image.Image) -> float:
    width, height = gray.size
    pixels = gray.load()
    values = []
    step = max(1, min(width, height) // 200)
    for x in range(0, width, step):
        values.append(pixels[x, 0])
        values.append(pixels[x, height - 1])
    for y in range(0, height, step):
        values.append(pixels[0, y])
        values.append(pixels[width - 1, y])
    return sum(values) / max(len(values), 1)


def build_object_mask(image: Image.Image, min_area_ratio: float = 0.002) -> Image.Image:
    gray = ImageOps.grayscale(image)
    threshold = otsu_threshold(gray)
    bg = border_mean(gray)
    if bg <= threshold:
        mask = gray.point(lambda v: 255 if v > threshold else 0)
    else:
        mask = gray.point(lambda v: 255 if v < threshold else 0)
    mask = mask.filter(ImageFilter.MedianFilter(5))
    mask = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))

    # Remove tiny isolated regions without requiring scipy/opencv.
    small = mask.resize((max(1, mask.width // 4), max(1, mask.height // 4)), Image.Resampling.NEAREST)
    pix = small.load()
    width, height = small.size
    visited = set()
    components = []
    for y in range(height):
        for x in range(width):
            if pix[x, y] == 0 or (x, y) in visited:
                continue
            stack = [(x, y)]
            visited.add((x, y))
            comp = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and pix[nx, ny] > 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            components.append(comp)
    if components:
        min_area = max(4, int(width * height * min_area_ratio))
        keep = [comp for comp in components if len(comp) >= min_area]
        if not keep:
            keep = sorted(components, key=len, reverse=True)[:4]
        clean = Image.new("L", small.size, 0)
        draw = ImageDraw.Draw(clean)
        for comp in keep:
            for point in comp:
                draw.point(point, fill=255)
        mask = clean.resize(mask.size, Image.Resampling.NEAREST)
        mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    return mask


def patch_mask_coverages(mask: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> List[float]:
    coverages = []
    binary = mask.convert("L")
    for box in boxes:
        crop = binary.crop(box)
        hist = crop.histogram()
        foreground = sum(hist[1:])
        area = max(crop.width * crop.height, 1)
        coverages.append(float(foreground) / float(area))
    return coverages


def apply_object_mask_to_patch_values(
    values: List[float],
    boxes: List[Tuple[int, int, int, int]],
    object_mask: Image.Image,
    min_coverage: float,
) -> Tuple[List[float], List[float]]:
    coverages = patch_mask_coverages(object_mask, boxes)
    masked = [float(value) if coverage >= min_coverage else 0.0 for value, coverage in zip(values, coverages)]
    return masked, coverages


def overlay_heatmap(image: Image.Image, heatmap: Image.Image, alpha: float) -> Image.Image:
    base = image.convert("RGB")
    heat = heatmap.convert("RGB")
    return Image.blend(base, heat, alpha=max(0.0, min(1.0, alpha)))


def overlay_mask(image: Image.Image, mask_path: str, alpha: float = 0.45) -> Optional[Image.Image]:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    mask = Image.open(path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", image.size, (255, 0, 0))
    base = image.convert("RGB")
    blended = Image.blend(base, red, alpha=max(0.0, min(1.0, alpha)))
    return Image.composite(blended, base, mask.point(lambda v: 255 if v > 0 else 0))


def annotate_image(
    image: Image.Image,
    lines: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    padding = 8
    line_h = 28
    banner_h = padding * 2 + line_h * len(lines)
    draw.rectangle((0, 0, out.width, banner_h), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_h), line, fill=(255, 255, 255), font=font)
    if boxes:
        for rank, box in enumerate(boxes, start=1):
            color = (255, 230, 0) if rank == 1 else (0, 255, 255)
            draw.rectangle(box, outline=color, width=4)
            draw.text((box[0] + 4, box[1] + 4), f"top{rank}", fill=color, font=font)
    return out


def draw_boxes_on_raw(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    label_prefix: str = "top",
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for rank, box in enumerate(boxes, start=1):
        color = (255, 230, 0) if rank == 1 else (0, 255, 255)
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 4, box[1] + 4), f"{label_prefix}{rank}", fill=color, font=font)
    return out


def top_token_boxes(values: List[float], shape: Tuple[int, int], image_size: Tuple[int, int], k: int) -> List[Tuple[int, int, int, int]]:
    h, w = shape
    width, height = image_size
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[: max(0, k)]
    boxes = []
    for idx, _ in indexed:
        r = idx // w
        c = idx % w
        x0 = int(c * width / w)
        x1 = int((c + 1) * width / w)
        y0 = int(r * height / h)
        y1 = int((r + 1) * height / h)
        boxes.append((x0, y0, x1, y1))
    return boxes


def patch_boxes(image_size: Tuple[int, int], grid: int) -> List[Tuple[int, int, int, int]]:
    return grid_boxes((grid, grid), image_size)


def grid_boxes(shape: Tuple[int, int], image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    width, height = image_size
    rows, cols = shape
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(c * width / cols)
            x1 = int((c + 1) * width / cols)
            y0 = int(r * height / rows)
            y1 = int((r + 1) * height / rows)
            boxes.append((x0, y0, x1, y1))
    return boxes


def make_occluded_image(image: Image.Image, box: Tuple[int, int, int, int], mode: str) -> Image.Image:
    out = image.convert("RGB").copy()
    x0, y0, x1, y1 = box
    if mode == "mean":
        pixels = list(out.getdata())
        mean = tuple(int(sum(pixel[i] for pixel in pixels) / max(len(pixels), 1)) for i in range(3))
        fill = mean
    elif mode == "gray":
        fill = (127, 127, 127)
    else:
        fill = (0, 0, 0)
    draw = ImageDraw.Draw(out)
    draw.rectangle((x0, y0, x1, y1), fill=fill)
    return out


def answer_scores_for_images(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    score_prefix: str,
    images: List[Image.Image],
    normal_ids: List[int],
    anomaly_ids: List[int],
) -> List[float]:
    if not images:
        return []
    image_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    inputs = processor(text=[score_prefix] * len(images), images=images, return_tensors="pt", **image_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    logits = outputs.logits[:, -1].float()
    normal_tensor = torch.tensor(normal_ids, device=logits.device, dtype=torch.long)
    anomaly_tensor = torch.tensor(anomaly_ids, device=logits.device, dtype=torch.long)
    normal_logits = logits.index_select(1, normal_tensor).max(dim=1).values
    anomaly_logits = logits.index_select(1, anomaly_tensor).max(dim=1).values
    return (anomaly_logits - normal_logits).detach().float().cpu().tolist()


def compute_occlusion_map(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    score_prefix: str,
    image: Image.Image,
    normal_ids: List[int],
    anomaly_ids: List[int],
) -> Tuple[List[float], List[float], List[Tuple[int, int, int, int]]]:
    grid = max(1, int(args.occlusion_grid))
    boxes = patch_boxes(image.size, grid)
    full_score = answer_scores_for_images(
        args,
        processor,
        model,
        score_prefix,
        [image],
        normal_ids,
        anomaly_ids,
    )[0]
    raw_contribs: List[float] = []
    for start in range(0, len(boxes), max(1, int(args.occlusion_batch_size))):
        batch_boxes = boxes[start : start + max(1, int(args.occlusion_batch_size))]
        batch_images = [make_occluded_image(image, box, args.occlusion_fill) for box in batch_boxes]
        masked_scores = answer_scores_for_images(
            args,
            processor,
            model,
            score_prefix,
            batch_images,
            normal_ids,
            anomaly_ids,
        )
        for masked_score in masked_scores:
            raw_contribs.append(float(full_score) - float(masked_score))
    if args.occlusion_positive_only:
        normalized = normalize_values([max(0.0, value) for value in raw_contribs])
    else:
        normalized = normalize_values(raw_contribs)
    return raw_contribs, normalized, boxes


def reduce_patch_attribution(tensor: torch.Tensor) -> List[float]:
    values = tensor.detach().float()
    if values.ndim == 0:
        return [float(values.cpu())]
    if values.ndim == 1:
        return values.abs().cpu().tolist()
    values = values.reshape(values.shape[0], -1)
    return values.sum(dim=1).cpu().tolist()


def compute_pixel_gradient_map(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    score_prefix: str,
    image: Image.Image,
    normal_ids: List[int],
    anomaly_ids: List[int],
) -> Tuple[List[float], List[float], Tuple[int, int], List[Tuple[int, int, int, int]]]:
    image_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    inputs = processor(text=[score_prefix], images=[image], return_tensors="pt", **image_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" not in inputs:
        raise RuntimeError("processor output has no pixel_values; cannot compute image gradient heatmap.")

    previous_requires_grad = [param.requires_grad for param in model.parameters()]
    try:
        for param in model.parameters():
            param.requires_grad_(False)

        pixel_values = inputs["pixel_values"].detach().clone().to(model.device)
        if not torch.is_floating_point(pixel_values):
            pixel_values = pixel_values.float()
        pixel_values.requires_grad_(True)
        inputs["pixel_values"] = pixel_values

        model.zero_grad(set_to_none=True)
        outputs = model(**inputs, return_dict=True, use_cache=False)
        logits = outputs.logits[0, -1].float()
        anomaly_logit = logit_group(logits, anomaly_ids, args.score_label_aggregation)
        normal_logit = logit_group(logits, normal_ids, args.score_label_aggregation)
        margin = anomaly_logit - normal_logit
        margin.backward()

        grad = pixel_values.grad
        if grad is None:
            raise RuntimeError("pixel_values.grad is None; the model may detach visual inputs internally.")
        signed = grad.float() * pixel_values.detach().float()
        if args.gradient_positive_only:
            patch_attr = torch.relu(signed).reshape(signed.shape[0], -1).sum(dim=1)
        else:
            patch_attr = signed.abs().reshape(signed.shape[0], -1).sum(dim=1)
        raw_values = reduce_patch_attribution(patch_attr)
    finally:
        for param, requires_grad in zip(model.parameters(), previous_requires_grad):
            param.requires_grad_(requires_grad)
        model.zero_grad(set_to_none=True)

    shape = visual_grid_shape(inputs, model, len(raw_values), image.size)
    boxes = grid_boxes(shape, image.size)
    normalized = normalize_values(raw_values)
    return raw_values, normalized, shape, boxes


def save_score_curve(case_dir: Path, layer_rows: List[Dict[str, Any]], selected_layer: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not layer_rows:
        return
    layers = [int(row["layer"]) for row in layer_rows]
    anomaly_probs = [float(row["anomaly_prob"]) for row in layer_rows]
    normal_probs = [float(row["normal_prob"]) for row in layer_rows]
    scores = [float(row["score"]) for row in layer_rows]
    plt.figure(figsize=(9, 5))
    plt.plot(layers, anomaly_probs, marker="o", label="anomaly prob")
    plt.plot(layers, normal_probs, marker="o", label="normal prob")
    plt.plot(layers, scores, marker="o", label="anomaly-normal score")
    plt.axvline(selected_layer, color="red", linestyle="--", label=f"heatmap layer {selected_layer}")
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title("Answer-token candidate trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(case_dir / "4-answer_token_layer_trend.png", dpi=180)
    plt.close()


def choose_heatmap_layer(layer_rows: List[Dict[str, Any]], deco_layers: List[int], mode: str) -> int:
    if mode == "final":
        return max(int(row["layer"]) for row in layer_rows)
    candidates = [row for row in layer_rows if int(row["layer"]) in set(deco_layers)]
    if not candidates:
        candidates = layer_rows
    if mode == "max_anomaly_score":
        return int(max(candidates, key=lambda row: float(row["score"]))["layer"])
    return int(max(candidates, key=lambda row: abs(float(row["score"])) * float(row["confidence"]))["layer"])


def run_one(
    args: argparse.Namespace,
    legacy: Any,
    processor: Any,
    model: Any,
    sample: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor has no tokenizer")

    image = Image.open(sample["image"]).convert("RGB")
    score_prefix = build_score_prefix(processor, sample["category"])
    image_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    inputs = processor(text=[score_prefix], images=[image], return_tensors="pt", **image_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    normal_map = first_token_ids(tokenizer, parse_csv_words(args.answer_normal_labels))
    anomaly_map = first_token_ids(tokenizer, parse_csv_words(args.answer_anomaly_labels))
    normal_ids = sorted(set(normal_map.values()))
    anomaly_ids = sorted(set(anomaly_map.values()))
    if not normal_ids or not anomaly_ids:
        raise RuntimeError("Failed to tokenize answer labels.")
    candidate_ids = sorted(set(normal_ids + anomaly_ids))

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=args.save_attention_heatmap,
            return_dict=True,
            use_cache=False,
        )

    probe_args = argparse.Namespace(score_label_aggregation=args.score_label_aggregation)
    layer_rows = legacy.compute_binary_layer_probe(
        args=probe_args,
        model=model,
        outputs=outputs,
        candidate_ids=candidate_ids,
        normal_ids=normal_ids,
        anomaly_ids=anomaly_ids,
        trend_layers=parse_range(args.trend_layers),
        deco_layers=set(parse_range(args.deco_layers)),
    )["layer_rows"]
    final_score = float(outputs.logits[0, -1].float()[anomaly_ids].max() - outputs.logits[0, -1].float()[normal_ids].max())
    final_pred = 1 if final_score >= args.decision_threshold else 0
    selected_layer = choose_heatmap_layer(layer_rows, parse_range(args.deco_layers), args.heatmap_layer_mode)

    case_name = f"{sample['category']}-{sample['label_name']}-{sample['image_id']}"
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    defect_text = ", ".join(sample["defect_labels"])
    pred_text = "anomaly" if final_pred else "normal"
    title_lines = [
        f"{sample['category']} / GT: {sample['label_name']} / defect: {defect_text}",
        f"pred: {pred_text} / score: {final_score:.4f} / heatmap layer: {selected_layer}",
    ]
    annotate_image(image, title_lines).save(case_dir / "1-original_annotated.jpg", quality=95)
    gt_mask_overlay = overlay_mask(image, sample.get("mask", ""), 0.45)
    if gt_mask_overlay is not None:
        annotate_image(gt_mask_overlay, title_lines).save(case_dir / "1b-ground_truth_mask_overlay.jpg", quality=95)
    object_mask = None
    object_coverages: List[float] = []
    if args.use_object_mask:
        object_mask = build_object_mask(image, args.object_mask_min_area_ratio)
        object_mask.save(case_dir / "1c-object_mask.png")
        object_overlay = overlay_mask(image, str(case_dir / "1c-object_mask.png"), 0.35)
        if object_overlay is not None:
            object_overlay.save(case_dir / "1d-object_mask_overlay_raw.jpg", quality=95)
    save_score_curve(case_dir, layer_rows, selected_layer)

    attention_info: Dict[str, Any] = {}
    if args.save_attention_heatmap:
        attentions = getattr(outputs, "attentions", None)
        if attentions is not None and attentions:
            attn_index = selected_layer - 1 if selected_layer > 0 else 0
            attn_index = max(0, min(attn_index, len(attentions) - 1))
            attn = attentions[attn_index]
            if attn is not None:
                input_ids = inputs["input_ids"][0]
                visual_positions = get_image_token_positions(processor, model, input_ids)
                query_index = inputs["input_ids"].shape[-1] - 1
                attn_values = attn[0, :, query_index, visual_positions].float().mean(dim=0).detach().cpu().tolist()
                attn_values = normalize_values([float(v) for v in attn_values])
                attn_shape = visual_grid_shape(inputs, model, len(attn_values), image.size)
                attn_heatmap = make_heatmap_image(attn_values, attn_shape, image.size)
                attn_boxes = top_token_boxes(attn_values, attn_shape, image.size, args.top_boxes)
                attn_overlay = overlay_heatmap(image, attn_heatmap, args.overlay_alpha)
                attn_heatmap.save(case_dir / "2-attention_heatmap_reference.jpg", quality=95)
                annotate_image(attn_overlay, title_lines, attn_boxes).save(
                    case_dir / "3-attention_overlay_reference.jpg",
                    quality=95,
                )
                attention_info = {
                    "attention_layer_index": attn_index,
                    "visual_token_count": len(attn_values),
                    "visual_grid_shape": list(attn_shape),
                    "attention_top_boxes": [list(box) for box in attn_boxes],
                }

    occlusion_raw: List[float] = []
    occlusion_values: List[float] = []
    occlusion_values_masked: List[float] = []
    occlusion_boxes: List[Tuple[int, int, int, int]] = []
    object_mask_fallback = False
    if args.save_occlusion_heatmap:
        occlusion_raw, occlusion_values, occlusion_boxes = compute_occlusion_map(
            args,
            processor,
            model,
            score_prefix,
            image,
            normal_ids,
            anomaly_ids,
        )
        occ_shape = (args.occlusion_grid, args.occlusion_grid)
        occlusion_values_masked = list(occlusion_values)
        if object_mask is not None:
            occlusion_values_masked, object_coverages = apply_object_mask_to_patch_values(
                occlusion_values,
                occlusion_boxes,
                object_mask,
                args.object_mask_min_coverage,
            )
            occlusion_values_masked = normalize_values(occlusion_values_masked)
            if max(occlusion_values_masked) <= 0.0 and max(occlusion_values) > 0.0:
                object_mask_fallback = True
                occlusion_values_masked = list(occlusion_values)
        occ_heatmap = make_heatmap_image(occlusion_values_masked, occ_shape, image.size)
        ranked_occ_boxes = [
            box
            for _, box in sorted(
                zip(occlusion_values_masked, occlusion_boxes),
                key=lambda item: item[0],
                reverse=True,
            )[: args.top_boxes]
        ]
        occ_overlay = overlay_heatmap(image, occ_heatmap, args.overlay_alpha)
        occ_heatmap.save(case_dir / "5-occlusion_margin_heatmap.jpg", quality=95)
        draw_boxes_on_raw(image, ranked_occ_boxes, "occ").save(
            case_dir / "5b-raw_occlusion_top_boxes_no_title.jpg",
            quality=95,
        )
        annotate_image(occ_overlay, title_lines, ranked_occ_boxes).save(
            case_dir / "6-occlusion_margin_overlay_top_boxes.jpg",
            quality=95,
        )

    gradient_raw: List[float] = []
    gradient_values: List[float] = []
    gradient_values_masked: List[float] = []
    gradient_boxes: List[Tuple[int, int, int, int]] = []
    gradient_shape: Tuple[int, int] = (0, 0)
    gradient_coverages: List[float] = []
    gradient_error = ""
    if args.save_gradient_heatmap:
        try:
            gradient_raw, gradient_values, gradient_shape, gradient_boxes = compute_pixel_gradient_map(
                args,
                processor,
                model,
                score_prefix,
                image,
                normal_ids,
                anomaly_ids,
            )
            gradient_values_masked = list(gradient_values)
            if object_mask is not None:
                gradient_values_masked, gradient_coverages = apply_object_mask_to_patch_values(
                    gradient_values,
                    gradient_boxes,
                    object_mask,
                    args.object_mask_min_coverage,
                )
                gradient_values_masked = normalize_values(gradient_values_masked)
                if max(gradient_values_masked) <= 0.0 and max(gradient_values) > 0.0:
                    gradient_values_masked = list(gradient_values)
            grad_heatmap = make_heatmap_image(gradient_values_masked, gradient_shape, image.size)
            ranked_grad_boxes = [
                box
                for _, box in sorted(
                    zip(gradient_values_masked, gradient_boxes),
                    key=lambda item: item[0],
                    reverse=True,
                )[: args.top_boxes]
            ]
            grad_overlay = overlay_heatmap(image, grad_heatmap, args.overlay_alpha)
            grad_heatmap.save(case_dir / "7-gradient_margin_heatmap.jpg", quality=95)
            draw_boxes_on_raw(image, ranked_grad_boxes, "grad").save(
                case_dir / "7b-raw_gradient_top_boxes_no_title.jpg",
                quality=95,
            )
            annotate_image(grad_overlay, title_lines, ranked_grad_boxes).save(
                case_dir / "8-gradient_margin_overlay_top_boxes.jpg",
                quality=95,
            )
        except Exception as exc:
            gradient_error = str(exc)
            print(f"[WARN] gradient heatmap failed for {case_name}: {exc}")

    case = {
        "image": sample["image"],
        "image_id": sample["image_id"],
        "category": sample["category"],
        "label": sample["label"],
        "label_name": sample["label_name"],
        "defect_labels": sample["defect_labels"],
        "mask": sample.get("mask", ""),
        "score_prefix": score_prefix,
        "answer_normal_tokenization": normal_map,
        "answer_anomaly_tokenization": anomaly_map,
        "final_score": final_score,
        "final_pred": pred_text,
        "selected_heatmap_layer": selected_layer,
        **attention_info,
        "occlusion_grid": args.occlusion_grid if args.save_occlusion_heatmap else None,
        "occlusion_raw_margin_drop": occlusion_raw,
        "occlusion_normalized_margin_drop": occlusion_values,
        "occlusion_object_masked_margin_drop": occlusion_values_masked,
        "occlusion_boxes": [list(box) for box in occlusion_boxes],
        "object_mask_enabled": bool(args.use_object_mask),
        "object_mask_fallback_to_unmasked": bool(object_mask_fallback),
        "object_mask_min_coverage": args.object_mask_min_coverage,
        "object_mask_patch_coverages": object_coverages,
        "gradient_raw_attribution": gradient_raw,
        "gradient_normalized_attribution": gradient_values,
        "gradient_object_masked_attribution": gradient_values_masked,
        "gradient_boxes": [list(box) for box in gradient_boxes],
        "gradient_grid_shape": list(gradient_shape),
        "gradient_object_mask_coverages": gradient_coverages,
        "gradient_error": gradient_error,
        "layer_rows": layer_rows,
        "files": {
            "original": str(case_dir / "1-original_annotated.jpg"),
            "ground_truth_mask_overlay": str(case_dir / "1b-ground_truth_mask_overlay.jpg")
            if sample.get("mask", "")
            else "",
            "object_mask": str(case_dir / "1c-object_mask.png") if args.use_object_mask else "",
            "object_mask_overlay_raw": str(case_dir / "1d-object_mask_overlay_raw.jpg") if args.use_object_mask else "",
            "attention_heatmap": str(case_dir / "2-attention_heatmap_reference.jpg") if args.save_attention_heatmap else "",
            "attention_overlay": str(case_dir / "3-attention_overlay_reference.jpg") if args.save_attention_heatmap else "",
            "trend": str(case_dir / "4-answer_token_layer_trend.png"),
            "occlusion_heatmap": str(case_dir / "5-occlusion_margin_heatmap.jpg") if args.save_occlusion_heatmap else "",
            "raw_occlusion_top_boxes_no_title": str(case_dir / "5b-raw_occlusion_top_boxes_no_title.jpg")
            if args.save_occlusion_heatmap
            else "",
            "occlusion_overlay": str(case_dir / "6-occlusion_margin_overlay_top_boxes.jpg") if args.save_occlusion_heatmap else "",
            "gradient_heatmap": str(case_dir / "7-gradient_margin_heatmap.jpg") if args.save_gradient_heatmap else "",
            "raw_gradient_top_boxes_no_title": str(case_dir / "7b-raw_gradient_top_boxes_no_title.jpg")
            if args.save_gradient_heatmap
            else "",
            "gradient_overlay": str(case_dir / "8-gradient_margin_overlay_top_boxes.jpg") if args.save_gradient_heatmap else "",
        },
    }
    save_json(case_dir / "case.json", case)
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Answer-token/anchor-layer heatmap probe for Qwen3-VL on VisA.")
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--visa-split-csv", default=None)
    parser.add_argument("--visa-split-set", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--categories", default="candle,pcb1,macaroni2,pipe_fryum")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-defect-type", default="anomaly_unknown")
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trend-layers", default="0:34")
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--heatmap-layer-mode", default="anchor_evidence", choices=["anchor_evidence", "max_anomaly_score", "final"])
    parser.add_argument("--score-label-aggregation", default="max", choices=["max", "logsumexp"])
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument("--answer-normal-labels", default="Normal,normal,Good,good,No,no")
    parser.add_argument("--answer-anomaly-labels", default="Anomaly,anomaly,Abnormal,abnormal,Defective,defective,Yes,yes")
    parser.add_argument("--save-attention-heatmap", action="store_true")
    parser.add_argument("--save-occlusion-heatmap", action="store_true", default=True)
    parser.add_argument("--disable-occlusion-heatmap", dest="save_occlusion_heatmap", action="store_false")
    parser.add_argument("--occlusion-grid", type=int, default=8)
    parser.add_argument("--occlusion-batch-size", type=int, default=8)
    parser.add_argument("--occlusion-fill", default="mean", choices=["mean", "gray", "black"])
    parser.add_argument("--occlusion-positive-only", action="store_true", default=True)
    parser.add_argument("--save-gradient-heatmap", action="store_true", default=True)
    parser.add_argument("--disable-gradient-heatmap", dest="save_gradient_heatmap", action="store_false")
    parser.add_argument(
        "--gradient-positive-only",
        action="store_true",
        default=True,
        help="Use ReLU(grad * pixel_values). If disabled, use absolute grad * pixel_values.",
    )
    parser.add_argument("--gradient-absolute", dest="gradient_positive_only", action="store_false")
    parser.add_argument("--use-object-mask", action="store_true", default=True)
    parser.add_argument("--disable-object-mask", dest="use_object_mask", action="store_false")
    parser.add_argument("--object-mask-min-coverage", type=float, default=0.15)
    parser.add_argument("--object-mask-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--top-boxes", type=int, default=3)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--log-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "0-config.json", vars(args))
    validate_paths(args)

    legacy = load_legacy_module()
    samples = sample_from_split(args)
    print(f"[INFO] selected samples: {len(samples)}")
    print(f"[INFO] categories: {args.categories}")
    if args.attn_implementation != "eager":
        print("[WARN] Attention heatmap usually requires --attn-implementation eager.")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    except Exception as exc:
        if not args.allow_attn_fallback:
            raise
        print(f"[WARN] failed to load with {args.attn_implementation}: {exc}")
        print("[WARN] fallback to eager for attention maps.")
        load_kwargs["attn_implementation"] = "eager"
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    cases = []
    with open(output_dir / "cases.jsonl", "w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(samples, desc="Answer heatmaps"), start=1):
            try:
                case = run_one(args, legacy, processor, model, sample, output_dir)
                cases.append(case)
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                f.flush()
                if idx % max(1, args.log_interval) == 0:
                    print(
                        f"[{idx}/{len(samples)}] {sample['category']}/{sample['label_name']}/"
                        f"{sample['image_id']} pred={case['final_pred']} score={case['final_score']:.4f} "
                        f"defect={','.join(sample['defect_labels'])}"
                    )
            except Exception as exc:
                failed = {**sample, "error": str(exc)}
                save_json(output_dir / f"failed-{idx:04d}.json", failed)
                print(f"[WARN] failed {sample['category']}/{sample['image_id']}: {exc}")
    save_json(output_dir / "summary.json", {"num_cases": len(cases), "cases": cases})
    print(f"[OK] saved to: {output_dir}")


if __name__ == "__main__":
    main()
