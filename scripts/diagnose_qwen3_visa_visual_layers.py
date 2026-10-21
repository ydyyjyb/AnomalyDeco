#!/usr/bin/env python3
"""Frozen Qwen3-VL visual-layer diagnostic on VisA.

For each category, build a patch memory from 1/2/4 normal support images.
Score held-out normal and anomaly images by nearest-neighbour cosine distance
at every native DeepStack output and at the final visual output.

No model parameters are updated and no test labels are used to construct the
memory or the anomaly scores. Labels are used only after scoring to compute
image-level AUROC.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_CATEGORIES = (
    "candle,capsules,cashew,chewinggum,fryum,macaroni1,macaroni2,"
    "pcb1,pcb3,pipe_fryum"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--visa-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--categories", default=DEFAULT_CATEGORIES)
    p.add_argument("--shots", default="1,2,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-normal-test", type=int, default=0)
    p.add_argument("--max-anomaly-test", type=int, default=0)
    p.add_argument("--min-pixels", type=int, default=262144)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def binary_auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean(values):
    return sum(values) / len(values) if values else None


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def visual_owner(model):
    if hasattr(model, "model") and hasattr(model.model, "get_image_features"):
        return model.model
    if hasattr(model, "get_image_features"):
        return model
    raise AttributeError("Cannot locate get_image_features on Qwen3-VL model")


@torch.inference_mode()
def extract_features(model, processor, image_path, min_pixels, max_pixels):
    image = Image.open(image_path).convert("RGB")
    inputs = processor.image_processor(
        images=[image],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    owner = visual_owner(model)
    device = next(owner.visual.parameters()).device
    pixel_values = inputs["pixel_values"].to(device=device, dtype=next(owner.visual.parameters()).dtype)
    grid_thw = inputs["image_grid_thw"].to(device)
    final_features, deep_features = owner.get_image_features(pixel_values, grid_thw)
    names = [f"deepstack_{idx}" for idx in owner.visual.deepstack_visual_indexes] + ["final"]
    tensors = list(deep_features) + [final_features]
    result = {}
    for name, tensor in zip(names, tensors):
        if isinstance(tensor, (tuple, list)):
            if len(tensor) != 1:
                raise RuntimeError(f"Expected one image feature tensor for {name}, got {len(tensor)}")
            tensor = tensor[0]
        result[name] = F.normalize(tensor.float(), dim=-1).cpu()
    return result, [int(x) for x in grid_thw[0].tolist()]


def patch_distances(query, memory):
    # Both are L2-normalized. Chunking bounds peak memory for larger shot banks.
    chunks = []
    for start in range(0, query.shape[0], 512):
        similarity = query[start : start + 512] @ memory.T
        chunks.append(1.0 - similarity.max(dim=1).values)
    return torch.cat(chunks)


def aggregate(distances, mode):
    if mode == "max":
        return float(distances.max().item())
    fraction = float(mode.removeprefix("top")) / 100.0
    k = max(1, int(math.ceil(distances.numel() * fraction)))
    return float(torch.topk(distances, k=k).values.mean().item())


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.visa_root).resolve()
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    shots = sorted({int(x) for x in args.shots.split(",") if x.strip()})
    max_shots = max(shots)
    score_modes = ["max", "top1", "top5"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.eval()
    owner = visual_owner(model)
    layer_names = [f"deepstack_{idx}" for idx in owner.visual.deepstack_visual_indexes] + ["final"]

    config = vars(args).copy()
    config.update({"resolved_categories": categories, "resolved_shots": shots, "layers": layer_names})
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    per_image = []
    support_manifest = []
    for cat_i, category in enumerate(categories, start=1):
        normal_dir = root / category / "Data" / "Images" / "Normal"
        anomaly_dir = root / category / "Data" / "Images" / "Anomaly"
        normals = image_files(normal_dir)
        anomalies = image_files(anomaly_dir)
        rng = random.Random(args.seed + cat_i)
        rng.shuffle(normals)
        rng.shuffle(anomalies)
        if len(normals) <= max_shots:
            raise RuntimeError(f"Not enough normal images for {category}: {len(normals)}")
        support_paths = normals[:max_shots]
        normal_tests = normals[max_shots:]
        if args.max_normal_test > 0:
            normal_tests = normal_tests[: args.max_normal_test]
        if args.max_anomaly_test > 0:
            anomalies = anomalies[: args.max_anomaly_test]

        print(f"[{cat_i}/{len(categories)}] {category}: support={len(support_paths)} normal_test={len(normal_tests)} anomaly={len(anomalies)}", flush=True)
        support_by_image = []
        expected_grid = None
        for p in support_paths:
            features, grid = extract_features(model, processor, p, args.min_pixels, args.max_pixels)
            support_by_image.append(features)
            support_manifest.append({"category": category, "image": str(p), "grid_thw": grid})
            expected_grid = expected_grid or grid

        memories = {}
        for shot in shots:
            memories[shot] = {
                layer: torch.cat([support_by_image[i][layer] for i in range(shot)], dim=0)
                for layer in layer_names
            }

        tests = [(p, 0) for p in normal_tests] + [(p, 1) for p in anomalies]
        for test_i, (path, label) in enumerate(tests, start=1):
            features, grid = extract_features(model, processor, path, args.min_pixels, args.max_pixels)
            record = {
                "category": category,
                "image": str(path),
                "label": label,
                "grid_t": grid[0],
                "grid_h": grid[1],
                "grid_w": grid[2],
                "visual_tokens": int(next(iter(features.values())).shape[0]),
            }
            for shot in shots:
                layer_distances = {}
                for layer in layer_names:
                    d = patch_distances(features[layer], memories[shot][layer])
                    layer_distances[layer] = d
                    for mode in score_modes:
                        record[f"s{shot}_{layer}_{mode}"] = aggregate(d, mode)
                stacked = torch.stack([layer_distances[layer] for layer in layer_names])
                for fusion_name, fused in [("fusion_mean", stacked.mean(0)), ("fusion_max", stacked.max(0).values)]:
                    for mode in score_modes:
                        record[f"s{shot}_{fusion_name}_{mode}"] = aggregate(fused, mode)
            per_image.append(record)
            if test_i % args.log_every == 0 or test_i == len(tests):
                print(f"  {test_i}/{len(tests)}", flush=True)
        write_csv(out / "per_image_scores.csv", per_image)

    methods = layer_names + ["fusion_mean", "fusion_max"]
    category_rows = []
    for category in categories:
        rows = [r for r in per_image if r["category"] == category]
        labels = [int(r["label"]) for r in rows]
        for shot in shots:
            for method in methods:
                for mode in score_modes:
                    key = f"s{shot}_{method}_{mode}"
                    category_rows.append({
                        "category": category,
                        "shots": shot,
                        "feature": method,
                        "aggregation": mode,
                        "normal_test": labels.count(0),
                        "anomaly_test": labels.count(1),
                        "image_auroc": binary_auc(labels, [float(r[key]) for r in rows]),
                    })

    macro_rows = []
    for shot in shots:
        for method in methods:
            for mode in score_modes:
                matches = [r for r in category_rows if r["shots"] == shot and r["feature"] == method and r["aggregation"] == mode]
                macro_rows.append({
                    "shots": shot,
                    "feature": method,
                    "aggregation": mode,
                    "categories": len(matches),
                    "macro_image_auroc": mean([r["image_auroc"] for r in matches]),
                })

    write_csv(out / "category_auroc.csv", category_rows)
    write_csv(out / "macro_auroc.csv", macro_rows)
    (out / "support_manifest.json").write_text(json.dumps(support_manifest, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"category": category_rows, "macro": macro_rows}, indent=2), encoding="utf-8")
    best = sorted(macro_rows, key=lambda x: x["macro_image_auroc"], reverse=True)[:10]
    print("Top macro results:")
    for row in best:
        print(row)


if __name__ == "__main__":
    main()
