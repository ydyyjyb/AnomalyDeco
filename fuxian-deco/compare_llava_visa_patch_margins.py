import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from probe_deco_layer_curve import build_prompt, parse_int_range, token_label


def parse_csv_words(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def first_piece_ids(tokenizer, words):
    ids = []
    detail = {}
    for word in words:
        variants = [word, " " + word] if not word.startswith(" ") else [word]
        word_ids = []
        for variant in variants:
            try:
                enc = tokenizer(variant, add_special_tokens=False).input_ids
            except Exception:
                enc = []
            if enc:
                word_ids.append(int(enc[0]))
        word_ids = sorted(set(word_ids))
        ids.extend(word_ids)
        detail[word] = word_ids
    return sorted(set(ids)), detail


def normalized_probs_over_ids(logits, token_ids):
    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    ids = torch.tensor(token_ids, dtype=torch.long, device=probs.device)
    vals = probs[ids]
    denom = vals.sum().clamp_min(1e-12)
    return vals / denom


def class_scores(candidate_ids, norm_probs, anomaly_ids, normal_ids):
    id_to_local = {int(tid): i for i, tid in enumerate(candidate_ids)}
    anomaly_score = sum(float(norm_probs[id_to_local[tid]].item()) for tid in anomaly_ids if tid in id_to_local)
    normal_score = sum(float(norm_probs[id_to_local[tid]].item()) for tid in normal_ids if tid in id_to_local)
    return anomaly_score, normal_score


def load_image_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image = parts[0]
            label = parts[1] if len(parts) > 1 else "unknown"
            records.append({"image": image, "label": label})
    return records


def preprocess_image(image_processor, image, device):
    image_tensor = image_processor.preprocess(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
    return image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)


def make_regions(image, grid, overlap):
    width, height = image.size
    regions = [
        {
            "region": "full",
            "kind": "full",
            "patch_index": 0,
            "patch_total": grid * grid if grid > 0 else 0,
            "row": -1,
            "col": -1,
            "box": (0, 0, width, height),
            "image": image,
        }
    ]
    if grid <= 0:
        return regions

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
            crop = image.crop((x0, y0, x1, y1))
            regions.append(
                {
                    "region": f"patch_r{row}_c{col}",
                    "kind": "patch",
                    "patch_index": row * grid + col + 1,
                    "patch_total": grid * grid,
                    "row": row,
                    "col": col,
                    "box": (x0, y0, x1, y1),
                    "image": crop,
                }
            )
    return regions


def build_region_question(base_question, region, grid, patch_context):
    if region["kind"] == "full":
        return base_question
    if not patch_context:
        return base_question
    return (
        f"{base_question}\n"
        f"You are seeing a local crop from a larger PCB image, not the full image. "
        f"This is patch {region['patch_index']} of {region['patch_total']} in a {grid}x{grid} grid "
        f"(row {region['row'] + 1}, column {region['col'] + 1}). "
        "Use this crop as local evidence, but remember that normal PCB structures can include holes, "
        "solder joints, traces, pads, and components. Do not call normal local PCB texture defective "
        "unless there is visible abnormal damage, stain, crack, scratch, missing material, or other defect evidence."
    )


def region_layer_rows(model, image_processor, input_ids, region, layers, candidate_ids, anomaly_ids, normal_ids):
    image_tensor = preprocess_image(image_processor, region["image"], model.device)
    with torch.inference_mode():
        logits_by_layer, _ = model(
            input_ids=input_ids,
            images=image_tensor,
            return_dict=True,
            output_hidden_states=True,
            early_exit_layers=layers,
        )

    rows = []
    x0, y0, x1, y1 = region["box"]
    for layer in layers:
        if layer not in logits_by_layer:
            continue
        layer_logits = logits_by_layer[layer][:, -1, :]
        norm_probs = normalized_probs_over_ids(layer_logits, candidate_ids)
        anomaly_score, normal_score = class_scores(candidate_ids, norm_probs, anomaly_ids, normal_ids)
        margin = anomaly_score - normal_score
        rows.append(
            {
                "region": region["region"],
                "kind": region["kind"],
                "row": region["row"],
                "col": region["col"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "layer": int(layer),
                "anomaly_score": anomaly_score,
                "normal_score": normal_score,
                "margin": margin,
                "pred": "anomaly" if anomaly_score >= normal_score else "normal",
            }
        )
    return rows


def summarize_regions(rows):
    by_region = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)

    summaries = []
    for region, region_rows in by_region.items():
        strongest = max(region_rows, key=lambda r: r["margin"])
        weakest = min(region_rows, key=lambda r: r["margin"])
        strongest_abs = max(region_rows, key=lambda r: abs(r["margin"]))
        mean_margin = sum(r["margin"] for r in region_rows) / len(region_rows)
        summaries.append(
            {
                "region": region,
                "kind": region_rows[0]["kind"],
                "row": region_rows[0]["row"],
                "col": region_rows[0]["col"],
                "x0": region_rows[0]["x0"],
                "y0": region_rows[0]["y0"],
                "x1": region_rows[0]["x1"],
                "y1": region_rows[0]["y1"],
                "max_margin": strongest["margin"],
                "max_margin_layer": strongest["layer"],
                "min_margin": weakest["margin"],
                "min_margin_layer": weakest["layer"],
                "strongest_abs_margin": strongest_abs["margin"],
                "strongest_abs_layer": strongest_abs["layer"],
                "mean_margin": mean_margin,
                "pred_by_max_margin": "anomaly" if strongest["margin"] >= 0 else "normal",
            }
        )
    return summaries


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def topk_mean(values, k):
    if not values:
        return 0.0
    values = sorted(values, reverse=True)
    k = max(1, min(int(k), len(values)))
    return mean(values[:k])


def patch_aggregate_stats(patch_summaries, topk):
    max_margins = [float(s["max_margin"]) for s in patch_summaries]
    mean_margins = [float(s["mean_margin"]) for s in patch_summaries]
    strongest_abs = [float(s["strongest_abs_margin"]) for s in patch_summaries]
    med = median(max_margins)
    best = max(patch_summaries, key=lambda s: s["max_margin"]) if patch_summaries else None
    return {
        "patch_count": len(patch_summaries),
        "best_patch": best["region"] if best else "",
        "best_patch_max_margin": best["max_margin"] if best else None,
        "best_patch_max_margin_layer": best["max_margin_layer"] if best else None,
        "patch_max_margin_mean": mean(max_margins),
        "patch_max_margin_std": std(max_margins),
        "patch_max_margin_median": med,
        "patch_topk_max_margin_mean": topk_mean(max_margins, topk),
        "patch_max_minus_median": (max(max_margins) - med) if max_margins else 0.0,
        "patch_mean_margin_mean": mean(mean_margins),
        "patch_mean_margin_std": std(mean_margins),
        "patch_strongest_abs_mean": mean(strongest_abs),
        "patch_strongest_abs_std": std(strongest_abs),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_full_vs_patch(rows, output_png):
    full = [r for r in rows if r["kind"] == "full"]
    patches = [r for r in rows if r["kind"] == "patch"]
    layers = sorted(set(r["layer"] for r in rows))

    full_by_layer = {r["layer"]: r["margin"] for r in full}
    patch_by_layer = {}
    for layer in layers:
        vals = [r["margin"] for r in patches if r["layer"] == layer]
        if vals:
            patch_by_layer[layer] = {"max": max(vals), "mean": sum(vals) / len(vals), "min": min(vals)}

    plt.figure(figsize=(10, 5.8))
    plt.plot(layers, [full_by_layer.get(layer, float("nan")) for layer in layers], marker="o", label="full image")
    plt.plot(layers, [patch_by_layer[layer]["max"] for layer in layers], marker="o", label="patch max")
    plt.plot(layers, [patch_by_layer[layer]["mean"] for layer in layers], marker="o", label="patch mean")
    plt.plot(layers, [patch_by_layer[layer]["min"] for layer in layers], marker="o", label="patch min")
    plt.axhline(0, color="black", linestyle=":", linewidth=1.2)
    plt.xlabel("Layer")
    plt.ylabel("Anomaly score - normal score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_patch_heatmap(image, summaries, output_png, score_key):
    patch_summaries = [s for s in summaries if s["kind"] == "patch"]
    if not patch_summaries:
        return

    grid = max(max(s["row"], s["col"]) for s in patch_summaries) + 1
    vals = [[0.0 for _ in range(grid)] for _ in range(grid)]
    for s in patch_summaries:
        vals[s["row"]][s["col"]] = float(s[score_key])

    plt.figure(figsize=(6.5, 5.6))
    im = plt.imshow(vals, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=score_key)
    for r in range(grid):
        for c in range(grid):
            plt.text(c, r, f"{vals[r][c]:.2f}", ha="center", va="center", color="black", fontsize=9)
    plt.xticks(range(grid), [f"c{c}" for c in range(grid)])
    plt.yticks(range(grid), [f"r{r}" for r in range(grid)])
    plt.title("Patch anomaly-normal margin")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for s in patch_summaries:
        val = max(-1.0, min(1.0, float(s[score_key])))
        if val >= 0:
            color = (255, 0, 0, int(35 + 120 * val))
        else:
            color = (0, 80, 255, int(35 + 120 * abs(val)))
        draw.rectangle((s["x0"], s["y0"], s["x1"], s["y1"]), outline=(255, 255, 255, 220), width=3, fill=color)
    overlay.save(output_png.with_name(output_png.stem + "_overlay.jpg"))


def create_relocalized_image(image, patch_summaries, threshold, dim_alpha, output_path):
    selected = [s for s in patch_summaries if float(s["mean_margin"]) >= float(threshold)]
    if not selected and patch_summaries:
        selected = [max(patch_summaries, key=lambda s: s["mean_margin"])]

    image = image.convert("RGB")
    muted = Image.blend(image, Image.new("RGB", image.size, (32, 32, 32)), float(dim_alpha))
    relocalized = muted.copy()
    for s in selected:
        box = (s["x0"], s["y0"], s["x1"], s["y1"])
        relocalized.paste(image.crop(box), box)
    draw = ImageDraw.Draw(relocalized, "RGBA")
    for s in selected:
        draw.rectangle((s["x0"], s["y0"], s["x1"], s["y1"]), outline=(255, 230, 0, 240), width=4)
    relocalized.save(output_path)
    return relocalized, selected


def build_relocalized_question(base_question, selected_count, threshold):
    return (
        f"{base_question}\n"
        f"The image is a full PCB view after patch screening. Regions whose patch mean anomaly-normal "
        f"margin is at least {threshold:.3f} are kept clear and outlined, while other regions are dimmed. "
        f"There are {selected_count} highlighted suspected regions. Use the full-board context and the "
        "highlighted regions together; do not classify normal PCB structures as defects unless abnormal "
        "damage evidence is visible."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-list", required=True, help="Text file: /path/to/image label per line.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="1:33")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--topk-patches", type=int, default=3)
    parser.add_argument("--patch-context", action="store_true", help="Tell the model when the input is a patch and where it is in the grid.")
    parser.add_argument("--relocalize-mean-threshold", type=float, default=None, help="If set, keep patches with mean_margin >= threshold and rerun on the masked full image.")
    parser.add_argument("--relocalize-dim-alpha", type=float, default=0.65)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--anomaly-words",
        default="defect,defective,anomaly,abnormal,scratch,crack,damage,damaged,stain,hole,flaw,faulty",
    )
    parser.add_argument("--normal-words", default="normal,standard,typical,clean,uniform,regular,good,intact")
    args = parser.parse_args()

    disable_torch_init()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_image_records(args.image_list)
    layers = parse_int_range(args.layers)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device="cuda",
    )
    model.eval()

    anomaly_ids, anomaly_detail = first_piece_ids(tokenizer, parse_csv_words(args.anomaly_words))
    normal_ids, normal_detail = first_piece_ids(tokenizer, parse_csv_words(args.normal_words))
    candidate_ids = sorted(set(anomaly_ids + normal_ids))

    all_layer_rows = []
    all_region_summaries = []
    image_summaries = []
    for record in records:
        image_path = record["image"]
        image = Image.open(image_path).convert("RGB")
        image_id = Path(image_path).stem
        label = record["label"]
        image_dir = out_dir / image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        regions = make_regions(image, args.grid, args.overlap)
        image_rows = []
        for region in regions:
            region_question = build_region_question(args.question, region, args.grid, args.patch_context)
            prompt = build_prompt(model, args.conv_mode, region_question, args.prefix)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            input_ids = input_ids.to(model.device)
            rows = region_layer_rows(model, image_processor, input_ids, region, layers, candidate_ids, anomaly_ids, normal_ids)
            for row in rows:
                row["image"] = image_path
                row["image_id"] = image_id
                row["label"] = label
                row["patch_index"] = region["patch_index"]
                row["patch_total"] = region["patch_total"]
                row["region_question"] = region_question
            image_rows.extend(rows)
            all_layer_rows.extend(rows)

        region_summaries = summarize_regions(image_rows)
        for summary in region_summaries:
            summary["image"] = image_path
            summary["image_id"] = image_id
            summary["label"] = label
        all_region_summaries.extend(region_summaries)

        full_summary = next(s for s in region_summaries if s["kind"] == "full")
        patch_summaries = [s for s in region_summaries if s["kind"] == "patch"]
        patch_stats = patch_aggregate_stats(patch_summaries, args.topk_patches)
        relocalized_stats = {
            "relocalized_patch_count": 0,
            "relocalized_image": "",
            "relocalized_max_margin": None,
            "relocalized_max_margin_layer": None,
            "relocalized_mean_margin": None,
            "relocalized_gain_over_full": None,
            "pred_relocalized": "not_run",
        }
        if args.relocalize_mean_threshold is not None:
            relocalized_path = image_dir / "relocalized_mean_mask.jpg"
            relocalized_image, selected_patches = create_relocalized_image(
                image,
                patch_summaries,
                args.relocalize_mean_threshold,
                args.relocalize_dim_alpha,
                relocalized_path,
            )
            relocalized_region = {
                "region": "relocalized_mean_mask",
                "kind": "relocalized",
                "patch_index": 0,
                "patch_total": len(selected_patches),
                "row": -1,
                "col": -1,
                "box": (0, 0, image.size[0], image.size[1]),
                "image": relocalized_image,
            }
            relocalized_question = build_relocalized_question(
                args.question,
                len(selected_patches),
                args.relocalize_mean_threshold,
            )
            relocalized_prompt = build_prompt(model, args.conv_mode, relocalized_question, args.prefix)
            relocalized_input_ids = tokenizer_image_token(
                relocalized_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
            relocalized_rows = region_layer_rows(
                model,
                image_processor,
                relocalized_input_ids,
                relocalized_region,
                layers,
                candidate_ids,
                anomaly_ids,
                normal_ids,
            )
            for row in relocalized_rows:
                row["image"] = image_path
                row["image_id"] = image_id
                row["label"] = label
                row["patch_index"] = 0
                row["patch_total"] = len(selected_patches)
                row["region_question"] = relocalized_question
            write_csv(image_dir / "relocalized_layer_margins.csv", relocalized_rows)
            all_layer_rows.extend(relocalized_rows)
            relocalized_summary = summarize_regions(relocalized_rows)[0]
            relocalized_stats = {
                "relocalized_patch_count": len(selected_patches),
                "relocalized_image": str(relocalized_path),
                "relocalized_max_margin": relocalized_summary["max_margin"],
                "relocalized_max_margin_layer": relocalized_summary["max_margin_layer"],
                "relocalized_mean_margin": relocalized_summary["mean_margin"],
                "relocalized_gain_over_full": relocalized_summary["max_margin"] - full_summary["max_margin"],
                "pred_relocalized": "anomaly" if relocalized_summary["max_margin"] >= 0 else "normal",
            }
        image_summary = {
            "image": image_path,
            "image_id": image_id,
            "label": label,
            "full_max_margin": full_summary["max_margin"],
            "full_max_margin_layer": full_summary["max_margin_layer"],
            **patch_stats,
            "patch_gain_over_full": (
                patch_stats["best_patch_max_margin"] - full_summary["max_margin"]
                if patch_stats["best_patch_max_margin"] is not None
                else None
            ),
            "topk_patch_gain_over_full": patch_stats["patch_topk_max_margin_mean"] - full_summary["max_margin"],
            "mean_patch_gain_over_full": patch_stats["patch_max_margin_mean"] - full_summary["max_margin"],
            "pred_full": "anomaly" if full_summary["max_margin"] >= 0 else "normal",
            "pred_patchmax": "anomaly" if patch_stats["best_patch_max_margin"] is not None and patch_stats["best_patch_max_margin"] >= 0 else "normal",
            "pred_patch_topk_mean": "anomaly" if patch_stats["patch_topk_max_margin_mean"] >= 0 else "normal",
            "pred_patch_mean": "anomaly" if patch_stats["patch_max_margin_mean"] >= 0 else "normal",
            **relocalized_stats,
        }
        image_summaries.append(image_summary)

        write_csv(image_dir / "layer_region_margins.csv", image_rows)
        write_csv(image_dir / "region_summary.csv", region_summaries)
        plot_full_vs_patch(image_rows, image_dir / "full_vs_patch_margin_by_layer.png")
        plot_patch_heatmap(image, region_summaries, image_dir / "patch_max_margin_heatmap.png", "max_margin")
        plot_patch_heatmap(image, region_summaries, image_dir / "patch_mean_margin_heatmap.png", "mean_margin")
        print(
            f"{image_id}: label={label} full_max={image_summary['full_max_margin']:.4f} "
            f"patch_max={image_summary['best_patch_max_margin']:.4f} "
            f"top{args.topk_patches}_mean={image_summary['patch_topk_max_margin_mean']:.4f} "
            f"std={image_summary['patch_max_margin_std']:.4f} "
            f"max-med={image_summary['patch_max_minus_median']:.4f} "
            f"gain={image_summary['patch_gain_over_full']:.4f} best={image_summary['best_patch']} "
            f"relocalized={image_summary['relocalized_max_margin']}"
        )

    write_csv(out_dir / "all_layer_region_margins.csv", all_layer_rows)
    write_csv(out_dir / "all_region_summary.csv", all_region_summaries)
    write_csv(out_dir / "image_summary.csv", image_summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "question": args.question,
                "prefix": args.prefix,
                "prompt": prompt,
                "layers": layers,
                "grid": args.grid,
                "overlap": args.overlap,
                "topk_patches": args.topk_patches,
                "patch_context": args.patch_context,
                "relocalize_mean_threshold": args.relocalize_mean_threshold,
                "relocalize_dim_alpha": args.relocalize_dim_alpha,
                "candidate_ids": candidate_ids,
                "candidate_tokens": [{"token_id": tid, "token": token_label(tokenizer, tid)} for tid in candidate_ids],
                "anomaly_tokenization": anomaly_detail,
                "normal_tokenization": normal_detail,
                "image_summaries": image_summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
