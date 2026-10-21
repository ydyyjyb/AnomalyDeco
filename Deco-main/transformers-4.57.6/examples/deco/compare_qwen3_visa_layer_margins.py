import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from probe_qwen3_deco_layer_curve import (
    Qwen3VLForConditionalGeneration,
    logits_by_layer_from_hidden_states,
    parse_int_range,
    prepare_inputs,
    token_label,
)
from probe_qwen3_deco_layer_curve_norm import first_piece_ids, normalized_probs_over_ids, parse_csv_words


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


def plot_margin(rows, output_png):
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)

    plt.figure(figsize=(10, 5.8))
    for image_id, image_rows in by_image.items():
        image_rows = sorted(image_rows, key=lambda r: r["layer"])
        label = image_rows[0]["label"]
        linestyle = "-" if label == "anomaly" else "--"
        plt.plot(
            [r["layer"] for r in image_rows],
            [r["margin"] for r in image_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.2,
            linestyle=linestyle,
            label=f"{image_id} ({label})",
        )
    plt.axhline(0, color="black", linestyle=":", linewidth=1.2)
    plt.xlabel("Layer")
    plt.ylabel("Anomaly score - normal score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-list", required=True, help="Text file: /path/to/image label per line.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="1:37")
    parser.add_argument(
        "--anomaly-words",
        default="defect,defective,anomaly,abnormal,scratch,crack,damage,damaged,stain,hole,flaw,faulty",
    )
    parser.add_argument("--normal-words", default="normal,standard,typical,clean,uniform,regular,good,intact")
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
    records = load_image_records(args.image_list)
    layers = parse_int_range(args.layers)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    anomaly_ids, anomaly_detail = first_piece_ids(tokenizer, parse_csv_words(args.anomaly_words))
    normal_ids, normal_detail = first_piece_ids(tokenizer, parse_csv_words(args.normal_words))
    candidate_ids = sorted(set(anomaly_ids + normal_ids))

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

    rows = []
    summaries = []
    for record in records:
        image_path = record["image"]
        image_id = Path(image_path).stem
        label = record["label"]
        inputs, prompt_text = prepare_inputs(
            processor,
            model,
            image_path,
            args.question,
            args.prefix,
            args.min_pixels,
            args.max_pixels,
        )
        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
        logits_by_layer = logits_by_layer_from_hidden_states(model, outputs.hidden_states, layers)
        best = None
        for layer in layers:
            if layer not in logits_by_layer:
                continue
            norm_probs = normalized_probs_over_ids(logits_by_layer[layer], candidate_ids)
            anomaly_score, normal_score = class_scores(candidate_ids, norm_probs, anomaly_ids, normal_ids)
            margin = anomaly_score - normal_score
            row = {
                "image": image_path,
                "image_id": image_id,
                "label": label,
                "layer": int(layer),
                "anomaly_score": anomaly_score,
                "normal_score": normal_score,
                "margin": margin,
                "pred": "anomaly" if anomaly_score >= normal_score else "normal",
            }
            rows.append(row)
            if best is None or abs(margin) > abs(best["margin"]):
                best = row
        summaries.append(
            {
                "image": image_path,
                "image_id": image_id,
                "label": label,
                "strongest_layer": best["layer"],
                "strongest_margin": best["margin"],
                "strongest_pred": best["pred"],
            }
        )
        print(
            f"{image_id}: label={label} strongest_layer={best['layer']} "
            f"margin={best['margin']:.4f} pred={best['pred']}"
        )

    csv_path = out_dir / "layer_class_margins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "question": args.question,
                "prefix": args.prefix,
                "prompt_text_example": prompt_text,
                "layers": layers,
                "candidate_ids": candidate_ids,
                "candidate_tokens": [{"token_id": tid, "token": token_label(tokenizer, tid)} for tid in candidate_ids],
                "anomaly_tokenization": anomaly_detail,
                "normal_tokenization": normal_detail,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    png_path = out_dir / "anomaly_normal_margin_by_layer.png"
    plot_margin(rows, png_path)

    print(f"saved: {csv_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
