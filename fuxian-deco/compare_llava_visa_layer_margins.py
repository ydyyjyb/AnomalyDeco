import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from probe_deco_layer_curve import build_prompt, load_image_tensor, parse_int_range, token_label


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
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-list", required=True, help="Text file: /path/to/image label per line.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="1:33")
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

    prompt = build_prompt(model, args.conv_mode, args.question, args.prefix)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)

    rows = []
    summaries = []
    for record in records:
        image_path = record["image"]
        image_id = Path(image_path).stem
        label = record["label"]
        image_tensor = load_image_tensor(image_processor, image_path, model.device)

        with torch.inference_mode():
            logits_by_layer, _ = model(
                input_ids=input_ids,
                images=image_tensor,
                return_dict=True,
                output_hidden_states=True,
                early_exit_layers=layers,
            )

        best = None
        for layer in layers:
            if layer not in logits_by_layer:
                continue
            layer_logits = logits_by_layer[layer][:, -1, :]
            norm_probs = normalized_probs_over_ids(layer_logits, candidate_ids)
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

    png_path = out_dir / "anomaly_normal_margin_by_layer.png"
    plot_margin(rows, png_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "question": args.question,
                "prefix": args.prefix,
                "prompt": prompt,
                "layers": layers,
                "candidate_ids": candidate_ids,
                "candidate_tokens": [{"token_id": tid, "token": token_label(tokenizer, tid)} for tid in candidate_ids],
                "anomaly_tokenization": anomaly_detail,
                "normal_tokenization": normal_detail,
                "summaries": summaries,
                "files": {"csv": str(csv_path), "plot": str(png_path)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"saved: {csv_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
