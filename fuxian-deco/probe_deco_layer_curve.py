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
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_int_range(value):
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(x) for x in value.split(",") if x.strip()]


def clean_token(token):
    token = token.replace("<0x0A>", "\\n")
    token = token.replace("\u2581", " ")
    return token.strip() or token


def token_label(tokenizer, token_id):
    return clean_token(tokenizer.convert_ids_to_tokens([int(token_id)])[0])


def first_token_id(tokenizer, text):
    ids = tokenizer(" " + text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Could not tokenize candidate word: {text!r}")
    return int(ids[0]), ids


def build_prompt(model, conv_mode, question, prefix):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if prefix:
        prompt += " " + prefix.strip()
    return prompt


def load_image_tensor(image_processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)


def select_deco_anchor(logits_by_layer, final_logits, early_exit_layers, top_k, top_p):
    final_probs = F.softmax(final_logits, dim=-1).squeeze(0)
    candidate_probs, candidate_ids = torch.topk(final_probs, dim=-1, k=top_k)
    cumulative = candidate_probs.cumsum(dim=-1)
    cutoff = torch.searchsorted(cumulative, torch.tensor(top_p, device=cumulative.device), right=False)
    cutoff = min(int(cutoff.item()) + 1, top_k)
    candidate_ids = candidate_ids[:cutoff]
    candidate_probs = candidate_probs[:cutoff]

    stacked = torch.stack([logits_by_layer[layer][:, -1, :] for layer in early_exit_layers], dim=0)
    early_probs = F.softmax(stacked, dim=-1)
    candidate_early_probs = early_probs[:, :, candidate_ids].squeeze(1)
    flat_idx = int(torch.argmax(candidate_early_probs).item())
    layer_idx = flat_idx // candidate_early_probs.size(1)
    token_idx = flat_idx % candidate_early_probs.size(1)

    selected_layer = int(early_exit_layers[layer_idx])
    selected_token_id = int(candidate_ids[token_idx].item())
    max_prob = float(candidate_early_probs[layer_idx, token_idx].item())

    selected_layer_logits = logits_by_layer[selected_layer][:, -1, :]
    indices_to_remove = torch.ones_like(selected_layer_logits, dtype=torch.bool)
    indices_to_remove[:, candidate_ids] = False
    calibrated_logits = final_logits + selected_layer_logits * max_prob
    calibrated_logits = calibrated_logits.masked_fill(indices_to_remove, -float("inf"))
    calibrated_probs = F.softmax(calibrated_logits, dim=-1).squeeze(0)
    calibrated_token_id = int(torch.argmax(calibrated_probs).item())

    return {
        "candidate_ids": [int(x) for x in candidate_ids.tolist()],
        "candidate_final_probs": [float(x) for x in candidate_probs.tolist()],
        "candidate_early_probs": candidate_early_probs.detach().float().cpu(),
        "selected_layer": selected_layer,
        "selected_token_id": selected_token_id,
        "selected_prob": max_prob,
        "calibrated_token_id": calibrated_token_id,
        "calibrated_token_prob": float(calibrated_probs[calibrated_token_id].item()),
    }


def plot_curves(rows, selected_layer, output_png):
    by_token = {}
    for row in rows:
        by_token.setdefault(row["token"], []).append(row)

    plt.figure(figsize=(10, 5.8))
    for token, token_rows in by_token.items():
        token_rows = sorted(token_rows, key=lambda r: r["layer"])
        plt.plot(
            [r["layer"] for r in token_rows],
            [r["prob"] for r in token_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=token,
        )
    plt.axvline(selected_layer, color="red", linestyle="--", linewidth=1.5, label=f"DeCo layer {selected_layer}")
    plt.xlabel("Layer")
    plt.ylabel("Next-token probability")
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--question", default="Please describe this image in detail.")
    parser.add_argument("--prefix", required=True, help="Assistant prefix before the probed next token.")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--early-exit-layers", default="20:29")
    parser.add_argument("--plot-layers", default="1:33")
    parser.add_argument("--candidate-words", default="", help="Optional comma-separated words to add to the plot.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    disable_torch_init()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device="cuda",
    )
    model.eval()

    prompt = build_prompt(model, args.conv_mode, args.question, args.prefix)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)
    image_tensor = load_image_tensor(image_processor, args.image_file, model.device)

    early_exit_layers = parse_int_range(args.early_exit_layers)
    plot_layers = parse_int_range(args.plot_layers)

    with torch.inference_mode():
        logits_by_layer, outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            return_dict=True,
            output_hidden_states=True,
            early_exit_layers=sorted(set(early_exit_layers + plot_layers)),
        )

    final_layer = int(model.config.num_hidden_layers)
    final_logits = outputs.logits[:, -1, :]
    final_probs = F.softmax(final_logits, dim=-1).squeeze(0)
    final_token_id = int(torch.argmax(final_probs).item())

    trace = select_deco_anchor(
        logits_by_layer=logits_by_layer,
        final_logits=final_logits,
        early_exit_layers=early_exit_layers,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    plot_token_ids = list(trace["candidate_ids"])
    word_token_notes = {}
    for word in [w.strip() for w in args.candidate_words.split(",") if w.strip()]:
        tid, ids = first_token_id(tokenizer, word)
        plot_token_ids.append(tid)
        word_token_notes[word] = ids
    plot_token_ids = sorted(set(plot_token_ids))

    rows = []
    for layer in plot_layers:
        layer_logits = logits_by_layer[layer][:, -1, :]
        probs = F.softmax(layer_logits, dim=-1).squeeze(0)
        for token_id in plot_token_ids:
            rows.append(
                {
                    "layer": int(layer),
                    "token_id": int(token_id),
                    "token": token_label(tokenizer, token_id),
                    "prob": float(probs[token_id].item()),
                    "is_deco_candidate": token_id in trace["candidate_ids"],
                    "is_selected_token": token_id == trace["selected_token_id"],
                }
            )

    csv_path = out_dir / "layer_token_probs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "layer_token_probs.png"
    plot_curves(rows, trace["selected_layer"], png_path)

    summary = {
        "image_file": args.image_file,
        "question": args.question,
        "prefix": args.prefix,
        "early_exit_layers": early_exit_layers,
        "plot_layers": plot_layers,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha": args.alpha,
        "final_layer": final_layer,
        "final_top_token_id": final_token_id,
        "final_top_token": token_label(tokenizer, final_token_id),
        "final_top_prob": float(final_probs[final_token_id].item()),
        "deco_selected_layer": trace["selected_layer"],
        "deco_selected_token_id": trace["selected_token_id"],
        "deco_selected_token": token_label(tokenizer, trace["selected_token_id"]),
        "deco_selected_preceding_prob": trace["selected_prob"],
        "deco_calibrated_token_id": trace["calibrated_token_id"],
        "deco_calibrated_token": token_label(tokenizer, trace["calibrated_token_id"]),
        "deco_calibrated_token_prob": trace["calibrated_token_prob"],
        "candidate_tokens": [
            {
                "rank": i + 1,
                "token_id": token_id,
                "token": token_label(tokenizer, token_id),
                "final_prob": trace["candidate_final_probs"][i],
            }
            for i, token_id in enumerate(trace["candidate_ids"])
        ],
        "candidate_word_tokenization": {
            word: [{"token_id": int(tid), "token": token_label(tokenizer, tid)} for tid in ids]
            for word, ids in word_token_notes.items()
        },
        "files": {
            "csv": str(csv_path),
            "plot": str(png_path),
        },
    }

    json_path = out_dir / "deco_layer_trace.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===== DeCo Layer Probe =====")
    print(f"final top token: {summary['final_top_token']!r} p={summary['final_top_prob']:.6f}")
    print(
        "DeCo selected layer: "
        f"{summary['deco_selected_layer']} "
        f"token={summary['deco_selected_token']!r} "
        f"preceding_prob={summary['deco_selected_preceding_prob']:.6f}"
    )
    print(
        "DeCo calibrated top token: "
        f"{summary['deco_calibrated_token']!r} p={summary['deco_calibrated_token_prob']:.6f}"
    )
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
