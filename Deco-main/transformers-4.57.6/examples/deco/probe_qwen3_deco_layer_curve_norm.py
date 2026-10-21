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
    first_token_id,
    get_norm_and_lm_head,
    logits_by_layer_from_hidden_states,
    parse_int_range,
    prepare_inputs,
    token_label,
)


def normalized_probs_over_ids(logits, token_ids):
    probs = F.softmax(logits.float(), dim=-1).squeeze(0)
    ids = torch.tensor(token_ids, dtype=torch.long, device=probs.device)
    vals = probs[ids]
    denom = vals.sum().clamp_min(1e-12)
    return vals / denom


def build_final_candidates(final_logits, top_k, top_p):
    final_probs = F.softmax(final_logits.float(), dim=-1).squeeze(0)
    candidate_probs, candidate_ids = torch.topk(final_probs, dim=-1, k=top_k)
    cumulative = candidate_probs.cumsum(dim=-1)
    cutoff = torch.searchsorted(cumulative, torch.tensor(top_p, device=cumulative.device), right=False)
    cutoff = min(int(cutoff.item()) + 1, top_k)
    candidate_ids = candidate_ids[:cutoff]
    candidate_probs = candidate_probs[:cutoff]
    norm_candidate_probs = candidate_probs / candidate_probs.sum().clamp_min(1e-12)
    return [int(x) for x in candidate_ids.tolist()], [float(x) for x in candidate_probs.tolist()], [float(x) for x in norm_candidate_probs.tolist()]


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
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                enc = []
            if enc:
                word_ids.append(int(enc[0]))
        word_ids = sorted(set(word_ids))
        ids.extend(word_ids)
        detail[word] = word_ids
    return sorted(set(ids)), detail


def select_deco_anchor_normalized(
    logits_by_layer,
    final_logits,
    early_exit_layers,
    top_k,
    top_p,
    alpha,
    candidate_mode,
    class_candidate_ids,
):
    if candidate_mode == "class_words":
        candidate_ids = list(class_candidate_ids)
        final_probs = F.softmax(final_logits.float(), dim=-1).squeeze(0)
        raw_final_probs = [float(final_probs[token_id].item()) for token_id in candidate_ids]
        norm_final_probs = [float(x.item()) for x in normalized_probs_over_ids(final_logits, candidate_ids)]
    else:
        candidate_ids, raw_final_probs, norm_final_probs = build_final_candidates(final_logits, top_k, top_p)

    if not candidate_ids:
        raise RuntimeError("No candidate token ids available.")

    layer_norm_rows = []
    best = None
    for layer in early_exit_layers:
        if layer not in logits_by_layer:
            continue
        norm_vals = normalized_probs_over_ids(logits_by_layer[layer], candidate_ids)
        layer_norm_rows.append((layer, norm_vals))
        local_idx = int(torch.argmax(norm_vals).item())
        score = float(norm_vals[local_idx].item())
        if best is None or score > best["score"]:
            best = {"layer": int(layer), "local_idx": local_idx, "score": score}

    if best is None:
        raise RuntimeError("No valid early-exit layer found for normalized DeCo selection.")

    selected_layer = best["layer"]
    selected_token_id = int(candidate_ids[best["local_idx"]])
    normalized_max_prob = float(best["score"])

    selected_logits = logits_by_layer[selected_layer].float()
    mask = torch.ones_like(selected_logits, dtype=torch.bool)
    mask[:, torch.tensor(candidate_ids, dtype=torch.long, device=selected_logits.device)] = False
    calibrated_logits = final_logits.float() + float(alpha) * normalized_max_prob * selected_logits
    calibrated_logits = calibrated_logits.masked_fill(mask, -float("inf"))
    calibrated_norm_probs = normalized_probs_over_ids(calibrated_logits, candidate_ids)
    calibrated_local_idx = int(torch.argmax(calibrated_norm_probs).item())
    calibrated_token_id = int(candidate_ids[calibrated_local_idx])

    return {
        "candidate_ids": candidate_ids,
        "candidate_final_probs_raw": raw_final_probs,
        "candidate_final_probs_norm": norm_final_probs,
        "selected_layer": selected_layer,
        "selected_token_id": selected_token_id,
        "selected_candidate_normalized_prob": normalized_max_prob,
        "calibrated_token_id": calibrated_token_id,
        "calibrated_candidate_normalized_prob": float(calibrated_norm_probs[calibrated_local_idx].item()),
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
            [r["normalized_prob"] for r in token_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=token,
        )
    plt.axvline(selected_layer, color="red", linestyle="--", linewidth=1.5, label=f"Norm-DeCo layer {selected_layer}")
    plt.xlabel("Layer")
    plt.ylabel("Candidate-set normalized probability")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--question", default="Please describe this image in detail.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidate-words", default="")
    parser.add_argument(
        "--candidate-mode",
        default="class_words",
        choices=["class_words", "final_topk"],
        help="class_words uses anomaly/normal lexicon as a closed candidate set; final_topk follows DeCo top-k/top-p candidates.",
    )
    parser.add_argument(
        "--anomaly-words",
        default="defect,defective,anomaly,abnormal,scratch,crack,damage,damaged,stain,hole,flaw,faulty",
    )
    parser.add_argument("--normal-words", default="normal,standard,typical,clean,uniform,regular,good,intact")
    parser.add_argument("--early-exit-layers", default="20:29")
    parser.add_argument("--plot-layers", default="1:37")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--output-dir", required=True)
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

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    anomaly_ids, anomaly_detail = first_piece_ids(tokenizer, parse_csv_words(args.anomaly_words))
    normal_ids, normal_detail = first_piece_ids(tokenizer, parse_csv_words(args.normal_words))
    class_candidate_ids = sorted(set(anomaly_ids + normal_ids))
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

    inputs, prompt_text = prepare_inputs(
        processor,
        model,
        args.image_file,
        args.question,
        args.prefix,
        args.min_pixels,
        args.max_pixels,
    )

    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("No hidden_states returned.")

    early_exit_layers = parse_int_range(args.early_exit_layers)
    plot_layers = parse_int_range(args.plot_layers)
    all_layers = sorted(set(early_exit_layers + plot_layers))
    logits_by_layer = logits_by_layer_from_hidden_states(model, outputs.hidden_states, all_layers)

    final_logits = outputs.logits[:, -1, :].float()
    final_probs = F.softmax(final_logits, dim=-1).squeeze(0)
    final_token_id = int(torch.argmax(final_probs).item())
    trace = select_deco_anchor_normalized(
        logits_by_layer,
        final_logits,
        early_exit_layers,
        args.top_k,
        args.top_p,
        args.alpha,
        args.candidate_mode,
        class_candidate_ids,
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
        if layer not in logits_by_layer:
            continue
        norm_vals = normalized_probs_over_ids(logits_by_layer[layer], plot_token_ids)
        for local_idx, token_id in enumerate(plot_token_ids):
            rows.append(
                {
                    "layer": int(layer),
                    "token_id": int(token_id),
                    "token": token_label(tokenizer, token_id),
                    "normalized_prob": float(norm_vals[local_idx].item()),
                    "is_deco_candidate": token_id in trace["candidate_ids"],
                    "is_selected_token": token_id == trace["selected_token_id"],
                }
            )

    csv_path = out_dir / "layer_token_probs.normalized.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "layer_token_probs.normalized.png"
    plot_curves(rows, trace["selected_layer"], png_path)

    summary = {
        "mode": "candidate_set_normalized",
        "normalization_note": (
            "If candidate_mode=final_topk, the final candidate set is formed from last-layer top-k/top-p "
            "raw probabilities. If candidate_mode=class_words, anomaly/normal lexicon tokens are used as a "
            "closed candidate set. Preceding-layer curves and DeCo layer selection use probabilities "
            "normalized within the candidate set. Plot rows normalize over plotted tokens, while DeCo "
            "selection normalizes over DeCo candidate tokens only."
        ),
        "model_path": args.model_path,
        "image_file": args.image_file,
        "question": args.question,
        "prefix": args.prefix,
        "prompt_text": prompt_text,
        "num_hidden_states": len(outputs.hidden_states),
        "early_exit_layers": early_exit_layers,
        "plot_layers": plot_layers,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha": args.alpha,
        "candidate_mode": args.candidate_mode,
        "anomaly_tokenization": anomaly_detail,
        "normal_tokenization": normal_detail,
        "class_candidate_ids": class_candidate_ids,
        "final_top_token_id": final_token_id,
        "final_top_token": token_label(tokenizer, final_token_id),
        "final_top_decoded": tokenizer.decode([final_token_id], skip_special_tokens=False),
        "final_top_prob_raw": float(final_probs[final_token_id].item()),
        "deco_selected_layer": trace["selected_layer"],
        "deco_selected_token_id": trace["selected_token_id"],
        "deco_selected_token": token_label(tokenizer, trace["selected_token_id"]),
        "deco_selected_decoded": tokenizer.decode([trace["selected_token_id"]], skip_special_tokens=False),
        "deco_selected_candidate_normalized_prob": trace["selected_candidate_normalized_prob"],
        "deco_calibrated_token_id": trace["calibrated_token_id"],
        "deco_calibrated_token": token_label(tokenizer, trace["calibrated_token_id"]),
        "deco_calibrated_decoded": tokenizer.decode([trace["calibrated_token_id"]], skip_special_tokens=False),
        "deco_calibrated_candidate_normalized_prob": trace["calibrated_candidate_normalized_prob"],
        "candidate_tokens": [
            {
                "rank": i + 1,
                "token_id": token_id,
                "token": token_label(tokenizer, token_id),
                "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
                "final_prob_raw": trace["candidate_final_probs_raw"][i],
                "final_prob_candidate_norm": trace["candidate_final_probs_norm"][i],
            }
            for i, token_id in enumerate(trace["candidate_ids"])
        ],
        "candidate_word_tokenization": {
            word: [{"token_id": int(tid), "token": token_label(tokenizer, tid)} for tid in ids]
            for word, ids in word_token_notes.items()
        },
        "files": {"csv": str(csv_path), "plot": str(png_path)},
    }

    json_path = out_dir / "deco_layer_trace.normalized.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===== Qwen3-VL Candidate-Normalized DeCo Probe =====")
    print(f"final top token: {summary['final_top_decoded']!r} raw_p={summary['final_top_prob_raw']:.6f}")
    print(
        "Norm-DeCo selected layer: "
        f"{summary['deco_selected_layer']} "
        f"token={summary['deco_selected_decoded']!r} "
        f"candidate_norm_prob={summary['deco_selected_candidate_normalized_prob']:.6f}"
    )
    print(
        "Norm-DeCo calibrated top token: "
        f"{summary['deco_calibrated_decoded']!r} "
        f"candidate_norm_prob={summary['deco_calibrated_candidate_normalized_prob']:.6f}"
    )
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
