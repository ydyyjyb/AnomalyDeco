import argparse
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

from probe_qwen3_deco_layer_curve import Qwen3VLForConditionalGeneration, get_norm_and_lm_head, parse_int_range
from eval_qwen3_visa_patch_phrase_auroc import (
    accuracy,
    binary_auc,
    collect_visa_samples,
    load_mask,
)
from eval_qwen3_visa_real_lite import (
    best_accuracy,
    build_real_question,
    category_defect_description,
    generate_reasoning,
    processor_inputs,
    real_lite_attention_map,
    resize_score_values_from_tensor,
    save_heatmap_overlay,
    save_map_only,
    token_to_word,
    write_csv,
    write_json,
)


def safe_div(num, den):
    return float(num) / float(den) if den else None


def mean_defined(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def f1_from_precision_recall(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def binary_classification_metrics(labels, preds):
    labels = [int(x) for x in labels]
    preds = [int(x) for x in preds]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = len(labels)
    anomaly_total = tp + fn
    normal_total = tn + fp
    anomaly_recall = safe_div(tp, anomaly_total)
    normal_recall = safe_div(tn, normal_total)
    balanced_acc = None
    if anomaly_recall is not None and normal_recall is not None:
        balanced_acc = 0.5 * (anomaly_recall + normal_recall)
    precision_anomaly = safe_div(tp, tp + fp)
    precision_normal = safe_div(tn, tn + fn)
    if anomaly_total and precision_anomaly is None:
        precision_anomaly = 0.0
    if normal_total and precision_normal is None:
        precision_normal = 0.0
    f1_anomaly = f1_from_precision_recall(precision_anomaly, anomaly_recall)
    f1_normal = f1_from_precision_recall(precision_normal, normal_recall)
    recall_all = mean_defined([normal_recall, anomaly_recall])
    precision_all = mean_defined([precision_normal, precision_anomaly])
    f1_all = mean_defined([f1_normal, f1_anomaly])
    recall_weighted = safe_div(
        (normal_recall or 0.0) * normal_total + (anomaly_recall or 0.0) * anomaly_total,
        total,
    )
    precision_weighted = safe_div(
        (precision_normal or 0.0) * normal_total + (precision_anomaly or 0.0) * anomaly_total,
        total,
    )
    f1_weighted = safe_div(
        (f1_normal or 0.0) * normal_total + (f1_anomaly or 0.0) * anomaly_total,
        total,
    )
    micro = safe_div(tp + tn, total)
    acc = safe_div(tp + tn, total)
    return {
        "acc": acc,
        "ACC": acc,
        "balanced_acc": balanced_acc,
        "anomaly_recall": anomaly_recall,
        "normal_recall": normal_recall,
        "recall_anomaly": anomaly_recall,
        "recall_normal": normal_recall,
        "recall_all": recall_all,
        "precision_anomaly": precision_anomaly,
        "precision_normal": precision_normal,
        "precision_all": precision_all,
        "f1_anomaly": f1_anomaly,
        "f1_normal": f1_normal,
        "f1_all": f1_all,
        "recall_weighted": recall_weighted,
        "precision_weighted": precision_weighted,
        "f1_weighted": f1_weighted,
        "recall_micro": micro,
        "precision_micro": micro,
        "f1_micro": micro,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": anomaly_total,
        "normal_total": normal_total,
        "evaluated_total": total,
    }


def metric_block(rows, score_key, pixel_labels, pixel_scores, threshold):
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    preds = [1 if score >= threshold else 0 for score in scores]
    best_acc, best_thr = best_accuracy(labels, scores)
    metrics = {
        "threshold": threshold,
        "best_acc": best_acc,
        "best_threshold": best_thr,
        "I-AUROC": binary_auc(labels, scores),
        "p-AUROC": binary_auc(pixel_labels, pixel_scores) if pixel_labels else None,
    }
    metrics.update(binary_classification_metrics(labels, preds))
    return metrics


def answer_only_reasoning(processor, model, tokenizer, image, record, category_bank, args):
    question = build_real_question(record, category_bank)
    prompt_inputs, full_sequence, generated_ids, generated_text, prompt_text = generate_reasoning(
        processor, model, tokenizer, image, record["image"], question, args
    )
    deco = {}
    if args.answer_deco:
        deco = answer_deco_trace(
            model,
            tokenizer,
            prompt_inputs,
            full_sequence,
            generated_ids,
            parse_int_range(args.answer_trend_layers),
            parse_int_range(args.deco_layers),
            args.deco_alpha,
        )
    return {
        "generated_text": generated_text,
        "prompt_text": prompt_text,
        "generated_tokens_total": len(generated_ids),
        "answer_deco": deco,
    }


def class_token_ids(tokenizer, words):
    ids = []
    for word in words:
        for text in (word, " " + word, word.lower(), " " + word.lower()):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def class_logit(logits, token_ids):
    selected = logits[:, torch.tensor(token_ids, dtype=torch.long, device=logits.device)]
    return torch.logsumexp(selected.float(), dim=-1)


def answer_class_probs(logits, yes_ids, no_ids):
    yes_logit = class_logit(logits, yes_ids)
    no_logit = class_logit(logits, no_ids)
    binary_logits = torch.stack([no_logit, yes_logit], dim=-1)
    probs = F.softmax(binary_logits, dim=-1)[0]
    return float(probs[1].item()), float(probs[0].item()), binary_logits


def find_generated_answer_token(generated_ids, yes_ids, no_ids):
    yes_set = set(int(x) for x in yes_ids)
    no_set = set(int(x) for x in no_ids)
    for local_idx in range(len(generated_ids) - 1, -1, -1):
        token_id = int(generated_ids[local_idx])
        if token_id in yes_set:
            return local_idx, "yes", token_id
        if token_id in no_set:
            return local_idx, "no", token_id
    return None, "", None


def answer_deco_trace(model, tokenizer, prompt_inputs, full_sequence, generated_ids, trend_layers, deco_layers, alpha):
    yes_ids = class_token_ids(tokenizer, ["Yes"])
    no_ids = class_token_ids(tokenizer, ["No"])
    if not yes_ids or not no_ids:
        return {"error": "failed to tokenize Yes/No candidates"}

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    answer_local_idx, generated_answer_token, generated_answer_token_id = find_generated_answer_token(
        generated_ids, yes_ids, no_ids
    )
    if answer_local_idx is None:
        return {
            "error": "no generated Yes/No token found",
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
        }
    predict_pos = prompt_len + int(answer_local_idx) - 1
    if predict_pos < 0:
        return {"error": "invalid answer prediction position"}

    forward_inputs = dict(prompt_inputs)
    forward_inputs["input_ids"] = full_sequence.to(model.device)
    forward_inputs["attention_mask"] = torch.ones_like(forward_inputs["input_ids"], device=model.device)
    with torch.inference_mode():
        outputs = model(**forward_inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        return {"error": "no hidden_states returned"}

    norm, lm_head = get_norm_and_lm_head(model)
    valid_trend_layers = [int(layer) for layer in trend_layers if 0 <= int(layer) < len(outputs.hidden_states)]
    valid_deco_layers = [int(layer) for layer in deco_layers if 0 <= int(layer) < len(outputs.hidden_states)]
    rows = []
    logits_by_layer = {}
    for layer in valid_trend_layers:
        h = outputs.hidden_states[layer][:, predict_pos, :]
        logits = lm_head(norm(h)).float()
        yes_prob, no_prob, binary_logits = answer_class_probs(logits, yes_ids, no_ids)
        rows.append(
            {
                "layer": layer,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "pred": "yes" if yes_prob >= no_prob else "no",
                "margin_yes_minus_no": yes_prob - no_prob,
            }
        )
        logits_by_layer[layer] = binary_logits

    final_layer = max(valid_trend_layers) if valid_trend_layers else len(outputs.hidden_states) - 1
    if final_layer not in logits_by_layer:
        h = outputs.hidden_states[final_layer][:, predict_pos, :]
        logits = lm_head(norm(h)).float()
        _, _, logits_by_layer[final_layer] = answer_class_probs(logits, yes_ids, no_ids)

    final_binary_logits = logits_by_layer[final_layer]
    final_probs = F.softmax(final_binary_logits, dim=-1)[0]
    final_no_prob = float(final_probs[0].item())
    final_yes_prob = float(final_probs[1].item())
    final_pred = "yes" if final_yes_prob >= final_no_prob else "no"

    best = None
    for layer in valid_deco_layers:
        if layer not in logits_by_layer:
            h = outputs.hidden_states[layer][:, predict_pos, :]
            logits = lm_head(norm(h)).float()
            _, _, logits_by_layer[layer] = answer_class_probs(logits, yes_ids, no_ids)
        layer_probs = F.softmax(logits_by_layer[layer], dim=-1)[0]
        layer_no = float(layer_probs[0].item())
        layer_yes = float(layer_probs[1].item())
        layer_max = max(layer_no, layer_yes)
        if best is None or layer_max > best["selected_prob"]:
            best = {
                "selected_layer": int(layer),
                "selected_prob": layer_max,
                "selected_pred": "yes" if layer_yes >= layer_no else "no",
                "selected_yes_prob": layer_yes,
                "selected_no_prob": layer_no,
            }

    if best is None:
        calibrated_probs = final_probs
        deco_pred = final_pred
        selected_layer_logits = torch.zeros_like(final_binary_logits)
    else:
        selected_layer_logits = logits_by_layer[best["selected_layer"]]
        calibrated_logits = final_binary_logits + float(alpha) * float(best["selected_prob"]) * selected_layer_logits
        calibrated_probs = F.softmax(calibrated_logits, dim=-1)[0]
        deco_pred = "yes" if float(calibrated_probs[1].item()) >= float(calibrated_probs[0].item()) else "no"

    return {
        "error": "",
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "generated_answer_token": generated_answer_token,
        "generated_answer_token_id": generated_answer_token_id,
        "answer_local_idx": int(answer_local_idx),
        "predict_pos": int(predict_pos),
        "final_layer": int(final_layer),
        "final_yes_prob": final_yes_prob,
        "final_no_prob": final_no_prob,
        "final_pred": final_pred,
        "deco_alpha": float(alpha),
        "deco_layers": valid_deco_layers,
        "selected_layer": best["selected_layer"] if best else "",
        "selected_prob": best["selected_prob"] if best else "",
        "selected_pred": best["selected_pred"] if best else "",
        "selected_yes_prob": best["selected_yes_prob"] if best else "",
        "selected_no_prob": best["selected_no_prob"] if best else "",
        "deco_yes_prob": float(calibrated_probs[1].item()),
        "deco_no_prob": float(calibrated_probs[0].item()),
        "deco_pred": deco_pred,
        "trend_rows": rows,
    }


def first_token_id(tokenizer, phrase):
    ids = tokenizer.encode(" " + str(phrase).strip(), add_special_tokens=False)
    return int(ids[0]) if ids else None


def collect_trend_candidates(tokenizer, selected_tokens, defect_bank, max_candidates):
    candidates = []
    seen_ids = set()
    for token in sorted(selected_tokens, key=lambda x: float(x.get("weight", 0.0)), reverse=True):
        token_id = int(token["token_id"])
        if token_id in seen_ids:
            continue
        candidates.append(
            {
                "token_id": token_id,
                "token": token.get("word") or token_to_word(tokenizer, token_id),
                "source": "selected_real_token",
                "ST": token.get("ST"),
                "SI": token.get("SI"),
                "weight": token.get("weight"),
            }
        )
        seen_ids.add(token_id)
        if len(candidates) >= max_candidates:
            return candidates
    for phrase in defect_bank:
        token_id = first_token_id(tokenizer, phrase)
        if token_id is None or token_id in seen_ids:
            continue
        candidates.append(
            {
                "token_id": token_id,
                "token": token_to_word(tokenizer, token_id),
                "source": "defect_bank_fallback",
                "phrase": phrase,
                "ST": None,
                "SI": None,
                "weight": None,
            }
        )
        seen_ids.add(token_id)
        if len(candidates) >= max_candidates:
            return candidates
    return candidates


def candidate_layer_trends(processor, model, tokenizer, image, record, category_bank, candidates, layers, args):
    if not candidates:
        return [], {}
    question = build_real_question(record, category_bank)
    inputs, prompt_text = processor_inputs(
        processor,
        model,
        image,
        record["image"],
        question,
        args.min_pixels,
        args.max_pixels,
    )
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    norm, lm_head = get_norm_and_lm_head(model)
    token_ids = torch.tensor([int(c["token_id"]) for c in candidates], dtype=torch.long, device=model.device)
    rows = []
    by_token = {c["token"]: [] for c in candidates}
    for layer in layers:
        if layer < 0 or layer >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[layer][:, -1, :]
        logits = lm_head(norm(h)).float()
        probs = F.softmax(logits, dim=-1)[0, token_ids].detach().cpu().tolist()
        for cand, prob in zip(candidates, probs):
            row = {
                "layer": int(layer),
                "token_id": int(cand["token_id"]),
                "token": cand["token"],
                "prob": float(prob),
                "source": cand.get("source", ""),
                "ST": cand.get("ST"),
                "SI": cand.get("SI"),
                "weight": cand.get("weight"),
            }
            rows.append(row)
            by_token[cand["token"]].append({"layer": int(layer), "prob": float(prob)})
    return rows, {"prompt_text": prompt_text, "by_token": by_token}


def save_trend_plot(rows, output_png):
    if not rows:
        return
    grouped = {}
    for row in rows:
        grouped.setdefault(row["token"], []).append(row)
    plt.figure(figsize=(10, 5.6))
    for token, token_rows in grouped.items():
        token_rows = sorted(token_rows, key=lambda x: int(x["layer"]))
        plt.plot(
            [r["layer"] for r in token_rows],
            [r["prob"] for r in token_rows],
            marker="o",
            linewidth=1.6,
            markersize=3,
            label=token,
        )
    plt.xlabel("Layer")
    plt.ylabel("next-token probability")
    plt.title("Candidate token probability trend across layers")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def save_answer_deco_plot(rows, selected_layer, output_png):
    if not rows:
        return
    rows = sorted(rows, key=lambda x: int(x["layer"]))
    plt.figure(figsize=(9.5, 5.2))
    plt.plot([r["layer"] for r in rows], [r["yes_prob"] for r in rows], marker="o", linewidth=1.8, label="Yes -> anomaly")
    plt.plot([r["layer"] for r in rows], [r["no_prob"] for r in rows], marker="o", linewidth=1.8, label="No -> normal")
    plt.plot(
        [r["layer"] for r in rows],
        [r["margin_yes_minus_no"] for r in rows],
        marker=".",
        linewidth=1.2,
        linestyle="--",
        label="Yes - No",
    )
    if selected_layer != "":
        plt.axvline(int(selected_layer), color="red", linestyle="--", linewidth=1.5, label=f"DeCo layer {selected_layer}")
    plt.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel("Probability / margin")
    plt.title("Answer Yes/No probability trend across layers")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def parse_generated_answer(text):
    matches = re.findall(r"answer\s*:\s*(yes|no)", str(text), flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return ""


def prediction_name(pred):
    return "anomaly" if int(pred) == 1 else "normal"


def answer_to_pred(answer):
    answer = str(answer or "").strip().lower()
    if answer == "yes":
        return 1
    if answer == "no":
        return 0
    return -1


def generated_answer_metric(rows, pred_key="answer_pred", score_key=None):
    valid = [row for row in rows if int(row.get(pred_key, -1)) in {0, 1}]
    if not valid:
        return {
            "acc": None,
            "I-AUROC": None,
            "coverage": 0.0,
            "valid": 0,
            "total": len(rows),
            "rule": "Answer: Yes => anomaly, Answer: No => normal",
        }
    labels = [int(row["label"]) for row in valid]
    preds = [int(row[pred_key]) for row in valid]
    scored = []
    if score_key:
        for row in valid:
            value = row.get(score_key, "")
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                scored.append((int(row["label"]), value))
    if scored:
        auc_labels = [x[0] for x in scored]
        auc_scores = [x[1] for x in scored]
        auc_score_source = score_key
    else:
        auc_labels = labels
        auc_scores = [float(x) for x in preds]
        auc_score_source = f"{pred_key} (binary fallback)"
    auroc = binary_auc(auc_labels, auc_scores)
    metrics = {
        "I-AUROC": auroc,
        "AUROC": auroc,
        "auroc_score_source": auc_score_source,
        "auroc_valid": len(auc_labels),
        "auroc_coverage": len(auc_labels) / len(valid) if valid else 0.0,
        "coverage": len(valid) / len(rows) if rows else 0.0,
        "valid": len(valid),
        "total": len(rows),
        "rule": f"{pred_key}: 1/Yes => anomaly, 0/No => normal",
    }
    metrics.update(binary_classification_metrics(labels, preds))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visa-split-csv", default="", help="Optional official VisA split CSV, e.g. split_csv/1cls.csv.")
    parser.add_argument("--visa-split-set", default="test", help="CSV split value(s) to evaluate, comma-separated. Default: test.")
    parser.add_argument("--categories", default="")
    parser.add_argument("--sample-normal", type=int, default=None, help="Randomly sample this many normal images. Default: use all.")
    parser.add_argument("--sample-anomaly", type=int, default=None, help="Randomly sample this many anomaly images. Default: use all.")
    parser.add_argument("--run-all", action="store_true", help="Explicitly evaluate all available images. Overrides sample-normal/sample-anomaly.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--real-top-percent", type=float, default=5.0)
    parser.add_argument("--real-threshold", type=float, default=0.5)
    parser.add_argument("--st-threshold", type=float, default=0.5)
    parser.add_argument("--si-threshold", type=float, default=0.12)
    parser.add_argument("--si-top-frac", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--pixel-auroc-size", type=int, default=256)
    parser.add_argument("--trend-layers", default="1:37")
    parser.add_argument("--max-trend-candidates", type=int, default=12)
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--save-per-image", action="store_true", help="Save per-image summary/trend files. Disabled by default for full runs.")
    parser.add_argument("--save-trends", action="store_true", help="Compute and save 1:37 candidate-token trends. Disabled by default because it is slow.")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--attention-max-generated-tokens", type=int, default=256, help="Cap generated tokens used for output_attentions teacher forcing.")
    parser.add_argument("--empty-cache-interval", type=int, default=100, help="Call torch.cuda.empty_cache every N images. Use 0 to disable.")
    parser.add_argument("--answer-only", action="store_true", help="Only generate and score final Answer: Yes/No; skip ReAL-lite attention maps.")
    parser.add_argument("--answer-deco", action="store_true", help="Trace Answer Yes/No probabilities across layers and apply DeCo to the final answer token.")
    parser.add_argument("--answer-trend-layers", default="0:33")
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--save-answer-deco-plots", action="store_true", help="Save per-image Answer Yes/No layer probability plots.")
    parser.add_argument("--answer-deco-plot-normal-per-category", type=int, default=None, help="When saving answer-DeCo plots, save at most this many normal samples per category.")
    parser.add_argument("--answer-deco-plot-anomaly-per-category", type=int, default=None, help="When saving answer-DeCo plots, save at most this many anomaly samples per category.")
    args = parser.parse_args()

    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration is not available from transformers.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_normal = None if args.run_all else args.sample_normal
    sample_anomaly = None if args.run_all else args.sample_anomaly
    samples, category_banks = collect_visa_samples(
        args.visa_root,
        args.categories,
        None,
        sample_normal,
        sample_anomaly,
        args.seed,
        args.visa_split_csv or None,
        args.visa_split_set,
    )
    if not samples:
        raise RuntimeError("No VisA samples found.")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa" if args.answer_only else "eager",
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    failed_rows = []
    pixel_labels = []
    pixel_scores = []
    eval_size = None if int(args.pixel_auroc_size) <= 0 else (int(args.pixel_auroc_size), int(args.pixel_auroc_size))
    trend_layers = parse_int_range(args.trend_layers)
    answer_deco_plot_counts = {}
    print(
        f"[INFO] Loaded {len(samples)} samples. answer_only={args.answer_only}, "
        f"save_per_image={args.save_per_image}, save_trends={args.save_trends}, "
        f"save_visualizations={args.save_visualizations}"
    )

    for idx, record in enumerate(samples, start=1):
        try:
            image = Image.open(record["image"]).convert("RGB")
            sample_dir = None
            should_save_answer_deco_plot = False
            if args.save_answer_deco_plots:
                plot_key = (record["category"], record["label_name"])
                plot_limit = (
                    args.answer_deco_plot_anomaly_per_category
                    if int(record["label"]) == 1
                    else args.answer_deco_plot_normal_per_category
                )
                current_count = int(answer_deco_plot_counts.get(plot_key, 0))
                should_save_answer_deco_plot = plot_limit is None or current_count < int(plot_limit)
                if should_save_answer_deco_plot:
                    answer_deco_plot_counts[plot_key] = current_count + 1
            if args.save_per_image or args.save_visualizations or args.save_trends or should_save_answer_deco_plot:
                sample_dir = out_dir / f"{idx:06d}-{record['category']}-{record['label_name']}-{record['image_id']}"
                sample_dir.mkdir(parents=True, exist_ok=True)
            bank = category_banks[record["category"]]

            if args.answer_only:
                result = answer_only_reasoning(processor, model, tokenizer, image, record, bank, args)
                real_score = None
                real_pred = -1
            else:
                result = real_lite_attention_map(processor, model, tokenizer, image, record, bank, args)
                real_score = float(result["score"])
                real_pred = 1 if real_score >= float(args.real_threshold) else 0
            generated_answer = parse_generated_answer(result["generated_text"])
            answer_pred = answer_to_pred(generated_answer)
            answer_deco = result.get("answer_deco", {}) or {}
            deco_answer = answer_deco.get("deco_pred", "")
            deco_answer_pred = answer_to_pred(deco_answer)
            predicted_anomaly_type = "none"
            selected_tokens = result.get("tokens", [])
            if real_pred == 1 and selected_tokens:
                predicted_anomaly_type = max(result["tokens"], key=lambda x: float(x.get("weight", 0.0))).get("word", "unknown")
            elif real_pred == 1:
                predicted_anomaly_type = "unspecified_anomaly"
            trend_rows, trend_summary = [], {}
            if args.save_trends and not args.answer_only:
                trend_candidates = collect_trend_candidates(
                    tokenizer,
                    result["tokens"],
                    bank["defect_bank"],
                    int(args.max_trend_candidates),
                )
                trend_rows, trend_summary = candidate_layer_trends(
                    processor,
                    model,
                    tokenizer,
                    image,
                    record,
                    bank,
                    trend_candidates,
                    trend_layers,
                    args,
                )
            row = {
                **record,
                "real_score": real_score if real_score is not None else "",
                "real_pred": real_pred,
                "real_pred_name": prediction_name(real_pred) if real_pred in {0, 1} else "not_run",
                "generated_answer": generated_answer,
                "answer_pred": answer_pred,
                "answer_pred_name": prediction_name(answer_pred) if answer_pred in {0, 1} else "unknown",
                "answer_correct": int(answer_pred == int(record["label"])) if answer_pred in {0, 1} else "",
                "deco_answer": deco_answer,
                "deco_answer_pred": deco_answer_pred,
                "deco_answer_pred_name": prediction_name(deco_answer_pred) if deco_answer_pred in {0, 1} else "unknown",
                "deco_answer_correct": int(deco_answer_pred == int(record["label"])) if deco_answer_pred in {0, 1} else "",
                "deco_selected_layer": answer_deco.get("selected_layer", ""),
                "deco_selected_prob": answer_deco.get("selected_prob", ""),
                "deco_selected_pred": answer_deco.get("selected_pred", ""),
                "deco_final_yes_prob": answer_deco.get("final_yes_prob", ""),
                "deco_final_no_prob": answer_deco.get("final_no_prob", ""),
                "deco_yes_prob": answer_deco.get("deco_yes_prob", ""),
                "deco_no_prob": answer_deco.get("deco_no_prob", ""),
                "deco_error": answer_deco.get("error", ""),
                "predicted_anomaly_type": predicted_anomaly_type,
                "correct": int(real_pred == int(record["label"])) if real_pred in {0, 1} else "",
                "real_raw_score": real_score if real_score is not None else "",
                "num_real_tokens": len(selected_tokens),
                "generated_tokens_total": result.get("generated_tokens_total", ""),
                "attention_tokens_used": result.get("attention_tokens_used", ""),
                "attention_truncated": result.get("attention_truncated", ""),
                "real_error": result.get("error", ""),
            }
            rows.append(row)

            if not args.answer_only:
                mask_labels = load_mask(record["mask"], eval_size or image.size)
                if mask_labels is not None:
                    pixel_labels.extend(mask_labels)
                    pixel_scores.extend(resize_score_values_from_tensor(result["pixel_map"], eval_size))

            if args.save_per_image and sample_dir is not None:
                write_json(
                    sample_dir / "summary.json",
                    {
                        **row,
                        "ground_truth": {
                            "label": int(record["label"]),
                            "label_name": record["label_name"],
                            "specific_anomaly_type": record.get("specific_anomaly_type", "normal" if int(record["label"]) == 0 else "anomaly_unspecified"),
                            "annotation_source": "official split CSV binary label; no per-image anomaly_prompt.json",
                            "mask": record.get("mask", ""),
                        },
                        "prediction": {
                            "pred_label": int(real_pred) if real_pred in {0, 1} else None,
                            "pred_label_name": prediction_name(real_pred) if real_pred in {0, 1} else "not_run",
                            "predicted_anomaly_type": predicted_anomaly_type,
                            "generated_answer_yes_no": generated_answer,
                            "answer_only_pred_label": answer_pred,
                            "answer_only_pred_label_name": prediction_name(answer_pred) if answer_pred in {0, 1} else "unknown",
                            "answer_only_decision_rule": "Answer: Yes => anomaly, Answer: No => normal",
                            "answer_only_is_correct": bool(answer_pred == int(record["label"])) if answer_pred in {0, 1} else None,
                            "deco_answer": deco_answer,
                            "deco_answer_pred_label": deco_answer_pred if deco_answer_pred in {0, 1} else None,
                            "deco_answer_pred_label_name": prediction_name(deco_answer_pred) if deco_answer_pred in {0, 1} else "unknown",
                            "deco_answer_is_correct": bool(deco_answer_pred == int(record["label"])) if deco_answer_pred in {0, 1} else None,
                            "decision_boundary": f"real_score >= {args.real_threshold} => anomaly, otherwise normal" if not args.answer_only else "not_run_in_answer_only_mode",
                            "is_correct": bool(real_pred == int(record["label"])) if real_pred in {0, 1} else None,
                        },
                        "scores": {
                            "real_score": real_score,
                            "real_raw_score": real_score,
                            "threshold": float(args.real_threshold),
                            "real_top_percent": float(args.real_top_percent),
                            "num_selected_real_tokens": len(selected_tokens),
                            "generated_tokens_total": result.get("generated_tokens_total", ""),
                            "attention_tokens_used": result.get("attention_tokens_used", ""),
                            "attention_truncated": result.get("attention_truncated", ""),
                            "ST_threshold": float(args.st_threshold),
                            "SI_threshold": float(args.si_threshold),
                        },
                        "answer_deco": answer_deco,
                        "judgement_basis": {
                            "summary": (
                                "Answer-only mode parses the final generated Answer: Yes/No."
                                if args.answer_only
                                else "Prediction is based on ReAL-lite anomaly_map top-percent mean. "
                                "The anomaly_map is a weighted aggregation of generated reasoning-token attentions "
                                "whose ST and SI pass thresholds."
                            ),
                            "selected_real_tokens": selected_tokens,
                            "candidate_token_trend_file": str(sample_dir / "candidate_token_layer_trends.csv") if args.save_trends else "",
                            "candidate_token_trend_plot": str(sample_dir / "candidate_token_layer_trends.png") if args.save_trends else "",
                        },
                        "generated_reasoning": result["generated_text"],
                        "prompt_text": result["prompt_text"],
                        "prompt_defect_description": category_defect_description(bank),
                        "note": "The prompt uses a built-in category-level VisA taxonomy shared by normal and anomaly samples. No anomaly_prompt.json or per-image defect description is read.",
                        "selected_real_tokens": selected_tokens,
                        "defect_bank": bank["defect_bank"],
                        "candidate_token_layer_trends": trend_summary,
                    },
                )
            if args.save_trends and sample_dir is not None:
                write_csv(sample_dir / "candidate_token_layer_trends.csv", trend_rows)
                write_json(sample_dir / "candidate_token_layer_trends.json", trend_summary)
            if should_save_answer_deco_plot and sample_dir is not None and answer_deco.get("trend_rows"):
                write_csv(sample_dir / "answer_deco_layer_trends.csv", answer_deco["trend_rows"])
                write_json(sample_dir / "answer_deco_trace.json", answer_deco)
                save_answer_deco_plot(
                    answer_deco["trend_rows"],
                    answer_deco.get("selected_layer", ""),
                    sample_dir / "answer_deco_layer_trends.png",
                )
            if args.save_visualizations and sample_dir is not None and not args.answer_only:
                save_heatmap_overlay(image, result["pixel_map"], sample_dir / "real_lite_anomaly_overlay.png", "ReAL-lite anomaly map")
                save_map_only(result["map"], sample_dir / "real_lite_visual_token_map.png", "ReAL-lite visual-token map")
                if args.save_trends:
                    save_trend_plot(trend_rows, sample_dir / "candidate_token_layer_trends.png")
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failed_rows.append({**record, "index": idx, "error": str(exc).splitlines()[0]})
            print(f"[WARN] CUDA OOM skipped {idx}/{len(samples)} {record['category']}/{record['label_name']}/{record['image_id']}: {str(exc).splitlines()[0]}")
            continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            failed_rows.append({**record, "index": idx, "error": str(exc).splitlines()[0]})
            print(f"[WARN] Runtime OOM skipped {idx}/{len(samples)} {record['category']}/{record['label_name']}/{record['image_id']}: {str(exc).splitlines()[0]}")
            continue
        finally:
            if int(args.empty_cache_interval) > 0 and idx % int(args.empty_cache_interval) == 0:
                torch.cuda.empty_cache()

        if idx == 1 or idx % max(1, int(args.log_interval)) == 0 or idx == len(samples):
            answer_metrics = generated_answer_metric(
                rows,
                "answer_pred",
                "deco_final_yes_prob" if args.answer_deco else None,
            )
            deco_metrics = generated_answer_metric(rows, "deco_answer_pred", "deco_yes_prob") if args.answer_deco else None
            if args.answer_only:
                print(
                    f"[{idx}/{len(samples)}] latest={record['category']}/{record['label_name']}/{record['image_id']} "
                    f"answer={generated_answer or 'unknown'} answer_acc={answer_metrics['acc']} "
                    f"balanced_acc={answer_metrics.get('balanced_acc')} "
                    f"anomaly_recall={answer_metrics.get('anomaly_recall')} "
                    f"normal_recall={answer_metrics.get('normal_recall')} "
                    f"coverage={answer_metrics['coverage']:.3f}"
                    + (
                        f" deco_acc={deco_metrics['acc']} deco_balanced_acc={deco_metrics.get('balanced_acc')} "
                        f"deco_anomaly_recall={deco_metrics.get('anomaly_recall')}"
                        if deco_metrics
                        else ""
                    )
                )
            else:
                real_metrics = metric_block(rows, "real_score", [], [], args.real_threshold)
                print(
                    f"[{idx}/{len(samples)}] latest={record['category']}/{record['label_name']}/{record['image_id']} "
                    f"real={real_score:.4f} pred={real_pred} answer={generated_answer or 'unknown'} "
                    f"answer_acc={answer_metrics['acc']} coverage={answer_metrics['coverage']:.3f} "
                    f"real_acc={real_metrics['acc']} real_balanced_acc={real_metrics.get('balanced_acc')}"
                )

    metrics = {
        "generated_answer_only": generated_answer_metric(
            rows,
            "answer_pred",
            "deco_final_yes_prob" if args.answer_deco else None,
        ),
        "num_images": len(rows),
        "num_failed": len(failed_rows),
        "num_pixel_labels": len(pixel_labels),
    }
    if args.answer_deco:
        metrics["deco_answer_only"] = generated_answer_metric(rows, "deco_answer_pred", "deco_yes_prob")
    if not args.answer_only:
        metrics["ReAL_lite_only"] = metric_block(rows, "real_score", pixel_labels, pixel_scores, args.real_threshold)

    per_category = {}
    for category in sorted(set(row["category"] for row in rows)):
        cat_rows = [row for row in rows if row["category"] == category]
        per_category[category] = {
            "generated_answer_only": generated_answer_metric(
                cat_rows,
                "answer_pred",
                "deco_final_yes_prob" if args.answer_deco else None,
            ),
        }
        if args.answer_deco:
            per_category[category]["deco_answer_only"] = generated_answer_metric(
                cat_rows,
                "deco_answer_pred",
                "deco_yes_prob",
            )
        if not args.answer_only:
            per_category[category]["ReAL_lite_only"] = metric_block(cat_rows, "real_score", [], [], args.real_threshold)
    metrics["per_category"] = per_category
    metrics["metric_definitions"] = {
        "normal_class": 0,
        "anomaly_class": 1,
        "ACC": "(TP + TN) / all evaluated samples",
        "AUROC": "ROC AUC from continuous Yes/anomaly probability; original uses final-layer Yes probability and DeCo uses calibrated Yes probability",
        "recall_normal": "TN / (TN + FP)",
        "recall_anomaly": "TP / (TP + FN)",
        "precision_normal": "TN / (TN + FN), treating normal as the positive class",
        "precision_anomaly": "TP / (TP + FP)",
        "f1_normal": "harmonic mean of precision_normal and recall_normal",
        "f1_anomaly": "harmonic mean of precision_anomaly and recall_anomaly",
        "*_all": "macro average of the normal and anomaly class metrics",
        "*_weighted": "class-support-weighted average",
        "*_micro": "global single-label micro average; equals ACC in binary classification",
    }

    summary_fields = [
        "ACC",
        "AUROC",
        "recall_normal",
        "recall_anomaly",
        "recall_all",
        "precision_normal",
        "precision_anomaly",
        "precision_all",
        "f1_normal",
        "f1_anomaly",
        "f1_all",
        "recall_weighted",
        "precision_weighted",
        "f1_weighted",
        "recall_micro",
        "precision_micro",
        "f1_micro",
        "balanced_acc",
        "coverage",
        "tp",
        "tn",
        "fp",
        "fn",
        "normal_total",
        "anomaly_total",
        "evaluated_total",
        "auroc_score_source",
        "auroc_coverage",
    ]
    summary_rows = []

    def append_summary(scope, category, method, block):
        summary_rows.append(
            {
                "scope": scope,
                "category": category,
                "method": method,
                **{field: block.get(field) for field in summary_fields},
            }
        )

    append_summary("overall", "all", "original", metrics["generated_answer_only"])
    if args.answer_deco:
        append_summary("overall", "all", "deco", metrics["deco_answer_only"])
    for category, category_metrics in per_category.items():
        append_summary("category", category, "original", category_metrics["generated_answer_only"])
        if args.answer_deco:
            append_summary("category", category, "deco", category_metrics["deco_answer_only"])

    write_csv(out_dir / "image_scores.csv", rows)
    write_csv(out_dir / "metrics_summary.csv", summary_rows)
    if failed_rows:
        write_csv(out_dir / "failed_samples.csv", failed_rows)
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "0-manifest.json",
        {
            "model_path": args.model_path,
            "visa_root": args.visa_root,
            "visa_split_csv": args.visa_split_csv,
            "visa_split_set": args.visa_split_set,
            "defect_bank_source": "builtin_visa_category_taxonomy",
            "uses_anomaly_prompt_json": False,
            "num_samples": len(samples),
            "logic": (
                "Answer-only: full-image reasoning, parse final Answer: Yes/No, optionally trace/apply DeCo on final answer token."
                if args.answer_only
                else "Pure ReAL-lite: full-image reasoning, generated-token to image-token attention, ST/SI token filtering, weighted attention-map aggregation."
            ),
            "args": vars(args),
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
