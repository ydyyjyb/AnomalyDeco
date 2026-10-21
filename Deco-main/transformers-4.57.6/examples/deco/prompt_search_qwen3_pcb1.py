import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from probe_qwen3_deco_layer_curve import (
    Qwen3VLForConditionalGeneration,
    get_norm_and_lm_head,
    parse_int_range,
    prepare_inputs,
    token_label,
)
from probe_qwen3_deco_layer_curve_norm import build_final_candidates, normalized_probs_over_ids


DEFAULT_PROMPTS = [
    {
        "question": "Classify the PCB image as normal or defective based only on visible evidence.",
        "prefix": "The PCB is",
    },
    {
        "question": "Decide whether this PCB board is normal or defective.",
        "prefix": "The condition label is",
    },
    {
        "question": "Inspect the PCB image for visible surface defects. Answer with normal or defective.",
        "prefix": "Answer:",
    },
    {
        "question": "Is this PCB image normal or abnormal? Use the visual evidence only.",
        "prefix": "The PCB appears",
    },
    {
        "question": "Judge whether this PCB contains visible anomaly, scratch, stain, hole, crack, or damage.",
        "prefix": "The visual diagnosis is",
    },
    {
        "question": "Please inspect the PCB surface and describe visible defects if any.",
        "prefix": "The PCB board shows a",
    },
    {
        "question": "Examine this PCB image for visual irregularities.",
        "prefix": "The visible surface condition is",
    },
    {
        "question": "Please judge whether there are abnormal defects in this PCB image.",
        "prefix": "The PCB board appears",
    },
    {
        "question": "Inspect the PCB carefully and identify any abnormal visual pattern.",
        "prefix": "The most visible condition is",
    },
    {
        "question": "Analyze this PCB image and report whether the board surface is normal or defective.",
        "prefix": "This PCB surface is",
    },
    {
        "question": "Look for scratches, stains, holes, cracks, or other visible PCB defects.",
        "prefix": "The visible defect is",
    },
    {
        "question": "Check the PCB for manufacturing anomalies and describe the visual evidence.",
        "prefix": "The abnormal region looks like a",
    },
    {
        "question": "Please describe the PCB board condition based only on visible image evidence.",
        "prefix": "The board condition is",
    },
]


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


def sample_images(root, n_anomaly, n_normal, seed):
    rng = random.Random(seed)
    root = Path(root)
    anomaly_dir = root / "pcb1" / "Data" / "Images" / "Anomaly"
    normal_dir = root / "pcb1" / "Data" / "Images" / "Normal"
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    anomaly = [p for p in anomaly_dir.rglob("*") if p.suffix.lower() in suffixes]
    normal = [p for p in normal_dir.rglob("*") if p.suffix.lower() in suffixes]
    if len(anomaly) < n_anomaly:
        raise RuntimeError(f"Not enough anomaly images: need {n_anomaly}, found {len(anomaly)}")
    if len(normal) < n_normal:
        raise RuntimeError(f"Not enough normal images: need {n_normal}, found {len(normal)}")
    chosen = [(p, "anomaly") for p in rng.sample(anomaly, n_anomaly)]
    chosen += [(p, "normal") for p in rng.sample(normal, n_normal)]
    rng.shuffle(chosen)
    return chosen


def load_prompt_candidates(path):
    if not path:
        return DEFAULT_PROMPTS
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("prompts", [])
        return [{"question": x["question"], "prefix": x["prefix"]} for x in data]
    prompts = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append({"question": obj["question"], "prefix": obj["prefix"]})
    return prompts


def logits_by_layer(model, hidden_states, layers):
    norm, lm_head = get_norm_and_lm_head(model)
    out = {}
    with torch.no_grad():
        for layer in layers:
            if 0 <= layer < len(hidden_states):
                h = hidden_states[layer][:, -1, :]
                out[int(layer)] = lm_head(norm(h)).float()
    return out


def select_norm_deco(
    logits_by_layer_map,
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
        probs = F.softmax(final_logits.float(), dim=-1)
        raw_final_probs = [float(probs[0, tid].item()) for tid in candidate_ids]
        norm_final_probs = normalized_probs_over_ids(final_logits, candidate_ids)
    else:
        candidate_ids, raw_final_probs, norm_final_probs = build_final_candidates(final_logits, top_k, top_p)

    if not candidate_ids:
        raise RuntimeError("No candidate token ids available.")

    best = None
    for layer in early_exit_layers:
        if layer not in logits_by_layer_map:
            continue
        norm_vals = normalized_probs_over_ids(logits_by_layer_map[layer], candidate_ids)
        local_idx = int(torch.argmax(norm_vals).item())
        score = float(norm_vals[local_idx].item())
        if best is None or score > best["score"]:
            best = {"layer": int(layer), "local_idx": local_idx, "score": score}
    if best is None:
        raise RuntimeError("No valid early-exit layer found.")

    selected_layer = best["layer"]
    selected_logits = logits_by_layer_map[selected_layer].float()
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=selected_logits.device)
    mask = torch.ones_like(selected_logits, dtype=torch.bool)
    mask[:, candidate_tensor] = False
    calibrated_logits = final_logits.float() + float(alpha) * float(best["score"]) * selected_logits
    calibrated_logits = calibrated_logits.masked_fill(mask, -float("inf"))
    calibrated_candidate_norm = normalized_probs_over_ids(calibrated_logits, candidate_ids)
    calibrated_local = int(torch.argmax(calibrated_candidate_norm).item())

    return {
        "candidate_ids": candidate_ids,
        "candidate_raw_final_probs": raw_final_probs,
        "candidate_norm_final_probs": norm_final_probs,
        "selected_layer": selected_layer,
        "selected_token_id": int(candidate_ids[best["local_idx"]]),
        "selected_norm_prob": float(best["score"]),
        "calibrated_token_id": int(candidate_ids[calibrated_local]),
        "calibrated_norm_probs": calibrated_candidate_norm,
        "calibrated_norm_prob": float(calibrated_candidate_norm[calibrated_local].item()),
    }


def class_scores(candidate_ids, norm_probs, anomaly_ids, normal_ids, anomaly_weight=1.0):
    id_to_local = {int(tid): i for i, tid in enumerate(candidate_ids)}
    anomaly_score = sum(float(norm_probs[id_to_local[tid]].item()) for tid in anomaly_ids if tid in id_to_local)
    normal_score = sum(float(norm_probs[id_to_local[tid]].item()) for tid in normal_ids if tid in id_to_local)
    weighted_anomaly_score = float(anomaly_weight) * anomaly_score
    if weighted_anomaly_score == 0.0 and normal_score == 0.0:
        pred = "unknown"
    elif weighted_anomaly_score >= normal_score:
        pred = "anomaly"
    else:
        pred = "normal"
    return pred, anomaly_score, normal_score


def confusion_counts(case_rows):
    counts = {
        "anomaly_correct": 0,
        "anomaly_total": 0,
        "normal_correct": 0,
        "normal_total": 0,
        "pred_anomaly": 0,
        "pred_normal": 0,
        "pred_unknown": 0,
    }
    for row in case_rows:
        label = row["label"]
        pred = row["pred"]
        if label == "anomaly":
            counts["anomaly_total"] += 1
            if pred == label:
                counts["anomaly_correct"] += 1
        elif label == "normal":
            counts["normal_total"] += 1
            if pred == label:
                counts["normal_correct"] += 1
        if pred == "anomaly":
            counts["pred_anomaly"] += 1
        elif pred == "normal":
            counts["pred_normal"] += 1
        else:
            counts["pred_unknown"] += 1
    counts["anomaly_accuracy"] = (
        counts["anomaly_correct"] / counts["anomaly_total"] if counts["anomaly_total"] else 0.0
    )
    counts["normal_accuracy"] = (
        counts["normal_correct"] / counts["normal_total"] if counts["normal_total"] else 0.0
    )
    return counts


def rescore_case_rows(case_rows, anomaly_weight):
    rescored = []
    for row in case_rows:
        new_row = dict(row)
        weighted_anomaly_score = float(anomaly_weight) * float(row["anomaly_score"])
        normal_score = float(row["normal_score"])
        if weighted_anomaly_score == 0.0 and normal_score == 0.0:
            pred = "unknown"
        elif weighted_anomaly_score >= normal_score:
            pred = "anomaly"
        else:
            pred = "normal"
        new_row["pred"] = pred
        new_row["correct"] = bool(pred == row["label"])
        new_row["anomaly_weight"] = float(anomaly_weight)
        new_row["weighted_anomaly_score"] = weighted_anomaly_score
        rescored.append(new_row)
    return rescored


def summarize_case_rows(case_rows):
    correct = sum(1 for r in case_rows if r["correct"])
    known = sum(1 for r in case_rows if r["pred"] != "unknown")
    confusion = confusion_counts(case_rows)
    selected_layers = [r["selected_layer"] for r in case_rows]
    return {
        "accuracy": correct / len(case_rows),
        "known_rate": known / len(case_rows),
        "correct": correct,
        "total": len(case_rows),
        **confusion,
        "selected_layers": selected_layers,
        "mean_selected_layer": sum(selected_layers) / len(selected_layers),
    }


def evaluate_case(processor, tokenizer, model, image_path, label, question, prefix, args, anomaly_ids, normal_ids):
    inputs, _ = prepare_inputs(
        processor,
        model,
        str(image_path),
        question,
        prefix,
        args.min_pixels,
        args.max_pixels,
    )
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
    layer_map = logits_by_layer(model, outputs.hidden_states, args._all_layers)
    final_logits = outputs.logits[:, -1, :].float()
    trace = select_norm_deco(
        layer_map,
        final_logits,
        args._early_exit_layers,
        args.top_k,
        args.top_p,
        args.alpha,
        args.candidate_mode,
        args._class_candidate_ids,
    )
    pred, anomaly_score, normal_score = class_scores(
        trace["candidate_ids"],
        trace["calibrated_norm_probs"],
        anomaly_ids,
        normal_ids,
        anomaly_weight=1.0,
    )
    candidate_text = [tokenizer.decode([tid], skip_special_tokens=False) for tid in trace["candidate_ids"]]
    return {
        "image": str(image_path),
        "label": label,
        "pred": pred,
        "correct": bool(pred == label),
        "anomaly_score": anomaly_score,
        "normal_score": normal_score,
        "anomaly_weight": 1.0,
        "weighted_anomaly_score": anomaly_score,
        "selected_layer": trace["selected_layer"],
        "selected_token": tokenizer.decode([trace["selected_token_id"]], skip_special_tokens=False),
        "calibrated_token": tokenizer.decode([trace["calibrated_token_id"]], skip_special_tokens=False),
        "candidate_tokens": candidate_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file", default=None, help="Optional JSON or JSONL with question/prefix records.")
    parser.add_argument("--n-anomaly", type=int, default=5)
    parser.add_argument("--n-normal", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accuracy", type=float, default=0.8)
    parser.add_argument(
        "--anomaly-weight-grid",
        default="1.0,1.25,1.5,1.75,2.0,2.5,3.0",
        help="Comma-separated weights for anomaly score rescoring. Model forward is run once per prompt.",
    )
    parser.add_argument("--early-exit-layers", default="20:29")
    parser.add_argument("--plot-layers", default="1:37")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.6)
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
    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow-attn-fallback", action="store_true")
    args = parser.parse_args()

    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("Qwen3VLForConditionalGeneration is not available.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sample_images(args.visa_root, args.n_anomaly, args.n_normal, args.seed)
    prompts = load_prompt_candidates(args.prompt_file)
    args._early_exit_layers = parse_int_range(args.early_exit_layers)
    args._plot_layers = parse_int_range(args.plot_layers)
    args._all_layers = sorted(set(args._early_exit_layers + args._plot_layers))
    args._anomaly_weight_grid = [float(x) for x in parse_csv_words(args.anomaly_weight_grid)]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    anomaly_ids, anomaly_detail = first_piece_ids(tokenizer, parse_csv_words(args.anomaly_words))
    normal_ids, normal_detail = first_piece_ids(tokenizer, parse_csv_words(args.normal_words))
    args._class_candidate_ids = sorted(set(anomaly_ids + normal_ids))

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

    manifest_path = out_dir / "sampled_cases.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [{"image": str(p), "label": label} for p, label in cases],
                "anomaly_tokenization": anomaly_detail,
                "normal_tokenization": normal_detail,
                "candidate_mode": args.candidate_mode,
                "class_candidate_ids": args._class_candidate_ids,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    round_rows = []
    best = None
    for round_idx, prompt in enumerate(prompts, start=1):
        question = prompt["question"]
        prefix = prompt["prefix"]
        print(f"\n===== Round {round_idx}/{len(prompts)} =====")
        print(f"question: {question}")
        print(f"prefix: {prefix}")

        case_rows = []
        for image_path, label in cases:
            row = evaluate_case(
                processor,
                tokenizer,
                model,
                image_path,
                label,
                question,
                prefix,
                args,
                anomaly_ids,
                normal_ids,
            )
            case_rows.append(row)
            print(
                f"{Path(image_path).name:>12s} label={label:<7s} pred={row['pred']:<7s} "
                f"ok={int(row['correct'])} layer={row['selected_layer']} "
                f"cal={row['calibrated_token']!r} a={row['anomaly_score']:.3f} n={row['normal_score']:.3f}"
            )

        weighted_summaries = []
        for anomaly_weight in args._anomaly_weight_grid:
            rescored_rows = rescore_case_rows(case_rows, anomaly_weight)
            weighted_summary = summarize_case_rows(rescored_rows)
            weighted_summary["anomaly_weight"] = anomaly_weight
            weighted_summaries.append(weighted_summary)
        best_weighted = max(
            weighted_summaries,
            key=lambda x: (x["accuracy"], x["anomaly_accuracy"], -abs(x["pred_anomaly"] - x["pred_normal"])),
        )
        final_case_rows = rescore_case_rows(case_rows, best_weighted["anomaly_weight"])
        summary = {
            "round": round_idx,
            "question": question,
            "prefix": prefix,
            **best_weighted,
            "case_file": str(out_dir / f"round_{round_idx:02d}_cases.json"),
        }
        (out_dir / f"round_{round_idx:02d}_cases.json").write_text(
            json.dumps(
                {
                    "summary": summary,
                    "weight_summaries": weighted_summaries,
                    "cases": final_case_rows,
                    "raw_cases_anomaly_weight_1": case_rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        round_rows.append(summary)
        if best is None or summary["accuracy"] > best["accuracy"]:
            best = summary
        print(
            f"round accuracy={summary['accuracy']:.3f}, known_rate={summary['known_rate']:.3f}, "
            f"anomaly_acc={summary['anomaly_accuracy']:.3f}, normal_acc={summary['normal_accuracy']:.3f}, "
            f"anomaly_weight={summary['anomaly_weight']:.2f}, "
            f"pred_counts=(a:{summary['pred_anomaly']}, n:{summary['pred_normal']}, u:{summary['pred_unknown']})"
        )
        if summary["accuracy"] >= args.target_accuracy:
            print(f"[STOP] target accuracy reached: {summary['accuracy']:.3f} >= {args.target_accuracy:.3f}")
            break

    with (out_dir / "round_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "round",
            "accuracy",
            "known_rate",
            "correct",
            "total",
            "anomaly_accuracy",
            "normal_accuracy",
            "pred_anomaly",
            "pred_normal",
            "pred_unknown",
            "anomaly_weight",
            "mean_selected_layer",
            "question",
            "prefix",
            "case_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in round_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    (out_dir / "best_prompt.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== Best Prompt =====")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
