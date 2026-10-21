#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisA pcb1 layer evidence analysis for Qwen3-VL.

This script does ONE thing:
1) Use anomaly_prompt.json to define anomaly evidence tokens (V_def)
2) Run anomaly images from VisA/category (default pcb1)
3) Compute per-layer token probability mass statistics
4) Draw figures similar to "Count vs Layer" for:
   - anomaly evidence distribution
   - visual information distribution

Notes:
- V_def is image-specific: each anomaly image uses its own JSON labels as positive evidence words.
- V_norm is a fixed normal-prior lexicon.
- V_hedge is optional and only saved as reference here; the default plots do not subtract it.
- "Visual information" here is an operational proxy, not a perfect ground-truth semantic probe:
  it measures probability mass on category/object-related words such as pcb/board/sensor/module/pin/component.

Outputs:
- <output_dir>/<category>_anomaly_evidence_counts.png
- <output_dir>/<category>_visual_evidence_counts.png
- <output_dir>/<category>_layer_evidence_stats.json
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VisA pcb1 layer evidence analysis")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to VisA root directory.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--category", type=str, default="pcb1", help="VisA category, default pcb1.")
    parser.add_argument("--gpu", type=str, default="", help="CUDA_VISIBLE_DEVICES interface, e.g. 0 or 6.")
    parser.add_argument("--output_dir", type=str, default="./runs/layer_evidence_analysis")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on anomaly images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pixels", type=int, default=262144)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--allow_attn_fallback", action="store_true")
    parser.add_argument("--thresholds", type=str, default="1e-6,5e-6,1e-5")
    parser.add_argument("--visual_words", type=str, default="pcb,board,circuit board,sensor,module,component,pin,solder,transducer")
    parser.add_argument("--norm_words", type=str, default="good,normal,intact,clean,smooth,uniform,no defect,undamaged")
    parser.add_argument("--hedge_words", type=str, default="maybe,seems,likely,appears,possibly,could,might")
    parser.add_argument("--gamma", type=float, default=8.0, help="For optional logsumexp stats in output JSON.")
    parser.add_argument("--lambda_norm", type=float, default=1.0)
    parser.add_argument("--eta_hedge", type=float, default=0.2)
    parser.add_argument("--prompt_mode", type=str, default="label_only", choices=["label_only", "json_style"])
    return parser.parse_args()


def parse_csv_words(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def split_prompt_labels(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]

def expand_defect_words(labels: List[str]) -> List[str]:
    expanded = []
    extras = {
        "excessive solder": ["solder", "excessive", "extra solder", "overflow", "metal residue"],
        "missing component": ["missing", "component", "absent", "lost", "not present"],
        "bent component": ["bent", "deformed", "misaligned", "crooked"],
        "scratch": ["scratch", "scratched", "mark", "surface damage", "damaged"],
    }

    for lb in labels:
        lb = lb.strip()
        if not lb:
            continue
        expanded.append(lb)
        if lb in extras:
            expanded.extend(extras[lb])

    # 去重
    seen = set()
    out = []
    for w in expanded:
        k = w.lower()
        if k not in seen:
            seen.add(k)
            out.append(w)
    return out


def resolve_image_path(category_dir: Path, image_key: str) -> Optional[Path]:
    p = Path(image_key)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(category_dir / image_key)
        candidates.append(category_dir / p.name)
        candidates.append(category_dir / "Data" / "Images" / "Anomaly" / p.name)
        candidates.append(category_dir / "Data" / "Images" / "Normal" / p.name)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None


def load_anomaly_prompt_annotations(category_dir: Path) -> List[Dict]:
    json_path = category_dir / "anomaly_prompt.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing anomaly_prompt.json: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} is not a list")

    items = []
    for item in data:
        if not isinstance(item, dict):
            continue

        image_key = None
        for k in ["image", "img", "img_path", "path", "file", "filename", "name"]:
            if k in item and str(item[k]).strip():
                image_key = str(item[k]).strip()
                break

        prompt_text = str(item.get("prompt", "")).strip()
        labels = split_prompt_labels(prompt_text)

        if image_key is None or len(labels) == 0:
            continue

        image_path = resolve_image_path(category_dir, image_key)
        if image_path is None:
            continue

        items.append(
            {
                "image_key": image_key,
                "image_path": str(image_path),
                "labels": labels,
            }
        )
    return items


def tokenize_words_to_ids(tokenizer, words: Sequence[str]) -> Set[int]:
    ids: Set[int] = set()
    for w in words:
        token_ids = tokenizer.encode(str(w), add_special_tokens=False)
        for tid in token_ids:
            ids.add(int(tid))
    return ids


def build_anomaly_prompt(category: str, all_defect_types: List[str], prompt_mode: str) -> str:
    candidate_text = ", ".join(["good"] + all_defect_types)
    if prompt_mode == "label_only":
        return (
            f"Category: {category}\n"
            f"Candidate labels: [{candidate_text}]\n"
            f"This is anomaly detection, not ordinary classification.\n"
            f"Output exactly one label from the candidate list.\n"
            f"Label:"
        )
    return (
        f"You are an image anomaly detection assistant.\n"
        f"Current category: {category}\n"
        f"Candidate labels: [{candidate_text}]\n"
        f"Output JSON only in this format:\n"
        f'{{"label": ""}}\n'
    )


def build_visual_prompt(category: str) -> str:
    return (
        f"Category: {category}\n"
        f"Describe the main visible object/content in one short phrase.\n"
        f"Answer:"
    )


def get_final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def layer_logits_from_hidden(hidden_state_last_token: torch.Tensor, lm_head, final_norm=None) -> torch.Tensor:
    x = hidden_state_last_token
    if final_norm is not None:
        x = final_norm(x)
    logits = lm_head(x)
    return logits


def safe_prob_mass(probs: torch.Tensor, token_ids: Sequence[int]) -> float:
    if len(token_ids) == 0:
        return 0.0
    valid_ids = [tid for tid in token_ids if 0 <= tid < probs.shape[-1]]
    if len(valid_ids) == 0:
        return 0.0
    return float(probs[valid_ids].sum().item())


def safe_logsum_score(probs: torch.Tensor, token_ids: Sequence[int], gamma: float) -> float:
    if len(token_ids) == 0:
        return float("-inf")
    valid_ids = [tid for tid in token_ids if 0 <= tid < probs.shape[-1]]
    if len(valid_ids) == 0:
        return float("-inf")
    p = probs[valid_ids].clamp_min(1e-12)
    vals = gamma * torch.log(p)
    m = torch.max(vals)
    score = (m + torch.log(torch.exp(vals - m).sum())) / gamma
    return float(score.item())


def run_single_forward(
    model,
    processor,
    prompt_text: str,
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
) -> List[torch.Tensor]:
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 1:
        raise RuntimeError("Model did not return hidden_states.")

    final_norm = get_final_norm_module(model)
    lm_head = model.get_output_embeddings()

    layer_probs = []
    for layer_idx in range(1, len(hidden_states)):
        hs = hidden_states[layer_idx][:, -1, :]
        logits = layer_logits_from_hidden(hs, lm_head, final_norm=final_norm)
        probs = torch.softmax(logits[0].float(), dim=-1).cpu()
        layer_probs.append(probs)

    return layer_probs


def draw_count_plot(
    save_path: Path,
    layers: List[int],
    threshold_to_counts: Dict[float, List[int]],
    title: str,
):
    plt.figure(figsize=(9, 5.5))
    for thr in sorted(threshold_to_counts.keys()):
        plt.plot(layers, threshold_to_counts[thr], linewidth=2, label=f"threshold={thr}")
    plt.xlabel("Layer", fontsize=16, fontweight="bold")
    plt.ylabel("Count", fontsize=16, fontweight="bold")
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_mean_mass_plot(
    save_path: Path,
    layers: List[int],
    mean_mass: List[float],
    title: str,
    ylabel: str,
):
    plt.figure(figsize=(9, 5.5))
    plt.plot(layers, mean_mass, linewidth=2)
    plt.xlabel("Layer", fontsize=16, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def draw_score_plot(
    save_path: Path,
    layers: List[int],
    scores: List[float],
    title: str,
    ylabel: str = "Mean score",
):
    plt.figure(figsize=(9, 5.5))
    plt.plot(layers, scores, linewidth=2)
    plt.xlabel("Layer", fontsize=16, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    category_dir = dataset_root / args.category
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_anomaly_prompt_annotations(category_dir)
    if len(items) == 0:
        raise RuntimeError(f"No anomaly annotations found for category={args.category}")

    if args.max_images is not None and len(items) > args.max_images:
        items = items[: args.max_images]

    all_defect_types = sorted({lb for item in items for lb in item["labels"]})
    thresholds = [float(x) for x in parse_csv_words(args.thresholds)]
    visual_words = parse_csv_words(args.visual_words)
    norm_words = parse_csv_words(args.norm_words)
    hedge_words = parse_csv_words(args.hedge_words)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    attn_impl = args.attn_implementation
    if device != "cuda" and attn_impl == "flash_attention_2":
        if args.allow_attn_fallback:
            attn_impl = "sdpa"
        else:
            raise RuntimeError("flash_attention_2 requires CUDA.")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else None,
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    tokenizer = processor.tokenizer
    v_norm_ids = tokenize_words_to_ids(tokenizer, norm_words)
    v_hedge_ids = tokenize_words_to_ids(tokenizer, hedge_words)
    v_vis_ids = tokenize_words_to_ids(tokenizer, visual_words)

    anomaly_prompt = build_anomaly_prompt(args.category, all_defect_types, args.prompt_mode)
    visual_prompt = build_visual_prompt(args.category)

    anomaly_count_by_thr: Dict[float, Optional[List[int]]] = {thr: None for thr in thresholds}
    visual_count_by_thr: Dict[float, Optional[List[int]]] = {thr: None for thr in thresholds}
    anomaly_mass_sum: Optional[List[float]] = None
    visual_mass_sum: Optional[List[float]] = None

    per_image_records = []

    pbar = tqdm(items, desc=f"Layer evidence analysis: {args.category}", ncols=120)
    for item in pbar:
        image_path = Path(item["image_path"])
        image = Image.open(image_path).convert("RGB")

        v_def_words = expand_defect_words(item["labels"])
        v_def_ids = tokenize_words_to_ids(tokenizer, v_def_words)

        anomaly_layer_probs = run_single_forward(
            model=model,
            processor=processor,
            prompt_text=anomaly_prompt,
            image=image,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        visual_layer_probs = run_single_forward(
            model=model,
            processor=processor,
            prompt_text=visual_prompt,
            image=image,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        n_layers = len(anomaly_layer_probs)
        if anomaly_mass_sum is None:
            anomaly_mass_sum = [0.0] * n_layers
            visual_mass_sum = [0.0] * n_layers
            for thr in thresholds:
                anomaly_count_by_thr[thr] = [0] * n_layers
                visual_count_by_thr[thr] = [0] * n_layers

        image_record = {
            "image_path": str(image_path),
            "labels": item["labels"],
            "anomaly_mass_by_layer": [],
            "visual_mass_by_layer": [],
            "anomaly_score_by_layer": [],
            "visual_score_by_layer": [],
        }

        for i in range(n_layers):
            probs_a = anomaly_layer_probs[i]
            probs_v = visual_layer_probs[i]

            def_mass = safe_prob_mass(probs_a, list(v_def_ids))
            norm_mass = safe_prob_mass(probs_a, list(v_norm_ids))
            hedge_mass = safe_prob_mass(probs_a, list(v_hedge_ids))

            vis_mass = safe_prob_mass(probs_v, list(v_vis_ids))
            vis_norm_mass = safe_prob_mass(probs_v, list(v_norm_ids))

            def_score = safe_logsum_score(probs_a, list(v_def_ids), args.gamma)
            norm_score = safe_logsum_score(probs_a, list(v_norm_ids), args.gamma)
            hedge_score = safe_logsum_score(probs_a, list(v_hedge_ids), args.gamma)

            vis_score = safe_logsum_score(probs_v, list(v_vis_ids), args.gamma)
            vis_norm_score = safe_logsum_score(probs_v, list(v_norm_ids), args.gamma)

            anom_score = None
            if (
                math.isfinite(def_score)
                and math.isfinite(norm_score)
                and math.isfinite(hedge_score)
            ):
                anom_score = def_score - args.lambda_norm * norm_score - args.eta_hedge * hedge_score

            vis_final_score = None
            if math.isfinite(vis_score) and math.isfinite(vis_norm_score):
                vis_final_score = vis_score - args.lambda_norm * vis_norm_score

            anomaly_mass_sum[i] += def_mass
            visual_mass_sum[i] += vis_mass

            for thr in thresholds:
                if def_mass >= thr:
                    anomaly_count_by_thr[thr][i] += 1
                if vis_mass >= thr:
                    visual_count_by_thr[thr][i] += 1

            image_record["anomaly_mass_by_layer"].append(def_mass)
            image_record["visual_mass_by_layer"].append(vis_mass)
            image_record["anomaly_score_by_layer"].append(anom_score)
            image_record["visual_score_by_layer"].append(vis_final_score)

        per_image_records.append(image_record)

    num_images = len(items)
    layers = list(range(1, len(anomaly_mass_sum) + 1))
    anomaly_mean_mass = [x / num_images for x in anomaly_mass_sum]
    visual_mean_mass = [x / num_images for x in visual_mass_sum]
    
    anomaly_mean_score = []
    visual_mean_score = []
    for i in range(len(layers)):
        vals_a = [rec["anomaly_score_by_layer"][i] for rec in per_image_records if rec["anomaly_score_by_layer"][i] is not None]
        vals_v = [rec["visual_score_by_layer"][i] for rec in per_image_records if rec["visual_score_by_layer"][i] is not None]
        anomaly_mean_score.append(sum(vals_a) / len(vals_a) if len(vals_a) > 0 else 0.0)
        visual_mean_score.append(sum(vals_v) / len(vals_v) if len(vals_v) > 0 else 0.0)

    anomaly_plot = output_dir / f"{args.category}_anomaly_evidence_counts.png"
    visual_plot = output_dir / f"{args.category}_visual_evidence_counts.png"
    anomaly_mean_plot = output_dir / f"{args.category}_anomaly_mean_mass.png"
    visual_mean_plot = output_dir / f"{args.category}_visual_mean_mass.png"
    anomaly_score_plot = output_dir / f"{args.category}_anomaly_score.png"
    visual_score_plot = output_dir / f"{args.category}_visual_score.png"

    draw_count_plot(
        save_path=anomaly_plot,
        layers=layers,
        threshold_to_counts=anomaly_count_by_thr,
        title=f"{args.category}: anomaly evidence distribution across layers",
    )
    draw_count_plot(
        save_path=visual_plot,
        layers=layers,
        threshold_to_counts=visual_count_by_thr,
        title=f"{args.category}: visual information distribution across layers",
    )
    draw_mean_mass_plot(
        save_path=anomaly_mean_plot,
        layers=layers,
        mean_mass=anomaly_mean_mass,
        title=f"{args.category}: mean anomaly evidence mass across layers",
        ylabel="Mean probability mass",
    )
    draw_mean_mass_plot(
        save_path=visual_mean_plot,
        layers=layers,
        mean_mass=visual_mean_mass,
        title=f"{args.category}: mean visual evidence mass across layers",
        ylabel="Mean probability mass",
    )
    draw_score_plot(
        save_path=anomaly_score_plot,
        layers=layers,
        scores=anomaly_mean_score,
        title=f"{args.category}: anomaly score across layers",
    )

    draw_score_plot(
        save_path=visual_score_plot,
        layers=layers,
        scores=visual_mean_score,
        title=f"{args.category}: visual score across layers",
    )

    save_json = {
        "meta": {
            "dataset_root": str(dataset_root),
            "model_name_or_path": args.model_name_or_path,
            "category": args.category,
            "num_images": num_images,
            "thresholds": thresholds,
            "prompt_mode": args.prompt_mode,
            "attn_implementation": attn_impl,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "token_sets": {
            "all_defect_types": all_defect_types,
            "visual_words": visual_words,
            "norm_words": norm_words,
            "hedge_words": hedge_words,
        },
        "layers": layers,
        "anomaly_count_by_threshold": {str(k): v for k, v in anomaly_count_by_thr.items()},
        "visual_count_by_threshold": {str(k): v for k, v in visual_count_by_thr.items()},
        "anomaly_mean_mass": anomaly_mean_mass,
        "visual_mean_mass": visual_mean_mass,
        "per_image_records": per_image_records,
        "output_files": {
            "anomaly_plot": str(anomaly_plot),
            "visual_plot": str(visual_plot),
            "anomaly_mean_plot": str(anomaly_mean_plot),
            "visual_mean_plot": str(visual_mean_plot),
        },
    }

    json_path = output_dir / f"{args.category}_layer_evidence_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Layer evidence analysis finished")
    print(f"Category        : {args.category}")
    print(f"Images analyzed : {num_images}")
    print(f"JSON stats      : {json_path}")
    print(f"Anomaly plot    : {anomaly_plot}")
    print(f"Visual plot     : {visual_plot}")
    print(f"Mean anomaly    : {anomaly_mean_plot}")
    print(f"Mean visual     : {visual_mean_plot}")
    print("=" * 80)


if __name__ == "__main__":
    main()
