#!/usr/bin/env python
import argparse
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for chunk in parse_csv(text):
        out.append(int(chunk))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Case-level interpretability analysis for Qwen3-VL under DeCo in MVTec anomaly detection."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--dataset_root", type=str, required=True, help="MVTec root for category metadata/support.")
    parser.add_argument("--category", type=str, required=True, help="Category name, e.g. bottle.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the single query image.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save analysis JSON.")
    parser.add_argument("--output_plot", type=str, default=None, help="Path to save layer-probability plot.")

    parser.add_argument(
        "--defect_types",
        type=str,
        default=None,
        help="Comma-separated allowed defect labels. If unset, auto-read from dataset_root/category/test/* excluding good.",
    )
    parser.add_argument(
        "--support_images",
        type=str,
        default=None,
        help="Optional comma-separated support image paths. If unset, auto sample from train/good (fallback test/good).",
    )
    parser.add_argument("--shot", type=int, default=3, help="Number of normal reference images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ground_truth_label", type=str, default="good", help="Ground-truth label for logging.")

    parser.add_argument(
        "--assistant_prefix",
        type=str,
        default='{"label": "',
        help="Optional assistant-side prefix appended after chat template. Useful for probing the label decision token.",
    )
    parser.add_argument(
        "--target_words",
        type=str,
        default=None,
        help="Comma-separated target words to probe. If unset, defaults to [good + defect_types].",
    )
    parser.add_argument(
        "--target_token_ids",
        type=str,
        default=None,
        help="Optional comma-separated token IDs to probe in addition to target words.",
    )
    parser.add_argument(
        "--analysis_step",
        type=int,
        default=0,
        help="Generation step index for hidden-state probing. 0 means the first generated token step.",
    )
    parser.add_argument(
        "--analysis_layer_start",
        type=int,
        default=1,
        help="Layer index start (inclusive) for plotting/probing. Default 1 skips embedding layer at 0.",
    )
    parser.add_argument(
        "--analysis_layer_end",
        type=int,
        default=None,
        help="Layer index end (exclusive) for plotting/probing. Default None uses all available layers.",
    )
    parser.add_argument(
        "--report_top_k",
        type=int,
        default=10,
        help="How many top tokens to record per layer in JSON.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=1, help="Generation length for probe run.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--alpha", type=float, default=0.6, help="DeCo alpha.")
    parser.add_argument("--threshold_top_p", type=float, default=0.9, help="DeCo top-p candidate cutoff.")
    parser.add_argument("--threshold_top_k", type=int, default=20, help="DeCo top-k candidate cutoff.")
    parser.add_argument("--start_layer", type=int, default=20, help="DeCo early-exit layer range start (inclusive).")
    parser.add_argument("--end_layer", type=int, default=29, help="DeCo early-exit layer range end (exclusive).")

    parser.add_argument("--min_pixels", type=int, default=640 * 640)
    parser.add_argument("--max_pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation.",
    )
    parser.add_argument(
        "--allow_attn_fallback",
        action="store_true",
        help="Fallback to sdpa when selected attention backend is unavailable.",
    )
    return parser.parse_args()


def load_eval_helpers(eval_script_path: Path):
    spec = importlib.util.spec_from_file_location("mvtec_eval_helpers", str(eval_script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper script: {eval_script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_defect_types(args: argparse.Namespace, helper_module) -> List[str]:
    if args.defect_types:
        return parse_csv(args.defect_types)

    dataset_root = Path(args.dataset_root)
    test_dir = dataset_root / args.category / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test dir for category '{args.category}': {test_dir}")
    return [p.name for p in sorted(test_dir.iterdir()) if p.is_dir() and p.name != "good"]


def resolve_support_paths(args: argparse.Namespace, helper_module) -> List[str]:
    shot = max(0, int(args.shot))
    if shot == 0:
        return []

    if args.support_images:
        provided = [str(Path(p).expanduser().resolve()) for p in parse_csv(args.support_images)]
        if len(provided) == 0:
            raise ValueError("--support_images is set but no valid path is parsed.")
        rng = random.Random(args.seed)
        if len(provided) >= shot:
            return rng.sample(provided, shot)
        return [rng.choice(provided) for _ in range(shot)]

    dataset_root = Path(args.dataset_root)
    train_good_dir = dataset_root / args.category / "train" / "good"
    test_good_dir = dataset_root / args.category / "test" / "good"

    support_pool = helper_module.list_images(train_good_dir)
    if len(support_pool) == 0:
        support_pool = helper_module.list_images(test_good_dir)

    if len(support_pool) == 0:
        raise RuntimeError(
            f"No support good images found for category '{args.category}'. "
            f"Checked: {train_good_dir} and {test_good_dir}."
        )

    support_pool_str = [str(p) for p in support_pool]
    rng = random.Random(args.seed)
    if len(support_pool_str) >= shot:
        return rng.sample(support_pool_str, shot)
    return [rng.choice(support_pool_str) for _ in range(shot)]


def resolve_target_words(args: argparse.Namespace, defect_types: List[str]) -> List[str]:
    if args.target_words:
        words = parse_csv(args.target_words)
    else:
        words = ["good"] + list(defect_types)
    out = []
    seen = set()
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def build_target_specs(
    tokenizer,
    target_words: List[str],
    target_token_ids: List[int],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    label_count: Dict[str, int] = {}

    def unique_label(base: str) -> str:
        n = label_count.get(base, 0) + 1
        label_count[base] = n
        return base if n == 1 else f"{base}#{n}"

    for word in target_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        used_space_prefix_fallback = False
        if len(ids) == 0 and not word.startswith(" "):
            ids = tokenizer.encode(" " + word, add_special_tokens=False)
            used_space_prefix_fallback = True
        if len(ids) == 0:
            continue
        first_id = int(ids[0])
        specs.append(
            {
                "display_name": unique_label(word),
                "source": "word",
                "target_text": word,
                "token_id": first_id,
                "token_text": tokenizer.convert_ids_to_tokens(first_id),
                "tokenized_ids": [int(x) for x in ids],
                "tokenized_tokens": tokenizer.convert_ids_to_tokens(ids),
                "uses_first_piece": len(ids) > 1,
                "used_space_prefix_fallback": used_space_prefix_fallback,
            }
        )

    for token_id in target_token_ids:
        token_text = tokenizer.convert_ids_to_tokens(int(token_id))
        base = f"id:{int(token_id)}"
        specs.append(
            {
                "display_name": unique_label(base),
                "source": "token_id",
                "target_text": None,
                "token_id": int(token_id),
                "token_text": token_text,
                "tokenized_ids": [int(token_id)],
                "tokenized_tokens": [token_text],
                "uses_first_piece": False,
                "used_space_prefix_fallback": False,
            }
        )
    return specs


def build_generation_kwargs(args: argparse.Namespace, early_exit_layers: List[int]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "num_beams": args.num_beams,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "use_deco": True,
        "alpha": args.alpha,
        "threshold_top_p": args.threshold_top_p,
        "threshold_top_k": args.threshold_top_k,
        "early_exit_layers": early_exit_layers,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "output_scores": True,
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def resolve_layer_indices(
    total_layers: int,
    analysis_layer_start: int,
    analysis_layer_end: Optional[int],
) -> List[int]:
    start = max(0, int(analysis_layer_start))
    end = total_layers if analysis_layer_end is None else min(total_layers, int(analysis_layer_end))
    if start >= end:
        raise ValueError(
            f"Invalid analysis layer range: start={start}, end={end}, total_layers={total_layers}. "
            "Adjust --analysis_layer_start/--analysis_layer_end."
        )
    return list(range(start, end))


def analyze_layer_probabilities(
    *,
    model,
    tokenizer,
    step_hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    target_specs: List[Dict[str, Any]],
    report_top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    norm_layer = model.model.language_model.norm
    lm_head = model.lm_head

    rows: List[Dict[str, Any]] = []
    curves: Dict[str, List[float]] = {spec["display_name"]: [] for spec in target_specs}
    dominant_target_by_layer: List[Dict[str, Any]] = []
    top_k_tokens_by_layer: List[Dict[str, Any]] = []

    with torch.no_grad():
        for layer_idx in layer_indices:
            last_token_hidden = step_hidden_states[layer_idx][:, -1, :]
            logits = lm_head(norm_layer(last_token_hidden))
            probs = F.softmax(logits.float(), dim=-1)[0]

            row: Dict[str, Any] = {"layer_index": int(layer_idx)}
            best_name = None
            best_prob = -1.0
            for spec in target_specs:
                token_id = int(spec["token_id"])
                p = float(probs[token_id].item())
                name = spec["display_name"]
                row[name] = p
                curves[name].append(p)
                if p > best_prob:
                    best_prob = p
                    best_name = name

            rows.append(row)
            dominant_target_by_layer.append(
                {
                    "layer_index": int(layer_idx),
                    "target": best_name,
                    "prob": float(best_prob),
                }
            )

            k = max(1, min(int(report_top_k), int(probs.shape[-1])))
            top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
            top_tokens = []
            for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                token_id = int(token_id)
                top_tokens.append(
                    {
                        "token_id": token_id,
                        "token": tokenizer.convert_ids_to_tokens(token_id),
                        "prob": float(prob),
                    }
                )
            top_k_tokens_by_layer.append(
                {
                    "layer_index": int(layer_idx),
                    "top_tokens": top_tokens,
                }
            )

    return rows, curves, dominant_target_by_layer, top_k_tokens_by_layer


def save_plot(layer_indices: List[int], curves: Dict[str, List[float]], output_plot: Path) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for label, ys in curves.items():
        plt.plot(layer_indices, ys, marker="o", linewidth=1.8, markersize=4.0, label=label)
    plt.xlabel("Layer Index")
    plt.ylabel("Next-token Probability")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.analysis_step < 0:
        raise ValueError("--analysis_step must be >= 0.")
    if args.max_new_tokens <= args.analysis_step:
        raise ValueError(
            f"--max_new_tokens ({args.max_new_tokens}) must be > --analysis_step ({args.analysis_step})."
        )
    if args.report_top_k <= 0:
        raise ValueError("--report_top_k must be > 0.")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    query_image_path = Path(args.image_path)
    if not query_image_path.exists():
        raise FileNotFoundError(f"image_path does not exist: {query_image_path}")

    early_exit_layers = list(range(args.start_layer, args.end_layer))
    if len(early_exit_layers) == 0:
        raise ValueError("`start_layer` must be smaller than `end_layer`.")

    eval_script_path = Path(__file__).with_name("eval_mvtec_qwen3_vl_thinking.py")
    helper_module = load_eval_helpers(eval_script_path)

    defect_types = resolve_defect_types(args, helper_module)
    support_paths = resolve_support_paths(args, helper_module)
    target_words = resolve_target_words(args, defect_types)
    target_token_ids = parse_int_csv(args.target_token_ids) if args.target_token_ids else []
    if len(target_words) == 0 and len(target_token_ids) == 0:
        raise ValueError("No targets to analyze. Set --target_words or --target_token_ids.")

    query_image = Image.open(query_image_path).convert("RGB")
    image_size = query_image.size
    support_images = [Image.open(p).convert("RGB") for p in support_paths]
    all_images = support_images + [query_image]

    messages = helper_module.build_messages(category=args.category, defect_types=defect_types, shot=args.shot)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = processor.tokenizer

    target_specs = build_target_specs(tokenizer, target_words, target_token_ids)
    if len(target_specs) == 0:
        raise RuntimeError(
            "All target words/token ids failed to map to valid tokens. "
            "Try explicit --target_token_ids from your tokenizer."
        )

    attn_impl = args.attn_implementation
    if device != "cuda" and attn_impl == "flash_attention_2":
        if args.allow_attn_fallback:
            print("[WARN] flash_attention_2 requires CUDA. Fallback to sdpa.")
            attn_impl = "sdpa"
        else:
            raise RuntimeError("flash_attention_2 requires CUDA. Use --allow_attn_fallback or --attn_implementation sdpa.")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else None,
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
    except Exception as e:
        if args.allow_attn_fallback and attn_impl == "flash_attention_2":
            print(f"[WARN] Failed to load with flash_attention_2: {e}")
            print("[WARN] Fallback to sdpa.")
            load_kwargs["attn_implementation"] = "sdpa"
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)
        else:
            raise

    if device != "cuda":
        model.to(device)
    model.eval()

    if not args.do_sample:
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.assistant_prefix:
        text = text + args.assistant_prefix

    inputs = processor(
        text=[text],
        images=all_images,
        return_tensors="pt",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_token_len = int(inputs["input_ids"].shape[-1])

    generation_kwargs = build_generation_kwargs(args, early_exit_layers)
    with torch.no_grad():
        output_dict = model.generate(**inputs, **generation_kwargs)

    if not hasattr(output_dict, "hidden_states") or output_dict.hidden_states is None:
        raise RuntimeError("Generation output has no hidden_states. Ensure return_dict_in_generate/output_hidden_states are enabled.")
    if len(output_dict.hidden_states) == 0:
        raise RuntimeError("Generation output hidden_states is empty.")
    if args.analysis_step >= len(output_dict.hidden_states):
        raise ValueError(
            f"analysis_step={args.analysis_step} out of range. "
            f"Only {len(output_dict.hidden_states)} generation steps are available."
        )

    if not hasattr(output_dict, "sequences"):
        raise RuntimeError("Expected GenerateDecoderOnlyOutput with `sequences`.")
    gen_ids = output_dict.sequences[:, input_token_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    step_hidden_states = output_dict.hidden_states[args.analysis_step]
    if step_hidden_states is None or len(step_hidden_states) == 0:
        raise RuntimeError(f"Hidden states for analysis_step={args.analysis_step} are empty.")
    total_layers = len(step_hidden_states)
    layer_indices = resolve_layer_indices(total_layers, args.analysis_layer_start, args.analysis_layer_end)

    (
        layer_probability_rows,
        target_curves,
        dominant_target_by_layer,
        top_k_tokens_by_layer,
    ) = analyze_layer_probabilities(
        model=model,
        tokenizer=tokenizer,
        step_hidden_states=step_hidden_states,
        layer_indices=layer_indices,
        target_specs=target_specs,
        report_top_k=args.report_top_k,
    )

    analysis_token_id = None
    analysis_token = None
    analysis_token_decoded = None
    if gen_ids.shape[-1] > args.analysis_step:
        analysis_token_id = int(gen_ids[0, args.analysis_step].item())
        analysis_token = tokenizer.convert_ids_to_tokens(analysis_token_id)
        analysis_token_decoded = tokenizer.decode([analysis_token_id], skip_special_tokens=False)

    parsed, parse_source = helper_module.parse_model_json(raw_text)
    prediction = helper_module.normalize_prediction(
        parsed=parsed,
        parse_source=parse_source,
        raw_text=raw_text,
        defect_types=defect_types,
        image_size=image_size,
    )
    gt_label_norm = helper_module.map_label(args.ground_truth_label, defect_types)

    output_json = Path(args.output_json)
    output_plot = Path(args.output_plot) if args.output_plot else output_json.with_suffix(".png")
    save_plot(layer_indices, target_curves, output_plot)

    output_data: Dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name_or_path": args.model_name_or_path,
            "dataset_root": str(dataset_root),
            "category": args.category,
            "image_path": str(query_image_path),
            "ground_truth_label_raw": args.ground_truth_label,
            "ground_truth_label_normalized": gt_label_norm,
            "defect_types": defect_types,
            "shot": args.shot,
            "support_image_paths": support_paths,
            "prompt_source": "eval_mvtec_qwen3_vl_thinking.py::build_messages",
            "assistant_prefix": args.assistant_prefix,
            "use_deco": True,
            "deco_alpha": args.alpha,
            "deco_threshold_top_p": args.threshold_top_p,
            "deco_threshold_top_k": args.threshold_top_k,
            "deco_early_exit_layers": early_exit_layers,
            "analysis_step": args.analysis_step,
            "analysis_layer_indices": layer_indices,
            "report_top_k": args.report_top_k,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "num_beams": args.num_beams,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "attn_implementation": load_kwargs["attn_implementation"],
            "allow_attn_fallback": args.allow_attn_fallback,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "prompt": {
            "text": text,
        },
        "targets": target_specs,
        "generation": {
            "generated_token_ids": [int(x) for x in gen_ids[0].tolist()],
            "raw_text": raw_text,
            "analysis_token_id": analysis_token_id,
            "analysis_token": analysis_token,
            "analysis_token_decoded": analysis_token_decoded,
            "parsed_json": parsed,
            "parse_source": parse_source,
            "normalized_prediction": prediction,
        },
        "analysis": {
            "layer_probabilities": layer_probability_rows,
            "target_curves": target_curves,
            "dominant_target_by_layer": dominant_target_by_layer,
            "top_k_tokens_by_layer": top_k_tokens_by_layer,
        },
        "artifacts": {
            "output_json": str(output_json),
            "output_plot": str(output_plot),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    final_layer_dominant = dominant_target_by_layer[-1] if len(dominant_target_by_layer) > 0 else None

    print("===== Case-level DeCo Interpretability =====")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"category: {args.category}")
    print(f"query_image: {query_image_path.name}")
    print(f"shot: {args.shot}")
    print(f"support_images: {[Path(p).name for p in support_paths]}")
    print(f"defect_types: {defect_types}")
    print(f"assistant_prefix: {args.assistant_prefix!r}")
    print(f"analysis_step: {args.analysis_step}")
    print(f"analysis_layers: [{layer_indices[0]}, {layer_indices[-1]}] total={len(layer_indices)}")
    print(f"targets: {[x['display_name'] for x in target_specs]}")
    if final_layer_dominant is not None:
        print(
            "final_layer_dominant_target: "
            f"{final_layer_dominant['target']} (p={final_layer_dominant['prob']:.6f})"
        )
    print(f"Saved JSON result to: {output_json}")
    print(f"Saved probability plot to: {output_plot}")


if __name__ == "__main__":
    main()
