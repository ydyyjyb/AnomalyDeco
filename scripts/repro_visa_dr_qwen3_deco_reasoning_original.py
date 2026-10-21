#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run VisA-D&R reasoning with Qwen3-VL and Qwen3-VL + DeCo.

This script intentionally follows the original VisA-D&R reasoning protocol:

1. Read the official ``visa_dr.json``.
2. Skip ``pcb2`` and ``pcb4`` by default.
3. For each anomaly image, ask the benchmark questions as-is.
4. Use image input only on the first question and keep previous Q/A history for
   later questions.
5. Compute ROUGE-L / SBERT metrics against the official answers.

No HDEC score, no Youden threshold, no fixed Answer, and no answer-consistency
post-processing are used here.  The only difference between the two reported
rows is whether generation is called with DeCo kwargs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from repro_visa_dr_reasoning import (
    category_from_rel,
    evaluate_reasoning,
    fmt,
    norm_rel_image,
    parse_int_range,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
)


DEFAULT_ADAR_ROOT = "/home/vina04/ADAR"
DEFAULT_DATA_ROOT = f"{DEFAULT_ADAR_ROOT}/evaluation/VisA-DR-official"
DEFAULT_VISA_DR_JSON = f"{DEFAULT_DATA_ROOT}/VisA/visa_dr.json"
DEFAULT_MODEL_PATH = f"{DEFAULT_ADAR_ROOT}/Deco-main/qwen3_4b_vl_instruct"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_ADAR_ROOT}/Deco-main/runs"
DEFAULT_SKIP_CLASSES = ["pcb2", "pcb4"]

VARIANT_CONFIGS = {
    "qwen3_original": {
        "method": "Qwen3-VL-4B-Instruct",
        "use_deco_generation": False,
    },
    "qwen3_deco": {
        "method": "Qwen3-VL-4B-Instruct + DeCo",
        "use_deco_generation": True,
    },
}


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def load_image(path: str, max_side: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_side and max_side > 0 and max(image.size) > max_side:
        if image.width >= image.height:
            new_width = max_side
            new_height = int(max_side * image.height / image.width)
        else:
            new_height = max_side
            new_width = int(max_side * image.width / image.height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def load_reasoning_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data = read_json(Path(args.visa_dr_json))
    skip = set(args.skip_classes)
    items: List[Dict[str, Any]] = []
    for i, raw in enumerate(data):
        rel = norm_rel_image(str(raw["image"]), args.data_root)
        category = category_from_rel(rel)
        if category in skip:
            continue
        abs_image = Path(args.data_root) / rel
        if not abs_image.exists():
            raise FileNotFoundError(f"missing image: {abs_image}")
        row = dict(raw)
        row["index"] = str(row.get("index", i))
        row["image"] = rel
        row["rel_image"] = rel
        row["abs_image"] = str(abs_image)
        row["category"] = category
        items.append(row)

    if args.limit and args.limit > 0:
        items = items[: args.limit]
    return items


def format_question(question: Any, answer_instruction: str = "") -> str:
    text = str(question)
    instruction = str(answer_instruction or "").strip()
    if instruction:
        text = f"{text}\n\n{instruction}"
    return text


def build_messages(
    item: Dict[str, Any],
    question_index: int,
    previous_responses: Sequence[str],
    answer_instruction: str = "",
) -> List[Dict[str, Any]]:
    questions = list(item["questions"])
    messages: List[Dict[str, Any]] = []
    for i in range(question_index):
        question_text = format_question(questions[i], answer_instruction)
        if i == 0:
            content = [{"type": "image"}, {"type": "text", "text": question_text}]
        else:
            content = [{"type": "text", "text": question_text}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(previous_responses[i])}]})

    question_text = format_question(questions[question_index], answer_instruction)
    if question_index == 0:
        content = [{"type": "image"}, {"type": "text", "text": question_text}]
    else:
        content = [{"type": "text", "text": question_text}]
    messages.append({"role": "user", "content": content})
    return messages


def load_qwen3(args: argparse.Namespace) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"

    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return processor, model


def encode_bad_words(tokenizer: Any, words: Sequence[str]) -> List[List[int]]:
    if tokenizer is None:
        return []
    bad_words: List[List[int]] = []
    seen = set()
    for word in words:
        text = str(word or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(int(x) for x in ids)
        if key in seen:
            continue
        seen.add(key)
        bad_words.append(list(key))
    return bad_words


class SuppressTokenIdsLogitsProcessor:
    """Hard-suppress selected token ids after all upstream logit edits.

    DeCo rewrites the next-token logits before Hugging Face's logits processor
    list is applied.  Passing this processor through ``generate`` therefore
    suppresses punctuation/URL tokens after the DeCo correction as well.
    """

    def __init__(self, token_ids: Sequence[int]):
        self.token_ids = sorted({int(token_id) for token_id in token_ids})

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.token_ids:
            scores[:, self.token_ids] = -float("inf")
        return scores


def collect_suppressed_token_ids(tokenizer: Any, texts: Sequence[str], pad_token_id: int | None, eos_token_id: int | None) -> List[int]:
    if tokenizer is None:
        return []
    excluded = {token_id for token_id in [pad_token_id, eos_token_id] if token_id is not None}
    # Do not suppress pure whitespace/newline tokens, because they are harmless
    # and some tokenizers compose punctuation with nearby whitespace.
    excluded_texts = {"", " ", "\n", "\t"}
    token_ids = set()
    for text in texts:
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        for token_id in ids:
            token_text = tokenizer.decode([token_id])
            if token_id in excluded or token_text in excluded_texts:
                continue
            token_ids.add(int(token_id))
    return sorted(token_ids)


def make_generation_kwargs(
    args: argparse.Namespace,
    use_deco_generation: bool,
    pad_token_id: int | None,
    eos_token_id: int | None,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.ban_markdown_url:
        banned = encode_bad_words(
            tokenizer,
            [
                "http",
                "https",
                "://",
                "www",
                ".com",
                "!",
                "！",
                "[",
                "]",
                "![",
                "![]",
                "![](",
                "](",
                "!!",
                "!!!",
                " !",
                " !!",
                "\n!",
                "\n!!",
            ],
        )
        if banned:
            kwargs["bad_words_ids"] = banned
        suppress_ids = collect_suppressed_token_ids(
            tokenizer,
            [
                "!",
                "!!",
                "!!!",
                " !",
                " !!",
                "\n!",
                "\n!!",
                "！",
                " ！",
                "[",
                "]",
                "![",
                "![]",
                "![](",
                "](",
                "http",
                "https",
                "://",
                "www",
                ".com",
                " https",
            ],
            pad_token_id,
            eos_token_id,
        )
        if suppress_ids:
            kwargs["logits_processor"] = [SuppressTokenIdsLogitsProcessor(suppress_ids)]
    if use_deco_generation:
        kwargs.update(
            {
                "use_deco": True,
                "alpha": args.deco_alpha,
                "threshold_top_p": args.deco_threshold_top_p,
                "threshold_top_k": args.deco_threshold_top_k,
                "early_exit_layers": parse_int_range(args.deco_layers),
            }
        )
    return kwargs


def run_variant_generation(
    args: argparse.Namespace,
    variant: str,
    items: Sequence[Dict[str, Any]],
    processor: Any,
    model: Any,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    import torch

    config = VARIANT_CONFIGS[variant]
    use_deco_generation = bool(config["use_deco_generation"])
    predictions_path = out_dir / "reasoning_predictions.jsonl"
    rows = read_jsonl(predictions_path) if args.resume and predictions_path.exists() else []
    done = {str(row.get("index")) for row in rows}
    todo = [item for item in items if str(item.get("index")) not in done]
    print(f"[{variant}] total={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return rows

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generation_kwargs = make_generation_kwargs(args, use_deco_generation, pad_token_id, eos_token_id, tokenizer)

    for item_i, item in enumerate(todo, start=1):
        if item_i == 1 or item_i % args.log_every == 0 or item_i == len(todo):
            print(f"[{variant}] image={item_i}/{len(todo)} {item['category']} {item['image']}", flush=True)

        image = load_image(str(item["abs_image"]), args.max_image_side)
        responses: List[str] = []
        token_counts: List[int] = []
        hit_max_new_tokens: List[bool] = []
        seconds_per_question: List[float] = []

        for q_idx in range(len(item["questions"])):
            messages = build_messages(item, q_idx, responses, args.answer_instruction)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            device = getattr(model, "device", None)
            if device is None:
                device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            start = time.time()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids = output_ids[:, input_len:]
            output = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            token_count = int((generated_ids != pad_token_id).sum().item()) if pad_token_id is not None else int(generated_ids.numel())
            responses.append(output)
            token_counts.append(token_count)
            hit_max_new_tokens.append(bool(token_count >= args.max_new_tokens))
            seconds_per_question.append(round(time.time() - start, 4))

        row = {
            "index": str(item["index"]),
            "category": item["category"],
            "image": item["image"],
            "questions": item["questions"],
            "answers": item["answers"],
            "responses": responses,
            "raw_output_token_counts": token_counts,
            "hit_max_new_tokens": hit_max_new_tokens,
            "seconds_per_question": seconds_per_question,
            "variant": variant,
            "method": config["method"],
            "use_deco_generation": use_deco_generation,
            "deco_alpha": args.deco_alpha if use_deco_generation else "",
            "deco_threshold_top_p": args.deco_threshold_top_p if use_deco_generation else "",
            "deco_threshold_top_k": args.deco_threshold_top_k if use_deco_generation else "",
            "deco_layers": args.deco_layers if use_deco_generation else "",
            "answer_instruction": args.answer_instruction,
        }
        rows.append(row)
        append_jsonl(predictions_path, row)

    return rows


def metric_summary_for_rows(args: argparse.Namespace, variant: str, rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    metrics = evaluate_reasoning(args, rows, out_dir)
    summary = {
        "variant": variant,
        "method": VARIANT_CONFIGS[variant]["method"],
        "model_path": args.model_path,
        "data_root": args.data_root,
        "visa_dr_json": args.visa_dr_json,
        "reasoning_rows": len(rows),
        "truncated_questions": sum(
            1
            for row in rows
            for hit in row.get("hit_max_new_tokens", [])
            if bool(hit)
        ),
        **metrics,
    }
    write_json(out_dir / "ours_summary.json", summary)
    write_csv(out_dir / "ours_summary.csv", [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-protocol VisA-D&R reasoning for Qwen3-VL and DeCo.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--visa-dr-json", default=DEFAULT_VISA_DR_JSON)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANT_CONFIGS), default=["qwen3_original", "qwen3_deco"])
    parser.add_argument("--skip-classes", nargs="*", default=DEFAULT_SKIP_CLASSES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--answer-instruction",
        default="",
        help="Optional uniform instruction appended to every benchmark question. Empty preserves the original questions exactly.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--ban-markdown-url",
        action="store_true",
        help="Globally block common Markdown-link and URL token sequences during generation.",
    )
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    parser.add_argument("--rouge-truncate-chars", type=int, default=512)
    parser.add_argument("--rouge-use-stemmer", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-cpu", action="store_true")
    args = parser.parse_args()

    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"{DEFAULT_OUTPUT_ROOT}/visa_dr_qwen3_deco_reasoning_original_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    master_dir = Path(args.out_dir)
    master_dir.mkdir(parents=True, exist_ok=True)
    write_json(master_dir / "run_config.json", vars(args))

    items = load_reasoning_items(args)
    print(
        f"[DATA] items={len(items)} data_root={args.data_root} visa_dr_json={args.visa_dr_json}",
        flush=True,
    )

    processor, model = load_qwen3(args)
    variant_rows: Dict[str, List[Dict[str, Any]]] = {}
    for variant in args.variants:
        variant_dir = master_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_json(variant_dir / "run_config.json", {**vars(args), "variant": variant, **VARIANT_CONFIGS[variant]})
        variant_rows[variant] = run_variant_generation(args, variant, items, processor, model, variant_dir)

    try:
        import gc
        import torch

        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    summaries: List[Dict[str, Any]] = []
    for variant in args.variants:
        variant_dir = master_dir / variant
        rows = variant_rows.get(variant) or read_jsonl(variant_dir / "reasoning_predictions.jsonl")
        summaries.append(metric_summary_for_rows(args, variant, rows, variant_dir))

    write_json(master_dir / "summary_comparison.json", summaries)
    write_csv(master_dir / "summary_comparison.csv", summaries)
    print("[DONE]", master_dir, flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
