#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Repair truncated VisA-D&R reasoning outputs by only increasing max_new_tokens.

The script reads an existing ``reasoning_predictions.jsonl`` and reruns only
images that contain ``hit_max_new_tokens=True``.  For a multi-turn sample, if
question k is truncated, generation is repeated from question k to the final
question because later turns depend on previous assistant answers.

No prompt, data, model, decoding strategy, or DeCo parameter is changed.  The
only generation parameter changed during repair is ``max_new_tokens``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from repro_visa_dr_reasoning import (
    evaluate_reasoning,
    parse_int_range,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_ADAR_ROOT = "/home/vina04/ADAR"
DEFAULT_DATA_ROOT = f"{DEFAULT_ADAR_ROOT}/evaluation/VisA-DR-official"


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_summary_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[0]) if rows else {}


def load_image(path: Path, max_side: int):
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


def first_truncated_index(row: Dict[str, Any]) -> int | None:
    for i, hit in enumerate(row.get("hit_max_new_tokens") or []):
        if bool(hit):
            return i
    return None


def count_truncated(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    total = 0
    hits = 0
    images = 0
    for row in rows:
        row_hits = [bool(x) for x in (row.get("hit_max_new_tokens") or [])]
        total += len(row_hits)
        hits += sum(1 for x in row_hits if x)
        images += int(any(row_hits))
    return total, hits, images


def build_messages(row: Dict[str, Any], question_index: int, previous_responses: Sequence[str]) -> List[Dict[str, Any]]:
    questions = list(row["questions"])
    messages: List[Dict[str, Any]] = []
    for i in range(question_index):
        if i == 0:
            content = [{"type": "image"}, {"type": "text", "text": str(questions[i])}]
        else:
            content = [{"type": "text", "text": str(questions[i])}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(previous_responses[i])}]})

    if question_index == 0:
        content = [{"type": "image"}, {"type": "text", "text": str(questions[question_index])}]
    else:
        content = [{"type": "text", "text": str(questions[question_index])}]
    messages.append({"role": "user", "content": content})
    return messages


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor

    if args.model_family == "qwen2":
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    elif args.model_family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"unsupported model family: {args.model_family}")

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
    model = ModelCls.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return processor, model


def make_generation_kwargs(
    args: argparse.Namespace,
    max_new_tokens: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if args.use_deco_generation:
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


def generate_one(
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    row: Dict[str, Any],
    image: Any,
    q_idx: int,
    max_new_tokens: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> Tuple[str, int, bool, float]:
    import torch

    messages = build_messages(row, q_idx, list(row["responses"]))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    kwargs = make_generation_kwargs(args, max_new_tokens, pad_token_id, eos_token_id)
    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    generated_ids = output_ids[:, input_len:]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    token_count = int((generated_ids != pad_token_id).sum().item()) if pad_token_id is not None else int(generated_ids.numel())
    return output, token_count, bool(token_count >= max_new_tokens), round(time.time() - start, 4)


def repair_rows(args: argparse.Namespace, rows: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    affected = [i for i, row in enumerate(rows) if first_truncated_index(row) is not None]
    print(f"[DATA] rows={len(rows)} affected_images={len(affected)}", flush=True)
    if not affected:
        return rows

    processor, model = load_model(args)
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    repair_log = out_dir / "repair_log.jsonl"
    if repair_log.exists() and not args.resume_repair_log:
        repair_log.unlink()

    for row_i, original_index in enumerate(affected, start=1):
        row = rows[original_index]
        start_idx = first_truncated_index(row)
        if start_idx is None:
            continue
        image_path = Path(args.data_root) / str(row["image"])
        image = load_image(image_path, args.max_image_side)
        print(
            f"[REPAIR] image={row_i}/{len(affected)} row_index={row.get('index')} "
            f"category={row.get('category')} start_q={start_idx + 1}",
            flush=True,
        )

        resolved = False
        for max_new_tokens in args.max_new_tokens_values:
            q_start = first_truncated_index(row)
            if q_start is None:
                resolved = True
                break
            print(f"[REPAIR] row_index={row.get('index')} max_new_tokens={max_new_tokens} from_q={q_start + 1}", flush=True)
            n_questions = min(len(row.get("questions") or []), len(row.get("responses") or []))
            for q_idx in range(q_start, n_questions):
                output, token_count, hit_max, seconds = generate_one(
                    args,
                    processor,
                    model,
                    row,
                    image,
                    q_idx,
                    int(max_new_tokens),
                    pad_token_id,
                    eos_token_id,
                )
                row["responses"][q_idx] = output
                row["raw_output_token_counts"][q_idx] = token_count
                row["hit_max_new_tokens"][q_idx] = hit_max
                row["seconds_per_question"][q_idx] = seconds
                append_jsonl(
                    repair_log,
                    {
                        "index": row.get("index"),
                        "category": row.get("category"),
                        "image": row.get("image"),
                        "question_index": q_idx,
                        "max_new_tokens": int(max_new_tokens),
                        "token_count": token_count,
                        "hit_max_new_tokens": hit_max,
                        "seconds": seconds,
                    },
                )
            if first_truncated_index(row) is None:
                resolved = True
                break

        row.setdefault("repair_meta", {})
        row["repair_meta"].update(
            {
                "repaired": True,
                "resolved": resolved,
                "max_new_tokens_values": list(args.max_new_tokens_values),
                "only_changed": "max_new_tokens",
            }
        )
        write_jsonl(out_dir / "reasoning_predictions.repair_checkpoint.jsonl", rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair truncated VisA-D&R reasoning outputs.")
    parser.add_argument("--model-family", choices=["qwen2", "qwen3"], required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--predictions-in", required=True)
    parser.add_argument("--source-summary-csv", default="")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--use-deco-generation", action="store_true")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument("--deco-layers", default="20:29")
    parser.add_argument("--max-new-tokens-values", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-repair-log", action="store_true")
    parser.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    parser.add_argument("--rouge-truncate-chars", type=int, default=512)
    parser.add_argument("--rouge-use-stemmer", action="store_true")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--sbert-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))
    shutil.copy2(args.predictions_in, out_dir / "reasoning_predictions.before_repair.jsonl")

    rows = read_jsonl(Path(args.predictions_in))
    before_total, before_hits, before_images = count_truncated(rows)
    repaired_rows = repair_rows(args, rows, out_dir)
    after_total, after_hits, after_images = count_truncated(repaired_rows)

    predictions_out = out_dir / "reasoning_predictions.jsonl"
    write_jsonl(predictions_out, repaired_rows)

    metrics = evaluate_reasoning(args, repaired_rows, out_dir)
    source_summary = read_summary_csv(Path(args.source_summary_csv)) if args.source_summary_csv else {}
    summary = {
        "method": args.method_name,
        "reasoning_rows": len(repaired_rows),
        "before_truncated_questions": before_hits,
        "before_images_with_truncation": before_images,
        "after_truncated_questions": after_hits,
        "after_images_with_truncation": after_images,
        "qa_pairs": after_total,
        "only_changed": "max_new_tokens",
        "max_new_tokens_values": " ".join(str(x) for x in args.max_new_tokens_values),
        "acc": source_summary.get("acc", ""),
        "precision": source_summary.get("precision", ""),
        "recall": source_summary.get("recall", ""),
        "f1": source_summary.get("f1", ""),
        **metrics,
    }
    write_json(out_dir / "repair_summary.json", summary)
    write_csv(out_dir / "repair_summary.csv", [summary])
    print("[DONE]", out_dir, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if after_hits:
        raise SystemExit(f"still truncated after repair: {after_hits} questions in {after_images} images")


if __name__ == "__main__":
    main()
