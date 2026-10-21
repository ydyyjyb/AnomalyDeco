#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMAD-style QA evaluation for ADAR/Qwen/JUDO-style industrial anomaly reasoning.

This reproduces the metric family used by JUDO:

  - Anomaly Discrimination: balanced accuracy over normal/abnormal samples for
    the MMAD "Anomaly Detection" question type.
  - Defect Classification, Defect Localization, Defect Description,
    Defect Analysis, Object Classification, Object Analysis: multiple-choice
    accuracy.
  - Average: mean over available task accuracies.

The script evaluates multiple-choice QA with a query image and one normal
template image, using the JUDO output format:

  <seg>...</seg>
  <think>...</think>
  <answer>A|B|C|D</answer>

It is intentionally separate from the ADAR main anomaly-score pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JUDO_COLUMNS = [
    "Anomaly Discrimination",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
    "Object Classification",
    "Object Analysis",
]

TYPE_ALIASES = {
    "Anomaly Detection": "Anomaly Discrimination",
    "Anomaly Discrimination": "Anomaly Discrimination",
    "Defect Classification": "Defect Classification",
    "Defect Type": "Defect Classification",
    "Defect Localization": "Defect Localization",
    "Defect Location": "Defect Localization",
    "Defect Description": "Defect Description",
    "Defect Appearance": "Defect Description",
    "Defect Analysis": "Defect Analysis",
    "Object Classification": "Object Classification",
    "Object Analysis": "Object Analysis",
    "Object Structure": "Object Analysis",
    "Object Details": "Object Analysis",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl_report(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def parse_int_range(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        return []
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def canonical_question_type(question_type: str) -> str:
    text = str(question_type or "unknown").strip()
    return TYPE_ALIASES.get(text, text)


def make_system_prompt(grid_size: int) -> str:
    return (
        "You are a helpful assistant for industrial quality control tasks. The user asks a question, and the Assistant solves it.\n"
        f"The query image is divided into a grid of {grid_size}×{grid_size} patches. "
        "Patch coordinates are (row, col) and both row and col start from 1 (top-left patch is (1,1)).\n\n"
        "Always respond in this exact order and format:\n"
        "1) <seg> The anomalous region patch list (use (r,c) and (r,c_start)-(r,c_end) for contiguous columns). If there is no anomaly, respond 'None'. </seg>\n"
        "2) <think> Provide your detailed reasoning so that it leads to the correct answer. </think>\n"
        "3) <answer>A|B|C|D</answer>\n"
    )


def make_answer_only_system_prompt() -> str:
    return (
        "You are answering an industrial quality-control multiple-choice question. "
        "You must choose exactly one option from A, B, C, or D. "
        "Do not answer None, N/A, uncertain, or with free-form text. "
        "Return exactly one tag in this format: <answer>A</answer>"
    )


def resolve_path(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else root / path


@lru_cache(maxsize=None)
def casefold_file_index(image_dir_text: str) -> Dict[str, Path]:
    image_dir = Path(image_dir_text)
    if not image_dir.is_dir():
        return {}
    return {p.name.casefold(): p for p in image_dir.iterdir() if p.is_file()}


def resolve_eval_image(args: argparse.Namespace, rel_or_abs: str) -> Path:
    """Resolve MMAD paths, reusing the already-installed VisA and MVTec data."""
    direct = resolve_path(Path(args.mmad_root), rel_or_abs)
    if direct.exists():
        return direct
    path = Path(str(rel_or_abs).replace("\\\\", "/"))
    parts = path.parts
    data_root = Path(args.data_root)
    candidates: List[Path] = []
    if len(parts) >= 5 and parts[0] == "MVTec-AD":
        candidates.append(data_root / "MVTec" / Path(*parts[1:]))
    elif len(parts) >= 5 and parts[0] == "VisA":
        category, split, label, filename = parts[1:5]
        folder = "Normal" if label.lower() in {"good", "normal"} else "Anomaly"
        image_dir = data_root / "VisA" / category / "Data" / "Images" / folder
        candidates.append(image_dir / filename)
        fallback = casefold_file_index(str(image_dir)).get(Path(filename).name.casefold())
        if fallback is not None:
            candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct


def resolve_grpo_path(args: argparse.Namespace, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    text = str(rel_or_abs).replace("\\", "/")
    parts = text.split("/")
    adar_root = Path(args.adar_root)
    data_root = Path(args.data_root)

    candidates: List[Path] = [
        Path(args.mmad_root) / text,
        adar_root / "JUDO" / text,
    ]
    if len(parts) >= 5 and parts[0] == "DS-MVTec":
        category = parts[1]
        kind = parts[2]
        defect = parts[3]
        filename = parts[4]
        if kind == "image":
            candidates.insert(0, data_root / "MVTec" / category / "test" / defect / filename)
    elif len(parts) >= 5 and parts[0] == "MVTec-AD":
        category = parts[1]
        split = parts[2]
        defect = parts[3]
        filename = parts[4]
        candidates.insert(0, data_root / "MVTec" / category / split / defect / filename)
    elif len(parts) >= 5 and parts[0] == "VisA":
        category = parts[1]
        label = parts[3].lower()
        filename = parts[4]
        folder = "Anomaly" if label in {"bad", "anomaly", "ng"} else "Normal"
        candidates.insert(0, data_root / "VisA" / category / "Data" / "Images" / folder / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def maybe_download_mmad(args: argparse.Namespace) -> None:
    root = Path(args.mmad_root)
    json_path = Path(args.mmad_json)
    if json_path.exists():
        return
    if not args.download_mmad:
        raise FileNotFoundError(f"MMAD json not found: {json_path}. Use --download-mmad or set --mmad-json.")
    root.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={args.mmad_repo!r}, repo_type='dataset', local_dir={str(root)!r}, local_dir_use_symlinks=False)"
    )
    subprocess.check_call([sys.executable, "-c", code])
    if not json_path.exists():
        candidates = sorted(root.rglob("mmad.json"))
        if candidates:
            args.mmad_json = str(candidates[0])
        else:
            raise FileNotFoundError(f"Downloaded MMAD but did not find mmad.json under {root}")


def load_mmad(args: argparse.Namespace) -> Dict[str, Any]:
    if args.grpo_jsonl:
        return {}
    maybe_download_mmad(args)
    with Path(args.mmad_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("MMAD json must be a dict: image_path -> metadata")
    return data


def load_grpo_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = Path(args.grpo_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"GRPO jsonl not found: {path}")
    items: List[Dict[str, Any]] = []
    allowed_prefixes = set(args.dataset_prefixes or [])
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image = str(obj.get("image", ""))
            prefix = image.split("/")[0] if image else ""
            if allowed_prefixes and prefix not in allowed_prefixes:
                continue
            qtype = canonical_question_type(str(obj.get("type", "unknown")))
            if args.question_types and qtype not in args.question_types:
                continue
            query_path = resolve_grpo_path(args, image)
            template = str(obj.get("template_image", ""))
            template_path = resolve_grpo_path(args, template)
            if args.skip_missing_images and (not query_path.exists() or not template_path.exists()):
                continue
            items.append(
                {
                    "dataset": prefix,
                    "image": image,
                    "query_image_path": str(query_path),
                    "template": template,
                    "template_image_path": str(template_path),
                    "qid": int(obj.get("id", len(items))),
                    "question": str(obj.get("problem", "")).strip(),
                    "options_text": "",
                    "options": {},
                    "correct_answer": str(obj.get("solution", "")).strip(),
                    "question_type_raw": str(obj.get("type", "unknown")),
                    "question_type": qtype,
                }
            )
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    return items


def parse_conversations(meta: Dict[str, Any], seed_key: str, shuffle_options: bool) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    conversation_keys = sorted(k for k in meta if str(k).startswith("conversation"))
    if not conversation_keys:
        return tasks
    for conv_key in conversation_keys[:1]:
        conv = meta.get(conv_key) or []
        for idx, qa in enumerate(conv):
            options = qa.get("Options") or qa.get("options") or {}
            if not isinstance(options, dict) or not options:
                continue
            items = list(options.items())
            if shuffle_options:
                rng = random.Random(stable_int(f"{seed_key}|{idx}"))
                rng.shuffle(items)
            correct_label = ""
            option_lines: List[str] = []
            option_map: Dict[str, str] = {}
            for i, (orig_label, option_text) in enumerate(items):
                new_label = chr(65 + i)
                option_lines.append(f"{new_label}. {option_text}")
                option_map[new_label] = str(option_text)
                if str(orig_label) == str(qa.get("Answer")):
                    correct_label = new_label
            if not correct_label:
                continue
            tasks.append(
                {
                    "qid": idx,
                    "question": str(qa.get("Question") or qa.get("question") or "").strip(),
                    "options_text": "\n".join(option_lines),
                    "options": option_map,
                    "correct_answer": correct_label,
                    "question_type_raw": str(qa.get("type", "unknown")),
                    "question_type": canonical_question_type(str(qa.get("type", "unknown"))),
                }
            )
    return tasks


def build_eval_items(args: argparse.Namespace, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if args.grpo_jsonl:
        return load_grpo_items(args)
    root = Path(args.mmad_root)
    items: List[Dict[str, Any]] = []
    for image_path, meta in sorted(data.items()):
        templates = list(meta.get("random_templates") or [])
        if not templates:
            continue
        template_idx = stable_int(str(image_path)) % len(templates)
        template_path = templates[template_idx]
        tasks = parse_conversations(meta, str(image_path), args.shuffle_options)
        for task in tasks:
            item = {
                "dataset": str(image_path).split("/")[0],
                "image": str(image_path),
                "query_image_path": str(resolve_eval_image(args, str(image_path))),
                "template": str(template_path),
                "template_image_path": str(resolve_eval_image(args, str(template_path))),
                **task,
            }
            if args.question_types and item["question_type"] not in args.question_types:
                continue
            items.append(item)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    return items


def load_done(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    rows = read_jsonl(path)
    keys = {(r.get("image"), int(r.get("qid", -1)), r.get("question")) for r in rows}
    return rows, keys


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:answer|答案)\s*[:：]\s*([A-D])\b", text, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1).upper() if match else "X"


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def generation_audit(ids: Any, eos_token_id: Any, pad_token_id: Any, max_new_tokens: int) -> Tuple[int, bool]:
    token_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    stop_ids = set()
    for value in (eos_token_id, pad_token_id):
        if isinstance(value, (list, tuple, set)):
            stop_ids.update(int(x) for x in value if x is not None)
        elif value is not None:
            stop_ids.add(int(value))
    for index, token_id in enumerate(token_ids):
        if int(token_id) in stop_ids:
            return index + 1, False
    return len(token_ids), len(token_ids) >= max_new_tokens


def batch_iter(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def load_image(path: str, size: int) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def load_model(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    family = args.model_family
    if family == "auto":
        cfg = Path(args.model_path) / "config.json"
        text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        if "qwen2_5_vl" in text.lower() or "qwen2.5" in text.lower():
            family = "qwen2_5"
        else:
            family = "qwen3"
    if family == "qwen2_5":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = ModelCls.from_pretrained(args.model_path, **model_kwargs).eval()
    return processor, tokenizer, model


def run_inference(args: argparse.Namespace, items: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    import torch

    results_path = out_dir / "mmad_qa_results.jsonl"
    done_rows, done_keys = load_done(results_path) if args.resume else ([], set())
    todo = [item for item in items if (item["image"], item["qid"], item["question"]) not in done_keys]
    print(f"[INFO] total={len(items)} done={len(done_rows)} todo={len(todo)}", flush=True)
    if not todo:
        return done_rows

    processor, tokenizer, model = load_model(args)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    system_prompt = make_answer_only_system_prompt() if args.answer_only else make_system_prompt(args.grid_size)

    rows = list(done_rows)
    for chunk in batch_iter(todo, args.batch_size):
        texts: List[str] = []
        images: List[Any] = []
        valid_chunk: List[Dict[str, Any]] = []
        for item in chunk:
            try:
                query_image = load_image(item["query_image_path"], args.image_size)
                template_image = load_image(item["template_image_path"], args.image_size)
            except Exception as exc:
                row = dict(item)
                row.update(
                    {
                        "raw_output": "",
                        "gpt_answer": "X",
                        "segmentation": "",
                        "thinking": "",
                        "correct": False,
                        "error": f"image_load_error: {exc}",
                    }
                )
                append_jsonl(results_path, [row])
                rows.append(row)
                continue
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "First image = QUERY; Second image = NORMAL template.\n"
                                f"Question: {item['question']}\n"
                                f"{item['options_text']}\n"
                                "Choose the single best option. "
                                "Your final output must contain one answer letter from the provided choices."
                            ),
                        },
                    ],
                },
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.extend([query_image, template_image])
            valid_chunk.append(item)

        if not valid_chunk:
            continue
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if args.use_deco_generation:
            generation_kwargs.update(
                {
                    "use_deco": True,
                    "alpha": args.deco_alpha,
                    "threshold_top_p": args.deco_threshold_top_p,
                    "threshold_top_k": args.deco_threshold_top_k,
                    "early_exit_layers": parse_int_range(args.deco_layers),
                }
            )

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        elapsed = (time.time() - t0) / max(1, len(valid_chunk))
        gen_ids = output_ids[:, input_len:]
        outputs = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        new_rows: List[Dict[str, Any]] = []
        for item, raw, generated_ids in zip(valid_chunk, outputs, gen_ids):
            raw = raw.strip()
            pred = extract_answer(raw)
            generated_tokens, output_truncated = generation_audit(
                generated_ids, eos_token_id, pad_token_id, args.max_new_tokens
            )
            row = dict(item)
            row.update(
                {
                    "raw_output": raw,
                    "gpt_answer": pred,
                    "segmentation": extract_tag(raw, "seg"),
                    "thinking": extract_tag(raw, "think"),
                    "correct": pred == item["correct_answer"],
                    "seconds_per_case": round(elapsed, 4),
                    "generated_tokens": generated_tokens,
                    "output_truncated": output_truncated,
                    "answer_tag_present": bool(extract_tag(raw, "answer")),
                    "error": "",
                    "use_deco_generation": bool(args.use_deco_generation),
                    "deco_alpha": args.deco_alpha if args.use_deco_generation else "",
                    "deco_threshold_top_p": args.deco_threshold_top_p if args.use_deco_generation else "",
                    "deco_threshold_top_k": args.deco_threshold_top_k if args.use_deco_generation else "",
                    "deco_layers": args.deco_layers if args.use_deco_generation else "",
                }
            )
            new_rows.append(row)
        append_jsonl(results_path, new_rows)
        rows.extend(new_rows)
        summarize_and_write(rows, out_dir, args.normal_flag)
    return rows


def safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def fmt_pct(value: Optional[float]) -> str:
    return "" if value is None else f"{100.0 * value:.2f}"


def summarize_rows(
    rows: Sequence[Dict[str, Any]], normal_flag: str, *, drop_invalid_predictions: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("correct_answer") not in {"A", "B", "C", "D", "E"}:
            continue
        # Preserve the top-level grouping used by the released JUDO summary.py.
        dataset = str(row.get("dataset", ""))
        by_dataset[dataset].append(row)

    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    valid_letters = {"A", "B", "C", "D"}  # official JUDO parser accepts A-D
    for dataset, all_group in sorted(by_dataset.items()):
        group = (
            [r for r in all_group if r.get("gpt_answer") in valid_letters]
            if drop_invalid_predictions
            else list(all_group)
        )
        out: Dict[str, Any] = {
            "dataset": dataset,
            "metric_protocol": "official_valid_only" if drop_invalid_predictions else "strict_invalid_as_wrong",
        }
        numeric_values: List[float] = []
        out["parse_valid_rate"] = fmt_pct(safe_div(
            sum(1 for r in all_group if r.get("gpt_answer") in valid_letters), len(all_group)
        ))
        for col in JUDO_COLUMNS:
            all_col_rows = [r for r in all_group if r.get("question_type") == col]
            col_rows = [r for r in group if r.get("question_type") == col]
            if col == "Anomaly Discrimination":
                normal_rows = [r for r in col_rows if normal_flag in str(r.get("image", "")).lower()]
                abnormal_rows = [r for r in col_rows if normal_flag not in str(r.get("image", "")).lower()]
                normal_acc = safe_div(sum(1 for r in normal_rows if r.get("correct")), len(normal_rows))
                abnormal_acc = safe_div(sum(1 for r in abnormal_rows if r.get("correct")), len(abnormal_rows))
                # Released JUDO summary sets the missing class accuracy to zero.
                normal_acc = 0.0 if normal_acc is None else normal_acc
                abnormal_acc = 0.0 if abnormal_acc is None else abnormal_acc
                acc = (normal_acc + abnormal_acc) / 2.0
                out["Overkill"] = fmt_pct(1.0 - normal_acc)
                out["Miss"] = fmt_pct(1.0 - abnormal_acc)
            else:
                acc = safe_div(sum(1 for r in col_rows if r.get("correct")), len(col_rows))
                acc = 0.0 if acc is None else acc
            out[col] = fmt_pct(acc)
            numeric_values.append(acc)
            detail.append(
                {
                    "dataset": dataset,
                    "question_type": col,
                    "total": len(all_col_rows),
                    "metric_denominator": len(col_rows),
                    "correct": sum(1 for r in col_rows if r.get("correct")),
                    "parse_valid": sum(1 for r in all_col_rows if r.get("gpt_answer") in valid_letters),
                    "accuracy": fmt_pct(acc),
                }
            )
        out["Average"] = fmt_pct(sum(numeric_values) / len(numeric_values) if numeric_values else None)
        out["total"] = len(all_group)
        out["valid"] = len(group)
        summary.append(out)

    if summary:
        avg: Dict[str, Any] = {
            "dataset": "Average",
            "metric_protocol": "official_valid_only" if drop_invalid_predictions else "strict_invalid_as_wrong",
        }
        for col in JUDO_COLUMNS + ["Average", "Overkill", "Miss", "parse_valid_rate"]:
            vals = [float(r[col]) for r in summary if r.get(col) not in {"", None}]
            avg[col] = f"{sum(vals) / len(vals):.2f}" if vals else ""
        avg["total"] = sum(int(r.get("total", 0)) for r in summary)
        avg["valid"] = sum(int(r.get("valid", 0)) for r in summary)
        summary.append(avg)
    return summary, detail


def summarize_and_write(rows: Sequence[Dict[str, Any]], out_dir: Path, normal_flag: str) -> None:
    official, official_detail = summarize_rows(rows, normal_flag, drop_invalid_predictions=True)
    strict, strict_detail = summarize_rows(rows, normal_flag, drop_invalid_predictions=False)
    write_csv(out_dir / "mmad_qa_judo_metrics.csv", official)
    write_json(out_dir / "mmad_qa_judo_metrics.json", official)
    write_csv(out_dir / "mmad_qa_by_task.csv", official_detail)
    write_csv(out_dir / "mmad_qa_judo_metrics_strict.csv", strict)
    write_csv(out_dir / "mmad_qa_by_task_strict.csv", strict_detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMAD-style QA metrics used by JUDO.")
    parser.add_argument("--adar-root", default=os.environ.get("ADAR_ROOT", str(Path.cwd())))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-family", default="auto", choices=["auto", "qwen3", "qwen2_5"])
    parser.add_argument("--data-root", default="")
    parser.add_argument("--mmad-root", default="")
    parser.add_argument("--mmad-json", default="")
    parser.add_argument("--grpo-jsonl", default="", help="Optional JUDO grpo_gt.jsonl-style QA file.")
    parser.add_argument("--mmad-repo", default="jiang-cc/MMAD")
    parser.add_argument("--download-mmad", action="store_true")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--answer-only", action="store_true", help="Use a strict answer-only prompt for rerunning invalid cases.")
    parser.add_argument("--use-deco-generation", action="store_true", help="Run model.generate with use_deco=True for JUDO-style QA generation.")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument("--deco-layers", default="20:29", help="Early-exit layers for DeCo generation; e.g. 20:29 means layers 20..28.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--shuffle-options", action="store_true", default=True)
    parser.add_argument("--no-shuffle-options", dest="shuffle_options", action="store_false")
    parser.add_argument("--normal-flag", default="good")
    parser.add_argument("--question-types", nargs="*", default=[])
    parser.add_argument("--dataset-prefixes", nargs="*", default=[], help="Optional MMAD/JUDO prefixes, e.g. DS-MVTec VisA.")
    parser.add_argument("--skip-missing-images", action="store_true", default=True)
    parser.add_argument("--no-skip-missing-images", dest="skip_missing_images", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Validate every image referenced by the selected dataset without loading the model.")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    adar_root = Path(args.adar_root)
    if not args.model_path:
        args.model_path = str(adar_root / "Deco-main" / "qwen3_4b_vl_instruct")
    if not args.data_root:
        args.data_root = str(adar_root / "evaluation" / "ODinW-13" / "data")
    if not args.mmad_root:
        args.mmad_root = str(adar_root / "evaluation" / "MMAD")
    if not args.mmad_json:
        args.mmad_json = str(Path(args.mmad_root) / "mmad.json")
    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(adar_root / "Deco-main" / "runs" / f"mmad_qa_eval_{stamp}")
    if args.question_types:
        args.question_types = [canonical_question_type(q) for q in args.question_types]
    args.normal_flag = str(args.normal_flag).lower()
    return args


def main() -> None:
    args = normalize_args(parse_args())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    metric_config = {
        "metric_grouping": "preserve raw top-level image path dataset names, matching released JUDO summary.py",
        "metric_handling": "primary CSV reproduces released valid-answer denominator; strict CSV counts invalid outputs as wrong",
        "deco_is_finetuning": False,
    }
    if args.collect_only and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(metric_config)
    else:
        config = dict(vars(args))
        config.update(
            {
                "protocol": "JUDO-style MMAD 1-shot; query image then one normal template; 16x16 seg, think, answer tags",
                "template_selection": "stable md5(image_path) index into random_templates",
                "option_order": "stable md5(image_path|question_id) shuffle; shared across model arms",
                **metric_config,
            }
        )
    write_json(config_path, config)

    data = load_mmad(args)
    items = build_eval_items(args, data)
    if not args.collect_only:
        missing = sorted(
            {
                str(path)
                for item in items
                for path in (Path(item["query_image_path"]), Path(item["template_image_path"]))
                if not path.is_file()
            }
        )
        if missing:
            write_json(out_dir / "missing_images.json", missing)
            raise FileNotFoundError(
                f"Full evaluation preflight failed: {len(missing)} referenced images are missing. "
                f"See {out_dir / 'missing_images.json'}"
            )
        if args.preflight_only:
            dataset_counts: Dict[str, int] = defaultdict(int)
            for item in items:
                dataset_counts[str(item["dataset"])] += 1
            write_json(
                out_dir / "preflight_report.json",
                {
                    "status": "ok",
                    "qa_items": len(items),
                    "unique_query_images": len({item["query_image_path"] for item in items}),
                    "unique_template_images": len({item["template_image_path"] for item in items}),
                    "missing_image_references": 0,
                    "qa_items_by_dataset": dict(sorted(dataset_counts.items())),
                },
            )
            print(f"[PREFLIGHT OK] qa_items={len(items)} out_dir={out_dir}", flush=True)
            return
    write_json(out_dir / "mmad_qa_items_preview.json", items[:100])
    if args.collect_only:
        rows = read_jsonl(out_dir / "mmad_qa_results.jsonl")
    else:
        rows = run_inference(args, items, out_dir)
    summarize_and_write(rows, out_dir, args.normal_flag)
    valid_rows = [r for r in rows if r.get("correct_answer") in {"A", "B", "C", "D", "E"}]
    write_jsonl_report(
        out_dir / "invalid_answer_cases.jsonl",
        [r for r in valid_rows if r.get("gpt_answer") not in {"A", "B", "C", "D", "E"}],
    )
    write_jsonl_report(
        out_dir / "truncated_cases.jsonl",
        [r for r in valid_rows if r.get("output_truncated")],
    )
    write_jsonl_report(
        out_dir / "inference_error_cases.jsonl",
        [r for r in valid_rows if r.get("error")],
    )
    write_json(
        out_dir / "generation_audit.json",
        {
            "rows": len(valid_rows),
            "invalid_answer_parse": sum(r.get("gpt_answer") not in {"A", "B", "C", "D", "E"} for r in valid_rows),
            "truncated_outputs": sum(bool(r.get("output_truncated")) for r in valid_rows),
            "missing_answer_tags": sum(not bool(r.get("answer_tag_present")) for r in valid_rows),
            "image_or_inference_errors": sum(bool(r.get("error")) for r in valid_rows),
        },
    )
    print(f"[OK] output_dir={out_dir}", flush=True)
    print(f"[OK] results={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
