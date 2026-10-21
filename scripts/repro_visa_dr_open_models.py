#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the open-weight VisA-D&R reasoning baselines from Anomaly-OV.

The benchmark protocol mirrors the existing Qwen2 reproduction in
``repro_visa_dr_qwen2vl_baseline.py``: use the 1,000 anomaly-only rows in
VisA-D&R after excluding pcb2/pcb4, preserve its sequential QA history, resize
the long image side to 1024, and decode greedily with max_new_tokens=4096.
The evaluation metrics are produced by ``repro_visa_dr_reasoning.py`` so the
SBERT checkpoint and ROUGE implementation stay identical to our Qwen2 runs.

This runner intentionally covers base open-weight models only. GPT-4V/4o have
no downloadable weights, and Anomaly-OV fine-tuned rows are not base models.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repro_visa_dr_qwen2vl_baseline import (  # noqa: E402
    append_jsonl,
    load_image_official,
    load_reasoning_items,
    read_jsonl,
)
from repro_visa_dr_reasoning import evaluate_reasoning  # noqa: E402


MODEL_SPECS = {
    "internvl2_8b": {
        "repo": "OpenGVLab/InternVL2-8B",
        "revision": "6fb9ad6924f69424e57fab2ab061d707688f0296",
        "display": "InternVL2-8B",
        "kind": "internvl",
    },
    "internvl2_26b": {
        "repo": "OpenGVLab/InternVL2-26B",
        "revision": "46f37972e05604cc44a59f72334df220fa197a7b",
        "display": "InternVL2-26B (8-bit)",
        "kind": "internvl",
    },
    "ixc_2_5_7b": {
        "repo": "internlm/internlm-xcomposer2d5-7b",
        "revision": "4aa81f2bbf20a9ddd4137dfe847c142adf07b652",
        "display": "IXC-2.5-7B",
        "kind": "ixc",
    },
    "llava_ov_0_5b": {
        "repo": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "revision": "74dd0bf867a4cda7950c17663794267c60cf4b40",
        "display": "LLaVA-OV-0.5B",
        "kind": "llava",
    },
    "llava_ov_7b": {
        "repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "revision": "0d50680527681998e456c7b78950205bedd8a068",
        "display": "LLaVA-OV-7B",
        "kind": "llava",
    },
}

PAPER_TABLE4 = {
    "Qwen2-VL-2B": (0.22, 0.55, 0.74),
    "Qwen2-VL-7B": (0.25, 0.61, 0.68),
    "InternVL2-8B": (0.23, 0.73, 0.80),
    "InternVL2-26B": (0.21, 0.74, 0.80),
    "IXC-2.5-7B": (0.21, 0.58, 0.77),
    "LLaVA-OV-0.5B": (0.20, 0.63, 0.81),
    "LLaVA-OV-7B": (0.24, 0.66, 0.79),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", default="/home/vina04/ADAR/evaluation/ODinW-13/data")
    p.add_argument("--visa-dr-json", default="/home/vina04/ADAR/evaluation/VisA-DR/visa_dr.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-image-side", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--limit-reasoning", type=int, default=0)
    p.add_argument("--limit-reasoning-per-category", type=int, default=0)
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sbert-batch-size", type=int, default=64)
    p.add_argument("--sbert-cpu", action="store_true")
    p.add_argument("--rouge-backend", choices=["auto", "rouge", "rouge-score", "internal_lcs"], default="auto")
    p.add_argument("--rouge-truncate-chars", type=int, default=512)
    p.add_argument("--rouge-use-stemmer", action="store_true")
    return p.parse_args()


def model_device(model: Any) -> torch.device:
    try:
        return model.device
    except Exception:
        for param in model.parameters():
            if param.device.type != "meta":
                return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_llava(model_path: str) -> Dict[str, Any]:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    return {"kind": "llava", "model": model, "processor": processor}


def load_internvl(model_path: str, load_in_8bit: bool) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": True,
    }
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        kwargs["max_memory"] = {0: "22GiB", "cpu": "26GiB"}
    model = AutoModel.from_pretrained(model_path, **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    return {"kind": "internvl", "model": model, "tokenizer": tokenizer}


def load_ixc(model_path: str) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model.tokenizer = tokenizer
    return {"kind": "ixc", "model": model, "tokenizer": tokenizer}


def load_backend(args: argparse.Namespace, kind: str) -> Dict[str, Any]:
    if kind == "llava":
        return load_llava(args.model_path)
    if kind == "internvl":
        return load_internvl(args.model_path, args.load_in_8bit)
    if kind == "ixc":
        return load_ixc(args.model_path)
    raise ValueError(f"unknown backend kind: {kind}")


def llava_answer(
    backend: Dict[str, Any], image: Image.Image, questions: Sequence[str],
    responses: Sequence[str], q_idx: int, max_new_tokens: int,
) -> Tuple[str, int, bool]:
    processor, model = backend["processor"], backend["model"]
    messages: List[Dict[str, Any]] = []
    for i in range(q_idx + 1):
        content: List[Dict[str, Any]] = []
        if i == 0:
            content.append({"type": "image"})
        content.append({"type": "text", "text": str(questions[i])})
        messages.append({"role": "user", "content": content})
        if i < q_idx:
            messages.append({"role": "assistant", "content": str(responses[i])})
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(model_device(model)) if hasattr(v, "to") else v for k, v in inputs.items()}
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = ids[:, inputs["input_ids"].shape[-1]:]
    count = int(generated.ne(pad_id).sum().item()) if pad_id is not None else int(generated.numel())
    text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, count, count >= max_new_tokens


def internvl_image_tensor(image: Image.Image, max_tiles: int = 12) -> torch.Tensor:
    """InternVL2 official dynamic 448px tiling (up to 12 tiles + thumbnail)."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_size = 448
    aspect = image.width / image.height
    ratios = sorted(
        {(i, j) for n in range(1, max_tiles + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= max_tiles},
        key=lambda x: (x[0] * x[1], x[0], x[1]),
    )
    best = (1, 1)
    best_diff = float("inf")
    area = image.width * image.height
    for ratio in ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_diff, best = diff, ratio
    cols, rows = best
    resized = image.resize((image_size * cols, image_size * rows), Image.Resampling.BICUBIC)
    tiles = []
    for k in range(cols * rows):
        left = (k % cols) * image_size
        top = (k // cols) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return torch.stack([transform(tile) for tile in tiles])


def internvl_answer(
    backend: Dict[str, Any], pixels: torch.Tensor, questions: Sequence[str],
    history: Any, q_idx: int, max_new_tokens: int,
) -> Tuple[str, Any, int, bool]:
    model, tokenizer = backend["model"], backend["tokenizer"]
    pixels = pixels.to(device=model_device(model), dtype=torch.bfloat16)
    query = str(questions[q_idx])
    if q_idx == 0:
        query = "<image>\n" + query
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    with torch.inference_mode():
        result = model.chat(
            tokenizer, pixels, query, generation_config,
            history=history, return_history=True,
        )
    text, history = result if isinstance(result, tuple) else (result, history)
    text = str(text).strip()
    count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, history, count, count >= max_new_tokens


def ixc_answer(
    backend: Dict[str, Any], image_path: str, question: str, first: bool,
    history: Any, max_new_tokens: int,
) -> Tuple[str, Any, int, bool]:
    model, tokenizer = backend["model"], backend["tokenizer"]
    query = f"Image <ImageHere>; {question}" if first else str(question)
    with torch.inference_mode():
        result = model.chat(
            tokenizer,
            query,
            [image_path],
            history=history,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_meta=True,
        )
    text, history = result if isinstance(result, tuple) else (result, history)
    text = str(text).strip()
    count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, history, count, count >= max_new_tokens


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    if args.model == "internvl2_26b" and not args.load_in_8bit:
        raise ValueError("InternVL2-26B is only run with the user-approved 8-bit quantization.")
    if args.model != "internvl2_26b" and args.load_in_8bit:
        raise ValueError("--load-in-8bit is reserved for InternVL2-26B in this experiment.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_args = SimpleNamespace(
        data_root=args.data_root,
        visa_dr_json=args.visa_dr_json,
        reasoning_skip_classes=["pcb2", "pcb4"],
        limit_reasoning=args.limit_reasoning,
        limit_reasoning_per_category=args.limit_reasoning_per_category,
    )
    items = load_reasoning_items(items_args)
    predictions_path = out_dir / "reasoning_predictions.jsonl"
    rows = read_jsonl(predictions_path) if args.resume and predictions_path.exists() else []
    done = {str(row.get("index")) for row in rows}
    todo = [item for item in items if str(item["index"]) not in done]

    config = vars(args).copy()
    config.update({
        "hf_repo": spec["repo"],
        "hf_revision": spec["revision"],
        "display_name": spec["display"],
        "quantization": "8-bit bitsandbytes" if args.load_in_8bit else "none",
        "dataset_count": len(items),
        "excluded_categories": ["pcb2", "pcb4"],
        "decoding": {"do_sample": False, "num_beams": 1},
        "prompt_protocol": "Anomaly-OV sequential QA; image in first user turn only",
    })
    write_json(out_dir / "run_config.json", config)
    print(f"[LOAD] {spec['display']} path={args.model_path}", flush=True)
    backend = load_backend(args, spec["kind"])
    print(f"[RUN] total={len(items)} completed={len(rows)} remaining={len(todo)}", flush=True)

    for n, item in enumerate(todo, start=1):
        image = load_image_official(item["abs_image"], args.max_image_side)
        responses: List[str] = []
        token_counts: List[int] = []
        hit_max: List[bool] = []
        question_seconds: List[float] = []
        if spec["kind"] == "internvl":
            pixels = internvl_image_tensor(image)
            history = None
            for q_idx, _ in enumerate(item["questions"]):
                start = time.time()
                text, history, count, hit = internvl_answer(
                    backend, pixels, item["questions"], history, q_idx, args.max_new_tokens
                )
                responses.append(text)
                token_counts.append(count)
                hit_max.append(hit)
                question_seconds.append(round(time.time() - start, 3))
        elif spec["kind"] == "llava":
            for q_idx, _ in enumerate(item["questions"]):
                start = time.time()
                text, count, hit = llava_answer(
                    backend, image, item["questions"], responses, q_idx, args.max_new_tokens
                )
                responses.append(text)
                token_counts.append(count)
                hit_max.append(hit)
                question_seconds.append(round(time.time() - start, 3))
        else:
            history = None
            with tempfile.TemporaryDirectory(prefix="visa_dr_ixc_image_") as temp_dir:
                resized_path = Path(temp_dir) / "image.jpg"
                image.save(resized_path, format="JPEG", quality=95)
                for q_idx, question in enumerate(item["questions"]):
                    start = time.time()
                    text, history, count, hit = ixc_answer(
                        backend, str(resized_path), question, q_idx == 0,
                        history, args.max_new_tokens,
                    )
                    responses.append(text)
                    token_counts.append(count)
                    hit_max.append(hit)
                    question_seconds.append(round(time.time() - start, 3))

        row = {
            "index": str(item["index"]),
            "category": item["category"],
            "image": item["image"],
            "questions": item["questions"],
            "answers": item["answers"],
            "responses": responses,
            "raw_output_token_counts": token_counts,
            "hit_max_new_tokens": hit_max,
            "seconds_per_question": question_seconds,
        }
        rows.append(row)
        append_jsonl(predictions_path, row)
        if n == 1 or n % args.log_every == 0 or n == len(todo):
            print(
                f"[RUN] {n}/{len(todo)} image={item['image']} "
                f"questions={len(responses)} truncations={sum(hit_max)}",
                flush=True,
            )

    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_args = SimpleNamespace(
        rouge_backend=args.rouge_backend,
        rouge_truncate_chars=args.rouge_truncate_chars,
        rouge_use_stemmer=args.rouge_use_stemmer,
        no_sbert=False,
        sbert_model=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        sbert_cpu=args.sbert_cpu,
    )
    metrics = evaluate_reasoning(metric_args, rows, out_dir)
    result = {
        "model": spec["display"],
        "hf_repo": spec["repo"],
        "hf_revision": spec["revision"],
        "quantization": config["quantization"],
        "reasoning_images": len(rows),
        "questions": sum(len(row.get("questions", [])) for row in rows),
        "truncated_answers": sum(sum(row.get("hit_max_new_tokens", [])) for row in rows),
        "max_new_tokens": args.max_new_tokens,
        "low_rouge_l": metrics.get("low_rouge_l"),
        "low_sbert": metrics.get("low_sbert"),
        "complex_sbert": metrics.get("complex_sbert"),
        "low_pairs": metrics.get("low_pairs"),
        "complex_pairs": metrics.get("complex_pairs"),
        "sbert_model": args.sbert_model,
        "rouge_backend_counts": metrics.get("rouge_backend_counts"),
    }
    write_json(out_dir / "open_model_summary.json", result)
    fields = list(result)
    with (out_dir / "open_model_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)
    print("[DONE] " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
