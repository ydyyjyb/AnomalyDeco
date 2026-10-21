import argparse
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser("Compare Qwen3-VL generation with/without DeCo")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    return parser.parse_args()


def find_first_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    max_len = min(a.numel(), b.numel())
    for idx in range(max_len):
        if int(a[idx]) != int(b[idx]):
            return idx
    return -1 if a.numel() == b.numel() else max_len


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    if args.image_path is not None:
        image = Image.open(Path(args.image_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_token_len = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "num_beams": args.num_beams,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    early_exit_layers = list(range(args.start_layer, args.end_layer))
    if len(early_exit_layers) == 0:
        raise ValueError("`start_layer` must be smaller than `end_layer`.")

    torch.manual_seed(args.seed)
    with torch.no_grad():
        output_ids_base = model.generate(**inputs, use_deco=False, **generation_kwargs)

    torch.manual_seed(args.seed)
    with torch.no_grad():
        output_ids_deco = model.generate(
            **inputs,
            use_deco=True,
            alpha=args.alpha,
            threshold_top_p=args.threshold_top_p,
            threshold_top_k=args.threshold_top_k,
            early_exit_layers=early_exit_layers,
            **generation_kwargs,
        )

    gen_ids_base = output_ids_base[:, input_token_len:]
    gen_ids_deco = output_ids_deco[:, input_token_len:]

    text_base = processor.batch_decode(gen_ids_base, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text_deco = processor.batch_decode(gen_ids_deco, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    first_diff = find_first_diff(gen_ids_base[0], gen_ids_deco[0])
    overlap_len = min(gen_ids_base.shape[-1], gen_ids_deco.shape[-1])
    token_overlap = (
        float((gen_ids_base[0, :overlap_len] == gen_ids_deco[0, :overlap_len]).sum().item()) / float(overlap_len)
        if overlap_len > 0
        else 1.0
    )

    print("=== Config ===")
    print(f"model: {args.model_name_or_path}")
    print(f"device: {model.device}")
    print(f"use_deco: False vs True")
    print(f"early_exit_layers: {early_exit_layers}")
    print(f"alpha={args.alpha}, threshold_top_p={args.threshold_top_p}, threshold_top_k={args.threshold_top_k}")
    print()

    print("=== Baseline (no DeCo) ===")
    print(text_base)
    print()

    print("=== DeCo ===")
    print(text_deco)
    print()

    print("=== Difference Summary ===")
    print(f"baseline_tokens: {gen_ids_base.shape[-1]}")
    print(f"deco_tokens: {gen_ids_deco.shape[-1]}")
    print(f"first_different_token_index: {first_diff}")
    print(f"token_overlap_ratio: {token_overlap:.4f}")


if __name__ == "__main__":
    main()
