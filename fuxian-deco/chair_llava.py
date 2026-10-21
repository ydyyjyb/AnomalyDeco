import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image_ids(args):
    if args.image_ids_file:
        ids = []
        with open(args.image_ids_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids.append(int(obj.get("image_id", obj.get("id"))))
        return ids

    if args.seed_file:
        ids = []
        with open(args.seed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "image_id" in obj:
                    ids.append(int(obj["image_id"]))
        if ids:
            return ids

    paths = sorted(Path(args.image_folder).glob("COCO_val2014_*.jpg"))
    ids = [int(p.stem.split("_")[-1]) for p in paths]
    return ids


def chunk_list(items, num_chunks, chunk_idx):
    if num_chunks <= 1:
        return items
    return items[chunk_idx::num_chunks]


def build_prompt(model, conv_mode, question):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def generated_token_ids(output_ids, input_ids):
    input_token_len = input_ids.shape[1]
    has_prompt_prefix = (
        output_ids.shape[1] >= input_token_len
        and torch.equal(output_ids[:, :input_token_len], input_ids)
    )
    return output_ids[:, input_token_len:] if has_prompt_prefix else output_ids


def generate_one(args, tokenizer, model, image_processor, image_id, debug=False):
    image_path = Path(args.image_folder) / f"COCO_val2014_{image_id:012d}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing COCO image: {image_path}")

    prompt, conv = build_prompt(model, args.conv_mode, args.question)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    generation_kwargs = {
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "stopping_criteria": [stopping_criteria],
        "use_deco": args.use_deco,
        "alpha": args.alpha,
        "threshold_top_p": args.threshold_top_p,
        "threshold_top_k": args.threshold_top_k,
        "early_exit_layers": list(range(args.start_layer, args.end_layer)),
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.inference_mode():
        output_dict = model.generate(input_ids, images=image_tensor, **generation_kwargs)

    output_ids = output_dict.sequences
    generated_ids = generated_token_ids(output_ids, input_ids)
    if debug:
        print(f"[DEBUG] input_ids shape: {tuple(input_ids.shape)}")
        print(f"[DEBUG] output_ids shape: {tuple(output_ids.shape)}")
        print(f"[DEBUG] generated_ids shape: {tuple(generated_ids.shape)}")
        print(f"[DEBUG] input prefix matches output: {generated_ids.shape[1] != output_ids.shape[1]}")
        print(f"[DEBUG] generated token id range: [{generated_ids.min().item()}, {generated_ids.max().item()}]")
        print(f"[DEBUG] generated decode: {tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]!r}")
    output_ids = generated_ids
    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    caption = caption.strip()
    if caption.endswith(stop_str):
        caption = caption[: -len(stop_str)].strip()

    return {
        "image_id": int(image_id),
        "caption": caption,
        "metadata": {
            "image": str(image_path),
            "use_deco": bool(args.use_deco),
            "alpha": args.alpha,
            "threshold_top_p": args.threshold_top_p,
            "threshold_top_k": args.threshold_top_k,
            "early_exit_layers": [args.start_layer, args.end_layer],
        },
    }


def eval_model(args):
    disable_torch_init()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Fix CUDA/NVML before running LLaVA-7B.")

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda",
    )
    model.eval()

    image_ids = load_image_ids(args)
    image_ids = chunk_list(image_ids, args.num_chunks, args.chunk_idx)
    if args.max_samples > 0:
        image_ids = image_ids[: args.max_samples]
    if not image_ids:
        raise RuntimeError("No COCO image ids found.")

    answers_file = Path(args.answers_file)
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    with answers_file.open("w", encoding="utf-8") as ans_file:
        for sample_idx, image_id in enumerate(tqdm(image_ids, desc="Generating CHAIR captions", ncols=120)):
            record = generate_one(args, tokenizer, model, image_processor, image_id, debug=args.debug_first and sample_idx == 0)
            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_file.flush()

    print(f"[OK] Saved captions: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True, help="COCO 2014 val2014 folder.")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--seed-file", type=str, default="./opera_log/llava-1.5/ours.jsonl")
    parser.add_argument("--image-ids-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--question", type=str, default="Please describe this image in detail.")
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-first", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
