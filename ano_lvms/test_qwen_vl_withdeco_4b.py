import argparse
import inspect
import os
import sys

import torch


def move_to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def build_model(tfm, model_path, torch_dtype, device):
    classes = [
        getattr(tfm, "AutoModelForImageTextToText", None),
        getattr(tfm, "AutoModelForVision2Seq", None),
        getattr(tfm, "AutoModelForCausalLM", None),
    ]
    for cls in classes:
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
            return model
        except Exception:
            continue
    raise RuntimeError(f"Cannot load model from {model_path} with available AutoModel classes.")


def decode_new_tokens(processor, output, input_len):
    gen = output.sequences[:, input_len:]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for DeCo on transformers-4.57.6 Qwen3-VL.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    tfm_src = os.path.join(repo_root, "transformers-4.57.6", "src")
    sys.path.insert(0, tfm_src)
    import transformers as tfm

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    tfm.set_seed(42)
    processor = tfm.AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = build_model(tfm, args.model_path, torch_dtype=torch_dtype, device=device)
    model.eval()

    sig = inspect.signature(model.generate)
    if "use_deco" not in sig.parameters:
        raise RuntimeError("Patched generate() does not expose `use_deco`.")

    num_layers = model.config.text_config.num_hidden_layers
    early_exit_layers = list(range(max(1, num_layers - 8), num_layers))
    if not early_exit_layers:
        raise RuntimeError("Failed to build `early_exit_layers`.")

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Answer in one short sentence: what is 2+3?"}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[prompt], return_tensors="pt")
    model_inputs = move_to_device(model_inputs, device=device, dtype=torch_dtype)
    input_len = model_inputs["input_ids"].shape[1]

    common_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    baseline = model.generate(**model_inputs, do_sample=False, num_beams=1, **common_kwargs)
    baseline_text = decode_new_tokens(processor, baseline, input_len)

    deco_greedy = model.generate(
        **model_inputs,
        do_sample=False,
        num_beams=1,
        use_deco=True,
        alpha=0.6,
        threshold_top_p=0.9,
        threshold_top_k=20,
        early_exit_layers=early_exit_layers,
        **common_kwargs,
    )
    deco_greedy_text = decode_new_tokens(processor, deco_greedy, input_len)

    deco_sample = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=1,
        use_deco=True,
        alpha=0.6,
        threshold_top_p=0.9,
        threshold_top_k=20,
        early_exit_layers=early_exit_layers,
        **common_kwargs,
    )
    deco_sample_text = decode_new_tokens(processor, deco_sample, input_len)

    beam_error_ok = False
    try:
        _ = model.generate(
            **model_inputs,
            do_sample=False,
            num_beams=2,
            use_deco=True,
            alpha=0.6,
            threshold_top_p=0.9,
            threshold_top_k=20,
            early_exit_layers=early_exit_layers,
            **common_kwargs,
        )
    except NotImplementedError:
        beam_error_ok = True

    if not beam_error_ok:
        raise AssertionError("Expected beam+DeCo to raise NotImplementedError.")

    print("PASS: generate signature contains use_deco")
    print(f"PASS: baseline greedy -> {baseline_text}")
    print(f"PASS: DeCo greedy     -> {deco_greedy_text}")
    print(f"PASS: DeCo sample     -> {deco_sample_text}")
    print("PASS: beam+DeCo raises NotImplementedError as expected")


if __name__ == "__main__":
    main()
