"""Cross-architecture adapter for the existing training-free main method.

This wrapper reuses the established VisA scoring/evaluation implementation in
0724_Mechanism_validation.py while loading Qwen2-VL, Qwen3-VL, or
LLaVA-OneVision. It intentionally does not modify the original experiment code.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


HERE = Path(__file__).resolve().parent
MECHANISM_SCRIPT = HERE / "0724_Mechanism_validation.py"
REFERENCE_HIDDEN_STATES = 37  # Qwen3-VL-4B: embedding output + 36 transformer blocks.
REFERENCE_TREND_STOP = 34
REFERENCE_DECO_START = 20
REFERENCE_DECO_STOP = 29


class ProcessorAdapter:
    """Drop Qwen-only pixel-bound kwargs for the LLaVA-OneVision processor."""

    def __init__(self, processor: Any, family: str):
        self._processor = processor
        self._family = family

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._family == "llava_onevision":
            kwargs.pop("min_pixels", None)
            kwargs.pop("max_pixels", None)
        return self._processor(*args, **kwargs)


def _load_mechanism_module() -> Any:
    spec = importlib.util.spec_from_file_location("mechanism_0724", MECHANISM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scoring implementation: {MECHANISM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_family(model_path: str) -> str:
    family = os.environ.get("ADAR_MODEL_FAMILY", "").strip().lower()
    if family in {"qwen2_vl", "qwen3_vl", "llava_onevision"}:
        return family
    raise RuntimeError(
        "Set ADAR_MODEL_FAMILY to one of: qwen2_vl, qwen3_vl, llava_onevision; "
        f"model path was {model_path!r}."
    )


def _text_config(config: Any) -> Any:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return config
    if isinstance(text_config, dict):
        return text_config
    return text_config


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def load_compatible_model(args: Any) -> tuple[Any, Any]:
    family = _model_family(args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    processor_adapter = ProcessorAdapter(processor, family)

    model_classes = {
        "qwen2_vl": Qwen2VLForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "llava_onevision": LlavaOnevisionForConditionalGeneration,
    }
    model_class = model_classes[family]
    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.local_files_only,
    }
    try:
        model = model_class.from_pretrained(args.model_path, **load_kwargs)
    except Exception as exc:
        if not args.allow_attn_fallback or args.attn_implementation == "sdpa":
            raise
        print(f"[WARN] {family}: failed to load with {args.attn_implementation}: {exc}")
        print("[WARN] Falling back to sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = model_class.from_pretrained(args.model_path, **load_kwargs)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = False

    text_config = _text_config(model.config)
    num_layers = _config_value(text_config, "num_hidden_layers")
    if num_layers is None:
        raise RuntimeError(f"Could not read text transformer depth for {family}.")
    num_layers = int(num_layers)

    # Keep the Qwen3 reference setting unchanged. For other backbones, map the
    # same layer fractions to their own depth; this is predeclared, not tuned on
    # the evaluation set.
    args.model_family = family
    args.model_depth = num_layers
    if family != "qwen3_vl":
        hidden_state_count = num_layers + 1
        trend_stop = max(2, round(REFERENCE_TREND_STOP * hidden_state_count / REFERENCE_HIDDEN_STATES))
        deco_start = max(1, math.floor(REFERENCE_DECO_START * hidden_state_count / REFERENCE_HIDDEN_STATES))
        deco_stop = min(
            hidden_state_count,
            max(deco_start + 1, math.ceil(REFERENCE_DECO_STOP * hidden_state_count / REFERENCE_HIDDEN_STATES)),
        )
        args.trend_layers = f"0:{trend_stop}"
        args.deco_layers = f"{deco_start}:{deco_stop}"

    print(
        f"[MODEL] family={family} class={model.__class__.__name__} depth={num_layers} "
        f"trend_layers={args.trend_layers} deco_layers={args.deco_layers} "
        f"processor={processor.__class__.__name__}"
    )
    if model.get_output_embeddings() is None:
        raise RuntimeError(f"{family} does not expose output embeddings required by HDEC scoring.")
    return processor_adapter, model


def main() -> None:
    mechanism = _load_mechanism_module()
    mechanism.load_model = load_compatible_model
    mechanism.main()


if __name__ == "__main__":
    main()
