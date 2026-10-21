import argparse
import csv
import importlib.util
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


SOURCE_KEY_MAP = {
    "text_prior": "text_prior_score",
    "final": "final_score",
    "anchor": "anchor_score",
    "corrected": "corrected_score",
    "final_debiased": "final_debiased_score",
    "anchor_debiased": "anchor_debiased_score",
    "corrected_debiased": "corrected_debiased_score",
    "defect_aware_pool": "defect_aware_pool_score",
    "hdec": "hdec_score",
    "hdec_debiased": "hdec_debiased_score",
    "anchor_hdec": "anchor_hdec_score",
    "anchor_hdec_debiased": "anchor_hdec_debiased_score",
    "suppression_hdec": "suppression_hdec_score",
    "suppression_hdec_debiased": "suppression_hdec_debiased_score",
    "suppression_pool_hdec": "suppression_pool_hdec_score",
    "suppression_pool_hdec_debiased": "suppression_pool_hdec_debiased_score",
    "generic_defect_pool": "generic_defect_pool_score",
    "specific_defect_pool": "specific_defect_pool_score",
    "generic_hdec": "generic_hdec_score",
    "generic_hdec_debiased": "generic_hdec_debiased_score",
    "specific_hdec": "specific_hdec_score",
    "specific_hdec_debiased": "specific_hdec_debiased_score",
}

PRIOR_KEY_MAP = {
    "final": "text_prior_score",
    "anchor": "anchor_prior_score",
    "corrected": "corrected_prior_score",
    "hdec": "hdec_prior_score",
    "anchor_hdec": "anchor_hdec_prior_score",
    "suppression_hdec": "suppression_hdec_prior_score",
    "suppression_pool_hdec": "suppression_pool_hdec_prior_score",
}

# Diagnostic source map from the previous full VisA run. This is not a trained model;
# if used on the same test split it should be reported as a diagnostic/fixed rule.
DEFAULT_SOURCE_MAP = {
    "candle": "anchor_hdec",
    "capsules": "final",
    "cashew": "final",
    "chewinggum": "hdec_debiased",
    "fryum": "hdec_debiased",
    "macaroni1": "hdec",
    "macaroni2": "anchor_hdec",
    "pcb1": "hdec",
    "pcb2": "final",
    "pcb3": "anchor",
    "pcb4": "final",
    "pipe_fryum": "final",
}

TARGETS = {
    "acc": 0.79,
    "AUROC": 0.911,
    "recall_all": 0.86,
    "precision_all": 0.95,
    "F1_all": 0.83,
}

DATASET_CHOICES = [
    "visa",
    "mvtec",
    "aitex",
    "elpv",
    "btad",
    "mpdd",
    "headct",
    "brainmri",
    "br35h",
]

NORMAL_DIR_NAMES = {
    "good",
    "normal",
    "ok",
    "no_defect",
    "nodefect",
    "no-defect",
    "nondefect",
    "non_defect",
    "defect_free",
}

AITEX_DEFECT_LABELS = [
    "fabric defect",
    "weaving defect",
    "thread defect",
    "hole",
    "stain",
    "broken yarn",
    "slub",
    "knot",
    "contamination",
    "texture anomaly",
]

ELPV_DEFECT_LABELS = [
    "solar cell defect",
    "crack",
    "micro crack",
    "finger interruption",
    "inactive area",
    "dark defect",
    "broken cell",
    "contamination",
    "solder defect",
    "edge defect",
]

BTAD_DEFECT_LABELS = [
    "surface defect",
    "scratch",
    "dent",
    "contamination",
    "deformation",
    "damaged part",
    "missing part",
    "color anomaly",
    "texture anomaly",
]

MPDD_GENERIC_DEFECT_LABELS = [
    "hole",
    "scratch",
    "bend",
    "broken part",
    "missing part",
    "contamination",
    "surface damage",
    "deformation",
    "manufacturing defect",
]

MEDICAL_DATASET_CONFIG = {
    "headct": {
        "category": "head CT scan",
        "defect_type": "intracranial hemorrhage",
        "defect_labels": [
            "intracranial hemorrhage",
            "brain bleeding",
            "hyperdense hemorrhagic region",
            "epidural hemorrhage",
            "subdural hemorrhage",
            "subarachnoid hemorrhage",
            "intraventricular hemorrhage",
            "intraparenchymal hemorrhage",
        ],
    },
    "brainmri": {
        "category": "brain MRI scan",
        "defect_type": "brain tumor",
        "defect_labels": [
            "brain tumor",
            "intracranial mass",
            "abnormal brain lesion",
            "space occupying lesion",
            "tumor tissue",
            "mass effect",
            "asymmetric abnormal region",
            "abnormal signal intensity",
        ],
    },
    "br35h": {
        "category": "brain MRI scan",
        "defect_type": "brain tumor",
        "defect_labels": [
            "brain tumor",
            "intracranial mass",
            "abnormal brain lesion",
            "space occupying lesion",
            "tumor tissue",
            "mass effect",
            "asymmetric abnormal region",
            "abnormal signal intensity",
        ],
    },
}

MATRIX_FAMILY_DESIGN = {
    "raw_source": "Direct score from one model evidence source.",
    "prior_debias_beta": "Subtract beta times text-only prior from a visual score.",
    "robust_normalized": "Normalize a source by per-category normal median/IQR.",
    "normal_zscore": "Normalize a source by per-category normal mean/std.",
    "category_fixed_normal_quantile": "Subtract a fixed per-category normal quantile boundary.",
    "category_source_selection": "Use a fixed category-to-source map from previous diagnostics.",
    "category_source_selection_robust": "Fixed category-to-source map plus normal robust normalization.",
    "multi_source_consistency": "Use agreement among several normalized sources.",
    "base_plus_consistency_gate": "Add a consensus bonus or penalty to one normalized base source.",
    "fixed_source_fusion": "Fuse several score sources with fixed rules.",
    "fixed_source_fusion_core": "Fuse the corrected/HDEC/anchor-HDEC core sources.",
    "difficulty_category_route": "Route known difficult categories to a stricter source.",
    "oracle_category_source_selection": "Upper-bound diagnostic selected with evaluated labels.",
    "oracle_category_source_selection_robust": "Oracle source selection plus normal robust normalization.",
}

HARD_CATEGORY_DEFECT_HINTS = {
    "capsules": [
        "deformed capsule",
        "damaged capsule",
        "leaking capsule",
        "foreign particle",
        "contamination",
        "bubble",
        "dent",
        "scratch",
        "stain",
        "color spot",
        "missing liquid",
        "broken shell",
    ],
    "macaroni2": [
        "broken macaroni",
        "chipped edge",
        "crack",
        "hole",
        "missing piece",
        "deformation",
        "fragment",
        "foreign particle",
        "dark spot",
        "stain",
        "burn mark",
        "surface damage",
    ],
    "pcb1": [
        "bent pin",
        "missing pin",
        "solder bridge",
        "excessive solder",
        "poor solder joint",
        "missing component",
        "misaligned component",
        "damaged component",
        "burn mark",
        "scratch",
        "crack",
        "contamination",
    ],
    "pcb2": [
        "bent pin",
        "missing pin",
        "solder bridge",
        "excessive solder",
        "poor solder joint",
        "missing component",
        "misaligned component",
        "damaged component",
        "burn mark",
        "scratch",
        "crack",
        "contamination",
    ],
    "pcb3": [
        "bent pin",
        "missing pin",
        "solder bridge",
        "excessive solder",
        "poor solder joint",
        "missing component",
        "misaligned component",
        "damaged component",
        "burn mark",
        "scratch",
        "crack",
        "contamination",
    ],
}


def load_legacy_module() -> Any:
    path = Path(__file__).with_name("0724_Mechanism_validation.py")
    spec = importlib.util.spec_from_file_location("mechanism_0724", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import legacy mechanism module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_words(text: str) -> List[str]:
    words = []
    seen = set()
    for chunk in str(text or "").split(","):
        word = chunk.strip()
        key = word.lower()
        if word and key not in seen:
            seen.add(key)
            words.append(word)
    return words


def parse_float_grid(text: str) -> List[float]:
    values = []
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    return values


def unique_preserve(items: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "0727_UP: training-free Qwen3-VL score-source and calibration matrix."
    )
    parser.add_argument("--dataset", type=str, default="visa", choices=DATASET_CHOICES)
    parser.add_argument("--visa-root", type=str, default="")
    parser.add_argument("--visa-split-csv", type=str, default=None)
    parser.add_argument("--visa-split-set", type=str, default="test")
    parser.add_argument("--mvtec-root", type=str, default="")
    parser.add_argument("--aitex-root", type=str, default="")
    parser.add_argument("--elpv-root", type=str, default="")
    parser.add_argument("--btad-root", type=str, default="")
    parser.add_argument("--mpdd-root", type=str, default="")
    parser.add_argument("--headct-root", type=str, default="")
    parser.add_argument("--brainmri-root", type=str, default="")
    parser.add_argument("--br35h-root", type=str, default="")
    parser.add_argument(
        "--elpv-label-policy",
        type=str,
        default="strict",
        choices=["strict", "threshold_0.5", "nonzero_anomaly"],
        help=(
            "ELPV labels.csv policy. strict keeps only 0.0 as normal and 1.0 as anomaly; "
            "threshold_0.5 uses label>=0.5; nonzero_anomaly treats any positive score as anomaly."
        ),
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--sample-normal", type=int, default=10)
    parser.add_argument("--sample-anomaly", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help=(
            "Strict few-shot support size per category. 0 keeps the original zero-shot/eval behavior. "
            "When >0, support normal images are selected separately and excluded from test metrics."
        ),
    )
    parser.add_argument(
        "--support-source",
        type=str,
        default="auto",
        choices=["auto", "train", "eval_normal_holdout"],
        help=(
            "Where few-shot normal support images come from. auto uses train normals when available "
            "and falls back to holding out eval normals only for datasets without train normals."
        ),
    )
    parser.add_argument(
        "--support-split-set",
        type=str,
        default="train",
        help="Split name used as normal support source for split-csv datasets such as VisA.",
    )
    parser.add_argument(
        "--strict-fewshot-support",
        action="store_true",
        help="Raise an error if a category has fewer than --shots available support normals.",
    )

    parser.add_argument("--min-pixels", type=int, default=640 * 640)
    parser.add_argument("--max-pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--allow-attn-fallback", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-deco", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=1)

    parser.add_argument("--trend-layers", type=str, default="0:34")
    parser.add_argument("--deco-layers", type=str, default="20:29")
    parser.add_argument("--deco-alpha", type=float, default=0.6)
    parser.add_argument("--deco-threshold-top-p", type=float, default=0.9)
    parser.add_argument("--deco-threshold-top-k", type=int, default=20)
    parser.add_argument(
        "--score-deco-mode",
        type=str,
        default="multi_top3",
        choices=["single", "multi_top3"],
    )
    parser.add_argument("--multi-deco-k", type=int, default=3)
    parser.add_argument(
        "--main-method-only",
        action="store_true",
        help="Only evaluate the formal main method E_raw_<main-method-source> and skip diagnostic/oracle/fusion experiments.",
    )
    parser.add_argument(
        "--main-method-source",
        type=str,
        default="hdec",
        choices=sorted(SOURCE_KEY_MAP.keys()),
        help="Score source used when --main-method-only is enabled. The formal setting is hdec.",
    )
    parser.add_argument(
        "--anchor-selection",
        type=str,
        default="conflict",
        choices=["conflict", "confidence", "evidence"],
    )
    parser.add_argument(
        "--score-label-aggregation",
        type=str,
        default="max",
        choices=["max", "logsumexp"],
    )
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument(
        "--metric-threshold-policies",
        type=str,
        default="youden,zero,normal_q95",
        help=(
            "Comma-separated policies: youden, best_f1, best_acc, zero, normal_q90, normal_q95, normal_q99, "
            "support_normal_q90, support_normal_q95, support_normal_q97, support_normal_q99, support_normal_max. "
            "The normal_q* policies use evaluated normal labels; support_normal_* uses only few-shot support normals."
        ),
    )
    parser.add_argument("--metric-threshold", type=float, default=0.0)
    parser.add_argument("--reason-audit", action="store_true")
    parser.add_argument("--reason-experiment", type=str, default="E_raw_hdec")
    parser.add_argument("--reason-max-new-tokens", type=int, default=1000)

    parser.add_argument("--no-text-prior-baseline", action="store_true")
    parser.add_argument("--normal-labels", type=str, default="good,normal,intact")
    parser.add_argument(
        "--generic-defect-labels",
        type=str,
        default="anomaly,defect,defective,abnormal,damage,damaged,crack,scratch,stain,hole,missing part,broken part,bent component,contamination,deformation",
    )
    parser.add_argument("--hedge-labels", type=str, default="maybe,possibly,likely,unclear,uncertain,appears,seems")
    parser.add_argument("--max-defect-labels", type=int, default=16)
    parser.add_argument(
        "--defect-label-json",
        type=str,
        default="",
        help=(
            "Optional category -> defect description JSON. If omitted, "
            "<dataset-root>/category_defect_labels.json is used when it exists."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="generic",
        choices=["generic", "category", "category_defects", "category_defects_guarded"],
        help=(
            "Prompt template forwarded to 0724_Mechanism_validation.py. "
            "Use category_defects_guarded for category-aware prompts with an anti-prior guard."
        ),
    )
    parser.add_argument(
        "--prompt-defect-limit",
        type=int,
        default=8,
        help="Maximum number of category-level defect descriptions inserted into category-defect prompts.",
    )
    parser.add_argument("--defect-aware-pool-k", type=int, default=3)
    parser.add_argument("--defect-aware-lambda-norm", type=float, default=1.0)
    parser.add_argument("--defect-aware-eta-hedge", type=float, default=0.2)
    parser.add_argument("--defect-aware-enable-hedge", action="store_true", default=True)
    parser.add_argument("--defect-aware-use-logsumexp", action="store_true")
    parser.add_argument("--defect-aware-gamma", type=float, default=8.0)

    parser.add_argument("--hdec-gate-gamma", type=float, default=8.0)
    parser.add_argument("--hdec-gate-tau", type=float, default=0.0)
    parser.add_argument("--hdec-beta-prior", type=float, default=1.0)
    parser.add_argument("--suppression-gate-gamma", type=float, default=1.0)
    parser.add_argument("--suppression-gate-tau", type=float, default=0.0)
    parser.add_argument("--suppression-pool-gate-gamma", type=float, default=8.0)
    parser.add_argument("--suppression-pool-gate-tau", type=float, default=0.0)

    parser.add_argument(
        "--source-candidates",
        type=str,
        default="final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,suppression_pool_hdec_debiased,generic_defect_pool,specific_defect_pool,generic_hdec_debiased,specific_hdec_debiased",
    )
    parser.add_argument("--beta-grid", type=str, default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--robust-sources", type=str, default="corrected,hdec_debiased,corrected_debiased")
    parser.add_argument(
        "--zscore-sources",
        type=str,
        default="corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,suppression_pool_hdec_debiased,generic_defect_pool,specific_defect_pool,generic_hdec_debiased,specific_hdec_debiased",
        help="Sources for per-category normal mean/std normalization experiments.",
    )
    parser.add_argument(
        "--quantile-center-sources",
        type=str,
        default="corrected_debiased,hdec_debiased,anchor_hdec,suppression_pool_hdec_debiased,generic_defect_pool,specific_defect_pool,generic_hdec_debiased,specific_hdec_debiased",
        help="Sources for fixed per-category normal-quantile centering experiments.",
    )
    parser.add_argument(
        "--normal-quantiles",
        type=str,
        default="0.9,0.95,0.99",
        help="Normal quantiles used by quantile-centering experiments.",
    )
    parser.add_argument(
        "--consensus-sources",
        type=str,
        default="final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,generic_defect_pool,specific_defect_pool,specific_hdec_debiased",
        help="Sources used to build multi-source consistency scores.",
    )
    parser.add_argument(
        "--consensus-base-sources",
        type=str,
        default="corrected_debiased,hdec_debiased,specific_hdec_debiased",
        help="Base sources receiving a consistency bonus/penalty.",
    )
    parser.add_argument("--consensus-gamma-grid", type=str, default="0.25,0.5,1.0")
    parser.add_argument(
        "--fusion-sources",
        type=str,
        default="final,anchor,corrected,corrected_debiased,hdec,hdec_debiased,anchor_hdec,generic_defect_pool,specific_defect_pool,specific_hdec_debiased",
        help="Sources used by source-fusion experiments.",
    )
    parser.add_argument(
        "--hard-categories",
        type=str,
        default="macaroni2,pcb1,pcb2,pcb3,capsules",
        help="Categories routed to stricter evidence-oriented source maps.",
    )
    parser.add_argument("--hard-category-source", type=str, default="specific_hdec_debiased")
    parser.add_argument("--easy-category-source", type=str, default="hdec_debiased")
    parser.add_argument("--disable-oracle-source-selection", action="store_true")
    parser.add_argument("--disable-default-source-map", action="store_true")

    parser.add_argument("--stream-jsonl", action="store_true")
    parser.add_argument("--include-generated-text", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _extract_reason_payload(text: str) -> Tuple[str, str]:
    payload: Dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
    reason = str(payload.get("reason", text)).strip()
    stance = str(payload.get("reason_conclusion", "")).strip().upper()
    aliases = {
        "NORMAL": "NORMAL",
        "ANOMALOUS": "ANOMALOUS",
        "ANOMALY": "ANOMALOUS",
        "ABNORMAL": "ANOMALOUS",
        "UNCERTAIN": "UNCERTAIN",
    }
    return reason, aliases.get(stance, "UNPARSEABLE")


def run_reason_audit(
    args: argparse.Namespace,
    legacy: Any,
    processor: Any,
    model: Any,
    cases: List[Dict[str, Any]],
    experiments: List[Dict[str, Any]],
    defect_bank_by_category: Dict[str, List[str]],
    output_dir: Path,
) -> None:
    from PIL import Image

    experiment = next((item for item in experiments if item["name"] == args.reason_experiment), None)
    if experiment is None:
        raise ValueError(f"Unknown reason experiment: {args.reason_experiment}")
    scores = [float(value) for value in experiment["scores"]]
    thresholds: Dict[str, float] = {}
    for category in sorted({case["category"] for case in cases}):
        idxs = [i for i, case in enumerate(cases) if case["category"] == category]
        thresholds[category] = float(
            legacy.select_threshold(
                [int(cases[i]["label"]) for i in idxs],
                [scores[i] for i in idxs],
                "youden",
                args.metric_threshold,
            )
        )
    save_json(output_dir / "reason_audit_youden_thresholds.json", thresholds)

    rows: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    cases_jsonl = output_dir / "reason_audit_cases.jsonl"
    contradictions_jsonl = output_dir / "reason_contradictions.jsonl"
    cases_jsonl.write_text("", encoding="utf-8")
    contradictions_jsonl.write_text("", encoding="utf-8")
    for idx, case in enumerate(tqdm(cases, desc="Reason audit", ncols=120)):
        category = case["category"]
        score = scores[idx]
        threshold = thresholds[category]
        answer = "ANOMALOUS" if score >= threshold else "NORMAL"
        defect_text = ", ".join(defect_bank_by_category.get(category, [])) or "unspecified visual defects"
        prompt = (
            f"The anomaly scoring module has fixed the final answer as {answer}.\n"
            f"Object category: {category}. Possible defect concepts: {defect_text}.\n"
            "Independently inspect the image. Briefly state only visible evidence, including evidence that "
            "conflicts with the fixed answer. Do not invent defects. Return JSON only:\n"
            '{"reason":"<visible evidence>","reason_conclusion":"NORMAL|ANOMALOUS|UNCERTAIN"}'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": case["image"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(case["image"]).convert("RGB")
        inputs = processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        try:
            import torch

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.reason_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            raw_reason = processor.batch_decode(
                output_ids[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            reason, reason_label = _extract_reason_payload(raw_reason)
            error = ""
        except Exception as exc:
            raw_reason, reason, reason_label, error = "", "", "UNPARSEABLE", str(exc)
        if reason_label == "UNCERTAIN":
            consistency = "uncertain"
        elif reason_label == "UNPARSEABLE":
            consistency = "unparseable"
        elif reason_label == answer:
            consistency = "consistent"
        else:
            consistency = "contradiction"
        row = {
            "dataset": args.dataset,
            "category": category,
            "image": case["image"],
            "image_id": case.get("image_id", ""),
            "ground_truth": "ANOMALOUS" if int(case["label"]) else "NORMAL",
            "score_experiment": args.reason_experiment,
            "hdec_score": score,
            "youden_threshold": threshold,
            "score_margin": score - threshold,
            "answer": answer,
            "answer_correct": answer == ("ANOMALOUS" if int(case["label"]) else "NORMAL"),
            "reason": reason,
            "raw_reason": raw_reason,
            "reason_label": reason_label,
            "consistency": consistency,
            "error": error,
        }
        rows.append(row)
        with open(cases_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if consistency == "contradiction":
            contradictions.append(row)
            with open(contradictions_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_rows = []
    for category in ["ALL"] + sorted({row["category"] for row in rows}):
        subset = rows if category == "ALL" else [row for row in rows if row["category"] == category]
        summary_rows.append(
            {
                "category": category,
                "total": len(subset),
                "consistent": sum(row["consistency"] == "consistent" for row in subset),
                "contradiction": sum(row["consistency"] == "contradiction" for row in subset),
                "uncertain": sum(row["consistency"] == "uncertain" for row in subset),
                "unparseable": sum(row["consistency"] == "unparseable" for row in subset),
                "consistency_rate": sum(row["consistency"] == "consistent" for row in subset) / max(1, len(subset)),
                "contradiction_rate": sum(row["consistency"] == "contradiction" for row in subset) / max(1, len(subset)),
                "answer_accuracy": sum(bool(row["answer_correct"]) for row in subset) / max(1, len(subset)),
            }
        )
    write_csv(output_dir / "reason_audit_summary.csv", summary_rows)


def prepare_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    args.skip_generation = not args.run_generation
    return args


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def get_dataset_root(args: argparse.Namespace, dataset: Optional[str] = None) -> Path:
    dataset = dataset or getattr(args, "dataset", "visa")
    attr_by_dataset = {
        "visa": "visa_root",
        "mvtec": "mvtec_root",
        "aitex": "aitex_root",
        "elpv": "elpv_root",
        "btad": "btad_root",
        "mpdd": "mpdd_root",
        "headct": "headct_root",
        "brainmri": "brainmri_root",
        "br35h": "br35h_root",
    }
    attr = attr_by_dataset.get(dataset)
    if attr is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root_text = str(getattr(args, attr, "") or "").strip()
    if dataset == "mvtec" and not root_text:
        root_text = str(getattr(args, "visa_root", "") or "").strip()
    if not root_text:
        raise ValueError(f"--{attr.replace('_', '-')} is required when --dataset {dataset}")
    return Path(root_text)


def sample_by_category_label(samples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.run_all:
        return samples
    rng = random.Random(args.seed)
    selected_samples: List[Dict[str, Any]] = []
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for sample in samples:
        buckets.setdefault((sample["category"], int(sample["label"])), []).append(sample)
    for key in sorted(buckets):
        bucket = buckets[key]
        k = args.sample_anomaly if key[1] == 1 else args.sample_normal
        if k is None or k < 0 or len(bucket) <= k:
            selected_samples.extend(bucket)
        else:
            selected_samples.extend(rng.sample(bucket, k))
    return selected_samples


def is_normal_defect_dir(name: str) -> bool:
    return name.strip().lower() in NORMAL_DIR_NAMES


def category_allowed(category: str, selected: Optional[set]) -> bool:
    return selected is None or category in selected


def mvtec_mask_path(root: Path, category: str, defect_type: str, image_path: Path) -> str:
    if is_normal_defect_dir(defect_type):
        return ""
    gt_dir = root / category / "ground_truth" / defect_type
    candidates = [
        gt_dir / f"{image_path.stem}_mask.png",
        gt_dir / f"{image_path.stem}_mask.bmp",
        gt_dir / f"{image_path.stem}_mask.tif",
        gt_dir / f"{image_path.stem}.png",
        gt_dir / f"{image_path.stem}.bmp",
        gt_dir / f"{image_path.stem}.tif",
        gt_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if gt_dir.exists():
        stem = image_path.stem.lower()
        for candidate in list_images(gt_dir):
            if stem in candidate.stem.lower():
                return str(candidate)
    return ""


def load_mvtec_like_samples(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    root = get_dataset_root(args, dataset)
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    samples: List[Dict[str, Any]] = []
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        category = category_dir.name
        if not category_allowed(category, selected):
            continue
        test_dir = category_dir / "test"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if is_normal_defect_dir(defect_type) else 1
            for image_path in list_images(defect_dir):
                samples.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "image": str(image_path),
                        "image_id": image_path.stem,
                        "label": label,
                        "label_name": "normal" if label == 0 else "anomaly",
                        "mask": mvtec_mask_path(root, category, defect_type, image_path),
                        "split_image": str(image_path.relative_to(root)).replace("\\", "/"),
                        "split_mask": "",
                        "anomaly_types": [] if label == 0 else [defect_type],
                        "defect_type": defect_type,
                    }
                )
    return sample_by_category_label(samples, args)


def load_mvtec_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return load_mvtec_like_samples(args, "mvtec")


def aitex_mask_path(root: Path, image_path: Path) -> str:
    mask_dir = root / "Mask_images"
    if not mask_dir.exists():
        return ""
    candidates = [
        mask_dir / image_path.name,
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}.bmp",
        mask_dir / f"{image_path.stem}.tif",
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}_mask.bmp",
        mask_dir / f"{image_path.stem}_mask.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    stem = image_path.stem.lower()
    for candidate in list_images(mask_dir):
        if stem in candidate.stem.lower() or candidate.stem.lower() in stem:
            return str(candidate)
    return ""


def load_aitex_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = get_dataset_root(args, "aitex")
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    category = "aitex_fabric"
    if not category_allowed(category, selected):
        return []
    samples: List[Dict[str, Any]] = []
    for image_path in list_images(root / "NODefect_images"):
        samples.append(
            {
                "dataset": "aitex",
                "category": category,
                "image": str(image_path),
                "image_id": image_path.stem,
                "label": 0,
                "label_name": "normal",
                "mask": "",
                "split_image": str(image_path.relative_to(root)).replace("\\", "/"),
                "split_mask": "",
                "anomaly_types": [],
                "defect_type": "good",
            }
        )
    for image_path in list_images(root / "Defect_images"):
        samples.append(
            {
                "dataset": "aitex",
                "category": category,
                "image": str(image_path),
                "image_id": image_path.stem,
                "label": 1,
                "label_name": "anomaly",
                "mask": aitex_mask_path(root, image_path),
                "split_image": str(image_path.relative_to(root)).replace("\\", "/"),
                "split_mask": "",
                "anomaly_types": ["fabric_defect"],
                "defect_type": "fabric_defect",
            }
        )
    return sample_by_category_label(samples, args)


def parse_elpv_rows(label_path: Path) -> List[Tuple[str, float, str]]:
    rows: List[Tuple[str, float, str]] = []
    with open(label_path, newline="", encoding="utf-8-sig") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if "," in raw:
                parts = [part.strip() for part in next(csv.reader([raw]))]
            elif ";" in raw:
                parts = [part.strip() for part in raw.split(";")]
            else:
                parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            cell_type = parts[2] if len(parts) > 2 else "cell"
            rows.append((parts[0], score, cell_type))
    return rows


def resolve_elpv_image(root: Path, rel_path: str) -> Optional[Path]:
    candidates = [
        root / rel_path,
        root / "images" / rel_path,
        root / "images" / Path(rel_path).name,
        root / "doc" / "images" / rel_path,
        root / "doc" / "images" / Path(rel_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def elpv_binary_label(score: float, policy: str) -> Optional[int]:
    if policy == "strict":
        if score <= 1e-8:
            return 0
        if score >= 1.0 - 1e-8:
            return 1
        return None
    if policy == "threshold_0.5":
        return 1 if score >= 0.5 else 0
    if policy == "nonzero_anomaly":
        return 1 if score > 0.0 else 0
    raise ValueError(f"Unsupported ELPV label policy: {policy}")


def load_elpv_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = get_dataset_root(args, "elpv")
    label_path = root / "labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"ELPV labels.csv not found: {label_path}")
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    samples: List[Dict[str, Any]] = []
    for rel_path, defect_score, cell_type in parse_elpv_rows(label_path):
        image_path = resolve_elpv_image(root, rel_path)
        if image_path is None or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = elpv_binary_label(defect_score, args.elpv_label_policy)
        if label is None:
            continue
        category = f"elpv_{cell_type}" if cell_type else "elpv"
        if selected and category not in selected and "elpv" not in selected:
            continue
        defect_type = "solar_cell_defect" if label == 1 else "good"
        samples.append(
            {
                "dataset": "elpv",
                "category": category,
                "image": str(image_path),
                "image_id": image_path.stem,
                "label": label,
                "label_name": "normal" if label == 0 else "anomaly",
                "mask": "",
                "split_image": str(image_path.relative_to(root)).replace("\\", "/"),
                "split_mask": "",
                "anomaly_types": [] if label == 0 else [defect_type],
                "defect_type": defect_type,
                "elpv_defect_score": defect_score,
            }
        )
    return sample_by_category_label(samples, args)


def medical_mask_path(root: Path, image_path: Path) -> str:
    """Resolve optional masks in the processed medical anomaly datasets."""
    mask_dir = root / "ground_truth"
    if not mask_dir.exists():
        return ""
    candidates = [
        mask_dir / image_path.name,
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def load_flat_medical_samples(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    """Load processed HeadCT/BrainMRI/Br35H folders with test/good and test/defect."""
    root = get_dataset_root(args, dataset)
    config = MEDICAL_DATASET_CONFIG[dataset]
    category = str(config["category"])
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    if not category_allowed(category, selected):
        return []

    samples: List[Dict[str, Any]] = []
    for label, folder_name in ((0, "good"), (1, "defect")):
        defect_type = "good" if label == 0 else str(config["defect_type"])
        for image_path in list_images(root / "test" / folder_name):
            samples.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "image": str(image_path),
                    "image_id": image_path.stem,
                    "label": label,
                    "label_name": "normal" if label == 0 else "anomaly",
                    "mask": "" if label == 0 else medical_mask_path(root, image_path),
                    "split_image": str(image_path.relative_to(root)).replace("\\", "/"),
                    "split_mask": "",
                    "anomaly_types": [] if label == 0 else [defect_type],
                    "defect_type": defect_type,
                }
            )
    return sample_by_category_label(samples, args)


def load_dataset_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = getattr(args, "dataset", "visa")
    if dataset == "mvtec":
        return load_mvtec_like_samples(args, "mvtec")
    if dataset == "mpdd":
        return load_mvtec_like_samples(args, "mpdd")
    if dataset == "btad":
        return load_mvtec_like_samples(args, "btad")
    if dataset == "aitex":
        return load_aitex_samples(args)
    if dataset == "elpv":
        return load_elpv_samples(args)
    if dataset in MEDICAL_DATASET_CONFIG:
        return load_flat_medical_samples(args, dataset)
    raise ValueError(f"load_dataset_samples does not handle dataset={dataset}")


def sample_image_key(sample: Dict[str, Any]) -> str:
    image = str(sample.get("image", "") or "")
    try:
        return str(Path(image).resolve())
    except Exception:
        return image


def annotate_sample_defaults(samples: List[Dict[str, Any]], dataset: str, split_role: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in samples:
        item = dict(sample)
        item.setdefault("dataset", dataset)
        item["split_role"] = split_role
        out.append(item)
    return out


def load_eval_samples(args: argparse.Namespace, legacy: Any) -> List[Dict[str, Any]]:
    """Load evaluation/test samples.

    For strict few-shot runs, always load the full requested evaluation split. The
    legacy --sample-normal/--sample-anomaly arguments are an evaluation sampler,
    not a few-shot protocol, so they must not affect --shots > 0 runs.
    """
    if args.shots > 0:
        old_run_all = args.run_all
        args.run_all = True
        try:
            if args.dataset == "visa":
                samples = legacy.load_split_samples(args)
            else:
                samples = load_dataset_samples(args)
        finally:
            args.run_all = old_run_all
    else:
        if args.dataset == "visa":
            samples = legacy.load_split_samples(args)
        else:
            samples = load_dataset_samples(args)
    return annotate_sample_defaults(samples, args.dataset, "test")


def make_normal_sample(
    dataset: str,
    category: str,
    image_path: Path,
    root: Path,
    split_name: str,
) -> Dict[str, Any]:
    try:
        split_image = str(image_path.relative_to(root)).replace("\\", "/")
    except ValueError:
        split_image = str(image_path)
    return {
        "dataset": dataset,
        "category": category,
        "image": str(image_path),
        "image_id": image_path.stem,
        "label": 0,
        "label_name": "normal",
        "mask": "",
        "split_image": split_image,
        "split_mask": "",
        "anomaly_types": [],
        "defect_type": "good",
        "split_name": split_name,
    }


def load_visa_train_normal_candidates(args: argparse.Namespace, legacy: Any) -> List[Dict[str, Any]]:
    old_split = args.visa_split_set
    old_run_all = args.run_all
    args.visa_split_set = args.support_split_set
    args.run_all = True
    try:
        samples = legacy.load_split_samples(args)
    finally:
        args.visa_split_set = old_split
        args.run_all = old_run_all
    normal_samples = [sample for sample in samples if int(sample.get("label", -1)) == 0]
    return annotate_sample_defaults(normal_samples, "visa", "support")


def load_mvtec_like_train_normal_candidates(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    root = get_dataset_root(args, dataset)
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    samples: List[Dict[str, Any]] = []
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        category = category_dir.name
        if not category_allowed(category, selected):
            continue
        train_dir = category_dir / "train"
        if not train_dir.exists():
            continue
        normal_dirs = [
            path
            for path in sorted(train_dir.iterdir())
            if path.is_dir() and is_normal_defect_dir(path.name)
        ]
        image_paths: List[Path] = []
        if normal_dirs:
            for normal_dir in normal_dirs:
                image_paths.extend(list_images(normal_dir))
        else:
            image_paths.extend(list_images(train_dir))
        for image_path in sorted(set(image_paths)):
            samples.append(make_normal_sample(dataset, category, image_path, root, "train"))
    return annotate_sample_defaults(samples, dataset, "support")


def load_flat_medical_train_normal_candidates(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    root = get_dataset_root(args, dataset)
    config = MEDICAL_DATASET_CONFIG[dataset]
    category = str(config["category"])
    selected = set(parse_csv_words(args.categories)) if args.categories else None
    if not category_allowed(category, selected):
        return []
    candidate_dirs = [root / "train" / "good", root / "train" / "normal", root / "train"]
    image_paths: List[Path] = []
    for folder in candidate_dirs:
        if folder.exists():
            image_paths.extend(list_images(folder))
            if image_paths:
                break
    samples = [make_normal_sample(dataset, category, image_path, root, "train") for image_path in sorted(set(image_paths))]
    return annotate_sample_defaults(samples, dataset, "support")


def load_train_normal_candidates(args: argparse.Namespace, legacy: Any) -> List[Dict[str, Any]]:
    dataset = getattr(args, "dataset", "visa")
    if dataset == "visa":
        return load_visa_train_normal_candidates(args, legacy)
    if dataset in {"mvtec", "mpdd", "btad"}:
        return load_mvtec_like_train_normal_candidates(args, dataset)
    if dataset in MEDICAL_DATASET_CONFIG:
        return load_flat_medical_train_normal_candidates(args, dataset)
    return []


def select_fewshot_support_and_test(
    args: argparse.Namespace,
    eval_samples: List[Dict[str, Any]],
    train_normal_candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if args.shots <= 0:
        return [], eval_samples, {
            "enabled": False,
            "shots": 0,
            "support_source": "none",
            "eval_total_before": len(eval_samples),
            "test_total_after": len(eval_samples),
            "support_total": 0,
            "per_category": {},
        }

    rng = random.Random(args.seed)
    categories = sorted({sample["category"] for sample in eval_samples})
    train_by_category: Dict[str, List[Dict[str, Any]]] = {}
    eval_normal_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for sample in train_normal_candidates:
        if int(sample.get("label", -1)) == 0:
            train_by_category.setdefault(sample["category"], []).append(sample)
    for sample in eval_samples:
        if int(sample.get("label", -1)) == 0:
            eval_normal_by_category.setdefault(sample["category"], []).append(sample)

    support_samples: List[Dict[str, Any]] = []
    per_category: Dict[str, Any] = {}
    for category in categories:
        train_bucket = sorted(train_by_category.get(category, []), key=sample_image_key)
        eval_bucket = sorted(eval_normal_by_category.get(category, []), key=sample_image_key)
        selected_source = "train"
        bucket = train_bucket
        if args.support_source == "eval_normal_holdout":
            selected_source = "eval_normal_holdout"
            bucket = eval_bucket
        elif args.support_source == "train":
            selected_source = "train"
            bucket = train_bucket
        elif args.support_source == "auto":
            if len(train_bucket) >= args.shots:
                selected_source = "train"
                bucket = train_bucket
            else:
                selected_source = "eval_normal_holdout"
                bucket = eval_bucket
        else:
            raise ValueError(f"Unsupported --support-source: {args.support_source}")

        if len(bucket) < args.shots:
            message = (
                f"Category {category!r} has only {len(bucket)} support normals from {selected_source}, "
                f"but --shots={args.shots}."
            )
            if args.strict_fewshot_support:
                raise RuntimeError(message)
            print(f"[WARN] {message} Using all available support normals for this category.")
        if not bucket:
            raise RuntimeError(
                f"No normal support images available for category {category!r}. "
                "Use --support-source auto/eval_normal_holdout or provide a train normal split."
            )
        chosen = rng.sample(bucket, min(args.shots, len(bucket)))
        for sample in chosen:
            item = dict(sample)
            item["split_role"] = "support"
            item["support_source"] = selected_source
            item["shots"] = args.shots
            support_samples.append(item)
        per_category[category] = {
            "requested_shots": args.shots,
            "selected_shots": len(chosen),
            "selected_source": selected_source,
            "available_train_normals": len(train_bucket),
            "available_eval_normals": len(eval_bucket),
            "support_images": [sample.get("split_image", sample.get("image", "")) for sample in chosen],
        }

    support_keys = {sample_image_key(sample) for sample in support_samples}
    test_samples = [sample for sample in eval_samples if sample_image_key(sample) not in support_keys]
    metadata = {
        "enabled": True,
        "shots": args.shots,
        "support_source": args.support_source,
        "support_split_set": args.support_split_set,
        "eval_total_before": len(eval_samples),
        "test_total_after": len(test_samples),
        "support_total": len(support_samples),
        "removed_eval_support_overlap": len(eval_samples) - len(test_samples),
        "per_category": per_category,
    }
    return support_samples, test_samples, metadata


def attach_reference_images_to_test_samples(
    test_samples: List[Dict[str, Any]],
    support_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    support_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for sample in support_samples:
        support_by_category.setdefault(sample["category"], []).append(sample)

    out: List[Dict[str, Any]] = []
    for sample in test_samples:
        refs = sorted(support_by_category.get(sample["category"], []), key=sample_image_key)
        item = dict(sample)
        item["reference_images"] = [ref["image"] for ref in refs]
        item["reference_image_ids"] = [ref.get("image_id", Path(ref["image"]).stem) for ref in refs]
        item["reference_image_count"] = len(refs)
        out.append(item)
    return out


def mvtec_defect_types(root: Path, category: str) -> List[str]:
    test_dir = root / category / "test"
    if not test_dir.exists():
        return []
    return [path.name for path in sorted(test_dir.iterdir()) if path.is_dir() and not is_normal_defect_dir(path.name)]


def dataset_defect_types(root: Path, category: str, dataset: str) -> List[str]:
    if dataset in {"mvtec", "mpdd", "btad"}:
        labels = mvtec_defect_types(root, category)
        if labels:
            return labels
        if dataset == "btad":
            return BTAD_DEFECT_LABELS
        if dataset == "mpdd":
            return MPDD_GENERIC_DEFECT_LABELS
        return []
    if dataset == "aitex":
        return AITEX_DEFECT_LABELS
    if dataset == "elpv":
        return ELPV_DEFECT_LABELS
    if dataset in MEDICAL_DATASET_CONFIG:
        return list(MEDICAL_DATASET_CONFIG[dataset]["defect_labels"])
    return []


def normalize_defect_label_items(items: Any) -> List[str]:
    labels: List[str] = []
    if not isinstance(items, list):
        return labels
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = (
                item.get("modified_label")
                or item.get("description")
                or item.get("label")
                or item.get("prompt")
                or ""
            )
        else:
            text = ""
        text = str(text).strip()
        if text:
            labels.append(text)
    return labels


def load_category_defect_label_json(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        return {}
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Defect label JSON must be a dict: {label_path}")
    out: Dict[str, List[str]] = {}
    for category, items in data.items():
        labels = normalize_defect_label_items(items)
        if labels:
            out[str(category)] = labels
    return out


def resolve_defect_label_json(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "defect_label_json", "") or "").strip()
    if explicit:
        return explicit
    try:
        dataset_root = get_dataset_root(args)
    except ValueError:
        return ""
    default_path = dataset_root / "category_defect_labels.json"
    return str(default_path) if default_path.exists() else ""


def get_probe(case: Dict[str, Any]) -> Dict[str, Any]:
    return case["hallucination_probe"]


def score_from_source(case: Dict[str, Any], source: str) -> float:
    key = SOURCE_KEY_MAP[source]
    return float(get_probe(case)[key])


def source_prior_score(case: Dict[str, Any], source: str) -> float:
    probe = get_probe(case)
    key = PRIOR_KEY_MAP.get(source)
    if key is not None and key in probe:
        return float(probe[key])
    return float(probe.get("text_prior_score", 0.0))


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    q = min(max(q, 0.0), 1.0)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def robust_normalize(cases: List[Dict[str, Any]], scores: List[float]) -> List[float]:
    by_category: Dict[str, List[int]] = {}
    for idx, case in enumerate(cases):
        by_category.setdefault(case["category"], []).append(idx)
    normalized = [0.0] * len(scores)
    for category, indexes in by_category.items():
        normal_scores = [scores[i] for i in indexes if int(cases[i]["label"]) == 0]
        reference = normal_scores if normal_scores else [scores[i] for i in indexes]
        median = percentile(reference, 0.5)
        q25 = percentile(reference, 0.25)
        q75 = percentile(reference, 0.75)
        iqr = max(q75 - q25, 1e-6)
        for i in indexes:
            normalized[i] = (scores[i] - median) / iqr
    return normalized


def zscore_normalize(cases: List[Dict[str, Any]], scores: List[float]) -> List[float]:
    by_category: Dict[str, List[int]] = {}
    for idx, case in enumerate(cases):
        by_category.setdefault(case["category"], []).append(idx)
    normalized = [0.0] * len(scores)
    for category, indexes in by_category.items():
        normal_scores = [scores[i] for i in indexes if int(cases[i]["label"]) == 0]
        reference = normal_scores if normal_scores else [scores[i] for i in indexes]
        mean = sum(reference) / max(len(reference), 1)
        var = sum((value - mean) ** 2 for value in reference) / max(len(reference), 1)
        std = max(math.sqrt(var), 1e-6)
        for i in indexes:
            normalized[i] = (scores[i] - mean) / std
    return normalized


def category_normal_quantile_center(
    cases: List[Dict[str, Any]],
    scores: List[float],
    quantile: float,
) -> List[float]:
    by_category: Dict[str, List[int]] = {}
    for idx, case in enumerate(cases):
        by_category.setdefault(case["category"], []).append(idx)
    centered = [0.0] * len(scores)
    for category, indexes in by_category.items():
        normal_scores = [scores[i] for i in indexes if int(cases[i]["label"]) == 0]
        reference = normal_scores if normal_scores else [scores[i] for i in indexes]
        threshold = percentile(reference, quantile)
        for i in indexes:
            centered[i] = scores[i] - threshold
    return centered


def normal_reference_by_category(
    reference_cases: List[Dict[str, Any]],
    reference_scores: List[float],
) -> Dict[str, List[float]]:
    by_category: Dict[str, List[float]] = {}
    for case, score in zip(reference_cases, reference_scores):
        if int(case.get("label", 0)) == 0:
            by_category.setdefault(case["category"], []).append(float(score))
    return by_category


def reference_scores_for_category(
    reference_by_category: Dict[str, List[float]],
    category: str,
) -> List[float]:
    values = reference_by_category.get(category, [])
    if values:
        return values
    pooled = [value for group in reference_by_category.values() for value in group]
    return pooled


def normalize_with_reference(
    cases: List[Dict[str, Any]],
    scores: List[float],
    reference_cases: Optional[List[Dict[str, Any]]],
    reference_scores: Optional[List[float]],
    mode: str,
    quantile: float = 0.95,
) -> List[float]:
    if not reference_cases or not reference_scores:
        if mode == "robust":
            return robust_normalize(cases, scores)
        if mode == "zscore":
            return zscore_normalize(cases, scores)
        if mode == "qcenter":
            return category_normal_quantile_center(cases, scores, quantile)
        raise ValueError(f"Unknown reference normalization mode: {mode}")

    reference_by_category = normal_reference_by_category(reference_cases, reference_scores)
    out: List[float] = []
    for case, score in zip(cases, scores):
        reference = reference_scores_for_category(reference_by_category, case["category"])
        if not reference:
            reference = [float(score)]
        if mode == "robust":
            median = percentile(reference, 0.5)
            q25 = percentile(reference, 0.25)
            q75 = percentile(reference, 0.75)
            denom = max(q75 - q25, 1e-6)
            out.append((float(score) - median) / denom)
        elif mode == "zscore":
            mean = sum(reference) / max(len(reference), 1)
            var = sum((value - mean) ** 2 for value in reference) / max(len(reference), 1)
            denom = max(math.sqrt(var), 1e-6)
            out.append((float(score) - mean) / denom)
        elif mode == "qcenter":
            out.append(float(score) - percentile(reference, quantile))
        else:
            raise ValueError(f"Unknown reference normalization mode: {mode}")
    return out


def transformed_source_scores(
    cases: List[Dict[str, Any]],
    source_scores: Dict[str, List[float]],
    sources: Iterable[str],
    transform: str,
    reference_cases: Optional[List[Dict[str, Any]]] = None,
    reference_source_scores: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[float]]:
    out = {}
    for source in sources:
        if source not in source_scores:
            continue
        scores = source_scores[source]
        if transform == "robust":
            out[source] = normalize_with_reference(
                cases,
                scores,
                reference_cases,
                reference_source_scores.get(source) if reference_source_scores else None,
                "robust",
            )
        elif transform == "zscore":
            out[source] = normalize_with_reference(
                cases,
                scores,
                reference_cases,
                reference_source_scores.get(source) if reference_source_scores else None,
                "zscore",
            )
        elif transform == "none":
            out[source] = list(scores)
        else:
            raise ValueError(f"Unknown score transform: {transform}")
    return out


def combine_source_scores(
    score_map: Dict[str, List[float]],
    sources: List[str],
    mode: str,
    weights: Optional[List[float]] = None,
) -> List[float]:
    active = [source for source in sources if source in score_map]
    if not active:
        raise ValueError("No active sources for fusion.")
    count = len(next(iter(score_map.values())))
    out = []
    if weights is None:
        weights = [1.0] * len(active)
    if len(weights) != len(active):
        raise ValueError("weights length must match active sources length")
    total_weight = sum(weights) or 1.0
    for idx in range(count):
        values = [score_map[source][idx] for source in active]
        if mode == "mean":
            out.append(sum(values) / len(values))
        elif mode == "weighted_mean":
            out.append(sum(value * weight for value, weight in zip(values, weights)) / total_weight)
        elif mode == "min":
            out.append(min(values))
        elif mode == "max":
            out.append(max(values))
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
    return out


def consensus_vote_scores(
    cases: List[Dict[str, Any]],
    source_scores: Dict[str, List[float]],
    sources: List[str],
    reference_cases: Optional[List[Dict[str, Any]]] = None,
    reference_source_scores: Optional[Dict[str, List[float]]] = None,
) -> List[float]:
    normalized = transformed_source_scores(
        cases,
        source_scores,
        sources,
        "robust",
        reference_cases,
        reference_source_scores,
    )
    active = [source for source in sources if source in normalized]
    if not active:
        raise ValueError("No active sources for consensus.")
    count = len(next(iter(normalized.values())))
    votes = []
    for idx in range(count):
        values = [math.tanh(normalized[source][idx]) for source in active]
        votes.append(sum(values) / len(values))
    return votes


def experiment_config(experiment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in experiment.items()
        if key not in {"scores", "support_scores"}
    }


def build_focused_defect_bank(
    legacy: Any,
    dataset_root: Path,
    category: str,
    generic_labels: List[str],
    max_labels: int,
    custom_labels_by_category: Optional[Dict[str, List[str]]] = None,
    dataset: str = "visa",
) -> List[str]:
    custom_labels = (custom_labels_by_category or {}).get(category, [])
    if custom_labels:
        prompt_labels = list(custom_labels)
    elif dataset in {"mvtec", "aitex", "elpv", "btad", "mpdd", "headct", "brainmri", "br35h"}:
        prompt_labels = dataset_defect_types(dataset_root, category, dataset)
    else:
        prompt_labels = legacy.build_category_defect_bank(dataset_root, category, [], max_labels)
    hard_hints = HARD_CATEGORY_DEFECT_HINTS.get(category, [])
    merged = []
    seen = set()
    for label in prompt_labels + hard_hints + generic_labels:
        clean = str(label).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            merged.append(clean)
    return merged[:max_labels]


def parse_support_threshold_policy(policy: str) -> Optional[float]:
    if policy in {"support_normal_max", "support_max"}:
        return 1.0
    if policy.startswith("support_normal_q"):
        q_text = policy.replace("support_normal_q", "")
        return float(q_text) / 100.0
    return None


def metrics_from_predictions(
    legacy: Any,
    labels: List[int],
    scores: List[float],
    preds: List[int],
    threshold_policy: str,
    threshold: Any,
    score_source: str,
    threshold_by_category: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    total = len(labels)
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    def div(num: float, den: float) -> Optional[float]:
        return num / den if den > 0 else None

    precision_anomaly = div(tp, tp + fp)
    recall_anomaly = div(tp, tp + fn)
    f1_anomaly = (
        2 * precision_anomaly * recall_anomaly / (precision_anomaly + recall_anomaly)
        if precision_anomaly is not None and recall_anomaly is not None and (precision_anomaly + recall_anomaly) > 0
        else None
    )
    precision_normal = div(tn, tn + fn)
    recall_normal = div(tn, tn + fp)
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if precision_normal is not None and recall_normal is not None and (precision_normal + recall_normal) > 0
        else None
    )
    metric = {
        "total": total,
        "acc": div(tp + tn, total),
        "AUROC": legacy.binary_auroc(labels, scores),
        "threshold_policy": threshold_policy,
        "threshold": threshold,
        "score_source": score_source,
        "recall_normal": recall_normal,
        "recall_anomaly": recall_anomaly,
        "recall_all": sum(x for x in [recall_normal, recall_anomaly] if x is not None)
        / max(1, len([x for x in [recall_normal, recall_anomaly] if x is not None])),
        "precision_normal": precision_normal,
        "precision_anomaly": precision_anomaly,
        "precision_all": sum(x for x in [precision_normal, precision_anomaly] if x is not None)
        / max(1, len([x for x in [precision_normal, precision_anomaly] if x is not None])),
        "F1_normal": f1_normal,
        "F1_anomaly": f1_anomaly,
        "F1_all": sum(x for x in [f1_normal, f1_anomaly] if x is not None)
        / max(1, len([x for x in [f1_normal, f1_anomaly] if x is not None])),
        "recall_micro": div(tp + tn, total),
        "precision_micro": div(tp + tn, total),
        "F1_micro": div(tp + tn, total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_anomaly": tp + fp,
        "pred_normal": tn + fn,
        "anomaly_total": tp + fn,
        "normal_total": tn + fp,
    }
    if threshold_by_category is not None:
        metric["threshold_by_category"] = json.dumps(threshold_by_category, ensure_ascii=False, sort_keys=True)
    return metric


def support_thresholds_by_category(
    support_cases: List[Dict[str, Any]],
    support_scores: List[float],
    policy: str,
) -> Dict[str, float]:
    quantile = parse_support_threshold_policy(policy)
    if quantile is None:
        raise ValueError(f"Not a support threshold policy: {policy}")
    reference = normal_reference_by_category(support_cases, support_scores)
    thresholds: Dict[str, float] = {}
    for category in sorted(reference):
        values = reference_scores_for_category(reference, category)
        if not values:
            raise RuntimeError(f"No support normal scores for category {category!r}")
        thresholds[category] = percentile(values, quantile)
    return thresholds


def metric_for_policy(
    legacy: Any,
    labels: List[int],
    scores: List[float],
    policy: str,
    fixed_threshold: float,
    source_name: str,
    cases: Optional[List[Dict[str, Any]]] = None,
    support_cases: Optional[List[Dict[str, Any]]] = None,
    support_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    support_quantile = parse_support_threshold_policy(policy)
    if support_quantile is not None:
        if cases is None or support_cases is None or support_scores is None:
            raise RuntimeError(f"{policy} requires few-shot support cases and support scores.")
        thresholds = support_thresholds_by_category(support_cases, support_scores, policy)
        pooled_support = [float(score) for case, score in zip(support_cases, support_scores) if int(case.get("label", 0)) == 0]
        pooled_threshold = percentile(pooled_support, support_quantile) if pooled_support else fixed_threshold
        preds = []
        used_thresholds = []
        for case, score in zip(cases, scores):
            threshold = thresholds.get(case["category"], pooled_threshold)
            preds.append(1 if float(score) >= threshold else 0)
            used_thresholds.append(threshold)
        unique_thresholds = sorted({float(value) for value in used_thresholds})
        threshold_value: Any = unique_thresholds[0] if len(unique_thresholds) == 1 else ""
        return metrics_from_predictions(
            legacy,
            labels,
            scores,
            preds,
            policy,
            threshold_value,
            source_name,
            thresholds,
        )
    if policy.startswith("normal_q"):
        q_text = policy.replace("normal_q", "")
        q = float(q_text) / 100.0
        normal_scores = [s for y, s in zip(labels, scores) if int(y) == 0]
        threshold = percentile(normal_scores, q)
        metric = legacy.binary_metrics(
            labels,
            scores,
            threshold_policy="fixed",
            fixed_threshold=threshold,
            score_source=source_name,
        )
        metric["threshold_policy"] = policy
        metric["threshold"] = threshold
        return metric
    return legacy.binary_metrics(
        labels,
        scores,
        threshold_policy=policy,
        fixed_threshold=fixed_threshold,
        score_source=source_name,
    )


def average_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "acc",
        "AUROC",
        "recall_normal",
        "recall_anomaly",
        "recall_all",
        "precision_normal",
        "precision_anomaly",
        "precision_all",
        "F1_normal",
        "F1_anomaly",
        "F1_all",
    ]
    averaged: Dict[str, Any] = {"total": len(metrics)}
    for key in keys:
        values = [float(m[key]) for m in metrics if m.get(key) is not None]
        averaged[key] = sum(values) / len(values) if values else None
    return averaged


def evaluate_experiment(
    legacy: Any,
    cases: List[Dict[str, Any]],
    scores: List[float],
    experiment: Dict[str, Any],
    threshold_policy: str,
    fixed_threshold: float,
    support_cases: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labels = [int(case["label"]) for case in cases]
    support_scores = experiment.get("support_scores")
    rows: List[Dict[str, Any]] = []
    common = {
        "experiment": experiment["name"],
        "family": experiment.get("family", ""),
        "source": experiment.get("source", ""),
        "sources": experiment.get("sources", ""),
        "beta": experiment.get("beta", ""),
        "gamma": experiment.get("gamma", ""),
        "quantile": experiment.get("quantile", ""),
        "fusion_mode": experiment.get("fusion_mode", ""),
        "weights": experiment.get("weights", ""),
        "normalization": experiment.get("normalization", "none"),
        "source_selection": experiment.get("source_selection", ""),
        "route": experiment.get("route", ""),
        "threshold_policy": threshold_policy,
        "note": experiment.get("note", ""),
    }
    overall_metric = metric_for_policy(
        legacy,
        labels,
        scores,
        threshold_policy,
        fixed_threshold,
        experiment["name"],
        cases,
        support_cases,
        support_scores,
    )
    rows.append({"scope": "overall", "category": "ALL", **common, **overall_metric})

    category_metrics = []
    for category in sorted({case["category"] for case in cases}):
        idxs = [i for i, case in enumerate(cases) if case["category"] == category]
        category_labels = [labels[i] for i in idxs]
        category_scores = [scores[i] for i in idxs]
        metric = metric_for_policy(
            legacy,
            category_labels,
            category_scores,
            threshold_policy,
            fixed_threshold,
            experiment["name"],
            [cases[i] for i in idxs],
            support_cases,
            support_scores,
        )
        category_metrics.append(metric)
        rows.append({"scope": "category", "category": category, **common, **metric})

    macro_metric = average_metrics(category_metrics)
    rows.append({"scope": "macro", "category": "ALL", **common, **macro_metric})
    return rows, macro_metric


def build_source_scores(cases: List[Dict[str, Any]], sources: Iterable[str]) -> Dict[str, List[float]]:
    score_map = {}
    for source in sources:
        if source not in SOURCE_KEY_MAP:
            raise ValueError(f"Unknown source: {source}")
        score_map[source] = [score_from_source(case, source) for case in cases]
    return score_map


def select_category_source_map(
    legacy: Any,
    cases: List[Dict[str, Any]],
    source_scores: Dict[str, List[float]],
    metric_name: str,
    threshold_policy: str,
    fixed_threshold: float,
) -> Dict[str, str]:
    selected = {}
    categories = sorted({case["category"] for case in cases})
    for category in categories:
        idxs = [i for i, case in enumerate(cases) if case["category"] == category]
        labels = [int(cases[i]["label"]) for i in idxs]
        best_source = None
        best_value = -1.0
        for source, scores in source_scores.items():
            category_scores = [scores[i] for i in idxs]
            metric = metric_for_policy(
                legacy,
                labels,
                category_scores,
                threshold_policy,
                fixed_threshold,
                source,
            )
            value = metric.get(metric_name)
            value = float(value) if value is not None else -1.0
            if value > best_value:
                best_value = value
                best_source = source
        selected[category] = best_source or next(iter(source_scores))
    return selected


def scores_from_source_map(
    cases: List[Dict[str, Any]],
    source_scores: Dict[str, List[float]],
    source_map: Dict[str, str],
) -> List[float]:
    out = []
    fallback = "corrected_debiased" if "corrected_debiased" in source_scores else next(iter(source_scores))
    for idx, case in enumerate(cases):
        source = source_map.get(case["category"], fallback)
        if source not in source_scores:
            source = fallback
        out.append(source_scores[source][idx])
    return out


def build_experiments(
    legacy: Any,
    args: argparse.Namespace,
    cases: List[Dict[str, Any]],
    support_cases: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if getattr(args, "main_method_only", False):
        source = str(getattr(args, "main_method_source", "hdec"))
        if source not in SOURCE_KEY_MAP:
            raise ValueError(f"Unknown main method source: {source}")
        source_scores = build_source_scores(cases, [source])
        support_source_scores = (
            build_source_scores(support_cases, [source])
            if support_cases
            else {}
        )
        experiment = {
            "name": f"E_raw_{source}",
            "family": "raw_source",
            "source": source,
            "scores": source_scores[source],
            "note": "Formal main method score source.",
        }
        if support_cases is not None:
            experiment["support_scores"] = support_source_scores[source]
        return [experiment], {}

    source_candidates = parse_csv_words(args.source_candidates)
    beta_grid = parse_float_grid(args.beta_grid)
    robust_sources = parse_csv_words(args.robust_sources)
    zscore_sources = parse_csv_words(args.zscore_sources)
    quantile_center_sources = parse_csv_words(args.quantile_center_sources)
    normal_quantiles = parse_float_grid(args.normal_quantiles)
    consensus_sources = parse_csv_words(args.consensus_sources)
    consensus_base_sources = parse_csv_words(args.consensus_base_sources)
    consensus_gamma_grid = parse_float_grid(args.consensus_gamma_grid)
    fusion_sources = parse_csv_words(args.fusion_sources)
    hard_categories = set(parse_csv_words(args.hard_categories))

    extra_sources = (
        robust_sources
        + zscore_sources
        + quantile_center_sources
        + consensus_sources
        + consensus_base_sources
        + fusion_sources
        + list(DEFAULT_SOURCE_MAP.values())
        + [args.hard_category_source, args.easy_category_source]
    )
    source_candidates = unique_preserve(source_candidates + extra_sources)
    unknown_sources = [source for source in source_candidates if source not in SOURCE_KEY_MAP]
    if unknown_sources:
        raise ValueError(f"Unknown source(s): {unknown_sources}")
    source_scores = build_source_scores(cases, source_candidates)
    support_source_scores = (
        build_source_scores(support_cases, source_candidates)
        if support_cases
        else {}
    )
    experiments = []
    source_maps: Dict[str, Any] = {}

    def add_experiment(experiment: Dict[str, Any], support_scores: Optional[List[float]] = None) -> None:
        if support_cases is not None:
            if support_scores is None:
                raise RuntimeError(f"Missing support_scores for experiment {experiment.get('name')}")
            experiment = dict(experiment)
            experiment["support_scores"] = support_scores
        experiments.append(experiment)

    for source in source_candidates:
        add_experiment(
            {
                "name": f"E_raw_{source}",
                "family": "raw_source",
                "source": source,
                "scores": source_scores[source],
            },
            support_source_scores.get(source),
        )

    beta_sources = [source for source in source_candidates if source in PRIOR_KEY_MAP]
    for source in beta_sources:
        raw = source_scores[source]
        priors = [source_prior_score(case, source) for case in cases]
        support_raw = support_source_scores.get(source, [])
        support_priors = [source_prior_score(case, source) for case in support_cases] if support_cases else []
        for beta in beta_grid:
            scores = [score - beta * prior for score, prior in zip(raw, priors)]
            support_scores = [score - beta * prior for score, prior in zip(support_raw, support_priors)] if support_cases else None
            add_experiment(
                {
                    "name": f"E_beta_{source}_b{beta:g}",
                    "family": "prior_debias_beta",
                    "source": source,
                    "beta": beta,
                    "scores": scores,
                },
                support_scores,
            )

    for source in robust_sources:
        if source not in source_scores:
            continue
        scores = normalize_with_reference(
            cases,
            source_scores[source],
            support_cases,
            support_source_scores.get(source),
            "robust",
        )
        support_scores = (
            normalize_with_reference(
                support_cases,
                support_source_scores[source],
                support_cases,
                support_source_scores[source],
                "robust",
            )
            if support_cases
            else None
        )
        add_experiment(
            {
                "name": f"E_robust_{source}",
                "family": "robust_normalized",
                "source": source,
                "normalization": "per_category_normal_median_iqr",
                "scores": scores,
                "note": "Uses few-shot support normal statistics when --shots > 0; otherwise diagnostic on evaluated split.",
            },
            support_scores,
        )

    for source in zscore_sources:
        if source not in source_scores:
            continue
        scores = normalize_with_reference(
            cases,
            source_scores[source],
            support_cases,
            support_source_scores.get(source),
            "zscore",
        )
        support_scores = (
            normalize_with_reference(
                support_cases,
                support_source_scores[source],
                support_cases,
                support_source_scores[source],
                "zscore",
            )
            if support_cases
            else None
        )
        add_experiment(
            {
                "name": f"E_zscore_{source}",
                "family": "normal_zscore",
                "source": source,
                "normalization": "per_category_normal_mean_std",
                "scores": scores,
                "note": "Centers each category with few-shot support normal mean/std when --shots > 0.",
            },
            support_scores,
        )

    for source in quantile_center_sources:
        if source not in source_scores:
            continue
        for quantile in normal_quantiles:
            scores = normalize_with_reference(
                cases,
                source_scores[source],
                support_cases,
                support_source_scores.get(source),
                "qcenter",
                quantile,
            )
            support_scores = (
                normalize_with_reference(
                    support_cases,
                    support_source_scores[source],
                    support_cases,
                    support_source_scores[source],
                    "qcenter",
                    quantile,
                )
                if support_cases
                else None
            )
            add_experiment(
                {
                    "name": f"E_qcenter_{source}_q{quantile:g}",
                    "family": "category_fixed_normal_quantile",
                    "source": source,
                    "quantile": quantile,
                    "normalization": f"score_minus_category_normal_q{quantile:g}",
                    "scores": scores,
                    "note": "Equivalent to a fixed per-category support-normal quantile boundary when threshold=0.",
                },
                support_scores,
            )

    if not args.disable_default_source_map:
        source_map = {
            category: DEFAULT_SOURCE_MAP.get(category, "corrected_debiased")
            for category in sorted({case["category"] for case in cases})
        }
        source_maps["fixed_default_source_map"] = source_map
        scores = scores_from_source_map(cases, source_scores, source_map)
        support_scores = scores_from_source_map(support_cases, support_source_scores, source_map) if support_cases else None
        add_experiment(
            {
                "name": "E_fixed_default_source_map",
                "family": "category_source_selection",
                "source_selection": "fixed_default_map",
                "scores": scores,
                "source_map": source_map,
                "note": "Fixed diagnostic rule map; do not call this trained.",
            },
            support_scores,
        )
        robust_scores = normalize_with_reference(
            cases,
            scores,
            support_cases,
            support_scores,
            "robust",
        )
        robust_support_scores = (
            normalize_with_reference(
                support_cases,
                support_scores,
                support_cases,
                support_scores,
                "robust",
            )
            if support_cases
            else None
        )
        add_experiment(
            {
                "name": "E_fixed_default_source_map_robust",
                "family": "category_source_selection_robust",
                "source_selection": "fixed_default_map",
                "normalization": "per_category_normal_median_iqr",
                "scores": robust_scores,
                "source_map": source_map,
                "note": "Fixed diagnostic rule map plus support-normal robust normalization when --shots > 0.",
            },
            robust_support_scores,
        )

    active_consensus_sources = [source for source in consensus_sources if source in source_scores]
    if active_consensus_sources:
        consensus = consensus_vote_scores(
            cases,
            source_scores,
            active_consensus_sources,
            support_cases,
            support_source_scores if support_cases else None,
        )
        support_consensus = (
            consensus_vote_scores(
                support_cases,
                support_source_scores,
                active_consensus_sources,
                support_cases,
                support_source_scores,
            )
            if support_cases
            else None
        )
        add_experiment(
            {
                "name": "E_consensus_vote_only",
                "family": "multi_source_consistency",
                "sources": ",".join(active_consensus_sources),
                "fusion_mode": "mean_tanh_robust_sign",
                "scores": consensus,
                "note": "Positive only when multiple normalized sources consistently favor anomaly.",
            },
            support_consensus,
        )
        robust_source_map = transformed_source_scores(
            cases,
            source_scores,
            active_consensus_sources,
            "robust",
            support_cases,
            support_source_scores if support_cases else None,
        )
        support_robust_source_map = transformed_source_scores(
            support_cases,
            support_source_scores,
            active_consensus_sources,
            "robust",
            support_cases,
            support_source_scores,
        ) if support_cases else {}
        for mode in ["mean", "min", "max"]:
            add_experiment(
                {
                    "name": f"E_consensus_{mode}_robust",
                    "family": "multi_source_consistency",
                    "sources": ",".join(active_consensus_sources),
                    "fusion_mode": mode,
                    "normalization": "per_source_per_category_normal_median_iqr",
                    "scores": combine_source_scores(robust_source_map, active_consensus_sources, mode),
                    "note": (
                        "mean balances sources; min is precision-strict consensus; "
                        "max is recall-oriented consensus."
                    ),
                },
                combine_source_scores(support_robust_source_map, active_consensus_sources, mode) if support_cases else None,
            )
        for base_source in consensus_base_sources:
            if base_source not in source_scores:
                continue
            base = normalize_with_reference(
                cases,
                source_scores[base_source],
                support_cases,
                support_source_scores.get(base_source),
                "robust",
            )
            support_base = (
                normalize_with_reference(
                    support_cases,
                    support_source_scores[base_source],
                    support_cases,
                    support_source_scores[base_source],
                    "robust",
                )
                if support_cases
                else None
            )
            for gamma in consensus_gamma_grid:
                scores = [base_score + gamma * vote for base_score, vote in zip(base, consensus)]
                support_scores = (
                    [base_score + gamma * vote for base_score, vote in zip(support_base, support_consensus)]
                    if support_cases and support_base is not None and support_consensus is not None
                    else None
                )
                add_experiment(
                    {
                        "name": f"E_consensus_bonus_{base_source}_g{gamma:g}",
                        "family": "base_plus_consistency_gate",
                        "source": base_source,
                        "sources": ",".join(active_consensus_sources),
                        "gamma": gamma,
                        "fusion_mode": "base_robust_plus_gamma_consensus",
                        "normalization": "per_category_normal_median_iqr",
                        "scores": scores,
                        "note": "Adds anomaly support only when several score sources agree.",
                    },
                    support_scores,
                )
    active_fusion_sources = [source for source in fusion_sources if source in source_scores]
    if active_fusion_sources:
        raw_source_map = transformed_source_scores(cases, source_scores, active_fusion_sources, "none")
        support_raw_source_map = (
            transformed_source_scores(support_cases, support_source_scores, active_fusion_sources, "none")
            if support_cases
            else {}
        )
        robust_source_map = transformed_source_scores(
            cases,
            source_scores,
            active_fusion_sources,
            "robust",
            support_cases,
            support_source_scores if support_cases else None,
        )
        support_robust_source_map = (
            transformed_source_scores(
                support_cases,
                support_source_scores,
                active_fusion_sources,
                "robust",
                support_cases,
                support_source_scores,
            )
            if support_cases
            else {}
        )
        add_experiment(
            {
                "name": "E_fusion_mean_raw",
                "family": "fixed_source_fusion",
                "sources": ",".join(active_fusion_sources),
                "fusion_mode": "mean",
                "scores": combine_source_scores(raw_source_map, active_fusion_sources, "mean"),
                "note": "Simple fixed-weight mean over raw score sources.",
            },
            combine_source_scores(support_raw_source_map, active_fusion_sources, "mean") if support_cases else None,
        )
        add_experiment(
            {
                "name": "E_fusion_mean_robust",
                "family": "fixed_source_fusion",
                "sources": ",".join(active_fusion_sources),
                "fusion_mode": "mean",
                "normalization": "per_source_per_category_normal_median_iqr",
                "scores": combine_source_scores(robust_source_map, active_fusion_sources, "mean"),
                "note": "Mean after each source is normalized by support normal statistics when --shots > 0.",
            },
            combine_source_scores(support_robust_source_map, active_fusion_sources, "mean") if support_cases else None,
        )
        strict_sources = [
            source
            for source in ["corrected_debiased", "hdec_debiased", "anchor_hdec", "specific_hdec_debiased"]
            if source in robust_source_map
        ]
        if len(strict_sources) >= 2:
            for mode in ["mean", "min", "max"]:
                add_experiment(
                    {
                        "name": f"E_fusion_core_{mode}_robust",
                        "family": "fixed_source_fusion_core",
                        "sources": ",".join(strict_sources),
                        "fusion_mode": mode,
                        "normalization": "per_source_per_category_normal_median_iqr",
                        "scores": combine_source_scores(robust_source_map, strict_sources, mode),
                        "note": "Core fusion of corrected, HDEC, and anchor-HDEC style scores.",
                    },
                    combine_source_scores(support_robust_source_map, strict_sources, mode) if support_cases else None,
                )
            weights = [0.2, 0.35, 0.2, 0.25][: len(strict_sources)]
            if len(weights) == len(strict_sources):
                add_experiment(
                    {
                        "name": "E_fusion_core_weighted_robust",
                        "family": "fixed_source_fusion_core",
                        "sources": ",".join(strict_sources),
                        "weights": ",".join(f"{w:g}" for w in weights),
                        "fusion_mode": "weighted_mean",
                        "normalization": "per_source_per_category_normal_median_iqr",
                        "scores": combine_source_scores(
                            robust_source_map,
                            strict_sources,
                            "weighted_mean",
                            weights,
                        ),
                        "note": "Fixed prior: HDEC gets larger weight, anchor/corrected stabilize it.",
                    },
                    combine_source_scores(
                        support_robust_source_map,
                        strict_sources,
                        "weighted_mean",
                        weights,
                    ) if support_cases else None,
                )

    categories = sorted({case["category"] for case in cases})
    source_map = {}
    for category in categories:
        if category in hard_categories:
            source_map[category] = args.hard_category_source
        else:
            source_map[category] = args.easy_category_source
    source_maps["difficulty_route"] = source_map
    routed_scores = scores_from_source_map(cases, source_scores, source_map)
    support_routed_scores = scores_from_source_map(support_cases, support_source_scores, source_map) if support_cases else None
    add_experiment(
        {
            "name": "E_difficulty_route",
            "family": "difficulty_category_route",
            "source_selection": "hard_categories_to_hard_source",
            "route": f"hard={args.hard_category_source};easy={args.easy_category_source}",
            "scores": routed_scores,
            "source_map": source_map,
            "note": "Fixed category rule: known difficult categories use stricter anchor-level evidence.",
        },
        support_routed_scores,
    )
    routed_robust_scores = normalize_with_reference(
        cases,
        routed_scores,
        support_cases,
        support_routed_scores,
        "robust",
    )
    support_routed_robust_scores = (
        normalize_with_reference(
            support_cases,
            support_routed_scores,
            support_cases,
            support_routed_scores,
            "robust",
        )
        if support_cases
        else None
    )
    add_experiment(
        {
            "name": "E_difficulty_route_robust",
            "family": "difficulty_category_route",
            "source_selection": "hard_categories_to_hard_source",
            "route": f"hard={args.hard_category_source};easy={args.easy_category_source}",
            "normalization": "per_category_normal_median_iqr",
            "scores": routed_robust_scores,
            "source_map": source_map,
            "note": "Difficulty route with support-normal robust normalization when --shots > 0.",
        },
        support_routed_robust_scores,
    )

    if not args.disable_oracle_source_selection:
        for metric_name in ["AUROC", "F1_all", "F1_anomaly", "acc"]:
            source_map = select_category_source_map(
                legacy,
                cases,
                source_scores,
                metric_name,
                "youden",
                args.metric_threshold,
            )
            key = f"oracle_best_{metric_name}"
            source_maps[key] = source_map
            scores = scores_from_source_map(cases, source_scores, source_map)
            support_scores = scores_from_source_map(support_cases, support_source_scores, source_map) if support_cases else None
            add_experiment(
                {
                    "name": f"E_oracle_category_best_{metric_name}",
                    "family": "oracle_category_source_selection",
                    "source_selection": f"best_{metric_name}",
                    "scores": scores,
                    "source_map": source_map,
                    "note": "Upper-bound diagnostic: selected with evaluated labels.",
                },
                support_scores,
            )
            robust_scores = normalize_with_reference(
                cases,
                scores,
                support_cases,
                support_scores,
                "robust",
            )
            robust_support_scores = (
                normalize_with_reference(
                    support_cases,
                    support_scores,
                    support_cases,
                    support_scores,
                    "robust",
                )
                if support_cases
                else None
            )
            add_experiment(
                {
                    "name": f"E_oracle_category_best_{metric_name}_robust",
                    "family": "oracle_category_source_selection_robust",
                    "source_selection": f"best_{metric_name}",
                    "normalization": "per_category_normal_median_iqr",
                    "scores": robust_scores,
                    "source_map": source_map,
                    "note": "Upper-bound diagnostic: selected with evaluated labels.",
                },
                robust_support_scores,
            )

    return experiments, source_maps


def target_gap(metric: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, target in TARGETS.items():
        value = metric.get(key)
        out[f"{key}_target"] = target
        out[f"{key}_gap"] = (float(value) - target) if value is not None else None
        out[f"{key}_pass"] = bool(value is not None and float(value) >= target)
    out["all_targets_pass"] = all(out[f"{key}_pass"] for key in TARGETS)
    return out


def compute_probe_cases(
    args: argparse.Namespace,
    legacy: Any,
    processor: Any,
    model: Any,
    samples: List[Dict[str, Any]],
    defect_bank_by_category: Dict[str, List[str]],
    normal_bank: List[str],
    hedge_bank: List[str],
    output_dir: Path,
    role: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cases: List[Dict[str, Any]] = []
    compact_cases: List[Dict[str, Any]] = []
    stream_jsonl = output_dir / f"{role}_cases.compact.jsonl"
    if args.stream_jsonl and stream_jsonl.exists():
        stream_jsonl.unlink()

    for idx, sample in enumerate(tqdm(samples, desc=f"0727_UP {role} probe", ncols=120), start=1):
        try:
            case = legacy.compute_probe(
                args,
                processor,
                model,
                sample,
                defect_bank_by_category[sample["category"]],
                normal_bank,
                hedge_bank,
            )
        except Exception as exc:
            case = {**sample, "error": str(exc)}
            print(f"[WARN] Failed {role} sample {sample['category']}/{sample.get('image_id', '')}: {exc}")
        for key in [
            "dataset",
            "anomaly_types",
            "defect_type",
            "split_role",
            "support_source",
            "shots",
            "split_name",
            "reference_images",
            "reference_image_ids",
            "reference_image_count",
        ]:
            if key in sample:
                case[key] = sample.get(key)
        cases.append(case)
        compact = legacy.compact_case(case, include_generated_text=args.include_generated_text)
        for key in [
            "dataset",
            "anomaly_types",
            "defect_type",
            "split_role",
            "support_source",
            "shots",
            "split_name",
            "reference_images",
            "reference_image_ids",
            "reference_image_count",
        ]:
            if key in case:
                compact[key] = case.get(key)
        compact_cases.append(compact)
        if args.stream_jsonl:
            with open(stream_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(compact, ensure_ascii=False) + "\n")
        if args.log_interval > 0 and idx % args.log_interval == 0:
            valid = sum(1 for item in cases if "hallucination_probe" in item)
            print(f"[INFO] {role} processed={idx}/{len(samples)} valid={valid}")

    valid_cases = [case for case in cases if "hallucination_probe" in case]
    failed_cases = [case for case in cases if "hallucination_probe" not in case]
    return valid_cases, failed_cases, compact_cases


def run() -> None:
    args = prepare_legacy_args(parse_args())
    legacy = load_legacy_module()
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "0-config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "0727_UP.py",
            "training_free": True,
            "targets": TARGETS,
            "args": vars(args),
        },
    )

    eval_samples = load_eval_samples(args, legacy)
    if not eval_samples:
        raise RuntimeError("No samples found.")
    train_normal_candidates = load_train_normal_candidates(args, legacy) if args.shots > 0 else []
    support_samples, samples, fewshot_metadata = select_fewshot_support_and_test(
        args,
        eval_samples,
        train_normal_candidates,
    )
    if args.shots > 0:
        samples = attach_reference_images_to_test_samples(samples, support_samples)
        fewshot_metadata["reference_prompt_enabled"] = True
        fewshot_metadata["reference_prompt_mode"] = "normal_references_then_query"
    else:
        fewshot_metadata["reference_prompt_enabled"] = False
    save_json(output_dir / "fewshot_split.json", fewshot_metadata)
    save_json(output_dir / "fewshot_support_samples.json", support_samples)
    if args.shots > 0:
        print(
            "[INFO] Strict few-shot split: "
            f"shots={args.shots} support={len(support_samples)} "
            f"test={len(samples)} eval_before={len(eval_samples)} "
            f"removed_overlap={fewshot_metadata.get('removed_eval_support_overlap', 0)}"
        )
    print(f"[INFO] Loaded {len(samples)} test samples.")

    normal_bank = parse_csv_words(args.normal_labels)
    hedge_bank = parse_csv_words(args.hedge_labels)
    generic_defect_labels = parse_csv_words(args.generic_defect_labels)
    categories = sorted({sample["category"] for sample in samples + support_samples})
    dataset_root = get_dataset_root(args)
    defect_label_json = resolve_defect_label_json(args)
    custom_defect_labels = load_category_defect_label_json(defect_label_json)
    if defect_label_json:
        print(f"[INFO] Loaded category defect labels: {defect_label_json}")
    defect_bank_by_category = {
        category: build_focused_defect_bank(
            legacy,
            dataset_root,
            category,
            generic_defect_labels,
            args.max_defect_labels,
            custom_defect_labels,
            args.dataset,
        )
        for category in categories
    }
    save_json(output_dir / "defect_bank_by_category.json", defect_bank_by_category)
    save_json(output_dir / "custom_defect_labels_by_category.json", custom_defect_labels)
    save_json(output_dir / "hard_category_defect_hints.json", HARD_CATEGORY_DEFECT_HINTS)

    processor, model = legacy.load_model(args)
    valid_support_cases: List[Dict[str, Any]] = []
    failed_support_cases: List[Dict[str, Any]] = []
    compact_support_cases: List[Dict[str, Any]] = []
    if support_samples:
        valid_support_cases, failed_support_cases, compact_support_cases = compute_probe_cases(
            args,
            legacy,
            processor,
            model,
            support_samples,
            defect_bank_by_category,
            normal_bank,
            hedge_bank,
            output_dir,
            "support",
        )

    valid_cases, failed_cases, compact_cases = compute_probe_cases(
        args,
        legacy,
        processor,
        model,
        samples,
        defect_bank_by_category,
        normal_bank,
        hedge_bank,
        output_dir,
        "test",
    )
    save_json(output_dir / "up_cases.json", compact_cases)
    save_json(output_dir / "failed_cases.json", failed_cases[:200])
    save_json(output_dir / "support_cases.json", compact_support_cases)
    save_json(output_dir / "failed_support_cases.json", failed_support_cases[:200])

    if not valid_cases:
        raise RuntimeError("No valid probe cases.")
    if args.shots > 0 and not valid_support_cases:
        raise RuntimeError("No valid few-shot support probe cases.")

    experiments, source_maps = build_experiments(legacy, args, valid_cases, valid_support_cases if args.shots > 0 else None)
    if args.reason_audit:
        run_reason_audit(
            args,
            legacy,
            processor,
            model,
            valid_cases,
            experiments,
            defect_bank_by_category,
            output_dir,
        )
    save_json(output_dir / "up_source_maps.json", source_maps)
    experiment_configs = [experiment_config(experiment) for experiment in experiments]
    save_json(output_dir / "up_experiment_configs.json", experiment_configs)
    write_csv(output_dir / "up_experiment_configs.csv", experiment_configs)
    save_json(output_dir / "up_matrix_family_design.json", MATRIX_FAMILY_DESIGN)

    threshold_policies = parse_csv_words(args.metric_threshold_policies)
    all_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        scores = experiment["scores"]
        for policy in threshold_policies:
            rows, macro_metric = evaluate_experiment(
                legacy,
                valid_cases,
                scores,
                experiment,
                policy,
                args.metric_threshold,
                valid_support_cases if args.shots > 0 else None,
            )
            for row in rows:
                row["shots"] = args.shots
                row["support_source"] = args.support_source if args.shots > 0 else ""
                row["support_total"] = fewshot_metadata.get("support_total", 0)
                row["eval_total_before"] = fewshot_metadata.get("eval_total_before", len(valid_cases))
                row["test_total_after"] = fewshot_metadata.get("test_total_after", len(valid_cases))
            all_rows.extend(rows)
            macro_row = {
                "scope": "macro",
                "category": "ALL",
                "experiment": experiment["name"],
                "family": experiment.get("family", ""),
                "source": experiment.get("source", ""),
                "sources": experiment.get("sources", ""),
                "beta": experiment.get("beta", ""),
                "gamma": experiment.get("gamma", ""),
                "quantile": experiment.get("quantile", ""),
                "fusion_mode": experiment.get("fusion_mode", ""),
                "weights": experiment.get("weights", ""),
                "normalization": experiment.get("normalization", "none"),
                "source_selection": experiment.get("source_selection", ""),
                "route": experiment.get("route", ""),
                "threshold_policy": policy,
                "shots": args.shots,
                "support_source": args.support_source if args.shots > 0 else "",
                "support_total": fewshot_metadata.get("support_total", 0),
                "eval_total_before": fewshot_metadata.get("eval_total_before", len(valid_cases)),
                "test_total_after": fewshot_metadata.get("test_total_after", len(valid_cases)),
                "note": experiment.get("note", ""),
                **macro_metric,
                **target_gap(macro_metric),
            }
            macro_rows.append(macro_row)

    write_csv(output_dir / "up_matrix_summary.csv", all_rows)
    write_csv(output_dir / "up_macro_summary.csv", macro_rows)
    write_csv(
        output_dir / "up_per_category_summary.csv",
        [row for row in all_rows if row.get("scope") == "category"],
    )

    best_by_metric = {}
    for key in ["acc", "AUROC", "recall_all", "precision_all", "F1_all", "F1_anomaly"]:
        candidates = [row for row in macro_rows if row.get(key) is not None]
        if candidates:
            best_by_metric[key] = max(candidates, key=lambda row: float(row[key]))
    save_json(output_dir / "up_best_by_metric.json", best_by_metric)

    print(f"[OK] Saved cases: {output_dir / 'up_cases.json'}")
    print(f"[OK] Saved matrix summary: {output_dir / 'up_matrix_summary.csv'}")
    print(f"[OK] Saved macro summary: {output_dir / 'up_macro_summary.csv'}")
    print(f"[OK] Saved per-category summary: {output_dir / 'up_per_category_summary.csv'}")
    print(f"[OK] Saved experiment configs: {output_dir / 'up_experiment_configs.json'}")
    print(f"[OK] Saved best-by-metric: {output_dir / 'up_best_by_metric.json'}")
    for key in ["acc", "AUROC", "recall_all", "precision_all", "F1_all"]:
        row = best_by_metric.get(key)
        if row:
            print(
                "[BEST] "
                f"{key}={row[key]} experiment={row['experiment']} "
                f"policy={row['threshold_policy']} family={row['family']}"
            )


if __name__ == "__main__":
    run()

