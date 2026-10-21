#!/usr/bin/env python
import argparse
import csv
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
GOOD_ALIASES = {"good", "normal", "ok", "no defect", "non-defective", "non_defective"}


def parse_csv_arg(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate image-level AUROC for Qwen3-VL (+/- DeCo) on MVTec/VisA ZSAD."
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root path (MVTec or VisA).")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", "mvtec", "visa", "flat_binary", "br35hdet", "brainmri", "headct"],
        help="Dataset type: mvtec, visa, flat_binary, br35hdet (TEST+annotations_test.json), brainmri (img_dir+ann_dir), headct (Dataset/labels.csv+head_ct), auto tries all.",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--category", type=str, default=None, help="Single category to evaluate AUROC.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to evaluate AUROC.")
    parser.add_argument("--max_samples_per_subtype", type=int, default=None)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--assistant_prefix", type=str, default='{"label": "')
    parser.add_argument(
        "--score_mode",
        type=str,
        default="one_minus_good",
        choices=["one_minus_good", "max_defect", "defect_minus_good", "defect_over_sum"],
        help="How to convert first-step token probabilities into anomaly score.",
    )
    parser.add_argument("--analysis_step", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--min_pixels", type=int, default=640 * 640,
        help="Min image pixels (lower = less VRAM; if OOM try reducing max_pixels).",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=16 * 16 * 2 * 2 * 2 * 6144,
        help="Max image pixels (reduce to e.g. 1572864 or 2097152 if CUDA OOM).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--allow_attn_fallback", action="store_true")

    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default="default",
        choices=["default", "defect_aware"],
    )
    parser.add_argument("--lambda_norm", type=float, default=1.0)
    parser.add_argument("--eta_hedge", type=float, default=0.0)
    parser.add_argument("--enable_hedge", action="store_true")
    parser.add_argument("--use_logsumexp", action="store_true")
    parser.add_argument("--gamma", type=float, default=8.0)
    parser.add_argument("--anchor_defect_words", type=str, default=None)
    parser.add_argument("--anchor_norm_words", type=str, default=None)
    parser.add_argument("--anchor_hedge_words", type=str, default=None)
    parser.add_argument("--deco_debug", action="store_true")

    parser.add_argument("--print_image_name", action="store_true")
    return parser.parse_args()


def load_eval_helpers(eval_script_path: Path):
    spec = importlib.util.spec_from_file_location("mvtec_eval_helpers", str(eval_script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper script: {eval_script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def resolve_selected_categories(category: Optional[str], categories: Optional[str]) -> Optional[List[str]]:
    selected = []
    if category:
        selected.append(category.strip())
    if categories:
        selected.extend(parse_csv_arg(categories))
    selected = list(dict.fromkeys([x for x in selected if x]))
    return selected if len(selected) > 0 else None


def discover_mvtec_like(
    dataset_root: Path,
    allowed_categories: Optional[List[str]],
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}

    categories = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    if allowed_categories is not None:
        allow = set(allowed_categories)
        categories = [p for p in categories if p.name in allow]

    for category_dir in categories:
        category = category_dir.name
        test_dir = category_dir / "test"
        if not test_dir.exists():
            continue

        train_good_dir = category_dir / "train" / "good"
        test_good_dir = test_dir / "good"
        subtype_dirs = [p for p in sorted(test_dir.iterdir()) if p.is_dir()]
        defect_types = [p.name for p in subtype_dirs if p.name != "good"]

        support_good_paths = list_images(train_good_dir)
        if len(support_good_paths) == 0:
            support_good_paths = list_images(test_good_dir)

        category_meta[category] = {
            "defect_types": defect_types,
            "support_good_paths": support_good_paths,
            "source": "mvtec_like",
        }

        for subtype_dir in subtype_dirs:
            image_paths = list_images(subtype_dir)
            if max_samples_per_subtype is not None:
                image_paths = image_paths[: max_samples_per_subtype]
            for image_path in image_paths:
                samples.append(
                    {
                        "category": category,
                        "image_path": str(image_path),
                        "ground_truth_label": subtype_dir.name,
                    }
                )
    return samples, category_meta


# 扁平二分类布局：根目录下直接有「正常」与「缺陷」两个文件夹（如 AITEX 的 NODefect_images + Defect_images）
FLAT_GOOD_NAMES = {"good", "normal", "nodefect_images", "no_defect", "non_defective"}
FLAT_DEFECT_NAMES = {"defect", "defect_images", "anomaly", "defective"}


def discover_flat_binary(
    dataset_root: Path,
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """根目录下有两类文件夹：正常/缺陷，无 category/test 层级。用于 AITEX 等。"""
    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}
    good_dir: Optional[Path] = None
    defect_dir: Optional[Path] = None
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        name_lower = p.name.lower().replace(" ", "_")
        if name_lower in FLAT_GOOD_NAMES or (name_lower.endswith("_images") and "no" in name_lower and "defect" in name_lower):
            good_dir = p
        elif name_lower in FLAT_DEFECT_NAMES or (name_lower.endswith("_images") and "defect" in name_lower and "no" not in name_lower):
            defect_dir = p
    if good_dir is None or defect_dir is None:
        return samples, category_meta
    good_paths = list_images(good_dir)
    defect_paths = list_images(defect_dir)
    if max_samples_per_subtype is not None:
        good_paths = good_paths[: max_samples_per_subtype]
        defect_paths = defect_paths[: max_samples_per_subtype]
    category = "default"
    category_meta[category] = {
        "defect_types": ["defect"],
        "support_good_paths": good_paths,
        "source": "flat_binary",
    }
    for image_path in good_paths:
        samples.append({"category": category, "image_path": str(image_path), "ground_truth_label": "good"})
    for image_path in defect_paths:
        samples.append({"category": category, "image_path": str(image_path), "ground_truth_label": "defect"})
    return samples, category_meta


def _br35hdet_resolve_image_path(dataset_root: Path, fn: str) -> Optional[Path]:
    """Resolve image path: TEST, TRAIN, VAL, then dataset root."""
    for sub in ("TEST", "test", "TRAIN", "train", "VAL", "val"):
        p = dataset_root / sub / fn
        if p.exists():
            return p
    p = dataset_root / fn
    return p if p.exists() else None


def discover_br35hdet(
    dataset_root: Path,
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Br35HDet: TEST/ + annotations_test.json（或 annotations_all.json）VGG 格式，regions 非空为缺陷；TRAIN/ + annotations_train.json 作 support。"""
    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}
    category = "default"
    test_json = dataset_root / "annotations_test.json"
    if not test_json.exists():
        test_json = dataset_root / "annotations_all.json"
    train_json = dataset_root / "annotations_train.json"
    test_dir = dataset_root / "TEST"
    if not test_dir.exists():
        test_dir = dataset_root / "test"
    if not test_json.exists():
        return samples, category_meta
    try:
        with open(test_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return samples, category_meta
    good_paths: List[Path] = []
    defect_paths: List[Path] = []
    train_good_paths: List[Path] = []
    for _key, item in data.items():
        if not isinstance(item, dict):
            continue
        fn = item.get("filename") or item.get("file_name", "")
        if not fn:
            continue
        img_path = _br35hdet_resolve_image_path(dataset_root, fn)
        if img_path is None:
            continue
        regions = item.get("regions", [])
        if regions:
            defect_paths.append(img_path)
        else:
            good_paths.append(img_path)
    train_dir = dataset_root / "TRAIN" if (dataset_root / "TRAIN").exists() else dataset_root / "train"
    if train_dir.exists() and train_json.exists():
        try:
            with open(train_json, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            for _key, item in train_data.items():
                if not isinstance(item, dict):
                    continue
                fn = item.get("filename") or item.get("file_name", "")
                if not fn:
                    continue
                img_path = train_dir / fn
                if img_path.exists() and not item.get("regions", []):
                    train_good_paths.append(img_path)
        except Exception:
            pass
    support_good = train_good_paths if train_good_paths else good_paths
    if max_samples_per_subtype is not None:
        good_paths = good_paths[: max_samples_per_subtype]
        defect_paths = defect_paths[: max_samples_per_subtype]
    category_meta[category] = {
        "defect_types": ["defect"],
        "support_good_paths": support_good,
        "source": "br35hdet",
    }
    for path in good_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "good"})
    for path in defect_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "defect"})
    return samples, category_meta


def discover_brainmri(
    dataset_root: Path,
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """BrainMRI: img_dir/test + ann_dir/test，mask 全黑为 normal，否则为 defect。"""
    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}
    category = "default"
    img_test = dataset_root / "img_dir" / "test"
    ann_test = dataset_root / "ann_dir" / "test"
    if not img_test.exists() or not ann_test.exists():
        return samples, category_meta
    good_paths: List[Path] = []
    defect_paths: List[Path] = []
    for img_path in list_images(img_test):
        name = img_path.name
        ann_path = ann_test / name
        if not ann_path.exists():
            defect_paths.append(img_path)
            continue
        try:
            mask = Image.open(ann_path).convert("L")
            data = list(mask.getdata())
            has_positive = max(data) > 0 if data else False
            if has_positive:
                defect_paths.append(img_path)
            else:
                good_paths.append(img_path)
        except Exception:
            defect_paths.append(img_path)
    support_good = list_images(dataset_root / "img_dir" / "train") or good_paths
    if max_samples_per_subtype is not None:
        good_paths = good_paths[: max_samples_per_subtype]
        defect_paths = defect_paths[: max_samples_per_subtype]
    category_meta[category] = {
        "defect_types": ["defect"],
        "support_good_paths": support_good,
        "source": "brainmri",
    }
    for path in good_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "good"})
    for path in defect_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "defect"})
    return samples, category_meta


def discover_headct(
    dataset_root: Path,
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Head-CT: Dataset/labels.csv (id, hemorrhage)，Dataset/head_ct/{id:03d}.png，hemorrhage 0=good 1=defect。"""
    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}
    category = "default"
    labels_path = dataset_root / "Dataset" / "labels.csv"
    img_dir = dataset_root / "Dataset" / "head_ct"
    if not labels_path.exists() or not img_dir.exists():
        return samples, category_meta
    good_paths: List[Path] = []
    defect_paths: List[Path] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            id_s = (row.get("id") or row.get("ID") or "").strip()
            try:
                id_int = int(id_s)
            except ValueError:
                continue
            hemorrhage = (
                row.get("hemorrhage") or row.get(" hemorrhage") or row.get("label") or "0"
            ).strip()
            is_anomaly = hemorrhage in ("1", "true", "yes")
            for fmt in (f"{id_int:03d}.png", f"{id_int:03d}.jpg", f"{id_int}.png", f"{id_int}.jpg"):
                p = img_dir / fmt
                if p.exists():
                    if is_anomaly:
                        defect_paths.append(p)
                    else:
                        good_paths.append(p)
                    break
    if max_samples_per_subtype is not None:
        good_paths = good_paths[: max_samples_per_subtype]
        defect_paths = defect_paths[: max_samples_per_subtype]
    category_meta[category] = {
        "defect_types": ["defect"],
        "support_good_paths": good_paths,
        "source": "headct",
    }
    for path in good_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "good"})
    for path in defect_paths:
        samples.append({"category": category, "image_path": str(path), "ground_truth_label": "defect"})
    return samples, category_meta


def try_read_visa_csv(dataset_root: Path) -> Optional[List[Dict[str, str]]]:
    candidates = [
        dataset_root / "split_csv" / "1cls.csv",
        dataset_root / "split_csv" / "all.csv",
        dataset_root / "split_csv" / "2cls.csv",
        dataset_root / "1cls.csv",
    ]
    csv_path = None
    for path in candidates:
        if path.exists():
            csv_path = path
            break
    if csv_path is None:
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(x) for x in reader]


def try_read_visa_per_category_csv(dataset_root: Path) -> Optional[List[Dict[str, str]]]:
    """VisA 每类一个 image_anno.csv 的布局：<dataset_root>/<category>/image_anno.csv，列 image, label, mask。"""
    rows: List[Dict[str, str]] = []
    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        csv_path = cat_dir / "image_anno.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    image_rel = str(r.get("image", r.get("img_path", r.get("path", "")))).strip()
                    if not image_rel:
                        continue
                    label = str(r.get("label", r.get("anomaly", "normal"))).strip()
                    rows.append({
                        "category": cat_dir.name,
                        "image": image_rel,
                        "label": label,
                        "split": "test",
                    })
        except Exception:
            continue
    return rows if len(rows) > 0 else None


def get_any_key(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    lower_map = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        rk = lower_map.get(key.lower())
        if rk is not None:
            return row.get(rk, default)
    return default


def normalize_binary_label(raw_label: Any) -> str:
    if raw_label is None:
        return "good"
    s = str(raw_label).strip().lower()
    # VisA 等數據集中常見用 0/1 標記：0=正常, 1=異常
    if s in GOOD_ALIASES or s in {"0", "0.0"}:
        return "good"
    if s in {"1", "1.0"} or s in {"anomaly", "abnormal", "defect", "bad", "ng"}:
        return "anomaly"
    return str(raw_label).strip()


def to_abs_image_path(dataset_root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_absolute():
        return p
    c1 = dataset_root / p
    if c1.exists():
        return c1
    c2 = dataset_root / "images" / p
    if c2.exists():
        return c2
    return c1


def discover_visa(
    dataset_root: Path,
    allowed_categories: Optional[List[str]],
    max_samples_per_subtype: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = try_read_visa_csv(dataset_root)
    if rows is None:
        rows = try_read_visa_per_category_csv(dataset_root)
    if rows is None:
        return discover_mvtec_like(dataset_root, allowed_categories, max_samples_per_subtype)

    tests: List[Dict[str, Any]] = []
    trains: List[Dict[str, Any]] = []
    allow = set(allowed_categories) if allowed_categories is not None else None

    for row in rows:
        category = str(get_any_key(row, ["object", "category", "cls", "class"], "")).strip()
        if not category:
            continue
        if allow is not None and category not in allow:
            continue

        split = str(get_any_key(row, ["split", "phase", "set"], "")).strip().lower()
        label = normalize_binary_label(get_any_key(row, ["label", "anomaly", "target"], "good"))
        image_rel = str(get_any_key(row, ["image", "img_path", "path"], "")).strip()
        if not image_rel:
            continue

        item = {
            "category": category,
            "image_path": str(to_abs_image_path(dataset_root, image_rel)),
            "ground_truth_label": label,
        }
        if split == "test":
            tests.append(item)
        elif split == "train":
            trains.append(item)

    tests_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    trains_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for x in tests:
        tests_by_cat.setdefault(x["category"], []).append(x)
    for x in trains:
        trains_by_cat.setdefault(x["category"], []).append(x)

    samples: List[Dict[str, Any]] = []
    category_meta: Dict[str, Dict[str, Any]] = {}

    for category, cat_tests in sorted(tests_by_cat.items(), key=lambda x: x[0]):
        support_good_paths = [
            Path(x["image_path"]) for x in trains_by_cat.get(category, []) if normalize_binary_label(x["ground_truth_label"]) == "good"
        ]
        if len(support_good_paths) == 0:
            support_good_paths = [
                Path(x["image_path"]) for x in cat_tests if normalize_binary_label(x["ground_truth_label"]) == "good"
            ]

        defect_types = sorted(
            {
                normalize_binary_label(x["ground_truth_label"])
                for x in cat_tests
                if normalize_binary_label(x["ground_truth_label"]) != "good"
            }
        )
        if len(defect_types) == 0:
            defect_types = ["anomaly"]

        category_meta[category] = {
            "defect_types": defect_types,
            "support_good_paths": support_good_paths,
            "source": "visa_csv",
        }

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for x in cat_tests:
            subtype = normalize_binary_label(x["ground_truth_label"])
            buckets.setdefault(subtype, []).append(x)
        for _, items in buckets.items():
            if max_samples_per_subtype is not None:
                items = items[: max_samples_per_subtype]
            samples.extend(items)

    return samples, category_meta


def discover_dataset(
    dataset_root: Path,
    dataset_type: str,
    allowed_categories: Optional[List[str]],
    max_samples_per_subtype: Optional[int],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if dataset_type == "mvtec":
        s, m = discover_mvtec_like(dataset_root, allowed_categories, max_samples_per_subtype)
        return "mvtec", s, m
    if dataset_type == "visa":
        s, m = discover_visa(dataset_root, allowed_categories, max_samples_per_subtype)
        return "visa", s, m
    if dataset_type == "flat_binary":
        s, m = discover_flat_binary(dataset_root, max_samples_per_subtype)
        return "flat_binary", s, m
    if dataset_type == "br35hdet":
        s, m = discover_br35hdet(dataset_root, max_samples_per_subtype)
        return "br35hdet", s, m
    if dataset_type == "brainmri":
        s, m = discover_brainmri(dataset_root, max_samples_per_subtype)
        return "brainmri", s, m
    if dataset_type == "headct":
        s, m = discover_headct(dataset_root, max_samples_per_subtype)
        return "headct", s, m

    # auto: 依次尝试 VisA CSV → VisA 每类 image_anno.csv → 扁平二分类 → Br35HDet → BrainMRI → Head-CT → MVTec 型
    rows = try_read_visa_csv(dataset_root)
    if rows is not None:
        s, m = discover_visa(dataset_root, allowed_categories, max_samples_per_subtype)
        return "visa", s, m
    rows = try_read_visa_per_category_csv(dataset_root)
    if rows is not None:
        s, m = discover_visa(dataset_root, allowed_categories, max_samples_per_subtype)
        return "visa", s, m
    s, m = discover_flat_binary(dataset_root, max_samples_per_subtype)
    if len(s) > 0:
        return "flat_binary", s, m
    s, m = discover_br35hdet(dataset_root, max_samples_per_subtype)
    if len(s) > 0:
        return "br35hdet", s, m
    s, m = discover_brainmri(dataset_root, max_samples_per_subtype)
    if len(s) > 0:
        return "brainmri", s, m
    s, m = discover_headct(dataset_root, max_samples_per_subtype)
    if len(s) > 0:
        return "headct", s, m
    s, m = discover_mvtec_like(dataset_root, allowed_categories, max_samples_per_subtype)
    return "mvtec", s, m


def choose_support_paths(pool: List[Path], shot: int, rng: random.Random) -> List[str]:
    if shot <= 0:
        return []
    pool = [p for p in pool if p.exists()]
    if len(pool) == 0:
        return []
    if len(pool) >= shot:
        return [str(p) for p in rng.sample(pool, shot)]
    return [str(rng.choice(pool)) for _ in range(shot)]


def resolve_token_id_for_word(tokenizer, word: str) -> Optional[int]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 0 and not word.startswith(" "):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 0:
        return None
    return int(ids[0])


def build_generation_kwargs(args: argparse.Namespace, early_exit_layers: List[int], tokenizer) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "num_beams": args.num_beams,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if args.use_deco:
        kwargs.update(
            {
                "use_deco": True,
                "alpha": args.alpha,
                "threshold_top_p": args.threshold_top_p,
                "threshold_top_k": args.threshold_top_k,
                "early_exit_layers": early_exit_layers,
                "anchor_strategy": args.anchor_strategy,
                "lambda_norm": args.lambda_norm,
                "eta_hedge": args.eta_hedge,
                "use_logsumexp": args.use_logsumexp,
                "gamma": args.gamma,
                "enable_hedge": args.enable_hedge,
                "anchor_defect_words": args.anchor_defect_words,
                "anchor_norm_words": args.anchor_norm_words,
                "anchor_hedge_words": args.anchor_hedge_words,
                "deco_debug": args.deco_debug,
                "tokenizer": tokenizer,
            }
        )
    else:
        kwargs["use_deco"] = False
    return {k: v for k, v in kwargs.items() if v is not None}


def compute_anomaly_score(
    probs: torch.Tensor,
    good_token_id: int,
    defect_token_ids: List[int],
    score_mode: str,
) -> Tuple[float, float, float]:
    p_good = float(probs[good_token_id].item())
    if len(defect_token_ids) > 0:
        p_def_max = max(float(probs[i].item()) for i in defect_token_ids)
        p_def_sum = sum(float(probs[i].item()) for i in defect_token_ids)
    else:
        p_def_max = 0.0
        p_def_sum = 0.0

    if score_mode == "one_minus_good":
        score = 1.0 - p_good
    elif score_mode == "max_defect":
        score = p_def_max
    elif score_mode == "defect_minus_good":
        score = p_def_max - p_good
    elif score_mode == "defect_over_sum":
        denom = p_def_sum + p_good
        score = (p_def_sum / denom) if denom > 0 else 0.0
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    return float(score), float(p_good), float(p_def_max)


def binary_auroc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score length mismatch")
    if len(y_true) == 0:
        return None

    pos = sum(1 for y in y_true if int(y) == 1)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None

    indexed = list(enumerate(y_score))
    indexed.sort(key=lambda x: float(x[1]))

    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and float(indexed[j + 1][1]) == float(indexed[i][1]):
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    sum_rank_pos = sum(ranks[idx] for idx, y in enumerate(y_true) if int(y) == 1)
    auc = (sum_rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.analysis_step < 0:
        raise ValueError("--analysis_step must be >= 0")
    if args.max_new_tokens <= args.analysis_step:
        raise ValueError("--max_new_tokens must be > --analysis_step")
    if args.shot < 0:
        raise ValueError("--shot must be >= 0")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    helper_module = load_eval_helpers(Path(__file__).with_name("eval_mvtec_qwen3_vl_thinking.py"))

    allowed_categories = resolve_selected_categories(category=args.category, categories=args.categories)

    resolved_dataset_type, samples, category_meta = discover_dataset(
        dataset_root=dataset_root,
        dataset_type=args.dataset_type,
        allowed_categories=allowed_categories,
        max_samples_per_subtype=args.max_samples_per_subtype,
    )
    discovered_categories = sorted(category_meta.keys())
    if allowed_categories is not None:
        missing = [x for x in allowed_categories if x not in set(discovered_categories)]
        if len(missing) > 0:
            raise ValueError(
                f"Requested categories not found: {missing}. "
                f"Available categories: {discovered_categories}"
            )
    if len(samples) == 0:
        raise RuntimeError("No samples found. Check dataset layout and category filters.")

    early_exit_layers = list(range(args.start_layer, args.end_layer))
    if args.use_deco and len(early_exit_layers) == 0:
        raise ValueError("When --use_deco is set, --start_layer must be smaller than --end_layer.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = processor.tokenizer

    good_token_id = resolve_token_id_for_word(tokenizer, "good")
    if good_token_id is None:
        raise RuntimeError("Failed to resolve token id for 'good'.")

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

    # 与模型实际所在设备一致，避免 device_map="auto" 将模型放到 CPU 时 inputs 仍在 cuda
    model_device = next(model.parameters()).device

    rng = random.Random(args.seed)
    support_cache: Dict[str, List[str]] = {}
    messages_cache: Dict[str, List[Dict[str, Any]]] = {}
    defect_types_cache: Dict[str, List[str]] = {}
    defect_token_ids_cache: Dict[str, List[int]] = {}

    for category, meta in category_meta.items():
        support_cache[category] = choose_support_paths(meta.get("support_good_paths", []), args.shot, rng)

        defect_types = [str(x).strip() for x in meta.get("defect_types", []) if str(x).strip()]
        if len(defect_types) == 0:
            defect_types = ["anomaly"]
        defect_types_cache[category] = defect_types

        messages_cache[category] = helper_module.build_messages(
            category=category,
            defect_types=defect_types,
            shot=args.shot,
        )

        token_ids = []
        for d in defect_types:
            tid = resolve_token_id_for_word(tokenizer, d)
            if tid is not None:
                token_ids.append(int(tid))
        defect_token_ids_cache[category] = sorted(set(token_ids))

    generation_kwargs = build_generation_kwargs(args, early_exit_layers, tokenizer)

    sample_plan_by_category: Dict[str, Dict[str, int]] = {}
    for s in samples:
        cat = str(s["category"])
        gt = helper_module.map_label(str(s["ground_truth_label"]), defect_types_cache.get(cat, []))
        item = sample_plan_by_category.setdefault(cat, {"total": 0, "anomaly": 0, "good": 0})
        item["total"] += 1
        if gt == "good":
            item["good"] += 1
        else:
            item["anomaly"] += 1

    print("===== Image-level AUROC Evaluation =====")
    print(f"dataset_root: {dataset_root}")
    print(f"resolved_dataset_type: {resolved_dataset_type}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"num_categories: {len(category_meta)}")
    print(f"categories: {sorted(category_meta.keys())}")
    print(f"num_samples: {len(samples)}")
    print(f"shot: {args.shot}")
    print(f"score_mode: {args.score_mode}")
    print(f"analysis_step: {args.analysis_step}")
    print(f"use_deco: {args.use_deco}")
    if args.use_deco:
        print(f"deco_alpha: {args.alpha}")
        print(f"deco_threshold_top_p: {args.threshold_top_p}")
        print(f"deco_threshold_top_k: {args.threshold_top_k}")
        print(f"deco_early_exit_layers: {early_exit_layers}")
        print(f"deco_anchor_strategy: {args.anchor_strategy}")
    print("----- Category Sample Plan -----")
    for cat in sorted(sample_plan_by_category.keys()):
        stat = sample_plan_by_category[cat]
        support_count = len(support_cache.get(cat, []))
        print(
            f"[{cat}] total={stat['total']} anomaly={stat['anomaly']} good={stat['good']} "
            f"support_images={support_count}"
        )
    print("========================================")

    results: List[Dict[str, Any]] = []
    progress = tqdm(samples, total=len(samples), desc="Evaluating AUROC", ncols=120, leave=True, mininterval=0.2)

    if device == "cuda":
        torch.cuda.empty_cache()

    for idx, sample in enumerate(progress, start=1):
        category = str(sample["category"])
        image_path = Path(sample["image_path"])
        gt_raw = str(sample["ground_truth_label"])
        gt_norm = helper_module.map_label(gt_raw, defect_types_cache[category])
        gt_is_anomaly = 0 if gt_norm == "good" else 1

        if not image_path.exists():
            results.append(
                {
                    "index": idx,
                    "category": category,
                    "image_path": str(image_path),
                    "ground_truth_label_raw": gt_raw,
                    "ground_truth_label_normalized": gt_norm,
                    "is_anomaly_gt": gt_is_anomaly,
                    "error": "image_not_found",
                }
            )
            continue

        query_image = Image.open(image_path).convert("RGB")
        support_paths = support_cache[category]
        if support_paths:
            support_images = [Image.open(p).convert("RGB") for p in support_paths]
        else:
            # Chat template has shot support + 1 query placeholders; processor expects len(images)==shot+1.
            # When no good samples (e.g. Br35HDet all anomaly), pad with query image to avoid IndexError.
            support_images = [query_image] * args.shot
        all_images = support_images + [query_image]

        text = processor.apply_chat_template(messages_cache[category], tokenize=False, add_generation_prompt=True)
        if args.assistant_prefix:
            text = text + args.assistant_prefix

        inputs = processor(
            text=[text],
            images=all_images,
            return_tensors="pt",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_token_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            output_dict = model.generate(**inputs, **generation_kwargs)

        scores = getattr(output_dict, "scores", None)
        if scores is None or len(scores) == 0:
            raise RuntimeError("No generation scores found; ensure output_scores=True.")
        if args.analysis_step >= len(scores):
            raise ValueError(
                f"analysis_step={args.analysis_step} out of range for sample with {len(scores)} generated steps."
            )

        step_scores = scores[args.analysis_step][0].float()
        probs = F.softmax(step_scores, dim=-1)
        anomaly_score, p_good, p_defect_max = compute_anomaly_score(
            probs=probs,
            good_token_id=good_token_id,
            defect_token_ids=defect_token_ids_cache.get(category, []),
            score_mode=args.score_mode,
        )

        gen_ids = output_dict.sequences[:, input_token_len:]
        raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parse_input_text = f"{args.assistant_prefix}{raw_text}" if args.assistant_prefix else raw_text
        parsed, parse_source = helper_module.parse_model_json(parse_input_text)
        prediction = helper_module.normalize_prediction(
            parsed=parsed,
            parse_source=parse_source,
            raw_text=parse_input_text,
            defect_types=defect_types_cache[category],
            image_size=query_image.size,
        )

        results.append(
            {
                "index": idx,
                "category": category,
                "image_path": str(image_path),
                "ground_truth_label_raw": gt_raw,
                "ground_truth_label_normalized": gt_norm,
                "is_anomaly_gt": gt_is_anomaly,
                "anomaly_score": anomaly_score,
                "p_good": p_good,
                "p_defect_max": p_defect_max,
                "prediction_label": prediction["label"],
                "parse_source": prediction.get("parse_source"),
                "strict_parse_ok": bool(prediction.get("strict_parse_ok", False)),
                "extraction_ok": bool(prediction.get("extraction_ok", False)),
            }
        )

        if args.print_image_name:
            progress.write(
                f"[{idx}/{len(samples)}] {image_path.name} gt={gt_norm} score={anomaly_score:.6f} pred={prediction['label']}"
            )
        progress.set_postfix({"cat": category, "score": f"{anomaly_score:.4f}"}, refresh=False)

        del inputs, output_dict
        if device == "cuda":
            torch.cuda.empty_cache()

    valid = [x for x in results if "error" not in x]
    by_category: Dict[str, Dict[str, Any]] = {}
    for item in valid:
        cat = str(item["category"])
        y = int(item["is_anomaly_gt"])
        s = float(item["anomaly_score"])
        slot = by_category.setdefault(cat, {"y_true": [], "y_score": [], "count": 0, "pos": 0, "neg": 0})
        slot["y_true"].append(y)
        slot["y_score"].append(s)
        slot["count"] += 1
        slot["pos"] += 1 if y == 1 else 0
        slot["neg"] += 1 if y == 0 else 0

    per_category: Dict[str, Any] = {}
    macro_vals: List[float] = []
    for cat, stats in sorted(by_category.items(), key=lambda x: x[0]):
        auc = binary_auroc(stats["y_true"], stats["y_score"])
        per_category[cat] = {
            "num_samples": stats["count"],
            "num_anomaly": stats["pos"],
            "num_good": stats["neg"],
            "image_auroc": auc,
            "defect_types": defect_types_cache.get(cat, []),
            "support_images": support_cache.get(cat, []),
        }
        if auc is not None:
            macro_vals.append(float(auc))

    macro_auc = (sum(macro_vals) / len(macro_vals)) if len(macro_vals) > 0 else None

    strict_parse_rate = (
        sum(1 for x in valid if bool(x.get("strict_parse_ok", False))) / len(valid)
        if len(valid) > 0
        else None
    )
    extraction_rate = (
        sum(1 for x in valid if bool(x.get("extraction_ok", False))) / len(valid)
        if len(valid) > 0
        else None
    )
    error_count = len(samples) - len(valid)

    output = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": str(dataset_root),
            "resolved_dataset_type": resolved_dataset_type,
            "model_name_or_path": args.model_name_or_path,
            "categories": sorted(by_category.keys()),
            "num_samples_total": len(samples),
            "num_samples_valid": len(valid),
            "num_samples_error": len(samples) - len(valid),
            "shot": args.shot,
            "assistant_prefix": args.assistant_prefix,
            "score_mode": args.score_mode,
            "analysis_step": args.analysis_step,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "num_beams": args.num_beams,
            "seed": args.seed,
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "allow_attn_fallback": args.allow_attn_fallback,
            "use_deco": args.use_deco,
            "deco_alpha": args.alpha if args.use_deco else None,
            "deco_threshold_top_p": args.threshold_top_p if args.use_deco else None,
            "deco_threshold_top_k": args.threshold_top_k if args.use_deco else None,
            "deco_early_exit_layers": early_exit_layers if args.use_deco else None,
            "deco_anchor_strategy": args.anchor_strategy if args.use_deco else None,
            "deco_lambda_norm": args.lambda_norm if args.use_deco else None,
            "deco_eta_hedge": args.eta_hedge if args.use_deco else None,
            "deco_enable_hedge": args.enable_hedge if args.use_deco else None,
            "deco_use_logsumexp": args.use_logsumexp if args.use_deco else None,
            "deco_gamma": args.gamma if args.use_deco else None,
            "deco_anchor_defect_words": args.anchor_defect_words if args.use_deco else None,
            "deco_anchor_norm_words": args.anchor_norm_words if args.use_deco else None,
            "deco_anchor_hedge_words": args.anchor_hedge_words if args.use_deco else None,
            "deco_debug": args.deco_debug if args.use_deco else None,
        },
        "summary": {
            "macro_image_auroc": macro_auc,
            "overall_num_samples": len(valid),
            "overall_num_anomaly": sum(int(x["is_anomaly_gt"]) for x in valid),
            "overall_num_good": sum(1 - int(x["is_anomaly_gt"]) for x in valid),
            "strict_parse_success_rate": strict_parse_rate,
            "extraction_success_rate": extraction_rate,
            "per_category": per_category,
        },
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("===== Category-level AUROC Result =====")
    print(f"num_valid_samples: {len(valid)}")
    print(f"num_error_samples: {error_count}")
    print(f"strict_parse_success_rate: {strict_parse_rate}")
    print(f"extraction_success_rate: {extraction_rate}")
    print(f"macro_image_auroc: {macro_auc}")
    print("----- Per-category AUROC -----")
    no_auc_categories: List[str] = []
    for cat in sorted(per_category.keys()):
        item = per_category[cat]
        auc = item["image_auroc"]
        if auc is None:
            no_auc_categories.append(cat)
        print(
            f"[{cat}] image_auroc={auc} total={item['num_samples']} "
            f"anomaly={item['num_anomaly']} good={item['num_good']}"
        )
    if len(no_auc_categories) > 0:
        print(f"[WARN] AUROC unavailable for categories with single-class samples: {no_auc_categories}")
    print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()
