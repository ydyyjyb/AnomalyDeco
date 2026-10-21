"""
ODinW dataset loading and processing utilities for Qwen3-VL.
"""
import os
import math
import random
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pycocotools.coco import COCO


def round_by_factor(number: int, factor: int) -> int:
    """Return the nearest integer divisible by factor."""
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32, # Qwen3-VL usually uses factor=16*2=32 (patch_size * merge_size)
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144, # Based on Qwen3-VL config
    temporal_factor: int = 1, # For images T=1
    num_frames: int = 1
) -> Tuple[int, int]:
    """
    Resize image logic adapted from video_processing_qwen3_vl.py.
    Calculates the expected resized dimensions for metadata purposes.
    """
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
        
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def load_odinw_config(config_path: str) -> Dict:
    """Load odinw13_config.py configuration file."""
    import runpy
    config = runpy.run_path(config_path)
    dataset_configs = config["datasets"]
    dataset_names = config["dataset_prefixes"]
    
    datasets = {}
    for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
        datasets[dataset_name] = dataset_config
    
    return datasets


def get_train_ann_path(val_ann_path: str) -> str:
    """Infer the training annotation path from the validation annotation path."""
    if "test" in val_ann_path:
        return val_ann_path.replace("test", "train")
    if "val" in val_ann_path:
        return val_ann_path.replace("val", "train")
    
    base_dir = os.path.dirname(val_ann_path)
    possible_train = os.path.join(base_dir, "train_annotations.json")
    if os.path.exists(possible_train):
        return possible_train
    
    # Fallback try generic train.json
    return os.path.join(base_dir, "train.json")


def format_few_shot_samples(coco_train: COCO, img_root: str, classes: List[str], num_shots: int) -> List[Dict]:
    """
    Select random few-shot examples and format them as message history.
    Normalizes coordinates to [0, 1000].
    """
    if num_shots <= 0 or coco_train is None:
        return []

    img_ids = list(coco_train.imgs.keys())
    selected_img_ids = random.sample(img_ids, min(num_shots, len(img_ids)))
    
    history_messages = []
    
    for img_id in selected_img_ids:
        img_info = coco_train.imgs[img_id]
        img_path = os.path.join(img_root, img_info['file_name'])
        ann_ids = coco_train.getAnnIds(imgIds=img_id)
        anns = coco_train.loadAnns(ann_ids)
        
        gt_json = []
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = coco_train.cats[cat_id]['name']
            
            x, y, w, h = ann['bbox']
            img_h, img_w = img_info['height'], img_info['width']
            
            # Normalize to 0-1000
            x1 = int((x / img_w) * 1000)
            y1 = int((y / img_h) * 1000)
            x2 = int(((x + w) / img_w) * 1000)
            y2 = int(((y + h) / img_h) * 1000)
            
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            x2 = max(0, min(1000, x2))
            y2 = max(0, min(1000, y2))
            
            gt_json.append({
                "bbox_2d": [x1, y1, x2, y2],
                "label": cat_name
            })
        
        obj_names = ", ".join(classes)
        prompt = f"Locate every instance that belongs to the following categories: '{obj_names}'. Report bbox coordinates in JSON format."

        # User Turn
        history_messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_path}, # Keep raw path for processor
                {"type": "text", "text": prompt}
            ]
        })
        
        # Assistant Turn
        history_messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"```json\n{json.dumps(gt_json)}\n```"}
            ]
        })
        
    return history_messages


def generate_odinw_jobs(data_dir: str, args) -> Tuple[List[Dict], Dict]:
    """Generate inference task list for ODinW dataset."""
    
    config_path = os.path.join(data_dir, "odinw13_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    datasets = load_odinw_config(config_path)
    
    question_list = []
    question_id = 0
    num_questions_per_dataset = {}
    
    # Qwen3-VL resolution parameters (used for metadata calculation)
    # These should match what is passed to processor kwargs
    min_pixels = 128 * 128
    max_pixels = 16 * 16 * 2 * 2 * 2 * 6144
    
    for data_name, data_config in datasets.items():
        print(f'Parsing ODinW:{data_name}')
        classes = list(data_config["metainfo"]["classes"])
        
        idx = data_config["data_root"].find('data/odinw/') + len('data/odinw/')
        sub_root = os.path.join(data_dir, data_config["data_root"][idx:])
        sub_anno = sub_root + data_config["ann_file"]
        sub_img_root = sub_root + data_config["data_prefix"]["img"]
        
        # Few-Shot Preparation
        coco_train = None
        if args.num_shots > 0:
            try:
                train_anno_path = get_train_ann_path(sub_anno)
                if not os.path.exists(train_anno_path):
                     print(f"Warning: Training annotation {train_anno_path} not found. Skipping few-shot for {data_name}.")
                else:
                    coco_train = COCO(train_anno_path)
            except Exception as e:
                print(f"Error loading training data: {e}")
                coco_train = None

        dataset = COCO(sub_anno)
        num_questions = 0
        
        for img_idx, img_meta in dataset.imgs.items():
            img_name = img_meta["file_name"]
            img_path = sub_img_root + img_name
            img_h = img_meta["height"]
            img_w = img_meta["width"]
            
            # Calculate resized dimensions for metadata/evaluation consistency
            # The actual resizing happens in Qwen3VLProcessor, but we need these
            # for the legacy evaluation script if it uses relative-to-resized coords.
            # Qwen3-VL outputs 0-1000 usually, but we keep this for compatibility.
            resized_h, resized_w = smart_resize(
                img_h, img_w, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            )
            
            img_annos = dataset.imgToAnns[img_idx]
            obj_names = ", ".join(classes)
            prompt = f"Locate every instance that belongs to the following categories: '{obj_names}'. Report bbox coordinates in JSON format."
            
            messages = []
            
            # Add Few-Shot History
            if args.num_shots > 0 and coco_train:
                history = format_few_shot_samples(coco_train, sub_img_root, classes, args.num_shots)
                messages.extend(history)
            
            # Add Current Test Image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path}, # Pass path, load later
                    {"type": "text", "text": prompt}
                ]
            })
            
            item = {
                "question_id": question_id,
                "annotation": img_annos,
                'messages': messages,
                "extra_info": {
                    'dataset_name': data_name,
                    'dataset_config': data_config,
                    'img_id': img_meta["id"],
                    'anno_path': sub_anno,
                    'resized_h': resized_h,
                    'resized_w': resized_w,
                    'img_h': img_h,
                    'img_w': img_w,
                    'img_path': img_path
                }
            }
            question_list.append(item)
            question_id += 1
            num_questions += 1
        
        num_questions_per_dataset[data_name] = num_questions
    
    for data_name, num_questions in num_questions_per_dataset.items():
        print(f'{data_name}: {num_questions}')
    print(f"Total ODinW questions: {len(question_list)}")
    
    return question_list, datasets