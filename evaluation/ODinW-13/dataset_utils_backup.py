"""
ODinW dataset loading and processing utilities.
"""
import os
import math
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
import glob
import random  # 新增
import numpy as np # 新增
def round_by_factor(number: int, factor: int) -> int:
    """Return the nearest integer divisible by factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the ceiling integer divisible by factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the floor integer divisible by factor."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 28, 
                 min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280, 
                 max_long_side: int = 8192) -> Tuple[int, int]:
    """Resize image to meet the following conditions:
        1. Both height and width are divisible by factor
        2. Total pixels are within [min_pixels, max_pixels]
        3. Longest side is within max_long_side
        4. Aspect ratio is preserved
    
    Args:
        height: Original image height
        width: Original image width
        factor: Size must be divisible by this factor
        min_pixels: Minimum pixel count
        max_pixels: Maximum pixel count
        max_long_side: Maximum longest side
    
    Returns:
        (resized_height, resized_width): Resized dimensions
    """
    if height < 2 or width < 2:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio must be smaller than 200, got {height} / {width}')

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def load_odinw_config(config_path: str) -> Dict:
    """Load odinw13_config.py configuration file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        datasets: Dictionary mapping dataset names to configurations
    """
    import runpy
    config = runpy.run_path(config_path)
    dataset_configs = config["datasets"]
    dataset_names = config["dataset_prefixes"]
    
    datasets = {}
    for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
        datasets[dataset_name] = dataset_config
    
    return datasets


def generate_odinw_jobs(data_dir: str, args) -> Tuple[List[Dict], Dict]:

    """Generate inference task list for ODinW dataset.
    
    Args:
        data_dir: Data directory path (containing odinw13_config.py)
        args: Command line arguments
    
    Returns:
        (question_list, datasets): Task list and dataset configurations
    """
    # 1. 获取外部参数
    # 假设 args 中通过命令行传入了这两个参数，如果没有则设为默认值
    ref_img_dir = getattr(args, 'ref_img_dir', None)
    num_shots = getattr(args, 'num_shots', 1) # 默认为 1-shot

    # 2. 预加载参考图片列表
    ref_images_pool = []
    if ref_img_dir and os.path.exists(ref_img_dir):
        # 支持常见的图片格式
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in exts:
            ref_images_pool.extend(glob.glob(os.path.join(ref_img_dir, ext)))
            # 同时也查找大写后缀
            ref_images_pool.extend(glob.glob(os.path.join(ref_img_dir, ext.upper())))
        
        print(f"Found {len(ref_images_pool)} reference images in {ref_img_dir}")
        if len(ref_images_pool) == 0:
            print("Warning: No images found in reference directory! Falling back to 0-shot.")
    elif ref_img_dir:
        print(f"Warning: Reference directory {ref_img_dir} does not exist!")



    # Load config
    config_path = os.path.join(data_dir, "odinw13_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    datasets = load_odinw_config(config_path)
    
    question_list = []
    question_id = 0
    num_questions_per_dataset = {}
    
    # Calculate image resolution parameters
    patch_size = 16
    merge_base = 2
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    min_pixels = pixels_per_token * 768
    max_pixels = pixels_per_token * 12800
    
    # Iterate through all datasets
    for data_name, data_config in datasets.items():
        print(f'Parsing ODinW:{data_name}')
        classes = list(data_config["metainfo"]["classes"])
        
        # Build data paths
        idx = data_config["data_root"].find('data/odinw/') + len('data/odinw/')
        sub_root = os.path.join(data_dir, data_config["data_root"][idx:])
        sub_anno = sub_root + data_config["ann_file"]
        sub_img_root = sub_root + data_config["data_prefix"]["img"]
        
        # Load COCO format annotations
        dataset = COCO(sub_anno)
        num_questions = 0
        
        # Iterate through all images
        for img_idx, img_meta in dataset.imgs.items():
            img_name = img_meta["file_name"]
            img_path = sub_img_root + img_name
            img_h = img_meta["height"]
            img_w = img_meta["width"]
            
            # Calculate resized image dimensions
            resized_h, resized_w = smart_resize(
                img_h, img_w, 
                factor=32, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels, 
                max_long_side=50000
            )
            
            # Get annotations
            img_annos = dataset.imgToAnns[img_idx]
            
            # Build class names list
            obj_names = ", ".join(classes)
            
            # Build prompt
            prompt = f"Locate every instance that belongs to the following categories: '{obj_names}'. Report bbox coordinates in JSON format. If it is not good, analyze the cause of the defect and provide corresponding solutions"
            
            messages = []
            
            # --- 1-Shot / Few-Shot 逻辑 ---
            if ref_images_pool and num_shots > 0:
                # 随机选取指定数量的参考图
                # 如果池子里的图不够，就允许重复选取
                if len(ref_images_pool) >= num_shots:
                    selected_refs = random.sample(ref_images_pool, num_shots)
                else:
                    selected_refs = random.choices(ref_images_pool, k=num_shots)
                
                for ref_path in selected_refs:
                    # 第一轮：User 输入正常图片
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image", 
                                "image": f"file://{ref_path}", 
                                "min_pixels": min_pixels, 
                                "max_pixels": max_pixels
                            },
                            {"type": "text", "text": prompt}
                        ]
                    })
                    
                    # 第一轮：Assistant 回答
                    # 关键点：假设“正常图片”不包含目标，返回空列表
                    # 这告诉模型：“这种样子的图片，什么都不要框选。”
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": "[]"}]
                    })


            # Build messages
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{img_path}", 
                            "min_pixels": min_pixels, 
                            "max_pixels": max_pixels
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            )
            
            # Build task item
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
    
    # Print statistics
    for data_name, num_questions in num_questions_per_dataset.items():
        print(f'{data_name}: {num_questions}')
    print(f"Total ODinW questions: {len(question_list)}")
    
    return question_list, datasets

