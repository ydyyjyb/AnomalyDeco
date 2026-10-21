import json
import re
import os
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm  # 导入进度条库

# --- 假设你已经安装并配置好了 SAM2 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===========================
# 配置区域
# ===========================
# SAM2 模型权重和配置文件路径
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
# 数据集根目录 (重要：请修改为你本地实际存放 data 文件夹的上一级目录)
# 例如，如果你的图片位于 D:/datasets/mvtec/data/bottle1/images/...
# 那么这里应该填写 D:/datasets/mvtec
DATA_ROOT_PATH = r"/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13"

# JSON 文件路径
JSON_FILE_PATH = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/results/input.json"

# 输出保存目录
OUTPUT_DIR = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/bottle1/viz_results"

# 可视化控制开关 (Request 4)
SHOW_ROI_BOX = True      # 是否在保存的图中画出输入给 SAM2 的矩形框 (绿色)
SHOW_SEG_CONTOUR = True  # 是否在保存的图中画出分割后的轮廓 (红色)
# ===========================


def load_predictor():
    """初始化 SAM2 预测器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 注意：build_sam2 的参数顺序可能随版本微调，请确保与你安装的版本一致
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    return SAM2ImagePredictor(sam2_model)

def extract_absolute_boxes(gen_text):
    """从 gen 文本中提取像素坐标 [x1, y1, x2, y2]"""
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', gen_text)
    return [list(map(int, m)) for m in matches]

def draw_and_save(image_np, boxes, masks, save_path):
    """根据开关控制，在原图绘制 ROI 和轮廓并保存"""
    # 转换颜色空间供 OpenCV 使用 (RGB -> BGR)
    vis_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 绘制矩形框 (ROI)
    if SHOW_ROI_BOX and boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色框

    # 绘制分割轮廓
    if SHOW_SEG_CONTOUR and masks:
        for mask in masks:
            # mask 是 bool 矩阵，转为 uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2) # 红色轮廓

    cv2.imwrite(save_path, vis_img)

# ===========================
# 3. 主批处理程序
# ===========================

def run_batch_process():
    # 初始化
    predictor = load_predictor()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 tqdm 包装循环
    # unit="img" 表示单位是图片，desc 是左侧显示的文字描述
    pbar = tqdm(data, desc="Processing Defects", unit="img")

    for item in pbar:
        qid = item.get("question_id")
        rel_path = item["extra_info"]["img_path"]
        gen_text = item["result"]["gen"]

        # 1. 路径拼接 (Request 1)
        # 移除前面的 './' 避免拼接错误
        clean_rel_path = rel_path.lstrip("./")
        full_path = os.path.join(DATA_ROOT_PATH, clean_rel_path)

        # 更新进度条描述信息，显示当前文件名
        pbar.set_postfix(file=os.path.basename(full_path))

        # 2. 检查是否为正常样本
        if '"label": "good"' in gen_text or '"good"' in gen_text.lower():
            continue

        # 3. 提取坐标 (Request 2: 直接作为像素坐标使用)
        boxes = extract_absolute_boxes(gen_text)
        if not boxes:
            continue

        # 4. 读取图片并推理
        if not os.path.exists(full_path):
            continue

        img_pil = Image.open(full_path).convert("RGB")
        img_np = np.array(img_pil)
        predictor.set_image(img_np)

        current_masks = []
        for box in boxes:
            # SAM2 基于 Box 进行分割 (Request 3)
            masks, _, _ = predictor.predict(
                box=np.array(box),
                multimask_output=False
            )
            current_masks.append(masks[0])

        # 5. 可视化并保存 (Request 4)
        save_filename = f"res_{qid}_{os.path.basename(full_path)}"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        draw_and_save(img_np, boxes, current_masks, save_path)

    print(f"\n✅ 所有任务处理完成，结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch_process()