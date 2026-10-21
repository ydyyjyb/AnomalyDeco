import json
import re
import os
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm

# --- SAM2 库导入 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===========================
# 1. 全局配置区域
# ===========================
SAM2_CHECKPOINT = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/sam2-main/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
# 数据集根目录 (重要：请修改为你本地实际存放 data 文件夹的上一级目录)
# 例如，如果你的图片位于 D:/datasets/mvtec/data/bottle1/images/...
# 那么这里应该填写 D:/datasets/mvtec
DATA_ROOT_PATH = r"/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13"

# JSON 文件路径
JSON_FILE_PATH = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/results/bottle1.json"

# 输出保存目录
OUTPUT_DIR = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/bottle1/viz_results"

# 可视化显示控制
SHOW_ROI_BOX = True         # 是否绘制绿色输入框
ALPHA = 0.2                 # 填充区域的透明度 (0为全透明，1为不透明)
CONTOUR_COLOR = (0, 0, 255) # BGR格式：红色 (用于边缘和填充)
ROI_COLOR = (0, 255, 0)     # BGR格式：绿色 (用于ROI框)

# 面积过滤阈值 (Request)
# 如果分割出来的轮廓面积小于这个像素值，将被滤除
MIN_AREA_THRESHOLD = 50     

# ===========================
# 2. 核心绘图与过滤函数
# ===========================

def draw_styled_mask(image_bgr, masks, boxes):
    """
    实现：红色轮廓 + 浅红色半透明填充 + 面积过滤
    """
    # 创建一个与原图等大的遮罩层用于填充
    overlay = image_bgr.copy()
    output = image_bgr.copy()

    # 1. 绘制 ROI 框 (如果开启)
    if SHOW_ROI_BOX and boxes:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), ROI_COLOR, 2)

    # 2. 处理分割掩码
    for mask in masks:
        # 将 mask 转为二进制图
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # 计算面积并过滤 (Request)
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_THRESHOLD:
                continue
            
            # 在 overlay 层上进行填充 (内部区域)
            cv2.drawContours(overlay, [cnt], -1, CONTOUR_COLOR, thickness=cv2.FILLED)
            # 在 output 层上绘制轮廓 (边缘线)
            cv2.drawContours(output, [cnt], -1, CONTOUR_COLOR, thickness=2)

    # 3. 将填充层与原图按透明度融合 (Request)
    # output = output * (1 - ALPHA) + overlay * ALPHA
    cv2.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0, output)
    
    return output

# ===========================
# 3. 主批处理程序 (集成 tqdm)
# ===========================

def run_batch_process():
    # 环境检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device))
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pbar = tqdm(data, desc="Batch Segmenting", unit="img")

    for item in pbar:
        qid = item.get("question_id")
        rel_path = item["extra_info"]["img_path"]
        gen_text = item["result"]["gen"]
        
        # 路径解析
        clean_rel_path = rel_path.lstrip("./")
        full_path = os.path.join(DATA_ROOT_PATH, clean_rel_path)
        pbar.set_postfix(file=os.path.basename(full_path))

        # 过滤正常样本
        if '"label": "good"' in gen_text or '"good"' in gen_text.lower():
            continue

        # 提取坐标 (正则)
        boxes = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', gen_text)
        boxes = [list(map(int, b)) for b in boxes]
        
        if not boxes or not os.path.exists(full_path):
            continue

        # 读取图片
        img_pil = Image.open(full_path).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # SAM2 推理
        predictor.set_image(img_np)
        current_masks = []
        for box in boxes:
            masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
            current_masks.append(masks[0])

        # 样式化渲染 (Request 关键步骤)
        final_viz = draw_styled_mask(img_bgr, current_masks, boxes)

        # 保存
        save_path = os.path.join(OUTPUT_DIR, f"styled_res_{qid}_{os.path.basename(full_path)}")
        cv2.imwrite(save_path, final_viz)

    print(f"\n✨ 处理完成！可视化结果存放在: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_batch_process()