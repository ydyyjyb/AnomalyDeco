import os
import json
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

# VOC 2012 类别列表
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def convert_voc_to_coco(xml_dir, img_dir, json_file, max_images=None):
    """
    Args:
        xml_dir: XML 文件夹路径
        img_dir: 图片文件夹路径
        json_file: 输出的 JSON 文件路径
        max_images: (int) 最大处理图片数量。如果为 None，则处理所有图片。
    """
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. 创建类别 ID 映射
    class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}
    for name, id in class_to_id.items():
        coco_output["categories"].append({
            "id": id,
            "name": name,
            "supercategory": "none"
        })

    # 2. 读取 XML 文件
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    # 为了保证每次取样一致，最好排序一下
    xml_files.sort()
    
    print(f"Found {len(xml_files)} xml files.")
    
    if max_images is not None:
        print(f"Sampling mode: Only processing the first {max_images} images.")
    
    ann_id = 0
    img_id = 0

    for xml_file in tqdm(xml_files):
        # 如果设置了最大数量限制，且已达到限制，则停止
        if max_images is not None and img_id >= max_images:
            break

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
        
        filename = root.find('filename').text
        
        # 检查图片是否存在
        if not os.path.exists(os.path.join(img_dir, filename)):
            # 尝试解决 jpg/JPG 大小写问题
            if os.path.exists(os.path.join(img_dir, filename.lower())):
                 filename = filename.lower()
            else:
                # print(f"Image not found: {filename}")
                continue

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 添加图片信息
        image_info = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_id
        }
        coco_output["images"].append(image_info)

        # 处理 Objects
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_to_id:
                continue
            
            category_id = class_to_id[name]
            bndbox = obj.find('bndbox')
            
            # VOC 格式: [xmin, ymin, xmax, ymax]
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            w = xmax - xmin
            h = ymax - ymin
            
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [] 
            }
            coco_output["annotations"].append(ann)
            ann_id += 1
            
        img_id += 1

    # 3. 保存 JSON
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(coco_output, f)
    print(f"\nProcessing Complete!")
    print(f"Processed Images: {img_id}")
    print(f"Total Annotations: {ann_id}")
    print(f"Saved to {json_file}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 原始 XML 路径 (请修改为你存放 XML 的实际路径)
    #    例如: /path/to/VOCdevkit/VOC2012/Annotations
    VOC_XML_DIR = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/VOC2012/Annotations"  
    
    # 2. 目标图片路径 (修改为我们刚才整理好的路径)
    TARGET_IMG_DIR = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/VOC2012/images"
    
    # 3. 输出 JSON 路径
    OUTPUT_JSON = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/VOC2012/annotations/val.json"
    
    # 4. 【关键修改】限制转换的图片数量
    #    设为 50 表示只取前 50 张图片用于快速测试
    #    设为 None 表示转换所有图片
    MAX_IMAGES = 5 
    
    # ===========================================
    
    convert_voc_to_coco(VOC_XML_DIR, TARGET_IMG_DIR, OUTPUT_JSON, max_images=MAX_IMAGES)