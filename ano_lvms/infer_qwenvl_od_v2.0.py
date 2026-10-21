import torch
import torch.nn.functional as F
import json
import logging
import os
import argparse
from typing import List, Optional, Union
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
# 尝试导入 Qwen3VLForConditionalGeneration，如果失败则使用 AutoModel
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from transformers.generation.logits_process import LogitsProcessorList
import re

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comparison_experiment.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("Qwen3-MVTec-Compare")

class Qwen3Detector:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        logger.info(f"正在加载模型: {model_path} ...")
        self.device = device
        self.dtype = dtype
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2", 
            trust_remote_code=True
        ).eval()

        # 自动定位组件
        self._locate_components()

    def _locate_components(self):
        try:
            # 1. 定位层数
            if hasattr(self.model.config, "text_config"):
                self.num_layers = self.model.config.text_config.num_hidden_layers
            else:
                # 尝试通用属性
                self.num_layers = getattr(self.model.config, "num_hidden_layers", 32)
            
            # 2. 定位 Norm 层 和 LM Head
            # Qwen 结构通常是 model.language_model.norm 或 model.norm
            if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
                self.norm_layer = self.model.model.language_model.norm
            elif hasattr(self.model, "language_model"):
                self.norm_layer = self.model.language_model.norm
            else:
                # 最后的尝试，可能在 model 下直接有 norm
                 self.norm_layer = getattr(self.model, "norm", None)

            self.lm_head = self.model.lm_head
            
            logger.info(f"✅ 组件定位成功: Layers={self.num_layers}")
            
        except Exception as e:
            logger.error(f"组件定位失败: {e}")
            logger.warning("将尝试使用默认配置运行，但 DeCo 可能无法正常工作。")

    def _parse_output(self, text, width, height):
        """
        解析模型输出，提取 defects 信息
        """
        result = {"thought": "", "defects": []}
        
        # 1. 提取思考过程
        if "</think>" in text:
            parts = text.split("</think>")
            result["thought"] = parts[0].replace("<think>", "").strip()
            remaining_text = parts[1]
        else:
            remaining_text = text

        # 2. 提取并清洗 JSON 字符串
        try:
            match = re.search(r'(\{.*\}|\[.*\])', remaining_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                # 移除不可见字符
                json_str = re.sub(r'[^\x00-\x7F]+', ' ', json_str) 
                
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    raw_list = data
                elif isinstance(data, dict) and "defects" in data:
                    raw_list = data["defects"]
                else:
                    raw_list = [data]

                for item in raw_list:
                    label = item.get("category") or item.get("defect") or "unknown"
                    
                    defect_info = {
                        "label": label,
                        "bbox": item.get("bbox", []),
                        "reason": item.get("analysis") or item.get("reason") or "No specific reason provided.",
                        "solution": item.get("solution") or ("No action needed" if label == "good" else "Replacement or repair required.")
                    }

                    # 坐标反归一化
                    if len(defect_info["bbox"]) == 4:
                        y1, x1, y2, x2 = defect_info["bbox"]
                        defect_info["bbox_abs"] = [
                            int((x1 / 1000) * width), int((y1 / 1000) * height),
                            int((x2 / 1000) * width), int((y2 / 1000) * height)
                        ]
                    
                    result["defects"].append(defect_info)

        except Exception as e:
            logger.warning(f"解析失败: {str(e)} | 原始内容片段: {remaining_text[:100]}")
                
        return result

    @torch.inference_mode()
    def generate_with_deco(
        self, 
        inputs, 
        max_new_tokens=512, 
        alpha=2.0,            
        threshold_top_p=0.9,  
        threshold_top_k=50,   
        early_exit_layers=None
    ):
        """
        DeCo/DoLa 生成函数
        """
        input_ids = inputs.input_ids
        batch_size, seq_length = input_ids.shape
        device = self.device
        
        if early_exit_layers is None:
            start_layer = self.num_layers // 2
            early_exit_layers = list(range(start_layer, self.num_layers - 1, 2))
        
        model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
        
        # Qwen3-VL 位置控制参数
        cache_position = torch.arange(seq_length, device=device)
        
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id or eos_token_id
        
        generated_ids = input_ids.clone()
        logits_processor = LogitsProcessorList()

        for step in range(max_new_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, 
                past_key_values=past_key_values,
                cache_position=cache_position, 
                use_cache=True,
                **model_kwargs
            )
            
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            
            next_token_logits = outputs.logits[:, -1, :] 

            # --- DeCo 核心逻辑 ---
            premature_logits_list = []
            for layer_idx in early_exit_layers:
                hidden = outputs.hidden_states[layer_idx + 1][:, -1, :] 
                hidden = self.norm_layer(hidden)
                logits = self.lm_head(hidden) 
                premature_logits_list.append(logits)
            
            stack_premature_logits = torch.stack(premature_logits_list)
            
            last_layer_probs = F.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_ids = torch.topk(last_layer_probs, k=threshold_top_k, dim=-1)
            
            stack_premature_probs = F.softmax(stack_premature_logits, dim=-1)
            expanded_ids = top_k_ids.unsqueeze(0).expand(len(early_exit_layers), -1, -1)
            
            candidate_premature_probs = torch.gather(stack_premature_probs, 2, expanded_ids)
            candidate_premature_logits = torch.gather(stack_premature_logits, 2, expanded_ids)
            
            max_premature_probs, layer_indices = torch.max(candidate_premature_probs, dim=0)
            
            gathered_logits = torch.gather(
                candidate_premature_logits, 
                0, 
                layer_indices.unsqueeze(0)
            ).squeeze(0)
            
            penalty = alpha * max_premature_probs * gathered_logits
            final_candidates_logits = torch.gather(next_token_logits, 1, top_k_ids)
            
            corrected_candidates_logits = final_candidates_logits - penalty
            
            modified_logits = torch.full_like(next_token_logits, float('-inf'))
            modified_logits.scatter_(1, top_k_ids, corrected_candidates_logits)
            
            processed_logits = logits_processor(input_ids, modified_logits)
            next_tokens = torch.argmax(processed_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            
            input_ids = next_tokens[:, None]
            past_key_values = outputs.past_key_values
            cache_position = cache_position[-1:] + 1
            
            if "attention_mask" in model_kwargs:
                mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [mask, mask.new_ones((batch_size, 1))], dim=-1
                )

            generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)

            if unfinished_sequences.max() == 0:
                break
                
        return generated_ids

    @torch.inference_mode()
    def predict_single_image(
        self, 
        image_path: str, 
        use_deco: bool = True, 
        deco_alpha: float = 2.0
    ):
        if not os.path.exists(image_path):
            logger.error(f"文件不存在: {image_path}")
            return None

        prompt = (
            "Locate every instance that belongs to the following categories: good,broken_large,broken_small,contamination. " \
            "If it is not good, Report bbox coordinates in JSON format and analyze the cause of the defect and provide corresponding solutions"
        )

        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]
        
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(self.device)

        # logger.info(f"处理图片: {os.path.basename(image_path)} | Mode: {'DeCo' if use_deco else 'Baseline'}")

        if use_deco:
            output_ids = self.generate_with_deco(
                inputs, 
                max_new_tokens=2048, 
                alpha=deco_alpha, 
                threshold_top_p=0.95,
                threshold_top_k=40
            )
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=2048, use_cache=True)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        with Image.open(image_path) as img:
            w, h = img.size
        parsed = self._parse_output(raw_text, w, h)
        
        return raw_text

def run_comparison(
    model_path, 
    mvtec_root, 
    output_dir="results",
    target_categories=None,
    deco_alpha=2.0
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = Qwen3Detector(model_path)
    
    # 扫描数据集
    image_tasks = []
    if not os.path.exists(mvtec_root):
        logger.error(f"数据集路径不存在: {mvtec_root}")
        return

    all_categories = [d for d in os.listdir(mvtec_root) if os.path.isdir(os.path.join(mvtec_root, d))]
    if target_categories:
        all_categories = [c for c in all_categories if c in target_categories]

    for cat in all_categories:
        test_dir = os.path.join(mvtec_root, cat, "test")
        if not os.path.exists(test_dir): continue
        
        defect_types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        for dtype in defect_types:
            dtype_dir = os.path.join(test_dir, dtype)
            for img_name in os.listdir(dtype_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_tasks.append({
                        "path": os.path.join(dtype_dir, img_name),
                        "category": cat,
                        "gt_label": dtype
                    })

    logger.info(f"共发现 {len(image_tasks)} 张测试图片")
    
    # 运行对比实验
    modes = [("Baseline", False), ("DeCo", True)]
    
    for mode_name, use_deco in modes:
        logger.info(f"\n开始运行 {mode_name} 实验...")
        output_file = os.path.join(output_dir, f"mvtec_result_{mode_name.lower()}.jsonl")
        
        # 如果文件存在，先清空
        if os.path.exists(output_file):
            os.remove(output_file)
            
        for task in tqdm(image_tasks, desc=f"{mode_name} Inference"):
            try:
                res = detector.predict_single_image(
                    task["path"], 
                    use_deco=use_deco,
                    deco_alpha=deco_alpha
                )
                
                if res:
                    record = {
                        "category": task["category"],
                        "file": os.path.basename(task["path"]),
                        "gt": task["gt_label"],
                        "pred": res["parsed"],
                        "raw": res["raw_text"],
                        "mode": mode_name
                    }
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Error processing {task['path']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VL DeCo vs Baseline Comparison on MVTec")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen-VL model")
    parser.add_argument("--mvtec_root", type=str, required=False, help="Path to MVTec dataset root (required for batch comparison)")
    parser.add_argument("--single_image", type=str, help="Path to a single image for testing")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--deco_alpha", type=float, default=2.0, help="Alpha value for DeCo")
    
    args = parser.parse_args()
    
    if args.single_image:
        logger.info(f"Running single image test: {args.single_image}")
        detector = Qwen3Detector(args.model_path)
        
        # Test Baseline
        logger.info("--- Testing Baseline ---")
        res_base = detector.predict_single_image(args.single_image, use_deco=False)
        if res_base:
            print(f"\n[Baseline Result]\n{json.dumps(res_base, indent=2, ensure_ascii=False)}")
            
        # Test DeCo
        logger.info(f"--- Testing DeCo (alpha={args.deco_alpha}) ---")
        res_deco = detector.predict_single_image(args.single_image, use_deco=True, deco_alpha=args.deco_alpha)
        if res_deco:
            print(f"\n[DeCo Result]\n{json.dumps(res_deco, indent=2, ensure_ascii=False)}")
            


    # run_comparison(
    #     args.model_path, 
    #     args.mvtec_root, 
    #     args.output_dir, 
    #     deco_alpha=args.deco_alpha
    # )
