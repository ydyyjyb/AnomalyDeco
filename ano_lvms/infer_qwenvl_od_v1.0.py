import torch
import torch.nn.functional as F
import json
import logging
import os
from typing import List, Optional, Union
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation.logits_process import LogitsProcessorList
import re
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mvtec_inference.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("Qwen3-MVTec")

class Qwen3Detector:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        logger.info(f"正在加载模型: {model_path} ...")
        self.device = device
        self.dtype = dtype
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2", 
            trust_remote_code=True
        ).eval()

        # # ------------------- [新增调试代码 开始] -------------------
        # logger.info("\n" + "="*20 + " 模型结构深度探测 " + "="*20)
        
        # # 1. 打印 Config 结构 (寻找 num_hidden_layers)
        # logger.info(f"Config 类型: {type(self.model.config)}")
        # if hasattr(self.model.config, "text_config"):
        #     logger.info("发现 'text_config'！尝试读取其中的层数...")
        #     logger.info(f"Text Config Keys: {list(self.model.config.text_config.__dict__.keys())[:10]}...") 
        # else:
        #     logger.info("未发现 'text_config'，打印根 Config Keys:")
        #     logger.info(list(self.model.config.__dict__.keys())[:10])

        # # 2. 打印模型主干组件 (寻找 norm 和 model/language_model)
        # logger.info("模型直属子模块:")
        # for name, module in self.model.named_children():
        #     logger.info(f"  ├─ {name} ({type(module).__name__})")
            
        #     # 如果是 model 或 language_model，进一步展开下一级
        #     if name in ["model", "language_model", "transformer"]:
        #         logger.info(f"  │   [展开 {name} 内部组件]:")
        #         for sub_name, _ in module.named_children():
        #             logger.info(f"  │     ├─ {sub_name}")
        
        # logger.info("="*50 + "\n")
        # # ------------------- [新增调试代码 结束] -------------------

# --- 根据日志精准定位组件 ---
        try:
            # 1. 定位层数 (日志显示在 text_config 中)
            self.num_layers = self.model.config.text_config.num_hidden_layers
            
            # 2. 定位 Norm 层
            # 日志显示结构为: self.model -> model -> language_model
            # 通常 Norm 层位于 language_model 的末尾
            self.norm_layer = self.model.model.language_model.norm
            
            # 3. 定位 LM Head
            self.lm_head = self.model.lm_head
            
            logger.info(f"✅ 组件定位成功: Layers={self.num_layers}, Norm={type(self.norm_layer).__name__}")
            
        except AttributeError as e:
            logger.error(f"组件定位失败: {e}")
            logger.error("请检查 language_model 内部是否名为 'norm' (也可能叫 'ln_f')")
            raise e

    def _parse_output(self, text, width, height):
        """
        修复版：支持过滤不可见字符，兼容 'good' (单对象) 和 'defects' (列表) 结构
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
            # 使用正则匹配最外层的 {} 或 []
            match = re.search(r'(\{.*\}|\[.*\])', remaining_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                # 【关键修复】移除不可见字符 (如 \xa0, \u200b 等)
                json_str = re.sub(r'[^\x00-\x7F]+', ' ', json_str) 
                
                data = json.loads(json_str)
                
                # 3. 归一化处理数据逻辑
                # 情况 A: 直接是列表 [ {defect...}, {defect...} ]
                if isinstance(data, list):
                    raw_list = data
                # 情况 B: 包含 defects 键的字典 { "defects": [...] }
                elif isinstance(data, dict) and "defects" in data:
                    raw_list = data["defects"]
                # 情况 C: 模型判断为 "good" 的单对象结构
                else:
                    raw_list = [data]

                for item in raw_list:
                    label = item.get("category") or item.get("defect") or "unknown"
                    
                    # 如果是 "good"，通常不需要上报 bbox，但如果模型输出了，我们也保留
                    defect_info = {
                        "label": label,
                        "bbox": item.get("bbox", []),
                        "reason": item.get("analysis") or item.get("reason") or "No specific reason provided.",
                        "solution": item.get("solution") or ("No action needed" if label == "good" else "Replacement or repair required.")
                    }

                    # 坐标反归一化
                    if len(defect_info["bbox"]) == 4:
                        y1, x1, y2, x2 = defect_info["bbox"] # 依照模型常用 ymin, xmin, ymax, xmax
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
        向量化优化版 DeCo/DoLa 生成函数
        支持 Batch Size > 1 并行推理，显著提升速度。
        """
        # --- 1. 初始化设置 ---
        input_ids = inputs.input_ids
        batch_size, seq_length = input_ids.shape
        device = self.device
        
        # 默认层选择
        if early_exit_layers is None:
            start_layer = self.num_layers // 2
            early_exit_layers = list(range(start_layer, self.num_layers - 1, 2))
        
        # 复制其他参数
        model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
        
        # Qwen3-VL 必需的位置控制参数
        cache_position = torch.arange(seq_length, device=device)
        
        # 状态变量
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id or eos_token_id
        
        # 结果容器 (batch_size, current_seq_len)
        generated_ids = input_ids.clone()
        
        # Logits 处理器 (如有需要可添加)
        logits_processor = LogitsProcessorList()

        for step in range(max_new_tokens):
            # --- 2. 模型前向传播 ---
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
            
            # --- 3. 获取 Final Layer Logits ---
            # next_token_logits: [batch_size, vocab_size]
            next_token_logits = outputs.logits[:, -1, :] 

            # --- 4. 向量化 DeCo 核心逻辑 (Vectorized DeCo) ---
            
            # (A) 预计算所有早退层的 Logits
            # 结果形状: [num_layers, batch_size, vocab_size]
            # 为了节省显存，如果 vocab 很大，这里会消耗较多显存。
            # 优化点：如果显存不足，可将这里的 stack 改为循环处理，但速度会变慢。
            premature_logits_list = []
            for layer_idx in early_exit_layers:
                # outputs.hidden_states[i+1] 是对应层输出
                hidden = outputs.hidden_states[layer_idx + 1][:, -1, :] 
                hidden = self.norm_layer(hidden)
                logits = self.lm_head(hidden) 
                premature_logits_list.append(logits)
            
            # Stack 起来方便并行计算
            # stack_premature_logits: [L, B, V]
            stack_premature_logits = torch.stack(premature_logits_list)
            
            # (B) 确定候选集 (Candidate Selection)
            # 仅在 Top-K 范围内计算 DeCo 修正，减少计算量并作为一种 Mask
            # probs: [B, V]
            last_layer_probs = F.softmax(next_token_logits, dim=-1)
            
            # 获取 Top-K 索引: [B, K]
            top_k_probs, top_k_ids = torch.topk(last_layer_probs, k=threshold_top_k, dim=-1)
            
            # (C) 动态层选择 (Dynamic Layer Selection) - 向量化操作
            
            # 计算早退层在所有 Vocab 上的概率
            # stack_premature_probs: [L, B, V]
            stack_premature_probs = F.softmax(stack_premature_logits, dim=-1)
            
            # 扩展 indices 以便 gather: [L, B, K]
            # 我们需要从 [L, B, V] 中取出 [L, B, K]
            expanded_ids = top_k_ids.unsqueeze(0).expand(len(early_exit_layers), -1, -1)
            
            # 提取候选词在早退层的概率: [L, B, K]
            candidate_premature_probs = torch.gather(stack_premature_probs, 2, expanded_ids)
            
            # 提取候选词在早退层的 Logits: [L, B, K]
            candidate_premature_logits = torch.gather(stack_premature_logits, 2, expanded_ids)
            
            # 寻找对每个候选词最"自信"的早退层
            # max_probs: [B, K], layer_indices: [B, K] (值为 0 到 num_layers-1)
            max_premature_probs, layer_indices = torch.max(candidate_premature_probs, dim=0)
            
            # 获取该最自信层对应的 Logits
            # 我们需要从 [L, B, K] 中根据 layer_indices [B, K] 取值 -> [B, K]
            # gather 需要 index 维度匹配，构造 [1, B, K]
            gathered_logits = torch.gather(
                candidate_premature_logits, 
                0, 
                layer_indices.unsqueeze(0)
            ).squeeze(0) # 结果: [B, K]
            
            # (D) 计算修正项并应用
            # Penalty = alpha * Prob * Logit
            # 结果: [B, K]
            penalty = alpha * max_premature_probs * gathered_logits
            
            # 获取原始 Final Layer 在候选位置的 Logits: [B, K]
            final_candidates_logits = torch.gather(next_token_logits, 1, top_k_ids)
            
            # 执行减法修正
            corrected_candidates_logits = final_candidates_logits - penalty
            
            # (E) 重组 Logits (Scatter Back)
            # 创建一个全是 -inf 的 Logits 矩阵
            modified_logits = torch.full_like(next_token_logits, float('-inf'))
            
            # 将修正后的 Top-K 值放回原位
            # scatter_ 讲 src 中的值填入 index 指定的位置
            modified_logits.scatter_(1, top_k_ids, corrected_candidates_logits)
            
            # -----------------------------------------------------
            
            # --- 5. 解码 (Decoding) ---
            # 使用 argmax (Greedy)
            # 由于非 Top-K 位置已经是 -inf，这里直接 argmax 就在候选集内选择
            processed_logits = logits_processor(input_ids, modified_logits)
            next_tokens = torch.argmax(processed_logits, dim=-1)

            # --- 6. 状态更新 ---
            # 处理已完成的序列 (Padding)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            
            # 更新 input_ids (仅保留当前步生成的 token 用于下一次输入)
            input_ids = next_tokens[:, None]
            
            # 更新 KV Cache
            past_key_values = outputs.past_key_values
            
            # 更新位置索引
            cache_position = cache_position[-1:] + 1
            
            # 更新 attention_mask (如果存在)
            if "attention_mask" in model_kwargs:
                mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [mask, mask.new_ones((batch_size, 1))], dim=-1
                )

            # 记录结果
            generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)

            # 全部序列生成完毕则提前退出
            if unfinished_sequences.max() == 0:
                break
                
        return generated_ids

    @torch.inference_mode()
    def predict_single_image(
        self, 
        image_path: str, 
        use_deco: bool = True, 
        deco_alpha: float = 2.0  # 建议 DoLa 使用 2.0 - 5.0 左右的 alpha (如果是减法逻辑)
    ):
        if not os.path.exists(image_path):
            logger.error(f"文件不存在: {image_path}")
            return None

        # 构建 Prompt
        prompt = (
            "Locate every instance that belongs to the following categories: good,broken_large,broken_small,contamination. " \
            "If it is not good, Report bbox coordinates in JSON format and analyze the cause of the defect and provide corresponding solutions"
        )

        # 构建输入
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]
        
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(self.device)

        logger.info(f"处理图片: {os.path.basename(image_path)} | Mode: {'DeCo' if use_deco else 'No DeCo'}")

        # 生成
        if use_deco:
            # 使用我们复刻的 DeCo/DoLa 逻辑
            # 注意：这里的 deco_alpha 如果是减法逻辑，建议设为正数，如 2.0
            output_ids = self.generate_with_deco(
                inputs, 
                max_new_tokens=16000, 
                alpha=deco_alpha, 
                threshold_top_p=0.95,
                threshold_top_k=40
            )
        else:
            output_ids = self.model.generate(**inputs, max_new_tokens=16000, use_cache=True)

        # 解码
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 解析结果
        with Image.open(image_path) as img:
            w, h = img.size
        parsed = self._parse_output(raw_text, w, h)
        
        return raw_text

    def run_mvtec_inference(
        self, 
        mvtec_root: str, 
        target_categories: List[str] = None,
        use_deco: bool = False,
        batch_size: int = 1,
        output_file: str = "mvtec_results.jsonl"
    ):
        # 扫描数据逻辑保持不变
        all_categories = [d for d in os.listdir(mvtec_root) if os.path.isdir(os.path.join(mvtec_root, d))]
        if target_categories:
            all_categories = [c for c in all_categories if c in target_categories]

        image_tasks = []
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
                            "gt_label": dtype,
                            "all_labels": defect_types
                        })

        logger.info(f"加载任务: {len(image_tasks)} Images")

        # 批处理循环 (仅支持 batch_size=1 当使用 DeCo 时，因为逻辑较复杂，batch化需要重写)
        if use_deco and batch_size > 1:
            logger.warning("DeCo 模式建议使用 batch_size=1 以获得最佳稳定性，已自动调整。")
            batch_size = 1

        for i in tqdm(range(0, len(image_tasks), batch_size), desc="Inferencing"):
            batch = image_tasks[i : i + batch_size]
            
            # 单张处理逻辑
            for task in batch:
                res = self.predict_single_image(
                    task["path"], 
                    task["category"], 
                    use_deco=use_deco,
                    deco_alpha=2.0 # 调节此参数控制抑制强度
                )
                
                if res:
                    record = {
                        "category": task["category"],
                        "file": os.path.basename(task["path"]),
                        "gt": task["gt_label"],
                        "pred": res["parsed"],
                        "raw": res["raw_text"]
                    }
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/ano_lvms/qwen3_4b_vl"
    MVTEC_DATA_ROOT = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec"
    TEST_IMG = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/bottle1/images/test_contamination_003.png"

    detector = Qwen3Detector(MODEL_PATH)
    
    # 单图测试：观察 Logits 修正是否解决了复读问题
    result = detector.predict_single_image(TEST_IMG, use_deco=True, deco_alpha=3.0)
    
    if result:
        print(result)
