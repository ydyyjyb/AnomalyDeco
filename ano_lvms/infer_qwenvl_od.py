import torch
import re
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
import time
import logging
import json
# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Qwen3-VL")

class Qwen3Detector:
    def __init__(self, model_path, device="cuda"):
        logger.info(f"正在从本地加载模型: {model_path}")
        start_time = time.time()
        self.device = device
        
        # 1. 加载处理器 (自动识别本地配置)
        self.processor = Qwen3VLProcessor.from_pretrained(model_path)
        
        # 2. 加载模型 (指定精度、显卡和 Flash Attention 2)
        # 注意: Flash Attention 2 大幅提升了处理视觉 Token 的速度
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2", # 关键提速配置
            trust_remote_code=True
        ).eval()
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}s")
        self._log_gpu_memory("加载模型后")
    def _log_gpu_memory(self, stage):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU 显存占用 [{stage}]: {memory_allocated:.2f} GB")

    def _parse_model_output(self, full_text, img_w, img_h):
        """
        解析模型输出，分离思考过程和JSON数据，并转换坐标
        """
        result = {
            "thought": "",
            "defects": []
        }

        # 1. 提取思考过程 <think>...</think>
        think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
        if think_match:
            result["thought"] = think_match.group(1).strip()
        
        # 2. 提取 JSON 部分
        # 移除 <think> 部分，避免干扰
        text_no_think = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
        
        # 寻找 JSON 结构的起始和结束 (寻找第一个 '{' 和最后一个 '}')
        try:
            start_idx = text_no_think.find('{')
            end_idx = text_no_think.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text_no_think[start_idx : end_idx + 1]
                data = json.loads(json_str)
                
                # 3. 处理坐标转换 (假设模型输出的是 0-1000 的归一化坐标)
                # 格式: [ymin, xmin, ymax, xmax] 或 [xmin, ymin, xmax, ymax]
                # Qwen 通常倾向于 [ymin, xmin, ymax, xmax] 但我们在 Prompt 里强制定义格式会更稳
                if "defects" in data:
                    for item in data["defects"]:
                        if "bbox" in item:
                            bbox = item["bbox"]
                            # 假设 Prompt 要求输出 [xmin, ymin, xmax, ymax] 且范围是 0-1000
                            # 进行转换到绝对像素
                            x1, y1, x2, y2 = bbox
                            
                            # 坐标反归一化 (如果模型输出的是0-1000)
                            # 如果模型直接输出了绝对坐标（数值大于1000），则不需要除以1000
                            scale_factor = 1000.0 if max(bbox) <= 1000 else 1.0
                            
                            item["bbox_abs"] = [
                                int((x1 / scale_factor) * img_w),
                                int((y1 / scale_factor) * img_h),
                                int((x2 / scale_factor) * img_w),
                                int((y2 / scale_factor) * img_h)
                            ]
                    result["defects"] = data["defects"]
            else:
                logger.warning("未能在输出中找到有效的 JSON 结构")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            logger.debug(f"尝试解析的字符串: {json_str}")
            
        return result

    def run_inference(self, image_path, prompt, max_new_tokens=1024):
        raw_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_image.size

        # 1. 输入处理日志
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text_prompt], images=[raw_image], return_tensors="pt").to(self.device)
        
        # 关键日志：视觉 Token 数量
        # Qwen3-VL 将图像转换为序列，视觉 Token 过多是推理变慢的主因
        num_visual_tokens = inputs.get("image_grid_thw", torch.tensor([[0,0,0]]))[0].prod()
        logger.info(f"图像尺寸: {orig_w}x{orig_h} | 视觉 Token 数量: {num_visual_tokens}")
        logger.info(f"输入总序列长度: {inputs.input_ids.shape[1]}")

        # 2. 推理阶段日志
        self._log_gpu_memory("推理开始前")
        start_time = time.time()
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_hidden_states=False,  # 允许提取中间层
                return_dict_in_generate=True
            )
        
        end_time = time.time()
        duration = end_time - start_time
        generated_ids = output.sequences[:, inputs.input_ids.shape[1]:]
        num_gen_tokens = generated_ids.shape[1]
        
        logger.info(f"生成完毕。生成 Token 数: {num_gen_tokens} | 耗时: {duration:.2f}s | 速度: {num_gen_tokens/duration:.2f} tokens/s")

        # 3. 提取中间层 (Hidden States) 日志
        # 提取最后一层隐藏状态的形状
        # last_hidden_layer = output.hidden_states[-1][-1] # [batch, 1, hidden_size]
        # logger.info(f"检测到中间层数据：第 {len(output.hidden_states[0])} 层，维度: {list(last_hidden_layer.shape)}")

        # 4. 坐标解析
        gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed_result = self._parse_model_output(gen_text, orig_w, orig_h) 
        
        return parsed_result

# --- 使用示例 ---
if __name__ == "__main__":
    MODEL_DIR = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/ano_lvms/qwen3_4b_vl"
    detector = Qwen3Detector(MODEL_DIR) # 替换为你的本地路径
    img_path = "/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/bottle1/images/test_contamination_003.png"
    result = detector.run_inference(img_path, 
    "Locate every instance that belongs to the following categories: 'good,broken_large,broken_small,contamination'. " \
    "If it is not good, Report bbox coordinates in JSON format and analyze the cause of the defect and provide corresponding solutions",
    max_new_tokens=16000)
        # 打印格式化结果
    print("\n" + "="*50)
    print("【思考过程】:")
    print(result["thought"])
    print("\n【检测结果 (JSON)】:")
    print(json.dumps(result["defects"], indent=2, ensure_ascii=False))
    
    # 如果有检测结果，打印转换后的绝对坐标
    if result["defects"]:
        print(f"\n【绝对坐标转换 (针对 {img_path} )】:")
        for defect in result["defects"]:
            print(f"类别: {defect['category']} | 原始Box: {defect['bbox']} | 像素Box: {defect.get('bbox_abs')}")
    print("="*50 + "\n")
    print(f"\n最终输出:\n{result}")
    