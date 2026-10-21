import os
import torch
import sys
from vllm import LLM, SamplingParams

def run_test():
    print("="*50)
    print("正在检查环境配置...")
    print("="*50)

    # 1. 检查 PyTorch 和 CUDA 基础环境
    try:
        print(f"Python 版本: {sys.version.split()[0]}")
        print(f"PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA 可用: 是")
            print(f"CUDA 版本: {cuda_version}")
            print(f"GPU 数量: {device_count}")
            print(f"GPU 型号: {device_name}")
        else:
            print("❌ 错误: 未检测到 CUDA 环境，vllm 无法在 CPU 模式下高效运行。")
            return
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return

    print("\n" + "="*50)
    print("正在初始化 vllm 引擎 (尝试加载 facebook/opt-125m)...")
    print("="*50)

    # 2. 初始化 vllm
    # 使用 facebook/opt-125m 因为它非常小，下载和加载都很快
    try:
        llm = LLM(
            model="facebook/opt-125m",
            tensor_parallel_size=1,        # 使用 1 张卡
            gpu_memory_utilization=0.6,    # 限制显存使用率，防止 OOM
            trust_remote_code=True,
            max_model_len=2048             # 显式指定长度避免警告
        )
    except Exception as e:
        print(f"\n❌ vllm 初始化失败。")
        print(f"错误详情: {e}")
        print("\n建议: 如果是 CUDA 驱动问题，请参考之前的解决方案升级驱动或降级 Torch。")
        return

    # 3. 准备测试 Prompt
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

    print("\n" + "="*50)
    print("开始推理测试...")
    print("="*50)

    # 4. 执行推理
    try:
        outputs = llm.generate(prompts, sampling_params)

        print("\n✅ 推理成功! 输出如下:\n")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Output: {generated_text!r}")
            print("-" * 30)
            
        print("\n✅ 测试通过: vllm 环境运行正常。")

    except Exception as e:
        print(f"❌ 推理过程中发生错误: {e}")


# CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python test_vllm.py
if __name__ == "__main__":
    run_test()