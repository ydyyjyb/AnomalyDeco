# test_flash_attention.py
import torch
from flash_attn import flash_attn_func
import time

def test_flash_attention():
    print("Testing Flash Attention...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 1024
    n_heads = 12
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, 
                   device='cuda', dtype=torch.float16, requires_grad=False)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, 
                   device='cuda', dtype=torch.float16, requires_grad=False)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, 
                   device='cuda', dtype=torch.float16, requires_grad=False)
    
    # 测试性能
    print(f"\nRunning Flash Attention with shape: {q.shape}")
    
    # 预热
    for _ in range(5):
        output = flash_attn_func(q, k, v)
    
    # 基准测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    iterations = 100
    for _ in range(iterations):
        output = flash_attn_func(q, k, v)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    print(f"Average time per iteration: {elapsed_time/iterations*1000:.2f} ms")
    print(f"Output shape: {output.shape}")
    print("Flash Attention test completed successfully!")
    
    return output

if __name__ == "__main__":
    test_flash_attention()