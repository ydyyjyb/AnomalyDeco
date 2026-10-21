import json

def convert_jsonl_to_json(input_file, output_file):
    data = []
    # 逐行读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    
    # 写入为标准的 JSON 数组格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 使用示例
convert_jsonl_to_json('/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/results/bottle1.jsonl', 
                      '/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/results/bottle1.json')