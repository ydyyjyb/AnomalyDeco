import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict, OrderedDict
import torch
from PIL import Image

# Transformers imports
from transformers import AutoProcessor, AutoModelForCausalLM

# pycocotools imports
from pycocotools.coco import COCO

# Local imports
from dataset_utils import load_odinw_config, generate_odinw_jobs
from eval_utils import compute_metrics

try:
    from processing_qwen3_vl import Qwen3VLProcessor
    from modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    print("✓ Successfully imported local Qwen3-VL classes.")
except ImportError as e:
    print(f"⚠️  Warning: Could not import local Qwen3-VL files ({e}).")
    print("   Falling back to AutoClass (this may fail if remote code is not in model_path).")
    from transformers import AutoProcessor, AutoModelForCausalLM
    Qwen3VLProcessor = AutoProcessor
    Qwen3VLForConditionalGeneration = AutoModelForCausalLM

def load_images_from_messages(messages):
    """Extract and load images from messages list."""
    images = []
    for message in messages:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image':
                    img_path = content['image']
                    if img_path.startswith("file://"):
                        img_path = img_path[7:]
                    try:
                        image = Image.open(img_path).convert("RGB")
                        images.append(image)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        # Create a dummy image to avoid crashing, though this ruins the sample
                        images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
    return images

def run_inference(args):
    """Run inference on the ODinW dataset using Qwen3-VL Transformers."""
    print("\n" + "="*80)
    print("🚀 ODinW Inference with Qwen3-VL (Transformers)")
    if args.num_shots > 0:
        print(f"   Few-Shot Mode: {args.num_shots} shot(s)")
    print("="*80 + "\n")
    
    # Generate task list
    question_list, datasets = generate_odinw_jobs(args.data_dir, args)
    print(f"✓ Generated {len(question_list)} inference jobs\n")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load Model and Processor
    print(f"Loading model from {args.model_path}")
        # 使用显式的类加载，指定 local_files_only=True 可以防止它错误地去连网找模型
    # 如果你的模型权重确实在本地，这能避免那个 Repo ID 错误
    try:
        processor = Qwen3VLProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype="auto"
        )
    except Exception as e:
        print("\n❌ Error loading model directly.")
        print("Please ensure 'processing_qwen3_vl.py' and 'modeling_qwen3_vl.py' are in the CURRENT directory.")
        print(f"Detailed error: {e}")
        return
    
    model.eval()
    print("✓ Model and Processor loaded\n")
    
    print("🚀 Running inference...")
    results = []
    start_time = time.time()
    
    # Qwen3-VL parameters for image processing
    image_resolution_kwargs = {
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels
    }

    # Iterate one by one (or small batches if implemented, but VL models vary in tokens/shape)
    for item in tqdm(question_list, desc="Inference"):
        messages = item['messages']
        
        # 1. Prepare Text Prompt
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Load Images
        image_inputs = load_images_from_messages(messages)
        
        # 3. Prepare Inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            **image_resolution_kwargs
        )
        inputs = inputs.to(model.device)
        
        # 4. Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty
            )
        
        # 5. Decode
        # Strip input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Handle </think> tag if present (from DeepSeek/reasoning models style if applicable)
        response_final = str(response).split("</think>")[-1].strip()
        
        result = {
            "question_id": item['question_id'],
            "annotation": item['annotation'],
            "extra_info": item['extra_info'],
            "result": {"gen": response_final, "gen_raw": response},
            "messages": item['messages'] # Save pure messages (without loaded images)
        }
        results.append(result)
        
        # Optional: Save interactively to prevent data loss
        if len(results) % 100 == 0:
             with open(args.output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res) + '\n')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(question_list):.2f} seconds/sample")
    
    # Final Save
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    
    # Save dataset config
    config_output = args.output_file.replace('.jsonl', '_datasets.json')
    with open(config_output, 'w') as f:
        datasets_serializable = {}
        for k, v in datasets.items():
            datasets_serializable[k] = {
                'metainfo': v['metainfo'],
                'data_root': v['data_root'],
                'ann_file': v['ann_file'],
                'data_prefix': v['data_prefix']
            }
        json.dump(datasets_serializable, f, indent=2)


def run_evaluation(args):
    """Run evaluation on inference results."""
    print("\n" + "="*80)
    print("🎯 ODinW Evaluation")
    print("="*80 + "\n")
    
    results = []
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found.")
        return

    print(f"✓ Loaded {len(results)} inference results\n")
    
    config_path = os.path.join(args.data_dir, "odinw13_config.py")
    datasets = load_odinw_config(config_path)
    
    all_outputs = defaultdict(list)
    for job in results:
        all_outputs[job["extra_info"]["dataset_name"]].append(job)
    
    all_results = {}
    
    for dataset_name, sub_jobs in all_outputs.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        
        anno_path = sub_jobs[0]["extra_info"]["anno_path"]
        coco_api = COCO(anno_path)
        classes = datasets[dataset_name]['metainfo']['classes']
        pred_bboxes_per_img = defaultdict(list)
        
        for job in sub_jobs:
            img_id = job["extra_info"]["img_id"]
            img_h = job["extra_info"]["img_h"]
            img_w = job["extra_info"]["img_w"]
            
            answer = job['result']['gen']
            answer = answer.replace("```json", "").replace("```", "").strip()
            
            import ast
            try:
                # Basic string cleaning
                if answer.startswith("[") and not answer.endswith("]"):
                    answer += "]"
                json_data = ast.literal_eval(answer)
                pred_bboxes = []
                pred_labels = []
                
                if isinstance(json_data, list):
                    for data in json_data:
                        if isinstance(data, dict) and len(data.get("bbox_2d", [])) == 4:
                            pred_bboxes.append(data["bbox_2d"])
                            pred_labels.append(data.get("label", ""))
            except Exception:
                pred_bboxes = []
                pred_labels = []
            
            # Coordinate conversion
            # Qwen3-VL (like Qwen2-VL) typically outputs normalized coordinates [0, 1000]
            # We convert these back to absolute pixel coordinates.
            if len(pred_bboxes) > 0:
                pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
                # Normalize factor
                scale = np.array([img_w, img_h, img_w, img_h]) / 1000.0
                pred_bboxes = pred_bboxes * scale
            else:
                pred_bboxes = np.array([]).reshape(-1, 4)
            
            pred_bboxes = pred_bboxes.tolist()
            
            pred_objs = defaultdict(list)
            for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
                pred_objs[str(pred_label).lower()].append(pred_bbox)
            
            class_names_lower = [name.lower() for name in classes]
            
            for k, v in pred_objs.items():
                if k in class_names_lower:
                    idx = class_names_lower.index(k)
                    pred_bboxes_per_img[img_id].append({
                        'label': idx, 
                        'bbox': v
                    })
        
        pred_results = []
        for k, v in pred_bboxes_per_img.items():
            bboxes = []
            labels = []
            for tmp in v:
                bboxes.extend(tmp['bbox'])
                labels.extend([tmp['label']] * len(tmp['bbox']))
            
            height = coco_api.imgs[k]["height"]
            width = coco_api.imgs[k]["width"]
            
            pred_tuple = (
                {'width': width, 'height': height, 'img_id': k},
                {
                    'img_id': k,
                    'bboxes': np.array(bboxes),
                    'scores': np.array([1.0] * len(bboxes)), # Dummy scores
                    'labels': np.array(labels),
                },
            )
            pred_results.append(pred_tuple)
        
        eval_results = compute_metrics(pred_results, _coco_api=coco_api)
        print(f"{dataset_name}: {eval_results}")
        all_results[dataset_name] = eval_results
    
    # Summarize results
    results_ordered = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    results_display = []
    
    for prefix, result in results_ordered.items():
        results_display.append([prefix] + [result[k] for k in metric_items])
    
    average_scores = []
    if results_display:
        for col_idx in range(len(metric_items)):
            average_scores.append(np.mean([line[col_idx + 1] for line in results_display]))
        results_display.append(['Average'] + average_scores)
        all_results.update({"Average": average_scores[0]})
        print(f"\nFinal Average mAP: {average_scores[0]:.4f}")
    
    try:
        from tabulate import tabulate
        print("\n" + "="*80)
        print(tabulate(results_display, headers=["Dataset"] + metric_items, tablefmt="fancy_outline", floatfmt=".3f"))
        print("="*80 + "\n")
    except ImportError:
        pass
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="ODinW Evaluation with Qwen3-VL")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--data-dir", type=str, required=True, help="Path to ODinW data")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    
    # Generation parameters
    infer_parser.add_argument("--max-new-tokens", type=int, default=16000)
    infer_parser.add_argument("--temperature", type=float, default=0.6) # Greedy for detection usually
    infer_parser.add_argument("--top-p", type=float, default=0.95)
    infer_parser.add_argument("--top-k", type=int, default=20)
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    
    # Image resolution
    infer_parser.add_argument("--min-pixels", type=int, default=128*128)
    infer_parser.add_argument("--max-pixels", type=int, default=16*16*2*2*2*6144)
    
    # Few-shot
    infer_parser.add_argument("--num-shots", type=int, default=0, help="Number of shots (0 for zero-shot)")
    infer_parser.add_argument("--ref-img-dir", type=str, default=None)

    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, required=True)
    eval_parser.add_argument("--input-file", type=str, required=True)
    eval_parser.add_argument("--output-file", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()