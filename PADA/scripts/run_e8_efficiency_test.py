import argparse
import csv
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MaPLe.model import MaPLe
from MaPLe.params_init import get_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--variants', default='none,mlp,transformer,kan')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--output', default='./result/e8_efficiency/efficiency.csv')
    known, extra = parser.parse_known_args()
    base_args = get_args(extra) if False else None
    return known


def make_base_args(prompt_module, gpu_id):
    import sys
    old = sys.argv
    sys.argv = [old[0], '--class_name', 'tile', '--prompt-module', prompt_module, '--gpu-id', str(gpu_id)]
    args = get_args()
    sys.argv = old
    return args


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for variant in args.variants.split(','):
        variant = variant.strip()
        model_args = make_base_args(variant, args.gpu_id)
        kwargs = vars(model_args)
        kwargs.update({'device': device, 'k_shot': 1, 'resolution': 400, 'img_resize': 240, 'img_cropsize': 240})
        model = MaPLe(model_args, **kwargs).to(device)
        model.eval_mode()
        model.build_text_feature_gallery()
        image = torch.randn(1, 3, 240, 240, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(args.warmup):
                model(image, 'cls', is_eval=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                model(image, 'cls', is_eval=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        rows.append({
            'prompt_module': variant,
            'trainable_params_m': trainable_params,
            'total_params_m': total_params,
            'gpu_memory_mb': memory_mb,
            'inference_time_ms': avg,
            'inference_time_std_ms': std,
            'fps': 1000.0 / avg if avg > 0 else 0,
        })
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
