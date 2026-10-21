import csv
import os

import numpy as np
import torch


def _save_tensor(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    np.save(path, value)


def _distance_rows(args, diagnostics):
    rows = []
    normal_anchor = diagnostics['normal_anchor']
    abnormal_anchor = diagnostics['abnormal_anchor']
    anchor_distance = float(diagnostics['anchor_distance'].reshape(-1)[0])

    groups = [
        ('generic', diagnostics['generic_abnormal_prompt_features']),
        ('category', diagnostics['category_abnormal_prompt_features']),
        ('learnable', diagnostics['learnable_abnormal_prompt_features']),
    ]
    all_distances = []
    group_distances = []
    for prompt_type, features in groups:
        if features.shape[0] == 0:
            group_distances.append((prompt_type, []))
            continue
        distances = torch.norm(features - normal_anchor, p=2, dim=-1)
        group_distances.append((prompt_type, distances))
        all_distances.append(distances)

    if all_distances:
        stacked = torch.cat(all_distances)
        min_distance = stacked.min()
        max_distance = stacked.max()
        denom = torch.clamp(max_distance - min_distance, min=1e-8)
    else:
        min_distance = torch.tensor(0.0)
        denom = torch.tensor(1.0)

    for prompt_type, distances in group_distances:
        for prompt_index, distance in enumerate(distances):
            dfa_weight = ((distance - min_distance) / denom * 2.0).item()
            rows.append({
                'dataset': args.dataset,
                'category': args.class_name,
                'shot': args.k_shot,
                'seed': args.seed,
                'use_dfa': args.use_dfa,
                'prompt_module': args.prompt_module,
                'dfa_variant': args.dfa_variant,
                'prompt_type': prompt_type,
                'prompt_index': prompt_index,
                'distance_to_normal_anchor': float(distance.item()),
                'dfa_weight': dfa_weight,
                'normal_abnormal_anchor_distance': anchor_distance,
                'abnormal_prompt_mean_distance': float(stacked.mean().item()) if all_distances else '',
            })
    return rows


def save_anchor_diagnostics(model, args, anchor_dir):
    os.makedirs(anchor_dir, exist_ok=True)
    diagnostics = model.collect_anchor_diagnostics()

    for name, value in diagnostics.items():
        if name == 'anchor_distance':
            continue
        _save_tensor(os.path.join(anchor_dir, f'{name}.npy'), value)

    rows = _distance_rows(args, diagnostics)
    csv_path = os.path.join(anchor_dir, 'anchor_distances.csv')
    fieldnames = [
        'dataset', 'category', 'shot', 'seed', 'use_dfa', 'prompt_module',
        'dfa_variant', 'prompt_type', 'prompt_index',
        'distance_to_normal_anchor', 'dfa_weight',
        'normal_abnormal_anchor_distance', 'abnormal_prompt_mean_distance',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dfa_csv_path = os.path.join(anchor_dir, 'dfa_weights.csv')
    with open(dfa_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        'anchor_dir': anchor_dir,
        'anchor_distances_path': csv_path,
        'dfa_weights_path': dfa_csv_path,
        'normal_abnormal_anchor_distance': float(diagnostics['anchor_distance'].reshape(-1)[0]),
        'abnormal_prompt_mean_distance': rows[0]['abnormal_prompt_mean_distance'] if rows else '',
    }
