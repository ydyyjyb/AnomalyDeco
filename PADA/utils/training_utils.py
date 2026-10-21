import random
import shutil
import time
import torch
import os
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
# from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 设置固定种子，因为神经网络权重初始化不一样，设置固定种子使得每次训练结果一致
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 从参数里建文件夹
def get_dir_from_args(TASK, root_dir, **kwargs):

    k_shot = kwargs['k_shot']
    dataset = kwargs['dataset']

    csv_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'csv')
    check_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'checkpoint')
    img_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'imgs')

    csv_path = os.path.join(csv_dir, f"Seed_{kwargs['seed']}-results.csv")
    check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt")

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return img_dir, csv_path, check_path


def get_run_artifact_paths(TASK, root_dir, **kwargs):
    dataset = kwargs['dataset']
    k_shot = kwargs['k_shot']
    seed = kwargs['seed']
    class_name = kwargs['class_name']
    experiment_name = kwargs.get('experiment_name', 'baseline')
    prompt_variant = kwargs.get('prompt_variant')
    fusion_variant = kwargs.get('fusion_variant')
    model_variant = kwargs.get('model_variant')
    dfa_variant = kwargs.get('dfa_variant')
    sensitivity_variant = kwargs.get('sensitivity_variant')
    memory_variant = kwargs.get('memory_variant')
    robustness_variant = kwargs.get('robustness_variant')
    run_parts = [root_dir, experiment_name, dataset, f'k_{k_shot}', f'seed_{seed}', class_name]
    if model_variant:
        run_parts.append(model_variant)
    if dfa_variant:
        run_parts.append(dfa_variant)
    if sensitivity_variant:
        run_parts.append(sensitivity_variant)
    if memory_variant:
        run_parts.append(memory_variant)
    if robustness_variant:
        run_parts.append(robustness_variant)
    if prompt_variant:
        run_parts.append(prompt_variant)
    if fusion_variant:
        run_parts.append(fusion_variant)
    run_parts.append(TASK.lower())
    run_dir = os.path.join(*run_parts)
    paths = {
        'run_dir': run_dir,
        'config_path': os.path.join(run_dir, 'config.json'),
        'summary_path': os.path.join(run_dir, 'summary.csv'),
        'heatmap_dir': os.path.join(run_dir, 'heatmaps'),
        'feature_dir': os.path.join(run_dir, 'features'),
        'anchor_dir': os.path.join(run_dir, 'anchors'),
        'checkpoint_dir': os.path.join(run_dir, 'checkpoints'),
    }
    for path in paths.values():
        if path.endswith('.json') or path.endswith('.csv'):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths
