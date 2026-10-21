import numpy as np
from torch.utils.data import DataLoader
# from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes
from .miniLED import load_miniled
from .Liver_AD import load_liver


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
    'miniled': load_miniled,
    'Liver_AD': load_liver,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):
    selected = kwargs['selected']
    dataset_name = kwargs['dataset']
    data_root = kwargs.get('data_root')
    if dataset_name == 'mvtec':
        data_root = kwargs.get('mvtec_root') or data_root
    elif dataset_name == 'visa':
        data_root = kwargs.get('visa_root') or data_root

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[dataset_name],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        selected=selected,
        data_root=data_root,
        support_sampling=kwargs.get('support_sampling', 'seed_file'),
        support_seed=kwargs.get('support_seed') if kwargs.get('support_seed') is not None else kwargs.get('seed'),
        support_perturbation=kwargs.get('support_perturbation', 'none'),
        max_test_samples=kwargs.get('max_test_samples')
    )

    if phase == 'train':
        # DataLoader 将dataset_inst 转化为可迭代的batch
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,    # shuffle=True 表示训练样本打乱顺序
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)

    return data_loader, dataset_inst
