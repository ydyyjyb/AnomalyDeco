import glob
import os
import random

mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood',
                   'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                   'screw', 'toothbrush', 'transistor', 'zipper']


MVTEC2D_DIR = './datasets/anomaly_detection/mvtec_anomaly_detection'


def _resolve_mvtec_root(category, data_root=None):
    base = data_root or os.environ.get('PADA_MVTEC_ROOT') or os.environ.get('PADA_DATA_ROOT') or MVTEC2D_DIR
    candidates = [
        base,
        os.path.join(base, 'mvtec_anomaly_detection'),
        os.path.join(base, 'MVTecAD'),
        os.path.join(base, 'MVTec_AD'),
        os.path.join(base, 'mvtec'),
    ]
    tried = []
    for candidate in candidates:
        train_dir = os.path.join(candidate, category, 'train')
        test_dir = os.path.join(candidate, category, 'test')
        tried.append(train_dir)
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            return candidate
    raise FileNotFoundError(
        'Could not find MVTecAD category data. Expected a directory like '
        f"'<root>/{category}/train'. Tried:\n  " + "\n  ".join(tried)
    )


def _read_seed_file_indices(seed_file, k_shot):
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):].strip()
            index = strip_line.split(' ')
            return [int(item) for item in index if item != '']
    return []


def _select_support_indices(labels, k_shot, support_sampling, support_seed, seed_file):
    normal_indices = [idx for idx, label in enumerate(labels) if label == 0]
    if k_shot <= 0:
        return normal_indices
    if support_sampling == 'seed_file':
        indices = _read_seed_file_indices(seed_file, k_shot)
        if len(indices) > 0:
            return indices
    if support_sampling == 'random':
        rng = random.Random(support_seed)
        return rng.sample(normal_indices, min(k_shot, len(normal_indices)))
    return normal_indices[:k_shot]


def load_mvtec(category, k_shot, selected, data_root=None, support_sampling='seed_file', support_seed=None):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = sorted(os.listdir(root_path))

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in mvtec_classes

    mvtec_root = _resolve_mvtec_root(category, data_root)
    test_img_path = os.path.join(mvtec_root, category, 'test')
    train_img_path = os.path.join(mvtec_root, category, 'train')
    ground_truth_path = os.path.join(mvtec_root, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    seed_file = os.path.join('./datasets/seeds_mvtec', category, 'selected_samples_per_run.txt')
    training_indx = _select_support_indices(train_tot_labels, k_shot, support_sampling, support_seed, seed_file)
    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
