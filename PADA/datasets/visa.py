import glob
import os
import random

visa_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                   'fryum', 'macaroni1', 'macaroni2',
                'pcb1', 'pcb2', 'pcb3','pcb4', 'pipe_fryum']

VISA_DIR = './datasets/anomaly_detection/VisA_pytorch/1cls'


def _resolve_visa_root(category, data_root=None):
    base = data_root or os.environ.get('PADA_VISA_ROOT') or os.environ.get('PADA_DATA_ROOT') or VISA_DIR
    candidates = [
        base,
        os.path.join(base, '1cls'),
        os.path.join(base, 'VisA_pytorch', '1cls'),
        os.path.join(base, 'VisA_pytorch'),
        os.path.join(base, 'visa', '1cls'),
        os.path.join(base, 'visa'),
        os.path.join(base, 'VisA', '1cls'),
        os.path.join(base, 'VisA'),
    ]
    tried = []
    for candidate in candidates:
        train_dir = os.path.join(candidate, category, 'train')
        test_dir = os.path.join(candidate, category, 'test')
        tried.append(train_dir)
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            return candidate, 'train_test'
        odinw_normal_dir = os.path.join(candidate, category, 'Data', 'Images', 'Normal')
        odinw_anomaly_dir = os.path.join(candidate, category, 'Data', 'Images', 'Anomaly')
        tried.append(odinw_normal_dir)
        if os.path.isdir(odinw_normal_dir) and os.path.isdir(odinw_anomaly_dir):
            return candidate, 'normal_anomaly'

    # Some shared ODinW/VisA mirrors add one or two extra nesting levels.
    # Search for '<root>/.../<category>/train' and return the directory that
    # contains category folders.
    if os.path.isdir(base):
        for root, dirs, _ in os.walk(base):
            rel_depth = os.path.relpath(root, base).count(os.sep)
            if rel_depth > 5:
                dirs[:] = []
                continue
            train_dir = os.path.join(root, category, 'train')
            test_dir = os.path.join(root, category, 'test')
            tried.append(train_dir)
            if os.path.isdir(train_dir) and os.path.isdir(test_dir):
                return root, 'train_test'
            odinw_normal_dir = os.path.join(root, category, 'Data', 'Images', 'Normal')
            odinw_anomaly_dir = os.path.join(root, category, 'Data', 'Images', 'Anomaly')
            tried.append(odinw_normal_dir)
            if os.path.isdir(odinw_normal_dir) and os.path.isdir(odinw_anomaly_dir):
                return root, 'normal_anomaly'
    raise FileNotFoundError(
        'Could not find VisA category data. Expected a directory like '
        f"'<root>/{category}/train'. Tried:\n  " + "\n  ".join(tried)
    )


def _glob_images(path):
    patterns = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png', '*.BMP', '*.bmp']
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(paths)


def _find_mask_path(mask_root, img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(mask_root, stem + '.png'),
        os.path.join(mask_root, stem + '.jpg'),
        os.path.join(mask_root, stem + '.JPG'),
        os.path.join(mask_root, os.path.basename(img_path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    recursive = []
    for ext in ['.png', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']:
        recursive.extend(glob.glob(os.path.join(mask_root, '**', stem + ext), recursive=True))
    if recursive:
        return sorted(recursive)[0]
    return 0


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


def load_visa(category, k_shot, selected, data_root=None, support_sampling='seed_file', support_seed=None):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = sorted(os.listdir(root_path))

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = _glob_images(os.path.join(root_path, defect_type))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = _glob_images(os.path.join(root_path, defect_type))
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '.png') for s in
                            img_paths]
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in visa_classes

    visa_root, visa_layout = _resolve_visa_root(category, data_root)

    if visa_layout == 'train_test':
        test_img_path = os.path.join(visa_root, category, 'test')
        train_img_path = os.path.join(visa_root, category, 'train')
        ground_truth_path = os.path.join(visa_root, category, 'ground_truth')

        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
        train_tot_types = load_phase(train_img_path, ground_truth_path)

        test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
        test_tot_types = load_phase(test_img_path, ground_truth_path)
    else:
        normal_dir = os.path.join(visa_root, category, 'Data', 'Images', 'Normal')
        anomaly_dir = os.path.join(visa_root, category, 'Data', 'Images', 'Anomaly')
        mask_dir = os.path.join(visa_root, category, 'Data', 'Masks')

        normal_paths = _glob_images(normal_dir)
        anomaly_paths = _glob_images(anomaly_dir)

        train_img_tot_paths = normal_paths
        train_gt_tot_paths = [0] * len(normal_paths)
        train_tot_labels = [0] * len(normal_paths)
        train_tot_types = ['good'] * len(normal_paths)

        test_img_tot_paths = list(normal_paths) + list(anomaly_paths)
        test_gt_tot_paths = [0] * len(normal_paths) + [_find_mask_path(mask_dir, path) for path in anomaly_paths]
        test_tot_labels = [0] * len(normal_paths) + [1] * len(anomaly_paths)
        test_tot_types = ['good'] * len(normal_paths) + ['anomaly'] * len(anomaly_paths)

    seed_file = os.path.join('./datasets/seeds_visa', category, 'selected_samples_per_run.txt')
    training_indx = _select_support_indices(train_tot_labels, k_shot, support_sampling, support_seed, seed_file)
    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]

    if visa_layout == 'normal_anomaly':
        support_paths = set(selected_train_img_tot_paths)
        keep_indices = [idx for idx, path in enumerate(test_img_tot_paths) if path not in support_paths]
        test_img_tot_paths = [test_img_tot_paths[idx] for idx in keep_indices]
        test_gt_tot_paths = [test_gt_tot_paths[idx] for idx in keep_indices]
        test_tot_labels = [test_tot_labels[idx] for idx in keep_indices]
        test_tot_types = [test_tot_types[idx] for idx in keep_indices]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
