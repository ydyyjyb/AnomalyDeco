import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, selected, data_root=None,
                 support_sampling='seed_file', support_seed=None, support_perturbation='none',
                 max_test_samples=None):

        self.load_function = load_function
        self.phase = phase
        self.data_root = data_root
        self.support_sampling = support_sampling
        self.support_seed = support_seed
        self.support_perturbation = support_perturbation
        self.max_test_samples = max_test_samples

        self.category = category  # 具体类别数据，比如transistors

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot, selected)  # self.labels => good : 0, anomaly : 1
        if self.phase == 'test' and self.max_test_samples is not None and self.max_test_samples > 0:
            self._limit_test_samples(self.max_test_samples)

    def load_dataset(self, k_shot, selected):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                      k_shot, selected,
                                                                                                      self.data_root,
                                                                                                      self.support_sampling,
                                                                                                      self.support_seed)
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def _limit_test_samples(self, max_test_samples):
        normal_indices = [idx for idx, label in enumerate(self.labels) if label == 0]
        anomaly_indices = [idx for idx, label in enumerate(self.labels) if label == 1]
        if normal_indices and anomaly_indices and max_test_samples >= 2:
            normal_quota = max(1, max_test_samples // 2)
            anomaly_quota = max_test_samples - normal_quota
            selected = normal_indices[:normal_quota] + anomaly_indices[:anomaly_quota]
        else:
            selected = list(range(min(max_test_samples, len(self.img_paths))))
        selected = selected[:max_test_samples]
        self.img_paths = [self.img_paths[idx] for idx in selected]
        self.gt_paths = [self.gt_paths[idx] for idx in selected]
        self.labels = [self.labels[idx] for idx in selected]
        self.types = [self.types[idx] for idx in selected]

    def __len__(self):
        return len(self.img_paths)

    def _perturb_support_image(self, img):
        if self.phase != 'train' or self.support_perturbation == 'none':
            return img
        if self.support_perturbation == 'brightness_low':
            return np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        if self.support_perturbation == 'brightness_high':
            return np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        if self.support_perturbation == 'noise':
            noise = np.random.normal(0, 255 * 0.02, img.shape)
            return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if self.support_perturbation == 'blur':
            return cv2.GaussianBlur(img, (3, 3), 0)
        if self.support_perturbation == 'rotate':
            h, w = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
            return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return img

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = self._perturb_support_image(img)

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[0]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255

        img = cv2.resize(img, (1024, 1024))
        gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        return img, gt, label, img_name, img_type
