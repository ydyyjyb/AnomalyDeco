import os
from datasets import dataset_classes
from multiprocessing import Pool
import time

if __name__ == '__main__':

    pool = Pool(processes=5)

    datasets = ['mvtec', 'visa']
    shots = [1, 2, 4]
    gpu_ids = [0, 1, 2]

    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for i, cls in enumerate(classes[:]):
                sh_method = f'python MaPLe_cls.py ' \
                            f'--dataset {dataset} ' \
                            f'--k_shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--gpu-id {gpu_ids[i % len(gpu_ids)]}' \

                # print(sh_method)
                time.sleep(5)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
