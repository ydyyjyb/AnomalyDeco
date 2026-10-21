import pandas as pd
import os
import json


METRIC_KEYS = [
    'i_roc', 'i_ap', 'p_roc', 'aupro',
    'score_margin',
    'normal_abnormal_anchor_distance',
    'abnormal_prompt_mean_distance',
]


def write_results(results:dict, cur_class, total_classes, csv_path):
    keys_ = list(results.keys())
    keys = METRIC_KEYS

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys_:
        if k not in df.columns:
            df[k] = 0.00

    for k in keys_:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name, dataset, csv_path):
    total_classes = [f"{dataset}-{name}" for name in total_classes]
    class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path)


def save_config(args_dict, config_path):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    serializable = {}
    for key, value in args_dict.items():
        try:
            json.dumps(value)
            serializable[key] = value
        except TypeError:
            serializable[key] = str(value)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def append_run_summary(metrics, args_dict, task, summary_path, checkpoint_path=None, heatmap_path=None, config_path=None):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    row = {
        'task': task,
        'experiment_name': args_dict.get('experiment_name', 'baseline'),
        'dataset': args_dict.get('dataset'),
        'category': args_dict.get('class_name'),
        'shot': args_dict.get('k_shot'),
        'seed': args_dict.get('seed'),
        'support_sampling': args_dict.get('support_sampling'),
        'support_seed': args_dict.get('support_seed'),
        'prompt_variant': args_dict.get('prompt_variant'),
        'use_generic_anomaly_prompt': args_dict.get('use_generic_anomaly_prompt'),
        'use_category_anomaly_prompt': args_dict.get('use_category_anomaly_prompt'),
        'use_learnable_anomaly_prompt': args_dict.get('use_learnable_anomaly_prompt'),
        'fusion_variant': args_dict.get('fusion_variant'),
        'use_text_branch': args_dict.get('use_text_branch'),
        'use_visual_memory': args_dict.get('use_visual_memory'),
        'fusion_type': args_dict.get('fusion_type'),
        'fusion_alpha': args_dict.get('fusion_alpha'),
        'model_variant': args_dict.get('model_variant'),
        'prompt_module': args_dict.get('prompt_module'),
        'kan_variant': args_dict.get('kan_variant'),
        'dfa_variant': args_dict.get('dfa_variant'),
        'use_dfa': args_dict.get('use_dfa'),
        'lambda_dfa': args_dict.get('lambda_dfa'),
        'sensitivity_variant': args_dict.get('sensitivity_variant'),
        'memory_variant': args_dict.get('memory_variant'),
        'memory_layers': args_dict.get('memory_layers'),
        'support_perturbation': args_dict.get('support_perturbation'),
        'robustness_variant': args_dict.get('robustness_variant'),
        'max_test_samples': args_dict.get('max_test_samples'),
        'save_cases': args_dict.get('save_cases'),
        'save_anchors': args_dict.get('save_anchors'),
        'trainable_params_m': args_dict.get('trainable_params_m'),
        'backbone': args_dict.get('backbone'),
        'pretrained_dataset': args_dict.get('pretrained_dataset'),
        'epochs': args_dict.get('Epoch'),
        'batch_size': args_dict.get('batch_size'),
        'lr': args_dict.get('lr'),
        'normal_prompt_length': args_dict.get('n_ctx'),
        'abnormal_prompt_length': args_dict.get('n_ctx_ab'),
        'prompt_depth': args_dict.get('prompt_depth'),
        'checkpoint_path': checkpoint_path,
        'heatmap_path': heatmap_path,
        'config_path': config_path,
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, '')

    df_new = pd.DataFrame([row])
    # Each run directory represents one task/category/shot/seed, so keep a
    # single latest summary row instead of appending duplicates on reruns.
    df_new.to_csv(summary_path, index=False, float_format='%.4f')
