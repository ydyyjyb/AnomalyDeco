import argparse
import os
import glob
import pandas as pd


METRICS = [
    'i_roc', 'i_ap', 'p_roc', 'aupro',
    'score_margin',
    'normal_abnormal_anchor_distance',
    'abnormal_prompt_mean_distance',
]


def load_summaries(root_dir, experiment_name):
    pattern = os.path.join(root_dir, experiment_name, '**', 'summary.csv')
    paths = sorted(glob.glob(pattern, recursive=True))
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df['summary_path'] = path
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    rows = []
    group_cols = ['experiment_name', 'dataset', 'shot', 'task']
    if 'model_variant' in df.columns and df['model_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('model_variant')
    if 'dfa_variant' in df.columns and df['dfa_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('dfa_variant')
    if 'sensitivity_variant' in df.columns and df['sensitivity_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('sensitivity_variant')
    if 'memory_variant' in df.columns and df['memory_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('memory_variant')
    if 'robustness_variant' in df.columns and df['robustness_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('robustness_variant')
    if 'prompt_variant' in df.columns and df['prompt_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('prompt_variant')
    if 'fusion_variant' in df.columns and df['fusion_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('fusion_variant')
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row['runs'] = len(group)
        row['categories'] = group['category'].nunique()
        row['seeds'] = group['seed'].nunique()
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors='coerce').dropna()
            if len(values) == 0:
                row[f'{metric}_mean'] = ''
                row[f'{metric}_std'] = ''
                row[f'{metric}_mean_std'] = ''
            else:
                mean = values.mean()
                std = values.std(ddof=0)
                row[f'{metric}_mean'] = mean
                row[f'{metric}_std'] = std
                row[f'{metric}_mean_std'] = f'{mean:.2f} +/- {std:.2f}'
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_categories(df):
    rows = []
    group_cols = ['experiment_name', 'dataset', 'shot', 'task', 'category']
    if 'model_variant' in df.columns and df['model_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('model_variant')
    if 'dfa_variant' in df.columns and df['dfa_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('dfa_variant')
    if 'sensitivity_variant' in df.columns and df['sensitivity_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('sensitivity_variant')
    if 'memory_variant' in df.columns and df['memory_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('memory_variant')
    if 'robustness_variant' in df.columns and df['robustness_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('robustness_variant')
    if 'prompt_variant' in df.columns and df['prompt_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('prompt_variant')
    if 'fusion_variant' in df.columns and df['fusion_variant'].fillna('').astype(str).ne('').any():
        group_cols.append('fusion_variant')
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row['runs'] = len(group)
        row['seeds'] = group['seed'].nunique()
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors='coerce').dropna()
            if len(values) == 0:
                row[f'{metric}_mean'] = ''
                row[f'{metric}_std'] = ''
            else:
                row[f'{metric}_mean'] = values.mean()
                row[f'{metric}_std'] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='./result')
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.root_dir, args.experiment_name, '_aggregate')
    os.makedirs(out_dir, exist_ok=True)

    df = load_summaries(args.root_dir, args.experiment_name)
    if df.empty:
        raise SystemExit(f'No summary.csv files found for {args.experiment_name}')

    all_runs_path = os.path.join(out_dir, 'all_runs.csv')
    summary_path = os.path.join(out_dir, 'mean_std_by_dataset_shot_task.csv')
    category_path = os.path.join(out_dir, 'mean_std_by_category.csv')

    df.to_csv(all_runs_path, index=False, float_format='%.4f')
    summarize(df).to_csv(summary_path, index=False, float_format='%.4f')
    summarize_categories(df).to_csv(category_path, index=False, float_format='%.4f')

    print(f'Wrote {all_runs_path}')
    print(f'Wrote {summary_path}')
    print(f'Wrote {category_path}')


if __name__ == '__main__':
    main()
