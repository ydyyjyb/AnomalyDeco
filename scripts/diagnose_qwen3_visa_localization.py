"""Follow-up localization diagnostic using the previous run's exact split.

Pixel metrics are explicitly evaluated at 256x256, not native resolution.
GT is read only after feature distances are computed. No per-image score
normalization is used for pooled AUROC. Native grid maps are retained.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from diagnose_qwen3_visa_visual_layers import extract_features, patch_distances, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous-run', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--limit-per-category', type=int, default=0)
    args = p.parse_args()
    previous = Path(args.previous_run)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((previous / 'run_config.json').read_text())
    support = json.loads((previous / 'support_manifest.json').read_text())
    tests = list(csv.DictReader((previous / 'per_image_scores.csv').open()))
    protocol = dict(previous_run=str(previous), shots=4, evaluation_size=256,
                    mask_resize='nearest', map_resize='bilinear_align_corners_false',
                    score_normalization='none', methods=cfg['layers'] + ['fusion_mean', 'fusion_max'],
                    limit_per_category=args.limit_per_category, previous_config=cfg)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(8)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_path'], dtype=torch.bfloat16,
        device_map='auto', attn_implementation='sdpa', local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(cfg['model_path'], local_files_only=True)
    layers = cfg['layers']
    methods = protocol['methods']
    metrics, cases = [], []
    for category in cfg['resolved_categories']:
        refs = [r for r in support if r['category'] == category]
        subset = [r for r in tests if r['category'] == category]
        if args.limit_per_category:
            subset = sum(([r for r in subset if int(r['label']) == y][:args.limit_per_category] for y in [0,1]), [])
        assert len(refs) == 4
        assert not {r['image'] for r in refs} & {r['image'] for r in subset}
        memories = []
        for ref in refs:
            f, _ = extract_features(model, processor, ref['image'], cfg['min_pixels'], cfg['max_pixels'])
            memories.append(f)
        bank = {l: torch.cat([f[l] for f in memories]) for l in layers}
        ys, maps = [], {m: [] for m in methods}
        catout = out / category
        catout.mkdir(exist_ok=True)
        for i, row in enumerate(subset):
            f, grid = extract_features(model, processor, row['image'], cfg['min_pixels'], cfg['max_pixels'])
            t,h,w = grid
            assert t == 1 and h % 2 == 0 and w % 2 == 0
            d = {l: patch_distances(f[l], bank[l]) for l in layers}
            stack = torch.stack(list(d.values()))
            d['fusion_mean'], d['fusion_max'] = stack.mean(0), stack.max(0).values
            native = {k: v.reshape(h//2, w//2).numpy() for k,v in d.items()}
            # Merger consumes contiguous 2x2 groups in row-major group order.
            assert all(v.numel() == h*w//4 for v in d.values())
            image = Image.open(row['image']).convert('RGB')
            if int(row['label']):
                mask_path = Path(cfg['visa_root']) / category / 'Data/Masks/Anomaly' / (Path(row['image']).stem + '.png')
                mask = Image.open(mask_path).convert('L')
                assert mask.size == image.size, (mask_path, mask.size, image.size)
                gt = np.asarray(mask.resize((256,256), Image.Resampling.NEAREST)) > 0
                assert gt.any(), f'Empty evaluation mask: {mask_path}'
            else:
                gt = np.zeros((256,256), dtype=bool)
            ys.append(gt.ravel())
            case = dict(category=category, image=row['image'], label=int(row['label']),
                        grid_h=h//2, grid_w=w//2, mask_fraction=float(gt.mean()))
            upmaps = {}
            for name, values in d.items():
                up = F.interpolate(values.reshape(1,1,h//2,w//2), size=(256,256), mode='bilinear', align_corners=False)[0,0].numpy()
                assert np.isfinite(up).all()
                maps[name].append(up.ravel())
                upmaps[name] = up
                # Reconcile image scores against previous run before reporting localization.
                k = max(1, int(np.ceil(len(values)*.05)))
                old = float(row[f's4_{name}_top5'])
                actual = float(values.topk(k).values.mean())
                assert abs(old-actual) < 1e-4, (name, old, actual)
                if gt.any():
                    case[name+'_pixel_auc'] = float(roc_auc_score(gt.ravel(), up.ravel()))
                    case[name+'_peak_hit'] = int(gt.ravel()[int(up.argmax())])
            cases.append(case)
            np.savez_compressed(catout / (str(row['label'])+'_'+Path(row['image']).stem+'.npz'), **native)
            if int(row['label']) and sum(c['category']==category and c['label']==1 for c in cases) <= 2:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4, figsize=(14,4))
                axes[0].imshow(image); axes[0].set_title('Image')
                axes[1].imshow(gt, cmap='gray'); axes[1].set_title('Ground truth')
                lo = min(upmaps['final'].min(), upmaps['fusion_mean'].min())
                hi = max(upmaps['final'].max(), upmaps['fusion_mean'].max())
                for ax, name in zip(axes[2:], ['final','fusion_mean']):
                    ax.imshow(upmaps[name], cmap='inferno', vmin=lo, vmax=hi); ax.set_title(name)
                    ax.contour(gt, levels=[.5], colors='cyan', linewidths=.5)
                for ax in axes: ax.axis('off')
                fig.tight_layout(); fig.savefig(catout / (Path(row['image']).stem+'.png'), dpi=130); plt.close(fig)
            if (i+1)%20 == 0: print(category, i+1, '/', len(subset), flush=True)
        y = np.concatenate(ys)
        for method in methods:
            selected = [c for c in cases if c['category']==category and c['label']==1]
            metrics.append(dict(category=category, feature=method, images=len(subset),
                pixel_auroc_256=float(roc_auc_score(y, np.concatenate(maps[method]))),
                anomaly_image_mean_pixel_auroc=float(np.mean([c[method+'_pixel_auc'] for c in selected])),
                peak_hit_rate=float(np.mean([c[method+'_peak_hit'] for c in selected]))))
        write_csv(out/'category_localization.csv', metrics)
        fields = sorted(set().union(*(r.keys() for r in cases)))
        with (out/'per_image_localization.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(cases)
        print(category, 'completed', flush=True)
    macro = [dict(feature=m, pixel_auroc_256=float(np.mean([r['pixel_auroc_256'] for r in metrics if r['feature']==m])),
                  peak_hit_rate=float(np.mean([r['peak_hit_rate'] for r in metrics if r['feature']==m]))) for m in methods]
    write_csv(out/'macro_localization.csv', macro)
    (out/'COMPLETED.json').write_text(json.dumps(macro, indent=2))
    print(json.dumps(macro, indent=2), flush=True)


if __name__ == '__main__':
    main()
