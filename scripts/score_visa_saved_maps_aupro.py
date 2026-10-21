"""Exact region-weighted AUPRO at FPR<=0.30 on stored 256x256 maps.

Each GT connected component has total weight one (8-connectivity).
A weighted ROC thus has TPR equal to mean per-region overlap, and FPR
equal to the fraction of background pixels above the score threshold.
Integrate linearly to FPR=0.30 and divide by 0.30. GT is evaluation-only.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    assert (run/'COMPLETED.json').exists()
    cfg = json.loads((run/'protocol.json').read_text())
    tests = list(csv.DictReader((run/'per_image_localization.csv').open()))
    root = Path(cfg['previous_config']['visa_root'])
    torch.set_num_threads(4)
    rows = []
    for category in cfg['previous_config']['resolved_categories']:
        subset = [r for r in tests if r['category']==category]
        targets, weights = [], []
        scores = {m: [] for m in cfg['methods']}
        n_regions = 0
        for r in subset:
            if int(r['label']):
                mask = Image.open(root/category/'Data/Masks/Anomaly'/ (Path(r['image']).stem+'.png')).convert('L')
                gt = np.asarray(mask.resize((256,256), Image.Resampling.NEAREST))>0
            else:
                gt = np.zeros((256,256), dtype=bool)
            components, n = label(gt, structure=np.ones((3,3)))
            n_regions += n
            w = np.ones(gt.shape, dtype=np.float64)
            for region in range(1,n+1):
                belongs = components==region
                w[belongs] = 1.0/belongs.sum()
            targets.append(gt.ravel()); weights.append(w.ravel())
            with np.load(run/category/(r['label']+'_'+Path(r['image']).stem+'.npz')) as maps:
                for m in cfg['methods']:
                    v = torch.from_numpy(maps[m]).reshape(1,1,*maps[m].shape)
                    scores[m].append(F.interpolate(v, size=(256,256), mode='bilinear', align_corners=False).numpy().ravel())
        y, weight = np.concatenate(targets), np.concatenate(weights)
        for m in cfg['methods']:
            fpr, pro, _ = roc_curve(y, np.concatenate(scores[m]), sample_weight=weight, drop_intermediate=False)
            cutoff = .30
            keep = fpr < cutoff
            x = np.append(fpr[keep], cutoff)
            z = np.append(pro[keep], np.interp(cutoff, fpr, pro))
            value = float(trapezoid(z, x)/cutoff)
            rows.append(dict(category=category, feature=m, regions=n_regions, aupro_256_fpr030=value))
        with (run/'category_aupro.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(category, 'AUPRO completed', flush=True)
    print(json.dumps({m:float(np.mean([r['aupro_256_fpr030'] for r in rows if r['feature']==m])) for m in cfg['methods']}, indent=2))


if __name__=='__main__':
    main()
