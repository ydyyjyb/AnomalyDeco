"""Fixed ROI proposal diagnostic; consumes saved heatmaps without inference.

Primary setting is 64/256 side length and top-3; other settings are ablations.
All anomaly images are evaluated. Crops are proposals, not detections.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
import torch
import torch.nn.functional as F


def boxes_from_map(scores, side, k=5):
    scores = scores.copy()
    yy, xx = np.indices(scores.shape)
    result = []
    for _ in range(k):
        y, x = np.unravel_index(scores.argmax(), scores.shape)
        assert np.isfinite(scores[y,x])
        x0, y0 = int(np.clip(x-side//2,0,256-side)), int(np.clip(y-side//2,0,256-side))
        box = [x0,y0,x0+side,y0+side]
        result.append(box)
        # Exclude all centers inside the previous crop; same rule for random.
        scores[(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])] = -np.inf
    return result


def coverage(boxes, gt, components, sizes):
    union = np.zeros(gt.shape, dtype=bool)
    for x0,y0,x1,y1 in boxes:
        union[y0:y1,x0:x1] = True
    covered = np.bincount(components[union].ravel(), minlength=len(sizes))
    fractions = covered[1:]/sizes[1:]
    return dict(defect_pixel_coverage=float((union&gt).sum()/gt.sum()),
                image_coverage_ge50=float((union&gt).sum()/gt.sum()>=.5),
                image_coverage_ge90=float((union&gt).sum()/gt.sum()>=.9),
                region_recall_ge50=float(np.mean(fractions>=.5)),
                region_recall_ge90=float(np.mean(fractions>=.9)),
                image_area_fraction=float(union.mean()))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args=parser.parse_args()
    run,out=Path(args.run),Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((run/'protocol.json').read_text())
    root=Path(protocol['previous_config']['visa_root'])
    rows=list(csv.DictReader((run/'per_image_localization.csv').open()))
    rows=[r for r in rows if int(r['label'])==1]
    methods=protocol['methods']
    design=dict(source=str(run), images=len(rows), side_lengths=[64,96], grid=256,
                ks=[1,3,5], primary_side=64, primary_k=3, random_repeats=10,
                seed=20260922, suppression='exclude centers inside previously selected crop',
                regions='8-connected components after nearest mask resize',
                region_aggregation='equal image weight, then equal category weight',
                training='none', gt_usage='evaluation only; no box selection or tuning',
                normal_images='excluded: ROI proposals are conditioned on an anomaly image')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    # Simple geometry and coverage checks.
    gt=np.zeros((256,256),bool); gt[100:105,100:105]=True
    cc,n=label(gt,np.ones((3,3))); sizes=np.bincount(cc.ravel())
    assert coverage([[90,90,120,120]],gt,cc,sizes)['defect_pixel_coverage']==1
    assert coverage([[0,0,20,20]],gt,cc,sizes)['defect_pixel_coverage']==0
    torch.set_num_threads(4)
    records=[]; examples=defaultdict(int)
    with (out/'boxes.jsonl').open('w') as boxfile:
        for i,r in enumerate(rows):
            proposed={}
            with np.load(run/r['category']/('1_'+Path(r['image']).stem+'.npz')) as maps:
                for m in methods:
                    z=maps[m]
                    up=F.interpolate(torch.from_numpy(z).reshape(1,1,*z.shape),size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy()
                    for side in [64,96]: proposed[(m,side)]=boxes_from_map(up,side)
            rng=np.random.default_rng(20260922+i)
            for rep in range(10):
                random_map=rng.random((256,256))
                for side in [64,96]: proposed[(f'random_{rep}',side)]=boxes_from_map(random_map,side)
            # Read GT only after proposals are fixed.
            mask=Image.open(root/r['category']/'Data/Masks/Anomaly'/(Path(r['image']).stem+'.png')).convert('L')
            gt=np.asarray(mask.resize((256,256),Image.Resampling.NEAREST))>0
            components,n=label(gt,np.ones((3,3))); sizes=np.bincount(components.ravel())
            assert n>0
            for (method,side),boxes in proposed.items():
                for k in [1,3,5]:
                    records.append(dict(category=r['category'],image=r['image'],method=method,
                                        side=side,k=k,**coverage(boxes[:k],gt,components,sizes)))
                boxfile.write(json.dumps(dict(category=r['category'],image=r['image'],method=method,side=side,boxes=boxes))+'\n')
            if examples[r['category']]<2:
                examples[r['category']]+=1
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axs=plt.subplots(1,4,figsize=(13,4))
                image=Image.open(r['image']).convert('RGB').resize((256,256))
                for ax,m in zip(axs[:3],['deepstack_5','final','fusion_mean']):
                    canvas=image.copy(); draw=ImageDraw.Draw(canvas)
                    for j,b in enumerate(proposed[(m,64)][:3]):
                        draw.rectangle(b,outline=['red','yellow','cyan'][j],width=2)
                    ax.imshow(canvas); ax.contour(gt,levels=[.5],colors='lime',linewidths=.7); ax.set_title(m)
                axs[3].imshow(gt,cmap='gray');axs[3].set_title('GT mask')
                for ax in axs: ax.axis('off')
                fig.tight_layout(); fig.savefig(out/(r['category']+'_'+Path(r['image']).stem+'.png'),dpi=120);plt.close(fig)
            if (i+1)%100==0:print(i+1,'/',len(rows),flush=True)
    write_csv(out/'per_image_roi.csv',records)
    fields=list(coverage([[0,0,256,256]],gt,components,sizes))
    groups=defaultdict(list)
    for r in records:
        method='random_mean10' if r['method'].startswith('random_') else r['method']
        groups[(r['category'],method,r['side'],r['k'])].append(r)
    catrows=[]
    for (cat,m,side,k),items in groups.items():
        catrows.append(dict(category=cat,method=m,side=side,k=k,**{f:float(np.mean([r[f] for r in items])) for f in fields}))
    write_csv(out/'category_roi.csv',catrows)
    groups=defaultdict(list)
    for r in catrows:groups[(r['method'],r['side'],r['k'])].append(r)
    summary=[dict(method=m,side=s,k=k,**{f:float(np.mean([r[f] for r in v])) for f in fields}) for (m,s,k),v in groups.items()]
    write_csv(out/'macro_roi.csv',summary)
    (out/'COMPLETED.json').write_text(json.dumps({'images':len(rows),'records':len(records)}))
    print(json.dumps([r for r in summary if r['side']==64 and r['k']==3],indent=2),flush=True)


if __name__=='__main__': main()
