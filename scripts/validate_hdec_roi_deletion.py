"""Paired ROI deletion diagnostic: 4-shot maps versus zero-reference HDEC.

Reuse the actual archived prompt banks/config and legacy compute_probe API.
No GT masks enter region selection, and no Youden thresholds are fitted.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


def save(p,x):
    p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--per-group',type=int,default=1,help='Images per category and ground-truth label.')
    cli=p.parse_args()
    root=Path('/home/vina04/ADAR')
    archive=root/'Deco-main/runs/six_datasets_0727_rerun_20260806_081642/visa'
    loc=root/'Deco-main/runs/qwen3_visa_localization_20260921'
    script=root/'Deco-main/transformers-4.57.6/examples/deco/0727_UP.py'
    spec=importlib.util.spec_from_file_location('main_up',script)
    up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
    old_argv=sys.argv
    sys.argv=[str(script),'--model-path',str(root/'Deco-main/qwen3_4b_vl_instruct'),'--output-dir',cli.out]
    args=up.parse_args();sys.argv=old_argv
    vars(args).update(json.loads((archive/'0-config.json').read_text())['args'])
    args.output_dir=cli.out;args.shots=0;args.run_generation=False;args.skip_generation=True
    args.local_files_only=True
    assert args.prompt_mode=='category_defects_guarded' and args.score_deco_mode=='multi_top3'
    out=Path(cli.out);out.mkdir(parents=True,exist_ok=True)
    banks=json.loads((archive/'defect_bank_by_category.json').read_text())
    normal_bank=up.parse_csv_words(args.normal_labels);hedge_bank=up.parse_csv_words(args.hedge_labels)
    groups=defaultdict(list)
    for r in csv.DictReader((loc/'per_image_localization.csv').open()):
        groups[(r['category'],int(r['label']))].append(r)
    rng=random.Random(20260929)
    assert cli.per_group>0
    selected=[]
    for _,rs in sorted(groups.items()):
        candidates=sorted(rs,key=lambda r:r['image'])
        selected.extend([rng.choice(candidates)] if cli.per_group==1 else rng.sample(candidates,cli.per_group))
    expected_scores=len(selected)*11
    assert len(selected)==20*cli.per_group
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageFilter,ImageStat
    torch.set_num_threads(4)
    manifest=[]
    for i,r in enumerate(selected):
        regions={}
        maps=np.load(loc/r['category']/(r['label']+'_'+Path(r['image']).stem+'.npz'))
        im=Image.open(r['image']).convert('RGB');w,h=im.size
        bw,bh=round(w*.25),round(h*.25)
        for m in ['deepstack_5','fusion_mean']:
            z=maps[m]
            z=F.interpolate(torch.from_numpy(z).reshape(1,1,*z.shape),size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy()
            y,x=np.unravel_index(z.argmax(),z.shape)
            x0=max(0,min(w-bw,round((x+.5)*w/256)-bw//2))
            y0=max(0,min(h-bh,round((y+.5)*h/256)-bh//2))
            regions[m]=[x0,y0,x0+bw,y0+bh]
        for rep in range(3):
            x0=rng.randrange(w-bw+1);y0=rng.randrange(h-bh+1)
            regions[f'random_{rep}']=[x0,y0,x0+bw,y0+bh]
        manifest.append(dict(**r,regions=regions,area_fraction=bw*bh/(w*h),size=[w,h]))
    protocol=dict(args=vars(args),archive=str(archive),seed=20260929,manifest=manifest,
                  heatmap_support_shots=4,hdec_prompt_shots=0,score='E_raw_hdec',
                  replacements=['blur','image_mean'],blur_radius='min(width,height)*0.04',
                  expected_scores=expected_scores,ground_truth_masks_used=False,thresholds='none',
                  defect_banks=banks,
                  code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [script,script.with_name('0724_Mechanism_validation.py')]})
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==protocol
    else:save(out/'protocol.json',protocol)
    print('PREFLIGHT',len(manifest),'images,',expected_scores,'scores',flush=True)
    if cli.prepare_only:return
    legacy=up.load_legacy_module()
    processor,model=legacy.load_model(args)
    dest=out/'scores.jsonl'
    records=[json.loads(s) for s in dest.open()] if dest.exists() else []
    done={(r['image'],r['condition']) for r in records}
    (out/'perturbed').mkdir(exist_ok=True)
    for i,r in enumerate(manifest):
        im=Image.open(r['image']).convert('RGB')
        blurred=im.filter(ImageFilter.GaussianBlur(min(im.size)*.04))
        color=tuple(round(x) for x in ImageStat.Stat(im).mean)
        conditions=[('original',None,None)]+[(fill+'_'+m,fill,b) for fill in ['blur','image_mean'] for m,b in r['regions'].items()]
        for name,fill,box in conditions:
            if (r['image'],name) in done:continue
            path=r['image']
            if fill:
                edited=im.copy()
                edited.paste(blurred.crop(box) if fill=='blur' else color,box)
                path=str(out/'perturbed'/f'{i:02d}_{name}.png');edited.save(path)
            sample=dict(category=r['category'],image=path,image_id=Path(path).stem,
                        label=int(r['label']),label_name='anomaly' if int(r['label']) else 'normal')
            t=time.time()
            case=legacy.compute_probe(args,processor,model,sample,banks[r['category']],normal_bank,hedge_bank)
            score=float(case['hallucination_probe']['hdec_score'])
            assert np.isfinite(score)
            row=dict(image=r['image'],category=r['category'],label=int(r['label']),
                     condition=name,score=score,seconds=time.time()-t,box=box,scored_image=path)
            with dest.open('a') as f:f.write(json.dumps(row)+'\n')
            records.append(row)
            print(i+1,'/',len(manifest),name,round(score,6),round(row['seconds'],2),flush=True)
    assert len(records)==expected_scores and len({(r['image'],r['condition']) for r in records})==expected_scores
    by_image=defaultdict(dict)
    for r in records:by_image[r['image']][r['condition']]=r['score']
    summary=[]
    for label in [0,1]:
        for fill in ['blur','image_mean']:
            for method in ['deepstack_5','fusion_mean']:
                deltas=[];random_deltas=[]
                for r in manifest:
                    if int(r['label'])!=label:continue
                    s=by_image[r['image']]
                    deltas.append(s['original']-s[fill+'_'+method])
                    random_deltas.append(mean(s['original']-s[fill+f'_random_{j}'] for j in range(3)))
                summary.append(dict(label=label,fill=fill,method=method,n=len(deltas),
                                    mean_drop=mean(deltas),random_mean_drop=mean(random_deltas),
                                    paired_advantage=mean(a-b for a,b in zip(deltas,random_deltas)),
                                    greater_than_random=sum(a>b for a,b in zip(deltas,random_deltas))))
    save(out/'summary.json',summary)
    save(out/'COMPLETED.json',dict(images=len(manifest),scores=len(records)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
