"""Audit actual oracle coverage and paired QA differences from saved outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

root=Path('/home/vina04/ADAR/Deco-main/runs')
run=root/'qwen3_visa_roi_qa_oracle100_20260927'
def load(p):
    return [json.loads(s) for s in p.open()]
rows=load(run/'oracle_mask_crop/reasoning_predictions.jsonl')
ids={r['index'] for r in rows}
assert len(rows)==len(ids)==100
assert all(len(r['responses'])==len(r['questions'])==4 for r in rows)
attempts=load(run/'attempts.jsonl')
assert len(attempts)==400 and all(r['attempts'][-1]['eos'] for r in attempts)
coverage=[]
for r in rows:
    path=Path('/home/vina04/ADAR/evaluation/ODinW-13/data/VisA')/r['category']/'Data/Masks/Anomaly'/(Path(r['image']).stem+'.png')
    gt=np.asarray(Image.open(path).convert('L').resize((256,256),Image.Resampling.NEAREST))>0
    union=np.zeros((256,256),bool)
    for x0,y0,x1,y1 in r['boxes']: union[y0:y1,x0:x1]=True
    coverage.append(dict(index=r['index'],coverage=float((union&gt).sum()/gt.sum()),area=float(union.mean())))
result=dict(actual_answers=sum(len(r['responses']) for r in rows),failed=0,
            coverage_mean=float(np.mean([r['coverage'] for r in coverage])),
            coverage_ge90=sum(r['coverage']>=.9 for r in coverage),
            coverage_zero=sum(r['coverage']==0 for r in coverage),
            area_mean=float(np.mean([r['area'] for r in coverage])),comparisons={})
for level,file in [('low','low_level_pairs.jsonl'),('complex','complex_pairs.jsonl')]:
    oracle={(r['index'],r['question_index']):r for r in load(run/'oracle_mask_crop'/file)}
    for base in ['original','fusion_crop']:
        pairs={(r['index'],r['question_index']):r for r in load(root/'qwen3_visa_roi_qa_full_20260922'/base/file) if r['index'] in ids}
        assert pairs.keys()==oracle.keys()
        for metric in ['rouge_l','sbert']:
            # Paired image resampling; keep within-image questions together.
            d=np.array([np.mean([oracle[k][metric]-pairs[k][metric] for k in oracle if k[0]==i]) for i in sorted(ids)])
            rng=np.random.default_rng(20260928)
            boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            result['comparisons'][f'{level}_{metric}_vs_{base}']=dict(delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
for qi in range(4):
    file='low_level_pairs.jsonl' if qi<2 else 'complex_pairs.jsonl'
    oracle=load(run/'oracle_mask_crop'/file)
    result[f'q{qi}_sbert']=float(np.mean([r['sbert'] for r in oracle if r['question_index']==qi]))
(run/'audit.json').write_text(json.dumps(result,indent=2))
(run/'actual_coverage.json').write_text(json.dumps(coverage,indent=2))
print(json.dumps(result,indent=2))
