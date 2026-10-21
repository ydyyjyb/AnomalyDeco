"""Read-only paired analysis of saved scores; no metric recomputation or tuning."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/qwen3_visa_roi_qa_full_20260922'


def load(p):
    return [json.loads(s) for s in p.open(encoding='utf-8')]


def key(p):
    return '/'.join(p.replace('\\','/').split('/')[-5:])


def main():
    data = {}
    for v in ['original','random_crop','fusion_crop']:
        pairs = load(RUN/v/'low_level_pairs.jsonl') + load(RUN/v/'complex_pairs.jsonl')
        data[v] = {(r['index'],r['question_index']):r for r in pairs}
        assert len(data[v]) == 4029
    assert data['original'].keys() == data['random_crop'].keys() == data['fusion_crop'].keys()
    coverage = {}
    with (ROOT/'outputs/qwen3_visa_roi_20260922/per_image_roi.csv').open() as f:
        for r in csv.DictReader(f):
            if r['method']=='fusion_mean' and r['side']=='64' and r['k']=='3':
                coverage[key(r['image'])] = float(r['defect_pixel_coverage'])
    groups = defaultdict(list)
    category_deltas = defaultdict(lambda:defaultdict(list))
    examples = []
    for k,a in data['original'].items():
        b,c = data['fusion_crop'][k],data['random_crop'][k]
        assert a['reference']==b['reference']==c['reference']
        cov = coverage[key(a['image'])]
        bucket = 'zero' if cov==0 else 'ge90' if cov>=.9 else 'partial'
        level = 'low' if k[1]<2 else 'complex'
        for metric in ['rouge_l','sbert']:
            name = level+'_'+metric
            record = dict(original=a[metric],fusion=b[metric],random=c[metric],
                          delta=b[metric]-a[metric],fusion_minus_random=b[metric]-c[metric])
            for dimension,value in [('category',a['category']),('coverage',bucket),('question',str(k[1])),('all','all')]:
                groups[(dimension,value,name)].append(record)
            category_deltas[name][a['category']].append(record['delta'])
        if k[1]==0:
            examples.append(dict(image=a['image'],coverage=cov,reference=a['reference'],
                                 original=a['prediction'],fusion=b['prediction'],
                                 sbert_delta=b['sbert']-a['sbert']))
    rows=[]
    for (dimension,value,metric),rs in groups.items():
        rows.append(dict(dimension=dimension,value=value,metric=metric,n=len(rs),
                         **{f:mean(r[f] for r in rs) for f in rs[0]}))
    with (RUN/'paired_analysis.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rng=random.Random(20260923)
    intervals={}
    for metric,cat in category_deltas.items():
        ds=[mean(cat[k]) for k in sorted(cat)]
        boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(20000))
        intervals[metric]=dict(category_macro_delta=mean(ds),category_bootstrap_ci95=[boot[499],boot[19499]],
                               positive_categories=sum(d>0 for d in ds),categories=len(ds))
    (RUN/'paired_category_bootstrap.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    examples.sort(key=lambda r:r['sbert_delta'])
    (RUN/'q0_examples_sorted.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(intervals,indent=2))
    for r in rows:
        if r['dimension'] in ['all','coverage','question']:
            print(r)
    print('COVERED EXAMPLES: worst and best SBERT; deliberately selected extremes, not representative')
    covered=[x for x in examples if x['coverage']>=.9]
    print(json.dumps([covered[0],covered[-1]],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
