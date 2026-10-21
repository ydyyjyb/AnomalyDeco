"""Compare identical validation samples across all three history modes."""
import csv
import json
from pathlib import Path
from statistics import mean
import sys

root = Path(sys.argv[1])
runs = {
    'full_history': root/'qwen3_visa_roi_qa_full_20260922',
    'independent': root/'qwen3_visa_roi_qa_independent_validation100_20260924',
    'questions_only': root/'qwen3_visa_roi_qa_questions_only_validation100_20260924',
}
def load(p):
    return [json.loads(s) for s in p.open(encoding='utf-8')]
protocols = {m:json.loads((p/'protocol.json').read_text()) for m,p in runs.items()}
assert protocols['independent']['manifest'] == protocols['questions_only']['manifest']
ids = {r['index'] for r in protocols['independent']['manifest']}
assert len(ids)==100
assert not ids.intersection(protocols['independent']['excluded_indices'])
rows=[]
for variant in ['original','random_crop','fusion_crop']:
    baseline={r['index']:r for r in load(runs['full_history']/variant/'reasoning_predictions.jsonl') if r['index'] in ids}
    for mode,path in runs.items():
        assert (path/'COMPLETED.json').exists()
        predictions={r['index']:r for r in load(path/variant/'reasoning_predictions.jsonl') if r['index'] in ids}
        assert predictions.keys()==baseline.keys()
        row=dict(history=mode,variant=variant,images=len(ids),
                 q0_identical=sum(r['responses'][0]==baseline[i]['responses'][0] for i,r in predictions.items()))
        for level,file in [('low','low_level_pairs.jsonl'),('complex','complex_pairs.jsonl')]:
            pairs=[r for r in load(path/variant/file) if r['index'] in ids]
            row[level+'_pairs']=len(pairs)
            for metric in ['rouge_l','sbert']:
                row[level+'_'+metric]=mean(r[metric] for r in pairs)
        rows.append(row)
out=root/'roi_history_validation100_summary.csv'
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
print(out)
