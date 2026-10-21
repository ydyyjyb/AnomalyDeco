"""Audit information carried by previous VisA-D&R questions; no model calls."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


TOKEN = re.compile(r"[a-z]+")
STOP = {
    'a','an','the','is','are','in','on','of','to','for','and','or','with','this','that',
    'these','those','what','why','how','can','could','would','should','be','been','there',
    'any','image','visible','considered','consider','anomaly','anomalies','future','might',
    'may','do','does','it','they','their','its','from','as','at','by','about','you','your',
}


def words(s):
    return {w for w in TOKEN.findall(s.lower()) if w not in STOP and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip', nargs='*', default=['pcb2','pcb4'])
    a = ap.parse_args()
    rows = json.loads(Path(a.json).read_text(encoding='utf-8'))
    rows = [r for r in rows if not any(f'/{x}/' in r['image'] for x in a.skip)]
    audit = []
    for r in rows:
        for i in range(1, len(r['questions'])):
            prior = words(' '.join(r['questions'][:i]))
            current_q = words(r['questions'][i])
            ref = words(r['answers'][i])
            shared = sorted(prior & ref)
            audit.append({
                'index':r['index'],'image':r['image'],'question_index':i,
                'prior_question_terms':sorted(prior),'current_question_terms':sorted(current_q),
                'reference_terms':sorted(ref),'prior_reference_overlap':shared,
                'prior_reference_overlap_n':len(shared),
                'prior_reference_recall':len(shared)/len(ref) if ref else 0,
            })
    summary = {
        'items':len(rows),'nonfirst_questions':len(audit),
        'with_nonempty_prior_reference_overlap':sum(x['prior_reference_overlap_n']>0 for x in audit),
        'fraction_with_overlap':sum(x['prior_reference_overlap_n']>0 for x in audit)/len(audit),
        'mean_prior_reference_recall':sum(x['prior_reference_recall'] for x in audit)/len(audit),
        'by_question_index':{},
    }
    for i in sorted({x['question_index'] for x in audit}):
        xs=[x for x in audit if x['question_index']==i]
        summary['by_question_index'][str(i)] = {
            'n':len(xs),'fraction_with_overlap':sum(x['prior_reference_overlap_n']>0 for x in xs)/len(xs),
            'mean_prior_reference_recall':sum(x['prior_reference_recall'] for x in xs)/len(xs),
        }
    ranked=sorted(audit,key=lambda x:(x['prior_reference_overlap_n'],x['prior_reference_recall']),reverse=True)
    Path(a.out).write_text(json.dumps({'summary':summary,'examples_top_overlap':ranked[:30],'all':audit},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    for x in ranked[:5]:
        print(x['index'],x['question_index'],x['prior_reference_overlap'])


if __name__=='__main__':
    main()
