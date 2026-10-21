"""Paired ROI QA pilot. No GT masks or reference answers enter generation.

Uses the previous stable generation settings, not an exact paper reproduction.
Stops on incomplete output; never silently scores truncated answers.
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from repro_visa_dr_qwen3_deco_reasoning_original import (
    append_jsonl, build_messages, load_qwen3, load_reasoning_items,
    make_generation_kwargs,
)
from repro_visa_dr_reasoning import evaluate_reasoning


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def key(path):
    parts = Path(path).parts
    return '/'.join(parts[-5:])


def pixel_hash(path):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    return im.size, hashlib.sha256(im.tobytes()).hexdigest()


def oracle_mask_boxes(item, visa_root, side=64, count=3):
    """Diagnostic-only GT-centered crop boxes, never available to the method."""
    from PIL import Image
    import numpy as np
    mask = Image.open(Path(visa_root)/item['category']/'Data/Masks/Anomaly'/
                      (Path(item['image']).stem + '.png')).convert('L')
    fg = np.argwhere(np.asarray(mask.resize((256,256), Image.Resampling.NEAREST)) > 0)
    assert len(fg), f"empty oracle mask: {item['image']}"
    # Centroid then deterministic farthest-point sampling keeps three crops and
    # covers separated defect components without using language labels.
    points = [fg.mean(axis=0)]
    while len(points) < count:
        d = np.min([np.sum((fg-p)**2, axis=1) for p in points], axis=0)
        points.append(fg[int(np.argmax(d))].astype(float))
    boxes=[]
    for y,x in points:
        x0=int(np.clip(round(x)-side//2, 0, 256-side))
        y0=int(np.clip(round(y)-side//2, 0, 256-side))
        boxes.append([x0,y0,x0+side,y0+side])
    return boxes


def qa_messages(item, qi, responses, instruction, independent=False, questions_only=False):
    if questions_only and qi > 0:
        # Use one user turn rather than inventing empty assistant answers.
        context = '\n'.join(f'{i+1}. {q}' for i,q in enumerate(item['questions'][:qi]))
        question = ('Previous questions (context only; no previous answers are provided):\n'
                    + context + '\n\nAnswer only the current question:\n' + item['questions'][qi])
        return build_messages({'questions': [question]}, 0, [], instruction)
    if independent:
        return build_messages({'questions': [item['questions'][qi]]}, 0, [], instruction)
    return build_messages(item, qi, responses, instruction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--roi', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--per-category', type=int, default=2)
    p.add_argument('--exclude-manifest', help='Prior protocol JSON whose image IDs must not be sampled.')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--independent-questions', action='store_true',
                   help='Present each original question with images, without previous Q/A.')
    p.add_argument('--questions-only-history', action='store_true',
                   help='Include preceding questions as text context, but no preceding answers.')
    p.add_argument('--variants', nargs='+', choices=['original','random_crop','fusion_crop','oracle_mask_crop','text_only'],
                   default=['original','random_crop','fusion_crop'])
    p.add_argument('--oracle-visa-root', default='',
                   help='Required only for the diagnostic oracle_mask_crop variant.')
    cli = p.parse_args()
    if cli.independent_questions and cli.questions_only_history:
        p.error('Select at most one history ablation')
    if 'oracle_mask_crop' in cli.variants and not cli.oracle_visa_root:
        p.error('--oracle-mask-crop requires --oracle-visa-root')
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**json.loads(Path(cli.config).read_text()))
    args.no_sbert = False
    args.sbert_cpu = True
    args.rouge_backend = 'rouge'  # No silent metric implementation fallback.
    args.max_new_tokens = 1024
    proposals = defaultdict(dict)
    for line in (Path(cli.roi)/'boxes.jsonl').open():
        r = json.loads(line)
        if r['side'] == 64 and r['method'] in ['fusion_mean', 'random_0']:
            proposals[key(r['image'])][r['method']] = r
    categories = defaultdict(list)
    for item in load_reasoning_items(args):
        assert key(item['image']) in proposals, item['image']
        categories[item['category']].append(item)
    rng = random.Random(20260922)
    excluded = set()
    if cli.exclude_manifest:
        excluded = {str(r['index']) for r in json.loads(Path(cli.exclude_manifest).read_text())['manifest']}
    items = []
    for cat in sorted(categories):
        candidates = sorted(categories[cat], key=lambda x: x['image'])
        rng.shuffle(candidates)
        candidates = [r for r in candidates if str(r['index']) not in excluded]
        if cli.per_category:
            assert len(candidates) >= cli.per_category
        items.extend(candidates[:cli.per_category] if cli.per_category else candidates)
    manifest = []
    for item in items:
        maps = proposals[key(item['image'])]
        assert set(maps) == {'fusion_mean', 'random_0'}
        original = maps['fusion_mean']['image']
        assert pixel_hash(original) == pixel_hash(item['abs_image']), item['image']
        assert len(item['questions']) == len(item['answers']) and len(item['questions']) >= 4
        manifest.append(dict(index=item['index'], image=item['image'],
                             sha256_rgb=pixel_hash(original)[1],
                             boxes={m: r['boxes'][:3] for m,r in maps.items()}))
    protocol = dict(args=vars(args), seed=20260922, images=len(items),
                    variants=cli.variants,
                    roi_source=cli.roi, crop_side_fraction=.25, crop_count=3,
                    random_repeat=0, eos_required=True, caps=[1024,4096],
                    no_fixed_answer=True, use_deco=False,
                    note='Stable concise QA settings, not exact paper reproduction; original has fewer visual tokens.',
                    manifest=manifest)
    if 'text_only' in cli.variants:
        protocol['text_only_control'] = 'Remove all image content; preserve questions, instruction and selected history mode.'
    if 'oracle_mask_crop' in cli.variants:
        protocol['oracle_diagnostic'] = {
            'uses_gt_masks_for_crop_centers': True,
            'not_a_deployable_method': True,
            'side': 64, 'count': 3,
            'selection':'mask centroid then deterministic farthest foreground points',
            'visa_root':cli.oracle_visa_root,
        }
    if cli.exclude_manifest:
        protocol['excluded_indices'] = sorted(excluded)
    if cli.independent_questions:
        protocol['history_mode'] = 'independent_questions'
    if cli.questions_only_history:
        protocol['history_mode'] = 'questions_only_history'
    config_path = out/'protocol.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == protocol, 'Resume protocol mismatch'
    else:
        save(config_path, protocol)
    print('PREFLIGHT OK', len(items), 'pixel-identical images', flush=True)
    if cli.prepare_only:
        return
    import torch
    from PIL import Image
    processor, model = load_qwen3(args)
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    kwargs = make_generation_kwargs(args, False, pad, eos, tokenizer)
    started = time.time()
    for item_i,item in enumerate(items):
        for variant in protocol['variants']:
            folder = out/variant
            folder.mkdir(exist_ok=True)
            dest = folder/'reasoning_predictions.jsonl'
            done = {json.loads(s)['index'] for s in dest.open()} if dest.exists() else set()
            if item['index'] in done:
                continue
            full = Image.open(item['abs_image']).convert('RGB')
            images = [] if variant == 'text_only' else [full.copy()]
            boxes = []
            if variant not in ['original', 'text_only']:
                if variant == 'oracle_mask_crop':
                    boxes = oracle_mask_boxes(item, cli.oracle_visa_root)
                else:
                    m = 'random_0' if variant == 'random_crop' else 'fusion_mean'
                    boxes = proposals[key(item['image'])][m]['boxes'][:3]
                for x0,y0,x1,y1 in boxes:
                    box = (round(x0*full.width/256), round(y0*full.height/256),
                           round(x1*full.width/256), round(y1*full.height/256))
                    images.append(full.crop(box))
            for im in images:
                im.thumbnail((args.max_image_side,args.max_image_side), Image.Resampling.LANCZOS)
            responses, details = [], []
            for qi in range(len(item['questions'])):
                messages = qa_messages(item, qi, responses, args.answer_instruction,
                                       cli.independent_questions, cli.questions_only_history)
                if variant == 'text_only':
                    for message in messages:
                        message['content'] = [part for part in message['content'] if part['type'] != 'image']
                if boxes:
                    # Same neutral framing in both crop arms, no defect location/type hints.
                    messages[0]['content'] = ([{'type':'text','text':'The first image is the full image. The next three images are enlarged crops from it.'}]
                        + [{'type':'image'} for _ in images] + [messages[0]['content'][-1]])
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], **({'images':images} if images else {}),
                                   padding=True, return_tensors='pt').to(model.device)
                if variant == 'text_only':
                    assert 'pixel_values' not in inputs and 'image_grid_thw' not in inputs
                n_input = inputs['input_ids'].shape[-1]
                attempts = []
                for cap in protocol['caps']:
                    torch.cuda.synchronize()
                    t = time.time()
                    with torch.inference_mode():
                        output = model.generate(**inputs, **{**kwargs,'max_new_tokens':cap})
                    torch.cuda.synchronize()
                    ids = output[0,n_input:].tolist()
                    decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    complete = bool(ids and ids[-1] == eos and decoded)
                    attempt = dict(cap=cap, tokens=len(ids), last_token=ids[-1] if ids else None,
                                   eos=complete, seconds=time.time()-t, text=decoded)
                    attempts.append(attempt)
                    if complete:
                        break
                append_jsonl(out/'attempts.jsonl', dict(index=item['index'],variant=variant,question=qi,attempts=attempts))
                if not complete:
                    save(out/'BLOCKED.json', dict(index=item['index'],variant=variant,question=qi,reason='No nonempty EOS-terminated answer at cap 4096'))
                    raise RuntimeError('Incomplete generation retained in attempts; metrics not reported')
                responses.append(decoded)
                details.append(dict(attempts=len(attempts),seconds=sum(a['seconds'] for a in attempts),
                                    tokens=len(ids),input_tokens=n_input,
                                    image_grid_thw=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else []))
            append_jsonl(dest, dict(index=item['index'],image=item['image'],category=item['category'],
                         questions=item['questions'],answers=item['answers'],responses=responses,
                         variant=variant,details=details,boxes=boxes,hit_max_new_tokens=[False]*len(responses)))
            print(item_i+1,'/',len(items),variant,item['image'],'seconds',round(sum(d['seconds'] for d in details),1),flush=True)
    del model
    torch.cuda.empty_cache()
    summaries = {}
    for variant in protocol['variants']:
        folder = out/variant
        rows = [json.loads(s) for s in (folder/'reasoning_predictions.jsonl').open()]
        assert len(rows) == len(items) and len({r['index'] for r in rows}) == len(items)
        summaries[variant] = evaluate_reasoning(args, rows, folder)
        summaries[variant]['generation_seconds'] = sum(d['seconds'] for r in rows for d in r['details'])
        summaries[variant]['retry_questions'] = sum(d['attempts']>1 for r in rows for d in r['details'])
    save(out/'summary.json',summaries)
    save(out/'COMPLETED.json',dict(images=len(items),answers=sum(len(i['questions']) for i in items)*len(protocol['variants']),seconds_this_run=time.time()-started))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
