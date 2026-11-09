#!/usr/bin/env python
"""CPU-only fresh-HC selection. No training, generation, or external data.

Reconstruct unstarted document positions at B stable. Select coherent official
train documents with frozen train-fitted hashed density ratios and full-document
overlap checks. Keep a separate source name so old resume indices remain valid.
"""
import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import zlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from saul.utils.io import iter_shards

spec=importlib.util.spec_from_file_location('curate94',ROOT/'scripts/94_recovery_curate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def body(text):
    words=text.split()
    if len(words)<=1536:return ' '.join(words)
    mid=len(words)//2
    return ' '.join(words[64:576]+words[mid-256:mid+256]+words[-512:])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--pool',type=int,default=150000)
    p.add_argument('--target-tokens',type=int,default=120000000)
    a=p.parse_args()
    os.chdir(ROOT)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    old=(ROOT/'data/tokens/legal32k').resolve()
    index=np.load(old/'train/high_courts/index.npy')
    st=torch.load(ROOT/'runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',map_location='cpu',weights_only=False,mmap=True)
    seen=np.zeros(len(index),dtype=bool)
    perm=np.random.default_rng([781,zlib.crc32(b'high_courts'),0]).permutation(len(index))
    for w,ws in st['loader']['workers'].items():
        assert ws['epoch']['high_courts']==0, 'Need multi-epoch union for this checkpoint'
        seen[perm[int(w)::4][:ws['cursor']['high_courts']]]=True
    del st,perm
    eligible=np.flatnonzero(~seen & (index[:,1]>=1024) & (index[:,1]<=24000) & (index[:,3]==0))
    selected_indices=set(np.random.default_rng(2781).choice(eligible,min(a.pool,len(eligible)),replace=False).tolist())
    stats=collections.Counter(unstarted_docs=int((~seen).sum()),length_anon_eligible=len(eligible),sampled=len(selected_indices))
    print('eligibility',dict(stats),flush=True)
    blocked=set()
    for r in c.read_json('data/raw/Phase_1_Pretraining/Test/test.json').values():blocked.update(c.grams(r['text']))
    dp=c.read_json('data/final/dev_prompts.json');dt=c.read_json('data/final/dev_truth.json')
    for k in dp:blocked.update(c.grams(dp[k]['text']+' '+dt[k]['text']))
    held=set()
    for split in ['val','dev']:
        for src in c.SOURCES:
            for r in iter_shards(f'data/final/{split}/{src}'):held.add((src,r['meta']['cluster']))
    boiler=collections.Counter()
    for src in c.SOURCES:
        for i,r in enumerate(iter_shards(f'data/final/train/{src}')):
            if i>=1000:break
            boiler.update(c.grams(r['text'])&blocked)
    exemptions={g for g,n in boiler.items() if n>=50};blocked-=exemptions
    vec=HashingVectorizer(n_features=10000,ngram_range=(1,2),alternate_sign=False,norm=None)
    fit_ids=set()
    fits={}
    for src in ['supreme_court','high_courts']:
        rows=[json.loads(l) for l in Path(f'runs/recovery_20261005/curation/filtered_{src}.jsonl').read_text().splitlines()]
        rows=sorted(rows,key=lambda r:c.hashed('fresh-fit:'+r['id']))[:1000]
        fit_ids.update(r['id'] for r in rows)
        freq=np.asarray(vec.transform([body(r['prompt']+' '+r['target']) for r in rows]).sum(axis=0)).ravel()+1
        fits[src]=freq/freq.sum()
    ratio=np.log(fits['supreme_court'])-np.log(fits['high_courts'])
    rows=[];batch=[];used=set()

    def flush():
        if not batch:return
        matrix=vec.transform([body(r['text']) for _,r in batch])
        scores=np.asarray(matrix@ratio).ravel()/np.maximum(1,np.asarray(matrix.sum(axis=1)).ravel())
        for (j,r),score in zip(batch,scores):
            u=min(1-1e-15,max(1e-15,(c.hashed('fresh-select:'+r['id'])+.5)/2**64))
            words=r['text'].split()
            rows.append(dict(id=r['id'],cluster=r['meta']['cluster'],index=j,tokens=int(index[j,1]),
                density=float(score),rank=float(score/.1-math.log(-math.log(u))),
                mention_supreme_court='supreme court' in r['text'].lower(),
                excerpt=' '.join(words[min(64,len(words)//4):min(64,len(words)//4)+160])))
        batch.clear()

    for j,r in enumerate(iter_shards('data/final/train/high_courts')):
        if j not in selected_indices:continue
        stats['examined']+=1
        assert len(r['text'].encode())==int(index[j,2]),('original index mismatch',j)
        cluster=r['meta']['cluster']
        if r['id'] in fit_ids or ('high_courts',cluster) in held or cluster in used:
            stats['cluster_or_fit_dropped']+=1;continue
        text=r['text'];words=text.split()
        alpha=sum(any(ch.isalpha() for ch in w) for w in words)/max(1,len(words))
        controls=sum(ord(ch)<32 and ch not in '\n\t\r' or 127<=ord(ch)<160 for ch in text)
        pairs=collections.Counter(zip(words,words[1:]))
        repeat=sum(n-2 for n in pairs.values() if n>2)/max(1,len(words)-1)
        if alpha<.75 or controls/max(1,len(text))>.001 or repeat>.35:
            stats['quality_dropped']+=1;continue
        if len(c.grams(text)&blocked)>=3:
            stats['overlap_dropped']+=1;continue
        used.add(cluster);batch.append((j,r));stats['accepted']+=1
        if len(batch)>=128:flush()
        if stats['examined']%2000==0:print('curation',dict(stats),flush=True)
    flush()
    # Save all candidate metadata for audit and alternate token-budget selection.
    with (out/'eligible_candidates.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    ranked=sorted(rows,key=lambda r:r['rank'],reverse=True)
    chosen=[];tokens=0
    for r in ranked:
        chosen.append(r);tokens+=r['tokens']
        if tokens>=a.target_tokens:break
    enough=tokens>=a.target_tokens
    summary={'status':'ready_for_review' if enough else 'insufficient_tokens','stats':dict(stats),
        'selected_docs':len(chosen),'selected_tokens':tokens,'target_tokens':a.target_tokens,
        'boilerplate_exemptions':len(exemptions),'density_fit_documents':len(fit_ids),
        'candidate_mean_density':float(np.mean([r['density'] for r in rows])) if rows else None,
        'selected_mean_density':float(np.mean([r['density'] for r in chosen])) if chosen else None,
        'candidate_sc_mention_fraction':float(np.mean([r['mention_supreme_court'] for r in rows])) if rows else None,
        'selected_sc_mention_fraction':float(np.mean([r['mention_supreme_court'] for r in chosen])) if chosen else None,
        'not_launched':True,
        'limitations':['Unstarted is relative to B stable stream; semantic novelty is not guaranteed.',
          'Length/quality filtering and source reweighting accompany DSIR-style selection.',
          'Density features fit training continuation windows; scoring full-document body samples can shift statistics.',
          'Strict overlap union can overexclude useful legal boilerplate.',
          'Manual review and realized token-mixture simulation are required before training.']}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'selected.jsonl').open('w') as f:
        for row in chosen:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if enough:
        d=out/'source';d.mkdir()
        # Keep original offsets into original tokens; only the index is a subset.
        # A NEW source name is required to avoid reinterpreting resumed old indices.
        np.save(d/'index.npy',index[[r['index'] for r in chosen]])
        os.symlink(old/'train/high_courts/tokens.bin',d/'tokens.bin')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
