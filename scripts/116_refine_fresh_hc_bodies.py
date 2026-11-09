#!/usr/bin/env python
"""Refine already screened HC records into prose bodies before DSIR selection.

CPU only. Inherits whole-document decontamination from 114. Keeps source IDs and
cluster lineage, and never changes the original corpus or active training run.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from saul.utils.io import iter_shards
from saul.tokenizer.train import load_raw_tokenizer
spec=importlib.util.spec_from_file_location('fresh114',ROOT/'scripts/114_prepare_fresh_hc.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
TITLE=re.compile(r'\b(?:COMMON JUDG[E]?MENT|J U D G (?:E )?M E N T|JUDG[E]?MENT|COMMON ORDER|C O M M O N O R D E R|O R D E R|ORDER)\b')


def extract(text):
    for m in TITLE.finditer(text):
        if m.start()>=.75*len(text):break
        before=text[max(0,m.start()-50):m.start()]
        after=text[m.end():]
        # Indian court captions often cite the challenged JUDGMENT/ORDER IN ...
        # before the parties. Those references are not the current body heading.
        if re.search(r'(?:AGAINST|IMPUGNED|COPY OF)\s+(?:THE\s+)?$',before):continue
        if re.match(r'\s*(?:/|IN\s+(?:WPC|WP|WA|W\.P|CIVIL|CRIMINAL)|DATED\b)',after):continue
        return text[m.end():].lstrip(' :.-\n'),len(text[:m.end()].split()),'first_uppercase_judgment_or_order_title'
    return text,0,'no_early_title'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--target-tokens',type=int,default=120000000)
    a=p.parse_args();os.chdir(ROOT)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    prev=Path(a.input)
    candidates={r['index']:{k:r[k] for k in ['id','index','cluster']} for r in map(json.loads,(prev/'eligible_candidates.jsonl').open())}
    oldindex=np.load('data/tokens/legal32k/train/high_courts/index.npy')
    vec=HashingVectorizer(n_features=10000,ngram_range=(1,2),alternate_sign=False,norm=None)
    fits={}
    for src in ['supreme_court','high_courts']:
        rows=[json.loads(l) for l in Path(f'runs/recovery_20261005/curation/filtered_{src}.jsonl').read_text().splitlines()]
        rows=sorted(rows,key=lambda r:f.c.hashed('fresh-fit:'+r['id']))[:1000]
        freq=np.asarray(vec.transform([f.body(r['prompt']+' '+r['target']) for r in rows]).sum(axis=0)).ravel()+1
        fits[src]=freq/freq.sum()
    ratio=np.log(fits['supreme_court'])-np.log(fits['high_courts'])
    tok=load_raw_tokenizer('artifacts/tokenizers/legal32k')
    bos,eos=tok.token_to_id('<|bos|>'),tok.token_to_id('<|eos|>')
    stats=collections.Counter();batch=[];rows=[];offset=0
    pool_path=out/'candidate_tokens.bin'
    with pool_path.open('wb') as pool:
        def flush():
            nonlocal offset
            if not batch:return
            matrix=vec.transform([f.body(r['text']) for r in batch])
            scores=np.asarray(matrix@ratio).ravel()/np.maximum(1,np.asarray(matrix.sum(axis=1)).ravel())
            encs=tok.encode_batch([r['text'] for r in batch],add_special_tokens=False)
            for r,score,enc in zip(batch,scores,encs):
                ids=np.asarray([bos]+enc.ids+[eos],dtype=np.uint16)
                if len(ids)<512:stats['short_tokens_dropped']+=1;continue
                u=min(1-1e-15,max(1e-15,(f.c.hashed('fresh-select:'+r['id'])+.5)/2**64))
                words=r['text'].split()
                row={k:v for k,v in r.items() if k!='text'}
                row.update(offset=offset,tokens=len(ids),bytes=len(r['text'].encode()),density=float(score),
                    rank=float(score/.1-math.log(-math.log(u))),
                    excerpt=' '.join(words[:160]),middle_excerpt=' '.join(words[len(words)//2:len(words)//2+100]),
                    body_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
                rows.append(row);pool.write(ids.tobytes());offset+=len(ids);stats['accepted']+=1
            batch.clear()
        for j,r in enumerate(iter_shards('data/final/train/high_courts')):
            if j not in candidates:continue
            stats['examined']+=1
            assert r['id']==candidates[j]['id'] and r['meta']['cluster']==candidates[j]['cluster']
            assert len(r['text'].encode())==int(oldindex[j,2])
            text,removed,rule=extract(r['text'])
            words=text.split()
            letters=[ch for ch in text if ch.isalpha()]
            caps=sum(ch.isupper() for ch in letters)/max(1,len(letters))
            latin=sum('a'<=ch.lower()<='z' for ch in letters)/max(1,len(letters))
            broken=len(re.findall(r'\b[b-hj-zB-HJ-Z]\s+[a-z]{2,}\b',text))/max(1,len(words))
            if len(words)<690:stats['short_body_dropped']+=1;continue
            if latin<.95:stats['non_english_script_dropped']+=1;continue
            if caps>.30:stats['uppercase_body_dropped']+=1;continue
            if broken>.015:stats['fragmented_ocr_dropped']+=1;continue
            stats['caption_removed_docs']+=int(removed>0)
            stats['caption_removed_words']+=removed
            batch.append(dict(id=r['id'],cluster=r['meta']['cluster'],original_index=j,text=text,
                caption_words_removed=removed,cut_rule=rule,uppercase_letter_fraction=caps,
                fragmented_word_proxy=broken,original_tokens=int(oldindex[j,1])))
            if len(batch)>=128:flush()
            if stats['examined']%5000==0:print(dict(stats),flush=True)
        flush()
    with (out/'eligible_bodies.jsonl').open('w') as file:
        for r in rows:file.write(json.dumps(r,ensure_ascii=False)+'\n')
    chosen=[];total=0
    for r in sorted(rows,key=lambda r:r['rank'],reverse=True):
        chosen.append(r);total+=r['tokens']
        if total>=a.target_tokens:break
    enough=total>=a.target_tokens
    if enough:
        source=out/'source';source.mkdir()
        pool=np.memmap(pool_path,dtype=np.uint16,mode='r')
        idx=[];pos=0
        with (source/'tokens.bin').open('wb') as file:
            for r in chosen:
                ids=pool[r['offset']:r['offset']+r['tokens']]
                assert int(ids[0])==bos and int(ids[-1])==eos
                file.write(ids.tobytes());idx.append([pos,len(ids),r['bytes'],0]);pos+=len(ids)
        np.save(source/'index.npy',np.asarray(idx,dtype=np.int64))
        assert pos==total and (source/'tokens.bin').stat().st_size==2*total
        del pool
    # Candidate pool is retained for reproducibility, under this task's archive.
    with (out/'selected.jsonl').open('w') as file:
        for r in chosen:file.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest=dict(status='ready_for_review' if enough else 'insufficient_tokens',stats=dict(stats),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_docs=len(chosen),selected_tokens=total,candidate_tokens=offset,
        selected_mean_density=float(np.mean([r['density'] for r in chosen])) if chosen else None,
        candidate_mean_density=float(np.mean([r['density'] for r in rows])) if rows else None,
        parent_manifest=str(prev/'manifest.json'),not_launched=True,
        limitations=['Title detection is heuristic; inspect sampled starts and middles before training.',
            'Prose-body selection removes some valid short orders and caption-heavy material from this added source only.',
            'This combines body extraction, quality filtering and DSIR-style selection; it is not an isolated DSIR causal experiment.',
            'Density fit rows, whole-document overlap exclusions and cluster separation are inherited from the parent curation.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
