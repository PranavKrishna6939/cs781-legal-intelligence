#!/usr/bin/env python
"""Prepare conservative surviving spans from previously collapsed SC records.

Every admitted fragment independently has ZERO non-boilerplate blocked 13grams.
Original held-out clusters stay excluded. This prepares data for review only.
"""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from saul.utils.io import iter_shards,read_json
from saul.tokenizer.train import load_raw_tokenizer
spec=importlib.util.spec_from_file_location('audit123',ROOT/'scripts/123_audit_restored_overlap.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
c=a.c
OUT=Path('/data/cs781-archive/recovery_20261006/restored_sc_spans')


def complements(intervals,n):
    result=[];end=0
    for left,right in a.merge(intervals):
        if left>end:result.append((end,left))
        end=right
    if end<n:result.append((end,n))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'source').mkdir()
    blocked=set()
    for row in read_json(ROOT/'data/raw/Phase_1_Pretraining/Test/test.json').values():blocked.update(c.grams(row['text']))
    dp=read_json(ROOT/'data/final/dev_prompts.json');dt=read_json(ROOT/'data/final/dev_truth.json')
    for key in dp:blocked.update(c.grams(dp[key]['text']+' '+dt[key]['text']))
    counts=collections.Counter()
    for src in c.SOURCES:
        for i,row in enumerate(iter_shards(str(ROOT/f'data/final/train/{src}'))):
            if i>=1000:break
            counts.update(c.grams(row['text'])&blocked)
    boiler={g for g,n in counts.items() if n>=50};blocked-=boiler
    held=set()
    for split in ['val','dev']:
        for row in iter_shards(str(ROOT/f'data/final/{split}/supreme_court')):held.add(row['meta']['cluster'])
    audit=read_json(a.OUT/'records.json')
    old={r['id']:r for r in read_json('/data/cs781-archive/recovery_20261005/restored_sc/restored_records.json')}
    eligible={r['id']:r for r in audit if old[r['id']]['old_words']<64 and
              r['matched_word_fraction']<.05 and r['longest_consecutive_blocked_window_words']<=128}
    assert len(eligible)==596
    admitted={r['id'] for r in map(json.loads,(ROOT/'runs/recovery_20261005/entity_recovery/random.jsonl').read_text().splitlines()) if r.get('restored')}
    assert not set(eligible)&admitted
    tok=load_raw_tokenizer(str(ROOT/'artifacts/tokenizers/legal32k'))
    bos,eos=tok.token_to_id('<|bos|>'),tok.token_to_id('<|eos|>')
    stats=collections.Counter();records=[];index=[];position=0
    source='/data/cs781-archive/recovery_20261005/restored_sc/recovered_train/supreme_court'
    with (OUT/'source/tokens.bin').open('wb') as tokens,(OUT/'fragments.jsonl').open('w') as textout:
        for row in iter_shards(source):
            if row['id'] not in eligible:continue
            stats['eligible_source_docs']+=1
            assert row['meta']['cluster'] not in held
            original=row['text'];lower=original.lower()
            # Exclude exceptional lowercase expansions rather than risk wrong character offsets.
            if len(lower)!=len(original):stats['unicode_offset_docs_excluded']+=1;continue
            matches=list(re.finditer(r'\w+',lower));words=[m.group() for m in matches]
            hit=[i for i in range(len(words)-12) if c.hashed(' '.join(words[i:i+13])) in blocked]
            excluded=a.merge([(max(0,i-64),min(len(words),i+13+64)) for i in hit])
            source_sha=hashlib.sha256(original.encode()).hexdigest()
            kept_doc=False
            for left,right in complements(excluded,len(words)):
                if right-left<690:continue
                char_left=0 if left==0 else matches[left].start()
                char_right=len(original) if right==len(words) else matches[right].start()
                fragment=original[char_left:char_right].strip()
                whitespace=fragment.split()
                if len(whitespace)<690:stats['short_whitespace_spans_excluded']+=1;continue
                # This independent check also catches any normalization/boundary mismatch.
                assert not c.grams(fragment)&blocked,('Residual blocked span',row['id'],left,right)
                letters=[ch for ch in fragment if ch.isalpha()]
                latin=sum('a'<=ch.lower()<='z' for ch in letters)/max(1,len(letters))
                caps=sum(ch.isupper() for ch in letters)/max(1,len(letters))
                broken=len(re.findall(r'\b[b-hj-zB-HJ-Z]\s+[a-z]{2,}\b',fragment))/len(whitespace)
                if latin<.95 or caps>.30 or broken>.005:
                    stats['quality_spans_excluded']+=1;continue
                ids=np.asarray([bos]+tok.encode(fragment,add_special_tokens=False).ids+[eos],dtype=np.uint16)
                rid=f"{row['id']}:span:{left}:{right}"
                record=dict(id=rid,source_id=row['id'],cluster=f"supreme_court:{row['meta']['cluster']}",
                    original_source_sha256=source_sha,original_char_bounds=[char_left,char_right],
                    normalized_word_bounds=[left,right],tokens=len(ids),words=len(whitespace),
                    fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
                    source_overlap_fraction=eligible[row['id']]['matched_word_fraction'],
                    zero_blocked_13grams=True,excerpt=' '.join(whitespace[:150]),
                    middle_excerpt=' '.join(whitespace[len(whitespace)//2:len(whitespace)//2+100]))
                records.append(record);textout.write(json.dumps(dict(**record,text=fragment),ensure_ascii=False)+'\n')
                tokens.write(ids.tobytes());index.append([position,len(ids),len(fragment.encode()),0]);position+=len(ids)
                stats['accepted_fragments']+=1;stats['accepted_words']+=len(whitespace);kept_doc=True
            stats['source_docs_with_fragments']+=int(kept_doc)
    np.save(OUT/'source/index.npy',np.asarray(index,dtype=np.int64))
    (OUT/'records.json').write_text(json.dumps(records))
    order=sorted(records,key=lambda r:c.hashed('span-review781:'+r['id']))
    sample=order[:10]+sorted(records,key=lambda r:r['tokens'],reverse=True)[:2]
    (OUT/'manual_sample.json').write_text(json.dumps(sample,indent=2))
    result=dict(status='ready_for_review',selected_docs=len(index),selected_tokens=position,
        stats=dict(stats),margin_normalized_words=64,boilerplate_exemptions=len(boiler),
        eligible_original_documents=596,all_fragments_zero_blocked_13grams=True,
        source='official restored training SC documents previously collapsed below 64 words',
        limitations=['Fragments remove matched passages and nearby context; they are not full judgments.',
            'Exact overlap and original held-cluster checks do not prove absence of every semantic duplicate.',
            'No active corpus was changed and no training job was launched.'])
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
