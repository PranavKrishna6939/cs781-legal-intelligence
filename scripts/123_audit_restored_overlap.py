#!/usr/bin/env python
"""CPU-only diagnosis of conservative whole-document recovery exclusions.

Does not admit documents, prepare tokens, or relax any current training gate.
Potential remaining spans are counts only, after removing every matched window
plus a 64-normalized-word margin. Semantic/cluster review would still be needed.
"""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from saul.utils.io import iter_shards,read_json
spec=importlib.util.spec_from_file_location('curate94',ROOT/'scripts/94_recovery_curate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
OUT=Path('/data/cs781-archive/recovery_20261006/restoration_overlap_audit')


def merge(intervals):
    result=[]
    for left,right in sorted(intervals):
        if result and left<=result[-1][1]:result[-1][1]=max(result[-1][1],right)
        else:result.append([left,right])
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    public=set();dev=set()
    for row in read_json(ROOT/'data/raw/Phase_1_Pretraining/Test/test.json').values():public.update(c.grams(row['text']))
    dp=read_json(ROOT/'data/final/dev_prompts.json');dt=read_json(ROOT/'data/final/dev_truth.json')
    for key in dp:dev.update(c.grams(dp[key]['text']+' '+dt[key]['text']))
    blocked=public|dev;counts=collections.Counter()
    for src in c.SOURCES:
        for i,row in enumerate(iter_shards(str(ROOT/f'data/final/train/{src}'))):
            if i>=1000:break
            counts.update(c.grams(row['text'])&blocked)
    boiler={g for g,n in counts.items() if n>=50}
    blocked-=boiler;public-=boiler;dev-=boiler
    held=set()
    for split in ['val','dev']:
        for row in iter_shards(str(ROOT/f'data/final/{split}/supreme_court')):held.add(row['meta']['cluster'])
    accepted={row['id'] for row in map(json.loads,(ROOT/'runs/recovery_20261005/entity_recovery/random.jsonl').read_text().splitlines()) if row.get('restored')}
    stats=collections.Counter();records=[]
    source='/data/cs781-archive/recovery_20261005/restored_sc/recovered_train/supreme_court'
    for row in iter_shards(source):
        stats['examined']+=1
        if row['id'] in accepted:stats['already_admitted']+=1;continue
        if row['meta']['cluster'] in held:stats['held_cluster_excluded']+=1;continue
        words=re.findall(r'\w+',row['text'].lower())
        hashes=[c.hashed(' '.join(words[i:i+13])) for i in range(len(words)-12)]
        positions=[i for i,g in enumerate(hashes) if g in blocked]
        distinct={hashes[i] for i in positions}
        if len(distinct)<3:stats['unselected_without_overlap_gate']+=1;continue
        stats['overlap_excluded_docs']+=1;stats['overlap_excluded_normalized_words']+=len(words)
        spans=merge([(i,i+13) for i in positions]); covered=sum(r-l for l,r in spans)
        expanded=merge([(max(0,l-64),min(len(words),r+64)) for l,r in spans])
        remaining=[];end=0
        for left,right in expanded:
            if left>end:remaining.append(left-end)
            end=right
        if end<len(words):remaining.append(len(words)-end)
        long_run=run=0;last=None
        for i in positions:
            run=run+1 if last is not None and i==last+1 else 1
            long_run=max(long_run,run);last=i
        covered_fraction=covered/max(1,len(words))
        stats['matched_normalized_words']+=covered
        stats['coverage_under_1pct_docs']+=int(covered_fraction<.01)
        stats['coverage_under_5pct_docs']+=int(covered_fraction<.05)
        stats['coverage_over_50pct_docs']+=int(covered_fraction>.5)
        stats['potential_remaining_words_in_690plus_spans']+=sum(n for n in remaining if n>=690)
        stats['potential_remaining_690plus_spans']+=sum(n>=690 for n in remaining)
        records.append(dict(id=row['id'],cluster=row['meta']['cluster'],normalized_words=len(words),
            unique_matched_13grams=len(distinct),public_matched_13grams=len(distinct&public),
            dev_matched_13grams=len(distinct&dev),matched_word_fraction=covered_fraction,
            longest_consecutive_blocked_window_words=long_run+12,
            potential_remaining_690plus_words=sum(n for n in remaining if n>=690)))
        if stats['examined']%500==0:print(json.dumps(dict(stats)),flush=True)
    result=dict(stats=dict(stats),boilerplate_exemptions=len(boiler),
        status='audit_only_no_new_training_data',margin_normalized_words=64,
        limitations=['Union overlap is not a single-reference longest-match alignment.',
            'Short shared legal quotations and near-duplicate cases are different possibilities; coverage alone does not certify safety.',
            'Remaining-span counts are not approved training text, not token counts, and not evidence of score improvement.',
            'No active corpus, source proportions, or queued experiment was changed.'])
    (OUT/'records.json').write_text(json.dumps(records))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
