#!/usr/bin/env python
"""Decontaminated restored gold continuations and deterministic entity negatives.

These are training candidates, not claimed human preferences or factuality labels.
No external model or evaluation-model features enter this builder.
"""
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from saul.data.split import cut_dev_example
from saul.utils.io import iter_shards, read_json, write_json

spec=importlib.util.spec_from_file_location('curate',Path(__file__).with_name('94_recovery_curate.py'))
curate=importlib.util.module_from_spec(spec);spec.loader.exec_module(curate)
ENTITY=re.compile(r"\b(?:[A-Z][a-z]{2,}(?:[ '-][A-Z][a-z]{2,})+|\d{2,}(?:[./-]\d+)*)(?!\w)")


def shared_entities(prompt,target):
    return sorted({m.group() for m in ENTITY.finditer(target)
                   if re.search(r'(?<!\w)'+re.escape(m.group())+r'(?!\w)',prompt)},key=lambda s:(-len(s),s))


def negative(row):
    for entity in shared_entities(row['prompt'],row['target']):
        if entity[0].isdigit():
            replacement=str(int(entity[0])+1)[-1]+entity[1:]
        else:
            # A spelling mutation isolates identity retention; no outcome reversal
            # heuristic, which can mislabel quoted/opposing legal propositions.
            last=entity[-1];replacement=entity[:-1]+('a' if last!='a' else 'e')
        if replacement in row['prompt'] or replacement in row['target']:continue
        rejected=re.sub(r'(?<!\w)'+re.escape(entity)+r'(?!\w)',lambda _:replacement,row['target'])
        assert rejected!=row['target'] and len(rejected.split())==128
        return dict(id=row['id'],cluster=row['cluster'],prompt=row['prompt'],chosen=row['target'],
                    rejected=rejected,entity=entity,replacement=replacement,
                    label_source='official gold versus programmatic identity corruption',
                    source_doc_sha256=row['source_doc_sha256'])


def main():
    out=Path('runs/recovery_20261005/entity_recovery');out.mkdir(parents=True,exist_ok=False)
    blocked=set()
    for path in ['data/raw/Phase_1_Pretraining/Test/test.json']:
        for r in read_json(path).values():blocked.update(curate.grams(r['text']))
    dp=read_json('data/final/dev_prompts.json');dt=read_json('data/final/dev_truth.json')
    for k in dp:blocked.update(curate.grams(dp[k]['text']+' '+dt[k]['text']))
    held=set()
    for split in ['val','dev']:
        for src in curate.SOURCES:
            for r in iter_shards(f'data/final/{split}/{src}'):held.add(f"{src}:{r['meta']['cluster']}")
    counts=collections.Counter()
    for src in curate.SOURCES:
        for i,r in enumerate(iter_shards(f'data/final/train/{src}')):
            if i>=1000:break
            counts.update(curate.grams(r['text'])&blocked)
    boiler={g for g,n in counts.items() if n>=50};blocked-=boiler
    from transformers import PreTrainedTokenizerFast
    tok=PreTrainedTokenizerFast.from_pretrained('artifacts/tokenizers/legal32k')
    stats=collections.Counter();random_rows=[];entity_rows=[];used=set()
    def encode(r,ex,seed):
        prompt,target=ex
        enc=tok(prompt+' '+target,add_special_tokens=False,return_offsets_mapping=True)
        if len(enc['input_ids'])+1>2048:return None
        first=next(i for i,(_,end) in enumerate(enc['offset_mapping']) if end>len(prompt))
        return dict(id=r['id'],source=r['source'],cluster=f"supreme_court:{r['meta']['cluster']}",
                    prompt=prompt,target=target,input_ids=[tok.bos_token_id]+enc['input_ids'],target_start=first+1,
                    source_doc_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                    generator=f'gold_cut_{seed}',split='train',restored=True,
                    shared_entities=shared_entities(prompt,target))
    source='/data/cs781-archive/recovery_20261005/restored_sc/recovered_train/supreme_court'
    for r in iter_shards(source):
        stats['examined']+=1;c=f"supreme_court:{r['meta']['cluster']}"
        if c in held or c in used:stats['cluster_dropped']+=1;continue
        if len(curate.grams(r['text'])&blocked)>=3:stats['overlap_dropped']+=1;continue
        windows=[]
        for seed in [1781,2781,3781,4781]:
            ex=cut_dev_example(r['text'],r['id'],seed=seed)
            if ex:
                row=encode(r,ex,seed)
                if row:windows.append(row)
        if not windows:stats['short_or_long_dropped']+=1;continue
        random_rows.append(windows[0])
        entity_rows.append(max(windows,key=lambda x:sum(len(e.split()) for e in x['shared_entities'])))
        used.add(c)
        if stats['examined']%500==0:print(dict(stats),flush=True)
    # Keep original source proportion approximately 78/8/7/7, with identical
    # unaffected-source replay across both restored-data variants.
    replay=[]
    for src,share in [('acts',.08),('tribunals',.07),('high_courts',.07)]:
        pool=[json.loads(l) for l in Path(f'runs/recovery_20261005/curation/filtered_{src}.jsonl').read_text().splitlines()]
        pool=[r for r in pool if r['cluster'] not in held and r['cluster'] not in used]
        pick=sorted(pool,key=lambda r:curate.hashed('restore:'+r['id']))[:round(len(random_rows)*share/.78)]
        replay.extend(pick);used.update(r['cluster'] for r in pick)
    variants={'random':random_rows+replay,'entity':entity_rows+replay}
    for name,rows in variants.items():
        rows.sort(key=lambda r:curate.hashed('restore-order:'+r['id']))
        assert len({r['cluster'] for r in rows})==len(rows)
        with (out/f'{name}.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    prefs=[p for r in entity_rows if (p:=negative(r))]
    # Cluster-disjoint held-out diagnostics for the synthetic task, never mixed
    # into the Phase-1 dev set or used to claim actual entity factuality gains.
    for name,want_hold in [('preferences_train',False),('preferences_holdout',True)]:
        with (out/f'{name}.jsonl').open('w') as f:
            for p in prefs:
                if (curate.hashed('pref-split:'+p['cluster'])%10==0)==want_hold:
                    f.write(json.dumps(p,ensure_ascii=False)+'\n')
    stats.update(accepted_restored=len(random_rows),replay=len(replay),synthetic_pairs=len(prefs))
    write_json(dict(stats=stats,boilerplate_removed=len(boiler),
        variant_counts={k:len(v) for k,v in variants.items()},
        entity_window_counts={k:sum(bool(r.get('shared_entities')) for r in v) for k,v in variants.items()},
        decontamination='full-document >=3 non-boilerplate normalized 13gram union overlaps with test prompts or dev prompt/truth; preserve official train clusters',
        experiments='matched original random window vs max shared entity evidence among four gold windows; same restored documents and unaffected-source replay',
        limitations=['Entity regex is a proxy and includes institutions, not a named-entity annotation.',
                    'Programmatic preference labels test copy consistency, not general legal reasoning.',
                    'No test truth or evaluation-model rewards used.']),str(out/'manifest.json'))
    print('COMPLETE',dict(stats),flush=True)


if __name__=='__main__':main()
