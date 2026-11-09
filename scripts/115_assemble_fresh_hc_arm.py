#!/usr/bin/env python
"""Assemble and simulate the fresh-HC arm; never starts GPU training.

Keep original sources and saved indices. New body documents use a fifth source
with fresh default cursors, so full optimizer/data-state resume remains defined.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import collections
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from saul.data.tokens import open_split
from saul.data.loader import PackedLMDataset,_WorkerStream


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--base-tokens',default=str(ROOT/'data/tokens/legal32k'))
    p.add_argument('--restored-spans')
    p.add_argument('--fresh-share',type=float,default=.12)
    p.add_argument('--restored-share',type=float,default=0.)
    p.add_argument('--tag',default='fresh_hc_bodies')
    p.add_argument('--run-name',default='hybrid_b_fresh_hc_anneal')
    a=p.parse_args();os.chdir(ROOT)
    source=Path(a.input).resolve();out=Path(a.out).resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='ready_for_review' and manifest['selected_tokens']>=120000000
    out.mkdir(parents=True,exist_ok=False);(out/'train').mkdir()
    old=Path(a.base_tokens).resolve()
    assert 0<a.fresh_share<.82 and 0<=a.restored_share<.82-a.fresh_share
    assert bool(a.restored_spans)==(a.restored_share>0)
    shares=json.loads((old/'token_shares.json').read_text())
    for name in shares['train']:os.symlink(old/'train'/name,out/'train'/name,target_is_directory=True)
    os.symlink(source/'source',out/'train/high_courts_fresh',target_is_directory=True)
    index=np.load(source/'source/index.npy')
    assert len(index)==manifest['selected_docs'] and int(index[:,1].sum())==manifest['selected_tokens']
    shares['train']['high_courts_fresh']={'docs':len(index),'tokens':int(index[:,1].sum()),
        'bytes':int(index[:,2].sum()),'anonymized_docs':0}
    extra_manifest=None
    if a.restored_spans:
        extra=Path(a.restored_spans).resolve()
        extra_manifest=json.loads((extra/'manifest.json').read_text())
        assert extra_manifest['status']=='ready_for_review' and extra_manifest['all_fragments_zero_blocked_13grams']
        extra_index=np.load(extra/'source/index.npy')
        assert len(extra_index)==extra_manifest['selected_docs']
        assert int(extra_index[:,1].sum())==extra_manifest['selected_tokens']>0
        os.symlink(extra/'source',out/'train/supreme_court_restored_spans',target_is_directory=True)
        shares['train']['supreme_court_restored_spans']={'docs':len(extra_index),'tokens':int(extra_index[:,1].sum()),
            'bytes':int(extra_index[:,2].sum()),'anonymized_docs':0}
    (out/'token_shares.json').write_text(json.dumps(shares,indent=2)+'\n')
    for name in ['val','token_bytes.npy','tokenizer_meta.json']:
        os.symlink(old/name,out/name,target_is_directory=name=='val')
    desired={'supreme_court':.82-a.fresh_share-a.restored_share,'acts':.14,'tribunals':.03,
             'high_courts':.01,'high_courts_fresh':a.fresh_share}
    if a.restored_spans:desired['supreme_court_restored_spans']=a.restored_share
    raw={s:desired[s]/(shares['train'][s]['tokens']/shares['train'][s]['docs']) for s in desired}
    draws={s:v/sum(raw.values()) for s,v in raw.items()}
    data={'_base_':'data.yaml','sources':list(draws),'mixture':{'anneal':draws,'anneal_use_test_mix':False,'anneal_min_share':0.0}}
    data_path=ROOT/f'configs/data_{a.tag}_20261006.yaml'
    train_path=ROOT/f'configs/train/recovery_{a.tag}_anneal.yaml'
    assert not data_path.exists() and not train_path.exists()
    data_path.write_text(yaml.safe_dump(data,sort_keys=False))
    config={'_base_':'recovery_b_anneal.yaml','tokens':str(out),'data_config':str(data_path.relative_to(ROOT)),
        'run_dir':f'/data/cs781-archive/recovery_20261006/{a.run_name}'}
    train_path.write_text(yaml.safe_dump(config,sort_keys=False))
    from saul.utils.config import load_config
    spec=importlib.util.spec_from_file_location('train30',ROOT/'scripts/30_train.py')
    train30=importlib.util.module_from_spec(spec);spec.loader.exec_module(train30)
    production_weights=train30._weights(load_config(str(train_path)))
    assert set(production_weights)==set(draws)
    assert all(abs(production_weights[s]-draws[s])<1e-12 for s in draws)
    spec=importlib.util.spec_from_file_location('audit111',ROOT/'scripts/111_audit_data_exposure.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    sources=open_split(str(out),'train',list(draws))
    ds=PackedLMDataset(sources,draws,2048,0,1,2,seed=781)
    st=torch.load(ROOT/'runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',map_location='cpu',weights_only=False,mmap=True)
    workers={int(k):v for k,v in st['loader']['workers'].items()}
    assert st['loader']['num_workers']==4
    real=_WorkerStream(ds,0,4,workers[0]);shadow=_WorkerStream(ds,0,4,workers[0])
    for _ in range(32):
        real.sequence();audit.simulate_one(shadow,collections.Counter());assert real.state()==shadow.state()
    counts=collections.Counter();ends={}
    for w,ws in workers.items():
        stream=_WorkerStream(ds,w,4,ws)
        for _ in range(1800*8*32//4):audit.simulate_one(stream,counts)
        ends[w]={'fresh_epoch':stream.epoch['high_courts_fresh'],'fresh_cursor':stream.cursor['high_courts_fresh']}
        if a.restored_spans:
            ends[w].update(restored_epoch=stream.epoch['supreme_court_restored_spans'],
                           restored_cursor=stream.cursor['supreme_court_restored_spans'])
        print('simulated worker',w,ends[w],flush=True)
    assert sum(counts.values())==1800*524288
    fractions={s:n/sum(counts.values()) for s,n in counts.items()}
    max_error=max(abs(fractions[s]-desired[s]) for s in desired)
    assert max_error<.02,('Realized token shares differ by more than 2 percentage points',fractions)
    result={'status':'prepared_not_launched','desired_token_shares':desired,'document_probabilities':draws,
        'simulated_full_anneal_token_shares':fractions,'simulated_full_anneal_tokens':dict(counts),
        'max_absolute_share_error':max_error,'fresh_stream_end_states':ends,
        'fresh_source_has_started_second_epoch':any(v['fresh_epoch']>0 for v in ends.values()),
        'production_state_matches_32_sequences':True,'train_config':str(train_path),'data_config':str(data_path),
        'base_tokens':str(old),'restored_spans':a.restored_spans,
        'additional_source_manifest':extra_manifest,
        'limitations':['New arm combines quality/body selection and source reweighting.',
            'Training validation mixture differs from the original; compare using fixed generation dev sets.',
            'Original HC source can independently read some selected documents; no semantic novelty guarantee.',
            'Source replacement and mixture changes are combined; interpret against the specified incumbent.']}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
