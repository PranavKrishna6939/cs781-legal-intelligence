#!/usr/bin/env python
"""Prepare an isolated CLM corpus using already decontaminated restored records.

Keep the original 50,134 document indices and all unaffected tokens. Only restore
IDs accepted by the existing full-document overlap/cluster audit. This deliberately
does not claim to restore all 5,984 records or certify the legacy corpus anew.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from saul.data.tokens import open_source
from saul.tokenizer.train import load_raw_tokenizer
from saul.utils.io import iter_shards


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    old=Path('data/tokens/legal32k').resolve()
    selected={r['id']:r for r in map(json.loads,Path('runs/recovery_20261005/entity_recovery/random.jsonl').read_text().splitlines()) if r.get('restored')}
    restored={r['id']:r for r in iter_shards('/data/cs781-archive/recovery_20261005/restored_sc/recovered_train/supreme_court') if r['id'] in selected}
    assert len(restored)==len(selected)==2872
    for rid,r in restored.items():
        assert hashlib.sha256(r['text'].encode()).hexdigest()==selected[rid]['source_doc_sha256']
        assert f"supreme_court:{r['meta']['cluster']}"==selected[rid]['cluster']
    tok=load_raw_tokenizer('artifacts/tokenizers/legal32k')
    bos,eos=tok.token_to_id('<|bos|>'),tok.token_to_id('<|eos|>')
    src=open_source(str(old),'train','supreme_court')
    dest=out/'train/supreme_court'
    dest.mkdir(parents=True)
    idx=src.index.copy()
    counts=np.load(old/'train_token_counts.npy').copy()
    stats=collections.Counter()
    changed=[]
    ids=[]
    position=0
    with (dest/'tokens.bin').open('wb') as f:
        for j,r in enumerate(iter_shards('data/final/train/supreme_court')):
            assert j<src.n_docs
            ids.append(r['id'])
            before=src.doc(j)
            assert len(r['text'].encode())==int(src.index[j,2]), ('index order mismatch',j)
            if r['id'] in restored:
                encoded=np.asarray([bos]+tok.encode(r['text'],add_special_tokens=False).ids+[eos],dtype=np.uint16)
                assert np.array_equal(encoded,before), ('token order mismatch',j)
                replacement=restored[r['id']]['text']
                after=np.asarray([bos]+tok.encode(replacement,add_special_tokens=False).ids+[eos],dtype=np.uint16)
                idx[j,2]=len(replacement.encode())
                idx[j,3]=0
                stats['replaced_docs']+=1
                stats['collapsed_docs_repaired']+=int(len(r['text'].split())<=2)
                stats['old_tokens_in_replaced_docs']+=len(before)
                stats['new_tokens_in_replaced_docs']+=len(after)
                counts+=np.bincount(after,minlength=len(counts))-np.bincount(before,minlength=len(counts))
                changed.append(j)
            else:
                after=before
            idx[j,0],idx[j,1]=position,len(after)
            f.write(after.tobytes())
            position+=len(after)
            if j and j%10000==0:print('processed',j,dict(stats),flush=True)
    assert len(ids)==src.n_docs and len(set(ids))==len(ids)
    assert stats['replaced_docs']==len(selected)
    assert (counts>=0).all()
    np.save(dest/'index.npy',idx)
    np.save(out/'train_token_counts.npy',counts)
    (out/'supreme_court_ids.json').write_text(json.dumps(ids))
    shares=json.loads((old/'token_shares.json').read_text())
    shares['train']['supreme_court']={'docs':len(idx),'tokens':position,'bytes':int(idx[:,2].sum()),'anonymized_docs':int(idx[:,3].sum())}
    for name in shares['train']:
        if name!='supreme_court':os.symlink(old/'train'/name,out/'train'/name,target_is_directory=True)
    for name in ['val','token_bytes.npy','tokenizer_meta.json']:
        os.symlink(old/name,out/name,target_is_directory=(name=='val'))
    (out/'token_shares.json').write_text(json.dumps(shares,indent=2)+'\n')
    # Same index/order permits full optimizer and data-cursor resume. Validate all
    # pending byte-independent token offsets against the replacement document.
    import torch
    st=torch.load('runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',map_location='cpu',weights_only=False,mmap=True)
    for ws in st['loader']['workers'].values():
        for name,j,off in ws['pending']:
            if name=='supreme_court':assert 0<=off<int(idx[j,1])
    means={name:v['tokens']/v['docs'] for name,v in shares['train'].items()}
    original_shares=json.loads((old/'token_shares.json').read_text())
    # Preserve expected token exposure of the winning document-draw mixture.
    from saul.data.stats import anneal_weights
    mixture=json.loads(Path('data/stats/test_source_mix.json').read_text())
    raw=anneal_weights({name:1 for name in means},mixture,.08)
    corrected={name:raw[name]*(original_shares['train'][name]['tokens']/original_shares['train'][name]['docs'])/means[name] for name in means}
    total=sum(corrected.values())
    corrected={name:v/total for name,v in corrected.items()}
    manifest={'status':'complete','stats':dict(stats),'original_tokens':src.n_tokens,'repaired_tokens':position,
        'added_tokens':position-src.n_tokens,'document_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'changed_document_indices':changed,'suggested_document_weights_preserving_expected_token_mix':corrected,
        'pending_offsets_valid':True,'all_changed_original_encodings_verified':True,
        'decontamination':'Reuse 2872 restored IDs accepted by scripts/104_curate_entity_recovery.py; full-document non-boilerplate 13grams and original held-out cluster exclusions.',
        'limitations':['Partial recovery: 2872/5984 restored records; original SFT length filter also limits this safe whitelist.',
                      'Legacy unaffected corpus is unchanged and not newly certified decontaminated.',
                      'Resume offsets in changed documents refer to changed text; document IDs and order are identical.',
                      'Corrected source weights preserve expected long-run token shares, not exact sample identities.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='changed_document_indices'},indent=2),flush=True)


if __name__=='__main__':main()
