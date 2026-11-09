#!/usr/bin/env python
"""Reconstruct original SC train documents with original clean IDs and split membership.

Reads only the official training source. Verifies every unmasked surviving document
byte-for-byte before declaring recovery complete. Never mutates the live token corpus.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from saul.data.clean import build_text,clean_document
from saul.data.stream import iter_json_array
from saul.utils.io import iter_shards,ShardWriter,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='/data/cs781-archive/recovery_20261005/official_raw/Phase_1_Pretraining/Train/train_supreme_court.json')
    p.add_argument('--out',default='/data/cs781-archive/recovery_20261005/restored_sc')
    a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    assert Path(a.raw).stat().st_size==1322567168
    existing={}
    for split in ['train','val','dev']:
        for r in iter_shards(f'data/final/{split}/supreme_court'):
            existing[r['id']]=dict(split=split,meta=r['meta'],sha=hashlib.sha256(r['text'].encode()).hexdigest(),
                                  words=len(r['text'].split()))
    all_train=ShardWriter(str(out/'train/supreme_court'))
    recovered=ShardWriter(str(out/'recovered_train/supreme_court'))
    n=verified=lost_words=restored_words=restored_docs=0
    matched=set(); audit=[]
    for raw in iter_json_array(a.raw):
        text,meta=build_text(raw,'supreme_court')
        text=clean_document(text,'supreme_court',keep_newlines=False)
        words=len(text.split())
        if words<64:continue
        rid=f'supreme_court:{n}';n+=1
        if rid not in existing:continue
        old=existing[rid];matched.add(rid)
        was=old['meta'].get('anonymized',False)
        if not was:
            assert hashlib.sha256(text.encode()).hexdigest()==old['sha'], ('Source ID mismatch',rid)
            verified+=1
        if old['split']!='train':continue
        updated={**old['meta'],'anonymized':False,'statute_masked':False,'was_anonymized':was}
        row=dict(id=rid,source='supreme_court',text=text,meta=updated)
        all_train.write(row)
        if was:
            recovered.write(row);restored_docs+=1;restored_words+=words
            if old['words']<64:lost_words+=words
            audit.append(dict(id=rid,old_words=old['words'],restored_words=words,cluster=updated['cluster']))
    all_train.close();recovered.close()
    assert matched==set(existing), ('Unmatched IDs',len(set(existing)-matched))
    summary=dict(clean_records=n,verified_unmasked=verified,matched=len(matched),train_docs=all_train.total,
                 restored_anonymized_docs=restored_docs,restored_words=restored_words,
                 original_words_in_collapsed_train_docs=lost_words,
                 collapsed_train_docs=sum(r['old_words']<64 for r in audit),
                 source='official Phase_1_Pretraining/Train/train_supreme_court.json',
                 note='Train split and near-duplicate clusters preserved. Apply test-prompt decontamination before derived SFT.')
    write_json(summary,str(out/'manifest.json'));write_json(audit,str(out/'restored_records.json'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
