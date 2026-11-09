#!/usr/bin/env python
"""Tighten the reviewed OCR proxy without repeating extraction/tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-fragment-rate',type=float,default=.005)
    p.add_argument('--target-tokens',type=int,default=120000000)
    a=p.parse_args();src=Path(a.input);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((src/'manifest.json').read_text())
    assert parent['status']=='ready_for_review'
    rows=[json.loads(l) for l in (src/'eligible_bodies.jsonl').open()]
    eligible=[r for r in rows if r['fragmented_word_proxy']<=a.max_fragment_rate]
    chosen=[];total=0
    for r in sorted(eligible,key=lambda x:x['rank'],reverse=True):
        chosen.append(r);total+=r['tokens']
        if total>=a.target_tokens:break
    assert total>=a.target_tokens,'Insufficient quality-filtered unique tokens'
    assert len({r['cluster'] for r in chosen})==len(chosen)
    pool=np.memmap(src/'candidate_tokens.bin',dtype=np.uint16,mode='r')
    target=out/'source';target.mkdir()
    idx=[];pos=0
    with (target/'tokens.bin').open('wb') as file:
        for r in chosen:
            tokens=pool[r['offset']:r['offset']+r['tokens']]
            assert len(tokens)==r['tokens']
            file.write(tokens.tobytes());idx.append([pos,len(tokens),r['bytes'],0]);pos+=len(tokens)
    np.save(target/'index.npy',np.asarray(idx,dtype=np.int64))
    assert pos==total and (target/'tokens.bin').stat().st_size==2*total
    with (out/'selected.jsonl').open('w') as file:
        for r in chosen:file.write(json.dumps(r,ensure_ascii=False)+'\n')
    sample=sorted(chosen,key=lambda r:hashlib.sha256(('final-review781:'+r['id']).encode()).hexdigest())[:12]
    (out/'manual_sample.json').write_text(json.dumps(sample,indent=2)+'\n')
    manifest={'status':'ready_for_review','selected_docs':len(chosen),'selected_tokens':total,
        'eligible_docs':len(eligible),'eligible_tokens':sum(r['tokens'] for r in eligible),
        'additional_fragment_filter_dropped':len(rows)-len(eligible),'max_fragment_rate':a.max_fragment_rate,
        'parent_manifest':str(src/'manifest.json'),'not_launched':True,
        'known_roster_selected':any(r['id']=='high_courts:1260319' for r in chosen),
        'known_fragmented_record_selected':any(r['id']=='high_courts:1607824' for r in chosen),
        'limitations':['Train-only OCR proxy can reject some legitimate abbreviations.',
            'Residual brief captions and occasional word fragments remain; this is not perfect text repair.',
            'Source/body/quality selection and mixture changes are a combined experimental arm.']}
    assert not manifest['known_roster_selected'] and not manifest['known_fragmented_record_selected']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
