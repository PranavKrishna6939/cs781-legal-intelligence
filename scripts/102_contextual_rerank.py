#!/usr/bin/env python
"""Collect own-model features through HTTP, then run predeclared reference-free selectors."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from saul.decode.mbr import rouge_matrix
from saul.serve.finish import filter_candidates


def zscore(x):
    return (x-x.mean())/max(float(x.std()),1e-6)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pool',required=True);p.add_argument('--prompts',required=True)
    p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:8000')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    prompts=json.loads(Path(a.prompts).read_text())
    rows=[json.loads(l) for l in Path(a.pool).read_text().splitlines()]
    groups=[dict(id=r['id'],prompt=prompts[r['id']]['text'],
                 candidates=filter_candidates(r['candidates'],r['finish_reasons'],128)) for r in rows]
    cached={}
    features_path=out/'features.jsonl'
    if features_path.exists():
        cached={r['id']:r for r in map(json.loads,features_path.read_text().splitlines())}
    todo=[g for g in groups if g['id'] not in cached]
    with features_path.open('a',buffering=1) as f:
        for i in range(0,len(todo),8):
            body=json.dumps(dict(groups=todo[i:i+8],context_tokens=1024)).encode()
            req=urllib.request.Request(a.url+'/v1/rerank_features',data=body,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as r: result=json.load(r)
            for row in result['groups']:
                f.write(json.dumps(row)+'\n');cached[row['id']]=row
            print('features',len(cached),'/',len(groups),flush=True)
    variants={name:{} for name in ['context_embed','context_mixed','context_mixed_ll01','context_mixed_ll03','likelihood']}
    diagnostics={}
    for g in groups:
        k=g['id'];feat=cached[k];n=len(g['candidates'])
        e=np.asarray(feat['contextual_matrix']);r=rouge_matrix(g['candidates']);ll=np.asarray(feat['mean_logprob'])
        assert np.isfinite(e).all() and np.isfinite(ll).all()
        def consensus(m):return (m.sum(1)-np.diag(m))/max(1,n-1)
        mixed=consensus(.5*e+.5*r)
        utilities=dict(context_embed=consensus(e),context_mixed=mixed,
                       context_mixed_ll01=zscore(mixed)+.1*zscore(ll),
                       context_mixed_ll03=zscore(mixed)+.3*zscore(ll),likelihood=ll)
        diagnostics[k]={}
        for name,values in utilities.items():
            j=int(values.argmax());variants[name][k]={'text':g['candidates'][j]};diagnostics[k][name]=j
    for name,pred in variants.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'prediction.json').write_text(json.dumps(pred,ensure_ascii=False))
        (d/'manifest.json').write_text(json.dumps(dict(pool=a.pool,prompts=a.prompts,selector=name,
                context_tokens=1024,continuation_features_only=True,reference_free=True),indent=2))
    (out/'choices.json').write_text(json.dumps(diagnostics))


if __name__=='__main__':main()
