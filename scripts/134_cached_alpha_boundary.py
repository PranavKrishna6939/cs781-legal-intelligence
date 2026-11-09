#!/usr/bin/env python
"""Apply the unchanged boundary rule to saved alpha-.25 utilities, CPU only.

Diagnostic preparation only. No references, new features, generation or scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ['base','extra','features','prompts','native','out']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    from transformers import PreTrainedTokenizerFast
    from saul.serve.finish import filter_candidates,postprocess,specials_of
    from saul.serve.boundary import boundary_choice,POLICY
    assert not torch.cuda.is_initialized()
    prompts=read(a.prompts);native=read(a.native/'prediction.json')
    features_list=[json.loads(line) for line in a.features.open()]
    features={r['id']:r['nested'] for r in features_list}
    assert len(features)==len(features_list) and set(features)==set(prompts)==set(native)
    pools=[]
    for source,seed in [(a.base,781),(a.extra,1781)]:
        m=read(source/'manifest.json')
        assert m['seed']==seed and m['sampling']['n']==64
        assert m['prompts_sha256']==sha(a.prompts)
        rows=[json.loads(line) for line in (source/'candidates.jsonl').open()]
        data={r['id']:r for r in rows}
        assert len(data)==len(rows) and set(data)==set(prompts)
        assert all(len(r['candidates'])==len(r['finish_reasons'])==64 for r in rows)
        pools.append(data)
    manifest=read(a.native/'manifest.json')
    assert manifest['checkpoint']==read(a.base/'manifest.json')['checkpoint']==read(a.extra/'manifest.json')['checkpoint']
    tok=PreTrainedTokenizerFast.from_pretrained(ROOT/'artifacts/tokenizers/legal32k')
    specials=specials_of(tok);pred={};decisions=[]
    for key in prompts:
        left,right=pools[0][key],pools[1][key]
        kept=filter_candidates(left['candidates']+right['candidates'],left['finish_reasons']+right['finish_reasons'],128)
        f=features[key]
        scores=.25*np.array(f['embedding_scores'])+.75*np.array(f['rouge_scores'])
        assert len(scores)==len(kept)
        index=int(scores.argmax())
        reconstructed=postprocess(kept[index],128,specials)
        assert reconstructed==native[key]['text']==f['alpha_picks']['0.25'], ('Native reconstruction mismatch',key)
        decision=boundary_choice(kept,scores.tolist(),prompts[key]['text'],index)
        selected=postprocess(kept[decision['selected_index']],128,specials)
        pred[key]={'text':selected}
        decisions.append(dict(id=key,native_index=index,**decision))
    assert not torch.cuda.is_initialized()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'prediction.json').write_text(json.dumps(pred,ensure_ascii=False))
    manifest.update(selector='alpha0.25_then_'+POLICY,auxiliary_not_promoted=True,selection_pending=False,
        native_prediction_sha256=sha(a.native/'prediction.json'),features_sha256=sha(a.features),
        boundary_source_sha256=sha(ROOT/'saul/serve/boundary.py'),
        incomplete_outputs={k:len(v['text'].split()) for k,v in pred.items() if len(v['text'].split())!=128})
    manifest['sampling']=dict(manifest['sampling'])
    manifest['sampling']['mbr']=dict(manifest['sampling']['mbr'],alpha=.25)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    summary=dict(n=len(pred),changed=sum(pred[k]!=native[k] for k in pred),
                 status='prepared_unscored',cuda_initialized=torch.cuda.is_initialized(),
                 references_used=False,native_reconstruction='all exact',policy=POLICY)
    (a.out/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
