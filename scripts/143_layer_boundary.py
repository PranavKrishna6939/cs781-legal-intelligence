#!/usr/bin/env python
"""Apply the fixed boundary policy to cached layer-12 N64 scores, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ARCH=Path('/data/cs781-archive/recovery_20261006')
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['screen','confirmation','test'],required=True)
    a=p.parse_args();part=a.partition
    import numpy as np
    import torch
    from transformers import PreTrainedTokenizerFast
    from saul.serve.finish import postprocess,specials_of
    from saul.serve.boundary import boundary_choice,POLICY
    assert not torch.cuda.is_initialized()
    if part=='test':
        native=ROOT/'submissions/B_layer12_mbr64_20261006'
        feature_path=ARCH/'layer_followup_queue/test_features/features.jsonl'
        prompts_path=ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'
        manifest=read(native/'manifest.json')
    else:
        native=ARCH/f'layer_mbr_exact/{part}/layer_12'
        feature_path=ARCH/f'layer_mbr_exact/{part}/features.jsonl'
        prompts_path=ROOT/f'runs/recovery_20261005/dev/{part}_prompts.json'
        manifest=read(ROOT/f'runs/recovery_20261005/followthrough/runtime_fixed/pool_64/{part}/manifest.json')
    prompts=read(prompts_path);baseline=read(native/'prediction.json')
    rows=[json.loads(line) for line in feature_path.open()];features={r['id']:r for r in rows}
    assert len(features)==len(rows) and set(features)==set(prompts)==set(baseline)
    assert manifest['prompts_sha256']==hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    specials=specials_of(PreTrainedTokenizerFast.from_pretrained(ROOT/'artifacts/tokenizers/legal32k'))
    pred={};decisions=[]
    for key in prompts:
        feature=features[key];layer=feature['layers']['12'];texts=feature['candidates'];scores=layer['scores']
        index=int(np.asarray(scores).argmax())
        assert postprocess(texts[index],128,specials)==layer['selected']==baseline[key]['text'],key
        choice=boundary_choice(texts,scores,prompts[key]['text'],index)
        pred[key]={'text':postprocess(texts[choice['selected_index']],128,specials)}
        decisions.append(dict(id=key,native_index=index,**choice))
    assert not torch.cuda.is_initialized()
    out=ARCH/'layer12_boundary'/part;out.mkdir(parents=True,exist_ok=False)
    (out/'prediction.json').write_text(json.dumps(pred,ensure_ascii=False))
    manifest.update(layer=12,selector='layer12_then_'+POLICY,selection_pending=False,auxiliary_not_promoted=True,
        boundary_source_sha256=hashlib.sha256((ROOT/'saul/serve/boundary.py').read_bytes()).hexdigest(),
        native_prediction_sha256=hashlib.sha256((native/'prediction.json').read_bytes()).hexdigest(),
        feature_cache_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        incomplete_outputs={k:len(v['text'].split()) for k,v in pred.items() if len(v['text'].split())!=128})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    summary=dict(n=len(pred),changed=sum(pred[k]!=baseline[k] for k in pred),native_reconstruction='all exact',
        cuda_initialized=False,status='prepared_unscored',policy=POLICY)
    (out/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
