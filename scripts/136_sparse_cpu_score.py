#!/usr/bin/env python
"""Paired CPU diagnostic on changed predictions, with cached-GPU parity checks."""
import argparse
import json
from pathlib import Path
import time

def read(path):return json.loads(Path(path).read_text())
def main():
    p=argparse.ArgumentParser()
    for name in ['baseline','candidate','truth']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    from bert_score import BERTScorer
    start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(2)
    base=read(a.baseline/'prediction.json');pred=read(a.candidate/'prediction.json')
    scores=read(a.baseline/'per_example_scores.json');truth=read(a.truth);keys=list(truth)
    assert set(base)==set(pred)==set(scores)==set(truth)
    changed=[k for k in keys if base[k]!=pred[k]];assert changed
    scorer=BERTScorer(lang='en',batch_size=8,device='cpu',rescale_with_baseline=False)
    assert scorer.hash==read(a.baseline/'scores.json')['bertscore_hash']
    refs=[truth[k]['text'].strip() for k in changed]
    _,_,old=scorer.score([base[k]['text'].strip() for k in changed],refs)
    _,_,new=scorer.score([pred[k]['text'].strip() for k in changed],refs)
    old=old.numpy();new=new.numpy()
    error=max(abs(float(old[i])-scores[k]['bertscore']) for i,k in enumerate(changed))
    assert error<1e-5,('CPU/GPU baseline discrepancy',error)
    deltas={k:0. for k in keys}
    for i,k in enumerate(changed):deltas[k]=float(new[i]-old[i])
    clusters=sorted({scores[k]['cluster'] for k in keys})
    groups=[[k for k in keys if scores[k]['cluster']==c] for c in clusters]
    sums=np.array([sum(deltas[k] for k in g) for g in groups]);sizes=np.array([len(g) for g in groups])
    draws=np.random.default_rng(781).integers(len(groups),size=(5000,len(groups)))
    boot=sums[draws].sum(1)/sizes[draws].sum(1)
    delta=float(np.mean(list(deltas.values())))
    result=dict(status='CPU_sparse_diagnostic_not_full_GPU_score',n=len(keys),changed=len(changed),
        baseline=str(a.baseline),candidate=str(a.candidate),delta_vs_baseline=delta,
        estimated_bertscore=read(a.baseline/'scores.json')['bertscore']+delta,
        paired_cluster_95ci=np.quantile(boot,[.025,.975]).tolist(),max_cpu_gpu_baseline_error=error,
        per_changed_delta={k:deltas[k] for k in changed},bertscore_hash=scorer.hash,
        seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized())
    assert not torch.cuda.is_initialized()
    (a.candidate/'cpu_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_changed_delta'}),flush=True)
if __name__=='__main__':main()
