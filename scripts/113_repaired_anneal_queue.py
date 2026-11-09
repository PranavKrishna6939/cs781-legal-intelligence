#!/usr/bin/env python
"""One isolated repair anneal, HTTP N64 evaluation and gated submission packaging."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('queue99',ROOT/'scripts/99_recovery_training_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
OUT=ROOT/'runs/recovery_20261006/repaired_anneal'
CK=Path('/data/cs781-archive/recovery_20261006/hybrid_b_repaired_anneal')
BASE=ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'


def state(stage,**extra):
    OUT.mkdir(parents=True,exist_ok=True)
    row=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    (OUT/'state.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)


def evaluate(checkpoint,name,partition):
    dest=OUT/name/partition
    state('evaluate',checkpoint=str(checkpoint),partition=partition)
    if not (dest/'prediction.json').exists():
        with q.server(checkpoint,'repair_'+name+'_'+partition,block_size=64):
            q.run(q.PY,'-u','scripts/91_recovery_decode.py','--checkpoint',checkpoint,
                  '--prompts',q.RUN/f'dev/{partition}_prompts.json','--out',dest,
                  '--n',64,'--ngram',4,'--allow_incomplete')
    manifest=q.read(dest/'manifest.json')
    assert manifest['checkpoint']==str(checkpoint) and manifest['sampling']['n']==64
    q.run(q.PY,'-u','scripts/92_recovery_score.py','--truth',q.RUN/f'dev/{partition}_truth.json',
          '--meta',q.RUN/f'dev/{partition}_meta.json','--runs',BASE/partition,dest)
    return dict(checkpoint=str(checkpoint),name=name,partition=partition,**q.read(dest/'scores.json'))


def main():
    os.chdir(ROOT)
    # Validate everything before allocating the GPU or starting a six-hour run.
    import torch
    from saul.utils.config import load_config
    from saul.train.schedule import in_decay
    cfg=load_config('configs/train/recovery_repaired_anneal.yaml')
    st=torch.load(q.RUN.parent/'hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',map_location='cpu',weights_only=False,mmap=True)
    assert st['step']==19202 and cfg.train.total_steps==21002
    assert in_decay(19202,21002,float(cfg.optim.decay_frac))
    assert not in_decay(19201,21002,float(cfg.optim.decay_frac))
    assert st['ema']['updates']==0 and st['loader']['num_workers']==cfg.train.num_workers
    assert q.read(Path(cfg.tokens)/'manifest.json')['status']=='complete'
    assert shutil.disk_usage(CK.parent).free>12*2**30
    for part in ['screen','confirmation']:
        assert (BASE/part/'prediction.json').exists()
        assert q.read(BASE/part/'manifest.json')['sampling']['n']==64
    smi=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    assert not smi, ('GPU already has compute processes',smi)
    del st
    if not (CK/'ckpt_final_ema/model.safetensors').exists():
        assert not CK.exists(),'Partial training directory: inspect before resuming'
        state('train',checkpoint=str(CK),steps=1800,tokens=1800*524288)
        q.run(q.PY,'-u','scripts/30_train.py','--config','configs/train/recovery_repaired_anneal.yaml',
              '--resume','runs/hybrid_294m/stable_cont_b/ckpt_stable')
    rows=[]
    for name in ['ckpt_final_ema','ckpt_final']:
        rows.append(evaluate(CK/name,name,'screen'))
        (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    best=max(rows,key=lambda r:r['bertscore'])
    promoted=False
    submission=None
    if best['delta_vs_first']>=.0005:
        confirmation=evaluate(Path(best['checkpoint']),best['name'],'confirmation')
        rows.append(confirmation)
        (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        promoted=confirmation['delta_vs_first']>=.0005 and confirmation['paired_cluster_95ci'][0]>0
        if promoted:
            submission=ROOT/'submissions/B_repaired_anneal_mbr64_20261006'
            state('generate_test',checkpoint=best['checkpoint'],submission=str(submission))
            q.run(q.PY,'-u','scripts/109_make_recovery_submission.py',
                  '--checkpoint',best['checkpoint'],'--out',submission)
    state('complete',promoted=promoted,results=str(OUT/'results.json'),submission=str(submission) if submission else None)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state('failed',error=repr(exc))
        raise
