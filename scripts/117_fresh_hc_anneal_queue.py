#!/usr/bin/env python
"""Serial second data arm, with deadline guard and the confirmed incumbent.

Start only after explicit local review marker and completed repair queue. All
decoding uses the existing HTTP server through the shared evaluation function.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('repair113',ROOT/'scripts/113_repaired_anneal_queue.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.OUT=ROOT/'runs/recovery_20261006/fresh_hc_anneal'
r.CK=Path('/data/cs781-archive/recovery_20261006/hybrid_b_fresh_hc_anneal')
FIRST=ROOT/'runs/recovery_20261006/repaired_anneal'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--deadline-utc',required=True,help='ISO UTC timestamp; reserve four hours')
    p.add_argument('--if-repair-promotes',help='Reviewed alternative config, used only after confirmed repair promotion')
    a=p.parse_args();os.chdir(ROOT)
    deadline=datetime.fromisoformat(a.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    from saul.utils.config import load_config
    original_config='configs/train/recovery_fresh_hc_bodies_anneal.yaml'
    def validated_config(path):
        cfg=load_config(path)
        audit=r.q.read(Path(cfg.tokens)/'manifest.json')
        assert audit['status']=='prepared_not_launched'
        review=r.q.read(Path(cfg.tokens)/'review.json')
        assert review['approved_for_training'] is True
        digest=hashlib.sha256((Path(cfg.tokens)/'manifest.json').read_bytes()).hexdigest()
        assert digest==review['manifest_sha256']
        assert audit['max_absolute_share_error']<.02
        return cfg,digest
    frozen={original_config:validated_config(original_config)[1]}
    if a.if_repair_promotes:frozen[a.if_repair_promotes]=validated_config(a.if_repair_promotes)[1]
    r.state('waiting_for_repair',deadline_utc=a.deadline_utc,conditional_config=a.if_repair_promotes)
    while True:
        first=r.q.read(FIRST/'state.json')
        if first['stage']=='complete':break
        if first['stage']=='failed':raise RuntimeError('Repair queue failed; inspect before taking GPU')
        if (deadline-datetime.now(timezone.utc)).total_seconds()<12*3600:
            r.state('skipped_deadline',reason='Need 8 hours for run/evaluation plus 4 hours buffer')
            return
        time.sleep(55)
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining<12*3600:
        r.state('skipped_deadline',remaining_hours=remaining/3600)
        return
    combined=bool(first.get('promoted') and a.if_repair_promotes)
    config_path=a.if_repair_promotes if combined else original_config
    cfg,digest=validated_config(config_path)
    assert digest==frozen[config_path], 'Prepared corpus changed while waiting'
    r.CK=Path(cfg.run_dir)
    assert r.CK.parent==Path('/data/cs781-archive/recovery_20261006')
    variant='repaired_sc_plus_restored_spans_plus_fresh_hc' if combined else 'fresh_hc_only'
    (r.OUT/'selected_arm.json').write_text(json.dumps(dict(variant=variant,config=config_path,
        manifest_sha256=digest,checkpoint=str(r.CK)),indent=2)+'\n')
    if first.get('promoted'):
        confirmed=[x for x in r.q.read(FIRST/'results.json') if x['partition']=='confirmation']
        best=max(confirmed,key=lambda x:x['bertscore'])
        r.BASE=FIRST/best['name']
    for part in ['screen','confirmation']:
        assert (r.BASE/part/'prediction.json').exists()
        assert r.q.read(r.BASE/part/'manifest.json')['sampling']['n']==64
    # The scoring script writes scores into its first run too. Stage read-only
    # prediction/manifest links so incumbent comparison history is not overwritten.
    origin_base=r.BASE
    for part in ['screen','confirmation']:
        anchor=r.OUT/'incumbent'/part;anchor.mkdir(parents=True,exist_ok=True)
        for filename in ['prediction.json','manifest.json']:
            target=anchor/filename
            if not target.exists():os.symlink(origin_base/part/filename,target)
    (r.OUT/'incumbent_source.json').write_text(json.dumps({'source':str(origin_base)},indent=2)+'\n')
    r.BASE=r.OUT/'incumbent'
    assert str(r.CK)==cfg.run_dir
    assert shutil.disk_usage(r.CK.parent).free>12*2**30
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    import torch
    from saul.train.schedule import in_decay
    st=torch.load(ROOT/'runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',map_location='cpu',weights_only=False,mmap=True)
    assert st['step']==19202 and cfg.train.total_steps==21002 and st['ema']['updates']==0
    assert in_decay(19202,21002,float(cfg.optim.decay_frac)) and not in_decay(19201,21002,float(cfg.optim.decay_frac))
    del st
    if not (r.CK/'ckpt_final_ema/model.safetensors').exists():
        assert not r.CK.exists(),'Partial run directory: inspect before resuming'
        r.state('train',checkpoint=str(r.CK),incumbent=str(r.BASE),remaining_hours=remaining/3600,variant=variant)
        r.q.run(r.q.PY,'-u','scripts/30_train.py','--config',config_path,
                '--resume','runs/hybrid_294m/stable_cont_b/ckpt_stable')
    rows=[]
    for name in ['ckpt_final_ema','ckpt_final']:
        rows.append(r.evaluate(r.CK/name,'fresh_'+name,'screen'))
        (r.OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    best=max(rows,key=lambda x:x['bertscore']);promoted=False;submission=None
    if best['delta_vs_first']>=.0005:
        conf=r.evaluate(Path(best['checkpoint']),best['name'],'confirmation')
        rows.append(conf);(r.OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        promoted=conf['delta_vs_first']>=.0005 and conf['paired_cluster_95ci'][0]>0
        if promoted:
            submission=ROOT/'submissions'/('B_repaired_spans_fresh_hc_mbr64_20261006' if combined else 'B_fresh_hc_anneal_mbr64_20261006')
            r.state('generate_test',checkpoint=best['checkpoint'],submission=str(submission))
            r.q.run(r.q.PY,'-u','scripts/109_make_recovery_submission.py','--checkpoint',best['checkpoint'],'--out',submission)
    r.state('complete',promoted=promoted,incumbent=str(r.BASE),results=str(r.OUT/'results.json'),variant=variant,
            submission=str(submission) if submission else None)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        r.state('failed',error=repr(exc))
        raise
