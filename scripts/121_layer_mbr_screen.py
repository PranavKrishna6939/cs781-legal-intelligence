#!/usr/bin/env python
"""Fixed-depth MBR screen on saved pools, exclusively through Saul HTTP.

No references are sent to the feature server. This is decoder selection on a
reused dev set, not JEPA training. A passing screen still needs confirmation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('witness119',ROOT/'scripts/119_jepa_witness_gate.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--partition',choices=['screen','confirmation'],default='screen')
    p.add_argument('--layers',nargs='+',type=int,default=[8,12,16,20])
    p.add_argument('--base',type=Path,default=ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64')
    p.add_argument('--checkpoint',type=Path,default=q.CHECKPOINT)
    a=p.parse_args();os.chdir(ROOT)
    assert 20 in a.layers and len(a.layers)<=4
    q.OUT=Path('/data/cs781-archive/recovery_20261006/layer_mbr_exact')/a.partition
    q.CHECKPOINT=a.checkpoint.resolve()
    q.OUT.mkdir(parents=True,exist_ok=True)
    assert not (q.OUT/'results.json').exists(),'Completed comparison exists'
    base=a.base.resolve()/a.partition
    assert q.read(base/'manifest.json')['checkpoint']==str(q.CHECKPOINT)
    # Full-group SDPA can exceed the memory remaining beside training. Do not
    # change packing silently: wait for both data queues and exclusive GPU use.
    for arm in ['repaired_anneal','fresh_hc_anneal']:
        status=q.read(ROOT/f'runs/recovery_20261006/{arm}/state.json')['stage']
        assert status in ['complete','skipped_deadline'],f'{arm} still owns its slot: {status}'
    nested_queue=ROOT/'runs/recovery_20261006/final_decode_queue/state.json'
    if nested_queue.exists():
        status=q.read(nested_queue)['stage']
        assert status in ['complete','skipped_deadline','screen_passed_deadline_stop'],f'Nested N128 has priority: {status}'
    alpha_queue=Path('/data/cs781-archive/recovery_20261006/n128_alpha025_followup/state.json')
    if alpha_queue.exists():
        status=q.read(alpha_queue)['stage']
        assert status in ['complete','skipped_deadline'],f'Alpha-.25 confirmation has priority: {status}'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    prompts=q.read(ROOT/f'runs/recovery_20261005/dev/{a.partition}_prompts.json')
    rows={r['id']:r for r in map(json.loads,(base/'candidates.jsonl').open())}
    assert set(rows)==set(prompts)
    protocol=dict(partition=a.partition,layers=a.layers,checkpoint=str(q.CHECKPOINT),
        candidate_source=str(base),alpha=.5,no_new_generation=True,
        screen_threshold=.0005,confirmation_threshold=.0005,confirmation_ci_lower_min=0.,
        adaptive_reused_dev=True,encoding='native_full_group_sdpa')
    if (q.OUT/'protocol.json').exists():
        assert q.read(q.OUT/'protocol.json')==protocol,'Cached run protocol differs'
    else:
        (q.OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cache=q.OUT/'features.jsonl'
    features={r['id']:r for r in map(json.loads,cache.open())} if cache.exists() else {}
    start=time.monotonic()
    def guard():
        if time.monotonic()-start>1800:
            raise TimeoutError('Layer comparison exceeded 30-minute feature budget')
    with q.server(),cache.open('a') as output:
        for i,key in enumerate(prompts):
            if key in features:continue
            guard()
            row=rows[key]
            result=q.post('/v1/layer_mbr_features',dict(candidates=row['candidates'],
                finish_reasons=row['finish_reasons'],layers=a.layers))
            assert result['layers']['20']['selected']==row['text'],f'Native parity mismatch: {key}'
            result['id']=key;features[key]=result
            output.write(json.dumps(result)+'\n');output.flush()
            if (i+1)%20==0:q.state('features',done=i+1,total=len(prompts),seconds=time.monotonic()-start)
    q.state('score',seconds=time.monotonic()-start)
    paths=[]
    for depth in [20]+[n for n in a.layers if n!=20]:
        path=q.OUT/f'layer_{depth}';path.mkdir(exist_ok=True)
        pred={k:{'text':features[k]['layers'][str(depth)]['selected']} for k in prompts}
        (path/'prediction.json').write_text(json.dumps(pred))
        paths.append(path)
    guard()
    subprocess.run([str(q.PY),'scripts/92_recovery_score.py',
        '--truth',str(ROOT/f'runs/recovery_20261005/dev/{a.partition}_truth.json'),
        '--meta',str(ROOT/f'runs/recovery_20261005/dev/{a.partition}_meta.json'),
        '--runs',*map(str,paths)],check=True,timeout=600)
    scores={p.name:q.read(p/'scores.json') for p in paths}
    (q.OUT/'results.json').write_text(json.dumps(scores,indent=2)+'\n')
    q.state('complete',scores=scores,seconds=time.monotonic()-start)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        q.OUT.mkdir(parents=True,exist_ok=True);q.state('failed',error=repr(exc));raise
