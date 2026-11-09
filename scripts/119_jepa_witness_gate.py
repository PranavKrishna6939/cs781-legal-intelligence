#!/usr/bin/env python
"""Reference-aware representation gate, never a submission or training reward.

Fit geometry only on cluster-disjoint official training continuations. Encode
through the Saul HTTP server, reproduce native N64 MBR, then score a small fixed
dev diagnostic. Fail closed on comparator mismatch or the GPU queue advancing.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT/'.venv/bin/python'
OUT = Path('/data/cs781-archive/recovery_20261006/j09_witness')
CHECKPOINT = Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
BASE = ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64/screen'
URL = 'http://127.0.0.1:8001'


def read(path):
    return json.loads(Path(path).read_text())


def post(route, data):
    request = urllib.request.Request(URL+route, data=json.dumps(data).encode(),
                                     headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def state(stage, **extra):
    row = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **extra)
    (OUT/'state.json').write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


def guard(start):
    if time.monotonic()-start > 1200:
        raise TimeoutError('Bounded gate exceeded 20 minutes')
    status = read(ROOT/'runs/recovery_20261006/repaired_anneal/state.json')
    if status['stage'] != 'train':
        raise RuntimeError('Primary queue needs exclusive GPU; stopping feature pilot')


@contextlib.contextmanager
def server():
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',8001)) != 0, 'Feature port occupied'
    log = (OUT/'server.log').open('w')
    proc = subprocess.Popen([str(PY),'-u','scripts/52_serve.py','--checkpoint',str(CHECKPOINT),
        '--tokenizer','artifacts/tokenizers/legal32k','--port','8001','--max_slots','1',
        '--max_prompt_len','2048','--max_new_tokens','512','--block_size','1'], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError(f'Feature server exited {proc.returncode}')
            try:
                with urllib.request.urlopen(URL+'/health',timeout=2) as response:
                    if response.status == 200: break
            except Exception:
                time.sleep(2)
        else: raise TimeoutError('Feature server startup')
        yield
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()


def train_rows():
    rows = [json.loads(line) for line in (ROOT/'runs/recovery_20261005/curation/random.jsonl').open()]
    # Cluster split precedes geometry fitting; reserve ~10% for future head diagnostics.
    rows = [r for r in rows if int(hashlib.sha256(r['cluster'].encode()).hexdigest()[:8],16)%10 != 0]
    rows.sort(key=lambda r: hashlib.sha256(('bank781'+r['id']).encode()).hexdigest())
    selected, seen = [], set()
    for row in rows:
        if row['cluster'] not in seen:
            selected.append(row); seen.add(row['cluster'])
        if len(selected)==512: break
    assert len(selected)==512
    return selected


def main():
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    os.chdir(ROOT); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'gate_result.json').exists(), 'Completed gate exists'
    start = time.monotonic(); guard(start)
    prompts = read(ROOT/'runs/recovery_20261005/dev/screen_prompts.json')
    truth = read(ROOT/'runs/recovery_20261005/dev/screen_truth.json')
    meta = read(ROOT/'runs/recovery_20261005/dev/screen_meta.json')
    keys = sorted(prompts, key=lambda k: hashlib.sha256(('j09gate781'+k).encode()).hexdigest())[:64]
    (OUT/'truth.json').write_text(json.dumps({k:truth[k] for k in keys}))
    (OUT/'meta.json').write_text(json.dumps({k:meta[k] for k in keys}))
    (OUT/'gate_protocol.json').write_text(json.dumps(dict(
        status='evaluation_only_not_deployable', ids=keys, n_anchors=128, tau=.1,
        bank_train_examples=512, backbone=str(CHECKPOINT),
        minimum_oracle_delta=.001, max_gpu_seconds=1200,
        training_evaluator_rewards=False, official_evaluator_use='diagnostic_only'), indent=2))
    records = {r['id']:r for r in map(json.loads,(BASE/'candidates.jsonl').open()) if r['id'] in keys}
    preds = {name:{} for name in ['native_mbr','oracle_witness','loo_witness']}
    features = []
    with server():
        state('fit_train_geometry')
        bank_path = OUT/'bank.json'
        if not bank_path.exists():
            rows = train_rows(); samples = []
            for i in range(0,len(rows),16):
                guard(start)
                result = post('/v1/latent_features',dict(texts=[r['target'] for r in rows[i:i+16]],sample_tokens=32))
                samples.extend(np.asarray(r['samples'],dtype=np.float32) for r in result['features'])
            raw = np.concatenate(samples); center = raw.mean(0)
            normalized = raw-center; normalized /= np.maximum(np.linalg.norm(normalized,axis=1,keepdims=True),1e-8)
            with threadpool_limits(limits=2):
                km = MiniBatchKMeans(n_clusters=128, batch_size=2048, n_init=3, random_state=781).fit(normalized)
            bank = dict(center=center.tolist(),anchors=km.cluster_centers_.tolist(),tau=.1)
            bank_path.write_text(json.dumps(bank))
            (OUT/'bank_train_ids.json').write_text(json.dumps([dict(id=r['id'],cluster=r['cluster']) for r in rows]))
        bank = read(bank_path)
        bank_id = post('/v1/latent_bank',bank)['bank_id']
        state('diagnostic_features', bank_id=bank_id)
        for index,key in enumerate(keys):
            guard(start)
            record = records[key]
            native = post('/v1/mbr_features',dict(candidates=record['candidates'],finish_reasons=record['finish_reasons']))
            if native['selected'] != record['text']:
                raise AssertionError(f'Native MBR mismatch for {key}; comparator invalid')
            kept = native['candidates']
            result = post('/v1/latent_features',dict(texts=kept,bank_id=bank_id))['features']
            reference = post('/v1/latent_features',dict(texts=[truth[key]['text']],bank_id=bank_id))['features'][0]
            prefix = post('/v1/latent_features',dict(texts=[prompts[key]['text']],kind='prefix'))['features'][0]
            support = np.asarray([r['support'] for r in result]); mass = np.asarray([r['mass'] for r in result])
            oracle = .5*(mass @ np.asarray(reference['support']) + support @ np.asarray(reference['mass']))
            n = len(kept)
            mean_s = (support.sum(0)-support)/max(1,n-1)
            mean_b = (mass.sum(0)-mass)/max(1,n-1)
            loo = .5*(mass*mean_s + support*mean_b).sum(1)
            for name,choice in [('native_mbr',native['selected']),('oracle_witness',kept[int(oracle.argmax())]),
                                ('loo_witness',kept[int(loo.argmax())])]:
                preds[name][key]={'text':choice}
            features.append(dict(id=key,candidates=kept,native_scores=native['scores'],support=support.tolist(),
                mass=mass.tolist(),reference=reference,prefix=prefix,oracle_scores=oracle.tolist(),loo_scores=loo.tolist()))
            if (index+1)%8==0: state('diagnostic_features',done=index+1,total=len(keys),seconds=time.monotonic()-start)
        (OUT/'gate_features.json').write_text(json.dumps(features))
    # Release served backbone before evaluation model is loaded. Never train on these scores.
    state('score_diagnostic',seconds=time.monotonic()-start)
    runs=[]
    for name,pred in preds.items():
        path=OUT/name; path.mkdir(exist_ok=True)
        (path/'prediction.json').write_text(json.dumps(pred)); runs.append(str(path))
    guard(start)
    subprocess.run([str(PY),'scripts/92_recovery_score.py','--truth',str(OUT/'truth.json'),
        '--meta',str(OUT/'meta.json'),'--runs',*runs],check=True,timeout=300)
    scores={name:read(OUT/name/'scores.json') for name in preds}
    result=dict(scores=scores,seconds=time.monotonic()-start,
        oracle_gate_passed=scores['oracle_witness']['delta_vs_first']>=.001,
        interpretation='Reference-aware diagnostic only; not expected BERTScore or a deployable selector')
    (OUT/'gate_result.json').write_text(json.dumps(result,indent=2)+'\n')
    state('complete',**result)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True); state('failed',error=repr(exc)); raise
