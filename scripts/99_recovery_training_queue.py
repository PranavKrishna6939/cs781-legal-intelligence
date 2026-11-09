#!/usr/bin/env python
"""Serial GPU queue: SFT pilot, checkpoint evaluation, then the missing B anneal.

Run only after the interactive decoder work releases the GPU. Every generation is
an HTTP request to a managed instance of the existing Saul inference server.
Stops on failures, persists state, and never uploads to Codabench.
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT/'.venv/bin/python')
RUN = ROOT/'runs/recovery_20261005'


def read(path):
    return json.loads(Path(path).read_text())


def state(stage, **extra):
    data = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
    (RUN/'queue_state.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)


def run(*args):
    print('RUN', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), cwd=ROOT, check=True)


@contextlib.contextmanager
def server(checkpoint, name, block_size=16):
    # Fail if somebody else owns the port; never send requests to an unrelated model.
    import socket
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',8000)) != 0, 'Port 8000 already occupied'
    with (ROOT/f'logs/recovery_{name}_server.log').open('w') as log:
        proc = subprocess.Popen([PY, '-u', 'scripts/52_serve.py', '--checkpoint', str(checkpoint),
                                 '--tokenizer', 'artifacts/tokenizers/legal32k', '--max_prompt_len', '2048',
                                 '--truncate_left', '--block_size', str(block_size)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+900
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Server {name} exited {proc.returncode}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError(f'Server {name} startup')
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()


def evaluate(checkpoint, name, partition='screen'):
    path = RUN/'models'/name/partition
    state('evaluate', checkpoint=str(checkpoint), name=name, partition=partition)
    with server(checkpoint, name):
        run(PY,'-u','scripts/91_recovery_decode.py', '--checkpoint',checkpoint,
            '--prompts',RUN/f'dev/{partition}_prompts.json','--out',path,'--allow_incomplete')
    anchor = RUN/('dense_d1' if partition=='screen' else 'dense_confirmation')/'baseline'
    run(PY,'-u','scripts/92_recovery_score.py','--truth',RUN/f'dev/{partition}_truth.json',
        '--meta',RUN/f'dev/{partition}_meta.json','--runs',anchor,path)
    return read(path/'scores.json')


def train_sft(data, name, checkpoint):
    path = RUN/name
    state('sft', data=str(data), checkpoint=str(checkpoint), out=str(path))
    run(PY,'-u','scripts/96_recovery_sft.py','--checkpoint',checkpoint,'--data',data,'--out',path,'--compile')
    scores=[]
    for ck in sorted(path.glob('ckpt_*'),key=lambda p:int(p.name.split('_')[-1])):
        score=evaluate(ck,f'{name}_{ck.name}')
        scores.append(dict(checkpoint=str(ck),name=f'{name}_{ck.name}',**score))
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip_sft',action='store_true')
    a=p.parse_args()
    os.chdir(ROOT)
    assert (RUN/'dense_confirmation/ngram_off/scores.json').exists(), 'Finish decoder confirmation first'
    manifest=read(RUN/'curation/manifest.json')
    assert manifest['actual_n']>=5000
    checkpoint=ROOT/'runs/dense_286m/anneal/ckpt_final_ema'
    summary={'sft':[],'anneal':[]}
    if not a.skip_sft:
        # Short full-model smoke never saves weights; the actual run reloads its start.
        state('sft_smoke')
        run(PY,'-u','scripts/96_recovery_sft.py','--checkpoint',checkpoint,
            '--data',RUN/'curation/random.jsonl','--out',RUN/'sft_smoke','--max_steps','2')
        summary['sft']=train_sft(RUN/'curation/random.jsonl','dense_sft_random',checkpoint)
        winner=max(summary['sft'],key=lambda s:s['bertscore'])
        # The fixed matched-data comparison is only worth its GPU time after a
        # positive continuation-SFT screen. This is a screen, not significance.
        if winner['delta_vs_first']>=.0005:
            summary['sft']+=train_sft(RUN/'curation/dsir.jsonl','dense_sft_dsir',checkpoint)
            winner=max(summary['sft'],key=lambda s:s['bertscore'])
            summary['sft_confirmation']=evaluate(winner['checkpoint'],winner['name'],'confirmation')
    (RUN/'training_summary.json').write_text(json.dumps(summary,indent=2))
    state('b_anneal')
    run('bash','scripts/95_recovery_b_anneal.sh')
    for tag in ['ckpt_final','ckpt_final_ema']:
        ck=Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal')/tag
        score=evaluate(ck,f'b_anneal_{tag}')
        summary['anneal'].append(dict(checkpoint=str(ck),name=f'b_anneal_{tag}',**score))
    winner=max(summary['anneal'],key=lambda s:s['bertscore'])
    summary['anneal_confirmation']=evaluate(winner['checkpoint'],winner['name'],'confirmation')
    (RUN/'training_summary.json').write_text(json.dumps(summary,indent=2))
    state('complete',summary=str(RUN/'training_summary.json'))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        state('failed',error=repr(exc))
        raise
