#!/usr/bin/env python
"""Short matched restored-data SFT experiment with managed inference servers."""
import contextlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python');RUN=ROOT/'runs/recovery_20261005'
ARCHIVE=Path('/data/cs781-archive/recovery_20261005')


def call(args,timeout=600):
    print('RUN',args,flush=True)
    child=subprocess.Popen(list(map(str,args)),cwd=ROOT,start_new_session=True)
    try:
        code=child.wait(timeout=timeout)
        if code:raise subprocess.CalledProcessError(code,args)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


@contextlib.contextmanager
def server(ck,name):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8000))!=0
    with (ROOT/f'logs/recovery_{name}_server.log').open('w') as log:
        p=subprocess.Popen([PY,'-u','scripts/52_serve.py','--checkpoint',str(ck),
             '--tokenizer','artifacts/tokenizers/legal32k','--max_slots','128',
             '--max_prompt_len','2048','--truncate_left'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            until=time.monotonic()+240
            while time.monotonic()<until:
                assert p.poll() is None,'Server exited'
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
                        if r.status==200:break
                except Exception:time.sleep(2)
            else:raise TimeoutError('Server startup')
            yield
        finally:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()


def evaluate(ck,name):
    dest=RUN/'entity_models'/name
    with server(ck,name):
        call([PY,'-u','scripts/91_recovery_decode.py','--checkpoint',ck,
              '--prompts',RUN/'dev/screen_prompts.json','--out',dest,'--allow_incomplete'])
    call([PY,'-u','scripts/92_recovery_score.py','--truth',RUN/'dev/screen_truth.json',
          '--meta',RUN/'dev/screen_meta.json','--runs',RUN/'dense_d1/baseline',dest],timeout=180)
    return dict(checkpoint=str(ck),name=name,**json.loads((dest/'scores.json').read_text()))


def main():
    os.chdir(ROOT)
    pid=int(sys.argv[1]);cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'scripts/30_train.py' in cmd and b'recovery_b_anneal.yaml' in cmd
    identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
    def resume():
        if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21]==identity:
            os.kill(pid,signal.SIGCONT)
    manifest=json.loads((RUN/'entity_recovery/manifest.json').read_text())
    assert manifest['stats']['accepted_restored']>=2000
    watcher=os.fork()
    if watcher==0:
        time.sleep(1800);resume();os._exit(0)
    def expired(signum,frame):raise TimeoutError('Entity pilot reached bounded timeslice')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    signal.alarm(1680)
    summary=[]
    try:
        os.kill(pid,signal.SIGSTOP);print('PAUSED',pid,flush=True);time.sleep(3)
        for variant in ['random','entity']:
            dest=ARCHIVE/f'dense_restored_{variant}'
            call([PY,'-u','scripts/96_recovery_sft.py','--checkpoint',
                  ROOT/'runs/dense_286m/anneal/ckpt_final_ema','--data',RUN/f'entity_recovery/{variant}.jsonl',
                  '--out',dest,'--lr','1e-5','--compile'])
            checkpoints=sorted(dest.glob('ckpt_*'),key=lambda p:int(p.name.split('_')[-1]))
            summary.append(evaluate(checkpoints[-1],f'{variant}_final'))
            (RUN/'entity_recovery/pilot_summary.json').write_text(json.dumps(summary,indent=2))
        # Predeclared early-stop check on the entity recipe only, not a wide grid.
        dest=ARCHIVE/'dense_restored_entity'
        ck=sorted(dest.glob('ckpt_*'),key=lambda p:int(p.name.split('_')[-1]))[1]
        summary.append(evaluate(ck,'entity_half'))
        (RUN/'entity_recovery/pilot_summary.json').write_text(json.dumps(summary,indent=2))
        print('COMPLETE',json.dumps(summary),flush=True)
    finally:
        signal.alarm(0);resume();os.kill(watcher,signal.SIGKILL);os.waitpid(watcher,0)
        print('RESUMED',pid,flush=True)


if __name__=='__main__':main()
