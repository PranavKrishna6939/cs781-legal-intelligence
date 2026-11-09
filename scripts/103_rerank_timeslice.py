#!/usr/bin/env python
"""Bounded GPU experiment; always resume the verified anneal process."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT/'.venv/bin/python')
RUN = ROOT/'runs/recovery_20261005'


def main():
    os.chdir(ROOT)
    pid = int(sys.argv[1])
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'scripts/30_train.py' in cmd and b'recovery_b_anneal.yaml' in cmd
    identity = Path(f'/proc/{pid}/stat').read_text().split()[21]
    def resume():
        if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21] == identity:
            os.kill(pid, signal.SIGCONT)
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',8000)) != 0
    # Separate watchdog survives an unexpected parent exit. Normal cleanup cancels it.
    watchdog = os.fork()
    if watchdog == 0:
        time.sleep(1200)
        resume()
        os._exit(0)
    server = None
    def expired(signum, frame):
        raise TimeoutError('Rerank timeslice exceeded 18 minutes')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(1080)
    try:
        os.kill(pid, signal.SIGSTOP)
        print('PAUSED', pid, flush=True)
        time.sleep(3)
        with (ROOT/'logs/recovery_contextual_server.log').open('w') as log:
            server = subprocess.Popen([PY,'-u','scripts/52_serve.py','--checkpoint',
                'runs/dense_286m/anneal/ckpt_final_ema','--tokenizer','artifacts/tokenizers/legal32k',
                '--max_prompt_len','2048','--truncate_left','--max_slots','32'],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                assert server.poll() is None, 'Server exited'
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
                        if r.status==200:break
                except Exception:time.sleep(2)
            else:raise TimeoutError('Server startup')
            subprocess.run([PY,'-u','scripts/102_contextual_rerank.py','--pool',
                str(RUN/'dense_d1/ngram_off/candidates.jsonl'),'--prompts',str(RUN/'dev/screen_prompts.json'),
                '--out',str(RUN/'contextual_rerank')],check=True,timeout=700)
            server.terminate();server.wait(timeout=30);server=None
        variants=['context_embed','context_mixed','context_mixed_ll01','context_mixed_ll03','likelihood']
        subprocess.run([PY,'-u','scripts/92_recovery_score.py','--truth',str(RUN/'dev/screen_truth.json'),
            '--meta',str(RUN/'dev/screen_meta.json'),'--runs',str(RUN/'dense_d1/ngram_off'),
            *[str(RUN/'contextual_rerank'/v) for v in variants]],check=True,timeout=300)
        print('COMPLETE',flush=True)
    finally:
        signal.alarm(0)
        if server is not None:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        resume()
        os.kill(watchdog,signal.SIGKILL)
        os.waitpid(watchdog,0)
        print('RESUMED',pid,flush=True)


if __name__=='__main__':main()
