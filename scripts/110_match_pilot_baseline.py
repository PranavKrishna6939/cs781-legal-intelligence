#!/usr/bin/env python
"""Match the restored-data pilot's 128-slot server capacity and recompute deltas."""
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/105_entity_recovery_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def main():
    os.chdir(ROOT);trainer=int(sys.argv[1]);previous=int(sys.argv[2])
    until=time.monotonic()+1800
    while Path(f'/proc/{previous}/cmdline').exists() and b'105_entity_recovery_pilot.py' in Path(f'/proc/{previous}/cmdline').read_bytes():
        if time.monotonic()>until:raise TimeoutError('Prior pilot did not finish')
        time.sleep(5)
    cmd=Path(f'/proc/{trainer}/cmdline').read_bytes();assert b'recovery_b_anneal.yaml' in cmd
    identity=Path(f'/proc/{trainer}/stat').read_text().split()[21]
    def resume():
        if Path(f'/proc/{trainer}/stat').exists() and Path(f'/proc/{trainer}/stat').read_text().split()[21]==identity:os.kill(trainer,signal.SIGCONT)
    watcher=os.fork()
    if watcher==0:time.sleep(900);resume();os._exit(0)
    def expired(signum,frame):raise TimeoutError('Matched baseline timeslice expired')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(780)
    try:
        os.kill(trainer,signal.SIGSTOP);print('PAUSED',trainer,flush=True);time.sleep(3)
        base=pilot.evaluate(ROOT/'runs/dense_286m/anneal/ckpt_final_ema','baseline_slots128')
        anchor=pilot.RUN/'entity_models/baseline_slots128'
        names=['random_final','entity_final','entity_half']
        pilot.call([pilot.PY,'-u','scripts/92_recovery_score.py','--truth',pilot.RUN/'dev/screen_truth.json',
              '--meta',pilot.RUN/'dev/screen_meta.json','--runs',anchor,*[pilot.RUN/'entity_models'/n for n in names]],timeout=240)
        path=pilot.RUN/'entity_recovery/pilot_summary.json'
        rows=json.loads(path.read_text());(path.parent/'pilot_summary_against_slots512.json').write_text(json.dumps(rows,indent=2))
        for r in rows:
            r.update(json.loads((pilot.RUN/'entity_models'/r['name']/'scores.json').read_text()))
            r['comparator']='dense baseline with matching 128-slot server'
        path.write_text(json.dumps(rows,indent=2))
        print('COMPLETE',json.dumps(rows),flush=True)
    finally:
        signal.alarm(0);resume();os.kill(watcher,signal.SIGKILL);os.waitpid(watcher,0)
        print('RESUMED',trainer,flush=True)


if __name__=='__main__':main()
