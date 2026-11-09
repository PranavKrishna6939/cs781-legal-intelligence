#!/usr/bin/env python
"""Wait for both data arms, then run the prepared nested-N128 experiment once."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/recovery_20261006/final_decode_queue'
CHILD = Path('/data/cs781-archive/recovery_20261006/nested_mbr128')
OWNS_STATE = False


def read(path):
    return json.loads(Path(path).read_text())


def state(stage, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = OUT / 'state.json.tmp'
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(OUT / 'state.json')
    print(json.dumps(row), flush=True)


def choose_incumbent(states):
    # Completed confirmation, never screen performance alone, determines the model.
    for arm in ['fresh_hc_anneal', 'repaired_anneal']:
        row = states[arm]
        if row['stage'] == 'complete' and row.get('promoted'):
            root = ROOT / 'runs/recovery_20261006' / arm
            confirmed = [r for r in read(root / 'results.json') if r['partition'] == 'confirmation']
            assert len(confirmed) == 1
            result = confirmed[0]
            return dict(checkpoint=result['checkpoint'], base=str(root / result['name']),
                        test_base=row['submission'], selected_from=arm)
    return dict(
        checkpoint='/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema',
        base=str(ROOT / 'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'),
        test_base=str(ROOT / 'submissions/B_anneal_ema_mbr64_20261005'),
        selected_from='scored_B_EMA_N64')


def main():
    global OWNS_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    deadline = datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset().total_seconds() == 0
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'state.json').exists(), 'Existing queue state: inspect before restarting'
    assert not (CHILD / 'state.json').exists(), 'Nested experiment already started'
    OWNS_STATE = True
    state('waiting_for_data', deadline_utc=args.deadline_utc, child_output=str(CHILD))
    while True:
        states = {arm: read(ROOT / 'runs/recovery_20261006' / arm / 'state.json')
                  for arm in ['repaired_anneal', 'fresh_hc_anneal']}
        assert all(row['stage'] != 'failed' for row in states.values()), 'Data queue failed; inspect before decoding'
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining < 4 * 3600:
            state('skipped_deadline', remaining_hours=remaining / 3600)
            return
        if all(row['stage'] in ['complete', 'skipped_deadline'] for row in states.values()):
            break
        time.sleep(55)
    incumbent = choose_incumbent(states)
    command = [sys.executable, '-u', str(ROOT / 'scripts/125_nested_mbr128.py'),
               '--checkpoint', incumbent['checkpoint'], '--base', incumbent['base'],
               '--test-base', incumbent['test_base'], '--out', str(CHILD),
               '--deadline-utc', args.deadline_utc]
    # Full identity/pool checks precede GPU allocation. The child also checks idle GPU.
    subprocess.run(command + ['--check-only'], check=True)
    (OUT / 'selected_incumbent.json').write_text(json.dumps(incumbent, indent=2) + '\n')
    state('running_nested128', **incumbent, child_output=str(CHILD))
    subprocess.run(command, check=True)
    result = read(CHILD / 'state.json')
    assert result['stage'] in ['complete', 'skipped_deadline', 'screen_passed_deadline_stop']
    state(result['stage'], **incumbent, child_output=str(CHILD),
          promoted=result.get('promoted', False), submission=result.get('submission'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OWNS_STATE:
            state('failed', error=repr(exc))
        raise
