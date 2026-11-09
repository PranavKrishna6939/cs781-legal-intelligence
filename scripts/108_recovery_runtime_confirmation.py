#!/usr/bin/env python
"""Finish pool comparisons under one graph-free compiled inference runtime.

Recompute N16 as well: changing compiler mode can change seeded sample paths.
All generation stays on the existing HTTP server; no training or uploads.
"""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('follow', ROOT/'scripts/107_recovery_followthrough.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
OUT = f.OUT/'runtime_fixed'


def evaluate(ck, name, anchor=None, partition='screen', n=16):
    dest = OUT/name/partition
    f.state('evaluate', checkpoint=str(ck), name='runtime_fixed/'+name,
            partition=partition, n=n)
    if not (dest/'prediction.json').exists():
        with f.queue.server(ck, 'runtime_fixed_'+name+'_'+partition, block_size=max(16, n)):
            f.call(f.PY, '-u', 'scripts/91_recovery_decode.py', '--checkpoint', ck,
                   '--prompts', f.RUN/f'dev/{partition}_prompts.json', '--out', dest,
                   '--n', n, '--ngram', 4, '--allow_incomplete')
    manifest = f.read(dest/'manifest.json')
    assert manifest['checkpoint'] == str(ck)
    assert manifest['sampling']['n'] == n
    if not (dest/'scores.json').exists():
        runs = [anchor, dest] if anchor else [dest]
        f.call(f.PY, '-u', 'scripts/92_recovery_score.py', '--truth',
               f.RUN/f'dev/{partition}_truth.json', '--meta',
               f.RUN/f'dev/{partition}_meta.json', '--runs', *runs)
    return dict(checkpoint=str(ck), name=name, candidate_count=n, path=str(dest),
                partition=partition, **f.read(dest/'scores.json'))


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    assert f.read(f.RUN/'queue_state.json')['stage'] == 'complete'
    ck = f.read(f.OUT/'base.json')['checkpoint']
    rows = []

    def save(row):
        rows.append(row)
        (OUT/'results.json').write_text(json.dumps(rows, indent=2))
        return row

    base = save(evaluate(ck, 'pool_16'))
    for n in (32, 64):
        save(evaluate(ck, f'pool_{n}', base['path'], n=n))
    best = max(rows, key=lambda r: r['bertscore'])
    if best['candidate_count'] != 16 and best['delta_vs_first'] >= .0005:
        reference = save(evaluate(ck, 'pool_16', partition='confirmation'))
        save(evaluate(ck, best['name'], reference['path'], 'confirmation', best['candidate_count']))
    f.state('complete', results=str(OUT/'results.json'),
            screen_winner=best['name'], runtime='inductor_without_cudagraphs')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        f.state('failed', error=repr(exc))
        raise
