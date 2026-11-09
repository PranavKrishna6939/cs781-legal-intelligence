#!/usr/bin/env python
"""Bounded HTTP N64 retest of an existing checkpoint after the main queues.

Requires explicit baseline directories and never uploads a submission. Baselines
are copied before scoring, preserving the original experiment's score records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baselines', required=True, nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--submission-name', required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    args.baselines = [path.resolve() for path in args.baselines]
    deadline = datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset().total_seconds() == 0
    assert Path(args.submission_name).name == args.submission_name
    assert (args.checkpoint/'model.safetensors').is_file()
    assert (args.checkpoint/'config.json').is_file()
    for part in ['screen', 'confirmation']:
        prompts = read(ROOT/f'runs/recovery_20261005/dev/{part}_prompts.json')
        for baseline in args.baselines:
            assert set(read(baseline/part/'prediction.json')) == set(prompts)
            assert not read(baseline/part/'manifest.json').get('selection_pending', False)
    if args.check_only:
        print('CHECKPOINT_FOLLOWUP_PREFLIGHT_PASSED; no GPU allocated')
        return
    for arm in ['repaired_anneal', 'fresh_hc_anneal', 'final_decode_queue']:
        status = read(ROOT/f'runs/recovery_20261006/{arm}/state.json')['stage']
        assert status in ['complete', 'skipped_deadline', 'screen_passed_deadline_stop'], (arm, status)
    alpha_queue = Path('/data/cs781-archive/recovery_20261006/n128_alpha025_followup/state.json')
    if alpha_queue.exists():
        status = read(alpha_queue)['stage']
        assert status in ['complete', 'skipped_deadline'], ('alpha025_followup', status)
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('queue130', ROOT/'scripts/99_recovery_training_queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)

    def remaining():
        return (deadline-datetime.now(timezone.utc)).total_seconds()

    def state(stage, **extra):
        row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **extra)
        (args.out/'state.json').write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    def run(*command):
        assert remaining() > 900, 'Final 15-minute buffer reached'
        subprocess.run(list(map(str, command)), check=True, cwd=ROOT,
                       timeout=max(1, remaining()-900))

    protocol = dict(checkpoint=str(args.checkpoint), baselines=list(map(str, args.baselines)),
                    n=64, seed=781, screen_min_delta=.0005, confirmation_min_delta=.0005,
                    confirmation_ci_lower_min=0, adaptive_reused_dev=True,
                    deadline_utc=args.deadline_utc)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')

    def evaluate(part):
        import numpy as np
        dest = args.out/'candidate'/part
        state('decode', partition=part, checkpoint=str(args.checkpoint))
        with queue.server(args.checkpoint, args.out.name+'_'+part, block_size=64):
            run(queue.PY, '-u', 'scripts/91_recovery_decode.py', '--checkpoint', args.checkpoint,
                '--prompts', ROOT/f'runs/recovery_20261005/dev/{part}_prompts.json',
                '--out', dest, '--n', 64, '--ngram', 4, '--allow_incomplete')
        manifest = read(dest/'manifest.json')
        assert manifest['checkpoint'] == str(args.checkpoint)
        assert manifest['sampling']['n'] == 64 and manifest['seed'] == 781
        anchors = []
        for index, baseline in enumerate(args.baselines):
            anchor = args.out/'controls'/str(index)/part
            anchor.mkdir(parents=True)
            for filename in ['prediction.json', 'manifest.json']:
                (anchor/filename).write_bytes((baseline/part/filename).read_bytes())
            anchors.append(anchor)
        state('score', partition=part)
        run(queue.PY, 'scripts/92_recovery_score.py',
            '--truth', ROOT/f'runs/recovery_20261005/dev/{part}_truth.json',
            '--meta', ROOT/f'runs/recovery_20261005/dev/{part}_meta.json',
            '--runs', *anchors, dest)
        candidate = read(dest/'per_example_scores.json')
        keys = list(candidate)
        clusters = sorted({candidate[k]['cluster'] for k in keys})
        groups = [[k for k in keys if candidate[k]['cluster'] == c] for c in clusters]
        comparisons = []
        for baseline, anchor in zip(args.baselines, anchors):
            control = read(anchor/'per_example_scores.json')
            delta = {k: candidate[k]['bertscore']-control[k]['bertscore'] for k in keys}
            sums = np.array([sum(delta[k] for k in group) for group in groups])
            sizes = np.array([len(group) for group in groups])
            draws = np.random.default_rng(781).integers(len(groups), size=(5000, len(groups)))
            boot = sums[draws].sum(1)/sizes[draws].sum(1)
            comparisons.append(dict(baseline=str(baseline), delta=float(np.mean(list(delta.values()))),
                                    paired_cluster_95ci=np.quantile(boot, [.025, .975]).tolist()))
        result = dict(partition=part, **read(dest/'scores.json'), comparisons=comparisons)
        (dest/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
        return result

    try:
        if remaining() < 2*3600:
            state('skipped_deadline', remaining_seconds=remaining())
            return
        rows = [evaluate('screen')]
        promoted = False
        submission = None
        if all(c['delta'] >= .0005 for c in rows[-1]['comparisons']):
            rows.append(evaluate('confirmation'))
            promoted = all(c['delta'] >= .0005 and c['paired_cluster_95ci'][0] > 0
                           for c in rows[-1]['comparisons'])
            if promoted:
                submission = ROOT/'submissions'/args.submission_name
                assert not submission.exists()
                state('generate_test', submission=str(submission))
                run(queue.PY, '-u', 'scripts/109_make_recovery_submission.py',
                    '--checkpoint', args.checkpoint, '--out', submission)
                manifest = read(submission/'manifest.json')
                manifest['followup_confirmation'] = rows[-1]
                manifest['checkpoint_sha256'] = hashlib.sha256(
                    (args.checkpoint/'model.safetensors').read_bytes()).hexdigest()
                (submission/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (args.out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        state('complete', promoted=promoted, submission=str(submission) if submission else None,
              results=rows)
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
