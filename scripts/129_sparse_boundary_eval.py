#!/usr/bin/env python
"""Bounded boundary-selector comparison on saved B-EMA N64 pools via HTTP.

Runs only while the fresh trainer has ample steps remaining. A separate CUDA
allocator cap protects the primary run. No sampling or retrieval is performed.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = str(ROOT / '.venv/bin/python')
CHECKPOINT = Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
BASE = ROOT / 'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'
OUT = Path('/data/cs781-archive/recovery_20261006/boundary_sparse')
TEST_BASE = ROOT / 'submissions/B_anneal_ema_mbr64_20261005'
SUBMISSION = ROOT / 'submissions/B_boundary_mbr64_20261006'
URL = 'http://127.0.0.1:8001'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_python(fraction, script, *args):
    code = ('import sys,torch,runpy; script=sys.argv.pop(1); '
            f'torch.cuda.set_per_process_memory_fraction({fraction}); '
            'runpy.run_path(script,run_name="__main__")')
    return [PY, '-u', '-c', code, str(script), *map(str, args)]


def guard(start):
    assert time.monotonic() - start < 900, 'Fifteen-minute pilot budget reached'
    assert read(ROOT / 'runs/recovery_20261006/fresh_hc_anneal/state.json')['stage'] == 'train'
    assert read(ROOT / 'runs/recovery_20261006/final_decode_queue/state.json')['stage'] == 'waiting_for_data'
    lines = (ROOT / 'logs/fresh_hc_anneal_queue_20261006.log').read_text().splitlines()
    last = next(line for line in reversed(lines) if line.startswith('step '))
    step = int(re.match(r'step (\d+)/', last).group(1))
    assert step < 20800, 'Too close to primary evaluation; await the main queue'


def post(route, body):
    req = urllib.request.Request(URL + route, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


@contextlib.contextmanager
def server(dest, fraction):
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 8001)) != 0, 'Feature port occupied'
    with (dest / 'server.log').open('w') as log:
        proc = subprocess.Popen(bounded_python(fraction, 'scripts/52_serve.py',
            '--checkpoint', CHECKPOINT, '--tokenizer', 'artifacts/tokenizers/legal32k',
            '--port', 8001, '--max_slots', 1, '--block_size', 1,
            '--max_prompt_len', 2048, '--max_new_tokens', 512),
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                assert proc.poll() is None, f'Feature server exited {proc.returncode}'
                try:
                    with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError('Feature server startup')
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=['screen', 'confirmation', 'test'], required=True)
    parser.add_argument('--out-root', type=Path, default=OUT)
    parser.add_argument('--feature-fraction', type=float, default=.32)
    parser.add_argument('--diagnostic-confirmation', action='store_true',
                        help='Explicit budget exception for a positive sub-threshold sparse screen; test gate unchanged')
    parser.add_argument('--package-incremental', action='store_true',
                        help='Package a labeled sub-threshold candidate only when screen and confirmation CIs are positive')
    args = parser.parse_args()
    assert .1 <= args.feature_fraction <= .4
    out_root = args.out_root.resolve()
    os.chdir(ROOT)
    start = time.monotonic()
    guard(start)
    part = args.partition
    assert not args.diagnostic_confirmation or part == 'confirmation'
    assert not args.package_incremental or part == 'test'
    if part == 'confirmation':
        screen_delta = read(out_root / 'screen/scores.json')['delta_vs_first']
        assert screen_delta >= .0005 or (args.diagnostic_confirmation and screen_delta > 0)
    if part == 'test':
        conf = read(out_root / 'confirmation/scores.json')
        if args.package_incremental:
            screen = read(out_root / 'screen/scores.json')
            assert screen['delta_vs_first'] > 0 and screen['paired_cluster_95ci'][0] > 0
            assert conf['delta_vs_first'] > 0 and conf['paired_cluster_95ci'][0] > 0
        else:
            assert conf['delta_vs_first'] >= .0005 and conf['paired_cluster_95ci'][0] > 0
    dest = SUBMISSION if part == 'test' else out_root / part
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'state.json').exists(), 'Existing run: inspect instead of overwriting'

    def state(stage, **extra):
        row = dict(stage=stage, partition=part, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        (dest / 'state.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    try:
        from transformers import PreTrainedTokenizerFast
        from saul.serve.boundary import boundary_choice, POLICY
        from saul.serve.finish import filter_candidates, postprocess, specials_of
        source = TEST_BASE if part == 'test' else BASE / part
        prompt_path = (ROOT / 'data/raw/Phase_1_Pretraining/Test/test.json' if part == 'test'
                       else ROOT / f'runs/recovery_20261005/dev/{part}_prompts.json')
        prompts, baseline = read(prompt_path), read(source / 'prediction.json')
        manifest = read(source / 'manifest.json')
        assert list(prompts) == list(baseline)
        assert manifest['checkpoint'] == str(CHECKPOINT)
        assert manifest['seed'] == 781 and manifest['sampling']['n'] == 64
        assert not manifest.get('selection_pending', False)
        assert manifest['prompts_sha256'] == sha(prompt_path)
        flags = [k for k in prompts if boundary_choice([baseline[k]['text']], [0.], prompts[k]['text'], 0)['flagged'][0]]
        rows = {r['id']: r for r in map(json.loads, (source / 'candidates.jsonl').open())}
        assert set(rows) == set(prompts)
        tok = PreTrainedTokenizerFast.from_pretrained(ROOT / 'artifacts/tokenizers/legal32k')
        lengths = {}
        for key in flags:
            row = rows[key]
            assert row['text'] == baseline[key]['text']
            assert len(row['candidates']) == len(row['finish_reasons']) == 64
            kept = filter_candidates(row['candidates'], row['finish_reasons'], 128)
            lengths[key] = sum(min(512, len(tok.encode(t, add_special_tokens=False))) for t in kept)
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        total = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= max(40000, total*args.feature_fraction+8192), f'Only {free} MiB free; wait for exclusive GPU'
        protocol = dict(policy=POLICY, checkpoint=str(CHECKPOINT), partition=part,
                        source=str(source), flagged_ids=flags, packed_tokens=lengths,
                        source_prediction_sha256=sha(source / 'prediction.json'),
                        source_candidates_sha256=sha(source / 'candidates.jsonl'),
                        boundary_source_sha256=sha(ROOT / 'saul/serve/boundary.py'),
                        feature_allocator_fraction=args.feature_fraction, scorer_allocator_fraction=.125,
                        max_seconds=900, native_parity_required=True,
                        screen_min_delta=.0005, confirmation_min_delta=.0005,
                        confirmation_ci_lower_min=0., new_sampling=False, retrieval=False)
        protocol['diagnostic_confirmation_budget_exception'] = args.diagnostic_confirmation
        protocol['incremental_packaging_exception'] = args.package_incremental
        (dest / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        prediction = {k: dict(v) for k, v in baseline.items()}
        state('features', flagged=len(flags), total=len(prompts))
        with server(dest, args.feature_fraction), (dest / 'features.jsonl').open('w') as output:
            # Hardest packed pool first; do not change native packing to fit.
            for key in sorted(flags, key=lambda k: lengths[k], reverse=True):
                guard(start)
                row = rows[key]
                body = dict(candidates=row['candidates'], finish_reasons=row['finish_reasons'])
                native = post('/v1/mbr_features', body)
                assert native['selected'] == baseline[key]['text'], f'Native parity: {key}'
                result = post('/v1/large_mbr_features', dict(body, prompt=prompts[key]['text']))
                assert result['selected'] == native['selected'], f'Tiled parity: {key}'
                difference = max(abs(a-b) for a, b in zip(native['scores'], result['scores']))
                assert len(native['scores']) == len(result['scores']) and difference <= 1e-5
                index = max(range(len(native['scores'])), key=lambda i: native['scores'][i])
                exact = boundary_choice(native['candidates'], native['scores'], prompts[key]['text'], index)
                exact_text = postprocess(native['candidates'][exact['selected_index']], 128, specials_of(tok))
                assert result['boundary']['selected'] == exact_text, f'Boundary utility parity: {key}'
                assert len(exact_text.split()) == 128
                prediction[key] = {'text': result['boundary']['selected']}
                output.write(json.dumps(dict(id=key, max_score_difference=difference, boundary=result['boundary'])) + '\n')
                output.flush()
        changed = [k for k in prompts if prediction[k] != baseline[k]]
        (dest / 'prediction.json').write_text(json.dumps(prediction, indent=2, ensure_ascii=True))
        manifest.update(selector=POLICY, selection_pending=False, component_pools=[str(source)],
                        changed_ids=changed, source_prediction_sha256=protocol['source_prediction_sha256'],
                        source_candidates_sha256=protocol['source_candidates_sha256'],
                        sparse_native_parity_checked_ids=flags, new_sampling=False,
                        incomplete_outputs={k:len(v['text'].split()) for k,v in prediction.items() if len(v['text'].split())!=128})
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        guard(start)
        if part != 'test':
            anchor = dest / 'incumbent'
            anchor.mkdir(exist_ok=True)
            for filename in ['prediction.json', 'manifest.json']:
                shutil.copyfile(source / filename, anchor / filename)
            state('score', changed=len(changed))
            subprocess.run(bounded_python(.125, 'scripts/92_recovery_score.py',
                '--truth', ROOT / f'runs/recovery_20261005/dev/{part}_truth.json',
                '--meta', ROOT / f'runs/recovery_20261005/dev/{part}_meta.json',
                '--runs', anchor, dest), cwd=ROOT, check=True, timeout=max(1, 900-(time.monotonic()-start)))
            state('complete', changed=len(changed), scores=read(dest / 'scores.json'))
        else:
            spec = importlib.util.spec_from_file_location('packer51', ROOT / 'scripts/51_make_submission.py')
            packer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(packer)
            assert list(prediction) == [str(i) for i in range(1, 1001)]
            check = packer.check_prediction(str(dest / 'prediction.json'), prompts, 128)
            assert check['n'] == check['exact_128'] == 1000 and check['special_token_leaks'] == 0
            with zipfile.ZipFile(dest / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.write(dest / 'prediction.json', arcname='prediction.json')
            with zipfile.ZipFile(dest / 'submission.zip') as archive:
                assert archive.namelist() == ['prediction.json'] and archive.testzip() is None
                assert archive.read('prediction.json') == (dest / 'prediction.json').read_bytes()
            manifest.update(validation=check, dev_confirmation=conf,
                            incremental_candidate=args.package_incremental,
                            original_0005_gate_passed=conf['delta_vs_first'] >= .0005,
                            prediction_sha256=sha(dest / 'prediction.json'), zip_sha256=sha(dest / 'submission.zip'))
            (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            state('complete', changed=len(changed), submission=str(dest),
                  promoted=not args.package_incremental, incremental_candidate=args.package_incremental)
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
