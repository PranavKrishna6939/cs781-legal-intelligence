#!/usr/bin/env python
"""Generate the confirmed B-EMA/MBR64 recipe through HTTP and package a strict submission."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    queue = load_module('queue', ROOT/'scripts/99_recovery_training_queue.py')
    submission = load_module('submission', ROOT/'scripts/51_make_submission.py')
    state_path = out/'state.json'

    def state(stage, **extra):
        data = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        state_path.write_text(json.dumps(data, indent=2))
        print(json.dumps(data), flush=True)

    try:
        test_path = ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'
        test = json.loads(test_path.read_text())
        assert list(test) == [str(i) for i in range(1, 1001)]
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(ROOT/'artifacts/tokenizers/legal32k')
        ids = tok([r['text'] for r in test.values()], add_special_tokens=False)['input_ids']
        lengths = [len(row)+(tok.bos_token_id is not None and (not row or row[0] != tok.bos_token_id)) for row in ids]
        assert max(lengths) <= 2048, 'Test would require prompt truncation'
        state('generate', checkpoint=str(checkpoint), n=64, prompts=1000,
              max_prompt_tokens=max(lengths))
        if not (out/'prediction.json').exists():
            with queue.server(checkpoint, out.name, block_size=64):
                queue.run(queue.PY, '-u', 'scripts/91_recovery_decode.py',
                          '--checkpoint', checkpoint, '--prompts', test_path,
                          '--out', out, '--n', 64)
        manifest = json.loads((out/'manifest.json').read_text())
        assert not manifest.get('selection_pending',False), 'Deferred MBR has not selected the final predictions'
        assert manifest['checkpoint'] == str(checkpoint)
        assert manifest['prompts_sha256'] == hashlib.sha256(test_path.read_bytes()).hexdigest()
        assert manifest['sampling']['n'] == 64 and manifest['seed'] == 781
        state('validate')
        pred_path = out/'prediction.json'
        prediction = json.loads(pred_path.read_text())
        pred_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=True), encoding='utf-8')
        check = submission.check_prediction(str(pred_path), test, 128)
        assert check['n'] == check['exact_128'] == 1000, check
        assert check['special_token_leaks'] == 0, check
        zip_path = out/'submission.zip'
        temporary_zip = out/'submission.zip.tmp'
        with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(pred_path, arcname='prediction.json')
        with zipfile.ZipFile(temporary_zip) as z:
            assert z.namelist() == ['prediction.json']
            assert z.read('prediction.json') == pred_path.read_bytes()
            assert z.testzip() is None
        temporary_zip.replace(zip_path)
        manifest.update(validation=check, runtime='inductor_without_cudagraphs',
                        prediction_sha256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                        zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest())
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        state('complete', zip=str(zip_path), **check)
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
