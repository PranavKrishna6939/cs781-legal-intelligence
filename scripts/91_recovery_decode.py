#!/usr/bin/env python
"""Run controlled decoding exclusively through the EC2 Saul HTTP inference server."""
import argparse
import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--prompts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed', type=int, default=781)
    p.add_argument('--ngram', type=int, default=4)
    p.add_argument('--max_ngram_occurrences', type=int, default=1)
    p.add_argument('--frequency', type=float, default=0.3)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--allow_incomplete', action='store_true', help='Dev diagnostics only; log short outputs instead of aborting')
    p.add_argument('--defer-mbr', action='store_true', help='Save candidate pools for a later HTTP rerank; selected text is provisional')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prediction.json').exists():
        raise SystemExit('Output already exists; choose a new run directory')
    raw = Path(a.prompts).read_bytes()
    prompts = json.loads(raw)
    params = dict(n=a.n, temperature=a.temperature, truncation='min_p', min_p=0.05,
                  no_repeat_ngram=a.ngram, frequency_penalty=a.frequency, frequency_window=64,
                  target_words=128, max_new_tokens=1024,
                  mbr=dict(enabled=a.n > 1 and not a.defer_mbr, utility='mixed', alpha=0.5))
    if a.max_ngram_occurrences != 1:
        params['max_ngram_occurrences'] = a.max_ngram_occurrences
    body = dict(prompts=[r['text'] for r in prompts.values()], sampling=params,
                return_candidates=True, rng_seed=a.seed)
    manifest = dict(checkpoint=a.checkpoint, prompts_sha256=hashlib.sha256(raw).hexdigest(),
                    sampling=params, seed=a.seed, url=a.url, n_prompts=len(prompts),selection_pending=a.defer_mbr)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.time()
    request = urllib.request.Request(a.url + '/v1/generate', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=7200) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Inference server returned HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
    assert len(payload['outputs']) == len(prompts)
    # Save unmodified candidate pools before validating the selected outputs.
    with (out / 'candidates.jsonl').open('w') as f:
        for k, r in zip(prompts, payload['outputs']):
            f.write(json.dumps(dict(id=k, **r), ensure_ascii=False) + '\n')
    predictions = {k: {'text': r['text']} for k, r in zip(prompts, payload['outputs'])}
    incomplete = {k: len(r['text'].split()) for k, r in predictions.items() if len(r['text'].split()) != 128}
    assert a.allow_incomplete or not incomplete, incomplete
    (out / 'prediction.json').write_text(json.dumps(predictions, ensure_ascii=False))
    manifest['seconds'] = time.time() - start
    manifest['candidate_count'] = sum(len(r['candidates']) for r in payload['outputs'])
    manifest['incomplete_outputs'] = incomplete
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
