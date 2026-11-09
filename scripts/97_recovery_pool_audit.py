#!/usr/bin/env python
"""Reuse server candidates: ROUGE-only MBR and dev-only oracle headroom audit.

The reference-aware oracle is a diagnostic only; it never emits submission predictions.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saul.decode.mbr import mbr_pick, rouge_matrix
from saul.serve.finish import filter_candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--truth', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--oracle_n', type=int, default=64)
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in (run/'candidates.jsonl').read_text().splitlines()]
    truth = json.loads(Path(a.truth).read_text())
    baseline = json.loads((run/'prediction.json').read_text())
    assert set(truth) == {r['id'] for r in records}
    pools, pred = {}, {}
    for r in records:
        kept = filter_candidates(r['candidates'], r['finish_reasons'], 128)
        assert all(len(t.split()) == 128 for t in kept)
        pools[r['id']] = kept
        pred[r['id']] = dict(text=kept[mbr_pick(rouge_matrix(kept))] if len(kept)>1 else kept[0])
    (out/'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
    (out/'manifest.json').write_text(json.dumps(dict(candidate_source=str(run), utility='rouge', new_decoding=False), indent=2))
    from bert_score import BERTScorer
    scorer = BERTScorer(lang='en', batch_size=8, device='cuda', rescale_with_baseline=False)
    keys = sorted(truth, key=lambda k: hashlib.sha256(('oracle781:'+k).encode()).hexdigest())[:a.oracle_n]
    texts, refs, bounds = [], [], []
    for k in keys:
        start = len(texts)
        texts.extend(pools[k]); refs.extend([truth[k]['text']]*len(pools[k]))
        bounds.append((start,len(texts)))
    _, _, f = scorer.score(texts, refs)
    _, _, actual = scorer.score([baseline[k]['text'] for k in keys], [truth[k]['text'] for k in keys])
    oracle = [float(f[s:e].max()) for s,e in bounds]
    actual = actual.tolist()
    report = dict(n=len(keys), bertscore_hash=scorer.hash, baseline=sum(actual)/len(actual),
                  oracle=sum(oracle)/len(oracle), mean_headroom=sum(o-b for o,b in zip(oracle,actual))/len(keys),
                  caution='Reference-aware dev upper bound, not an attainable/test score or a deployable selector',
                  examples={k:dict(baseline=b,oracle=o,candidate_scores=f[s:e].tolist())
                            for k,b,o,(s,e) in zip(keys,actual,oracle,bounds)})
    (out/'oracle_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}), flush=True)


if __name__ == '__main__':
    main()
