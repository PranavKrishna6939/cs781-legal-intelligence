#!/usr/bin/env python
"""Official-equivalent scoring with one cached scorer and cluster-paired intervals."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--truth', required=True)
    p.add_argument('--meta', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    a = p.parse_args()
    import numpy as np
    from bert_score import BERTScorer
    spec = importlib.util.spec_from_file_location('official', Path(__file__).resolve().parents[1] / 'resources/evaluation_script/evaluate_phase1.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    truth = official.load_json(a.truth)
    meta = json.loads(Path(a.meta).read_text())
    keys = list(truth)
    refs = [truth[k]['text'].strip() for k in keys]
    scorer = BERTScorer(lang='en', batch_size=8, device='cuda', rescale_with_baseline=False)
    baseline = None
    clusters = sorted({meta[k]['cluster'] for k in keys})
    groups = [np.array([i for i, k in enumerate(keys) if meta[k]['cluster'] == c]) for c in clusters]
    for run in a.runs:
        path = Path(run)
        if (path/'manifest.json').exists():
            assert not json.loads((path/'manifest.json').read_text()).get('selection_pending',False), 'Deferred MBR predictions are provisional'
        pred = official.load_json(path / 'prediction.json')
        official.validate(pred, 'prediction')
        assert set(pred) == set(truth)
        texts = [pred[k]['text'].strip() for k in keys]
        _, _, f1 = scorer.score(texts, refs)
        values = f1.cpu().numpy()
        rouge = [official.rouge_l_f1(t, r) for t, r in zip(texts, refs)]
        row = dict(n=len(keys), bertscore=float(values.mean()), rouge_l=float(np.mean(rouge)),
                   bleu4=official.bleu4([t.split() for t in texts], [r.split() for r in refs]),
                   bertscore_hash=scorer.hash, clusters=len(clusters))
        if baseline is None:
            baseline = values.copy()
        delta = values - baseline
        rng = np.random.default_rng(781)
        sums = np.array([delta[g].sum() for g in groups])
        sizes = np.array([len(g) for g in groups])
        draws = rng.integers(len(groups), size=(5000, len(groups)))
        boot = sums[draws].sum(1) / sizes[draws].sum(1)
        row['delta_vs_first'] = float(delta.mean())
        row['paired_cluster_95ci'] = np.quantile(boot, [0.025, 0.975]).tolist()
        (path / 'scores.json').write_text(json.dumps(row, indent=2))
        per = {k: dict(bertscore=float(values[i]), rouge_l=rouge[i], **meta[k]) for i, k in enumerate(keys)}
        (path / 'per_example_scores.json').write_text(json.dumps(per))
        print(str(path), json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
