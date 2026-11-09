#!/usr/bin/env python
"""CPU-only audit of document draws versus tokens and saved stream coverage.

Uses the production worker's pick/permutation rules and validates its length-only
simulation against actual packed output state. Does not modify training data.
"""
import argparse
import collections
import importlib.util
import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from saul.data.loader import PackedLMDataset, _WorkerStream
from saul.data.tokens import open_split
from saul.utils.config import load_config


def simulate_one(stream, counts):
    ds = stream.ds
    rng = random.Random(ds.seed * 1_000_000_007 + stream.w * 1_000_003 + stream.seq)
    filled = 0
    while filled < ds.seq_len:
        name, j, off = stream._pick(rng)
        available = max(0, int(ds.sources[name].index[j, 1]) - off)
        if not available:
            continue
        take = min(available, ds.seq_len - filled)
        counts[name] += take
        if available - take >= ds.min_remainder:
            stream.pending.append((name, j, off + take))
        filled += take
    stream.seq += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--sequences-per-worker', type=int, default=4096)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('train30', ROOT/'scripts/30_train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = load_config('configs/train/recovery_b_anneal.yaml')
    weights = train._weights(cfg)
    sources = open_split(cfg.tokens, 'train', list(weights))
    ds = PackedLMDataset(sources, weights, 2048, 0, 1, 2, seed=781)
    # Test the audit itself against the real loader, including pending fragments.
    real = _WorkerStream(ds, 0, 4, None)
    shadow = _WorkerStream(ds, 0, 4, None)
    for _ in range(32):
        real.sequence()
        simulate_one(shadow, collections.Counter())
        assert real.state() == shadow.state()
    report = {'audit_matches_production_for_32_sequences': True,
              'document_draw_probabilities': weights,
              'mean_document_tokens': {n: s.n_tokens/s.n_docs for n,s in sources.items()}}
    norm = sum(weights[n]*s.n_tokens/s.n_docs for n,s in sources.items())
    report['long_run_expected_token_shares'] = {
        n: weights[n]*s.n_tokens/s.n_docs/norm for n,s in sources.items()}
    report['checkpoints'] = {}
    saved_states = {}
    paths = {
        'B_stable': ROOT/'runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt',
        'B_anneal_final': Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final/trainer.pt'),
    }
    for tag, path in paths.items():
        st = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        saved = st['loader']
        saved_states[tag] = saved
        workers = {int(k): v for k,v in saved['workers'].items()}
        W = int(saved['num_workers'])
        coverage = {}
        for name, src in sources.items():
            total_started = unique_started = 0
            per_worker = []
            for w, ws in workers.items():
                shard_docs = len(range(w, src.n_docs, W))
                ep, cursor = int(ws['epoch'][name]), int(ws['cursor'][name])
                total_started += ep*shard_docs + cursor
                unique_started += shard_docs if ep else cursor
                per_worker.append({'worker':w,'epoch':ep,'cursor':cursor})
            coverage[name] = {'docs':src.n_docs,'document_starts':total_started,
                'mean_document_starts':total_started/src.n_docs,
                'unique_started_fraction':unique_started/src.n_docs,
                'workers':per_worker}
        counts = collections.Counter()
        for w, ws in workers.items():
            stream = _WorkerStream(ds, w, W, ws)
            for _ in range(a.sequences_per_worker):
                simulate_one(stream, counts)
        report['checkpoints'][tag] = {'step':st['step'], 'coverage':coverage,
            'next_anneal_tokens_simulated':sum(counts.values()),
            'next_anneal_token_shares':{n:c/sum(counts.values()) for n,c in counts.items()}}
        del st
    # Reconstruct the actual anneal's source totals, including pending fragments.
    # The only unrecorded loss is a dropped one-token remainder per started doc.
    bounds = {}
    for name, src in sources.items():
        emitted_plus_dropped = starts = 0
        for w in range(4):
            before = saved_states['B_stable']['workers']
            after = saved_states['B_anneal_final']['workers']
            b = before[w] if w in before else before[str(w)]
            e = after[w] if w in after else after[str(w)]
            be, ee = b['epoch'][name], e['epoch'][name]
            for epoch in range(be, ee+1):
                perm = np.random.default_rng([781,zlib.crc32(name.encode()),epoch]).permutation(src.n_docs)[w::4]
                lo = b['cursor'][name] if epoch == be else 0
                hi = e['cursor'][name] if epoch == ee else len(perm)
                ids = perm[lo:hi]
                emitted_plus_dropped += int(src.index[ids,1].sum())
                starts += len(ids)
            for sign, ws in ((1,b),(-1,e)):
                emitted_plus_dropped += sign*sum(int(src.index[j,1])-off
                    for n,j,off in ws['pending'] if n == name)
        bounds[name] = {'tokens_upper':emitted_plus_dropped,
                        'tokens_lower':emitted_plus_dropped-starts}
    actual_budget = (21002-19202)*524288
    assert sum(v['tokens_lower'] for v in bounds.values()) <= actual_budget <= sum(v['tokens_upper'] for v in bounds.values())
    report['actual_completed_B_anneal'] = {'total_tokens':actual_budget,
        'source_token_bounds':bounds,
        'source_share_bounds':{n:[v['tokens_lower']/actual_budget,v['tokens_upper']/actual_budget] for n,v in bounds.items()}}
    report['limitations'] = [
        'Coverage counts document starts, not all document tokens consumed; pending fragments remain.',
        'Coverage assumes the current seed, document order and worker count persisted across resumes.',
        'Simulations follow anneal weights, including when started from the stable checkpoint.',
        'Long-run formula ignores tiny dropped remainders and finite pending-pool transients.'
    ]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
