#!/usr/bin/env python
"""Official-train gold continuations and a matched, DSIR-style selection variant.

No generated targets. One window per source document, deterministic document reservoir.
The full source document is checked against public test prompts and held-out dev text.
Boilerplate exemptions require >=50 observed official-train documents.
"""
import argparse
import collections
import hashlib
import heapq
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saul.data.split import cut_dev_example
from saul.utils.io import iter_shards, read_json, write_json

SOURCES = ['supreme_court', 'acts', 'tribunals', 'high_courts']


def hashed(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


def grams(text):
    words = re.findall(r'\w+', text.lower())
    return {hashed(' '.join(words[i:i+13])) for i in range(len(words)-12)}


def features(text):
    words = re.findall(r'\w+', text.lower())
    # Stable hashed unigram/bigram multinomial, 10k buckets as in the curation report.
    return collections.Counter([hashed(w) % 10000 for w in words] +
                               [hashed(words[i] + ' ' + words[i+1]) % 10000 for i in range(len(words)-1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/recovery_20261005/curation')
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--pool', type=int, default=26000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test = read_json('data/raw/Phase_1_Pretraining/Test/test.json')
    dev_p = read_json('data/final/dev_prompts.json')
    dev_t = read_json('data/final/dev_truth.json')
    blocked = set()
    for r in test.values():
        blocked.update(grams(r['text']))
    for k in dev_p:
        blocked.update(grams(dev_p[k]['text'] + ' ' + dev_t[k]['text']))
    held_clusters = set()
    for split in ['dev', 'val']:
        for src in SOURCES:
            for r in iter_shards(f'data/final/{split}/{src}'):
                held_clusters.add((src, r['meta']['cluster']))
    print('blocklist', len(blocked), 'heldout_clusters', len(held_clusters), flush=True)
    # Pass one retains only IDs, so corpus-sized text never accumulates in memory.
    reservoirs = {}
    boilerplate_counts = collections.Counter()
    for src in SOURCES:
        heap = []
        for doc_number, r in enumerate(iter_shards(f'data/final/train/{src}')):
            if doc_number < 1000:
                boilerplate_counts.update(grams(r['text']) & blocked)
            if r['meta'].get('anonymized') or r['meta'].get('statute_masked'):
                continue
            if (src, r['meta']['cluster']) in held_clusters:
                continue
            if len(r['text'].split()) < 690:
                continue
            rank = hashed('pool781:' + r['id'])
            item = (-rank, r['id'])
            if len(heap) < a.pool:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        reservoirs[src] = {rid for _, rid in heap}
        print('reservoir', src, len(heap), flush=True)
    boilerplate = {g for g, count in boilerplate_counts.items() if count >= 50}
    blocked.difference_update(boilerplate)
    print('observed_train_boilerplate_removed', len(boilerplate), flush=True)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained('artifacts/tokenizers/legal32k')
    pools = {}
    stats = {}
    for src in SOURCES:
        pool = []
        counters = collections.Counter()
        used_clusters = set()
        for r in iter_shards(f'data/final/train/{src}'):
            if r['id'] not in reservoirs[src]:
                continue
            counters['candidate_docs'] += 1
            if r['meta']['cluster'] in used_clusters:
                counters['duplicate_cluster'] += 1
                continue
            # Union threshold can overexclude generic boilerplate; it cannot let a
            # >=3-match single test input through. Log this stricter-than-report choice.
            if len(grams(r['text']) & blocked) >= 3:
                counters['overlap_dropped'] += 1
                continue
            ex = cut_dev_example(r['text'], r['id'], seed=1781)
            if ex is None:
                continue
            prompt, target = ex
            joint = prompt + ' ' + target
            enc = tok(joint, add_special_tokens=False, return_offsets_mapping=True)
            if len(enc['input_ids']) + 1 > 2048:
                counters['long_dropped'] += 1
                continue
            # Include a boundary-spanning BPE token in the continuation loss.
            first = next(i for i, (_, end) in enumerate(enc['offset_mapping']) if end > len(prompt))
            ids = [tok.bos_token_id] + enc['input_ids']
            start = first + 1
            row = dict(id=r['id'], source=src, cluster=f"{src}:{r['meta']['cluster']}",
                       prompt=prompt, target=target, input_ids=ids, target_start=start,
                       source_doc_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                       generator='gold_cut_1781', split='train')
            assert start > 0 and len(ids) - start > 0
            pool.append(row)
            used_clusters.add(r['meta']['cluster'])
        pools[src] = pool
        counters['accepted'] = len(pool)
        stats[src] = dict(counters)
        with (out / f'filtered_{src}.jsonl').open('w') as f:
            for row in pool:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('filtered', src, stats[src], flush=True)
    # Fit on one fold of official-train documents, score a separate fold. Never fit
    # selection densities to dev/test targets or use the official BERTScore model.
    target_counts, raw_counts = collections.Counter(), collections.Counter()
    fit_ids = set()
    for src in ['supreme_court', 'tribunals', 'high_courts']:
        fit = sorted(pools[src], key=lambda r: hashed('fit:' + r['id']))[:min(1000, len(pools[src])//5)]
        for r in fit:
            fit_ids.add(r['id'])
            dest = target_counts if src == 'supreme_court' else raw_counts
            dest.update(features(r['prompt'] + ' ' + r['target']))
    nt, nr = sum(target_counts.values()) + 10000, sum(raw_counts.values()) + 10000
    log_ratio = [math.log((target_counts[i]+1)/nt) - math.log((raw_counts[i]+1)/nr) for i in range(10000)]
    for src in SOURCES:
        pools[src] = [r for r in pools[src] if r['id'] not in fit_ids]
        for r in pools[src]:
            f = features(r['prompt'] + ' ' + r['target'])
            r['dsir_mean_log_ratio'] = sum(log_ratio[i]*c for i,c in f.items()) / max(1, sum(f.values()))
    proportions = dict(zip(SOURCES, [.78, .08, .07, .07]))
    available_n = min(int(len(pools[src])/proportions[src]) for src in SOURCES)
    actual_n = min(a.n, (available_n//1000)*1000)
    assert actual_n >= 5000, ('Too few decontaminated examples for useful pilot', available_n)
    quotas = {src: round(actual_n*proportions[src]) for src in SOURCES}
    selected = {}
    for variant in ['random', 'dsir']:
        rows = []
        for src in SOURCES:
            pool = pools[src]
            n = quotas[src]
            assert len(pool) >= n, (src, len(pool), n)
            if variant == 'dsir' and src in ['tribunals', 'high_courts']:
                # Length-normalized density with fixed temperature, plus Gumbel
                # sampling: explicitly a small-data DSIR-style variant, not a
                # reproduction of the report's unread-token anneal proposal.
                def rank(r):
                    u = (hashed('select781:' + r['id']) + .5) / 2**64
                    u = min(1-1e-15, max(1e-15, u))
                    return r['dsir_mean_log_ratio'] / .1 - math.log(-math.log(u))
                pick = sorted(pool, key=rank, reverse=True)[:n]
            else:
                pick = sorted(pool, key=lambda r: hashed('select781:' + r['id']))[:n]
            rows.extend(pick)
        rows.sort(key=lambda r: hashed('order781:' + r['id']))
        with (out / f'{variant}.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        selected[variant] = {'n': len(rows), 'mean_tokens': sum(len(r['input_ids']) for r in rows)/len(rows),
                             'mean_dsir': {s: sum(r['dsir_mean_log_ratio'] for r in rows if r['source']==s)/quotas[s] for s in SOURCES}}
    write_json(dict(stats=stats, quotas=quotas, requested_n=a.n, actual_n=actual_n,
                    selected=selected, density_fit_docs=len(fit_ids),
                    decontamination='full-document >=3 shared normalized 13grams against union of test prompts and dev prompt+truth; boilerplate requires >=50 observed train docs in first 1000/source',
                    boilerplate_removed=len(boilerplate),
                    caveats=['Conservative overlap gate may overexclude boilerplate.',
                             'Documents are not certified unread: training stream exposure is not reconstructed.',
                             'DSIR-style mean log-density selector; source proportions fixed at 78/8/7/7.',
                             'No hard-example oversampling in this initial controlled pair.']), str(out/'manifest.json'))
    print('COMPLETE', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
