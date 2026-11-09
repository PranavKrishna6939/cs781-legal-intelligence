#!/usr/bin/env python
"""Freeze document-cluster-disjoint screen/confirmation sets and gold penalty diagnostics."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saul.utils.io import iter_shards, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/recovery_20261005/dev')
    a = p.parse_args()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained('artifacts/tokenizers/legal32k')
    prompts = read_json('data/final/dev_prompts.json')
    truth = read_json('data/final/dev_truth.json')
    meta = read_json('data/final/dev_meta.json')
    wanted = {v['doc_id']: k for k, v in meta.items()}
    found = set()
    for source in sorted({v['source'] for v in meta.values()}):
        for rec in iter_shards(f'data/final/dev/{source}'):
            if rec['id'] in wanted:
                k = wanted[rec['id']]
                meta[k]['cluster'] = f"{source}:{rec['meta']['cluster']}"
                found.add(k)
    assert found == set(meta), 'Missing dev source/cluster metadata'
    groups = collections.defaultdict(list)
    diagnostics = {}
    for k in prompts:
        groups[meta[k]['cluster']].append(k)
        prompt = prompts[k]['text']
        joint = prompt + ' ' + truth[k]['text']
        enc = tok(joint, add_special_tokens=False, return_offsets_mapping=True)
        target = [t for t, (_, e) in zip(enc['input_ids'], enc['offset_mapping']) if e > len(prompt)]
        seen = set()
        banned = []
        for i in range(3, len(target)):
            gram = tuple(target[i-3:i+1])
            if gram in seen:
                banned.append(i)
            seen.add(gram)
        plen = len(tok.encode(prompt, add_special_tokens=False)) + 1
        diagnostics[k] = {'prompt_tokens_with_bos': plen, 'gold_tokens': len(target),
                          'gold_4gram_banned_positions': banned,
                          'gold_4gram_banned_count': len(banned)}
    # Preserve full-dev source proportions. Select whole clusters, never split a document family.
    counts = collections.Counter(v['source'] for v in meta.values())
    remaining = set(groups)
    partitions = {}
    for name, size in [('screen', 384), ('confirmation', 1000)]:
        picked = []
        for src, count in sorted(counts.items()):
            target_count = round(size * count / len(meta))
            candidates = sorted((g for g in remaining if meta[groups[g][0]]['source'] == src),
                                key=lambda g: hashlib.sha256(('recovery781:' + g).encode()).hexdigest())
            n = 0
            for g in candidates:
                if n >= target_count:
                    break
                picked.extend(groups[g]); n += len(groups[g]); remaining.remove(g)
        keys = [k for k in prompts if k in set(picked)]
        partitions[name] = keys
        for suffix, data in [('prompts', prompts), ('truth', truth), ('meta', meta)]:
            write_json({k: data[k] for k in keys}, f'{a.out}/{name}_{suffix}.json')
    assert not ({meta[k]['cluster'] for k in partitions['screen']} &
                {meta[k]['cluster'] for k in partitions['confirmation']})
    write_json(diagnostics, f'{a.out}/gold_penalty_diagnostics.json')
    summary = {'partitions': {n: {'n': len(ks), 'sources': dict(collections.Counter(meta[k]['source'] for k in ks))}
                              for n, ks in partitions.items()},
               'full_dev': len(prompts), 'gold_examples_with_banned_4gram': sum(bool(v['gold_4gram_banned_count']) for v in diagnostics.values()),
               'gold_banned_tokens': sum(v['gold_4gram_banned_count'] for v in diagnostics.values()),
               'gold_tokens': sum(v['gold_tokens'] for v in diagnostics.values()),
               'prompt_max_tokens': max(v['prompt_tokens_with_bos'] for v in diagnostics.values()),
               'overflow_policy': 'explicit truncate_left to 2048 tokens, matching historical dev',
               'overflow_ids': [k for k, v in diagnostics.items() if v['prompt_tokens_with_bos'] > 2048]}
    write_json(summary, f'{a.out}/manifest.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
