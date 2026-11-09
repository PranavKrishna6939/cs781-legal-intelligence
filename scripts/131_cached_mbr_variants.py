#!/usr/bin/env python3
"""Materialize predeclared N128/N64 selector diagnostics from HTTP feature caches.

CPU only: no generation, feature extraction, scoring, promotion or test package.
Run after the corresponding partition has finished in script 125.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def materialize(root, partition, out):
    protocol = read(root/'protocol.json')
    parent = root/'selected128'/partition
    feature_path = parent/'features.jsonl'
    rows = [json.loads(line) for line in feature_path.open()]
    features = {row['id']: row for row in rows}
    assert len(features) == len(rows), 'Duplicate cached IDs'
    prompts_path = ROOT/f'runs/recovery_20261005/dev/{partition}_prompts.json'
    prompts = read(prompts_path)
    assert set(features) == set(prompts), 'Partition feature collection is incomplete'
    manifests = {pool: read(root/directory/partition/'manifest.json')
                 for pool, directory in [('nested', 'selected128'), ('control', 'tiled64')]}
    for pool, manifest in manifests.items():
        assert manifest['checkpoint'] == protocol['checkpoint']
        assert not manifest.get('selection_pending', False)
        assert manifest['prompts_sha256'] == hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        native_path = root/('selected128' if pool == 'nested' else 'tiled64')/partition/'prediction.json'
        native = read(native_path)
        assert all(features[k][pool]['selected'] == native[k]['text'] for k in prompts)
    out.mkdir(parents=True, exist_ok=False)
    outputs = []
    for pool, n in [('nested', 128), ('control', 64)]:
        for selector in ['alpha_025', 'alpha_075', 'boundary']:
            name = f'n{n}_{selector}'
            pred = {}
            for key in prompts:
                feature = features[key][pool]
                if selector == 'boundary':
                    text = feature['boundary']['selected']
                else:
                    text = feature['alpha_picks']['0.25' if selector == 'alpha_025' else '0.75']
                assert isinstance(text, str)
                pred[key] = {'text': text}
            dest = out/name
            dest.mkdir()
            (dest/'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
            manifest = dict(manifests[pool], selector=f'cached_{name}',
                            feature_cache=str(feature_path),
                            feature_cache_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                            selection_pending=False, auxiliary_not_promoted=True,
                            incomplete_outputs={k: len(v['text'].split()) for k, v in pred.items()
                                                if len(v['text'].split()) != 128})
            manifest['sampling'] = dict(manifest['sampling'])
            manifest['sampling']['mbr'] = dict(manifest['sampling']['mbr'],
                alpha=.25 if selector == 'alpha_025' else .75 if selector == 'alpha_075' else .5)
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            outputs.append(str(dest))
    result = dict(partition=partition, runs=outputs, status='prepared_unscored',
                  adaptive_reused_dev=True, test_package_created=False)
    (out/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--partition', choices=['screen', 'confirmation'], required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root.resolve(), args.partition, args.out.resolve()), indent=2))


if __name__ == '__main__':
    main()
