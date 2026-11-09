#!/usr/bin/env python
"""Stage 05: word/byte shares per split+source and the estimated test source mix.

A TF-IDF + logistic-regression classifier is trained on up to 5,000 training documents per source (first 3,000
words each, whitespace-joined like the prompts) and applied to the 1,000 test prompts. The averaged class
probabilities become data/stats/test_source_mix.json, which drives the anneal-phase mixture.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.data.stats import estimate_mix, train_source_classifier  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ensure_dir, iter_shards, read_json, write_json  # noqa: E402
from saul.utils.seed import stable_unit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--per_source", type=int, default=5000)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    texts, labels, shares = [], [], {}
    for split in ("train", "val", "dev"):
        shares[split] = {}
        for src in cfg.sources:
            d = os.path.join(cfg.paths.final, split, src)
            if not os.path.isdir(d):
                continue
            nd = nw = nb = 0
            taken = 0
            for rec in iter_shards(d):
                nd += 1
                w = rec["text"].split()
                nw += len(w)
                nb += len(rec["text"].encode("utf-8"))
                if split == "train" and taken < a.per_source and stable_unit(rec["id"], "clf") < 0.2:
                    # random 512-word window, like a test prompt
                    s = int(stable_unit(rec["id"], "win") * max(1, len(w) - 512))
                    texts.append(" ".join(w[s:s + 512]))
                    labels.append(src)
                    taken += 1
            shares[split][src] = {"docs": nd, "words": nw, "bytes": nb}
    stats_dir = ensure_dir(cfg.paths.stats)
    write_json(shares, os.path.join(stats_dir, "split_shares.json"))
    clf = train_source_classifier(texts, labels)
    test = read_json(os.path.join(cfg.paths.raw, cfg.hf.test_file))
    mix = estimate_mix(clf, (v["text"] for v in test.values()))
    write_json(mix, os.path.join(stats_dir, "test_source_mix.json"))
    dev = os.path.join(cfg.paths.final, "dev_prompts.json")
    if os.path.exists(dev):
        dev_mix = estimate_mix(clf, (v["text"] for v in read_json(dev).values()))
        write_json(dev_mix, os.path.join(stats_dir, "dev_source_mix_predicted.json"))
    print("estimated test source mix:", {k: round(v, 3) for k, v in mix.items()})


if __name__ == "__main__":
    main()
