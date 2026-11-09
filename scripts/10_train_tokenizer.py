#!/usr/bin/env python
"""Stage 10: train the legal byte-level BPE variants on a source-weighted sample of the cleaned+anonymized train split.

    python scripts/10_train_tokenizer.py --config configs/tokenizer.yaml --variants legal32k legal24k legal49k
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.tokenizer.specials import build_special_tokens  # noqa: E402
from saul.tokenizer.train import REGEXES, sample_texts, save_tokenizer, train_bpe  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import list_shards, read_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dcfg = load_config(cfg.data_config)
    tags_path = dcfg.anonymize.tags_file
    tags = read_json(tags_path)["tags"] if os.path.exists(tags_path) else [dcfg.anonymize.default_tags.name,
                                                                          dcfg.anonymize.default_tags.org]
    specials = build_special_tokens(tags)
    shares = read_json(os.path.join(dcfg.paths.stats, "split_shares.json"))["train"]
    source_bytes = {s: shares[s]["bytes"] for s in dcfg.sources}
    tot_w = sum(float(v) for v in cfg.source_weights.values())
    sample_bytes = float(cfg.sample_bytes)
    targets = {s: sample_bytes * float(cfg.source_weights[s]) / tot_w for s in dcfg.sources}
    shard_dirs = {s: list_shards(os.path.join(dcfg.paths.final, "train", s)) for s in dcfg.sources}
    for name in a.variants or [cfg.default]:
        v = cfg.variants[name]
        print(f"training {name}: vocab={v.vocab_size} regex={v.regex} specials={len(specials)} (tags={tags})")
        tok = train_bpe(sample_texts(shard_dirs, targets, source_bytes), v.vocab_size, specials,
                        cfg.min_frequency, REGEXES[v.regex])
        out = os.path.join(cfg.out_root, name)
        save_tokenizer(tok, out, specials, {"regex": v.regex, "anon_tags": tags, "sample_targets": targets})
        print(f"saved {out}")


if __name__ == "__main__":
    main()
