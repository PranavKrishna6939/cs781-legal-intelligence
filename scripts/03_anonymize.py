#!/usr/bin/env python
"""Stage 03b: drop dedup losers and inject IL-TUR-style anonymization into a deterministic subset of documents.

Outputs
  data/anon/{source}/shard_*.jsonl.gz    kept documents; meta.anonymized / meta.statute_masked flags
  data/stats/anon_stats.json            per-source kept docs/words/bytes (used by stage 04 for split budgets)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.data.anonymize import anonymize_names_orgs, anonymize_statutes  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ShardWriter, ensure_dir, iter_shards, read_json, write_json  # noqa: E402
from saul.utils.seed import stable_unit  # noqa: E402

_A: dict = {}


def _init(frac: float, sfrac: float, roles: dict) -> None:
    _A.update(frac=frac, sfrac=sfrac, roles=roles)


def _anon(rec: dict) -> dict:
    u = stable_unit(rec["id"], "anon")
    roles = _A["roles"]
    rec["meta"]["anonymized"] = False
    rec["meta"]["statute_masked"] = False
    if u < _A["frac"]:
        rec["text"] = anonymize_names_orgs(rec["text"], roles["name"], roles["org"])
        rec["meta"]["anonymized"] = True
        if roles.get("statute") and u < _A["sfrac"]:
            rec["text"] = anonymize_statutes(rec["text"], roles["statute"])
            rec["meta"]["statute_masked"] = True
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    if os.path.exists(cfg.anonymize.tags_file):
        roles = read_json(cfg.anonymize.tags_file)["roles"]
    else:
        print(f"[warn] {cfg.anonymize.tags_file} missing; using default tags. Run 03_inspect_downstream_tags.py first!")
        roles = dict(cfg.anonymize.default_tags)
    # statute masking is applied to a subset of the anonymised docs: u < statute_frac <= frac
    sfrac = min(cfg.anonymize.statute_frac, cfg.anonymize.frac)
    stats = {}
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    for src in cfg.sources:
        keep = np.load(os.path.join(cfg.paths.dedup, src, "keep.npy"))
        cluster = np.load(os.path.join(cfg.paths.dedup, src, "cluster.npy"))
        out_dir = os.path.join(cfg.paths.anon, src)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            raise SystemExit(f"{out_dir} not empty")
        w = ShardWriter(out_dir, max_records=cfg.clean.shard_records)
        n_words = n_bytes = n_anon = 0

        def kept():
            for rec in iter_shards(os.path.join(cfg.paths.clean, src)):
                n = int(rec["id"].split(":")[1])
                if keep[n]:
                    rec["meta"]["cluster"] = int(cluster[n])
                    yield rec

        with ctx.Pool(a.workers, initializer=_init, initargs=(cfg.anonymize.frac, sfrac, roles)) as pool:
            for rec in pool.imap(_anon, kept(), chunksize=256):
                w.write(rec)
                n_words += len(rec["text"].split())
                n_bytes += len(rec["text"].encode("utf-8"))
                n_anon += rec["meta"]["anonymized"]
        w.close()
        stats[src] = {"docs": w.total, "words": n_words, "bytes": n_bytes, "anonymized": n_anon}
        print(f"[{src}] {stats[src]}")
    write_json(stats, os.path.join(ensure_dir(cfg.paths.stats), "anon_stats.json"))


if __name__ == "__main__":
    main()
