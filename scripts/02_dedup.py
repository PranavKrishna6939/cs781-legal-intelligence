#!/usr/bin/env python
"""Stage 02: global exact + MinHash-LSH near-duplicate removal across all four sources.

Priority when duplicates collide: sources in `cfg.sources` order (supreme_court first, i.e. test-like text wins),
then the longer document.

Outputs (per source, aligned with the doc index n of data/clean/{source})
  data/dedup/{source}/keep.npy      bool, document survives dedup
  data/dedup/{source}/cluster.npy   int64 global cluster id (used for cluster-aware splitting)
  data/stats/dedup_stats.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.data.dedup import (band_keys, exact_hash, lsh_clusters, make_permutations,  # noqa: E402
                             minhash_signature, shingle_hashes)
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ensure_dir, list_shards, write_json  # noqa: E402

_P: dict = {}


def _init(num_perm: int, bands: int, rows: int, k: int) -> None:
    a, b = make_permutations(num_perm)
    _P.update(a=a, b=b, bands=bands, rows=rows, k=k)


def _process_shard(path: str):
    ex, sigs, words = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            t = rec["text"]
            ex.append(exact_hash(t))
            sigs.append(minhash_signature(shingle_hashes(t, _P["k"]), _P["a"], _P["b"]))
            words.append(len(t.split()))
    sig = np.stack(sigs) if sigs else np.zeros((0, len(_P["a"])), np.uint32)
    keys = band_keys(sig, _P["bands"], _P["rows"]) if len(sig) else np.zeros((0, _P["bands"]), np.uint64)
    return (np.asarray(ex, dtype=np.uint64), keys, (sig & 0xFFFF).astype(np.uint16),
            np.asarray(words, dtype=np.int32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dc = cfg.dedup
    out_root = ensure_dir(cfg.paths.dedup)
    sources = list(cfg.sources)
    counts = [len(np.load(os.path.join(cfg.paths.clean, s, "words.npy"))) for s in sources]
    N = int(sum(counts))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    print(f"dedup over {N:,} docs: " + ", ".join(f"{s}={c:,}" for s, c in zip(sources, counts)))

    sig16 = np.lib.format.open_memmap(os.path.join(out_root, "sig16.npy"), mode="w+", dtype=np.uint16,
                                      shape=(N, dc.num_perm))
    keys = np.zeros((N, dc.bands), dtype=np.uint64)
    exact = np.zeros(N, dtype=np.uint64)
    words = np.zeros(N, dtype=np.int32)
    prio = np.zeros(N, dtype=np.int8)
    t0 = time.time()
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(dc.workers, initializer=_init, initargs=(dc.num_perm, dc.bands, dc.rows, dc.shingle_words)) as pool:
        for si, src in enumerate(sources):
            pos = int(offsets[si])
            for ex, kk, s16, w in pool.imap(_process_shard, list_shards(os.path.join(cfg.paths.clean, src))):
                n = len(ex)
                exact[pos:pos + n] = ex
                keys[pos:pos + n] = kk
                sig16[pos:pos + n] = s16
                words[pos:pos + n] = w
                prio[pos:pos + n] = si
                pos += n
            assert pos == offsets[si + 1], f"{src}: {pos} != {offsets[si+1]} (clean output changed?)"
            print(f"  signatures {src} done ({time.time()-t0:.0f}s)", flush=True)
    sig16.flush()

    # exact dedup: first occurrence in priority order wins
    _, first = np.unique(exact, return_index=True)
    exact_keep = np.zeros(N, dtype=bool)
    exact_keep[first] = True
    active = exact_keep & (words >= dc.min_words_for_minhash)
    roots, merges = lsh_clusters(keys, sig16, active, dc.threshold)
    roots = np.where(active, roots, np.arange(N))
    # representative per cluster: best priority, then longest
    order = np.lexsort((-words.astype(np.int64), prio, roots))
    r_sorted = roots[order]
    is_first = np.ones(N, dtype=bool)
    is_first[1:] = r_sorted[1:] != r_sorted[:-1]
    is_rep = np.zeros(N, dtype=bool)
    is_rep[order[is_first]] = True
    keep = exact_keep & is_rep
    cluster = roots

    stats = {"total": N, "exact_dups": int(N - exact_keep.sum()), "near_merges": int(merges),
             "kept": int(keep.sum()), "per_source": {}}
    for si, src in enumerate(sources):
        s, e = offsets[si], offsets[si + 1]
        d = ensure_dir(os.path.join(out_root, src))
        np.save(os.path.join(d, "keep.npy"), keep[s:e])
        np.save(os.path.join(d, "cluster.npy"), cluster[s:e])
        stats["per_source"][src] = {
            "docs": int(e - s), "kept": int(keep[s:e].sum()),
            "exact_removed": int((~exact_keep[s:e]).sum()),
            "near_removed": int((exact_keep[s:e] & ~keep[s:e]).sum()),
            "kept_words": int(words[s:e][keep[s:e]].sum(dtype=np.int64)),
        }
    stats["seconds"] = round(time.time() - t0, 1)
    write_json(stats, os.path.join(ensure_dir(cfg.paths.stats), "dedup_stats.json"))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
