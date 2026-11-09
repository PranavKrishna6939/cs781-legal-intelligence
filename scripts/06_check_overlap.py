#!/usr/bin/env python
"""Stage 06: 13-gram overlap between evaluation texts and the training corpus.

Checks (report 3.2 step 5, 3.3): (a) Phase-1 test prompts vs the Phase-1 *train* split; (b) Phase-6 test documents
vs the Supreme-Court file (the In-Abs headnote source). Results are for disclosure / TA questions and for trusting
the dev set; nothing is filtered automatically.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ensure_dir, iter_jsonl, list_shards, read_json, write_json  # noqa: E402

N = 13
_Q: dict = {}


def ngram_hashes(text: str) -> np.ndarray:
    w = text.lower().split()
    if len(w) < N:
        return np.empty(0, np.uint64)
    h = np.fromiter((zlib.crc32(x.encode()) for x in w), dtype=np.uint64, count=len(w))
    n = len(w) - N + 1
    acc = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N):
            acc = acc * np.uint64(0x100000001B3) + h[j:j + n]
    return acc


def _init(query: np.ndarray) -> None:
    _Q["q"] = np.unique(query)


def _scan(path: str) -> np.ndarray:
    q = _Q["q"]
    found = []
    if q.size == 0:
        return np.empty(0, np.uint64)
    for rec in iter_jsonl([path]):
        g = ngram_hashes(rec["text"])
        if g.size:
            idx = np.minimum(np.searchsorted(q, g), q.size - 1)
            m = q[idx] == g
            if m.any():
                found.append(np.unique(g[m]))
    return np.unique(np.concatenate(found)) if found else np.empty(0, np.uint64)


def overlap_report(queries: dict[str, str], shard_paths: list[str], workers: int) -> dict:
    qh = {k: ngram_hashes(v) for k, v in queries.items()}
    allq = np.unique(np.concatenate([x for x in qh.values() if x.size])) if qh else np.empty(0, np.uint64)
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(allq,)) as pool:
        parts = pool.map(_scan, shard_paths)
    hit = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
    per = {k: float(np.isin(v, hit).mean()) if v.size else 0.0 for k, v in qh.items()}
    vals = np.array(list(per.values()))
    return {
        "n": len(per), "mean_overlap": float(vals.mean()) if vals.size else 0.0,
        "n_over_50pct": int((vals > 0.5).sum()), "n_over_90pct": int((vals > 0.9).sum()),
        "n_any": int((vals > 0).sum()), "per_item": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    out = {}
    test = read_json(os.path.join(cfg.paths.raw, cfg.hf.test_file))
    shards = [p for s in cfg.sources for p in list_shards(os.path.join(cfg.paths.final, "train", s))]
    out["phase1_test_vs_train"] = overlap_report({k: v["text"] for k, v in test.items()}, shards, a.workers)
    print("phase1 test vs train:", {k: v for k, v in out["phase1_test_vs_train"].items() if k != "per_item"})
    p6 = os.path.join(cfg.paths.raw, "Phase_6_Summ/Test/test.json")
    if os.path.exists(p6):
        data = read_json(p6)
        items = data if isinstance(data, list) else [{"id": k, "document": v} for k, v in data.items()]
        docs = {str(r["id"]): (r.get("document") or r.get("text") or "") for r in items}
        sc = [p for sp in ("train", "val", "dev") for p in list_shards(os.path.join(cfg.paths.final, sp, "supreme_court"))]
        out["phase6_test_vs_sc"] = overlap_report(docs, sc, a.workers)
        print("phase6 test vs SC:", {k: v for k, v in out["phase6_test_vs_sc"].items() if k != "per_item"})
    write_json(out, os.path.join(ensure_dir(cfg.paths.stats), "overlap.json"))


if __name__ == "__main__":
    main()
