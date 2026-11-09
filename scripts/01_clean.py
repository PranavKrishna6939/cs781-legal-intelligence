#!/usr/bin/env python
"""Stage 01: stream-parse the four Phase-1 JSON arrays, build + clean document text, write gzip JSONL shards.

Outputs
  data/clean/{source}/shard_*.jsonl.gz     records {id, source, text, meta}; id = "{source}:{n}" (n contiguous)
  data/clean/{source}/words.npy             int32 word counts aligned with n
  data/stats/test_prompt_format.json        formatting stats of the 1,000 official test prompts
  data/stats/whitespace_decision.json       the whitespace convention applied to all training text
  data/stats/clean_stats.json               per-source doc / word / byte counts

    python scripts/01_clean.py --config configs/data.yaml [--sources supreme_court acts] [--limit 1000]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.data.clean import (build_text, clean_document, decide_whitespace_mode,  # noqa: E402
                             inspect_prompts)
from saul.data.stream import iter_json_array  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ShardWriter, ensure_dir, read_json, write_json  # noqa: E402
from saul.utils.seed import stable_unit  # noqa: E402

_W: dict = {}


def _init_worker(source: str, mode: str, mixed_p: float, min_words: int, rep_min: int, rep_max: int) -> None:
    _W.update(source=source, mode=mode, mixed_p=mixed_p, min_words=min_words, rep_min=rep_min, rep_max=rep_max)


def _clean_batch(batch: list[dict]) -> list[tuple[str, dict, int, int] | None]:
    out = []
    src = _W["source"]
    for rec in batch:
        text, meta = build_text(rec, src)
        if not text:
            out.append(None)
            continue
        mode = _W["mode"]
        if mode == "mixed":
            keep_nl = stable_unit(text[:256], "ws") >= _W["mixed_p"]
        else:
            keep_nl = mode == "keep_newlines"
        text = clean_document(text, src, keep_newlines=keep_nl, hc_rep_min=_W["rep_min"], hc_rep_max=_W["rep_max"])
        nw = len(text.split())
        if nw < _W["min_words"]:
            out.append(None)
            continue
        out.append((text, meta, nw, len(text.encode("utf-8"))))
    return out


def _batches(it, size: int, limit: int | None):
    buf, n = [], 0
    for rec in it:
        buf.append(rec)
        n += 1
        if len(buf) >= size:
            yield buf
            buf = []
        if limit is not None and n >= limit:
            break
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max raw records per source (debug)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    stats_dir = ensure_dir(cfg.paths.stats)

    # --- decide the whitespace convention from the official test prompts --------------------------------
    test_path = os.path.join(cfg.paths.raw, cfg.hf.test_file)
    test = read_json(test_path)
    fmt = inspect_prompts(v["text"] for v in test.values())
    mode = decide_whitespace_mode(fmt, cfg.clean.whitespace_mode)
    write_json(fmt, os.path.join(stats_dir, "test_prompt_format.json"))
    write_json({"configured": cfg.clean.whitespace_mode, "applied": mode, "test_format": fmt},
               os.path.join(stats_dir, "whitespace_decision.json"))
    print(f"test prompt format: {json.dumps(fmt)}\n-> whitespace mode: {mode}")

    sources = a.sources or list(cfg.sources)
    all_stats = {}
    stats_path = os.path.join(stats_dir, "clean_stats.json")
    if os.path.exists(stats_path):
        all_stats = read_json(stats_path)
    for src in sources:
        t0 = time.time()
        in_path = os.path.join(cfg.paths.raw, cfg.hf.phase1_files[src])
        out_dir = os.path.join(cfg.paths.clean, src)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            raise SystemExit(f"{out_dir} is not empty; delete it to re-run cleaning for {src}")
        writer = ShardWriter(out_dir, max_records=cfg.clean.shard_records)
        words: list[int] = []
        n_in = n_bytes = 0
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(cfg.clean.workers, initializer=_init_worker,
                      initargs=(src, mode, cfg.clean.mixed_collapse_prob, cfg.clean.min_words,
                                cfg.clean.hc_repeated_line_min_count, cfg.clean.hc_repeated_line_max_chars)) as pool:
            for res in pool.imap(_clean_batch, _batches(iter_json_array(in_path), cfg.clean.batch_size, a.limit),
                                 chunksize=1):
                for r in res:
                    n_in += 1
                    if r is None:
                        continue
                    text, meta, nw, nb = r
                    n = len(words)
                    writer.write({"id": f"{src}:{n}", "source": src, "text": text, "meta": meta})
                    words.append(nw)
                    n_bytes += nb
                if n_in % 200_000 < cfg.clean.batch_size:
                    print(f"  {src}: {n_in:,} read, {len(words):,} kept, {time.time()-t0:.0f}s", flush=True)
        writer.close()
        np.save(os.path.join(out_dir, "words.npy"), np.asarray(words, dtype=np.int32))
        all_stats[src] = {
            "docs_in": n_in, "docs_kept": len(words), "words": int(np.sum(words, dtype=np.int64)),
            "bytes": n_bytes, "seconds": round(time.time() - t0, 1),
        }
        print(f"[{src}] {all_stats[src]}")
        write_json(all_stats, stats_path)


if __name__ == "__main__":
    main()
