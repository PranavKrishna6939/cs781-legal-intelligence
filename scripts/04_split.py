#!/usr/bin/env python
"""Stage 04: cluster-aware train/val/dev split and the Phase-1 dev prompt set.

Outputs
  data/final/{train,val,dev}/{source}/shard_*.jsonl.gz
  data/final/dev_prompts.json, dev_truth.json            full dev set (evaluate_phase1.py schema)
  data/final/dev_prompts_fast.json, dev_truth_fast.json  500-prompt SC-heavy subset for sweeps
  data/final/dev_meta.json                               id -> {source, doc_id}
  data/stats/split_stats.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.data.split import assign_split, cut_dev_example  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ShardWriter, ensure_dir, iter_shards, read_json, write_json  # noqa: E402
from saul.utils.seed import stable_unit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    sp = cfg.split
    anon_stats = read_json(os.path.join(cfg.paths.stats, "anon_stats.json"))
    final = cfg.paths.final
    for s in ("train", "val", "dev"):
        d = os.path.join(final, s)
        if os.path.isdir(d) and any(os.listdir(os.path.join(d, x)) for x in os.listdir(d)
                                    if os.path.isdir(os.path.join(d, x))):
            raise SystemExit(f"{d} not empty")

    stats = {}
    dev_candidates: dict[str, list[tuple[float, str, str, str]]] = {}
    for src in cfg.sources:
        st = anon_stats[src]
        val_words = sp.val_tokens_per_source / sp.tokens_per_word
        f_val = min(0.2, val_words / max(st["words"], 1))
        f_dev = min(0.2, sp.dev_docs[src] * sp.dev_oversample / max(st["docs"], 1))
        writers = {s: ShardWriter(os.path.join(final, s, src), max_records=cfg.clean.shard_records)
                   for s in ("train", "val", "dev")}
        counts = {s: {"docs": 0, "words": 0} for s in writers}
        cands = []
        for rec in iter_shards(os.path.join(cfg.paths.anon, src)):
            split = assign_split(rec["meta"]["cluster"], f_val, f_dev, src)
            writers[split].write(rec)
            counts[split]["docs"] += 1
            counts[split]["words"] += len(rec["text"].split())
            # Test prompts never contain injected tags, so dev prompts come only from untouched documents.
            if split == "dev" and not rec["meta"].get("anonymized", False):
                ex = cut_dev_example(rec["text"], rec["id"], sp.prompt_words, sp.cont_words,
                                     sp.dev_min_offset_words, sp.seed)
                if ex is not None:
                    cands.append((stable_unit(rec["id"], "devorder"), rec["id"], ex[0], ex[1]))
        for w in writers.values():
            w.close()
        cands.sort()
        dev_candidates[src] = cands[: sp.dev_docs[src]]
        stats[src] = {"f_val": f_val, "f_dev": f_dev, **counts, "dev_prompts": len(dev_candidates[src])}
        if len(dev_candidates[src]) < sp.dev_docs[src]:
            print(f"[warn] {src}: only {len(dev_candidates[src])} dev prompts (wanted {sp.dev_docs[src]})")
        print(f"[{src}] {stats[src]}")

    prompts, truth, meta = {}, {}, {}
    fast_p, fast_t = {}, {}
    k = 0
    n_fast_other = 0
    for src in cfg.sources:
        for i, (_, doc_id, p, t) in enumerate(dev_candidates[src]):
            key = f"dev_{k:05d}"
            k += 1
            prompts[key] = {"text": p}
            truth[key] = {"text": t}
            meta[key] = {"source": src, "doc_id": doc_id}
            if src == "supreme_court" and i < sp.fast_subset["supreme_court"]:
                fast_p[key], fast_t[key] = prompts[key], truth[key]
            elif src != "supreme_court" and n_fast_other < sp.fast_subset["other"] and i < sp.fast_subset["other"] // 3 + 1:
                fast_p[key], fast_t[key] = prompts[key], truth[key]
                n_fast_other += 1
    ensure_dir(final)
    write_json(prompts, os.path.join(final, "dev_prompts.json"))
    write_json(truth, os.path.join(final, "dev_truth.json"))
    write_json(fast_p, os.path.join(final, "dev_prompts_fast.json"))
    write_json(fast_t, os.path.join(final, "dev_truth_fast.json"))
    write_json(meta, os.path.join(final, "dev_meta.json"))
    stats["dev_total"] = len(prompts)
    stats["dev_fast"] = len(fast_p)
    write_json(stats, os.path.join(ensure_dir(cfg.paths.stats), "split_stats.json"))
    print(f"dev prompts: {len(prompts)} (fast subset {len(fast_p)})")


if __name__ == "__main__":
    main()
