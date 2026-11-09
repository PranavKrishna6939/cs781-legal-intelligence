#!/usr/bin/env python
"""Build a 1,000-prompt dev set from training documents (memorization upper bound).

800 Supreme Court and 200 Acts. Same cut as the held-out dev set: 512-word prompt,
128-word continuation, offset at least 300 words, untouched documents only.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.data.split import cut_dev_example  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import iter_shards, write_json  # noqa: E402
from saul.utils.seed import stable_unit  # noqa: E402

QUOTA = {"supreme_court": 800, "acts": 200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out_prompts", default="data/final/dev_intrain_prompts.json")
    ap.add_argument("--out_truth", default="data/final/dev_intrain_truth.json")
    ap.add_argument("--out_meta", default="data/final/dev_intrain_meta.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sp = cfg.split
    prompts, truth, meta = {}, {}, {}
    k = 0
    for src, n_keep in QUOTA.items():
        cands = []
        shard_dir = os.path.join(cfg.paths.final, "train", src)
        for rec in iter_shards(shard_dir):
            if rec.get("meta", {}).get("anonymized", False):
                continue
            ex = cut_dev_example(rec["text"], rec["id"], sp.prompt_words, sp.cont_words,
                                 sp.dev_min_offset_words, sp.seed)
            if ex is None:
                continue
            cands.append((stable_unit(rec["id"], "intrain"), rec["id"], ex[0], ex[1]))
        if len(cands) < n_keep:
            raise SystemExit(f"{src}: only {len(cands)} usable training docs, wanted {n_keep}")
        cands.sort()
        for _, doc_id, prompt, cont in cands[:n_keep]:
            key = f"in_{k:05d}"
            k += 1
            prompts[key] = {"text": prompt}
            truth[key] = {"text": cont}
            meta[key] = {"source": src, "doc_id": doc_id}
        print(f"{src}: kept {n_keep} of {len(cands)}", flush=True)
    write_json(prompts, a.out_prompts)
    write_json(truth, a.out_truth)
    write_json(meta, a.out_meta)
    print(f"wrote {len(prompts)} prompts", flush=True)


if __name__ == "__main__":
    main()
