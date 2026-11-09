#!/usr/bin/env python
"""Stage 41: score checkpoints on fixed held-out sets (loss and bits/byte), independent of any training mixture.

    python scripts/41_eval_val.py --checkpoints runs/dense_286m/stable/ckpt_stable runs/dense_286m/anneal/ckpt_* \
        --out runs/dense_286m/val_eval.json

Sets: `stable` (the stable pretraining mixture) and one per source (e.g. `supreme_court`, the test-like source).
Every checkpoint sees exactly the same sequences.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from saul.data.stats import mixture_weights  # noqa: E402
from saul.data.tokens import open_split  # noqa: E402
from saul.eval.val import evaluate_lm, fixed_val_batches  # noqa: E402
from saul.model.modeling_saul import SaulForCausalLM  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--tokens", default="data/tokens/legal32k")
    ap.add_argument("--data_config", default="configs/data.yaml")
    ap.add_argument("--sequences", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sources", nargs="*", default=["supreme_court"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dcfg = load_config(a.data_config)
    val = open_split(a.tokens, "val", list(dcfg.sources))
    counts = {s: v["tokens"] for s, v in read_json(os.path.join(a.tokens, "token_shares.json"))["train"].items()}
    sets = {"stable": mixture_weights(counts, dict(dcfg.mixture.stable))}
    sets.update({s: {s: 1.0} for s in a.sources})
    batches = {name: fixed_val_batches({s: val[s] for s in w if s in val}, w, a.seq_len, a.batch, a.sequences, 0, 1, 2)
               for name, w in sets.items()}
    token_bytes = np.load(os.path.join(a.tokens, "token_bytes.npy"))
    ckpts = sorted({p for pat in a.checkpoints for p in glob.glob(pat) if os.path.isdir(p)})
    rows = []
    for ck in ckpts:
        model = SaulForCausalLM.from_pretrained(ck).cuda()
        row = {"checkpoint": ck}
        for name, bs in batches.items():
            r = evaluate_lm(model, bs, token_bytes, "cuda", bf16=True)
            row[f"{name}_loss"], row[f"{name}_bpb"] = r["val_loss"], r.get("val_bpb")
        rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(rows, a.out)


if __name__ == "__main__":
    main()
