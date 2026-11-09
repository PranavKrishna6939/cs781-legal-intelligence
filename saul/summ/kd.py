"""Sequence-level distillation into conv_mini.

The teacher is an already-trained run (E1). This writes greedy summaries for the 6,830
training documents, then trains conv_mini with equal gold and teacher cross-entropy.
There is no online logit KL: a second 300M forward does not fit this window.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

from saul.summ.data import holdout_ids, load_rows, p6_dir
from saul.summ.decode import generate, lengths_for, predicted_words
from saul.summ.train import load_run


def dump_teacher(run: str, dest: str, slots: int, lam: float) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_run(run, device)
    held = holdout_ids()
    rows = [row for row in load_rows("train") if row["id"] not in held]
    ridge = None
    if meta.get("ridge") is not None:
        import numpy as np
        ridge = np.array(meta["ridge"], dtype=np.float64)
    words = predicted_words(
        model, tok, rows, meta["len_ids"], meta["lmax"], meta["strip"], device, meta.get("length_method", "ridge"), ridge,
    )
    targets, forced = lengths_for(words, lam, meta["len_ids"])
    texts = generate(
        model, tok, rows, targets, forced, meta["lmax"], meta["strip"], device, n=1, slots=slots,
    )
    payload = [{"id": row["id"], "summary": text} for row, text in zip(rows, texts)]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"[p6] teacher summaries {len(payload)} -> {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--name", default="M3")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pack-tokens", type=int, default=8192)
    ap.add_argument("--step-tokens", type=int, default=32768)
    args = ap.parse_args()
    dest = os.path.join(p6_dir(), "exp", args.name, "teacher.json")
    if not os.path.exists(dest):
        dump_teacher(args.teacher_run, dest, args.slots, args.lam)
    cmd = [
        sys.executable, "-m", "saul.summ.train",
        "--model", "conv_mini", "--technique", "full", "--name", args.name,
        "--epochs", str(args.epochs), "--lmax", str(args.lmax), "--teacher", dest,
        "--pack-tokens", str(args.pack_tokens), "--step-tokens", str(args.step_tokens),
        "--lr", "5e-5",
    ]
    print("[p6] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
