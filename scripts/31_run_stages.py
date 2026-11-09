#!/usr/bin/env python
"""Stage 31: stable pretraining, a decay branch, then a context-extension branch.

  A  stable mixture, stops at the first decay step (ckpt_stable); no compute is spent on a decay that B redoes.
  B  resumes ckpt_stable (optimizer, step, data stream) and runs the WSD decay on the anneal mixture.
  C  starts from B's final weights with a fresh short schedule at 8K context and RoPE base 5e5.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(config: str, run_dir: str, extra: list[str]) -> None:
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "30_train.py"),
           "--config", config, f"run_dir={run_dir}", *extra]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--run_dir", default="runs/dense_286m")
    ap.add_argument("--extend_steps", type=int, default=600, help="C: 600 x 524K tokens = 315M tokens at 8K")
    ap.add_argument("--skip_extend", action="store_true")
    a = ap.parse_args()
    stable_dir = os.path.join(a.run_dir, "stable")
    anneal_dir = os.path.join(a.run_dir, "anneal")
    if not os.path.exists(os.path.join(stable_dir, "ckpt_stable")):
        run(a.config, stable_dir, ["train.mixture=stable", "--stop_at_stable"])
    if not os.path.exists(os.path.join(anneal_dir, "ckpt_final")):
        run(a.config, anneal_dir, ["train.mixture=anneal", "--resume", os.path.join(stable_dir, "ckpt_stable")])
    if a.skip_extend:
        return
    run(a.config, os.path.join(a.run_dir, "extend"), [
        "train.mixture=anneal", "train.seq_len=8192", "train.batch_size=2", "train.grad_accum=32",
        f"train.total_steps={a.extend_steps}", "train.long_doc_prob=0.5", "train.long_doc_min=4096",
        "train.mntp=false", "train.eval_every=200", "train.ckpt_every=200",
        "optim.muon_lr=0.0005", "optim.adamw_lr=0.0005", "optim.embed_lr=0.0005",
        "optim.warmup_frac=0.05", "optim.decay_frac=0.3",
        "model.rope_base=500000", "--init_from", os.path.join(anneal_dir, "ckpt_final"),
    ])


if __name__ == "__main__":
    main()
