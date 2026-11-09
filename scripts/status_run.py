#!/usr/bin/env python3
"""One-line status of a pretraining stage directory: latest step, loss, tokens/s, ETA, last evals.

    python3 scripts/status_run.py runs/dense_286m/stable 7200
"""
import json
import os
import sys

run_dir, target = sys.argv[1], int(sys.argv[2])
path = os.path.join(run_dir, "metrics.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
train = [r for r in rows if "loss" in r]
evals = [r for r in rows if "val_loss" in r]
if not train:
    print(f"{run_dir}: no metrics yet")
    sys.exit(0)
t = train[-1]
tail = train[-10:]
mean = sum(r["loss"] for r in tail) / len(tail)
eta = (target - t["step"]) * 524288 / max(t["tokens_per_s"], 1) / 3600
print(f"{run_dir}: step {t['step']}/{target}  loss(last10) {mean:.4f}  tok/s {t['tokens_per_s']:,.0f}  "
      f"gnorm {t['gnorm']:.3f}  eta {eta:.1f} h")
for e in evals[-2:]:
    print(f"  eval step {e['step']}: val_loss {e['val_loss']:.4f}  bits/byte {e.get('val_bpb', float('nan')):.4f}")
