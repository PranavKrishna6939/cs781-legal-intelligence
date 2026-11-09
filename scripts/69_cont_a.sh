#!/bin/bash
# Hybrid continuation A: resume ckpt_stable at peak LR, same stable mix, stop at step 19202.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
.venv/bin/python - << 'PY'
import torch
st = torch.load("runs/hybrid_294m/stable/ckpt_stable/trainer.pt", map_location="cpu", weights_only=False)
print("resume_step", st["step"], flush=True)
if int(st["step"]) != 7200:
    raise SystemExit(f"expected step 7200, got {st['step']}")
PY
exec .venv/bin/python -u scripts/30_train.py --config configs/train/hybrid.yaml \
  run_dir=runs/hybrid_294m/stable_cont_a \
  train.mixture=stable train.total_steps=21600 train.ckpt_every=1000 \
  optim.decay_frac=0.111 \
  --resume runs/hybrid_294m/stable/ckpt_stable \
  --stop_at_stable
