#!/bin/bash
# Steady-state tokens/s for one micro-batch size. Same 524288 tokens/step. Does not save.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
BS=$1
ACC=$2
.venv/bin/python -u scripts/30_train.py --config configs/train/hybrid.yaml \
  run_dir=runs/bench_b${BS} \
  train.mixture=stable train.batch_size=$BS train.grad_accum=$ACC \
  train.log_every=8 train.eval_every=0 train.ckpt_every=0 \
  wandb.enabled=false \
  --resume runs/hybrid_294m/stable/ckpt_stable \
  --bench 32
