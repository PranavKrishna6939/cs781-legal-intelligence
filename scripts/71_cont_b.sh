#!/bin/bash
# Hybrid continuation B: same resume as A, but a Supreme Court-heavier stable mixture.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
.venv/bin/python - << 'PY'
import torch
from saul.data.stats import mixture_weights
from saul.utils.config import load_config
from saul.utils.io import read_json
st = torch.load("runs/hybrid_294m/stable/ckpt_stable/trainer.pt", map_location="cpu", weights_only=False)
print("resume_step", st["step"], flush=True)
if int(st["step"]) != 7200:
    raise SystemExit(f"expected step 7200, got {st['step']}")
dcfg = load_config("configs/data_stable_sc_heavy.yaml")
shares = read_json("data/tokens/legal32k/token_shares.json")["train"]
counts = {s: shares[s]["tokens"] for s in shares}
w = mixture_weights(counts, dict(dcfg.mixture.stable))
print("stable_weights", {k: round(v, 4) for k, v in w.items()}, flush=True)
PY
exec .venv/bin/python -u scripts/30_train.py --config configs/train/hybrid.yaml \
  run_dir=runs/hybrid_294m/stable_cont_b \
  train.mixture=stable data_config=configs/data_stable_sc_heavy.yaml \
  train.total_steps=21600 train.ckpt_every=1000 \
  optim.decay_frac=0.111 \
  --resume runs/hybrid_294m/stable/ckpt_stable \
  --stop_at_stable
