#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
.venv/bin/python - <<'PY'
import importlib.util
import os
import shutil
import torch
from saul.utils.config import load_config
from saul.train.schedule import in_decay, wsd_multiplier
cfg = load_config('configs/train/recovery_b_anneal.yaml')
st = torch.load('runs/hybrid_294m/stable_cont_b/ckpt_stable/trainer.pt', map_location='cpu', weights_only=False)
step = int(st['step'])
total, frac = int(cfg.train.total_steps), float(cfg.optim.decay_frac)
assert step == 19202 and total-step == 1800
assert int(total*(1-frac)) == step
assert in_decay(step, total, frac) and not in_decay(step-1, total, frac)
assert st['ema']['updates'] == 0 and st['ema']['shadow'] is None
assert st['loader']['num_workers'] == cfg.train.num_workers
assert all(g['_base_lr'] == (0.003 if g.get('name') == 'embed' else 0.002) for g in st['adamw']['param_groups'])
assert shutil.disk_usage('/data/cs781-archive').free > 8 * 2**30, 'Need 8 GiB free for raw final+trainer+EMA'
assert not os.path.exists(cfg.run_dir), 'Run directory already exists'
spec = importlib.util.spec_from_file_location('train_main', 'scripts/30_train.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print('weights', mod._weights(cfg), flush=True)
print('full_resume', step, 'end', total, 'decay_start', int(total*(1-frac)),
      'first_lr_mult', wsd_multiplier(step,total,float(cfg.optim.warmup_frac),frac),
      'last_lr_mult', wsd_multiplier(total-1,total,float(cfg.optim.warmup_frac),frac),
      'tokens', 1800*524288, flush=True)
PY
if [[ "${1:-}" == "--preflight-only" ]]; then
    exit 0
fi
exec .venv/bin/python -u scripts/30_train.py --config configs/train/recovery_b_anneal.yaml \
    --resume runs/hybrid_294m/stable_cont_b/ckpt_stable
