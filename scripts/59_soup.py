#!/usr/bin/env python
"""Average safetensors checkpoints that share one architecture (a weight soup).

    python scripts/59_soup.py --out runs/hybrid_294m/soup_late \
        runs/hybrid_294m/anneal/ckpt_8500 \
        runs/hybrid_294m/anneal/ckpt_9000 \
        runs/hybrid_294m/anneal/ckpt_final_ema

Configs must match on every architecture field. Extend checkpoints (RoPE base 5e5) must not be
mixed with anneal checkpoints (RoPE base from the original model config).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from saul.utils.io import write_json  # noqa: E402

def _config(path: str) -> dict:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("checkpoints", nargs="+")
    a = ap.parse_args()
    if len(a.checkpoints) < 2:
        raise SystemExit("a soup needs at least two checkpoints")
    cfgs = [_config(p) for p in a.checkpoints]
    for p, cfg in zip(a.checkpoints, cfgs):
        if cfg != cfgs[0]:
            raise SystemExit(f"config.json mismatch: {a.checkpoints[0]} vs {p}")
    states = [load_file(os.path.join(p, "model.safetensors")) for p in a.checkpoints]
    keys = list(states[0])
    for p, st in zip(a.checkpoints, states):
        if list(st) != keys and set(st) != set(keys):
            raise SystemExit(f"tensor key mismatch: {p}")
        for k in keys:
            if st[k].shape != states[0][k].shape:
                raise SystemExit(f"shape mismatch on {k} in {p}")
    n = float(len(states))
    averaged = {}
    for k in keys:
        acc = None
        for st in states:
            t = st[k].float()
            acc = t if acc is None else acc + t
        averaged[k] = (acc / n).to(dtype=states[0][k].dtype)
    os.makedirs(a.out, exist_ok=True)
    save_file(averaged, os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfgs[0], f, indent=2)
        f.write("\n")
    write_json({"ingredients": [os.path.abspath(p) for p in a.checkpoints], "uniform": True},
               os.path.join(a.out, "soup.json"))
    n_params = sum(t.numel() for t in averaged.values())
    print(f"wrote {a.out} ({n_params:,} params, {len(a.checkpoints)} ingredients)", flush=True)
    del averaged
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
