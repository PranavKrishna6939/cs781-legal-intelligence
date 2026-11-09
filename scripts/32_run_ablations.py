#!/usr/bin/env python
"""Stage 32: run the proxy ablations one after another and tabulate them.

    python scripts/32_run_ablations.py [--configs configs/ablations/a03_gr.yaml ...]

* Runs every `configs/ablations/*.yaml` in name order; each run lives in runs/ablations/<name>.
* A run with `ckpt_final` is skipped, so the driver can be restarted after a crash or reboot.
* A failed run is recorded and the queue continues.
* A run whose token directory is not ready yet (the vocab variants) moves to the end of the queue and waits.
* artifacts/ablations/summary.md and summary.json are rewritten after every run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.utils.config import load_config  # noqa: E402


def read_metrics(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def summarize(name: str, run_dir: str, status: str, seconds: float | None) -> dict:
    rows = read_metrics(run_dir)
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "val_loss" in r]
    last_eval = evals[-1] if evals else {}
    tail = train[-20:]
    params = None
    log = os.path.join(run_dir, "train.log")
    if os.path.exists(log):
        m = re.search(r"params=([\d,]+)", open(log, errors="replace").read())
        params = int(m.group(1).replace(",", "")) if m else None
    return {
        "name": name, "status": status, "params": params,
        "steps": train[-1]["step"] if train else 0,
        "train_loss_tail": sum(r["loss"] for r in tail) / len(tail) if tail else None,
        "val_loss": last_eval.get("val_loss"), "val_bpb": last_eval.get("val_bpb"),
        "ema_val_bpb": last_eval.get("ema_val_bpb"),
        "tokens_per_s": sum(r["tokens_per_s"] for r in tail) / len(tail) if tail else None,
        "gpu_gb": max((r.get("gpu_gb", 0) for r in train), default=None),
        "hours": None if seconds is None else seconds / 3600,
    }


def write_summary(out: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    lines = ["| ablation | status | params | steps | train loss (last 20 logs) | val loss | val bits/byte | "
             "EMA val bits/byte | tokens/s | peak GB | hours |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['status']} | {f(r['params'], ',')} | {r['steps']} | "
                     f"{f(r['train_loss_tail'], '.4f')} | {f(r['val_loss'], '.4f')} | {f(r['val_bpb'], '.4f')} | "
                     f"{f(r['ema_val_bpb'], '.4f')} | {f(r['tokens_per_s'], ',.0f')} | {f(r['gpu_gb'], '.1f')} | "
                     f"{f(r['hours'], '.2f')} |")
    open(out, "w").write("\n".join(lines) + "\n")
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(rows, fh, indent=2)


def tokens_ready(cfg_path: str) -> bool:
    cfg = load_config(cfg_path)
    return os.path.exists(os.path.join(cfg.tokens, "token_shares.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--out", default="artifacts/ablations/summary.md")
    ap.add_argument("--wait_hours", type=float, default=12.0, help="max wait for a deferred run's token files")
    a = ap.parse_args()
    queue = a.configs or sorted(glob.glob("configs/ablations/*.yaml"))
    results: dict[str, dict] = {}
    deferred_since: dict[str, float] = {}
    while queue:
        cfg = queue.pop(0)
        name = os.path.splitext(os.path.basename(cfg))[0]
        run_dir = os.path.join("runs", "ablations", name)
        if os.path.exists(os.path.join(run_dir, "ckpt_final")):
            results[name] = summarize(name, run_dir, "done (cached)", None)
            write_summary(a.out, list(results.values()))
            continue
        if not tokens_ready(cfg):
            first = deferred_since.setdefault(name, time.time())
            if time.time() - first > a.wait_hours * 3600:
                results[name] = summarize(name, run_dir, "skipped: tokens never ready", None)
                write_summary(a.out, list(results.values()))
                continue
            queue.append(cfg)
            if all(not tokens_ready(c) for c in queue):
                print(f"[wait] token files not ready for {[os.path.basename(c) for c in queue]}", flush=True)
                time.sleep(300)
            continue
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, "scripts/30_train.py", "--config", cfg, f"run_dir={run_dir}"]
        print(f"[run] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(os.path.join(run_dir, "train.log"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "done" if rc == 0 and os.path.exists(os.path.join(run_dir, "ckpt_final")) else f"failed (rc={rc})"
        results[name] = summarize(name, run_dir, status, time.time() - t0)
        print(f"[{status}] {name}: {json.dumps(results[name])}", flush=True)
        write_summary(a.out, list(results.values()))
    print(open(a.out).read())


if __name__ == "__main__":
    main()
