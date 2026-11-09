"""Distil the ensemble's per-label probabilities into Conv-Mini."""
from __future__ import annotations

import json
import os

import numpy as np

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.score import rare_macro, sigmoid
from saul.utils.io import ensure_dir, write_json


def dump_teacher(members: list[str], temperature: float = 2.0,
                 temperatures: dict | None = None) -> str:
    """Mean of D0-calibrated member train logits, softened again at `temperature`."""
    tables = []
    order = None
    used = []
    loaded = []
    for name in members:
        path = os.path.join(exp_dir(name), "train_logits.npz")
        if not os.path.exists(path):
            print(f"[kd] skip {name}, no train logits", flush=True)
            continue
        blob = np.load(path, allow_pickle=True)
        ids = [str(i) for i in blob["ids"]]
        loaded.append((name, ids, np.asarray(blob["logits"], dtype=np.float64)))
        if order is None or len(ids) > len(order):
            order = ids
    if order is None:
        raise SystemExit("[kd] no member wrote train logits")
    for name, ids, matrix in loaded:
        index = {case_id: i for i, case_id in enumerate(ids)}
        if any(case_id not in index for case_id in order):
            print(f"[kd] skip {name}, train logits do not cover the teacher order", flush=True)
            continue
        member_t = 1.0
        if temperatures and name in temperatures:
            member_t = float(temperatures[name])
        logits = matrix[[index[i] for i in order]] / member_t
        tables.append(logits)
        used.append(name)
    if not tables:
        raise SystemExit("[kd] no member wrote train logits")
    members = used
    mean = np.mean(tables, axis=0) / temperature
    probs = sigmoid(mean)
    directory = os.path.join(p5_dir(), "kd")
    ensure_dir(directory)
    path = os.path.join(directory, "teacher.npz")
    np.savez(path, ids=np.asarray(order), probs=probs.astype(np.float32))
    write_json({"members": members, "temperature": temperature, "n": len(order)}, os.path.join(directory, "teacher.json"))
    print(f"[kd] teacher {path} n={len(order)}", flush=True)
    return path


def student_args(epochs: int, teacher: str | None, exp: str, loss: str) -> list[str]:
    args = [
        "--exp", exp, "--model", "conv_mini", "--technique", "full", "--context", "echo",
        "--head", "laat", "--loss", loss, "--epochs", str(epochs), "--limit", "2048",
        "--attn", "sdpa", "--no-wandb",
    ]
    if teacher:
        args += ["--teacher", teacher]
    return args


def rare_report(exp: str, freq_path: str | None = None) -> dict:
    from saul.lsi.decide import load_logits
    blob = load_logits(exp, "dev")
    if "labels" not in blob:
        return {}
    freq = np.ones(N_LABELS)
    path = freq_path or os.path.join(p5_dir(), "label_stats.json")
    if os.path.exists(path):
        freq = np.asarray(json.load(open(path, encoding="utf-8"))["freq"])
    pred = sigmoid(blob["logits"]) >= 0.5
    return {"rare20": rare_macro(blob["labels"], pred, freq, 20)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", default="")
    parser.add_argument("--from-mix", default="")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()
    members = [name for name in args.members.split(",") if name]
    temperatures = None
    if args.from_mix:
        path = os.path.join(exp_dir(args.from_mix), "mix.json")
        if not os.path.exists(path):
            raise SystemExit(f"[kd] no mix report at {path}")
        info = json.load(open(path, encoding="utf-8"))
        members = list(info.get("members") or members)
        temperatures = info.get("temperature") or None
    if members:
        dump_teacher(members, args.temperature, temperatures)


if __name__ == "__main__":
    main()
