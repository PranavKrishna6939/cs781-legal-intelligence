"""Distil the ensemble into Conv-Mini, and train the gold-label control."""
from __future__ import annotations

import json
import os

import numpy as np

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.ensemble import _align, _done, mix_logits
from saul.cjpe.data import load_split
from saul.utils.io import ensure_dir


def dump_teacher(members: list[str], temperature: float = 2.0) -> str:
    dest = os.path.join(p4_dir(), "kd")
    ensure_dir(dest)
    path = os.path.join(dest, "teacher.npz")
    pool = load_split("pool")
    present = [exp for exp in members if os.path.exists(os.path.join(exp_dir(exp), "logits_pool.npz"))]
    if not present:
        raise RuntimeError("no pool logits for the teacher")
    rows = []
    for exp in present:
        plain = _done(exp).get("loss") == "ce"
        rows.append(_align(exp, "pool", pool.ids, plain))
    mixed = mix_logits(rows) / temperature
    mixed -= mixed.max(axis=1, keepdims=True)
    probs = np.exp(mixed)
    probs /= probs.sum(axis=1, keepdims=True)
    np.savez(path, ids=np.asarray(pool.ids), probs=probs.astype(np.float32), members=np.asarray(present))
    with open(os.path.join(dest, "teacher_done.json"), "w", encoding="utf-8") as handle:
        json.dump({"members": present, "temperature": temperature}, handle)
    print(f"[kd] teacher {path} members={present}", flush=True)
    return path


def _args(argv: list[str]):
    from saul.cjpe.train import build_parser
    args = build_parser().parse_args(argv)
    args.gate = None
    return args


def train_student(epochs: int = 2, teacher: str | None = None) -> None:
    from saul.cjpe.train import train_one
    teacher = teacher or os.path.join(p4_dir(), "kd", "teacher.npz")
    train_one(_args([
        "--exp", "A9", "--model", "conv_mini", "--technique", "full", "--context", "causal",
        "--limit", "4096", "--epochs", str(epochs), "--loss", "kd", "--teacher", teacher,
        "--batch-tokens", "16384",
    ]))


def train_gold(epochs: int = 1) -> None:
    from saul.cjpe.train import train_one
    train_one(_args([
        "--exp", "A9gold", "--model", "conv_mini", "--technique", "full", "--context", "causal",
        "--limit", "4096", "--epochs", str(epochs), "--loss", "logit_adjust",
        "--batch-tokens", "16384", "--no-pool-logits",
    ]))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--student", action="store_true")
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--members", default="")
    args = parser.parse_args()
    if args.teacher:
        dump_teacher([item for item in args.members.split(",") if item])
    if args.student:
        train_student()
    if args.gold:
        train_gold()


if __name__ == "__main__":
    main()
