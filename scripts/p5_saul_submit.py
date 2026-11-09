"""Build the Phase 5 zips from finished SAUL-LSI runs.

S1-S4 are the single models. S0 is a cross-fitted weighted average of their fused logits,
with per-label thresholds shrunk at kappa 50 and at least one label per case.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.data import dev_half_array
from saul.lsi.decide import at_least_one, quick_macro, shrink_thresholds, optimal_label_thresholds, best_global_threshold
from saul.lsi.score import macro_f1, sigmoid
from saul.lsi.submit import rows_from_matrix, test_ids, write_zip

MEMBERS = (
    ("E2_ced", "P5_S1_ced"),
    ("E1_dense_lorma", "P5_S2_dense"),
    ("E3_hybrid_adapter", "P5_S3_hybrid"),
    ("E4_conv_kd", "P5_S4_kd_convmini"),
)


def _load(exp: str, kind: str) -> tuple[list[str], np.ndarray]:
    blob = np.load(os.path.join(exp_dir(exp), f"{kind}_logits.npz"))
    return [str(i) for i in blob["ids"]], blob["logits"].astype(np.float64)


def _align(ids: list[str], logits: np.ndarray, order: list[str]) -> np.ndarray:
    index = {case_id: i for i, case_id in enumerate(ids)}
    missing = [case_id for case_id in order if case_id not in index]
    if missing:
        raise SystemExit(f"missing {len(missing)} ids, first {missing[0]}")
    return logits[[index[case_id] for case_id in order]]


def _zscore(dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = dev.mean(axis=0, keepdims=True)
    std = dev.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (dev - mean) / std, (test - mean) / std


def _thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return shrink_thresholds(
        optimal_label_thresholds(probs, labels),
        best_global_threshold(probs, labels),
        labels.sum(0),
        50,
    )


def _predict(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return at_least_one(probs, thresholds).astype(np.int64)


def _fit_weights(members: list[np.ndarray], labels: np.ndarray, half: np.ndarray) -> np.ndarray:
    n = len(members)
    best_w, best_f = np.full(n, 1.0 / n), -1.0
    grid = range(11)
    for raw in np.ndindex(*([11] * n)):
        if sum(raw) != 10:
            continue
        weight = np.asarray(raw, dtype=np.float64) / 10.0
        mixed = sum(w * m for w, m in zip(weight, members))
        score = quick_macro(mixed, labels, half)["macro_f1"]
        if score > best_f:
            best_f, best_w = score, weight
    _ = grid
    return best_w


def _one(name: str, dev: np.ndarray, test: np.ndarray, labels: np.ndarray, half, dev_ids, test_order) -> None:
    scored = quick_macro(dev, labels, half)
    print(f"[submit] {name} dev {scored['macro_f1']:.4f} half {scored['half']}", flush=True)
    probs = sigmoid(dev)
    # Thresholds are the mean of the two cross-fit vectors, matching what test is allowed to see.
    folds = []
    for fit_name in ("A", "B"):
        rows = half == fit_name
        folds.append(_thresholds(probs[rows], labels[rows]))
    thresholds = np.mean(folds, axis=0)
    pred = _predict(sigmoid(test), thresholds)
    directory = os.path.join(p5_dir(), "..", "submissions", "phase5", name)
    # p5_dir is ~/cs781/p5, submissions live next to it.
    directory = os.path.join(os.path.dirname(p5_dir()), "submissions", "phase5", name)
    write_zip(rows_from_matrix(test_order, pred), directory, expected=test_order, require_label=True)


def main() -> None:
    dev_meta = json.load(open(os.path.join(p5_dir(), "tok", "dev.meta.json"), encoding="utf-8"))
    dev_ids = [row["id"] for row in dev_meta]
    labels = np.zeros((len(dev_ids), N_LABELS), dtype=np.float64)
    for index, row in enumerate(dev_meta):
        for lab in row["labels"]:
            labels[index, int(lab)] = 1.0
    half = dev_half_array(dev_ids)
    test_order = test_ids()
    ready = []
    dev_z, test_z = [], []
    for exp, zip_name in MEMBERS:
        if not os.path.exists(os.path.join(exp_dir(exp), "dev_logits.npz")):
            print(f"[submit] skip {exp}, no dev logits", flush=True)
            continue
        d_ids, d_logits = _load(exp, "dev")
        t_ids, t_logits = _load(exp, "test")
        dev = _align(d_ids, d_logits, dev_ids)
        test = _align(t_ids, t_logits, test_order)
        _one(zip_name, dev, test, labels, half, dev_ids, test_order)
        zd, zt = _zscore(dev, test)
        dev_z.append(zd)
        test_z.append(zt)
        ready.append(exp)
    if len(dev_z) < 2:
        return
    weight = _fit_weights(dev_z, labels, half)
    print("[submit] weights", dict(zip(ready, [round(float(w), 2) for w in weight])), flush=True)
    mixed_dev = sum(w * m for w, m in zip(weight, dev_z))
    mixed_test = sum(w * m for w, m in zip(weight, test_z))
    _one("P5_S0_avg", mixed_dev, mixed_test, labels, half, dev_ids, test_order)


if __name__ == "__main__":
    main()
