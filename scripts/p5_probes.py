"""Five CPU probe zips. They do not touch the GPU.

1. zlex, per-label thresholds (dev ~0.441)
2. zlex, one global threshold (checks whether per-label thresholds overfit dev)
3. kNN label vote
4. z-scored blend of zlex and kNN
5. that blend, with exact-duplicate texts overwritten by the copied gold labels
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.data import dev_half_array, text_hash
from saul.lsi.decide import (
    at_least_one, best_global_threshold, optimal_label_thresholds, quick_macro, shrink_thresholds,
)
from saul.lsi.score import macro_f1, sigmoid
from saul.lsi.submit import rows_from_matrix, test_ids, write_zip
from scripts.p5_tfidf import _labels, _read

TAU = 0.05
K = 30


def _dev_labels() -> tuple[list[str], np.ndarray, np.ndarray]:
    import json
    meta = json.load(open(os.path.join(p5_dir(), "tok", "dev.meta.json"), encoding="utf-8"))
    ids = [row["id"] for row in meta]
    labels = np.zeros((len(ids), N_LABELS), dtype=np.float64)
    for index, row in enumerate(meta):
        for lab in row["labels"]:
            labels[index, int(lab)] = 1.0
    return ids, labels, dev_half_array(ids)


def _vote(idx: np.ndarray, sim: np.ndarray, pool: np.ndarray) -> np.ndarray:
    chosen = idx[:, :K]
    scores = sim[:, :K]
    weights = np.exp((scores - scores.max(axis=1, keepdims=True)) / TAU)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkc->nc", weights, pool[chosen])


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _zscore(dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = dev.mean(axis=0, keepdims=True)
    std = dev.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (dev - mean) / std, (test - mean) / std


def _label_thresholds(logits: np.ndarray, labels: np.ndarray, half: np.ndarray) -> np.ndarray:
    probs = sigmoid(logits)
    folds = []
    for name in ("A", "B"):
        rows = half == name
        folds.append(shrink_thresholds(
            optimal_label_thresholds(probs[rows], labels[rows]),
            best_global_threshold(probs[rows], labels[rows]),
            labels[rows].sum(0),
            50,
        ))
    return np.mean(folds, axis=0)


def _global_threshold(logits: np.ndarray, labels: np.ndarray, half: np.ndarray) -> float:
    probs = sigmoid(logits)
    values = [best_global_threshold(probs[half == name], labels[half == name]) for name in ("A", "B")]
    return float(np.mean(values))


def _apply(logits: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return at_least_one(sigmoid(logits), np.broadcast_to(thresholds, logits.shape)).astype(np.int64)


def _dup_map(query: list[dict], pool: list[dict], pool_y: np.ndarray | None = None) -> list[np.ndarray | None]:
    if pool_y is None:
        pool_y = _labels(pool)
    table: dict[str, list[int]] = {}
    for index, row in enumerate(pool):
        table.setdefault(text_hash(row["text"]), []).append(index)
    out = []
    for row in query:
        hits = table.get(text_hash(row["text"]))
        if not hits:
            out.append(None)
            continue
        mean = pool_y[hits].mean(axis=0)
        out.append((mean >= 0.5).astype(np.int64))
    return out


def _override(pred: np.ndarray, copies: list[np.ndarray | None]) -> np.ndarray:
    out = pred.copy()
    for index, gold in enumerate(copies):
        if gold is not None and int(gold.sum()) > 0:
            out[index] = gold
    return out


def _write(name: str, pred: np.ndarray, order: list[str]) -> None:
    directory = os.path.join(os.path.dirname(p5_dir()), "submissions", "phase5", name)
    write_zip(rows_from_matrix(order, pred), directory, expected=order, require_label=True)
    filled = pred.sum(axis=1)
    print(f"[probe] {name} labels/case mean {filled.mean():.2f} empty {(filled == 0).sum()}", flush=True)


def main() -> None:
    feats = os.path.join(p5_dir(), "feats")
    dev_ids, labels, half = _dev_labels()
    test_order = test_ids()
    zlex_dev = np.load(os.path.join(feats, "zlex_dev.npy"))
    zlex_test = np.load(os.path.join(feats, "zlex_test.npy"))
    pool = np.load(os.path.join(feats, "knn_labels.npy"))
    dev_knn = np.load(os.path.join(feats, "knn_dev.npz"))
    test_knn = np.load(os.path.join(feats, "knn_test.npz"))
    knn_dev = _logit(_vote(dev_knn["idx"], dev_knn["sim"], pool))
    knn_test = _logit(_vote(test_knn["idx"], test_knn["sim"], pool))

    zlex_thr = _label_thresholds(zlex_dev, labels, half)
    zlex_pred = _apply(zlex_test, zlex_thr)
    print("[probe] zlex", quick_macro(zlex_dev, labels, half)["macro_f1"], flush=True)
    _write("P5_probe_zlex", zlex_pred, test_order)

    global_t = _global_threshold(zlex_dev, labels, half)
    parts = []
    probs = sigmoid(zlex_dev)
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        threshold = best_global_threshold(probs[half == fit_name], labels[half == fit_name])
        pred = probs[half == score_name] >= threshold
        parts.append(macro_f1(labels[half == score_name], pred))
    print(f"[probe] zlex global t={global_t:.3f} dev {float(np.mean(parts)):.4f}", flush=True)
    global_test = _apply(zlex_test, np.full(N_LABELS, global_t))
    _write("P5_probe_zlex_global", global_test, test_order)

    knn_thr = _label_thresholds(knn_dev, labels, half)
    print("[probe] knn", quick_macro(knn_dev, labels, half)["macro_f1"], flush=True)
    _write("P5_probe_knn", _apply(knn_test, knn_thr), test_order)

    z_dev, z_test = _zscore(zlex_dev, zlex_test)
    k_dev, k_test = _zscore(knn_dev, knn_test)
    best_l, best_f = 0.0, -1.0
    for lam in np.linspace(0, 1, 11):
        score = quick_macro((1 - lam) * z_dev + lam * k_dev, labels, half)["macro_f1"]
        if score > best_f:
            best_l, best_f = float(lam), score
    mixed_dev = (1 - best_l) * z_dev + best_l * k_dev
    mixed_test = (1 - best_l) * z_test + best_l * k_test
    mix_thr = _label_thresholds(mixed_dev, labels, half)
    mix_pred = _apply(mixed_test, mix_thr)
    print(f"[probe] blend knn weight {best_l:.1f} dev {best_f:.4f}", flush=True)
    _write("P5_probe_blend", mix_pred, test_order)

    train, dev, test = _read("train"), _read("dev"), _read("test")
    train_y, dev_y = _labels(train), _labels(dev)
    dev_copies = _dup_map(dev, train, train_y)
    # Dev half A may also copy the other half. Rebuild per half so the dev number is not leaked.
    by_half = np.array([row_id for row_id in half])
    dev_pred = _apply(mixed_dev, mix_thr)
    overridden = dev_pred.copy()
    for name in ("A", "B"):
        other_rows = np.flatnonzero(by_half != name)
        copies = _dup_map(
            [dev[i] for i in np.flatnonzero(by_half == name)],
            train + [dev[i] for i in other_rows],
            np.vstack([train_y, dev_y[other_rows]]),
        )
        rows = np.flatnonzero(by_half == name)
        for local, gold in enumerate(copies):
            if gold is not None:
                overridden[rows[local]] = gold
    print(f"[probe] blend+dup dev {macro_f1(labels, overridden):.4f} "
          f"hits {sum(gold is not None for gold in dev_copies)}", flush=True)
    test_copies = _dup_map(test, train + dev, np.vstack([train_y, dev_y]))
    print(f"[probe] test duplicate hits {sum(gold is not None for gold in test_copies)}", flush=True)
    _write("P5_probe_blend_dup", _override(mix_pred, test_copies), test_order)


if __name__ == "__main__":
    main()
