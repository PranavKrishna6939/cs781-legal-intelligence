"""Paired bootstrap over per-example scores. Adopt a change only if the 95% interval excludes 0."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, n: int = 10_000, seed: int = 781) -> dict:
    rng = np.random.RandomState(seed)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("score vectors must be aligned on the same examples")
    diff = a - b
    idx = rng.randint(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(len(diff)),
    }
