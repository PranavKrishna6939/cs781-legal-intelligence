"""Macro scores, paired bootstrap, and the adoption rule from the plan."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from saul.cjpe.const import ADOPT_GAIN


def macro_scores(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    kw = dict(average="macro", zero_division=0, labels=[0, 1])
    return {
        "f1": float(f1_score(y_true, y_pred, **kw)),
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
    }


def predict_logits(logits, threshold: float = 0.0) -> np.ndarray:
    """Predict 1 when label-1 log-odds exceed `threshold`. Ties go to 0."""
    logits = np.asarray(logits, dtype=np.float64)
    return (logits[:, 1] - logits[:, 0] > threshold).astype(np.int64)


def half_gains(y, pred_new, pred_base, half) -> tuple[float, float]:
    """Gain of pred_new over pred_base on each half. `half` is 'A' or 'B' per row."""
    half = np.asarray(half)
    gains = []
    for name in ("A", "B"):
        m = half == name
        if not m.any():
            gains.append(0.0)
            continue
        new = macro_scores(y[m], pred_new[m])["f1"]
        base = macro_scores(y[m], pred_base[m])["f1"]
        gains.append(new - base)
    return gains[0], gains[1]


def paired_bootstrap(y, pred_new, pred_base, n: int = 1000, seed: int = 781) -> float:
    """One-sided p that the new predictions are not better (share of resamples with gain <= 0)."""
    y = np.asarray(y)
    pred_new = np.asarray(pred_new)
    pred_base = np.asarray(pred_base)
    rng = np.random.default_rng(seed)
    n_cases = len(y)
    worse = 0
    for _ in range(n):
        idx = rng.integers(0, n_cases, n_cases)
        gain = macro_scores(y[idx], pred_new[idx])["f1"] - macro_scores(y[idx], pred_base[idx])["f1"]
        if gain <= 0:
            worse += 1
    return worse / max(n, 1)


def adopt(base_f1: float, new_f1: float, half_gain: tuple[float, float], p_value: float,
          margin: float = ADOPT_GAIN) -> bool:
    """A variant is kept only with +0.7 points on both D-sel halves and bootstrap p < 0.1."""
    if new_f1 - base_f1 < margin:
        return False
    if half_gain[0] < margin or half_gain[1] < margin:
        return False
    return p_value < 0.1
