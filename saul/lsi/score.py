"""Multi-label scores and the adoption rule."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from saul.lsi.const import ADOPT_GAIN, N_LABELS, SEED


def _binary(y) -> np.ndarray:
    return (np.asarray(y) > 0).astype(np.int64)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(_binary(y_true), _binary(y_pred), average="macro", zero_division=0))


def micro_f1(y_true, y_pred) -> float:
    return float(f1_score(_binary(y_true), _binary(y_pred), average="micro", zero_division=0))


def subset_accuracy(y_true, y_pred) -> float:
    y_true = _binary(y_true)
    y_pred = _binary(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).all(axis=1).mean())


def per_label_f1(y_true, y_pred) -> np.ndarray:
    return f1_score(_binary(y_true), _binary(y_pred), average=None, zero_division=0,
                    labels=list(range(np.asarray(y_true).shape[1])))


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": micro_f1(y_true, y_pred),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
    }


def half_macro(y_true, y_pred, half) -> dict[str, float]:
    half = np.asarray(half)
    out = {}
    for name in ("A", "B"):
        mask = half == name
        out[name] = macro_f1(y_true[mask], y_pred[mask]) if mask.any() else 0.0
    return out


def paired_bootstrap(y_true, pred_new, pred_base, n: int = 400, seed: int = SEED) -> float:
    """Share of resamples where the new macro-F1 is not higher. One-sided p."""
    y_true = _binary(y_true)
    pred_new = _binary(pred_new)
    pred_base = _binary(pred_base)
    rng = np.random.default_rng(seed)
    n_cases = len(y_true)
    worse = 0
    for _ in range(n):
        idx = rng.integers(0, n_cases, n_cases)
        gain = macro_f1(y_true[idx], pred_new[idx]) - macro_f1(y_true[idx], pred_base[idx])
        if gain <= 0:
            worse += 1
    return worse / max(n, 1)


def adopt(base_f1: float, new_f1: float, half_gain: tuple[float, float], p_value: float,
          margin: float = ADOPT_GAIN) -> bool:
    """Keep a variant only with +0.7 points on both dev halves and bootstrap p < 0.1."""
    if new_f1 - base_f1 < margin:
        return False
    if half_gain[0] < margin or half_gain[1] < margin:
        return False
    return p_value < 0.1


def sigmoid(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def rare_macro(y_true, y_pred, freq, k: int = 20) -> float:
    order = np.argsort(np.asarray(freq))[:k]
    y_true = _binary(y_true)[:, order]
    y_pred = _binary(y_pred)[:, order]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def n_labels() -> int:
    return N_LABELS
