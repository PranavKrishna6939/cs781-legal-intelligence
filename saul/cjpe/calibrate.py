"""Prior correction, temperature, and cross-fitted threshold and stacker."""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from saul.cjpe.const import POINT, PRIOR0, PRIOR1, STACK_GAIN, THRESH_GAIN
from saul.cjpe.score import macro_scores, predict_logits

PRIOR_SHIFT = math.log(PRIOR0 / PRIOR1)


def apply_prior(logits: np.ndarray, plain_ce: bool) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).copy()
    if plain_ce:
        logits[:, 1] += PRIOR_SHIFT
    return logits


def log_odds(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return logits[:, 1] - logits[:, 0]


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.5, hi: float = 3.0, steps: int = 26) -> float:
    labels = np.asarray(labels)
    best_t, best = 1.0, _nll(logits, labels, 1.0)
    for temperature in np.linspace(lo, hi, steps):
        score = _nll(logits, labels, float(temperature))
        if score < best:
            best, best_t = score, float(temperature)
    return best_t


def checked_temperature(logits: np.ndarray, labels: np.ndarray, half: np.ndarray) -> dict:
    """Fit on D-sel-A. Keep it only when D-sel-B macro-F1 does not fall."""
    half = np.asarray(half)
    labels = np.asarray(labels)
    fit, check = half == "A", half == "B"
    temperature = fit_temperature(logits[fit], labels[fit])
    base = macro_scores(labels[check], predict_logits(logits[check]))["f1"]
    scaled = macro_scores(labels[check], predict_logits(logits[check] / temperature))["f1"]
    kept = scaled + 1e-12 >= base
    return {"temperature": temperature if kept else 1.0, "fit": temperature, "kept": kept, "check_f1": scaled, "base_f1": base}


def fit_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    best_t, best = 0.0, macro_scores(labels, predict_logits(np.stack([-scores, scores], axis=1), 0.0))["f1"]
    grid = np.quantile(scores, np.linspace(0.05, 0.95, 37))
    for threshold in grid:
        f1 = macro_scores(labels, (scores > threshold).astype(np.int64))["f1"]
        if f1 > best:
            best, best_t = f1, float(threshold)
    return best_t


def cross_fitted_threshold(scores: np.ndarray, labels: np.ndarray, half: np.ndarray) -> dict:
    """Threshold fit on one half is scored only on the other half."""
    half = np.asarray(half)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.zeros(len(labels), dtype=np.int64)
    used = {}
    gains = []
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        threshold = fit_threshold(scores[fit], labels[fit])
        used[f"t_for_{score_name}"] = threshold
        pred[score] = (scores[score] > threshold).astype(np.int64)
        tuned = macro_scores(labels[score], pred[score])["f1"]
        base = macro_scores(labels[score], (scores[score] > 0).astype(np.int64))["f1"]
        gains.append(tuned - base)
    gain = float(np.mean(gains))
    return {
        "gain": gain,
        "kept": gain >= THRESH_GAIN,
        "threshold": 0.0 if gain < THRESH_GAIN else float(np.mean(list(used.values()))),
        "pred": pred,
        **used,
    }


def _stack_matrix(member_logits: list[np.ndarray], log_length: np.ndarray, decade: np.ndarray) -> np.ndarray:
    cols = [log_odds(logits)[:, None] for logits in member_logits]
    cols.append(np.asarray(log_length, dtype=np.float64)[:, None])
    cols.append(np.asarray(decade, dtype=np.float64)[:, None])
    return np.concatenate(cols, axis=1)


def cross_fitted_stacker(member_logits: list[np.ndarray], labels: np.ndarray, half: np.ndarray,
                         log_length: np.ndarray, decade: np.ndarray, base_scores: np.ndarray) -> dict:
    half = np.asarray(half)
    labels = np.asarray(labels)
    features = _stack_matrix(member_logits, log_length, decade)
    pred = np.zeros(len(labels), dtype=np.int64)
    oof = np.full(len(labels), 0.5, dtype=np.float64)
    gains = []
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name
        clf = LogisticRegression(max_iter=200)
        clf.fit(features[fit], labels[fit])
        proba = clf.predict_proba(features[score])
        classes = list(clf.classes_)
        positive = proba[:, classes.index(1)] if 1 in classes else np.zeros(int(score.sum()))
        oof[score] = positive
        pred[score] = (positive > 0.5).astype(np.int64)
        tuned = macro_scores(labels[score], pred[score])["f1"]
        base = macro_scores(labels[score], (base_scores[score] > 0).astype(np.int64))["f1"]
        gains.append(tuned - base)
    gain = float(np.mean(gains))
    return {"gain": gain, "kept": gain >= STACK_GAIN, "pred": pred, "oof_positive": oof, "point": POINT}
