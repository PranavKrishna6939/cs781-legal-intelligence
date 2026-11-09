"""Summary-length prediction. The stop length is lambda times the prediction.

The self-predicted length is the expected bucket from one forward at the summary marker.
The fallback is a ridge on document features. The trainer keeps whichever has the lower
holdout mean absolute error.
"""
from __future__ import annotations

import math
import re

import numpy as np

from saul.summ.data import BUCKET_EDGES, bucket_midpoints, bucket_of

_FEATURES = ("section", "article", "appeal", "whether", "dissent")


def features(document: str) -> np.ndarray:
    words = document.split()
    low = document.lower()
    counts = [len(re.findall(rf"\b{name}\b", low)) for name in _FEATURES]
    civil = 1.0 if "civil appeal" in low[:400] else 0.0
    criminal = 1.0 if "criminal appeal" in low[:400] else 0.0
    return np.array([math.log(len(words) + 1), *counts, civil, criminal], dtype=np.float64)


def fit_ridge(rows: list[dict], lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge on log(summary words). Returns the weight vector, bias last."""
    x = np.stack([features(row["document"]) for row in rows])
    y = np.log(np.array([max(1, len(row["summary"].split())) for row in rows], dtype=np.float64))
    x = np.concatenate([x, np.ones((len(rows), 1))], axis=1)
    reg = lam * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def predict_ridge(weights: np.ndarray, document: str) -> float:
    x = np.concatenate([features(document), [1.0]])
    return float(math.exp(x @ weights))


def ridge_mae(weights: np.ndarray, rows: list[dict]) -> float:
    err = [abs(predict_ridge(weights, row["document"]) - len(row["summary"].split())) for row in rows]
    return float(np.mean(err)) if err else 0.0


def words_from_probs(probs) -> float:
    mids = bucket_midpoints()
    return float(sum(float(p) * m for p, m in zip(probs, mids)))


def words_from_bucket(index: int) -> float:
    return bucket_midpoints()[index]


def gold_bucket(summary: str) -> int:
    return bucket_of(len(summary.split()))


def stop_words(predicted: float, lam: float, lo: int = 64, hi: int = 4096) -> int:
    return int(min(hi, max(lo, round(lam * predicted))))
