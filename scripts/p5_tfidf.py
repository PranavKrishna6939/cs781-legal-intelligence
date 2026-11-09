"""TF-IDF one-vs-rest member. CPU only. Writes logits in the same npz format as the neural runs."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, raw_dir
from saul.lsi.const import N_LABELS, SEED
from saul.lsi.data import dev_half_array, dedupe_labels, text_hash
from saul.lsi.decide import quick_macro
from saul.utils.io import ensure_dir, write_json


def _read(name: str) -> list[dict]:
    folder = "Test" if name == "test" else "Train"
    path = os.path.join(raw_dir(), folder, f"{name}.json")
    rows = json.load(open(path, encoding="utf-8"))
    return [{"id": str(row["id"]), "text": row["text"], "labels": dedupe_labels(row.get("labels", []))} for row in rows]


def _labels(rows: list[dict]) -> np.ndarray:
    mlb = MultiLabelBinarizer(classes=list(range(N_LABELS)))
    return mlb.fit_transform([row["labels"] for row in rows]).astype(np.float32)


def _features(train, dev, test, chars: bool):
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=300_000, sublinear_tf=True)
    xtr = word.fit_transform(row["text"] for row in train).tocsr()
    xdv = word.transform(row["text"] for row in dev).tocsr()
    xte = word.transform(row["text"] for row in test).tocsr()
    if not chars:
        for matrix in (xtr, xdv, xte):
            matrix.sum_duplicates()
            matrix.sort_indices()
        return xtr, xdv, xte
    char = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False)
    xtr = sparse.hstack([xtr, char.transform(row["text"] for row in train)]).tocsr()
    xdv = sparse.hstack([xdv, char.transform(row["text"] for row in dev)]).tocsr()
    xte = sparse.hstack([xte, char.transform(row["text"] for row in test)]).tocsr()
    for matrix in (xtr, xdv, xte):
        matrix.sum_duplicates()
        matrix.sort_indices()
    return xtr, xdv, xte


def _fit_lr(x, y, C: float, balanced: bool):
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, solver="liblinear", class_weight="balanced" if balanced else None),
        n_jobs=4,
    )
    clf.fit(x, y)
    return clf.predict_proba(x) if False else clf


def _proba(clf, x, kind: str) -> np.ndarray:
    return clf.predict_proba(x)


def _logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _save(exp: str, split: str, ids, logits, labels=None) -> None:
    directory = exp_dir(exp)
    ensure_dir(directory)
    payload = {"ids": np.asarray(ids), "logits": logits.astype(np.float32)}
    if labels is not None:
        payload["labels"] = labels.astype(np.float32)
    np.savez(os.path.join(directory, f"{split}_logits.npz"), **payload)


def main() -> None:
    train, dev, test = _read("train"), _read("dev"), _read("test")
    ytr, ydv = _labels(train), _labels(dev)
    half = dev_half_array([row["id"] for row in dev])
    # Drop train rows whose text is in dev so the member's dev logits are not memorised copies.
    dev_hash = {text_hash(row["text"]) for row in dev}
    keep = [i for i, row in enumerate(train) if text_hash(row["text"]) not in dev_hash]
    configs = [("B0", False, 8.0, True, "lr")]
    for C in (2.0, 4.0, 8.0, 16.0):
        for balanced in (True, False):
            configs.append((f"grid_c{C}_b{int(balanced)}", True, C, balanced, "lr"))
    configs.append(("svm", True, 1.0, True, "svm"))
    best = None
    x_cache = {}
    for name, chars, C, balanced, kind in configs:
        print(f"[tfidf] {name}", flush=True)
        if chars not in x_cache:
            x_cache[chars] = _features(train, dev, test, chars)
        xtr, xdv, xte = x_cache[chars]
        if kind == "svm":
            # Platt scaling is a sigmoid fit on a train-internal split, not a fixed squash of the margin.
            clf = OneVsRestClassifier(
                CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced"), method="sigmoid", cv=2),
                n_jobs=4,
            )
        else:
            clf = OneVsRestClassifier(
                LogisticRegression(C=C, solver="liblinear", class_weight="balanced" if balanced else None),
                n_jobs=4,
            )
        clf.fit(xtr[keep], ytr[keep])
        dev_p = _proba(clf, xdv, kind)
        scored = quick_macro(_logit(dev_p), ydv, half)
        print(f"[tfidf] {name} macro {scored['macro_f1']:.4f}", flush=True)
        if best is None or scored["macro_f1"] > best[0]:
            best = (scored["macro_f1"], name, chars, C, balanced, kind, _logit(dev_p), _logit(_proba(clf, xte, kind)), scored)
    assert best is not None
    score, name, chars, C, balanced, kind, dev_logits, test_logits, scored = best
    _save("B1", "dev", [row["id"] for row in dev], dev_logits, ydv)
    _save("B1", "test", [row["id"] for row in test], test_logits)
    # Refit the winner and write train logits for the KD teacher. The running copy of this
    # script does not see this change until it is started again.
    xtr, _, _ = x_cache[chars]
    if kind == "svm":
        winner = OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced"), method="sigmoid", cv=2),
            n_jobs=4,
        )
    else:
        winner = OneVsRestClassifier(
            LogisticRegression(C=C, solver="liblinear", class_weight="balanced" if balanced else None),
            n_jobs=4,
        )
    winner.fit(xtr[keep], ytr[keep])
    _save("B1", "train", [row["id"] for row in train], _logit(_proba(winner, xtr, kind)), ytr)
    write_json({"macro_f1": score, "config": name, "half": scored["half"]}, os.path.join(exp_dir("B1"), "done.json"))
    print(json.dumps({"B1": score, "config": name}), flush=True)


if __name__ == "__main__":
    main()
