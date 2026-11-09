"""Cross-fitted lexical logits and retrieval neighbours for SAUL-LSI. CPU only.

Word 1-2-gram TF-IDF, vocabulary fit on train + dev + test text. LinearSVC C=0.5, decision values
times 4, are z_lex. Train rows are 5-fold out-of-fold, grouped by text hash. Dev half A is scored
by train + half B, and test is the mean of the two half-fits.

Neighbours are the top 30 by cosine in that same TF-IDF space. Train excludes the case itself.
Dev half A retrieves from train + half B. Test retrieves from train + dev.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import p5_dir
from saul.lsi.const import N_LABELS, SEED
from saul.lsi.data import dev_half_array, text_hash
from saul.lsi.decide import quick_macro
from scripts.p5_tfidf import _labels, _read

SCALE = 4.0
C = 0.5
TOPK = 30
BLOCK = 2000


def _check_ids(name: str, rows: list[dict]) -> None:
    path = os.path.join(p5_dir(), "tok", f"{name}.meta.json")
    meta = json.load(open(path, encoding="utf-8"))
    meta_ids = [row["id"] for row in meta]
    raw_ids = [row["id"] for row in rows]
    if meta_ids != raw_ids:
        raise SystemExit(f"[side] {name} ids differ from tok meta ({len(meta_ids)} vs {len(raw_ids)})")


def _vectorise(train, dev, test) -> dict:
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=300_000, min_df=2,
                           sublinear_tf=True, dtype=np.float32)
    corpus = [row["text"] for row in train + dev + test]
    print(f"[side] fitting TF-IDF on {len(corpus)} cases", flush=True)
    word.fit(corpus)
    return {name: word.transform([row["text"] for row in rows]).tocsr()
            for name, rows in (("train", train), ("dev", dev), ("test", test))}


def _fit(x, y):
    return OneVsRestClassifier(LinearSVC(C=C), n_jobs=6).fit(x, y)


def _folds(train: list[dict]) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(train):
        groups.setdefault(text_hash(row["text"]), []).append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)
    folds = np.zeros(len(train), dtype=np.int8)
    for index, key in enumerate(keys):
        for row in groups[key]:
            folds[row] = index % 5
    return folds


def _zlex(train, dev, test, x, ytr, ydv, half) -> None:
    feats = os.path.join(p5_dir(), "feats")
    os.makedirs(feats, exist_ok=True)
    folds = _folds(train)
    train_logits = np.zeros((len(train), N_LABELS), dtype=np.float32)
    for fold in range(5):
        start = time.time()
        keep = folds != fold
        score = folds == fold
        clf = _fit(x["train"][keep], ytr[keep])
        train_logits[score] = clf.decision_function(x["train"][score]) * SCALE
        print(f"[side] OOF fold {fold} ({time.time() - start:.0f}s)", flush=True)
    dev_logits = np.zeros_like(ydv)
    test_logits = np.zeros((len(test), N_LABELS), dtype=np.float32)
    for fold in ("A", "B"):
        start = time.time()
        fit_rows = np.flatnonzero(half == fold)
        score_rows = np.flatnonzero(half != fold)
        clf = _fit(sparse.vstack([x["train"], x["dev"][fit_rows]]).tocsr(), np.vstack([ytr, ydv[fit_rows]]))
        dev_logits[score_rows] = clf.decision_function(x["dev"][score_rows]) * SCALE
        test_logits += clf.decision_function(x["test"]) * SCALE / 2
        print(f"[side] dev fold {fold} ({time.time() - start:.0f}s)", flush=True)
    np.save(os.path.join(feats, "zlex_train.npy"), train_logits)
    np.save(os.path.join(feats, "zlex_dev.npy"), dev_logits)
    np.save(os.path.join(feats, "zlex_test.npy"), test_logits)
    scored = quick_macro(dev_logits, ydv, half)
    print(json.dumps({"zlex_dev_macro": round(scored["macro_f1"], 4), "half": scored["half"],
                      "train_abs_mean": round(float(np.abs(train_logits).mean()), 3)}), flush=True)


def _topk(query: sparse.csr_matrix, pool: sparse.csr_matrix, pool_index: np.ndarray,
          exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    idx = np.zeros((query.shape[0], TOPK), dtype=np.int32)
    sim = np.zeros((query.shape[0], TOPK), dtype=np.float32)
    pool_t = pool.T.tocsr()
    for start in range(0, query.shape[0], BLOCK):
        rows = query[start:start + BLOCK]
        scores = (rows @ pool_t).toarray().astype(np.float32, copy=False)
        if exclude_self:
            for local in range(scores.shape[0]):
                scores[local, start + local] = -1.0
        local_idx = np.argpartition(-scores, TOPK, axis=1)[:, :TOPK]
        local_sim = np.take_along_axis(scores, local_idx, 1)
        order = np.argsort(-local_sim, axis=1)
        local_idx = np.take_along_axis(local_idx, order, 1)
        local_sim = np.take_along_axis(local_sim, order, 1)
        idx[start:start + rows.shape[0]] = pool_index[local_idx]
        sim[start:start + rows.shape[0]] = local_sim
        print(f"[side] knn {start + rows.shape[0]}/{query.shape[0]}", flush=True)
    return idx, sim


def _knn(train, dev, x, ytr, ydv, half) -> None:
    feats = os.path.join(p5_dir(), "feats")
    n_train = len(train)
    labels = np.vstack([ytr, ydv]).astype(np.float32)
    np.save(os.path.join(feats, "knn_labels.npy"), labels)
    train_index = np.arange(n_train, dtype=np.int32)
    idx, sim = _topk(x["train"], x["train"], train_index, exclude_self=True)
    np.savez(os.path.join(feats, "knn_train.npz"), idx=idx, sim=sim)
    dev_idx = np.zeros((len(dev), TOPK), dtype=np.int32)
    dev_sim = np.zeros((len(dev), TOPK), dtype=np.float32)
    for name in ("A", "B"):
        rows = np.flatnonzero(half == name)
        other = np.flatnonzero(half != name)
        pool = sparse.vstack([x["train"], x["dev"][other]]).tocsr()
        pool_index = np.concatenate([train_index, n_train + other.astype(np.int32)])
        dev_idx[rows], dev_sim[rows] = _topk(x["dev"][rows], pool, pool_index, exclude_self=False)
        print(f"[side] knn dev half {name}", flush=True)
    np.savez(os.path.join(feats, "knn_dev.npz"), idx=dev_idx, sim=dev_sim)
    pool = sparse.vstack([x["train"], x["dev"]]).tocsr()
    pool_index = np.arange(n_train + len(dev), dtype=np.int32)
    test_idx, test_sim = _topk(x["test"], pool, pool_index, exclude_self=False)
    np.savez(os.path.join(feats, "knn_test.npz"), idx=test_idx, sim=test_sim)


def main() -> None:
    train, dev, test = _read("train"), _read("dev"), _read("test")
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        _check_ids(name, rows)
    ytr, ydv = _labels(train), _labels(dev)
    half = dev_half_array([row["id"] for row in dev])
    x = _vectorise(train, dev, test)
    feats = os.path.join(p5_dir(), "feats")
    os.makedirs(feats, exist_ok=True)
    for name, matrix in x.items():
        sparse.save_npz(os.path.join(feats, f"tfidf_{name}.npz"), matrix)
    _knn(train, dev, x, ytr, ydv, half)
    _zlex(train, dev, test, x, ytr, ydv, half)


if __name__ == "__main__":
    main()
