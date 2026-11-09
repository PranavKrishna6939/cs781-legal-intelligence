"""L1: lexical member, swept on train → dev, then refit on train plus one dev half (CF2). CPU only.

Dev half B is scored by the fit on train + half A, and the other way round. Test is the mean of both fits.
Also writes the word TF-IDF matrices used by the kNN member.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir
from saul.lsi.data import dev_half_array
from saul.lsi.decide import quick_macro
from saul.utils.io import ensure_dir, write_json
from scripts.p5_tfidf import _labels, _read

SCALE = 4.0


def features(train, dev, test, ngram_max: int) -> dict:
    word = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=600_000 if ngram_max > 2 else 300_000,
                           sublinear_tf=True, min_df=2, dtype=np.float32)
    word.fit([row["text"] for row in train] + [row["text"] for row in dev])
    return {name: word.transform([row["text"] for row in rows]).tocsr()
            for name, rows in (("train", train), ("dev", dev), ("test", test))}


def add_char(x: dict, rows: dict, bits: int = 18) -> dict:
    char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** bits, alternate_sign=False,
                             dtype=np.float32, norm="l2")
    out = {}
    for name, matrix in x.items():
        c = char.transform([row["text"] for row in rows[name]])
        out[name] = sparse.hstack([matrix, 0.5 * c]).tocsr()
    return out


def fit(x, y, C: float):
    # liblinear copies X to float64 in every worker; the char view only fits in RAM with two workers.
    jobs = 6 if x.nnz < 150_000_000 else 2
    return OneVsRestClassifier(LinearSVC(C=C), n_jobs=jobs).fit(x, y)


def main() -> None:
    directory = exp_dir("L1cf2")
    ensure_dir(directory)
    ensure_dir(os.path.join(p5_dir(), "feats"))
    train, dev, test = _read("train"), _read("dev"), _read("test")
    rows = {"train": train, "dev": dev, "test": test}
    ytr, ydv = _labels(train), _labels(dev)
    half = dev_half_array([row["id"] for row in dev])
    sweep = {}
    views = {}
    views["w12"] = features(train, dev, test, 2)
    for name, matrix in views["w12"].items():
        sparse.save_npz(os.path.join(p5_dir(), "feats", f"word_{name}.npz"), matrix)
    views["w13"] = features(train, dev, test, 3)
    views["w12c"] = add_char(views["w12"], rows)
    best = None
    for view in ("w12", "w13", "w12c"):
        for C in (0.25, 0.5, 1.0):
            start = time.time()
            x = views[view]
            clf = fit(x["train"], ytr, C)
            scored = quick_macro(clf.decision_function(x["dev"]) * SCALE, ydv, half)
            sweep[f"{view}_C{C}"] = {"macro": round(scored["macro_f1"], 4), "half": scored["half"]}
            print(f"[l1] {view} C={C}: macro {scored['macro_f1']:.4f} ({time.time() - start:.0f}s)", flush=True)
            if best is None or scored["macro_f1"] > best[0]:
                best = (scored["macro_f1"], view, C)
    write_json(sweep, os.path.join(directory, "sweep.json"))
    _, view, C = best
    x = views[view]
    dev_logits = np.zeros(ydv.shape, dtype=np.float32)
    test_logits = np.zeros((len(test), ydv.shape[1]), dtype=np.float32)
    for fold in ("A", "B"):
        start = time.time()
        fit_rows = np.flatnonzero(half == fold)
        score_rows = np.flatnonzero(half != fold)
        clf = fit(sparse.vstack([x["train"], x["dev"][fit_rows]]).tocsr(), np.vstack([ytr, ydv[fit_rows]]), C)
        dev_logits[score_rows] = clf.decision_function(x["dev"][score_rows]) * SCALE
        test_logits += clf.decision_function(x["test"]) * SCALE / 2
        print(f"[l1] CF2 fold {fold} ({time.time() - start:.0f}s)", flush=True)
    np.savez(os.path.join(directory, "dev_logits.npz"), ids=np.asarray([row["id"] for row in dev]),
             logits=dev_logits, labels=ydv)
    np.savez(os.path.join(directory, "test_logits.npz"), ids=np.asarray([row["id"] for row in test]),
             logits=test_logits)
    clf = fit(x["train"], ytr, C)
    np.savez(os.path.join(directory, "trainonly_dev_logits.npz"), ids=np.asarray([row["id"] for row in dev]),
             logits=clf.decision_function(x["dev"]) * SCALE, labels=ydv)
    np.savez(os.path.join(directory, "trainonly_test_logits.npz"), ids=np.asarray([row["id"] for row in test]),
             logits=clf.decision_function(x["test"]) * SCALE)
    scored = quick_macro(dev_logits, ydv, half)
    write_json({"macro_f1": scored["macro_f1"], "half": scored["half"], "view": view, "C": C,
                "trainonly": sweep[f"{view}_C{C}"]}, os.path.join(directory, "done.json"))
    print(json.dumps({"L1cf2": scored["macro_f1"], "half": scored["half"], "view": view, "C": C}), flush=True)


if __name__ == "__main__":
    main()
