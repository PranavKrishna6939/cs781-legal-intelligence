"""How much of the label signal sits past the first N words. Word TF-IDF + LinearSVC, CPU only."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir
from saul.lsi.data import dev_half_array
from saul.lsi.decide import quick_macro
from scripts.p5_tfidf import _labels, _read


def main() -> None:
    train, dev = _read("train"), _read("dev")
    ytr, ydv = _labels(train), _labels(dev)
    half = dev_half_array([row["id"] for row in dev])
    result = {}
    views = [("head512", 512, 0), ("head384_tail128", 384, 128), ("tail512", 0, 512),
             ("head1024", 1024, 0), ("head768_tail256", 768, 256), ("full", 0, 0)]
    for name, head, tail in views:
        start = time.time()

        def cut(rows):
            if not head and not tail:
                return [row["text"] for row in rows]
            out = []
            for row in rows:
                words = row["text"].split()
                if len(words) <= head + tail:
                    out.append(row["text"])
                else:
                    out.append(" ".join(words[:head] + (words[-tail:] if tail else [])))
            return out

        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=300_000, sublinear_tf=True, min_df=2)
        xtr = vec.fit_transform(cut(train))
        xdv = vec.transform(cut(dev))
        clf = OneVsRestClassifier(LinearSVC(C=0.5), n_jobs=6).fit(xtr, ytr)
        scored = quick_macro(clf.decision_function(xdv) * 4.0, ydv, half)
        result[name] = round(scored["macro_f1"], 4)
        print(f"[trunc] {name}: macro {scored['macro_f1']:.4f} ({time.time() - start:.0f}s)", flush=True)
    os.makedirs(exp_dir("probe"), exist_ok=True)
    with open(os.path.join(exp_dir("probe"), "trunc.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main()
