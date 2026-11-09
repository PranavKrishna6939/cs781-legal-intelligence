"""TF-IDF logistic regression on the last 2,000 words. CPU floor for the report."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from saul.cjpe import p4_dir
from saul.cjpe.score import macro_scores


def _rows(name: str) -> list[dict]:
    path = os.path.join(p4_dir(), "text", f"{name}.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tail(text: str, n: int = 2000) -> str:
    words = text.split()
    return " ".join(words[-n:])


def main() -> None:
    pool, dsel, test = _rows("pool"), _rows("dsel"), _rows("test")
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=200000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=200000)
    train_text = [_tail(row["text"]) for row in pool]
    dev_text = [_tail(row["text"]) for row in dsel]
    test_text = [_tail(row["text"]) for row in test]
    x_word = word.fit_transform(train_text)
    x_char = char.fit_transform(train_text)
    from scipy.sparse import hstack
    train_x = hstack([x_word, x_char])
    labels = np.asarray([int(row["label"]) for row in pool])
    clf = LogisticRegression(class_weight="balanced", max_iter=200)
    clf.fit(train_x, labels)
    dev_x = hstack([word.transform(dev_text), char.transform(dev_text)])
    test_x = hstack([word.transform(test_text), char.transform(test_text)])
    dev_proba = clf.predict_proba(dev_x)
    test_proba = clf.predict_proba(test_x)
    classes = list(clf.classes_)
    def logits(proba):
        out = np.zeros((len(proba), 2), dtype=np.float32)
        for index, label in enumerate(classes):
            out[:, int(label)] = np.log(np.clip(proba[:, index], 1e-6, 1))
        return out
    dev_logits, test_logits = logits(dev_proba), logits(test_proba)
    pred = (dev_logits[:, 1] > dev_logits[:, 0]).astype(np.int64)
    gold = np.asarray([int(row["label"]) for row in dsel])
    scores = macro_scores(gold, pred)
    dest = os.path.join(p4_dir(), "exp", "B0")
    os.makedirs(dest, exist_ok=True)
    np.savez(os.path.join(dest, "logits_dsel.npz"), ids=np.asarray([r["id"] for r in dsel]), logits=dev_logits, labels=gold)
    np.savez(os.path.join(dest, "logits_test.npz"), ids=np.asarray([r["id"] for r in test]), logits=test_logits)
    with open(os.path.join(dest, "done.json"), "w", encoding="utf-8") as handle:
        json.dump({"exp": "B0", "dev_f1": scores["f1"], "dev_p": scores["precision"], "dev_r": scores["recall"]}, handle, indent=2)
    print(f"[b0] dev_f1 {scores['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
