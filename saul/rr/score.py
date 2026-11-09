"""13-label macro-F1, matching evaluate_phase3.py (absent labels score 0)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from saul.ner.registry import repo_root
from saul.rr.data import LABELS, N_LABELS, raw_dir


def macro_scores(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    f1s, precs, recs = [], [], []
    per = {}
    for c in range(N_LABELS):
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if tp + fp + fn == 0:
            prec = rec = f1 = 0.0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        per[c] = f1
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "per_label": per,
    }


def official_macro(docs, pred: dict[str, list[str]]) -> dict[str, float]:
    """Score with resources/evaluation_script/evaluate_phase3.py."""
    script = os.path.join(repo_root(), "resources", "evaluation_script", "evaluate_phase3.py")
    vocab = os.path.join(raw_dir(), "Train", "label_vocab.json")
    if not os.path.exists(vocab):
        vocab_data = {name: i for i, name in enumerate(LABELS)}
    else:
        vocab_data = None
    cl, it, guess = {}, {}, {}
    for doc in docs:
        labels = pred.get(doc.doc_id)
        if labels is None or doc.labels is None:
            continue
        bucket = cl if doc.domain == "CL" else it
        bucket[doc.doc_id] = doc.labels
        guess[doc.doc_id] = labels
    with tempfile.TemporaryDirectory() as tmp:
        cl_path = os.path.join(tmp, "cl.json")
        it_path = os.path.join(tmp, "it.json")
        pr_path = os.path.join(tmp, "prediction.json")
        vb_path = os.path.join(tmp, "labels.json")
        for path, obj in ((cl_path, cl), (it_path, it), (pr_path, guess)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        with open(vb_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data if vocab_data is not None else json.load(open(vocab, encoding="utf-8")), f)
        proc = subprocess.run(
            [sys.executable, script, "--truth-cl", cl_path, "--truth-it", it_path,
             "--prediction", pr_path, "--labels", vb_path],
            capture_output=True, text=True, check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("Macro-F1"):
            out["macro_f1"] = float(line.split()[-1])
        elif line.startswith("Macro-Precision"):
            out["macro_precision"] = float(line.split()[-1])
        elif line.startswith("Macro-Recall"):
            out["macro_recall"] = float(line.split()[-1])
    return out


def confusion_pair(y_true, y_pred, a: int, b: int) -> dict:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    gold_a = yt == a
    return {
        "gold_a": int(gold_a.sum()),
        "a_as_b": int(np.sum(gold_a & (yp == b))),
        "b_as_a": int(np.sum((yt == b) & (yp == a))),
    }
