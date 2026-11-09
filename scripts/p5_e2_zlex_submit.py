"""Write P5_E2_zlex: blend of E2_ced logits with cross-fitted TF-IDF LinearSVC margins (lambda 0.45),
per-label thresholds fitted on each dev half, shrunk to the global threshold, then averaged."""
import json
import os

import numpy as np

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.data import dev_half_array
from saul.lsi.decide import at_least_one, best_global_threshold, optimal_label_thresholds, shrink_thresholds
from saul.lsi.score import sigmoid
from saul.lsi.submit import rows_from_matrix, test_ids, write_zip

LAM = 0.45


def align(blob, order):
    index = {str(i): n for n, i in enumerate(blob["ids"])}
    return np.asarray(blob["logits"], dtype=np.float64)[[index[i] for i in order]]


def main() -> None:
    meta = json.load(open(os.path.join(p5_dir(), "tok", "dev.meta.json")))
    dev_ids = [r["id"] for r in meta]
    labels = np.zeros((len(dev_ids), N_LABELS))
    for i, r in enumerate(meta):
        for lab in r["labels"]:
            labels[i, int(lab)] = 1.0
    half = dev_half_array(dev_ids)
    ed = align(np.load(os.path.join(exp_dir("E2_ced"), "dev_logits.npz")), dev_ids)
    et = align(np.load(os.path.join(exp_dir("E2_ced"), "test_logits.npz")), test_ids())
    zd = np.load(os.path.join(p5_dir(), "feats", "zlex_dev.npy")).astype(np.float64)
    zt = np.load(os.path.join(p5_dir(), "feats", "zlex_test.npy")).astype(np.float64)
    probs = sigmoid((1 - LAM) * ed + LAM * zd)
    folds = []
    for name in ("A", "B"):
        rows = half == name
        folds.append(shrink_thresholds(
            optimal_label_thresholds(probs[rows], labels[rows]),
            best_global_threshold(probs[rows], labels[rows]),
            labels[rows].sum(0), 50))
    thr = np.mean(folds, axis=0)
    order = test_ids()
    pred = at_least_one(sigmoid((1 - LAM) * et + LAM * zt), thr).astype(np.int64)
    print("empty", int((pred.sum(1) == 0).sum()), "mean", round(float(pred.sum(1).mean()), 2), flush=True)
    directory = os.path.join(os.path.dirname(p5_dir()), "submissions", "phase5", "P5_E2_zlex")
    write_zip(rows_from_matrix(order, pred), directory, expected=order, require_label=True)


if __name__ == "__main__":
    main()
