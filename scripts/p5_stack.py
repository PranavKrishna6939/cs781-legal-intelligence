"""Phase 5 stack: members + rule features → per-label scores → cross-fitted thresholds → submission.

Usage: python -m scripts.p5_stack --members L1cf2,E2_ced_full --out S0 [--zip]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.data import dev_half_array
from saul.lsi.submit import rows_from_matrix, write_zip

KAPPA = 50
GAIN = 0.003


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def label_thresholds(probs, labels):
    """Best-F1 cut per label from one sort per column."""
    out = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        order = np.argsort(-probs[:, j])
        s = probs[order, j]
        tp = np.cumsum(labels[order, j])
        pos = tp[-1]
        if pos == 0:
            out[j] = 1.01
            continue
        k = np.arange(1, len(s) + 1)
        f1 = 2 * tp / (k + pos)
        best = int(np.argmax(f1))
        out[j] = s[best] if best + 1 >= len(s) else 0.5 * (s[best] + s[best + 1])
    return out


def macro_f1(labels, pred):
    tp = (labels * pred).sum(0)
    denom = labels.sum(0) + pred.sum(0)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def micro_f1(labels, pred):
    tp = (labels * pred).sum()
    return float(2 * tp / max(labels.sum() + pred.sum(), 1))


def global_threshold(probs, labels):
    grid = np.linspace(0.05, 0.95, 19)
    scores = [macro_f1(labels, probs >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def fit_thresholds(probs, labels, kappa=KAPPA):
    per = label_thresholds(probs, labels)
    g = global_threshold(probs, labels)
    n = labels.sum(0)
    return (n * per + kappa * g) / (n + kappa)


def at_least_one(probs, thr):
    pred = probs >= thr
    empty = ~pred.any(1)
    top = (probs / np.clip(thr, 1e-6, None)).argmax(1)
    pred[empty, top[empty]] = True
    return pred


def crossfit(logits, labels, half):
    probs = sigmoid(logits)
    pred = np.zeros_like(labels, dtype=bool)
    parts = {}
    for fit, score in (("A", "B"), ("B", "A")):
        f, s = half == fit, half == score
        pred[s] = at_least_one(probs[s], fit_thresholds(probs[f], labels[f]))
        parts[score] = macro_f1(labels[s], pred[s])
    return {"macro": float(np.mean(list(parts.values()))), "half": parts, "pred": pred}


def load_member(name, dev_ids, test_ids):
    out = []
    for kind, ids in (("dev", dev_ids), ("test", test_ids)):
        z = np.load(os.path.join(exp_dir(name), f"{kind}_logits.npz"))
        order = {str(i): k for k, i in enumerate(z["ids"])}
        out.append(z["logits"][[order[i] for i in ids]].astype(np.float32))
    return out


def weighted(members_dev, members_test, labels, half):
    """Coordinate ascent on one weight per member, cross-fitted macro-F1."""
    m = len(members_dev)
    w = np.ones(m)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    best = crossfit(sum(w[i] * members_dev[i] for i in range(m)) / w.sum(), labels, half)["macro"]
    for _ in range(2):
        for i in range(m):
            for value in grid:
                trial = w.copy()
                trial[i] = value
                if trial.sum() == 0:
                    continue
                score = crossfit(sum(trial[k] * members_dev[k] for k in range(m)) / trial.sum(), labels, half)["macro"]
                if score > best + 1e-5:
                    best, w = score, trial
    dev = sum(w[i] * members_dev[i] for i in range(m)) / w.sum()
    test = sum(w[i] * members_test[i] for i in range(m)) / w.sum()
    return dev, test, w


def partners(labels, k=3):
    co = labels.T @ labels
    np.fill_diagonal(co, 0)
    return np.argsort(-co, axis=1)[:, :k]


def stacker(members_dev, members_test, rules, base_dev, base_test, labels, half, train_y):
    near = partners(train_y)

    def block(members, base, kind, j):
        cols = [m[:, j] for m in members]
        cols.append(logit(rules[f"knn_{kind}"][:, j]))
        cols.append(rules[f"sec_{kind}"][:, j])
        dup = rules[f"dup_{kind}"]
        cols.append(dup[:, j])
        cols.append(dup[:, N_LABELS])
        cols.append(base[:, near[j]].mean(1))
        return np.stack(cols, 1)

    dev_out = np.zeros_like(base_dev)
    test_out = np.zeros_like(base_test)
    for j in range(N_LABELS):
        xd = block(members_dev, base_dev, "dev", j)
        xt = block(members_test, base_test, "test", j)
        for fit, score in (("A", "B"), ("B", "A")):
            f, s = half == fit, half == score
            clf = LogisticRegression(C=1.0, max_iter=400).fit(xd[f], labels[f, j])
            dev_out[s, j] = clf.decision_function(xd[s])
            test_out[:, j] += clf.decision_function(xt) / 2
    return dev_out, test_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--no-rules", action="store_true")
    args = parser.parse_args()
    rules = dict(np.load(os.path.join(p5_dir(), "feats", "rules.npz")))
    dev_ids = [str(i) for i in rules["dev_ids"]]
    test_ids = [str(i) for i in rules["test_ids"]]
    labels = np.load(os.path.join(exp_dir("L1cf2"), "dev_logits.npz"))["labels"].astype(np.float32)
    half = dev_half_array(dev_ids)
    names = args.members.split(",")
    loaded = [load_member(n, dev_ids, test_ids) for n in names]
    mdev = [zscore(d) for d, _ in loaded]
    mtest = [zscore(t) for _, t in loaded]
    report = {"members": {}}
    for n, d in zip(names, mdev):
        report["members"][n] = {k: v for k, v in crossfit(d, labels, half).items() if k != "pred"}
    base_dev, base_test, w = weighted(mdev, mtest, labels, half)
    base = crossfit(base_dev, labels, half)
    report["weighted"] = {"weights": dict(zip(names, w.tolist())), "macro": base["macro"], "half": base["half"]}
    final_dev, final_test, chosen = base_dev, base_test, "weighted"
    if not args.no_rules:
        from scripts.p5_tfidf import _labels, _read
        train_y = _labels(_read("train"))
        sdev, stest = stacker(mdev, mtest, rules, base_dev, base_test, labels, half, train_y)
        st = crossfit(sdev, labels, half)
        report["stacker"] = {"macro": st["macro"], "half": st["half"]}
        if all(st["half"][h] - base["half"][h] >= GAIN for h in ("A", "B")):
            final_dev, final_test, chosen = sdev, stest, "stacker"
    report["chosen"] = chosen
    final = crossfit(final_dev, labels, half)
    report["final"] = {"macro": final["macro"], "half": final["half"], "micro": micro_f1(labels, final["pred"]),
                       "subset": float((final["pred"] == (labels > 0)).all(1).mean()),
                       "labels_per_case": float(final["pred"].sum(1).mean())}
    thr = fit_thresholds(sigmoid(final_dev), labels)
    pred = at_least_one(sigmoid(final_test), thr)
    report["test_labels_per_case"] = float(pred.sum(1).mean())
    out_dir = os.path.join(p5_dir(), "subs", args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "scores.npz"), dev=final_dev, test=final_test, dev_ids=np.asarray(dev_ids),
             test_ids=np.asarray(test_ids))
    if args.zip:
        write_zip(rows_from_matrix(test_ids, pred), out_dir, expected=test_ids, require_label=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps({k: report[k] for k in ("weighted", "chosen", "final")} | {"stacker": report.get("stacker")}),
          flush=True)


if __name__ == "__main__":
    main()
