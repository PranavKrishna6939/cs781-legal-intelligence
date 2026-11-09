"""Document decoder: triangle-averaged marginals, per-domain Viterbi, nested bias check.

Scores are macro-F1 in [0, 1]. The plan's +1.0 and 2.5 are percentage points, so the
thresholds below are 0.01 and 0.025. Dissent and PrecedentOverruled biases stay 0.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from saul.rr.data import FROZEN_BIAS, LABEL_ID, LABELS, N_LABELS, labeled_docs, load_docs, p3_dir
from saul.rr.infer import load_probs
from saul.rr.score import macro_scores

LAMBDAS = (0.0, 0.25, 0.5, 1.0)
BIAS_GRID = np.linspace(-2.0, 2.0, 17)
FROZEN = {LABEL_ID[name] for name in FROZEN_BIAS}


def should_collapse(untuned: float, tuned: float, nested: float) -> bool:
    gain = (nested - untuned) * 100.0
    gap = (tuned - nested) * 100.0
    return gain < 1.0 or gap > 2.5


def transitions(docs, ids: set[str] | None = None) -> dict[str, np.ndarray]:
    counts = {dom: np.ones((N_LABELS, N_LABELS), dtype=np.float64) for dom in ("CL", "IT")}
    for doc in docs:
        if doc.labels is None:
            continue
        if ids is not None and doc.doc_id not in ids:
            continue
        prev = None
        for lab in doc.label_ids:
            if prev is not None:
                counts[doc.domain][prev, lab] += 1
            prev = lab
    out = {}
    for dom, table in counts.items():
        out[dom] = np.log(table / table.sum(1, keepdims=True))
    return out


def viterbi(probs: np.ndarray, log_t: np.ndarray, bias: np.ndarray, lam: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-8, None))
    n, k = logp.shape
    score = logp[0] + bias
    back = np.zeros((n, k), dtype=np.int16)
    trans = lam * log_t
    for i in range(1, n):
        cand = score[:, None] + trans + (logp[i] + bias)[None, :]
        back[i] = cand.argmax(0).astype(np.int16)
        score = cand[back[i], np.arange(k)]
    path = np.empty(n, dtype=np.int16)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _score(docs, probs, trans, lam: dict[str, float], bias: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    gold, pred = [], []
    paths = {}
    for doc in docs:
        if doc.doc_id not in probs or doc.labels is None:
            continue
        path = viterbi(probs[doc.doc_id], trans[doc.domain], bias, lam[doc.domain])
        paths[doc.doc_id] = path
        gold.extend(doc.label_ids)
        pred.extend(path.tolist())
    if not gold:
        return 0.0, paths
    return macro_scores(gold, pred)["macro_f1"], paths


def _best_lambda(docs, probs, trans, bias: np.ndarray) -> dict[str, float]:
    best_l, best_s = {dom: 0.0 for dom in ("CL", "IT")}, -1.0
    for lc in LAMBDAS:
        for li in LAMBDAS:
            lam = {"CL": lc, "IT": li}
            score, _ = _score(docs, probs, trans, lam, bias)
            if score > best_s:
                best_s = score
                best_l = lam
    return best_l


def tune_bias(docs, probs, trans, lam: dict[str, float], passes: int = 3) -> np.ndarray:
    bias = np.zeros(N_LABELS, dtype=np.float64)
    for _ in range(passes):
        for lab in range(N_LABELS):
            if lab in FROZEN:
                bias[lab] = 0.0
                continue
            best_v, best_s = 0.0, -1.0
            for value in BIAS_GRID:
                bias[lab] = float(value)
                score, _ = _score(docs, probs, trans, lam, bias)
                value = float(value)
                better = score > best_s + 1e-12
                # A tie keeps the smaller bias, so 0 wins when the grid does not help.
                closer = best_s >= 0 and abs(score - best_s) <= 1e-12 and abs(value) < abs(best_v) - 1e-12
                if better or closer or best_s < 0:
                    best_s = score
                    best_v = value
            bias[lab] = best_v
    bias[list(FROZEN)] = 0.0
    return bias


def load_split_probs(exp: str, kind: str) -> dict[str, np.ndarray] | None:
    root = os.path.join(p3_dir(), exp)
    if kind == "test":
        dest = os.path.join(root, "test")
        if not os.path.exists(os.path.join(dest, "done.json")):
            return None
        return {name[:-4]: load_probs(os.path.join(dest, name)) for name in os.listdir(dest) if name.endswith(".npz")}
    probs = {}
    for k in range(5):
        folder = os.path.join(root, f"fold{k}", "oof")
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.endswith(".npz"):
                probs[name[:-4]] = load_probs(os.path.join(folder, name))
    return probs or None


def average_marginals(exps: list[str], weights: list[float] | None = None, kind: str = "test") -> dict[str, np.ndarray]:
    weights = [1.0] * len(exps) if weights is None else weights
    tables = []
    used = []
    for exp, weight in zip(exps, weights):
        table = load_split_probs(exp, kind)
        if table is None:
            print(f"[mix] drop {exp}: no {kind} cache")
            continue
        tables.append((weight, table))
        used.append(exp)
    if not tables:
        raise FileNotFoundError(f"no caches for {exps}")
    ids = set(tables[0][1])
    for _, table in tables[1:]:
        ids &= set(table)
    total = sum(w for w, _ in tables)
    out = {}
    for doc_id in ids:
        acc = None
        for weight, table in tables:
            part = table[doc_id].astype(np.float64) * (weight / total)
            acc = part if acc is None else acc + part
        acc = np.clip(acc, 1e-8, None)
        acc /= acc.sum(1, keepdims=True)
        out[doc_id] = acc.astype(np.float32)
    return out


def _docs_for(ids: set[str] | None, docs):
    if ids is None:
        return [d for d in docs if d.labels is not None]
    return [d for d in docs if d.doc_id in ids and d.labels is not None]


def _shared_lambda(docs, probs, trans, bias: np.ndarray) -> dict[str, float]:
    best_v, best_s = 0.0, -1.0
    for value in LAMBDAS:
        lam = {"CL": float(value), "IT": float(value)}
        score, _ = _score(docs, probs, trans, lam, bias)
        if score > best_s:
            best_s = score
            best_v = float(value)
    return {"CL": best_v, "IT": best_v}


def _nested_shared(docs, probs, by_fold, all_ids) -> float:
    """Held-out macro-F1 of one global λ and zero biases, tuned on the other folds."""
    scores = []
    bias0 = np.zeros(N_LABELS)
    for ids in by_fold:
        train_ids = all_ids - set(ids)
        subset = _docs_for(train_ids, docs)
        trans = transitions(docs, train_ids)
        lam = _shared_lambda(subset, probs, trans, bias0)
        held = _docs_for(set(ids), docs)
        score, _ = _score(held, probs, trans, lam, bias0)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def fit_decoder(exp: str, docs=None, probs=None) -> dict:
    docs = labeled_docs(docs or load_docs())
    probs = probs if probs is not None else load_split_probs(exp, "oof")
    if not probs:
        raise FileNotFoundError(f"no OOF cache for {exp}")
    folds = json.load(open(os.path.join(p3_dir(), "folds.json"), encoding="utf-8"))
    by_fold = []
    for fold in folds:
        ids = set(fold)
        by_fold.append(ids)
    all_ids = set().union(*by_fold)

    def _fit(ids):
        subset = _docs_for(ids, docs)
        trans = transitions(docs, ids)
        bias0 = np.zeros(N_LABELS)
        lam = _best_lambda(subset, probs, trans, bias0)
        bias = tune_bias(subset, probs, trans, lam)
        lam = _best_lambda(subset, probs, trans, bias)
        return trans, lam, bias

    trans, lam, bias = _fit(all_ids)
    untuned, _ = _score(docs, probs, trans, {"CL": 0.0, "IT": 0.0}, np.zeros(N_LABELS))
    tuned, _ = _score(docs, probs, trans, lam, bias)
    nested_scores = []
    for k, ids in enumerate(by_fold):
        train_ids = all_ids - ids
        tr, la, bi = _fit(train_ids)
        held = _docs_for(ids, docs)
        score, _ = _score(held, probs, tr, la, bi)
        base, _ = _score(held, probs, tr, {"CL": 0.0, "IT": 0.0}, np.zeros(N_LABELS))
        nested_scores.append({"fold": k, "tuned": score, "untuned": base})
    nested = float(np.mean([r["tuned"] for r in nested_scores]))
    nested_base = float(np.mean([r["untuned"] for r in nested_scores]))
    collapsed = should_collapse(nested_base, tuned, nested)
    nested_kept = nested
    if collapsed:
        bias = np.zeros(N_LABELS)
        lam = _shared_lambda(docs, probs, trans, bias)
        nested_kept = _nested_shared(docs, probs, by_fold, all_ids)
    kept, _ = _score(docs, probs, trans, lam, bias)
    cfg = {
        "exp": exp,
        "lambda": lam,
        "bias": {LABELS[i]: float(bias[i]) for i in range(N_LABELS)},
        "untuned": untuned,
        "tuned": tuned,
        "kept": kept,
        "nested": nested,
        "nested_kept": nested_kept,
        "nested_folds": nested_scores,
        "collapsed": collapsed,
        "stack_kept": False,
        "exact_kept": False,
    }
    return cfg


def predict_docs(docs, probs, trans, lam, bias) -> dict[str, list[str]]:
    pred = {}
    for doc in docs:
        if doc.doc_id not in probs:
            continue
        path = viterbi(probs[doc.doc_id], trans[doc.domain], bias, lam[doc.domain])
        pred[doc.doc_id] = [LABELS[int(i)] for i in path]
    return pred


def bias_array(cfg: dict) -> np.ndarray:
    return np.array([cfg["bias"][name] for name in LABELS], dtype=np.float64)


def exact_prior(train_docs) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for doc in train_docs:
        if doc.labels is None:
            continue
        for sent, lab in zip(doc.sentences, doc.labels):
            counts.setdefault(sent, {})
            counts[sent][lab] = counts[sent].get(lab, 0) + 1
    prior = {}
    for sent, hist in counts.items():
        total = sum(hist.values())
        lab, n = max(hist.items(), key=lambda kv: kv[1])
        if total >= 2 and n / total >= 0.9:
            prior[sent] = lab
    return prior


def apply_exact(docs, pred: dict[str, list[str]], prior: dict[str, str]) -> dict[str, list[str]]:
    out = {}
    for doc in docs:
        if doc.doc_id not in pred:
            continue
        labels = list(pred[doc.doc_id])
        for i, sent in enumerate(doc.sentences):
            if sent in prior:
                labels[i] = prior[sent]
        out[doc.doc_id] = labels
    return out


def _stack_features(docs, probs) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    xs, ys, index = [], [], []
    for doc in docs:
        if doc.doc_id not in probs:
            continue
        p = probs[doc.doc_id]
        n = p.shape[0]
        dom = 1.0 if doc.domain == "IT" else 0.0
        for i in range(n):
            cols = [p[i]]
            for off in (-1, 1, -3, 3):
                j = min(max(i + off, 0), n - 1)
                cols.append(p[j])
            rel = i / max(n - 1, 1)
            xs.append(np.concatenate(cols + [np.array([rel, dom])]))
            if doc.labels is not None:
                ys.append(LABEL_ID[doc.labels[i]])
            index.append((doc.doc_id, i))
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros(0, dtype=np.int64), []
    return np.stack(xs), np.array(ys) if ys else np.zeros(0, dtype=np.int64), index


def fit_stacker(docs, probs, folds=None) -> dict | None:
    del folds
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    x, y, _ = _stack_features([d for d in docs if d.labels is not None], probs)
    if len(y) < 20:
        return None
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(x, y)
    return {"clf": clf}


def stack_predict(clf, docs, probs) -> dict[str, list[str]]:
    x, _, index = _stack_features(docs, probs)
    pred_ids = clf.predict(x)
    out: dict[str, list[str]] = {}
    for (doc_id, i), lab in zip(index, pred_ids):
        out.setdefault(doc_id, [])
        while len(out[doc_id]) <= i:
            out[doc_id].append(LABELS[0])
        out[doc_id][i] = LABELS[int(lab)]
    return out


def nested_stack_gain(docs, probs, folds, base_score: float) -> float:
    """Mean held-out fold macro-F1 of the stacker, minus the kept decoder's nested score.

    The decoder's nested number is also a mean of per-fold macro-F1, so the gain is in
    the same units. +1.0 in the plan is 0.01 here.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return -1.0
    scores = []
    for ids in folds:
        train = [d for d in docs if d.labels is not None and d.doc_id not in ids]
        held = [d for d in docs if d.labels is not None and d.doc_id in ids]
        if not train or not held:
            continue
        x, y, _ = _stack_features(train, probs)
        if len(y) < 20 or len(set(y.tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(x, y)
        got = stack_predict(clf, held, probs)
        gold, pred = [], []
        for doc in held:
            gold.extend(doc.label_ids)
            pred.extend(LABEL_ID[lab] for lab in got[doc.doc_id])
        scores.append(macro_scores(gold, pred)["macro_f1"])
    if not scores:
        return -1.0
    return float(np.mean(scores)) - base_score


def exact_oof(docs, pred: dict[str, list[str]]) -> dict:
    """Leave-one-document prior. The scored document does not vote for its own labels."""
    counts: dict[str, dict[str, int]] = {}
    for doc in docs:
        if doc.labels is None:
            continue
        for sent, lab in zip(doc.sentences, doc.labels):
            bucket = counts.setdefault(sent, {})
            bucket[lab] = bucket.get(lab, 0) + 1
    gold, base, after = [], [], []
    overridden = 0
    total = 0
    for doc in docs:
        if doc.doc_id not in pred or doc.labels is None:
            continue
        labels = list(pred[doc.doc_id])
        own: dict[str, dict[str, int]] = {}
        for sent, lab in zip(doc.sentences, doc.labels):
            own.setdefault(sent, {})
            own[sent][lab] = own[sent].get(lab, 0) + 1
        for i, sent in enumerate(doc.sentences):
            total += 1
            hist = dict(counts.get(sent, {}))
            for lab, n in own.get(sent, {}).items():
                hist[lab] = hist.get(lab, 0) - n
                if hist[lab] <= 0:
                    hist.pop(lab, None)
            total_n = sum(hist.values())
            if hist and total_n >= 2:
                chosen, n = max(hist.items(), key=lambda kv: kv[1])
                if n / total_n >= 0.9:
                    labels[i] = chosen
                    overridden += 1
        gold.extend(doc.label_ids)
        base.extend(LABEL_ID[x] for x in pred[doc.doc_id])
        after.extend(LABEL_ID[x] for x in labels)
    before = macro_scores(gold, base)["macro_f1"] if gold else 0.0
    after_s = macro_scores(gold, after)["macro_f1"] if gold else 0.0
    return {
        "before": before,
        "after": after_s,
        "rate": overridden / max(total, 1),
        "kept": after_s >= before - 1e-12,
    }


_CLOSING = re.compile(r"secretary is directed", re.I)
DISSENT_MIN_TAIL = 10


def dissent_start(doc) -> int | None:
    """Start of a separate member's order appended to a CCI order, or None.

    Both labelled dissents (CCI Sugar Mills, CCI Singhania) begin right after the
    majority's closing "Secretary is directed ..." line and run to the last
    sentence. No other labelled document has ten or more sentences after such a
    line. HC/SC/COMPAT judgments only cite dissents, so the rule is CCI-only.
    """
    if doc.domain != "CL" or not doc.doc_id.startswith("CCI"):
        return None
    n = len(doc.sentences)
    for i, sent in enumerate(doc.sentences):
        if _CLOSING.search(sent) and n - (i + 1) >= DISSENT_MIN_TAIL:
            return i + 1
    return None


def apply_dissent(docs, pred: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {}
    for doc in docs:
        if doc.doc_id not in pred:
            continue
        labels = list(pred[doc.doc_id])
        start = dissent_start(doc)
        if start is not None:
            labels[start:] = ["Dissent"] * (len(labels) - start)
        out[doc.doc_id] = labels
    return out


def dissent_oof(docs, pred: dict[str, list[str]]) -> dict:
    gold, before, after = [], [], []
    fired = []
    ruled = apply_dissent(docs, pred)
    for doc in docs:
        if doc.doc_id not in pred or doc.labels is None:
            continue
        gold.extend(doc.label_ids)
        before.extend(LABEL_ID[x] for x in pred[doc.doc_id])
        after.extend(LABEL_ID[x] for x in ruled[doc.doc_id])
        if dissent_start(doc) is not None:
            fired.append(doc.doc_id)
    b = macro_scores(gold, before)["macro_f1"] if gold else 0.0
    a = macro_scores(gold, after)["macro_f1"] if gold else 0.0
    return {"before": b, "after": a, "fired": fired, "kept": a >= b - 1e-12}


def save_cfg(exp: str, cfg: dict) -> str:
    path = os.path.join(p3_dir(), exp, "decode.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {k: v for k, v in cfg.items() if k != "clf"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    return path


def decode_experiment(exp: str, kind: str = "test", use_stack: bool = True, use_exact: bool = True,
                      use_dissent: bool = False) -> dict[str, list[str]]:
    docs = load_docs()
    labeled = labeled_docs(docs)
    cfg_path = os.path.join(p3_dir(), exp, "decode.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    else:
        cfg = fit_decoder(exp, labeled)
        save_cfg(exp, cfg)
    target = [d for d in docs if d.split == "test"] if kind == "test" else labeled
    probs = load_split_probs(exp, "oof" if kind == "oof" else "test")
    if kind == "test" and probs is None:
        raise FileNotFoundError(exp)
    if kind == "oof":
        probs = load_split_probs(exp, "oof")
    trans = transitions(labeled)
    lam = cfg["lambda"]
    bias = bias_array(cfg)
    pred = predict_docs(target, probs, trans, lam, bias)
    if use_stack and cfg.get("stack_kept"):
        oof = load_split_probs(exp, "oof")
        fitted = fit_stacker(labeled, oof) if oof else None
        if fitted is not None:
            pred = stack_predict(fitted["clf"], target, probs)
    if use_exact and cfg.get("exact_kept"):
        pred = apply_exact(target, pred, exact_prior(labeled))
    if use_dissent:
        pred = apply_dissent(target, pred)
    return pred
