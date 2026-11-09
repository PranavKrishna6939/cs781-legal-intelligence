"""Strict span scores. A hit requires the same start, end, and label."""
from __future__ import annotations

from saul.ner.labels import LABELS


def strict_scores(gold: dict[str, list], pred: dict[str, list], labels: list[str] | None = None) -> dict:
    labels = labels or LABELS
    stats = {lab: [0, 0, 0] for lab in labels}
    for did, spans in gold.items():
        g = {(int(a), int(b), c) for a, b, c in spans}
        p = {(int(a), int(b), c) for a, b, c in pred.get(did, [])}
        for lab in labels:
            gs = {x for x in g if x[2] == lab}
            ps = {x for x in p if x[2] == lab}
            stats[lab][0] += len(gs & ps)
            stats[lab][1] += len(ps - gs)
            stats[lab][2] += len(gs - ps)
    per = {}
    f1s, ps, rs = [], [], []
    for lab in labels:
        tp, fp, fn = stats[lab]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"p": prec, "r": rec, "f": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
        ps.append(prec)
        rs.append(rec)
    n = max(len(labels), 1)
    return {
        "macro_f1": sum(f1s) / n,
        "macro_p": sum(ps) / n,
        "macro_r": sum(rs) / n,
        "per_label": per,
    }


def gold_map(docs) -> dict[str, list]:
    return {d.doc_id: [(a, b, c) for a, b, c in d.gold] for d in docs}
