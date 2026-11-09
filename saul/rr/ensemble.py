"""Equal-weight mixes. There is no weight search on the score path."""
from __future__ import annotations

import json
import os

import numpy as np

from saul.rr.data import LABEL_ID, labeled_docs, load_docs, p3_dir
from saul.rr.decode import average_marginals, load_split_probs
from saul.rr.score import macro_scores


def oof_argmax_f1(exp: str) -> float | None:
    table = load_split_probs(exp, "oof")
    if not table:
        return None
    docs = {d.doc_id: d for d in labeled_docs(load_docs())}
    gold, pred = [], []
    for doc_id, probs in table.items():
        doc = docs.get(doc_id)
        if doc is None or doc.labels is None:
            continue
        gold.extend(doc.label_ids)
        pred.extend(probs.argmax(1).tolist())
    if not gold:
        return None
    return macro_scores(gold, pred)["macro_f1"]


def members_within(exps: list[str], margin: float = 0.08) -> list[str]:
    scores = {e: oof_argmax_f1(e) for e in exps}
    known = {e: s for e, s in scores.items() if s is not None}
    if not known:
        return []
    best = max(known.values())
    return [e for e, s in known.items() if s >= best - margin]


def decoded_oof(exp: str) -> float | None:
    """Score of the decoder that would be submitted, else argmax of the cached marginals."""
    path = os.path.join(p3_dir(), exp, "decode.json")
    if os.path.exists(path):
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("kept") is not None:
            return float(blob["kept"])
        if blob.get("tuned") is not None and not blob.get("collapsed"):
            return float(blob["tuned"])
    return oof_argmax_f1(exp)


def g2_note(singles: list[str], mix_score: float) -> dict:
    best = max(decoded_oof(e) or 0.0 for e in singles)
    return {"best_single": best, "mix": mix_score, "miss": mix_score < best + 0.01}


def save_mix_probs(name: str, exps: list[str], kind: str = "test") -> dict[str, np.ndarray]:
    probs = average_marginals(exps, kind=kind)
    dest = os.path.join(p3_dir(), name, "fold0", "oof") if kind == "oof" else os.path.join(p3_dir(), name, kind)
    os.makedirs(dest, exist_ok=True)
    for doc_id, table in probs.items():
        weight = np.ones(table.shape[0], dtype=np.float32)
        np.savez_compressed(
            os.path.join(dest, f"{doc_id}.npz"),
            marginals=(table * weight[:, None]).astype(np.float32),
            weight=weight,
        )
    if kind == "test":
        with open(os.path.join(dest, "done.json"), "w", encoding="utf-8") as f:
            json.dump({"folds": 5, "members": exps, "weights": "equal"}, f)
    return probs


def label_ids(names: list[str]) -> list[int]:
    return [LABEL_ID[n] for n in names]
