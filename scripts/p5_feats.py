"""Retrieval and rule members for the Phase 5 stack. CPU only.

R1 kNN label vote, R2 surviving section numbers, R3 exact-duplicate labels. Dev half A only sees train +
dev half B (and the other way round); test sees train + all of dev.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import p5_dir
from saul.lsi.const import N_LABELS
from saul.lsi.data import dev_half_array, text_hash
from saul.lsi.decide import quick_macro
from scripts.p5_tfidf import _labels, _read

K = 30
TAU = 0.05
SEC = re.compile(r"(?:[Ss]ections?|SECTIONS?|[Ss]ecs?\.|[Uu]/[Ss]s?\.?|[Ss]s\.)\s*"
                 r"((?:\d{1,3}\s*-?\s*[A-Z]{0,2}\b[\s,/&]*(?:and|r/w|read with|&|,|/)?\s*)+)")
NUM = re.compile(r"(\d{1,3})\s*-?\s*([A-Z]{0,2})\b")


def section_numbers(text: str) -> set[str]:
    out = set()
    for match in SEC.finditer(text):
        for number, suffix in NUM.findall(match.group(1)):
            out.add(number + suffix)
    return out


def knn_vote(query: sparse.csr_matrix, pool: sparse.csr_matrix, pool_y: np.ndarray, exclude_self=None) -> np.ndarray:
    out = np.zeros((query.shape[0], N_LABELS), dtype=np.float32)
    pool_t = pool.T.tocsr()
    for start in range(0, query.shape[0], 1000):
        sims = (query[start:start + 1000] @ pool_t).toarray()
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        top_s = np.take_along_axis(sims, top, 1)
        weights = np.exp((top_s - top_s.max(1, keepdims=True)) / TAU)
        weights /= weights.sum(1, keepdims=True)
        out[start:start + 1000] = np.einsum("nk,nkl->nl", weights, pool_y[top])
    return out


def section_table(texts: list[str], y: np.ndarray, alpha: float = 5.0, min_count: int = 5) -> dict:
    prior = y.mean(0)
    counts = collections.Counter()
    joint = collections.defaultdict(lambda: np.zeros(N_LABELS))
    for text, row in zip(texts, y):
        for number in section_numbers(text):
            counts[number] += 1
            joint[number] += row
    return {n: (joint[n] + alpha * prior) / (c + alpha) for n, c in counts.items() if c >= min_count}


def section_feature(texts: list[str], table: dict) -> np.ndarray:
    out = np.zeros((len(texts), N_LABELS), dtype=np.float32)
    for i, text in enumerate(texts):
        rows = [table[n] for n in section_numbers(text) if n in table]
        if rows:
            out[i] = np.max(rows, axis=0)
    return out


def duplicate_feature(query: list[dict], pool: list[dict], pool_y: np.ndarray) -> np.ndarray:
    """Mean gold vector over exact-text copies; column N_LABELS is the hit flag."""
    table = collections.defaultdict(list)
    for row, y in zip(pool, pool_y):
        table[text_hash(row["text"])].append(y)
    out = np.zeros((len(query), N_LABELS + 1), dtype=np.float32)
    for i, row in enumerate(query):
        hits = table.get(text_hash(row["text"]))
        if hits:
            out[i, :N_LABELS] = np.mean(hits, axis=0)
            out[i, N_LABELS] = 1.0
    return out


def main() -> None:
    feats = os.path.join(p5_dir(), "feats")
    train, dev, test = _read("train"), _read("dev"), _read("test")
    ytr, ydv = _labels(train), _labels(dev)
    half = dev_half_array([row["id"] for row in dev])
    w = {name: sparse.load_npz(os.path.join(feats, f"word_{name}.npz")).tocsr() for name in ("train", "dev", "test")}
    knn_dev = np.zeros_like(ydv)
    sec_dev = np.zeros_like(ydv)
    dup_dev = np.zeros((len(dev), N_LABELS + 1), dtype=np.float32)
    for name in ("A", "B"):
        rows = np.flatnonzero(half == name)
        other = np.flatnonzero(half != name)
        pool = sparse.vstack([w["train"], w["dev"][other]]).tocsr()
        pool_y = np.vstack([ytr, ydv[other]])
        knn_dev[rows] = knn_vote(w["dev"][rows], pool, pool_y)
        pool_rows = train + [dev[i] for i in other]
        table = section_table([r["text"] for r in pool_rows], pool_y)
        sec_dev[rows] = section_feature([dev[i]["text"] for i in rows], table)
        dup_dev[rows] = duplicate_feature([dev[i] for i in rows], pool_rows, pool_y)
        print(f"[feats] dev half {name} done", flush=True)
    pool = sparse.vstack([w["train"], w["dev"]]).tocsr()
    pool_y = np.vstack([ytr, ydv])
    knn_test = knn_vote(w["test"], pool, pool_y)
    table = section_table([r["text"] for r in train + dev], pool_y)
    sec_test = section_feature([r["text"] for r in test], table)
    dup_test = duplicate_feature(test, train + dev, pool_y)
    np.savez(os.path.join(feats, "rules.npz"), knn_dev=knn_dev, knn_test=knn_test, sec_dev=sec_dev,
             sec_test=sec_test, dup_dev=dup_dev, dup_test=dup_test,
             dev_ids=np.asarray([r["id"] for r in dev]), test_ids=np.asarray([r["id"] for r in test]))
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    report = {
        "knn_alone": quick_macro(logit(knn_dev), ydv, half)["macro_f1"],
        "dev_with_section_number": float((sec_dev.max(1) > 0).mean()),
        "test_with_section_number": float((sec_test.max(1) > 0).mean()),
        "dev_duplicates": int(dup_dev[:, -1].sum()),
        "test_duplicates": int(dup_test[:, -1].sum()),
    }
    with open(os.path.join(feats, "rules.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
