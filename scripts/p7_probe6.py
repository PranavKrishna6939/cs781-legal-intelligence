"""Dev probes for the plan revision: n-gram BM25, pseudo-relevance feedback, statutes, train co-citation."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import join_text, micro_f1
from p7_common import OUT, apply_mask, decode, gold_of, halves, masks, split_bundle, zrows

STATUTE = re.compile(
    r"\b(?:section|sec\.|s\.|article|art\.|order|rule)\s*(\d+[a-z]?(?:\s*\(\s*\d+\s*\))?)", re.I,
)
ACT = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5}\s+(?:Act|Code))\b")


def bm25_counts(docs, queries, ngram, k1=1.5, b=0.75):
    vec = HashingVectorizer(
        ngram_range=(ngram, ngram), n_features=2 ** 22, alternate_sign=False, norm=None,
        lowercase=True, token_pattern=r"[a-z0-9]+",
    )
    counts = vec.transform(docs).tocsr()
    query = vec.transform(queries).tocsr()
    query.data[:] = 1.0
    n = counts.shape[0]
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    length = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    norm = k1 * (1 - b + b * length / max(length.mean(), 1.0))
    weighted = counts.astype(np.float32).copy()
    rows = np.repeat(np.arange(n), np.diff(weighted.indptr))
    weighted.data = weighted.data * (k1 + 1) / (weighted.data + norm[rows])
    weighted = weighted.multiply(idf).tocsr()
    return (query @ weighted.T).toarray()


def statutes(text: str) -> set:
    found = {m.group(1).replace(" ", "").lower() for m in STATUTE.finditer(text)}
    found |= {m.group(1).lower() for m in ACT.finditer(text)}
    return found


def evaluate(name, score, dev, gold, self_mask, after, meta, out):
    a_ids, b_ids = halves(dev["qids"])
    masked = apply_mask(score, self_mask, after)
    for kind in ("row", "column", "sinkhorn"):
        pred = decode(dev["qids"], dev["cand"], masked, kind, None, tuple(meta["mass_coef"]), meta["eps"], 0.5)
        fa = micro_f1(gold, pred, [q for q in dev["qids"] if q in a_ids])["f1"]
        fb = micro_f1(gold, pred, [q for q in dev["qids"] if q in b_ids])["f1"]
        fall = micro_f1(gold, pred, dev["qids"])["f1"]
        out[f"{name}|{kind}"] = {"all": round(fall, 4), "A": round(fa, 4), "B": round(fb, 4)}
        print(f"{name:28s} {kind:8s} all {fall:.4f} A {fa:.4f} B {fb:.4f}", flush=True)


def main():
    meta = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    dev = split_bundle("Dev")
    gold = gold_of(dev["queries"])
    self_mask, after = masks(dev["qids"], dev["cand"], dev["q_sents"], dev["sents"])
    e0 = np.load(os.path.join(OUT, "e0", "dev.npz"))
    names = ["full", "head", "context", "shingle"]
    e0_score = sum(w * zrows(e0[n]) for w, n in zip(meta["weights"], names))
    out = {}
    evaluate("E0", e0_score, dev, gold, self_mask, after, meta, out)

    qfull = [join_text(dev["queries"][q]["text"]) for q in dev["qids"]]
    qhead = [" ".join(s[:40]) for s in dev["q_sents"]]
    cfull = dev["docs"]
    chead = [" ".join(s[:40]) for s in dev["sents"]]
    grams = {}
    for n in (2, 3, 4):
        grams[f"full{n}"] = bm25_counts(cfull, qfull, n)
        grams[f"head{n}"] = bm25_counts(chead, qhead, n)
        evaluate(f"bm25 full {n}-gram", grams[f"full{n}"], dev, gold, self_mask, after, meta, out)
        evaluate(f"bm25 head40 {n}-gram", grams[f"head{n}"], dev, gold, self_mask, after, meta, out)
    ngram_mix = sum(zrows(v) for v in grams.values()) / len(grams)
    evaluate("ngram mix", ngram_mix, dev, gold, self_mask, after, meta, out)
    evaluate("E0 + ngram mix", e0_score + ngram_mix, dev, gold, self_mask, after, meta, out)

    # Pseudo-relevance feedback in candidate space: Rocchio over the candidate-candidate
    # similarity of the top-k candidates by the current score.
    cand_head = bm25_counts(chead, chead, 3)
    np.fill_diagonal(cand_head, 0.0)
    cand_sim = zrows(cand_head)
    base = apply_mask(e0_score + ngram_mix, self_mask, after)
    for k in (3, 5):
        top = np.argsort(-base, axis=1)[:, :k]
        prf = np.stack([cand_sim[top[i]].mean(axis=0) for i in range(len(top))])
        for beta in (0.25, 0.5):
            evaluate(f"PRF k{k} b{beta}", e0_score + ngram_mix + beta * zrows(prf), dev, gold, self_mask, after, meta, out)

    qs = [statutes(t) for t in qfull]
    cs = [statutes(t) for t in cfull]
    jac = np.zeros((len(qs), len(cs)), dtype=np.float32)
    for i, a in enumerate(qs):
        if not a:
            continue
        for j, bset in enumerate(cs):
            if bset:
                jac[i, j] = len(a & bset) / len(a | bset)
    evaluate("statute jaccard", jac, dev, gold, self_mask, after, meta, out)
    evaluate("E0 + ngram + 0.25 statute", e0_score + ngram_mix + 0.25 * zrows(jac), dev, gold, self_mask, after, meta, out)

    train = split_bundle("Train")
    train_gold = gold_of(train["queries"])
    dev_pool = set(dev["cand"])
    co = defaultdict(set)
    for ids in train_gold.values():
        inside = [c for c in ids if c in dev_pool]
        for a in inside:
            for bb in inside:
                if a != bb:
                    co[a].add(bb)
    owner = defaultdict(set)
    for q, ids in gold.items():
        for c in ids:
            owner[c].add(q)
    pairs = same = 0
    for a, partners in co.items():
        for bb in partners:
            if owner[a] and owner[bb]:
                pairs += 1
                same += bool(owner[a] & owner[bb])
    test = split_bundle("Test")
    test_pool = set(test["cand"])
    test_edges = sum(
        1 for ids in train_gold.values()
        for a in ids for bb in ids if a != bb and a in test_pool and bb in test_pool
    )
    out["cocitation"] = {
        "dev_pool_docs_with_train_cocitation": len(co),
        "dev_pairs_both_gold": pairs,
        "dev_pairs_same_query": same,
        "precision": same / max(pairs, 1),
        "test_directed_edges": test_edges,
    }
    print("co-citation", out["cocitation"], flush=True)

    train_self, train_after = masks(train["qids"], train["cand"], train["q_sents"], train["sents"])
    tcand = {c: i for i, c in enumerate(train["cand"])}
    all_masked = 0
    for r, q in enumerate(train["qids"]):
        cols = [tcand[c] for c in train_gold[q] if c in tcand]
        if cols and all(train_after[r, c] for c in cols):
            all_masked += 1
    out["train_queries_all_gold_date_masked"] = all_masked
    print("train queries whose gold is all date-masked", all_masked, flush=True)
    json.dump(out, open(os.path.join(OUT, "probe6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
