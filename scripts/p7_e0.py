"""E0: Okapi BM25, citation-context BM25, IDF shingles, train-fit fusion, Sinkhorn."""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import join_text
from p7_common import OUT
from p7_common import (
    apply_mask,
    bm25_matrix,
    both_halves_at_least,
    check_submission,
    citation_contexts,
    citator_matrix,
    decode,
    fit_marker_mass,
    gold_of,
    half_scores,
    halves,
    masks,
    passages,
    reduce_passages,
    save_metrics,
    split_bundle,
    write_zip,
    zrows,
)

K1_GRID = (0.6, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.3, 0.5, 0.75, 0.9)
EPS_GRID = (0.1, 0.2, 0.5, 1.0, 2.0)
WEIGHT_STEP = 0.25


def first_sentences(sent_lists, n: int) -> list[str]:
    return [" ".join(sents[:n]) for sents in sent_lists]


def shingle_overlap(docs: list[str], queries: list[str]) -> np.ndarray:
    counter = CountVectorizer(
        lowercase=True, token_pattern=r"[a-z0-9]+", ngram_range=(8, 8),
        min_df=2, max_features=200_000, binary=True,
    )
    counted = counter.fit_transform(docs)
    query = counter.transform(queries)
    df = np.asarray(counted.sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((counted.shape[0] + 1.0) / (df + 1.0)).astype(np.float32)
    return (query.multiply(idf) @ counted.T).toarray()


def context_bm25(bundle, k1: float, b: float) -> np.ndarray:
    owners = []
    texts = []
    for cand_i, sents in enumerate(bundle["sents"]):
        for passage in passages(sents):
            owners.append(cand_i)
            texts.append(passage)
    owners = np.asarray(owners)
    queries = []
    spans = []
    for sents in bundle["q_sents"]:
        ctx = citation_contexts(sents) or ["context: " + " ".join(sents[:8])]
        spans.append((len(queries), len(queries) + len(ctx)))
        queries.extend(ctx)
    print(f"[e0] context bm25 passages {len(texts)} contexts {len(queries)}", flush=True)
    raw = bm25_matrix(texts, queries, k1, b, max_features=200_000)
    per_context = reduce_passages(raw, owners, len(bundle["cand"]), "max")
    out = np.zeros((len(bundle["qids"]), len(bundle["cand"])), dtype=np.float32)
    for row, (lo, hi) in enumerate(spans):
        out[row] = per_context[lo:hi].max(axis=0)
    return out


def grid_k1(bundle, gold, self_mask, after) -> tuple[float, float]:
    qtext = [join_text(bundle["queries"][q]["text"]) for q in bundle["qids"]]
    best_f1, best_k1, best_b = -1.0, 1.2, 0.75
    for k1, b in itertools.product(K1_GRID, B_GRID):
        scores = apply_mask(bm25_matrix(bundle["docs"], qtext, k1, b), self_mask, after)
        pred = decode(bundle["qids"], bundle["cand"], scores, "row")
        from p7_bm25 import micro_f1
        f1 = micro_f1(gold, pred, bundle["qids"])["f1"]
        print(f"[e0] k1 {k1} b {b} train f1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_k1, best_b = f1, k1, b
    print(f"[e0] chosen k1 {best_k1} b {best_b} train {best_f1:.4f}", flush=True)
    return best_k1, best_b


def features_of(bundle, k1, b) -> dict[str, np.ndarray]:
    qids = bundle["qids"]
    qtext = [join_text(bundle["queries"][q]["text"]) for q in qids]
    print(f"[e0] full bm25 {bundle['split']}", flush=True)
    full = bm25_matrix(bundle["docs"], qtext, k1, b)
    print(f"[e0] head40 bm25 {bundle['split']}", flush=True)
    head = bm25_matrix(first_sentences(bundle["sents"], 40), first_sentences(bundle["q_sents"], 40), k1, b)
    ctx = context_bm25(bundle, k1, b)
    print(f"[e0] shingles {bundle['split']}", flush=True)
    shingles = shingle_overlap(bundle["docs"], qtext)
    citator = citator_matrix(qtext, qids, bundle["docs"], bundle["cand"])
    return {"full": full, "head": head, "context": ctx, "shingle": shingles, "citator": citator}


def weight_grid(step: float, n: int):
    units = int(round(1.0 / step))
    for combo in itertools.product(range(units + 1), repeat=n):
        if sum(combo) == units:
            yield tuple(c * step for c in combo)


def fuse(parts: list[np.ndarray], weights: tuple[float, ...], self_mask, after) -> np.ndarray:
    score = sum(w * zrows(part) for w, part in zip(weights, parts))
    return apply_mask(score, self_mask, after)


def marker_counts(bundle) -> np.ndarray:
    return np.array([len(citation_contexts(sents)) or 1 for sents in bundle["q_sents"]], dtype=np.float64)


def fit_decoder(train_scores, train, gold):
    counts = marker_counts(train)
    gold_counts = np.array([len(gold[q]) for q in train["qids"]], dtype=np.float64)
    coef = fit_marker_mass(counts, gold_counts)
    best = (-1.0, 0.5, "column")
    from p7_bm25 import micro_f1
    for kind, eps in [("column", 0.5), ("sinkhorn", None)]:
        if kind == "column":
            pred = decode(train["qids"], train["cand"], train_scores, "column", tau=0.5)
            f1 = micro_f1(gold, pred, train["qids"])["f1"]
            print(f"[e0] train column f1 {f1:.4f}", flush=True)
            if f1 > best[0]:
                best = (f1, 0.5, "column")
        else:
            for eps in EPS_GRID:
                pred = decode(train["qids"], train["cand"], train_scores, "sinkhorn", counts, coef, eps)
                f1 = micro_f1(gold, pred, train["qids"])["f1"]
                print(f"[e0] train sinkhorn eps {eps} f1 {f1:.4f}", flush=True)
                if f1 > best[0]:
                    best = (f1, eps, "sinkhorn")
    return coef, best[1], best[2]


def main() -> None:
    train = split_bundle("Train")
    dev = split_bundle("Dev")
    test = split_bundle("Test")
    gold = gold_of(train["queries"])
    dev_gold = gold_of(dev["queries"])
    self_mask, after = masks(train["qids"], train["cand"], train["q_sents"], train["sents"])
    k1, b = grid_k1(train, gold, self_mask, after)
    train_f = features_of(train, k1, b)
    dev_f = features_of(dev, k1, b)
    names = ["full", "head", "context", "shingle"]
    os.makedirs(os.path.join(OUT, "e0"), exist_ok=True)
    for tag, bundle, feats in (("train", train, train_f), ("dev", dev, dev_f)):
        np.savez(
            os.path.join(OUT, "e0", f"{tag}.npz"),
            qids=np.asarray(bundle["qids"]), cand=np.asarray(bundle["cand"]),
            **{name: feats[name].astype(np.float32) for name in names + ["citator"]},
        )
    os.makedirs(os.path.join(OUT, "e0"), exist_ok=True)
    for tag, bundle, feats in (("train", train, train_f), ("dev", dev, dev_f)):
        np.savez(
            os.path.join(OUT, "e0", f"{tag}.npz"),
            qids=np.asarray(bundle["qids"]), cand=np.asarray(bundle["cand"]),
            **{name: feats[name].astype(np.float32) for name in names + ["citator"]},
        )
    dev_mask = masks(dev["qids"], dev["cand"], dev["q_sents"], dev["sents"])
    best = None
    for weights in weight_grid(WEIGHT_STEP, len(names)):
        fused = fuse([train_f[n] for n in names], weights, self_mask, after)
        pred = decode(train["qids"], train["cand"], fused, "row")
        from p7_bm25 import micro_f1
        f1 = micro_f1(gold, pred, train["qids"])["f1"]
        if best is None or f1 > best[0]:
            best = (f1, weights, fused)
            print(f"[e0] weights {weights} train {f1:.4f}", flush=True)
    print(f"[e0] best weights {best[1]} train {best[0]:.4f}", flush=True)
    coef, eps, kind = fit_decoder(best[2], train, gold)
    dev_counts = marker_counts(dev)
    dev_score = fuse([dev_f[n] for n in names], best[1], *dev_mask)
    dev_pred = decode(dev["qids"], dev["cand"], dev_score, kind, dev_counts, coef, eps, 0.5)
    report = half_scores(dev_gold, dev_pred, dev["qids"])
    cit_weights = tuple(w * 0.75 for w in best[1]) + (0.25,)
    cit = fuse([dev_f[n] for n in names] + [dev_f["citator"]], cit_weights, *dev_mask)
    cit_pred = decode(dev["qids"], dev["cand"], cit, kind, dev_counts, coef, eps, 0.5)
    cit_report = half_scores(dev_gold, cit_pred, dev["qids"])
    # structural reference: full+head fusion used by the uploaded zips, column softmax
    ref = fuse([dev_f["full"], dev_f["head"]], (0.75, 0.25), *dev_mask)
    ref_pred = decode(dev["qids"], dev["cand"], ref, "column", tau=0.5)
    ref_report = half_scores(dev_gold, ref_pred, dev["qids"])
    payload = {
        "k1": k1, "b": b, "weights": best[1], "decoder": kind, "eps": eps,
        "mass_coef": coef, "dev": report, "dev_citator": cit_report, "dev_struct_ref": ref_report,
        "gate": both_halves_at_least(report, 0.47),
    }
    path = save_metrics("e0", payload)
    print("[e0] dev", {k: round(report[k]["f1"], 4) for k in ("all", "A", "B")}, "gate", payload["gate"], flush=True)
    print("[e0] citator", {k: round(cit_report[k]["f1"], 4) for k in ("all", "A", "B")}, flush=True)
    print("[e0] struct ref", {k: round(ref_report[k]["f1"], 4) for k in ("all", "A", "B")}, flush=True)
    print("[e0] metrics", path, flush=True)
    test_f = features_of(test, k1, b)
    np.savez(
        os.path.join(OUT, "e0", "test.npz"),
        qids=np.asarray(test["qids"]), cand=np.asarray(test["cand"]),
        **{name: test_f[name].astype(np.float32) for name in names + ["citator"]},
    )
    if not payload["gate"]:
        print("[e0] gate failed, no zip", flush=True)
        return
    test_mask = masks(test["qids"], test["cand"], test["q_sents"], test["sents"])
    test_score = fuse([test_f[n] for n in names], best[1], *test_mask)
    test_pred = decode(test["qids"], test["cand"], test_score, kind, marker_counts(test), coef, eps, 0.5)
    check_submission(test_pred, test["qids"], test["cand"])
    write_zip("P7_E0", test_pred)
    print("[e0] wrote P7_E0", flush=True)


if __name__ == "__main__":
    main()
