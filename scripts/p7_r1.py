"""R1: n-gram Okapi backbone. k1, b, fusion weights, and the decoder are fit on train.

Saves p7/r1/{train,dev,test}.npz for the ranker. Writes a zip only when the dev gate
in the plan passes: +0.02 all-dev over the reference, and neither half worse by more than 0.01.
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import micro_f1
from p7_common import (
    OUT,
    apply_mask,
    citation_contexts,
    decode,
    fit_marker_mass,
    gold_of,
    half_scores,
    masks,
    save_metrics,
    split_bundle,
    write_zip,
    year_of,
    years,
    zrows,
)
from p7_e0 import marker_counts, weight_grid
from p7_probe6 import STATUTE, ACT

K1_GRID = (1.5, 2.0, 3.0, 5.0)
B_GRID = (0.75, 0.9, 1.0)
EPS_GRID = (0.1, 0.2, 0.5, 1.0, 2.0)
TAU_GRID = (0.2, 0.5, 1.0)
N_FEATURES = 2 ** 22


def hashing(n: int) -> HashingVectorizer:
    return HashingVectorizer(
        ngram_range=(n, n), n_features=N_FEATURES, alternate_sign=False, norm=None,
        lowercase=True, token_pattern=r"[a-z0-9]+",
    )


def doc_stats(vec: HashingVectorizer, docs: list[str]):
    counts = vec.transform(docs).tocsr().astype(np.float32)
    n = counts.shape[0]
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    length = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    return counts, idf, length


def query_counts(vec: HashingVectorizer, queries: list[str]):
    query = vec.transform(queries).tocsr().astype(np.float32)
    if query.nnz:
        query.data[:] = 1.0
    return query


def okapi(counts, idf, length, query, k1: float, b: float) -> np.ndarray:
    average = max(float(length.mean()) if len(length) else 1.0, 1.0)
    norm = k1 * (1.0 - b + b * length / average)
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    data = counts.data * (k1 + 1.0) / (counts.data + norm[rows]) * idf[counts.indices]
    weighted = csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)
    return (query @ weighted.T).toarray().astype(np.float32)


def max_pool(raw: np.ndarray, spans: list[tuple[int, int]], n_cand: int) -> np.ndarray:
    out = np.zeros((len(spans), n_cand), dtype=np.float32)
    for row, (lo, hi) in enumerate(spans):
        out[row] = raw[lo:hi].max(axis=0) if hi > lo else 0.0
    return out


def contexts_of(sent_lists) -> tuple[list[str], list[tuple[int, int]]]:
    queries: list[str] = []
    spans = []
    for sents in sent_lists:
        ctx = citation_contexts(sents) or [" ".join(sents[:8])]
        spans.append((len(queries), len(queries) + len(ctx)))
        queries.extend(ctx)
    return queries, spans


def statutes_of(text: str) -> set[str]:
    found = {m.group(1).replace(" ", "").lower() for m in STATUTE.finditer(text)}
    found |= {m.group(1).lower() for m in ACT.finditer(text)}
    return found


def jaccard(left: list[set[str]], right: list[set[str]]) -> np.ndarray:
    out = np.zeros((len(left), len(right)), dtype=np.float32)
    for i, a in enumerate(left):
        if not a:
            continue
        for j, b in enumerate(right):
            if b:
                out[i, j] = len(a & b) / len(a | b)
    return out


def prepare(split: str) -> dict:
    bundle = split_bundle(split)
    self_mask, after = masks(bundle["qids"], bundle["cand"], bundle["q_sents"], bundle["sents"])
    q_text = [" ".join(s) for s in bundle["q_sents"]]
    c_text = list(bundle["docs"])
    q_head = [" ".join(s[:40]) for s in bundle["q_sents"]]
    c_head = [" ".join(s[:40]) for s in bundle["sents"]]
    ctx, spans = contexts_of(bundle["q_sents"])
    cy = years(bundle["sents"]).astype(np.float64)
    cy = np.where(cy > 0, cy, 0.0)
    qy = np.array([year_of(sents)[0] or 0 for sents in bundle["q_sents"]], dtype=np.float64)
    known = (cy[None, :] > 0) & (qy[:, None] > 0)
    gap = np.where(known, qy[:, None] - cy[None, :], 0).astype(np.float32)
    pack = {
        "split": split,
        "qids": bundle["qids"],
        "cand": bundle["cand"],
        "self_mask": self_mask,
        "after": after,
        "q_text": q_text,
        "c_text": c_text,
        "q_head": q_head,
        "c_head": c_head,
        "ctx": ctx,
        "spans": spans,
        "year_gap": gap,
        "marker_counts": marker_counts(bundle),
        "q_statute": [statutes_of(t) for t in q_text],
        "c_statute": [statutes_of(t) for t in c_text],
        "gold": gold_of(bundle["queries"]) if split != "Test" else {},
    }
    print(
        f"[r1] {split} queries {len(bundle['qids'])} pool {len(bundle['cand'])} contexts {len(ctx)}",
        flush=True,
    )
    return pack


def align_e0(split: str, qids, cand) -> dict[str, np.ndarray]:
    stored = np.load(os.path.join(OUT, "e0", f"{split}.npz"))
    sq = [str(x) for x in stored["qids"]]
    sc = [str(x) for x in stored["cand"]]
    same = sq == [str(q) for q in qids] and sc == [str(c) for c in cand]
    out = {}
    for name in ("full", "head", "context", "shingle", "citator"):
        mat = stored[name]
        if same:
            out[name] = mat
        else:
            qi = {q: i for i, q in enumerate(sq)}
            ci = {c: i for i, c in enumerate(sc)}
            out[name] = mat[np.ix_([qi[str(q)] for q in qids], [ci[str(c)] for c in cand])]
    return out


def sweep_full(pack, n: int, gold) -> tuple[float, float, np.ndarray, float]:
    vec = hashing(n)
    print(f"[r1] {pack['split']} full {n}-gram counts", flush=True)
    counts, idf, length = doc_stats(vec, pack["c_text"])
    query = query_counts(vec, pack["q_text"])
    best = (-1.0, 2.0, 0.9, None)
    for k1, b in itertools.product(K1_GRID, B_GRID):
        raw = okapi(counts, idf, length, query, k1, b)
        pred = decode(pack["qids"], pack["cand"], apply_mask(raw, pack["self_mask"], pack["after"]), "row")
        f1 = micro_f1(gold, pred, pack["qids"])["f1"]
        print(f"[r1] full {n}g k1 {k1} b {b} train {f1:.4f}", flush=True)
        if f1 > best[0]:
            best = (f1, k1, b, raw)
    print(f"[r1] full {n}g chosen k1 {best[1]} b {best[2]} train {best[0]:.4f}", flush=True)
    return best[1], best[2], best[3], best[0]


def score_full(pack, n: int, k1: float, b: float) -> np.ndarray:
    vec = hashing(n)
    print(f"[r1] {pack['split']} full {n}-gram k1 {k1} b {b}", flush=True)
    counts, idf, length = doc_stats(vec, pack["c_text"])
    return okapi(counts, idf, length, query_counts(vec, pack["q_text"]), k1, b)


def score_head(pack, n: int, k1: float, b: float) -> np.ndarray:
    vec = hashing(n)
    print(f"[r1] {pack['split']} head40 {n}-gram", flush=True)
    counts, idf, length = doc_stats(vec, pack["c_head"])
    return okapi(counts, idf, length, query_counts(vec, pack["q_head"]), k1, b)


def score_ctx(pack, n: int, k1: float, b: float) -> np.ndarray:
    vec = hashing(n)
    print(f"[r1] {pack['split']} context {n}-gram contexts {len(pack['ctx'])}", flush=True)
    counts, idf, length = doc_stats(vec, pack["c_text"])
    raw = okapi(counts, idf, length, query_counts(vec, pack["ctx"]), k1, b)
    return max_pool(raw, pack["spans"], len(pack["cand"]))


def train_f1(pack, gold, scores, kind, counts, coef, eps, tau) -> float:
    pred = decode(pack["qids"], pack["cand"], scores, kind, counts, coef, eps, tau)
    return micro_f1(gold, pred, pack["qids"])["f1"]


def search_weights(parts: list[np.ndarray], pack, gold, label: str):
    zipped = [zrows(p) for p in parts]
    counts = pack["marker_counts"]
    ranked = []
    best_row = -1.0
    for weights in weight_grid(0.25, len(parts)):
        fused = apply_mask(sum(w * z for w, z in zip(weights, zipped) if w), pack["self_mask"], pack["after"])
        f1 = train_f1(pack, gold, fused, "row", counts, (0.0, 1.0), 0.5, 0.5)
        ranked.append((f1, weights))
        if f1 > best_row:
            best_row = f1
            print(f"[r1] {label} weights {weights} train-row {f1:.4f}", flush=True)
    ranked.sort(key=lambda item: item[0], reverse=True)
    coef = fit_marker_mass(counts, np.array([len(gold[q]) for q in pack["qids"]], dtype=np.float64))
    best = (-1.0, None, None, None, None, None)

    def fused_of(weights):
        return apply_mask(sum(w * z for w, z in zip(weights, zipped) if w), pack["self_mask"], pack["after"])

    for _, weights in ranked[:15]:
        fused = fused_of(weights)
        for tau in TAU_GRID:
            f1 = train_f1(pack, gold, fused, "column", counts, coef, 0.5, tau)
            if f1 > best[0]:
                best = (f1, weights, fused, "column", tau, coef)
                print(f"[r1] {label} column tau {tau} {weights} train {f1:.4f}", flush=True)
        for eps in EPS_GRID:
            f1 = train_f1(pack, gold, fused, "sinkhorn", counts, coef, eps, 0.5)
            if f1 > best[0]:
                best = (f1, weights, fused, "sinkhorn", eps, coef)
                print(f"[r1] {label} sinkhorn eps {eps} {weights} train {f1:.4f}", flush=True)
    print(f"[r1] {label} chosen {best[3]} {best[4]} weights {best[1]} train {best[0]:.4f}", flush=True)
    return best


def apply_choice(parts, pack, choice) -> np.ndarray:
    _, weights, _, kind, knob, coef = choice
    fused = apply_mask(sum(w * zrows(p) for w, p in zip(weights, parts)), pack["self_mask"], pack["after"])
    return fused, kind, knob, coef


def decode_choice(pack, fused, kind, knob, coef):
    tau = knob if kind == "column" else 0.5
    eps = knob if kind == "sinkhorn" else 0.5
    return decode(pack["qids"], pack["cand"], fused, kind, pack["marker_counts"], coef, eps, tau)


def gate(report: dict, ref: dict) -> bool:
    return (
        report["all"]["f1"] >= ref["all"]["f1"] + 0.02
        and report["A"]["f1"] >= ref["A"]["f1"] - 0.01
        and report["B"]["f1"] >= ref["B"]["f1"] - 0.01
    )


def named_zip(name: str, pred: dict) -> str:
    path = write_zip(name, pred)
    outer = os.path.join(os.path.dirname(os.path.dirname(path)), f"{name}.zip")
    shutil.copy(path, outer)
    print(f"[r1] wrote {path} and {outer}", flush=True)
    return outer


def save_views(split: str, pack, views: dict) -> None:
    dest = os.path.join(OUT, "r1")
    os.makedirs(dest, exist_ok=True)
    np.savez(
        os.path.join(dest, f"{split}.npz"),
        qids=np.asarray(pack["qids"]), cand=np.asarray(pack["cand"]),
        **{k: v.astype(np.float32) for k, v in views.items()},
    )


def main() -> None:
    e0_meta = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    ref = e0_meta["dev"]
    ref_cit = e0_meta["dev_citator"]
    train = prepare("Train")
    dev = prepare("Dev")
    gold = train["gold"]
    dev_gold = dev["gold"]
    params = {}
    train_full = {}
    for n in (4, 5, 3):
        k1, b, raw, f1 = sweep_full(train, n, gold)
        params[f"full{n}"] = {"k1": k1, "b": b, "train_row": f1}
        train_full[f"full{n}"] = raw
    dev_full = {f"full{n}": score_full(dev, n, params[f"full{n}"]["k1"], params[f"full{n}"]["b"]) for n in (3, 4, 5)}
    train_e0 = align_e0("train", train["qids"], train["cand"])
    dev_e0 = align_e0("dev", dev["qids"], dev["cand"])
    names = ["full", "head", "context", "shingle", "full4", "full5"]
    train_parts = [train_e0["full"], train_e0["head"], train_e0["context"], train_e0["shingle"], train_full["full4"], train_full["full5"]]
    dev_parts = [dev_e0["full"], dev_e0["head"], dev_e0["context"], dev_e0["shingle"], dev_full["full4"], dev_full["full5"]]
    lexical = search_weights(train_parts, train, gold, "lexical")
    citator = search_weights(train_parts + [train_e0["citator"]], train, gold, "citator")

    dev_fused, kind, knob, coef = apply_choice(dev_parts, dev, lexical)
    dev_pred = decode_choice(dev, dev_fused, kind, knob, coef)
    report = half_scores(dev_gold, dev_pred, dev["qids"])
    dev_cit, ckind, cknob, ccoef = apply_choice(dev_parts + [dev_e0["citator"]], dev, citator)
    cit_pred = decode_choice(dev, dev_cit, ckind, cknob, ccoef)
    cit_report = half_scores(dev_gold, cit_pred, dev["qids"])
    ok = gate(report, ref)
    ok_cit = gate(cit_report, ref_cit)
    print("[r1] dev", {k: round(report[k]["f1"], 4) for k in ("all", "A", "B")}, "gate", ok, flush=True)
    print("[r1] citator", {k: round(cit_report[k]["f1"], 4) for k in ("all", "A", "B")}, "gate", ok_cit, flush=True)

    test = prepare("Test")
    test_full = {f"full{n}": score_full(test, n, params[f"full{n}"]["k1"], params[f"full{n}"]["b"]) for n in (3, 4, 5)}
    test_e0 = align_e0("test", test["qids"], test["cand"])
    test_parts = [test_e0["full"], test_e0["head"], test_e0["context"], test_e0["shingle"], test_full["full4"], test_full["full5"]]
    if ok:
        fused, kind, knob, coef = apply_choice(test_parts, test, lexical)
        pred = decode_choice(test, fused, kind, knob, coef)
        from p7_common import check_submission
        check_submission(pred, test["qids"], test["cand"])
        named_zip("P7_R1_ngram", pred)
    else:
        print("[r1] lexical gate failed, no zip", flush=True)
    if ok_cit:
        fused, kind, knob, coef = apply_choice(test_parts + [test_e0["citator"]], test, citator)
        pred = decode_choice(test, fused, kind, knob, coef)
        from p7_common import check_submission
        check_submission(pred, test["qids"], test["cand"])
        named_zip("P7_R1_ngram_citator", pred)
    else:
        print("[r1] citator gate failed, no zip", flush=True)

    # Extra views for the ranker. Head and context use the full-4 parameters.
    k1, b = params["full4"]["k1"], params["full4"]["b"]
    for pack, full, e0 in (
        (train, train_full, train_e0),
        (dev, dev_full, dev_e0),
        (test, test_full, test_e0),
    ):
        views = dict(full)
        views.update(e0)
        views["year_gap"] = pack["year_gap"]
        views["statute"] = jaccard(pack["q_statute"], pack["c_statute"])
        for n in (1, 3, 4):
            views[f"head{n}"] = score_head(pack, n, k1, b)
        for n in (1, 4):
            views[f"ctx{n}"] = score_ctx(pack, n, k1, b)
        save_views({"Train": "train", "Dev": "dev", "Test": "test"}[pack["split"]], pack, views)
        print(f"[r1] saved {pack['split']} views {sorted(views)}", flush=True)

    payload = {
        "params": params,
        "lexical": {"weights": lexical[1], "features": names, "decoder": lexical[3], "knob": lexical[4], "mass_coef": lexical[5], "train": lexical[0]},
        "citator": {"weights": citator[1], "decoder": citator[3], "knob": citator[4], "mass_coef": citator[5], "train": citator[0]},
        "dev": report,
        "dev_citator": cit_report,
        "gate": ok,
        "gate_citator": ok_cit,
        "ref_dev": {k: ref[k]["f1"] for k in ("all", "A", "B")},
        "ref_citator": {k: ref_cit[k]["f1"] for k in ("all", "A", "B")},
    }
    path = save_metrics("r1", payload)
    print("[r1] metrics", path, flush=True)


if __name__ == "__main__":
    main()
