"""Shared units, masks, scores, and decoding for Saul-CiteRank.

Word budgets stand in for the plan's token budgets (192 / 96 / 160) at roughly
1.4 tokens per word, so lexical and neural code chunk the same sentences.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from p7_bm25 import K, load_pool, load_queries, micro_f1, write_zip
from p7_probe5 import cited_by, own_keys

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
OUT = os.path.join(ROOT, "p7")
WORD = re.compile(r"[a-z0-9]+", re.I)
YEAR = re.compile(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})")
ANY_YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")
HEADER_CITE = re.compile(r"CITATION\s*:\s*\d{4}")
MARKER = re.compile(r"\bCITATION\b")

PASSAGE_WORDS = 140
STRIDE_WORDS = 70
CONTEXT_WORDS = 120


def words_of(text: str) -> list[str]:
    return WORD.findall(text.lower())


def year_of(sents) -> tuple[int | None, str]:
    head = " ".join(sents[:30])
    match = YEAR.search(head)
    if match:
        return int(match.group(1)), "header"
    found = [int(y) for y in ANY_YEAR.findall(" ".join(sents))]
    return (max(found), "max") if found else (None, "none")


def marker_sentences(sents) -> list[int]:
    hits = []
    for i, sent in enumerate(sents):
        if not MARKER.search(sent):
            continue
        if i < 4 and HEADER_CITE.search(sent):
            continue
        hits.append(i)
    return hits


def citation_contexts(sents) -> list[str]:
    hits = marker_sentences(sents)
    windows = [(max(0, i - 2), min(len(sents), i + 2)) for i in hits]
    merged: list[list[int]] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out = []
    for lo, hi in merged:
        raw = " ".join(sents[lo:hi]).split()
        if len(raw) > CONTEXT_WORDS:
            lower = [tok.lower().strip(":,.") for tok in raw]
            pos = next((i for i, tok in enumerate(lower) if tok == "citation"), 0)
            start = max(0, min(pos - CONTEXT_WORDS // 2, len(raw) - CONTEXT_WORDS))
            raw = raw[start:start + CONTEXT_WORDS]
        out.append("context: " + " ".join(raw))
    return out


def headnote_text(sents) -> str | None:
    start = next((i for i, sent in enumerate(sents[:25]) if "HEADNOTE" in sent), None)
    if start is None:
        return None
    taken: list[str] = []
    for sent in sents[start:start + 8]:
        taken.extend(sent.split())
        if len(taken) >= PASSAGE_WORDS:
            break
    return "headnote: " + " ".join(taken[:PASSAGE_WORDS])


def passages(sents) -> list[str]:
    out = []
    head = headnote_text(sents)
    if head:
        out.append(head)
    sent_words = [sent.split() for sent in sents]
    i = 0
    n = len(sents)
    while i < n:
        buf: list[str] = []
        j = i
        while j < n and len(buf) < PASSAGE_WORDS:
            buf.extend(sent_words[j])
            j += 1
        if buf:
            out.append("passage: " + " ".join(buf[:PASSAGE_WORDS]))
        if j >= n:
            break
        skip = 0
        k = i
        while k < j and skip < STRIDE_WORDS:
            skip += len(sent_words[k])
            k += 1
        i = max(k, i + 1)
    return out or ["passage: " + " ".join(sents)[:2000]]


def split_bundle(split: str) -> dict:
    ids, docs, sents = load_pool(split)
    queries = load_queries(split)
    qids = list(queries.keys())
    q_sents = [queries[q]["text"] if isinstance(queries[q]["text"], list) else [docs[0]] for q in qids]
    return {
        "split": split,
        "queries": queries,
        "qids": qids,
        "q_sents": q_sents,
        "cand": [str(c) for c in ids],
        "docs": docs,
        "sents": sents,
    }


def years(sent_lists) -> np.ndarray:
    return np.array([year_of(sents)[0] or -1 for sents in sent_lists])


def masks(qids, cand, q_sents, c_sents) -> tuple[np.ndarray, np.ndarray]:
    cand = np.asarray(cand)
    self_mask = np.array([[c == q for c in cand] for q in qids])
    cy = years(c_sents)
    qy = np.array([year_of(sents)[0] or 10_000 for sents in q_sents])
    after = (cy[None, :] > qy[:, None]) & (cy[None, :] > 0)
    return self_mask, after


def apply_mask(scores: np.ndarray, self_mask, after, fill=-1e9) -> np.ndarray:
    out = np.array(scores, dtype=np.float64, copy=True)
    out[self_mask] = fill
    out[after] = fill
    return out


def zrows(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores) & (scores > -1e8)
    masked = np.where(finite, scores, np.nan)
    mu = np.nanmean(masked, axis=1, keepdims=True)
    sd = np.nanstd(masked, axis=1, keepdims=True)
    z = (masked - mu) / (sd + 1e-9)
    return np.where(finite, np.nan_to_num(z, nan=0.0), -8.0)


def halves(qids) -> tuple[set, set]:
    order = np.array(sorted(qids))
    rng = np.random.default_rng(781)
    rng.shuffle(order)
    mid = len(order) // 2
    return set(order[:mid]), set(order[mid:])


def gold_of(queries) -> dict:
    return {q: list(v.get("relevant_candidates") or []) for q, v in queries.items()}


def topk(qids, cand, scores, k: int = K) -> dict:
    cand = np.asarray(cand)
    order = np.argsort(-scores, axis=1)[:, :k]
    return {q: [str(cand[i]) for i in order[r]] for r, q in enumerate(qids)}


def recall_at(gold, ranked, qids, ks=(40, 100)) -> dict:
    out = {}
    for k in ks:
        hit = tot = 0
        for q in qids:
            g = set(gold[q])
            tot += len(g)
            hit += len(g & set(ranked[q][:k]))
        out[f"recall@{k}"] = hit / max(tot, 1)
    return out


def score_split(gold, pred, qids) -> dict:
    ranked = {q: pred[q] for q in qids}
    metrics = micro_f1(gold, ranked, qids)
    metrics.update(recall_at(gold, ranked, qids))
    return metrics


def half_scores(gold, pred, qids) -> dict:
    a, b = halves(qids)
    return {
        "all": score_split(gold, pred, qids),
        "A": score_split(gold, pred, [q for q in qids if q in a]),
        "B": score_split(gold, pred, [q for q in qids if q in b]),
    }


def both_halves_at_least(report: dict, bar: float) -> bool:
    return report["A"]["f1"] >= bar and report["B"]["f1"] >= bar


def column_softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    finite = scores > -1e8
    shifted = np.where(finite, scores, -1e8)
    shifted = shifted - shifted.max(axis=0, keepdims=True)
    col = np.exp(shifted / tau) * finite
    denom = col.sum(axis=0, keepdims=True)
    safe = np.divide(col, denom, out=np.zeros_like(col), where=denom > 0)
    return np.log(safe + 1e-30)


def sinkhorn(scores: np.ndarray, row_mass: np.ndarray, eps: float, iters: int = 40) -> np.ndarray:
    """Entropic assignment. Columns sum to at most 1 via a slack row."""
    n_q, n_c = scores.shape
    finite = scores > -1e8
    shifted = np.where(finite, scores, scores.max() if np.isfinite(scores).any() else 0.0)
    kernel = np.exp((shifted - shifted.max()) / max(eps, 1e-3))
    kernel = np.where(finite, kernel, 0.0)
    mass = np.clip(row_mass.astype(np.float64), 1e-6, None)
    if mass.sum() > n_c - 1:
        mass *= (n_c - 1) / mass.sum()
    slack_mass = n_c - mass.sum()
    u = np.ones(n_q)
    u_slack = 1.0
    v = np.ones(n_c)
    for _ in range(iters):
        offered = kernel @ v
        u = mass / np.maximum(offered, 1e-12)
        u_slack = slack_mass / max(float(v.sum()), 1e-12)
        demand = kernel.T @ u + u_slack
        v = 1.0 / np.maximum(demand, 1e-12)
    return u[:, None] * kernel * v[None, :]


def fit_marker_mass(marker_counts: np.ndarray, gold_counts: np.ndarray) -> tuple[float, float]:
    """gold ≈ a + b * markers, the row marginal used by Sinkhorn."""
    x = np.column_stack([np.ones(len(marker_counts)), marker_counts])
    coef, *_ = np.linalg.lstsq(x, gold_counts, rcond=None)
    return float(coef[0]), float(coef[1])


def row_mass_from(marker_counts: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return np.clip(intercept + slope * marker_counts, 1.0, 40.0)


def hungarian(scores: np.ndarray, slots: np.ndarray):
    from scipy.optimize import linear_sum_assignment

    n_q = scores.shape[0]
    slots = np.maximum(1, slots.astype(int))
    if slots.sum() > scores.shape[1]:
        slots = np.maximum(1, np.floor(slots * scores.shape[1] * 0.98 / slots.sum()).astype(int))
    rows = np.repeat(np.arange(n_q), slots)
    chosen_row, chosen_col = linear_sum_assignment(-scores[rows])
    buckets = {i: [] for i in range(n_q)}
    for i, j in zip(chosen_row, chosen_col):
        buckets[rows[i]].append((scores[rows[i], j], j))
    order = np.zeros((n_q, K), dtype=int)
    for i in range(n_q):
        got = [j for _, j in sorted(buckets[i], reverse=True)]
        for j in np.argsort(-scores[i]):
            if len(got) >= K:
                break
            if j not in got:
                got.append(int(j))
        order[i] = got[:K]
    return order


def decode(qids, cand, scores, kind: str, marker_counts=None, mass_coef=(0.0, 1.0), eps=0.5, tau=0.5) -> dict:
    if kind == "row":
        decoded = scores
    elif kind == "column":
        decoded = column_softmax(scores, tau)
    elif kind == "sinkhorn":
        counts = np.ones(len(qids)) if marker_counts is None else np.asarray(marker_counts, dtype=np.float64)
        mass = row_mass_from(counts, mass_coef[0], mass_coef[1])
        decoded = sinkhorn(scores, mass, eps)
    elif kind == "hungarian":
        counts = np.full(len(qids), K) if marker_counts is None else np.clip(marker_counts, K, 40)
        order = hungarian(scores, counts)
        cand = np.asarray(cand)
        return {q: [str(cand[i]) for i in order[r]] for r, q in enumerate(qids)}
    else:
        raise ValueError(kind)
    return topk(qids, cand, decoded, K)


def check_submission(pred: dict, qids, cand) -> None:
    pool = set(map(str, cand))
    if set(pred) != set(qids):
        raise SystemExit(f"queries {len(pred)} != {len(qids)}")
    for q, ids in pred.items():
        if len(ids) != K or len(set(ids)) != K:
            raise SystemExit(f"{q} does not have 7 unique ids")
        if q in ids:
            raise SystemExit(f"{q} retrieved itself")
        if any(i not in pool for i in ids):
            raise SystemExit(f"{q} has an id outside the pool")


def save_metrics(name: str, payload: dict) -> str:
    dest = os.path.join(OUT, name)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "metrics.json")
    json.dump(payload, open(path, "w"), indent=1)
    return path


def citator_matrix(q_docs, qids, c_docs, cand) -> np.ndarray:
    cited = [cited_by(doc) for doc in c_docs]
    own = [own_keys(doc) for doc in q_docs]
    out = np.zeros((len(qids), len(cand)), dtype=np.float32)
    index = {c: i for i, c in enumerate(cand)}
    for r, keys in enumerate(own):
        if not keys:
            continue
        for c, edges in zip(cand, cited):
            if c != qids[r] and keys & edges:
                out[r, index[c]] = 1.0
    return out


def bm25_matrix(docs: list[str], queries: list[str], k1: float, b: float, max_features: int = 300_000) -> np.ndarray:
    """Okapi BM25. Rows are queries, columns are docs."""
    counter = CountVectorizer(
        lowercase=True, token_pattern=r"[a-z0-9]+", max_features=max_features, min_df=1,
    )
    counts = counter.fit_transform(docs)
    query_counts = counter.transform(queries)
    n_docs = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    length = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average = float(length.mean()) if len(length) else 1.0
    norm = k1 * (1.0 - b + b * length / max(average, 1.0))
    weighted = counts.astype(np.float32).copy()
    weighted.data = weighted.data * (k1 + 1.0) / (weighted.data + norm[weighted.nonzero()[0]])
    weighted = weighted.multiply(idf)
    return (query_counts.astype(np.float32) @ weighted.T).toarray()


def reduce_passages(passage_scores: np.ndarray, owner: np.ndarray, n_cand: int, how: str = "max") -> np.ndarray:
    """passage_scores is (n_query_units, n_passages). owner maps passage -> candidate."""
    n_units = passage_scores.shape[0]
    out = np.full((n_units, n_cand), -1e9, dtype=np.float32)
    for cand_i in range(n_cand):
        cols = np.where(owner == cand_i)[0]
        if len(cols) == 0:
            continue
        block = passage_scores[:, cols]
        out[:, cand_i] = block.max(axis=1) if how == "max" else block.mean(axis=1)
    return out


def logsumexp_max(unit_by_cand: np.ndarray, tau: float) -> np.ndarray:
    """unit_by_cand is (n_units, n_cand) of max-passage scores. Returns (n_cand,)."""
    shifted = unit_by_cand - unit_by_cand.max(axis=0, keepdims=True)
    return tau * np.log(np.exp(shifted / tau).sum(axis=0) + 1e-12) + unit_by_cand.max(axis=0)


def pairwise_L(context: np.ndarray, cand: np.ndarray, owner: np.ndarray, n_cand: int, tau: float) -> np.ndarray:
    """context (m, d), cand (p, d), both L2-normalised. One query -> (n_cand,)."""
    if len(context) == 0 or len(cand) == 0:
        return np.zeros(n_cand, dtype=np.float32)
    sims = context @ cand.T
    per_slot = reduce_passages(sims, owner, n_cand, "max")
    return logsumexp_max(per_slot, tau).astype(np.float32)


def pairwise_G(passages_q: np.ndarray, cand: np.ndarray, owner: np.ndarray, n_cand: int, topk_query: int) -> np.ndarray:
    if len(passages_q) == 0 or len(cand) == 0:
        return np.zeros(n_cand, dtype=np.float32)
    sims = passages_q @ cand.T
    distinct = np.partition(sims, max(0, sims.shape[1] - 3), axis=1)[:, -3:].mean(axis=1)
    keep = np.argsort(-distinct)[:topk_query]
    per = reduce_passages(sims[keep], owner, n_cand, "max")
    return per.mean(axis=0).astype(np.float32)
