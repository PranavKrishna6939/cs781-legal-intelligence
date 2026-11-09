"""TF-IDF retrieval for Phase 7. Fit on each split's own candidate pool."""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
DATA = os.path.join(ROOT, "data", "raw", "Phase_7_PCR")
OUT = os.path.join(ROOT, "p7")
SUB = os.path.join(ROOT, "submissions", "phase7")
K = 7


def join_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def load_queries(split: str) -> dict:
    path = os.path.join(DATA, split, f"{split.lower()}.json")
    return json.load(open(path, encoding="utf-8"))


def load_pool(split: str) -> tuple[list[str], list[str]]:
    path = os.path.join(DATA, split, f"{split.lower()}_candidates.json")
    raw = json.load(open(path, encoding="utf-8"))
    ids = list(raw.keys())
    docs = [join_text(raw[i]["text"]) for i in ids]
    sents = [raw[i]["text"] if isinstance(raw[i]["text"], list) else [docs[n]] for n, i in enumerate(ids)]
    return ids, docs, sents


def query_text(row: dict, first_sents: int | None) -> str:
    text = row["text"]
    if isinstance(text, list) and first_sents:
        text = text[:first_sents]
    return join_text(text)


def micro_f1(gold: dict[str, list[str]], pred: dict[str, list[str]], queries: list[str]) -> dict:
    tp = fp = fn = 0
    for qid in queries:
        g = set(gold[qid])
        p = set(pred[qid][:K])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def rank_split(cand_ids, cand_docs, queries: dict, ngram, first_sents: int | None, max_features: int):
    vec = TfidfVectorizer(
        ngram_range=ngram, sublinear_tf=True, max_features=max_features, lowercase=True,
    )
    cand_matrix = vec.fit_transform(cand_docs if first_sents is None else [
        " ".join(s[:first_sents]) if isinstance(s, list) else s for s in cand_docs
    ])
    qids = list(queries.keys())
    q_text = [query_text(queries[i], first_sents) for i in qids]
    scores = linear_kernel(vec.transform(q_text), cand_matrix)
    return qids, np.asarray(cand_ids), scores


def top_ids(qids, cand_ids, scores, n: int) -> dict[str, list[str]]:
    out = {}
    for row, qid in enumerate(qids):
        take = min(n, scores.shape[1])
        order = np.argpartition(-scores[row], take - 1)[:take]
        order = order[np.argsort(-scores[row, order])]
        out[qid] = [str(cand_ids[i]) for i in order]
    return out


def write_zip(name: str, pred: dict) -> str:
    dest = os.path.join(SUB, name)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "prediction.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(pred, handle)
    zpath = os.path.join(dest, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="prediction.json")
    return zpath


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dev = load_queries("Dev")
    test = load_queries("Test")
    train = load_queries("Train")
    gold = {qid: [c for c in row.get("relevant_candidates") or [] if c] for qid, row in dev.items()}
    lengths = [len(v) for v in gold.values()]
    print(
        f"[bm25] queries train {len(train)} dev {len(dev)} test {len(test)} "
        f"gold/dev mean {np.mean(lengths):.2f} p50 {np.median(lengths):.0f} "
        f"min {min(lengths)} max {max(lengths)}",
        flush=True,
    )

    views = {
        "word12_full": dict(ngram=(1, 2), first_sents=None, max_features=300_000),
        "word13_head40": dict(ngram=(1, 3), first_sents=40, max_features=300_000),
    }
    rng = np.random.default_rng(781)
    dev_ids = np.array(sorted(dev))
    rng.shuffle(dev_ids)
    half = len(dev_ids) // 2
    part = {"A": set(dev_ids[:half]), "B": set(dev_ids[half:])}

    results = {}
    saved = {}
    for name, cfg in views.items():
        pool_ids, pool_docs, pool_sents = load_pool("Dev")
        docs = pool_sents if cfg["first_sents"] else pool_docs
        qids, cand_ids, scores = rank_split(pool_ids, docs, dev, cfg["ngram"], cfg["first_sents"], cfg["max_features"])
        pred = top_ids(qids, cand_ids, scores, K)
        scores_by = {half_name: micro_f1(gold, pred, sorted(ids)) for half_name, ids in part.items()}
        full = micro_f1(gold, pred, list(dev))
        results[name] = {"A": scores_by["A"], "B": scores_by["B"], "all": full}
        saved[name] = (qids, cand_ids, scores, pred)
        print(
            f"[bm25] {name} A {scores_by['A']['f1']:.4f} B {scores_by['B']['f1']:.4f} all {full['f1']:.4f}",
            flush=True,
        )

    chosen = max(results, key=lambda name: results[name]["A"]["f1"])
    print(f"[bm25] chosen on A: {chosen} half B {results[chosen]['B']['f1']:.4f}", flush=True)

    # Test ranking uses the chosen view, fit on the test pool only.
    cfg = views[chosen]
    pool_ids, pool_docs, pool_sents = load_pool("Test")
    docs = pool_sents if cfg["first_sents"] else pool_docs
    qids, cand_ids, scores = rank_split(pool_ids, docs, test, cfg["ngram"], cfg["first_sents"], cfg["max_features"])
    pred = top_ids(qids, cand_ids, scores, 15)
    submit = {qid: ids[:K] for qid, ids in pred.items()}
    zpath = write_zip("P7_bm25", submit)
    np.savez(
        os.path.join(OUT, "bm25_test.npz"),
        qids=np.asarray(qids), cand_ids=np.asarray(cand_ids), scores=scores.astype(np.float32),
    )
    json.dump({qid: ids for qid, ids in pred.items()}, open(os.path.join(OUT, "bm25_test_top15.json"), "w"))

    # Train-pool BM25 hard negatives for the bi-encoder.
    t_ids, t_docs, _t_sents = load_pool("Train")
    tq, tc, ts = rank_split(t_ids, t_docs, train, (1, 2), None, 300_000)
    hard = {}
    for row, qid in enumerate(tq):
        gold_ids = set(train[qid].get("relevant_candidates") or [])
        order = np.argsort(-ts[row])
        picked = []
        for index in order:
            cid = str(tc[index])
            if cid not in gold_ids:
                picked.append(cid)
            if len(picked) == 20:
                break
        hard[qid] = picked
    json.dump(hard, open(os.path.join(OUT, "bm25_train_hard.json"), "w"))

    # Dev scores for the chosen view, full pool, so fusion can reuse them.
    d_ids, d_docs, d_sents = load_pool("Dev")
    docs = d_sents if cfg["first_sents"] else d_docs
    dq, dc, ds = rank_split(d_ids, docs, dev, cfg["ngram"], cfg["first_sents"], cfg["max_features"])
    np.savez(
        os.path.join(OUT, "bm25_dev.npz"),
        qids=np.asarray(dq), cand_ids=np.asarray(dc), scores=ds.astype(np.float32),
    )
    blob = {"chosen": chosen, "views": results, "zip": zpath, "gold_mean": float(np.mean(lengths))}
    json.dump(blob, open(os.path.join(OUT, "bm25.json"), "w"), indent=2)
    print(f"[bm25] wrote {zpath}", flush=True)


if __name__ == "__main__":
    main()
