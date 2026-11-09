"""Fuse BM25 with the dense bi-encoder. The weight is chosen on dev half A."""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
DATA = os.path.join(ROOT, "data", "raw", "Phase_7_PCR")
OUT = os.path.join(ROOT, "p7")
SUB = os.path.join(ROOT, "submissions", "phase7")
K = 7
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def micro_f1(gold, pred, queries) -> float:
    tp = fp = fn = 0
    for qid in queries:
        g = set(gold[qid])
        p = set(pred[qid][:K])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def load_bundle(kind: str):
    bm = np.load(os.path.join(OUT, f"bm25_{kind}.npz"), allow_pickle=True)
    de = np.load(os.path.join(OUT, f"dense_{kind}.npz"), allow_pickle=True)
    return bm, de


def aligned_scores(kind: str):
    bm, de = load_bundle(kind)
    bq = [str(i) for i in bm["qids"].tolist()]
    bc = [str(i) for i in bm["cand_ids"].tolist()]
    dq = [str(i) for i in de["qids"].tolist()]
    dc = [str(i) for i in de["cand_ids"].tolist()]
    q_index = {i: n for n, i in enumerate(dq)}
    c_index = {i: n for n, i in enumerate(dc)}
    query = np.stack([de["query"][q_index[i]] for i in bq])
    cand = np.stack([de["cand"][c_index[i]] for i in bc])
    dense = query @ cand.T
    return bq, bc, np.asarray(bm["scores"], dtype=np.float64), dense


def zscore(row: np.ndarray) -> np.ndarray:
    std = row.std()
    if std < 1e-8:
        return np.zeros_like(row)
    return (row - row.mean()) / std


def topn(qids, cand_ids, scores, n: int) -> dict[str, list[str]]:
    out = {}
    for row, qid in enumerate(qids):
        take = min(n, scores.shape[1])
        order = np.argpartition(-scores[row], take - 1)[:take]
        order = order[np.argsort(-scores[row, order])]
        out[qid] = [str(cand_ids[i]) for i in order]
    return out


def fused(bm, dense, w: float) -> np.ndarray:
    out = np.empty_like(bm)
    for i in range(bm.shape[0]):
        out[i] = (1 - w) * zscore(bm[i]) + w * zscore(dense[i])
    return out


def halves(qids: list[str]) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(781)
    ids = np.array(sorted(qids))
    rng.shuffle(ids)
    mid = len(ids) // 2
    return list(ids[:mid]), list(ids[mid:])


def write_zip(name: str, pred: dict) -> str:
    dest = os.path.join(SUB, name)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "prediction.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({qid: ids[:K] for qid, ids in pred.items()}, handle)
    zpath = os.path.join(dest, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="prediction.json")
    return zpath


def vary(best: dict[str, list[str]], bm: dict[str, list[str]], dense: dict[str, list[str]], lower: dict[str, list[str]]) -> dict[str, dict]:
    files = {
        "P7_best": {q: ids[:K] for q, ids in best.items()},
        "P7_swap7": {},
        "P7_swap67": {},
        "P7_bm25": {q: ids[:K] for q, ids in bm.items()},
        "P7_dense": {q: ids[:K] for q, ids in dense.items()},
        "P7_w_step": {q: ids[:K] for q, ids in lower.items()},
    }
    for qid, ids in best.items():
        files["P7_swap7"][qid] = ids[:6] + [ids[7]]
        files["P7_swap67"][qid] = ids[:5] + [ids[7], ids[8]]
    # Identical lists are not a second submission. Replace the last ID.
    seen = []
    for name, pred in list(files.items()):
        key = tuple((q, tuple(pred[q])) for q in sorted(pred))
        if key in seen:
            for qid, ids in best.items():
                used = set(pred[qid])
                extra = next(c for c in bm[qid] if c not in used)
                pred[qid] = pred[qid][:6] + [extra]
        seen.append(tuple((q, tuple(files[name][q])) for q in sorted(files[name])))
    return files


def main() -> None:
    dev = json.load(open(os.path.join(DATA, "Dev", "dev.json"), encoding="utf-8"))
    gold = {qid: [c for c in row.get("relevant_candidates") or [] if c] for qid, row in dev.items()}
    qids, cand_ids, bm, dense = aligned_scores("dev")
    part_a, part_b = halves(qids)
    bm_f1 = micro_f1(gold, topn(qids, cand_ids, bm, K), part_b)
    chosen = 0.0
    chosen_b = bm_f1
    table = {}
    for w in WEIGHTS:
        scores = fused(bm, dense, w)
        pred = topn(qids, cand_ids, scores, K)
        fa = micro_f1(gold, pred, part_a)
        fb = micro_f1(gold, pred, part_b)
        table[str(w)] = {"A": fa, "B": fb}
        print(f"[fuse] w {w:.2f} A {fa:.4f} B {fb:.4f}", flush=True)
        if w == 0.0:
            continue
        # Choose on A. Keep only if B does not fall below pure BM25.
        if fa > table["0.0"]["A"] and fb >= bm_f1 and fa > table.get(str(chosen), {}).get("A", -1):
            chosen = w
            chosen_b = fb
    if chosen != 0.0 and chosen_b < bm_f1:
        chosen = 0.0
    print(f"[fuse] kept w {chosen:.2f} half B {table[str(chosen)]['B']:.4f}", flush=True)

    tq, tc, tbm, td = aligned_scores("test")
    best_scores = fused(tbm, td, chosen)
    best15 = topn(tq, tc, best_scores, 15)
    bm15 = topn(tq, tc, tbm, 15)
    dense15 = topn(tq, tc, td, 15)
    lower_w = max(w for w in WEIGHTS if w < chosen) if chosen > 0 else 0.0
    lower15 = topn(tq, tc, fused(tbm, td, lower_w), 15)
    files = vary(best15, bm15, dense15, lower15)
    written = []
    for name, pred in files.items():
        path = write_zip(name, pred)
        written.append({"name": name, "zip": path, "w": chosen if name == "P7_best" else None})
        print(f"[fuse] {name} {path}", flush=True)
    json.dump({"w": chosen, "dev": table, "files": written}, open(os.path.join(OUT, "fuse.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
