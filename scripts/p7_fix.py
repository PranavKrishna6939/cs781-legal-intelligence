import json
import re
import sys

import numpy as np

sys.path.insert(0, "scripts")
from p7_bm25 import K, load_pool, load_queries, micro_f1, rank_split, write_zip

YEAR = re.compile(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})")
ANY_YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")


def year_of(sents):
    m = YEAR.search(" ".join(sents[:30]))
    if m:
        return int(m.group(1)), "header"
    ys = [int(y) for y in ANY_YEAR.findall(" ".join(sents))]
    return (max(ys), "max") if ys else (None, "none")

VIEWS = {"full": ((1, 2), None), "head40": ((1, 3), 40)}
FUSE = {"full": 0.75, "head40": 0.25}


def z(scores):
    return (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-9)


def scored(split):
    ids, docs, sents = load_pool(split)
    qs = load_queries(split)
    out = {}
    for name, (ngram, first) in VIEWS.items():
        qids, cand, s = rank_split(ids, docs, qs, ngram, first, 300_000)
        out[name] = z(s)
    cy = np.array([year_of(s)[0] or -1 for s in sents])
    qy = np.array([year_of(qs[q]["text"])[0] or 10_000 for q in qids])
    self_mask = np.array([[c == q for c in cand] for q in qids])
    after = (cy[None, :] > qy[:, None]) & (cy[None, :] > 0)
    return qs, qids, cand, out, self_mask, after


def predict(qids, cand, s, self_mask, after, penalty):
    s = s.copy()
    s[self_mask] = -1e9
    s = s - penalty * after
    order = np.argsort(-s, 1)[:, :K]
    return {q: [str(cand[i]) for i in order[r]] for r, q in enumerate(qids)}


def main():
    qs, qids, cand, views, self_mask, after = scored("Dev")
    gold = {q: v["relevant_candidates"] for q, v in qs.items()}
    rows = {}
    for vname in ["full", "head40", "fuse"]:
        s = views[vname] if vname != "fuse" else sum(FUSE[k] * views[k] for k in FUSE)
        for pen in [0.0, 0.5, 1.0, 2.0, 1e9]:
            for drop_self in [False, True]:
                m = self_mask if drop_self else np.zeros_like(self_mask)
                pred = predict(qids, cand, s, m, after, pen)
                rows[f"{vname}|pen{pen:g}|self{int(drop_self)}"] = micro_f1(gold, pred, qids)["f1"]
    for k, v in sorted(rows.items(), key=lambda x: -x[1]):
        print(f"{v:.4f} {k}")
    json.dump(rows, open("p7/fix_dev.json", "w"), indent=1)

    tq, tqids, tcand, tviews, tself, tafter = scored("Test")
    fused = sum(FUSE[k] * tviews[k] for k in FUSE)
    picks = {
        "P7_fix_fuse_self": (fused, 0.0),
        "P7_fix_fuse_date": (fused, float(sys.argv[1]) if len(sys.argv) > 1 else 1.0),
        "P7_fix_head40_date": (tviews["head40"], float(sys.argv[1]) if len(sys.argv) > 1 else 1.0),
    }
    for name, (s, pen) in picks.items():
        pred = predict(tqids, tcand, s, tself, tafter, pen)
        assert len(pred) == len(tq) and all(len(set(v)) == K and q not in v for q, v in pred.items())
        write_zip(name, pred)
        print("wrote", name)


if __name__ == "__main__":
    main()
