import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "scripts")
from p7_bm25 import K, micro_f1
from p7_fix import FUSE, scored
from p7_probe2 import pick


def col_softmax(s, t):
    col = np.exp((s - s.max(0, keepdims=True)) / t)
    return np.log(col / col.sum(0, keepdims=True) + 1e-30)


def assign(qids, cand, s, slots):
    rows = np.repeat(np.arange(len(qids)), slots)
    r, c = linear_sum_assignment(-s[rows])
    out = {q: [] for q in qids}
    for i, j in zip(r, c):
        out[qids[rows[i]]].append((s[rows[i], j], str(cand[j])))
    pred = {}
    for n, q in enumerate(qids):
        got = [cid for _, cid in sorted(out[q], reverse=True)][:K]
        for j in np.argsort(-s[n]):
            if len(got) == K:
                break
            if str(cand[j]) not in got:
                got.append(str(cand[j]))
        pred[q] = got
    return pred


def main():
    qs, qids, cand, views, self_mask, after = scored("Dev")
    gold = {q: v["relevant_candidates"] for q, v in qs.items()}
    base = sum(FUSE[k] * views[k] for k in FUSE)
    base[self_mask] = -50
    base[after] = -50
    is_query = np.isin(cand, list(qs))
    res = {}
    for drop_q in [False, True]:
        b = base.copy()
        if drop_q:
            b[:, is_query] = -50
        tag = f"dropq{int(drop_q)}"
        res[f"{tag}_rowonly"] = micro_f1(gold, pick(qids, cand, b), qids)["f1"]
        for t in [0.5, 1.0]:
            res[f"{tag}_col_t{t}"] = micro_f1(gold, pick(qids, cand, col_softmax(b, t)), qids)["f1"]
        marks = np.array([" ".join(qs[q]["text"]).count("CITATION") for q in qids])
        for name, slots in [("7", np.full(len(qids), K)), ("marks", np.clip(marks, 1, 40)),
                            ("marks_ge7", np.clip(marks, K, 40))]:
            if slots.sum() > b.shape[1]:
                slots = np.maximum(1, (slots * b.shape[1] * 0.98 / slots.sum()).astype(int))
            res[f"{tag}_assign_{name}"] = micro_f1(gold, assign(qids, cand, b, slots), qids)["f1"]
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
