import collections
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from p7_bm25 import K, load_queries, micro_f1
from p7_fix import FUSE, scored


def pick(qids, cand, s):
    order = np.argsort(-s, 1)[:, :K]
    return {q: [str(cand[i]) for i in order[r]] for r, q in enumerate(qids)}


def main():
    qs, qids, cand, views, self_mask, after = scored("Dev")
    gold = {q: v["relevant_candidates"] for q, v in qs.items()}
    base = sum(FUSE[k] * views[k] for k in FUSE)
    base[self_mask] = -50
    base[after] = -50
    res = {"base": micro_f1(gold, pick(qids, cand, base), qids)["f1"]}
    for t in [0.05, 0.1, 0.2, 0.5]:
        col = np.exp((base - base.max()) / t)
        col = col / col.sum(0, keepdims=True)
        row = np.exp((base - base.max(1, keepdims=True)) / t)
        row = row / row.sum(1, keepdims=True)
        res[f"dual_softmax_t{t}"] = micro_f1(gold, pick(qids, cand, np.log(col + 1e-30) + np.log(row + 1e-30)), qids)["f1"]
        res[f"col_softmax_t{t}"] = micro_f1(gold, pick(qids, cand, np.log(col + 1e-30)), qids)["f1"]
    for a in [0.25, 0.5, 0.75, 1.0]:
        cz = (base - base.mean(0, keepdims=True)) / (base.std(0, keepdims=True) + 1e-9)
        res[f"rowz+{a}colz"] = micro_f1(gold, pick(qids, cand, base + a * cz), qids)["f1"]
    qset = set(qs)
    res["gold_links"] = sum(len(g) for g in gold.values())
    res["gold_that_are_queries"] = sum(c in qset for g in gold.values() for c in g)
    pop = collections.Counter(c for g in gold.values() for c in g)
    res["cands_cited_by_k_queries"] = dict(sorted(collections.Counter(pop.values()).items()))
    marks = [" ".join(qs[q]["text"]).count("CITATION") for q in qids]
    res["corr_markers_vs_gold"] = float(np.corrcoef(marks, [len(gold[q]) for q in qids])[0, 1])
    res["mean_markers"] = float(np.mean(marks))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
