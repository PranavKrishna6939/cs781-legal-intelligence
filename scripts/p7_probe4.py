import json
import sys

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, "scripts")
from p7_bm25 import join_text, load_pool, micro_f1
from p7_fix import FUSE, scored
from p7_probe2 import pick
from p7_probe3 import col_softmax


def main():
    qs, qids, cand, views, self_mask, after = scored("Dev")
    gold = {q: set(v["relevant_candidates"]) for q, v in qs.items()}
    ids, docs, _ = load_pool("Dev")
    assert list(ids) == [str(c) for c in cand]
    res = {}
    g = np.array([[c in gold[q] for c in cand] for q in qids])
    qnum = np.array([int(q) for q in qids])[:, None]
    cnum = np.array([int(c) for c in cand])[None, :]
    res["gold_id_lt_query_id"] = float((cnum < qnum)[g].mean())
    res["pool_id_lt_query_id"] = float((cnum < qnum)[~self_mask].mean())
    base = sum(FUSE[k] * views[k] for k in FUSE)
    base[self_mask] = -50
    base[after] = -50
    for n in [8, 5]:
        vec = HashingVectorizer(ngram_range=(n, n), binary=True, norm=None, alternate_sign=False, n_features=2**24, lowercase=True, token_pattern=r"[a-z0-9]+")
        C = vec.transform(docs)
        Q = vec.transform([join_text(qs[q]["text"]) for q in qids])
        ov = (Q @ C.T).toarray().astype(np.float32)
        ov[self_mask] = 0
        ov[after] = 0
        top1 = ov.argmax(1)
        has = ov.max(1) > 0
        res[f"ov{n}_queries_with_any"] = float(has.mean())
        res[f"ov{n}_top1_precision"] = float(np.mean([cand[j] in gold[q] for q, j, h in zip(qids, top1, has) if h]))
        res[f"ov{n}_pairs_gt0_precision"] = float(g[ov > 0].mean())
        res[f"ov{n}_pairs_gt0_recall"] = float((ov[g] > 0).mean())
        for a in [0.5, 1.0, 2.0]:
            s = base + a * np.log1p(ov)
            res[f"ov{n}_a{a}"] = micro_f1(gold, pick(qids, cand, s), qids)["f1"]
            res[f"ov{n}_a{a}_col05"] = micro_f1(gold, pick(qids, cand, col_softmax(s, 0.5)), qids)["f1"]
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
