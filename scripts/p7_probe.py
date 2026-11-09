import collections
import json
import re
import sys

import numpy as np

sys.path.insert(0, "scripts")
from p7_bm25 import load_pool, load_queries, micro_f1

YEAR = re.compile(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})")
ANY_YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")


def year_of(sents):
    text = " ".join(sents[:30])
    m = YEAR.search(text)
    if m:
        return int(m.group(1)), "header"
    ys = [int(y) for y in ANY_YEAR.findall(" ".join(sents))]
    return (max(ys), "max") if ys else (None, "none")


out = {}
pools = {}
for split in ["Train", "Dev", "Test"]:
    ids, docs, sents = load_pool(split)
    pools[split] = (ids, sents)
    qs = load_queries(split)
    out[split] = {"queries": len(qs), "pool": len(ids)}
    out[split]["pool_is_query"] = len(set(ids) & set(qs))
    out[split]["citation_marker_q"] = float(np.mean(["CITATION" in " ".join(q["text"]) for q in qs.values()]))
    cy = {c: year_of(s) for c, s in zip(ids, sents)}
    out[split]["cand_year_src"] = collections.Counter(v[1] for v in cy.values())
    if split == "Test":
        qy = {q: year_of(v["text"]) for q, v in qs.items()}
        out[split]["query_year_src"] = collections.Counter(v[1] for v in qy.values())
        continue
    gold = {q: set(v["relevant_candidates"]) for q, v in qs.items()}
    qy = {q: year_of(v["text"]) for q, v in qs.items()}
    out[split]["query_year_src"] = collections.Counter(v[1] for v in qy.values())
    ok = tot = 0
    kept = []
    for q, g in gold.items():
        y = qy[q][0]
        if y is None:
            continue
        for c in g:
            cyr = cy.get(c, (None,))[0]
            if cyr is None:
                continue
            tot += 1
            ok += cyr <= y
        kept.append(np.mean([cy[c][0] is None or cy[c][0] <= y for c in ids]))
    out[split]["gold_not_after_query"] = ok / max(tot, 1)
    out[split]["pool_kept_by_date"] = float(np.mean(kept)) if kept else None
    pop = collections.Counter(c for g in gold.values() for c in g)
    out[split]["gold_cands_distinct"] = len(pop)
    out[split]["top_pop"] = pop.most_common(5)
    out[split]["frac_pool_ever_gold"] = len(pop) / len(ids)
    if split == "Dev":
        qids = sorted(gold)
        pred = {q: [c for c, _ in pop.most_common(7)] for q in qids}
        out[split]["popularity_only_f1_insample"] = micro_f1(gold, pred, qs)
out["pool_overlap"] = {
    "train_dev": len(set(pools["Train"][0]) & set(pools["Dev"][0])),
    "train_test": len(set(pools["Train"][0]) & set(pools["Test"][0])),
    "dev_test": len(set(pools["Dev"][0]) & set(pools["Test"][0])),
}
print(json.dumps(out, indent=1, default=str))
