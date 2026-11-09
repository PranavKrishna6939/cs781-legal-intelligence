import json
import re
import sys

sys.path.insert(0, "scripts")
from p7_bm25 import join_text, load_pool, load_queries

OWN = re.compile(r"(\d{4})\s+AIR\s*(\d+)")
CITATOR = re.compile(r"CITATOR INFO\s*:(.*?)(?:ACT\s*:|HEADNOTE\s*:|$)", re.S)
ENTRY = re.compile(r"\b[A-Z]{1,3}\s+(\d{4})\s+SC\s*(\d+)")


def own_keys(text):
    head = text[:1500]
    m = re.search(r"CITATION\s*:(.*?)(?:CITATOR INFO|ACT\s*:|HEADNOTE|$)", head, re.S)
    return {(y, int(n)) for y, n in OWN.findall(m.group(1))} if m else set()


def cited_by(text):
    m = CITATOR.search(text[:4000])
    return {(y, int(n)) for y, n in ENTRY.findall(m.group(1))} if m else set()


def main():
    for split in ["Train", "Dev", "Test"]:
        qs = load_queries(split)
        ids, docs, _ = load_pool(split)
        cby = {c: cited_by(d) for c, d in zip(ids, docs)}
        qkey = {q: own_keys(join_text(v["text"])) for q, v in qs.items()}
        hits = {q: {c for c in ids if c != q and qkey[q] & cby[c]} for q in qs}
        res = {
            "queries_with_own_AIR": sum(bool(k) for k in qkey.values()),
            "cands_with_citator": sum(bool(v) for v in cby.values()),
            "pred_links": sum(len(h) for h in hits.values()),
            "queries_with_any_link": sum(bool(h) for h in hits.values()),
        }
        if split != "Test":
            gold = {q: set(v["relevant_candidates"]) for q, v in qs.items()}
            tp = sum(len(hits[q] & gold[q]) for q in qs)
            res["link_precision"] = tp / max(res["pred_links"], 1)
            res["link_recall_of_all_gold"] = tp / sum(len(g) for g in gold.values())
        print(split, json.dumps(res))


if __name__ == "__main__":
    main()
