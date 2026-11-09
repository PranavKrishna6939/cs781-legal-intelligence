"""Reporter-citation matching: citations written in the query body against each candidate's own header citations."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import join_text
from p7_common import OUT, gold_of, split_bundle

HEADER = re.compile(r"CITATION\s*:(.*?)(?:CITATOR INFO|ACT\s*:|HEADNOTE|$)", re.S)
# Header forms: "1971 AIR 2410", "1972 SCR (1) 736", "1985 SCC (3) 120", "1971 SCC (1) 9".
HEAD_AIR = re.compile(r"(\d{4})\s+AIR\s+(\d+)")
HEAD_VOL = re.compile(r"(\d{4})\s+(SCR|SCC|SCALE|JT)\s*\(\s*(\d+)\s*\)\s*(\d+)")
# "1962 SCR Supl. (3) 436" and "1962 SCR Supp. (1) 100".
HEAD_SUPP = re.compile(r"(\d{4})\s+(SCR|SCC)\s+Sup(?:p|l)\.?\s*\(\s*(\d+)\s*\)\s*(\d+)", re.I)
# "1952 SCR 597": no volume.
HEAD_BARE = re.compile(r"(\d{4})\s+(SCR|SCC)\s+(\d+)")
# "JT 1991 (2) 45", year after the reporter.
HEAD_JT = re.compile(r"\bJT\s+(\d{4})\s*\(\s*(\d+)\s*\)\s*(\d+)")
# Body forms.
BODY_AIR = re.compile(r"AIR\s*(\d{4})\s*S\.?\s*C\.?\s*(\d+)")
BODY_AIR2 = re.compile(r"(\d{4})\s+AIR\s+(\d+)")
BODY_VOL = re.compile(r"[\(\[]\s*(\d{4})\s*[\)\]]\s*(\d+)\s+(SCR|SCC|S\.C\.R\.|S\.C\.C\.)\s+(\d+)")
BODY_VOL2 = re.compile(r"(\d{4})\s+(SCR|SCC|S\.C\.R\.|S\.C\.C\.)\s*\(\s*(\d+)\s*\)\s*(\d+)")
BODY_BARE = re.compile(r"(\d{4})\s+(SCR|SCC|S\.C\.R\.|S\.C\.C\.)\s+(\d+)")
BODY_SUPP = re.compile(
    r"(\d{4})\s+Supp(?:l)?\.?\s*(?:\(\s*(\d+)\s*\)|(\d+))\s+(SCR|SCC|S\.C\.R\.|S\.C\.C\.)\s+(\d+)", re.I,
)
BODY_SUPP2 = re.compile(
    r"(\d{4})\s+(SCR|SCC|S\.C\.R\.|S\.C\.C\.)\s+Supp(?:l)?\.?\s*\(\s*(\d+)\s*\)\s*(\d+)", re.I,
)


def norm_rep(rep: str) -> str:
    return rep.replace(".", "").upper()


def header_keys(text: str) -> set:
    m = HEADER.search(text[:2000])
    if not m:
        return set()
    body = m.group(1)
    keys = {("AIR", y, int(p)) for y, p in HEAD_AIR.findall(body)}
    keys |= {(norm_rep(r), y, int(v), int(p)) for y, r, v, p in HEAD_VOL.findall(body)}
    keys |= {(norm_rep(r), y, int(v), int(p)) for y, r, v, p in HEAD_SUPP.findall(body)}
    keys |= {(norm_rep(r), y, 0, int(p)) for y, r, p in HEAD_BARE.findall(body)}
    keys |= {("JT", y, int(v), int(p)) for y, v, p in HEAD_JT.findall(body)}
    return keys


def body_keys(text: str) -> set:
    keys = {("AIR", y, int(p)) for y, p in BODY_AIR.findall(text)}
    keys |= {("AIR", y, int(p)) for y, p in BODY_AIR2.findall(text)}
    keys |= {(norm_rep(r), y, int(v), int(p)) for y, v, r, p in BODY_VOL.findall(text)}
    keys |= {(norm_rep(r), y, int(v), int(p)) for y, r, v, p in BODY_VOL2.findall(text)}
    keys |= {(norm_rep(r), y, 0, int(p)) for y, r, p in BODY_BARE.findall(text)}
    for y, vol_a, vol_b, rep, page in BODY_SUPP.findall(text):
        keys.add((norm_rep(rep), y, int(vol_a or vol_b or 0), int(page)))
    keys |= {(norm_rep(r), y, int(v), int(p)) for y, r, v, p in BODY_SUPP2.findall(text)}
    return keys


def body_text(text: str) -> str:
    """Drop the query's own header so it does not match itself."""
    m = HEADER.search(text[:2000])
    return text[m.end():] if m else text


def probe(split: str) -> dict:
    bundle = split_bundle(split)
    has_gold = split != "Test"
    gold = gold_of(bundle["queries"]) if has_gold else {}
    cand = bundle["cand"]
    own = [header_keys(doc) for doc in bundle["docs"]]
    index: dict = {}
    for j, keys in enumerate(own):
        for k in keys:
            index.setdefault(k, set()).add(j)
    links = tp = 0
    gold_total = sum(len(set(map(str, v))) for v in gold.values()) if has_gold else 0
    per_query = []
    edges = []
    for qi, q in enumerate(bundle["qids"]):
        qtext = body_text(" ".join(bundle["q_sents"][qi]))
        keys = body_keys(qtext)
        hit = set()
        for k in keys:
            hit |= index.get(k, set())
        hit = {j for j in hit if cand[j] != q}
        links += len(hit)
        per_query.append(len(hit))
        for j in hit:
            edges.append((q, cand[j]))
        if has_gold:
            g = set(map(str, gold[q]))
            tp += len({cand[j] for j in hit} & g)
    out = {
        "split": split,
        "cands_with_header_keys": int(sum(bool(k) for k in own)),
        "pool": len(cand),
        "links": links,
        "queries_with_link": int(sum(1 for n in per_query if n)),
        "queries": len(bundle["qids"]),
        "mean_links_per_query": float(np.mean(per_query)),
    }
    if has_gold:
        out.update({
            "tp": tp,
            "precision": tp / max(links, 1),
            "recall_of_gold": tp / max(gold_total, 1),
            "gold_total": gold_total,
        })
    json.dump(edges, open(os.path.join(OUT, f"reporter_{split.lower()}.json"), "w"))
    return out


def main():
    res = [probe(s) for s in ("Train", "Dev", "Test")]
    for r in res:
        print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "probe8.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
