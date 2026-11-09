"""Precision and recall of the header parser against gold role strings.

Folds 0–3 are the development check. Fold 4 is printed afterwards and is the blind check:
if it drops by more than 0.10, simplify the patterns in saul/ner/header.py. Do not add new ones.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.data import load_folds, load_split  # noqa: E402
from saul.ner.header import counsel_cues, norm_key, parse_header, witness_cues  # noqa: E402


def _sets(docs):
    """Return predicted and gold normalised keys for the acceptance table."""
    app_p, app_g = set(), set()
    side_ok = side_n = 0
    cou_p, cou_g = set(), set()
    wit_p, wit_hit = set(), set()
    for doc in docs:
        header = parse_header(doc.text)
        gold = {}
        for s, e, lab in doc.gold:
            if s < header.end and lab in ("APP", "RESP", "A.COUNSEL", "R.COUNSEL"):
                gold.setdefault(norm_key(doc.text[s:e]), lab)
        for party in header.parties:
            key = norm_key(party.name)
            if not key:
                continue
            if party.kind == "party" and party.side == "A":
                app_p.add((doc.doc_id, key, "APP"))
            elif party.kind == "party" and party.side == "R":
                app_p.add((doc.doc_id, key, "RESP"))
            elif party.kind == "counsel":
                lab = "A.COUNSEL" if party.side == "A" else "R.COUNSEL"
                cou_p.add((doc.doc_id, key, lab))
        for key, lab in gold.items():
            if lab in ("APP", "RESP"):
                app_g.add((doc.doc_id, key, lab))
            else:
                cou_g.add((doc.doc_id, key, lab))
        for key, lab in gold.items():
            if lab not in ("APP", "RESP"):
                continue
            pred = [p for p in header.parties if p.kind == "party" and norm_key(p.name) == key]
            if not pred:
                continue
            side_n += 1
            want = "A" if lab == "APP" else "R"
            side_ok += int(pred[0].side == want)
        wit_gold = {norm_key(doc.text[s:e]) for s, e, lab in doc.gold if lab == "WIT"}
        for cue in witness_cues(doc.text):
            key = norm_key(cue.name)
            wit_p.add((doc.doc_id, key))
            if key in wit_gold:
                wit_hit.add((doc.doc_id, key))
        _ = counsel_cues
    return app_p, app_g, side_ok, side_n, cou_p, cou_g, wit_p, wit_hit


def _pr(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r


def _report(name, docs):
    app_p, app_g, side_ok, side_n, cou_p, cou_g, wit_p, wit_hit = _sets(docs)
    ap, ar = _pr(app_p, app_g)
    cp, cr = _pr(cou_p, cou_g)
    wp = len(wit_hit) / len(wit_p) if wit_p else 0.0
    side = side_ok / side_n if side_n else 0.0
    print(f"{name:16} APP/RESP P={ap:.3f} R={ar:.3f} side={side:.3f} "
          f"counsel P={cp:.3f} R={cr:.3f} witness P={wp:.3f} n={len(docs)}")
    return ap, ar, side, cp, wp


def main() -> None:
    docs = load_split("train")
    folds = load_folds(docs)
    by_id = {d.doc_id: d for d in docs}
    dev = [by_id[i] for ids in folds[:4] for i in ids if i in by_id]
    blind = [by_id[i] for i in folds[4] if i in by_id]
    d = _report("folds 0-3", dev)
    b = _report("fold 4 (blind)", blind)
    print("acceptance folds 0-3: APP/RESP P>=0.85 R>=0.75 side>=0.97 counsel P>=0.85 witness P>=0.90")
    ok = d[0] >= 0.85 and d[1] >= 0.75 and d[2] >= 0.97 and d[3] >= 0.85 and d[4] >= 0.90
    print("folds 0-3", "PASS" if ok else "FAIL")
    for name, i in (("P", 0), ("R", 1)):
        if d[i] - b[i] > 0.10:
            print(f"fold 4 {name} dropped by {d[i] - b[i]:.3f}; simplify patterns, do not add any")


if __name__ == "__main__":
    main()
