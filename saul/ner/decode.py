"""Document-level decoding: string voting, propagation, thresholds, boundary rules.

Mention probabilities are summed across occurrences of the same string (not log-summed,
which would punish a string for being repeated). Strings of 4 characters or fewer, and
digit-only strings, are scored per mention and are not propagated.
"""
from __future__ import annotations

import json
import os

import numpy as np

from saul.ner.labels import LABELS, N_LABELS, NONE, PERSON_LABELS, SUBTOKEN_LABELS
from saul.ner.metrics import gold_map, strict_scores

HONORIFICS = (
    "Hon'ble Justice ", "Hon'ble ", "Justice ", "Advocate ",
    "Shri ", "Smt. ", "Mr. ", "Ms. ", "Mrs. ",
)
TRAILING = (", J.", ", J", " J.", "& Ors.", "& Ors", " & Ors.", ", learned", " learned counsel")
SHORT = 4


def default_config() -> dict:
    return {
        "tau": {lab: 0.5 for lab in LABELS},
        "propagate": {lab: True for lab in LABELS},
        "rules": {
            "honorific": False,
            "trailing": False,
            "whitespace": False,
            "prec_paren": False,
            "subtoken": False,
        },
    }


def _word_bounded(text: str, start: int, end: int) -> bool:
    left = start == 0 or not text[start - 1].isalnum()
    right = end == len(text) or not text[end].isalnum()
    return left and right


def pool_mentions(mentions: list[dict], text: str) -> list[dict]:
    """Group mentions. Long strings share one summed probability vector."""
    groups: dict[tuple, list] = {}
    order: list[tuple] = []
    for m in mentions:
        s, e = int(m["start"]), int(m["end"])
        if e <= s or s < 0 or e > len(text):
            continue
        frag = text[s:e]
        probs = np.asarray(m["probs"], dtype=np.float64)
        short = len(frag) <= SHORT or frag.isdigit()
        key = (s, e) if short else ("S", frag)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((s, e, probs))
    out = []
    for key in order:
        items = groups[key]
        mass = np.zeros(len(items[0][2]), dtype=np.float64)
        occ = []
        for s, e, probs in items:
            mass += probs
            occ.append((s, e))
        frag = text[occ[0][0]:occ[0][1]] if isinstance(key[0], int) else key[1]
        out.append({
            "string": frag,
            "mass": mass,
            "occurrences": occ,
            "short": isinstance(key[0], int),
        })
    return out


def accept_groups(groups: list[dict], cfg: dict) -> list[dict]:
    tau = cfg["tau"]
    accepted = []
    for g in groups:
        mass = g["mass"]
        if mass.shape[0] == N_LABELS + 1:
            label_mass = mass[:N_LABELS]
        else:
            label_mass = mass
        if float(label_mass.sum()) <= 0:
            continue
        lab_i = int(label_mass.argmax())
        score = float(label_mass[lab_i])
        # None-class mass can veto a weak label when it is larger than every label.
        if mass.shape[0] == N_LABELS + 1 and float(mass[NONE]) > score:
            continue
        lab = LABELS[lab_i]
        if score < float(tau.get(lab, 0.5)):
            continue
        accepted.append({**g, "label": lab, "score": score})
    return accepted


def propagate(text: str, accepted: list[dict], cfg: dict) -> list[tuple[int, int, str]]:
    """Place accepted strings. Long strings expand to every whole-word hit, longest first."""
    placed: list[tuple[int, int, str, int]] = []
    prop = cfg["propagate"]
    expandable = []
    fixed = []
    for g in accepted:
        do_prop = (not g["short"]) and prop.get(g["label"], True)
        if do_prop:
            expandable.append(g)
        else:
            for s, e in g["occurrences"]:
                fixed.append((s, e, g["label"], len(text[s:e])))
    expandable.sort(key=lambda g: len(g["string"]), reverse=True)
    occupied = np.zeros(len(text), dtype=bool)

    def _take(s: int, e: int, lab: str) -> None:
        if e <= s or s < 0 or e > len(text) or occupied[s:e].any():
            return
        if not _word_bounded(text, s, e):
            return
        occupied[s:e] = True
        placed.append((s, e, lab, e - s))

    for g in expandable:
        frag = g["string"]
        if not frag:
            continue
        start = 0
        while True:
            i = text.find(frag, start)
            if i < 0:
                break
            _take(i, i + len(frag), g["label"])
            start = i + 1
    fixed.sort(key=lambda x: x[3], reverse=True)
    for s, e, lab, _ in fixed:
        _take(s, e, lab)
    return [(s, e, lab) for s, e, lab, _ in placed]


def _shift_edges(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def apply_rules(text: str, spans: list[tuple[int, int, str]], cfg: dict) -> list[tuple[int, int, str]]:
    rules = cfg["rules"]
    out = []
    seen = set()
    for s, e, lab in spans:
        if rules.get("whitespace"):
            s, e = _shift_edges(text, s, e)
        frag = text[s:e]
        if rules.get("honorific") and lab in PERSON_LABELS:
            for h in HONORIFICS:
                if frag.startswith(h) and len(frag) > len(h):
                    s += len(h)
                    frag = text[s:e]
                    break
        if rules.get("trailing") and lab in PERSON_LABELS:
            for t in TRAILING:
                if frag.endswith(t) and len(frag) > len(t):
                    e -= len(t)
                    frag = text[s:e]
                    break
        if rules.get("prec_paren") and lab == "PREC" and "(" in frag and ")" not in frag:
            k = text.find(")", e, min(len(text), e + 80))
            if k > e:
                e = k + 1
        if e <= s:
            continue
        key = (s, e, lab)
        if key not in seen:
            seen.add(key)
            out.append(key)
    if rules.get("subtoken"):
        out = _subtoken(text, out, seen)
    return out


def _subtoken(text: str, spans: list[tuple[int, int, str]], seen: set) -> list[tuple[int, int, str]]:
    extra = []
    occupied = np.zeros(len(text), dtype=bool)
    for s, e, _ in spans:
        occupied[s:e] = True
    for s, e, lab in spans:
        if lab not in SUBTOKEN_LABELS:
            continue
        frag = text[s:e]
        if len(frag) < 5:
            continue
        start = 0
        while True:
            i = text.find(frag, start)
            if i < 0:
                break
            j = i + len(frag)
            left = i == 0 or not text[i - 1].isalnum()
            inside = j < len(text) and text[j].isalnum()
            if left and inside and not occupied[i:j].any():
                key = (i, j, lab)
                if key not in seen:
                    seen.add(key)
                    extra.append(key)
                    occupied[i:j] = True
            start = i + 1
    return spans + extra


def decode_doc(text: str, mentions: list[dict], cfg: dict) -> list[tuple[int, int, str]]:
    groups = pool_mentions(mentions, text)
    accepted = accept_groups(groups, cfg)
    spans = propagate(text, accepted, cfg)
    return apply_rules(text, spans, cfg)


def decode_all(docs_text: dict[str, str], mentions_by_doc: dict[str, list], cfg: dict) -> dict[str, list]:
    return {did: decode_doc(text, mentions_by_doc.get(did, []), cfg) for did, text in docs_text.items()}


def _with(cfg: dict, **changes) -> dict:
    out = json.loads(json.dumps(cfg))
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def tune(docs, mentions_by_doc: dict[str, list], cfg: dict | None = None) -> dict:
    """Coordinate ascent on per-label thresholds, then propagation and rule switches."""
    cfg = json.loads(json.dumps(cfg or default_config()))
    texts = {d.doc_id: d.text for d in docs}
    gold = gold_map(docs)
    grid = [round(x, 2) for x in np.linspace(0.15, 0.9, 16)]

    def score(c: dict) -> float:
        pred = decode_all(texts, mentions_by_doc, c)
        return float(strict_scores(gold, pred)["macro_f1"])

    for _ in range(2):
        for lab in LABELS:
            best_t, best_f = cfg["tau"][lab], score(cfg)
            for t in grid:
                trial = _with(cfg, tau={lab: t})
                f = score(trial)
                if f > best_f + 1e-6:
                    best_f, best_t = f, t
            cfg["tau"][lab] = best_t
    for lab in LABELS:
        if not cfg["propagate"][lab]:
            continue
        trial = _with(cfg, propagate={lab: False})
        if score(trial) > score(cfg) + 1e-6:
            cfg["propagate"][lab] = False
    for rule in ("honorific", "trailing", "whitespace", "prec_paren", "subtoken"):
        trial = _with(cfg, rules={rule: True})
        if score(trial) > score(cfg) + 1e-6:
            cfg["rules"][rule] = True
    _floor_empty(cfg, docs, mentions_by_doc)
    cfg["oof_f1"] = score(cfg)
    return cfg


def _floor_empty(cfg: dict, docs, mentions_by_doc: dict[str, list]) -> None:
    """If a label would emit nothing, lower its threshold to the 5th percentile of its string scores."""
    texts = {d.doc_id: d.text for d in docs}
    pred = decode_all(texts, mentions_by_doc, cfg)
    counts = {lab: 0 for lab in LABELS}
    for spans in pred.values():
        for _, _, lab in spans:
            if lab in counts:
                counts[lab] += 1
    scores = {lab: [] for lab in LABELS}
    for doc in docs:
        for g in pool_mentions(mentions_by_doc.get(doc.doc_id, []), doc.text):
            mass = g["mass"][:N_LABELS]
            if float(mass.sum()) <= 0:
                continue
            lab = LABELS[int(mass.argmax())]
            scores[lab].append(float(mass.max()))
    for lab in LABELS:
        if counts[lab] > 0 or not scores[lab]:
            continue
        cfg["tau"][lab] = float(np.percentile(scores[lab], 5))


def mentions_from_npz(path: str) -> list[dict]:
    """Rebuild the mention list stored beside a document npz, if the json sidecar is absent."""
    z = np.load(path)
    out = []
    if "span_s" in z and len(z["span_s"]):
        for s, e, probs in zip(z["span_s"].tolist(), z["span_e"].tolist(), z["span_probs"].tolist()):
            out.append({"start": int(s), "end": int(e), "probs": probs})
    return out


def load_mention_dir(directory: str) -> dict[str, list]:
    path = os.path.join(directory, "oof_mentions.jsonl")
    by: dict[str, list] = {}
    if not os.path.exists(path):
        return by
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by.setdefault(rec["doc"], []).append(rec)
    return by


def propagation_oracle(text: str, gold: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """First gold label of each string, copied to every whole-word occurrence. The 0.99 ceiling."""
    first: dict[str, str] = {}
    for s, e, lab in sorted(gold):
        first.setdefault(text[s:e], lab)
    occupied = np.zeros(len(text), dtype=bool)
    placed = []
    for frag in sorted(first, key=len, reverse=True):
        if not frag:
            continue
        start = 0
        while True:
            i = text.find(frag, start)
            if i < 0:
                break
            j = i + len(frag)
            if _word_bounded(text, i, j) and not occupied[i:j].any():
                occupied[i:j] = True
                placed.append((i, j, first[frag]))
            start = i + 1
    return placed
