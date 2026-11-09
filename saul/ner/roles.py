"""Entity-level role resolver (plan §S.5).

One label per surface string in a document, from header evidence and token masses.
The gradient-boosted tree and the logistic regression are both fit; the caller
keeps whichever wins nested macro-F1.
"""
from __future__ import annotations

import math
import re

import numpy as np

from saul.ner.header import Party, norm_key
from saul.ner.labels import LABELS, O
from saul.ner.tokdec import HONORIFIC, _bounded, label_mass

ROLE_LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH"]
CLASSES = ROLE_LABELS + ["NONE"]
BLOCK_LABELS = {"PREC", "CASENO", "STAT", "COURT"}
N_FEATURES = 60
ORG_RE = re.compile(
    r"\b(?:Ltd|Limited|Company|Corporation|Board|State of|Union of India|Government|"
    r"Commission|Authority|Bank|Trust|Society|Association|Directorate|Department|"
    r"Ministry|Custodian|Council|University)\b",
    re.I,
)
HONOR_BEFORE = re.compile(r"(?:Mr|Ms|Mrs|Shri|Smt|Sri|Dr)\.?\s*$", re.I)
KEYWORDS = [
    re.compile(p, re.I)
    for p in (
        r"\bappellant",
        r"\bpetitioner",
        r"\bapplicant",
        r"\brespondent",
        r"\baccused",
        r"\b(?:deceased|late)\b",
        r"\bcomplainant",
        r"\b(?:victim|prosecutrix)",
        r"\b(?:learned counsel|advocate)",
        r"(?:\bJ\.|\bJustice|\bHon'?ble)",
        r"(?:\bP\.?W\b|\bD\.?W\b|\bwitness|\bdeposed)",
        r"(?:\bfather|\bson|\bwife|\bdaughter|\bS/o|\bW/o)",
    )
]


def _word_hits(text: str, frag: str, case: bool) -> list[tuple[int, int]]:
    flags = 0 if case else re.I
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(frag) + r"(?![A-Za-z0-9])", flags)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _inside(s: int, e: int, blocked: list[tuple[int, int]]) -> bool:
    return any(a < e and b > s for a, b in blocked)


class Entity:
    def __init__(self, key: str):
        self.key = key
        self.surfaces: set[str] = set()
        self.mentions: list[tuple[int, int, str]] = []  # start, end, surface


def collect_entities(doc, spans: list[tuple], header_parties: list[Party],
                     witnesses: list[Party], counsels: list[Party]) -> list[Entity]:
    """Candidates from tokdec role spans and from the parser, grouped by normalised key."""
    seeds: list[str] = []
    for s, e, lab in spans:
        if lab in ROLE_LABELS:
            seeds.append(doc.text[s:e])
    for party in list(header_parties) + list(witnesses) + list(counsels):
        if party.name:
            seeds.append(party.name)
    blocked = [(s, e) for s, e, lab in spans if lab in BLOCK_LABELS]
    by_key: dict[str, Entity] = {}
    seen_span: set[tuple[int, int]] = set()
    for seed in seeds:
        key = norm_key(seed)
        if not key or key in by_key and seed in by_key[key].surfaces:
            continue
        ent = by_key.setdefault(key, Entity(key))
        for s, e in _word_hits(doc.text, seed, case=False):
            if (s, e) in seen_span or _inside(s, e, blocked):
                continue
            surface = doc.text[s:e]
            if norm_key(surface) != key:
                continue
            # Case-sensitive occurrence of this exact surface.
            if not _bounded(doc.text, s, e):
                continue
            seen_span.add((s, e))
            ent.surfaces.add(surface)
            ent.mentions.append((s, e, surface))
    for ent in by_key.values():
        ent.mentions.sort()
    return [e for e in by_key.values() if e.mentions]


def _token_idxs(doc, s: int, e: int) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(doc.starts, doc.ends)) if int(a) < e and int(b) > s]


def _mass_row(doc, logp, s: int, e: int) -> np.ndarray | None:
    idxs = _token_idxs(doc, s, e)
    if not idxs or logp is None:
        return None
    mass = label_mass(logp)
    return mass[idxs].mean(0)


def _o_mass(doc, logp, s: int, e: int) -> float:
    idxs = _token_idxs(doc, s, e)
    if not idxs or logp is None:
        return 0.0
    p = np.exp(logp[idxs].astype(np.float64))
    return float(p[:, O].mean())


def _flags(ent: Entity, header_parties: list[Party], witnesses: list[Party], counsels: list[Party],
           header_end: int) -> dict:
    keys = {norm_key(p.name): p for p in header_parties}
    party = keys.get(ent.key)
    return {
        "a_party": bool(party and party.kind == "party" and party.side == "A"),
        "r_party": bool(party and party.kind == "party" and party.side == "R"),
        "a_counsel": any(norm_key(p.name) == ent.key and p.side == "A" and p.kind == "counsel" for p in header_parties),
        "r_counsel": any(norm_key(p.name) == ent.key and p.side == "R" and p.kind == "counsel" for p in header_parties),
        "judge": any(norm_key(p.name) == ent.key and p.kind == "judge" for p in header_parties),
        "witness": any(norm_key(p.name) == ent.key for p in witnesses),
        "cue_a": sum(1 for p in counsels if norm_key(p.name) == ent.key and p.side == "A"),
        "cue_r": sum(1 for p in counsels if norm_key(p.name) == ent.key and p.side == "R"),
        "has_a": any(p.kind == "party" and p.side == "A" for p in header_parties),
        "has_r": any(p.kind == "party" and p.side == "R" for p in header_parties),
        "header_end": header_end,
    }


def featurize(doc, ent: Entity, spans: list[tuple], logp, flags: dict, prior: np.ndarray | None) -> np.ndarray:
    """60 features, f1–f60, in the order fixed by plan §S.5."""
    x = np.zeros(N_FEATURES, dtype=np.float64)
    role_i = [LABELS.index(l) for l in ROLE_LABELS]
    masses = []
    o_masses = []
    for s, e, _ in ent.mentions:
        row = _mass_row(doc, logp, s, e)
        masses.append(row if row is not None else np.zeros(len(LABELS)))
        o_masses.append(_o_mass(doc, logp, s, e))
    M = np.stack(masses) if masses else np.zeros((1, len(LABELS)))
    mean = M[:, role_i].mean(0)
    mx = M[:, role_i].max(0)
    x[0:7] = mean
    x[7:14] = mx
    x[14] = float(np.mean(o_masses)) if o_masses else 0.0
    pred_lab = {}
    for s, e, lab in spans:
        if lab in ROLE_LABELS:
            pred_lab[(s, e)] = lab
    n = max(len(ent.mentions), 1)
    counts = np.zeros(7)
    uncovered = 0
    for s, e, _ in ent.mentions:
        lab = pred_lab.get((s, e))
        if lab in ROLE_LABELS:
            counts[ROLE_LABELS.index(lab)] += 1
        else:
            uncovered += 1
    x[15:22] = counts / n
    x[22] = uncovered / n
    x[23] = float(flags["a_party"])
    x[24] = float(flags["r_party"])
    x[25] = float(flags["a_counsel"])
    x[26] = float(flags["r_counsel"])
    x[27] = float(flags["judge"])
    x[28] = float(flags["witness"])
    x[29] = float(flags["cue_a"])
    x[30] = float(flags["cue_r"])
    x[31] = float(flags["has_a"])
    x[32] = float(flags["has_r"])
    kw = np.zeros(12)
    for s, e, _ in ent.mentions:
        win = doc.text[max(0, s - 80):e + 80]
        for i, pat in enumerate(KEYWORDS):
            kw[i] += len(pat.findall(win))
    x[33:45] = kw / n
    x[45] = math.log1p(len(ent.mentions))
    first = ent.mentions[0][0]
    x[46] = first / max(len(doc.text), 1)
    x[47] = float(first < int(flags["header_end"]))
    main = max(ent.surfaces, key=len)
    letters = re.sub(r"[^A-Za-z]", "", main)
    x[48] = float(bool(letters) and letters.isupper())
    x[49] = float(len(main.split()))
    x[50] = float(any(ORG_RE.search(s) for s in ent.surfaces))
    honor = 0
    for s, _, _surf in ent.mentions:
        if HONOR_BEFORE.search(doc.text[max(0, s - 12):s]):
            honor = 1
            break
    x[51] = float(honor)
    if prior is None:
        prior = np.zeros(8)
    x[52:60] = prior[:8]
    return x


def prior_vector(counts: dict[str, int] | None) -> np.ndarray:
    """7 label shares plus log(1 + count). `counts` maps a role label to a tally."""
    v = np.zeros(8, dtype=np.float64)
    if not counts:
        return v
    total = sum(counts.get(l, 0) for l in ROLE_LABELS)
    if total:
        for i, lab in enumerate(ROLE_LABELS):
            v[i] = counts.get(lab, 0) / total
    v[7] = math.log1p(total)
    return v


def string_priors(docs, fold_of: dict[str, int]) -> dict[int, dict[str, dict[str, int]]]:
    """fold -> exact surface -> label -> count, built from the other folds' gold."""
    by_fold: dict[int, list] = {}
    for d in docs:
        by_fold.setdefault(fold_of.get(d.doc_id, -1), []).append(d)
    out: dict[int, dict[str, dict[str, int]]] = {}
    folds = sorted(by_fold)
    for held in folds:
        counts: dict[str, dict[str, int]] = {}
        for f, group in by_fold.items():
            if f == held:
                continue
            for d in group:
                for s, e, lab in d.gold:
                    if lab not in ROLE_LABELS:
                        continue
                    frag = d.text[s:e]
                    bucket = counts.setdefault(frag, {})
                    bucket[lab] = bucket.get(lab, 0) + 1
        out[held] = counts
    return out


def entity_target(doc, ent: Entity) -> str:
    gold = {(s, e): lab for s, e, lab in doc.gold}
    votes: dict[str, int] = {}
    for s, e, _ in ent.mentions:
        lab = gold.get((s, e))
        if lab in ROLE_LABELS:
            votes[lab] = votes.get(lab, 0) + 1
    if not votes:
        return "NONE"
    return max(votes, key=votes.get)


def document_rows(doc, spans, logp, parties, witnesses, counsels, header_end, prior_map):
    rows = []
    targets = []
    entities = collect_entities(doc, spans, parties, witnesses, counsels)
    for ent in entities:
        flags = _flags(ent, parties, witnesses, counsels, header_end)
        # Prior is for the exact surface with the most mentions.
        surface = max(ent.surfaces, key=lambda s: sum(1 for m in ent.mentions if m[2] == s))
        prior = prior_vector((prior_map or {}).get(surface))
        rows.append(featurize(doc, ent, spans, logp, flags, prior))
        targets.append(entity_target(doc, ent))
    return entities, rows, targets


def make_classifier(kind: str):
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=0,
        )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])


def fit_classifier(kind: str, x: np.ndarray, y: list[str]):
    clf = make_classifier(kind)
    clf.fit(x, y)
    return clf


def predict_proba(clf, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    classes = list(clf.classes_) if hasattr(clf, "classes_") else list(clf.named_steps["clf"].classes_)
    proba = clf.predict_proba(x)
    return classes, proba


def apply_decisions(doc, spans: list[tuple], entities: list[Entity], classes: list[str], proba: np.ndarray,
                    cfg: dict, header_end: int) -> list[tuple]:
    """Relabel, place body mentions, or drop, per the tuned thresholds."""
    theta_place = cfg.get("theta_place") or {lab: 0.5 for lab in ROLE_LABELS}
    theta_drop = float(cfg.get("theta_drop", 1.01))
    place_header = cfg.get("p2_place_header") or {lab: False for lab in ROLE_LABELS}
    pred = {(s, e): lab for s, e, lab in spans}
    drop: set[tuple[int, int]] = set()
    relabel: dict[tuple[int, int], str] = {}
    add: list[tuple] = []
    occupied = [False] * (len(doc.text) + 1)
    for s, e, _ in spans:
        for i in range(max(s, 0), min(e, len(doc.text))):
            occupied[i] = True
    blocked = [(s, e) for s, e, lab in spans if lab in BLOCK_LABELS]
    for ent, row in zip(entities, proba):
        k = int(row.argmax())
        lab = classes[k]
        p = float(row[k])
        if lab == "NONE" and p >= theta_drop:
            for s, e, _ in ent.mentions:
                drop.add((s, e))
            continue
        if lab == "NONE" or p < float(theta_place.get(lab, 0.5)):
            continue
        for s, e, _surf in ent.mentions:
            if _inside(s, e, blocked):
                continue
            in_header = s < header_end
            had = (s, e) in pred
            if had:
                relabel[(s, e)] = lab
            elif (not in_header) or place_header.get(lab, False):
                if not any(occupied[s:e]):
                    add.append((s, e, lab))
                    for i in range(s, e):
                        occupied[i] = True
    out = []
    for s, e, lab in spans:
        if (s, e) in drop and lab in ROLE_LABELS:
            continue
        out.append((s, e, relabel.get((s, e), lab)))
    out.extend(add)
    return sorted(set(out))


def apply_rule_fallback(doc, spans: list[tuple], parties: list[Party], witnesses: list[Party],
                        header_end: int, cfg: dict) -> list[tuple]:
    """APP/RESP from parser sides, WIT from witness cues, counsel from parser lists. No judges."""
    label_of = {}
    for p in parties:
        if p.kind == "party" and p.side == "A":
            label_of.setdefault(norm_key(p.name), "APP")
        elif p.kind == "party" and p.side == "R":
            label_of.setdefault(norm_key(p.name), "RESP")
        elif p.kind == "counsel" and p.side == "A":
            label_of.setdefault(norm_key(p.name), "A.COUNSEL")
        elif p.kind == "counsel" and p.side == "R":
            label_of.setdefault(norm_key(p.name), "R.COUNSEL")
    for p in witnesses:
        label_of.setdefault(norm_key(p.name), "WIT")
    place_header = cfg.get("p2_place_header") or {}
    pred = {(s, e): lab for s, e, lab in spans}
    occupied = [False] * (len(doc.text) + 1)
    for s, e, _ in spans:
        for i in range(s, min(e, len(doc.text))):
            occupied[i] = True
    relabel = {}
    add = []
    for ent in collect_entities(doc, spans, parties, witnesses, []):
        lab = label_of.get(ent.key)
        if not lab:
            continue
        for s, e, _surf in ent.mentions:
            if (s, e) in pred:
                relabel[(s, e)] = lab
                continue
            if s < header_end and not place_header.get(lab, False):
                continue
            if any(occupied[s:e]):
                continue
            add.append((s, e, lab))
            for i in range(s, e):
                occupied[i] = True
    out = [(s, e, relabel.get((s, e), lab)) for s, e, lab in spans]
    out.extend(add)
    return sorted(set(out))
