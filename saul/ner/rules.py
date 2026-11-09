"""Boundary and label rules P3-a … P3-f (plan §S.6). Each rule is a config switch."""
from __future__ import annotations

import re

from saul.ner.labels import LABELS
from saul.ner.tokdec import PERSONS, _bounded, label_mass

CITE = re.compile(
    r",?\s*(?:\(\d{4}\)\s*\d+\s*SCC\s*\d+|AIR\s*\d{4}\s*[A-Z][A-Za-z.]*\s*\d+|"
    r"\[\d{4}\]\s*\d+\s*S\.?C\.?R\.?\s*\d+|\d{4}\s*\(\d+\)\s*[A-Z]{2,5}\s*\d+)"
)
VS_SEP = re.compile(r"(?<![A-Za-z])(?:v\.|vs\.|versus)(?![A-Za-z])", re.I)
SHORT_CTX = re.compile(r"(?:case|\(supra\)|supra|'s case)", re.I)
PREC_SKIP = {"state", "union of india", "commissioner"}
ORG_SUF = re.compile(
    r"(?:[A-Z][\w.&'-]*\s+){0,6}(?:Authority|Commission|Board|Ministry|Department|"
    r"Directorate|Tribunal|Government|Council|Corporation)\b"
)
GOV_OF = re.compile(r"^Government of .+", re.I)
YEAR_TAIL = re.compile(r"\s+(?:of|OF)\s+\d{4}|/\d{4}")
SENT_PUNCT = set(".!?;")


def _tok_at(doc, pos: int) -> int | None:
    for i, (s, e) in enumerate(zip(doc.starts, doc.ends)):
        if s <= pos < e:
            return i
    return None


def _occupied(text: str, spans: list[tuple]) -> list[bool]:
    occ = [False] * (len(text) + 1)
    for s, e, _ in spans:
        for i in range(max(s, 0), min(e, len(text))):
            occ[i] = True
    return occ


def _free(occ, s: int, e: int) -> bool:
    return not any(occ[s:e])


def _mark(occ, s: int, e: int) -> None:
    for i in range(s, e):
        occ[i] = True


def p3_caseno(doc, spans: list[tuple]) -> list[tuple]:
    out = []
    for s, e, lab in spans:
        if lab != "CASENO":
            out.append((s, e, lab))
            continue
        i = _tok_at(doc, s)
        if i is not None:
            taken = 0
            j = i - 1
            while j >= 0 and taken < 6:
                ts, te = int(doc.starts[j]), int(doc.ends[j])
                frag = doc.text[ts:te]
                if not frag or frag[0].islower() or frag[0] in SENT_PUNCT or frag.isdigit():
                    break
                if frag[:1].isupper() or (frag.startswith("(") and frag.endswith(")") and re.fullmatch(r"\([A-Za-z.]+\)", frag)):
                    s = ts
                    taken += 1
                    j -= 1
                    continue
                break
        tail = doc.text[e:e + 16]
        m = YEAR_TAIL.match(tail) if re.search(r"\d\s*$", doc.text[s:e]) else None
        if m:
            e = e + m.end()
        out.append((s, e, lab))
    return out


def p3_prec_split(doc, spans: list[tuple]) -> list[tuple]:
    out = []
    for s, e, lab in spans:
        if lab != "PREC":
            out.append((s, e, lab))
            continue
        frag = doc.text[s:e]
        parts = re.split(r";|,\s+", frag)
        if len(parts) > 1 and all(VS_SEP.search(p) for p in parts):
            cursor = s
            for p in parts:
                at = doc.text.find(p, cursor, e + 2)
                if at < 0:
                    continue
                ps, pe = at, at + len(p)
                ps = ps + (len(p) - len(p.lstrip()))
                pe = at + len(p.rstrip())
                if pe > ps:
                    out.append((ps, pe, "PREC"))
                cursor = pe
            continue
        out.append((s, e, lab))
    return out


def p3_prec_cite(doc, spans: list[tuple]) -> list[tuple]:
    out = []
    for s, e, lab in spans:
        if lab == "PREC":
            m = CITE.match(doc.text[e:e + 40])
            if m and m.end() <= 40:
                e = e + m.end()
        out.append((s, e, lab))
    return out


def p3_prec_short(doc, spans: list[tuple]) -> list[tuple]:
    extra = []
    occ = _occupied(doc.text, spans)
    for s, e, lab in spans:
        if lab != "PREC":
            continue
        frag = doc.text[s:e]
        m = VS_SEP.search(frag)
        if not m:
            continue
        first = frag[:m.start()].strip(" ,.;")
        key = first.casefold()
        words = first.split()
        if key in PREC_SKIP:
            continue
        if len(words) < 2 and not (len(words) == 1 and len(words[0]) >= 6):
            continue
        pat = re.compile(re.escape(first) + r".{0,12}?(?:case|\(supra\)|supra|'s case)", re.I)
        for hit in pat.finditer(doc.text):
            hs, he = hit.start(), hit.start() + len(first)
            if doc.text[hs:he].casefold() != first.casefold():
                continue
            if not _bounded(doc.text, hs, he) or not _free(occ, hs, he):
                continue
            extra.append((hs, he, "PREC"))
            _mark(occ, hs, he)
    return spans + extra


def p3_auth_gov(doc, spans: list[tuple], lexicon: dict[str, str] | None) -> list[tuple]:
    """Replace `Government of <Place>` with the head noun when the lexicon says so.

    The longer span is the boundary error, so it does not stay beside the head noun.
    """
    if not lexicon or lexicon.get("Government") != "AUTH":
        return spans
    out = []
    for s, e, lab in spans:
        if lab == "AUTH" and GOV_OF.match(doc.text[s:e]):
            at = doc.text.lower().find("government", s, e)
            if at >= s:
                out.append((at, at + len("Government"), "AUTH"))
                continue
        out.append((s, e, lab))
    return out


def p3_auth_cand(doc, spans: list[tuple], logp, theta: float) -> list[tuple]:
    mass = label_mass(logp) if logp is not None else None
    auth_i = LABELS.index("AUTH")
    occ = _occupied(doc.text, spans)
    extra = []
    if mass is None:
        return spans
    for m in ORG_SUF.finditer(doc.text):
        s, e = m.start(), m.end()
        if not _bounded(doc.text, s, e) or not _free(occ, s, e):
            continue
        idxs = [i for i, (a, b) in enumerate(zip(doc.starts, doc.ends)) if a < e and b > s]
        if not idxs:
            continue
        if float(mass[idxs, auth_i].mean()) < theta:
            continue
        extra.append((s, e, "AUTH"))
        _mark(occ, s, e)
    return spans + extra


def p3_court_split(doc, spans: list[tuple]) -> list[tuple]:
    out = []
    for s, e, lab in spans:
        if lab != "COURT":
            out.append((s, e, lab))
            continue
        frag = doc.text[s:e]
        parts = re.split(r",\s+|\s+and\s+", frag)
        if len(parts) > 1 and all(re.search(r"Court|Tribunal", p) for p in parts):
            cursor = s
            for p in parts:
                at = doc.text.find(p, cursor, e + 2)
                if at < 0:
                    continue
                ps = at + (len(p) - len(p.lstrip()))
                pe = at + len(p.rstrip())
                if pe > ps:
                    out.append((ps, pe, "COURT"))
                cursor = pe
            continue
        out.append((s, e, lab))
    return out


def p3_junk(spans: list[tuple], text: str) -> list[tuple]:
    out = []
    for s, e, lab in spans:
        frag = text[s:e]
        letters = len(re.findall(r"[A-Za-z]", frag))
        if lab in PERSONS and (letters < 2 or (e - s) > 60):
            continue
        if frag and not re.search(r"[A-Za-z0-9]", frag):
            continue
        out.append((s, e, lab))
    return out


def apply_rules(doc, spans: list[tuple], cfg: dict, logp=None, lexicon: dict | None = None) -> list[tuple]:
    if cfg.get("p3_caseno"):
        spans = p3_caseno(doc, spans)
    if cfg.get("p3_prec_split"):
        spans = p3_prec_split(doc, spans)
    if cfg.get("p3_prec_cite"):
        spans = p3_prec_cite(doc, spans)
    if cfg.get("p3_prec_short"):
        spans = p3_prec_short(doc, spans)
    if cfg.get("p3_auth_gov"):
        spans = p3_auth_gov(doc, spans, lexicon)
    if cfg.get("p3_auth_cand"):
        spans = p3_auth_cand(doc, spans, logp, float(cfg.get("theta_auth_cand", 0.15)))
    if cfg.get("p3_court_split"):
        spans = p3_court_split(doc, spans)
    if cfg.get("p3_junk"):
        spans = p3_junk(spans, doc.text)
    spans = [sp for sp in spans if sp[1] > sp[0]]
    return sorted(set(spans))
