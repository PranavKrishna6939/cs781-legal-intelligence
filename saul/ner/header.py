"""Deterministic header parser for Indian judgments (plan §S.4).

Patterns were specified from training folds 0–3. Fold 4 is scored only by
scripts/p2_header_eval.py and is not used to add patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from saul.ner.tokdec import HONORIFIC

# Optional plural, with or without a space: APPELLANTS, APPELLANT (S), RESPONDENT(S).
_PLURAL = r"(?:S|\s*\(\s*S\s*\))?"
SIDE_A = rf"(?:APPELLANT|PETITIONER|APPLICANT|PLAINTIFF|COMPLAINANT|REVISIONIST){_PLURAL}"
SIDE_R = rf"(?:RESPONDENT|DEFENDANT|OPPOSITE\s+PART(?:Y|IES)|NON-APPLICANT|ACCUSED){_PLURAL}"
VS = r"(?:VERSUS|Versus|versus|Vs\.?|VS\.?|V/S|v\.)"
DOTS = r"(?:\.{2,}|…|-{2,}|:)?"
SCI = re.compile(
    r"PETITIONER:\s*(?P<a>.+?)\s+Vs\.?\s+RESPONDENT:\s*(?P<r>.+?)\s+DATE OF JUDGMENT",
    re.S,
)
BENCH = re.compile(
    r"BENCH:\s*(?P<j>.+?)\s+(?=BENCH:|CITATION:|ACT:|HEADNOTE:|JUDGMENT:)",
    re.S,
)
ROLEW = re.compile(
    rf"(?is)(?P<a>.{{2,400}}?)\s*{DOTS}\s*{SIDE_A}\s*(?:Through\s*:?\s*(?P<ca>.{{0,300}}?))?\s*{VS}\s+"
    rf"(?P<r>.{{2,400}}?)\s*{DOTS}\s*{SIDE_R}"
    rf"(?:\s*Through\s*:?\s*(?P<cr>.{{0,300}}?)(?=CORAM|JUDGMENT|J U D G|\.{{3,}}|$))?"
)
COLON = re.compile(
    rf"(?i)(?<![Ff]or\s)(?:Petitioner|Appellant|Applicant){_PLURAL}\s*:\s*-?\s*(?P<a>.+?)\s+"
    rf"(?:Respondent|Opposite Part(?:y|ies)){_PLURAL}\s*:\s*-?\s*(?P<r>.+?)\s+"
    r"(?=Counsel for|Hon'?ble|CORAM|-{5,})"
)
# Used only when no role-word pair matched. Names stay short so a bare "Vs." in the body is not a party.
_NAME = r"[A-Z][\w.@']+(?:\s+(?:of|and|the)\s+[A-Z][\w.@']+|\s+[A-Z][\w.@']+){0,8}"
BARE_VS = re.compile(
    rf"(?P<a>{_NAME})\s+Vs\.?\s+(?P<r>{_NAME})"
    r"(?=\s+and\s+others|\s+&|\s+Mr\.|\s+for\s+the|\s*$)"
)
COUNSEL_HDR = [
    re.compile(
        r"(?i)Counsel for (?:the )?(?P<side>Petitioner|Appellant|Applicant|Respondent)s?\s*:-?\s*"
        r"(?P<list>.+?)(?=Counsel for|Hon'?ble|CORAM|$)"
    ),
    re.compile(
        r"(?i)For (?:the )?(?P<side>Appellant|Petitioner|Applicant|Respondent|State)s?"
        r"(?:\s*No\.?\s*[\d,& ]+)?\s*:-?\s*(?P<list>.+?)"
        r"(?=For (?:the )?(?:Appellant|Petitioner|Applicant|Respondent|State)|Heard on|Judgment on|"
        r"Date of|CORAM|Hon'?ble|JUDGMENT|$)"
    ),
]
KARNATAKA = re.compile(
    r"(?s)BETWEEN\s*:?\s*(?P<a>.+?)\s+(?:\.{3,}\s*)?(?:APPELLANT|PETITIONER)S?\s+AND\s*:?\s*"
    r"(?P<r>.+?)\s+(?:\.{3,}\s*)?RESPONDENTS?"
)
ANCHOR = re.compile(
    r"JURISDICTION|OF\s+\d{4}\)?|NO\.?\s*[_\d/.\-]+(?:\s*OF\s*\d{4})?\)?|"
    r"\(Arising out of .*?\)|IN THE MATTER OF\s*:?|BETWEEN\s*:?|\bWITH\b|\n",
    re.I | re.S,
)
ENUM = re.compile(r"(?:^|\s)\d+[.\)\]]\s+")
AND_ENUM = re.compile(r"\s+(?:&|and)\s+(?=\d+[.\)\]])", re.I)
AT_SPLIT = re.compile(r"\s+@\s+")
INNER_NAME = re.compile(r"\((?:SHRI|SMT|Sri|Shri|Smt)\.?\s+([^)]+)\)")
CUTS = [
    re.compile(
        r"\b(?:S/o|W/o|D/o|wd/o|Aged|Age|Occu|R/o|resident|Residing|through|having|"
        r"A COMPANY|REP(?:\.|RESENTED)? BY|rep\. by|its)\b",
        re.I,
    ),
    re.compile(r"\s*(?:&|and)\s*(?:ORS|Ors|ANR|Anr|Others|others|another|Another)\b\.?", re.I),
    re.compile(r"\s*\((?:DEAD|D)\).*", re.I | re.S),
    re.compile(r"\s*BY\s+L\.?R'?S?\.?.*", re.I | re.S),
    re.compile(r"…"),
    re.compile(r"\.{2,}"),
]
ORG_WORD = re.compile(
    r"\b(?:LTD|LIMITED|COMPANY|CORPORATION|BOARD|STATE|UNION|ASSOCIATION|TRUST|BANK|"
    r"SOCIETY|COMMISSION|AUTHORITY)\b",
    re.I,
)
THE_LEAD = re.compile(r"(?:THE\s+)", re.I)
MS_LEAD = re.compile(r"^(?:M/s\.?|M/S\.?)\s*", re.I)
COUNSEL_SPLIT = re.compile(r",\s*|\s+and\s+|\s*&\s*", re.I)
COUNSEL_SUF = re.compile(
    r"\s*(?:Sr\.?\s*Adv(?:ocate)?\.?|Adv(?:ocate)?\.?|Senior Advocate|AOR|learned\b.*|counsel\b.*)$",
    re.I,
)
BENCH_SPLIT = re.compile(r"(?:\s*\((?:J|CJ)\)\s*)|\s+&\s+|\s+BENCH:\s*", re.I)
CORAM = re.compile(
    r"CORAM:\s*(?:HON'?BLE\s+)?(?:MR\.|MRS\.|MS\.)?\s*JUSTICE\s+(?P<j>[A-Z][A-Z .]+)",
    re.I,
)
BEFORE = re.compile(
    r"BEFORE THE HON'?BLE(?:\s+(?:MR\.|MRS\.|MS\.|DR\.))?\s*JUSTICE\s+(?P<j>.+?)"
    r"(?:\s+AND THE HON'?BLE(?:\s+(?:MR\.|MRS\.|MS\.|DR\.))?\s*JUSTICE\s+(?P<j2>.+?))?"
    r"(?=\s+(?:AND|DATED|WRIT|CIVIL|CRIMINAL|APPEAL)|$)",
    re.I,
)
OPEN_J = re.compile(
    r"(?<![A-Za-z])(?P<j>(?:[A-Z][\w.']{0,24}\s+){0,4}[A-Z][\w.']{1,24}),\s*J\.\s"
)
SIG_J = re.compile(r"\.{3,}\s*,?\s*J\.?\s*\((?P<j>[A-Z][A-Z .]+)\)")
PWMARK = r"(?:P|D|C)\.?\s?W\.?\s*[-.:]?\s*\d+"
WIT_BEFORE = re.compile(
    rf"(?P<name>(?:(?:Smt|Shri|Sri|Dr|Mr|Mrs)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}})\s*[\(\[,]?\s*{PWMARK}"
)
WIT_AFTER = re.compile(
    rf"{PWMARK}\s*[\)\],:\-]?\s*(?:(?:Smt|Shri|Sri|Dr|Mr|Mrs)\.?\s+)?"
    rf"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}})"
)
WIT_STOP = {
    "the", "this", "he", "she", "in", "on", "it", "exhibit", "ex", "court",
    "state", "accused", "deceased", "witness",
}
COUNSEL_FWD = re.compile(
    r"(?:Mr|Ms|Mrs|Shri|Smt|Dr)\.?\s+(?P<name>(?:[A-Z][\w.'-]*\s?){1,5}),?\s+"
    r"(?:learned\s+)?(?:senior\s+)?(?:counsel|advocate|Sr\.?\s*Adv\.?|ASG|AAG|"
    r"(?:Additional\s+)?Solicitor General|Standing Counsel|Panel Lawyer)"
    r"[^.]{0,60}?\bfor\s+(?:the\s+)?(?P<side>appellant|petitioner|applicant|complainant|"
    r"respondent|State|Union)",
    re.I,
)
COUNSEL_REV = re.compile(
    r"counsel for the (?P<side>appellant|petitioner|applicant|complainant|respondent|State|Union)s?"
    r"\s*,?\s*(?:Mr|Ms|Shri|Smt)\.?\s+(?P<name>(?:[A-Z][\w.'-]*\s?){1,5})",
    re.I,
)
PARA = re.compile(r"(?:\n\s*1\.\s|\b1\.\s+(?:Leave granted|This|The|In|Heard))")
A_SIDE = re.compile(
    r"APPELLANT|PETITIONER|APPLICANT|PLAINTIFF|COMPLAINANT|REVISIONIST", re.I
)
HEADER_LIMIT = 4000


@dataclass
class Party:
    name: str
    side: str
    kind: str
    start: int
    end: int
    fmt: str


@dataclass
class Header:
    parties: list[Party] = field(default_factory=list)
    end: int = 0
    fmt: str = "NONE"


def norm_key(name: str) -> str:
    """Casefold, drop a leading honorific, and drop punctuation."""
    text = name.strip()
    while True:
        m = HONORIFIC.match(text)
        if not m:
            break
        text = text[m.end():]
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _side_of(label: str) -> str:
    if re.search(r"state|respondent|defendant|opposite|non-applicant|accused|union", label, re.I):
        return "R"
    if A_SIDE.search(label):
        return "A"
    return ""


def _region_end(text: str, party_end: int) -> int:
    limit = min(HEADER_LIMIT, len(text))
    if party_end <= 0:
        return limit
    m = PARA.search(text, party_end, limit)
    if m:
        return m.start()
    return limit


def _after_anchor(fragment: str, abs_start: int) -> tuple[str, int]:
    last = None
    for m in ANCHOR.finditer(fragment):
        last = m
    if last is None:
        return fragment, abs_start
    return fragment[last.end():], abs_start + last.end()


def _cut_piece(fragment: str) -> str:
    cut_at = len(fragment)
    for pat in CUTS:
        m = pat.search(fragment)
        if m:
            cut_at = min(cut_at, m.start())
    return fragment[:cut_at]


def _strip_the(fragment: str) -> str:
    m = THE_LEAD.match(fragment)
    if not m:
        return fragment
    rest = fragment[m.end():]
    if len(rest.split()) <= 4 and not ORG_WORD.search(rest):
        return rest
    return fragment


def _strip_honorific(fragment: str) -> str:
    fragment = re.sub(r"^(?:Ku|Km)\.?\s+", "", fragment)
    while True:
        m = HONORIFIC.match(fragment)
        if not m:
            return fragment
        fragment = fragment[m.end():]


def _clean_name(fragment: str, strip_ms: bool) -> str:
    fragment = _cut_piece(fragment).strip()
    if strip_ms:
        fragment = MS_LEAD.sub("", fragment)
    fragment = _strip_the(fragment)
    fragment = _strip_honorific(fragment)
    return fragment.strip(" \t,;:")


def _emit_name(fragment: str, abs_start: int, side: str, kind: str, fmt: str, strip_ms: bool) -> list[Party]:
    """Turn one raw piece into one or two parties, with offsets into the original text."""
    raw = fragment
    base = abs_start
    inner = INNER_NAME.search(raw)
    pieces: list[tuple[str, int]] = []
    if inner and kind == "party":
        pieces.append((inner.group(1), base + inner.start(1)))
        after = raw[inner.end():]
        if after.strip():
            pieces.append((after, base + inner.end()))
    else:
        pieces.append((raw, base))
    out = []
    for piece, origin in pieces:
        cleaned = _clean_name(piece, strip_ms)
        if len(re.sub(r"[^A-Za-z]", "", cleaned)) < 2 or len(cleaned.split()) > 14:
            continue
        # Locate the cleaned surface inside the original piece (first occurrence).
        at = piece.find(cleaned)
        if at < 0:
            at = piece.lower().find(cleaned.lower())
        if at < 0:
            continue
        start = origin + at
        end = start + len(cleaned)
        out.append(Party(cleaned, side, kind, start, end, fmt))
        if cleaned.endswith("."):
            bare = cleaned[:-1].rstrip()
            if len(re.sub(r"[^A-Za-z]", "", bare)) >= 2:
                out.append(Party(bare, side, kind, start, start + len(bare), fmt))
    return out


def _split_block(block: str, abs_start: int) -> list[tuple[str, int]]:
    """Split a party block into (piece, absolute start) pairs."""
    if not block:
        return []
    # Separators are dropped: a piece ends where a marker starts and the next
    # piece begins where that marker ends.
    spans = [(0, len(block))]
    markers = [(m.start(), m.end()) for m in ENUM.finditer(block)]
    markers += [(m.start(), m.end()) for m in AND_ENUM.finditer(block)]
    markers += [(m.start(), m.end()) for m in AT_SPLIT.finditer(block)]
    for a, b in sorted(markers):
        nxt = []
        for s, e in spans:
            if b <= s or a >= e:
                nxt.append((s, e))
                continue
            if s < a:
                nxt.append((s, a))
            if b < e:
                nxt.append((b, e))
        spans = nxt
    pieces = []
    for a, b in spans:
        frag = block[a:b]
        if frag.strip():
            pieces.append((frag, abs_start + a))
    return pieces or [(block, abs_start)]


def _tail(fragment: str, abs_start: int) -> tuple[str, int]:
    """A role word can sit hundreds of characters after the caption. Keep the last clause."""
    if len(fragment) <= 90:
        return fragment, abs_start
    cuts = [m.end() for m in re.finditer(r"\s{2,}|\s\+\s+|\n|(?:19|20)\d{2}\s+", fragment)]
    if not cuts:
        return fragment, abs_start
    at = cuts[-1]
    return fragment[at:], abs_start + at


def _parties_from_block(block: str, abs_start: int, side: str, fmt: str, strip_ms: bool) -> list[Party]:
    if fmt == "F-ROLEWORD":
        block, abs_start = _after_anchor(block, abs_start)
    block, abs_start = _tail(block, abs_start)
    out = []
    for piece, origin in _split_block(block, abs_start):
        out.extend(_emit_name(piece, origin, side, "party", fmt, strip_ms))
    return out


def _counsel_pieces(listing: str, abs_start: int, side: str, fmt: str) -> list[Party]:
    out = []
    pos = 0
    for part in COUNSEL_SPLIT.split(listing):
        at = listing.find(part, pos)
        if at < 0:
            at = pos
        pos = at + len(part)
        cleaned = COUNSEL_SUF.sub("", part).strip(" \t,;:.")
        cleaned = _strip_honorific(cleaned).strip()
        if len(re.sub(r"[^A-Za-z]", "", cleaned)) < 2:
            continue
        rel = part.find(cleaned)
        if rel < 0:
            continue
        start = abs_start + at + rel
        out.append(Party(cleaned, side, "counsel", start, start + len(cleaned), fmt))
    return out


def _bench_names(block: str, abs_start: int, fmt: str) -> list[Party]:
    out = []
    parts = BENCH_SPLIT.split(block)
    cursor = 0
    for part in parts:
        at = block.find(part, cursor)
        if at < 0:
            at = cursor
        cursor = at + len(part)
        name = part.strip(" \t,;:")
        if len(re.sub(r"[^A-Za-z]", "", name)) < 2:
            continue
        rel = part.find(name)
        start = abs_start + at + max(rel, 0)
        out.append(Party(name, "", "judge", start, start + len(name), fmt))
    return out


def _from_sci(window: str, strip_ms: bool) -> list[Party]:
    out = []
    for m in SCI.finditer(window):
        out.extend(_parties_from_block(m.group("a"), m.start("a"), "A", "F-SCI", strip_ms))
        out.extend(_parties_from_block(m.group("r"), m.start("r"), "R", "F-SCI", strip_ms))
    for m in BENCH.finditer(window):
        out.extend(_bench_names(m.group("j"), m.start("j"), "F-SCI"))
    return out


def _from_roleword(window: str, strip_ms: bool) -> list[Party]:
    out = []
    for m in ROLEW.finditer(window):
        out.extend(_parties_from_block(m.group("a"), m.start("a"), "A", "F-ROLEWORD", strip_ms))
        out.extend(_parties_from_block(m.group("r"), m.start("r"), "R", "F-ROLEWORD", strip_ms))
        if m.group("ca"):
            out.extend(_counsel_pieces(m.group("ca"), m.start("ca"), "A", "F-ROLEWORD"))
        if m.group("cr"):
            out.extend(_counsel_pieces(m.group("cr"), m.start("cr"), "R", "F-ROLEWORD"))
    return out


def _from_colon(window: str, strip_ms: bool) -> list[Party]:
    out = []
    for m in COLON.finditer(window):
        out.extend(_parties_from_block(m.group("a"), m.start("a"), "A", "F-COLON", strip_ms))
        out.extend(_parties_from_block(m.group("r"), m.start("r"), "R", "F-COLON", strip_ms))
    return out


def _from_karnataka(window: str, strip_ms: bool) -> list[Party]:
    out = []
    for m in KARNATAKA.finditer(window):
        out.extend(_parties_from_block(m.group("a"), m.start("a"), "A", "F-NONE", strip_ms))
        out.extend(_parties_from_block(m.group("r"), m.start("r"), "R", "F-NONE", strip_ms))
    return out


def _counsel_headers(window: str) -> list[Party]:
    out = []
    for pat in COUNSEL_HDR:
        for m in pat.finditer(window):
            out.extend(_counsel_pieces(m.group("list"), m.start("list"), _side_of(m.group("side")), "F-COUNSEL"))
    return out


def _judges(text: str, limit: int) -> list[Party]:
    out = []
    region = text[:limit]
    for pat, fmt in ((CORAM, "F-CORAM"), (BEFORE, "F-BEFORE"), (OPEN_J, "F-J")):
        for m in pat.finditer(region):
            for grp in ("j", "j2"):
                if grp not in m.groupdict() or not m.group(grp):
                    continue
                name = m.group(grp).strip(" \t,;:")
                if len(re.sub(r"[^A-Za-z]", "", name)) < 2 or re.search(r"judgment|j u d", name, re.I):
                    continue
                start = m.start(grp)
                out.append(Party(name, "", "judge", start, start + len(name), fmt))
    for m in SIG_J.finditer(text):
        name = m.group("j").strip(" \t,;:")
        if len(re.sub(r"[^A-Za-z]", "", name)) >= 2:
            out.append(Party(name, "", "judge", m.start("j"), m.start("j") + len(name), "F-SIG"))
    return out


def _dominant(parties: list[Party]) -> str:
    counts: dict[str, int] = {}
    for p in parties:
        if p.kind == "party":
            counts[p.fmt] = counts.get(p.fmt, 0) + 1
    if not counts:
        return "NONE"
    return max(counts, key=counts.get)


def parse_header(text: str, strip_ms: bool = False) -> Header:
    window = text[:HEADER_LIMIT]
    found: list[Party] = []
    found += _from_sci(window, strip_ms)
    found += _from_roleword(window, strip_ms)
    found += _from_colon(window, strip_ms)
    found += _from_karnataka(window, strip_ms)
    if not any(p.kind == "party" for p in found):
        for m in BARE_VS.finditer(window[:2000]):
            found += _parties_from_block(m.group("a"), m.start("a"), "A", "F-BARE", strip_ms)
            found += _parties_from_block(m.group("r"), m.start("r"), "R", "F-BARE", strip_ms)
    found += _counsel_headers(window)
    party_end = max((p.end for p in found if p.kind == "party"), default=0)
    end = _region_end(text, party_end)
    found += _judges(text, max(end, min(HEADER_LIMIT, len(text))))
    kept = [p for p in found if p.kind == "judge" or p.start < end]
    # Drop duplicate (start, end, kind, side) rows from overlapping patterns.
    seen = set()
    unique = []
    for p in kept:
        key = (p.start, p.end, p.kind, p.side, p.name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return Header(unique, end, _dominant(unique))


def _strip_name_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    name = text[start:end]
    m = HONORIFIC.match(name)
    if m and m.end() < len(name):
        start += m.end()
        name = text[start:end]
    name = name.strip(" \t,;:")
    return name, start, start + len(name)


def witness_cues(text: str) -> list[Party]:
    out = []
    seen = set()
    for pat in (WIT_BEFORE, WIT_AFTER):
        for m in pat.finditer(text):
            name, start, end = _strip_name_span(text, m.start("name"), m.end("name"))
            if norm_key(name) in WIT_STOP or len(re.sub(r"[^A-Za-z]", "", name)) < 2:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            out.append(Party(name, "", "witness", start, end, "F-WIT"))
    return out


def counsel_cues(text: str) -> list[Party]:
    out = []
    for pat in (COUNSEL_FWD, COUNSEL_REV):
        for m in pat.finditer(text):
            name, start, end = _strip_name_span(text, m.start("name"), m.end("name"))
            name = COUNSEL_SUF.sub("", name).strip(" \t,;:.")
            end = start + len(name)
            if len(re.sub(r"[^A-Za-z]", "", name)) < 2:
                continue
            out.append(Party(name, _side_of(m.group("side")), "counsel", start, end, "F-CUE"))
    return out


def cast_list(text: str, strip_ms: bool = False) -> str:
    """The parsed cast line fed to every window when header_mode is 'parsed'."""
    header = parse_header(text, strip_ms=strip_ms)
    wits = witness_cues(text)

    def join(pred) -> str:
        names = []
        for p in header.parties:
            if pred(p) and p.name not in names:
                names.append(p.name)
        return "; ".join(names)

    wit_names = []
    for p in wits:
        if p.name not in wit_names:
            wit_names.append(p.name)
    return (
        f"APPELLANT: {join(lambda p: p.kind == 'party' and p.side == 'A')} | "
        f"RESPONDENT: {join(lambda p: p.kind == 'party' and p.side == 'R')} | "
        f"A-COUNSEL: {join(lambda p: p.kind == 'counsel' and p.side == 'A')} | "
        f"R-COUNSEL: {join(lambda p: p.kind == 'counsel' and p.side == 'R')} | "
        f"BENCH: {join(lambda p: p.kind == 'judge')} | "
        f"WITNESS: {'; '.join(wit_names)}"
    )
