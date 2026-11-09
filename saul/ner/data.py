"""Phase-2 documents, token windows, BIOES targets, and the 5-fold split.

A window is [header tokens of the document] + <|sep|> + [a body slice]. Tags are
ignored on the header, the separator, and padding. Gold spans that do not land on
token boundaries are mapped to the tightest covering token span.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from saul.ner.labels import (
    IGNORE, LABELS, N_LABELS, N_SPAN, N_TAGS, NONE, O, PAD_ID, RARE_LABELS, SEP_ID,
    label_id, tag_id, tag_parts,
)
from saul.ner.registry import p2_dir, repo_root
from saul.tokenizer.train import load_raw_tokenizer

# <|cls|>. Reserved in the legal32k tokenizer and never emitted by encode().
PREAMBLE_ID = 5
HEADER_DENSE = 192
BODY_DENSE = 1792
STRIDE_DENSE = 1280
HEADER_ECHO = 128
BODY_ECHO = 896
STRIDE_ECHO = 640
MAX_SPAN = 64


@dataclass
class Doc:
    doc_id: str
    text: str
    origin: str
    ids: list[int]
    starts: list[int]
    ends: list[int]
    tags: list[int]
    gold: list[tuple[int, int, str]]
    rare_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Window:
    doc_id: str
    input_ids: list[int]
    char_starts: list[int]
    char_ends: list[int]
    tags: list[int]
    body_start: int
    body_len: int
    body_tok0: int
    doc_tok: int
    rare: bool
    empty: bool
    # body-local (token start inclusive, token end exclusive, label index)
    gold_spans: list[tuple[int, int, int]]
    # False for self-training windows: uncertain tokens stay IGNORE instead of being taught as O / none.
    supervise_none: bool = True
    teacher_probs: np.ndarray | None = None
    teacher_spans: list | None = None


def data_root() -> str:
    return os.environ.get("SAUL_LNER_DIR", os.path.join(repo_root(), "data", "raw", "Phase_2_LNER"))


def tokenizer_path() -> str:
    return os.environ.get("SAUL_TOKENIZER", os.path.join(repo_root(), "artifacts", "tokenizers", "legal32k"))


def load_tokenizer():
    return load_raw_tokenizer(tokenizer_path())


def _strip_start(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def tokenize_doc(tok, text: str) -> tuple[list[int], list[int], list[int]]:
    enc = tok.encode(text, add_special_tokens=False)
    ids, starts, ends = [], [], []
    for tid, (s, e) in zip(enc.ids, enc.offsets):
        if e <= s:
            continue
        s = _strip_start(text, s, e)
        if s >= e:
            continue
        ids.append(int(tid))
        starts.append(int(s))
        ends.append(int(e))
    return ids, starts, ends


def _cover(starts: list[int], ends: list[int], gs: int, ge: int) -> tuple[int, int] | None:
    """Tightest token span [i, j) covering the character span [gs, ge)."""
    i = j = None
    for t, (s, e) in enumerate(zip(starts, ends)):
        if e <= gs or s >= ge:
            continue
        if i is None:
            i = t
        j = t + 1
    if i is None:
        return None
    return i, j


def assign_tags(starts, ends, gold: list[tuple[int, int, str]]) -> tuple[list[int], int, int]:
    """Return token tags, the number of exact boundary matches, and the number of gold spans used."""
    tags = [O] * len(starts)
    exact = used = 0
    start_set = set(starts)
    end_set = set(ends)
    occupied = [False] * len(starts)
    for gs, ge, lab in gold:
        cover = _cover(starts, ends, gs, ge)
        if cover is None:
            continue
        i, j = cover
        if any(occupied[i:j]):
            continue
        li = label_id(lab)
        if j == i + 1:
            tags[i] = tag_id(li, "S")
        else:
            tags[i] = tag_id(li, "B")
            for t in range(i + 1, j - 1):
                tags[t] = tag_id(li, "I")
            tags[j - 1] = tag_id(li, "E")
        for t in range(i, j):
            occupied[t] = True
        used += 1
        if gs in start_set and ge in end_set and starts[i] == gs and ends[j - 1] == ge:
            exact += 1
    return tags, exact, used


def load_split(name: str, tok=None) -> list[Doc]:
    """name is 'train' (fold1+fold2) or 'test' (fold3)."""
    root = data_root()
    tok = tok or load_tokenizer()
    docs: list[Doc] = []
    if name == "test":
        text = json.load(open(os.path.join(root, "Test", "fold3-text.json"), encoding="utf-8"))
        pairs = [("test", text, {})]
    else:
        pairs = []
        for fold in ("fold1", "fold2"):
            text = json.load(open(os.path.join(root, "Train", f"{fold}-text.json"), encoding="utf-8"))
            gold = json.load(open(os.path.join(root, "Train", f"{fold}-gold.json"), encoding="utf-8"))
            pairs.append((fold, text, gold))
    for origin, texts, golds in pairs:
        for did, raw in texts.items():
            ids, starts, ends = tokenize_doc(tok, raw)
            spans = []
            for sp in golds.get(did, []):
                spans.append((int(sp[0]), int(sp[1]), sp[2]))
            spans.sort()
            tags, _, _ = assign_tags(starts, ends, spans)
            rare = {}
            for _, _, lab in spans:
                if lab in RARE_LABELS:
                    rare[lab] = rare.get(lab, 0) + 1
            docs.append(Doc(str(did), raw, origin, ids, starts, ends, tags, spans, rare))
    return docs


def make_folds(docs: list[Doc], n_folds: int = 5, seed: int = 781) -> list[list[str]]:
    """Greedy balance on origin and rare-label counts. Exactly len/n_folds docs each."""
    order = sorted(docs, key=lambda d: (-sum(d.rare_counts.values()), d.origin, d.doc_id))
    folds: list[list[Doc]] = [[] for _ in range(n_folds)]
    rare_tot = [0] * n_folds
    origin_tot = [{} for _ in range(n_folds)]
    for doc in order:
        def key(i: int):
            return (rare_tot[i], origin_tot[i].get(doc.origin, 0), len(folds[i]), i)
        i = min(range(n_folds), key=key)
        folds[i].append(doc)
        rare_tot[i] += sum(doc.rare_counts.values())
        origin_tot[i][doc.origin] = origin_tot[i].get(doc.origin, 0) + 1
    # Deterministic order inside a fold so split 0 is stable.
    rng = random.Random(seed)
    out = []
    for fold in folds:
        ids = [d.doc_id for d in fold]
        rng.shuffle(ids)
        out.append(ids)
    return out


def folds_path() -> str:
    return os.path.join(p2_dir(), "folds.json")


def save_folds(folds: list[list[str]], path: str | None = None) -> str:
    path = path or folds_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    return path


def load_folds(docs: list[Doc], path: str | None = None) -> list[list[str]]:
    path = path or folds_path()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    folds = make_folds(docs)
    try:
        save_folds(folds, path)
    except OSError:
        pass
    return folds


def window_shape(context: str, header: int | None = None) -> tuple[int, int, int]:
    if context == "echo":
        return HEADER_ECHO if header is None else header, BODY_ECHO, STRIDE_ECHO
    return HEADER_DENSE if header is None else header, BODY_DENSE, STRIDE_DENSE


def _parsed_header_ids(doc: Doc, tok, header_len: int) -> list[int]:
    from saul.ner.header import cast_list
    parsed, _, _ = tokenize_doc(tok, cast_list(doc.text))
    parsed = parsed[:min(96, header_len)]
    parsed = parsed + doc.ids[:header_len - len(parsed)]
    return parsed[:header_len]


def build_windows(doc: Doc, header_len: int, body_len: int, stride: int, header_mode: str = "raw",
                  tok=None, preamble: bool = False) -> list[Window]:
    n = len(doc.ids)
    if n == 0:
        return []
    if header_mode == "parsed":
        header_ids = _parsed_header_ids(doc, tok, header_len)
        h = len(header_ids)
    else:
        h = min(header_len, n)
        header_ids = doc.ids[:h]
    if n <= body_len:
        starts_at = [0]
    else:
        starts_at = list(range(0, n - body_len + 1, stride))
        if starts_at[-1] != n - body_len:
            starts_at.append(n - body_len)
    windows = []
    for b0 in starts_at:
        b1 = min(n, b0 + body_len)
        body_ids = doc.ids[b0:b1]
        body_tags = _close_edges(doc.tags[b0:b1])
        lead = [PREAMBLE_ID] if preamble and b0 == 0 else []
        ids = header_ids + [SEP_ID] + lead + body_ids
        pad = h + 1 + len(lead)
        cstarts = [-1] * pad + doc.starts[b0:b1]
        cends = [-1] * pad + doc.ends[b0:b1]
        tags = [IGNORE] * pad + body_tags
        rare = any(tag_is_rare(t) for t in body_tags)
        empty = all(t == O for t in body_tags)
        spans = _body_spans(body_tags)
        windows.append(Window(
            doc.doc_id, ids, cstarts, cends, tags, pad, b1 - b0, b0, n, rare, empty, spans, True,
        ))
    return windows


def _close_edges(tags: list[int]) -> list[int]:
    """An entity cut by the window edge would start on I/E or end on B/I, which the CRF
    scores at -1e4. Re-tag the cut piece as a complete entity."""
    tags = list(tags)
    if not tags:
        return tags
    kind, lab = tag_parts(tags[0])
    if kind == "I":
        tags[0] = tag_id(lab, "B") if len(tags) > 1 and tag_parts(tags[1])[0] in ("I", "E") else tag_id(lab, "S")
    elif kind == "E":
        tags[0] = tag_id(lab, "S")
    kind, lab = tag_parts(tags[-1])
    if kind == "I":
        tags[-1] = tag_id(lab, "E")
    elif kind == "B":
        tags[-1] = tag_id(lab, "S")
    return tags


def tag_is_rare(tag: int) -> bool:
    if tag <= 0:
        return False
    return LABELS[(tag - 1) // 4] in RARE_LABELS


def _body_spans(tags: list[int]) -> list[tuple[int, int, int]]:
    """Gold token spans inside one body, as (start, end exclusive, label index)."""
    out = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t <= 0:
            i += 1
            continue
        kind_i = (t - 1) % 4
        lab = (t - 1) // 4
        if kind_i == 3:  # S
            out.append((i, i + 1, lab))
            i += 1
            continue
        if kind_i == 0:  # B
            j = i + 1
            while j < len(tags) and tags[j] > 0 and (tags[j] - 1) // 4 == lab and (tags[j] - 1) % 4 in (1, 2):
                if (tags[j] - 1) % 4 == 2:
                    out.append((i, j + 1, lab))
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
            continue
        i += 1
    return out


def epoch_windows(windows: list[Window], rng: np.random.Generator, empty_p: float = 0.3,
                  rare_boost: bool = True) -> list[Window]:
    if not rare_boost:
        chosen = list(windows)
        rng.shuffle(chosen)
        return chosen
    rare = [w for w in windows if w.rare]
    empty = [w for w in windows if w.empty]
    rest = [w for w in windows if not w.rare and not w.empty]
    chosen = rest + rare + rare
    chosen += [w for w in empty if rng.random() < empty_p]
    rng.shuffle(chosen)
    return chosen


def collate(windows: list[Window], max_span: int = MAX_SPAN) -> dict[str, torch.Tensor]:
    width = max(len(w.input_ids) for w in windows)
    body = max(w.body_len for w in windows)
    b = len(windows)
    input_ids = torch.full((b, width), PAD_ID, dtype=torch.long)
    doc_ids = torch.full((b, width), -1, dtype=torch.long)
    tags = torch.full((b, width), IGNORE, dtype=torch.long)
    body_mask = torch.zeros(b, width, dtype=torch.bool)
    span = torch.full((b, body, max_span), IGNORE, dtype=torch.long)
    for i, w in enumerate(windows):
        n = len(w.input_ids)
        input_ids[i, :n] = torch.tensor(w.input_ids, dtype=torch.long)
        doc_ids[i, :n] = 0
        tags[i, :n] = torch.tensor(w.tags, dtype=torch.long)
        body_mask[i, w.body_start:w.body_start + w.body_len] = True
        if w.supervise_none:
            for off in range(max_span):
                last = w.body_len - off
                if last <= 0:
                    break
                span[i, :last, off] = NONE
        for s, e, lab in w.gold_spans:
            length = e - s
            if 1 <= length <= max_span and e <= w.body_len:
                span[i, s, length - 1] = lab
    batch = {
        "input_ids": input_ids,
        "doc_ids": doc_ids,
        "tags": tags,
        "body_mask": body_mask,
        "span_targets": span,
    }
    if any(w.teacher_probs is not None for w in windows):
        teacher = torch.zeros(b, body, N_TAGS, dtype=torch.float32)
        for i, w in enumerate(windows):
            if w.teacher_probs is not None:
                n = min(w.body_len, w.teacher_probs.shape[0])
                teacher[i, :n] = torch.tensor(np.asarray(w.teacher_probs[:n]), dtype=torch.float32)
        batch["teacher_probs"] = teacher
    return batch


def tag_weights(docs: list[Doc]) -> torch.Tensor:
    """Sqrt-inverse frequency over the 48 entity tags, mean 1. O stays at 1."""
    counts = np.ones(N_TAGS - 1, dtype=np.float64)
    for doc in docs:
        for t in doc.tags:
            if t > 0:
                counts[t - 1] += 1
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    full = np.ones(N_TAGS, dtype=np.float32)
    full[1:] = w.astype(np.float32)
    return torch.tensor(full)


def span_weights(docs: list[Doc]) -> torch.Tensor:
    counts = np.ones(N_LABELS, dtype=np.float64)
    for doc in docs:
        for _, _, lab in doc.gold:
            counts[label_id(lab)] += 1
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    full = np.empty(N_SPAN, dtype=np.float32)
    full[:N_LABELS] = w.astype(np.float32)
    full[NONE] = 0.3
    return torch.tensor(full)


def docs_for_ids(docs: list[Doc], ids: list[str]) -> list[Doc]:
    want = set(ids)
    return [d for d in docs if d.doc_id in want]


def windows_for(docs: list[Doc], context: str, header: int | None = None, header_mode: str = "raw",
                preamble: bool = False) -> list[Window]:
    h, b, s = window_shape(context, header)
    tok = load_tokenizer() if header_mode == "parsed" else None
    out = []
    for doc in docs:
        out.extend(build_windows(doc, h, b, s, header_mode=header_mode, tok=tok, preamble=preamble))
    return out


def swap_role_strings(doc: Doc, donors: list[Doc], rng: np.random.Generator) -> Doc:
    """Replace every whole-word hit of each role gold string with a same-label string from another document."""
    pool: dict[str, list[str]] = {lab: [] for lab in RARE_LABELS}
    for donor in donors:
        if donor.doc_id == doc.doc_id:
            continue
        for s, e, lab in donor.gold:
            if lab in pool:
                pool[lab].append(donor.text[s:e])
    mapping: dict[str, str] = {}
    for s, e, lab in doc.gold:
        if lab not in pool:
            continue
        frag = doc.text[s:e]
        if frag in mapping:
            continue
        choices = [c for c in pool[lab] if c and c != frag]
        if choices:
            mapping[frag] = str(choices[int(rng.integers(0, len(choices)))])
    if not mapping:
        return doc
    hits = []
    for frag, repl in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(frag) + r"(?![A-Za-z0-9])", doc.text):
            hits.append((m.start(), m.end(), repl))
    hits.sort()
    chosen = []
    last = -1
    for s, e, repl in hits:
        if s >= last:
            chosen.append((s, e, repl))
            last = e

    def map_point(p: int) -> int | None:
        delta = 0
        for s, e, repl in chosen:
            if p >= e:
                delta += len(repl) - (e - s)
            elif p > s:
                return None
            else:
                break
        return p + delta

    parts = []
    pos = 0
    for s, e, repl in chosen:
        parts.append(doc.text[pos:s])
        parts.append(repl)
        pos = e
    parts.append(doc.text[pos:])
    text = "".join(parts)
    gold = []
    replaced = {(s, e): repl for s, e, repl in chosen}
    consumed = set()
    for s, e, lab in doc.gold:
        if (s, e) in replaced:
            ns = map_point(s)
            if ns is None:
                continue
            gold.append((ns, ns + len(replaced[(s, e)]), lab))
            consumed.add((s, e))
            continue
        ns, ne = map_point(s), map_point(e)
        if ns is None or ne is None or ne <= ns:
            continue
        gold.append((ns, ne, lab))
    tok = load_tokenizer()
    ids, starts, ends = tokenize_doc(tok, text)
    tags, _, _ = assign_tags(starts, ends, gold)
    rare = {}
    for _, _, lab in gold:
        if lab in RARE_LABELS:
            rare[lab] = rare.get(lab, 0) + 1
    return Doc(doc.doc_id, text, doc.origin, ids, starts, ends, tags, gold, rare)
