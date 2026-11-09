"""Phase-3 documents, sentence windows, and the 5-fold split.

A window is a run of whole sentences, each prefixed with <|sent|>. Dense and CED
windows stop at 2,048 tokens. Echo windows stop at 1,024 because the backbone
repeats the window. The next window starts at the first sentence that begins
after the midpoint of the current one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from saul.ner.data import load_tokenizer
from saul.ner.registry import repo_root

SENT_ID = 10
PAD_ID = 2
SENT_BUDGET = 128
HEAD_KEEP = 96
TAIL_KEEP = 32
DENSE_LIMIT = 2048
ECHO_LIMIT = 1024

LABELS = [
    "Fact",
    "Issue",
    "ArgumentPetitioner",
    "ArgumentRespondent",
    "Statute",
    "Dissent",
    "PrecedentReliedUpon",
    "PrecedentNotReliedUpon",
    "PrecedentOverruled",
    "RulingByLowerCourt",
    "RatioOfTheDecision",
    "RulingByPresentCourt",
    "None",
]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
N_LABELS = len(LABELS)
RARE_LABELS = {
    "Issue",
    "None",
    "RulingByPresentCourt",
    "PrecedentNotReliedUpon",
    "Dissent",
    "Statute",
    "RulingByLowerCourt",
}
FROZEN_BIAS = ("Dissent", "PrecedentOverruled")
DOMAIN_ID = {"CL": 0, "IT": 1}


def p3_dir() -> str:
    return os.environ.get("SAUL_P3_DIR", "/data/cs781-p3")


def raw_dir() -> str:
    env = os.environ.get("SAUL_RR_DIR")
    if env:
        return env
    candidates = [
        os.path.join(p3_dir(), "raw", "Phase_3_RR"),
        os.path.join(repo_root(), "data", "raw", "Phase_3_RR"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def assert_vocab(path: str | None = None) -> list[str]:
    path = path or os.path.join(raw_dir(), "Train", "label_vocab.json")
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        labels = [name for name, _ in sorted(data.items(), key=lambda kv: kv[1])]
    else:
        labels = list(data)
    if labels != LABELS:
        raise ValueError(f"label vocab mismatch: {labels}")
    return labels


@dataclass
class RRDoc:
    doc_id: str
    domain: str
    sentences: list[str]
    labels: list[str] | None
    split: str

    @property
    def label_ids(self) -> list[int] | None:
        if self.labels is None:
            return None
        return [LABEL_ID[lab] for lab in self.labels]


@dataclass
class Window:
    doc_id: str
    domain: str
    input_ids: list[int]
    sent_index: list[int]
    marker_pos: list[int]
    tok_start: list[int]
    tok_end: list[int]
    labels: list[int]
    rel_pos: list[float]
    window_pos: list[float]
    rare: bool
    split: str = ""
    teacher: np.ndarray | None = field(default=None)


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _from_records(records, domain: str, split: str, labeled: bool) -> list[RRDoc]:
    docs = []
    if isinstance(records, dict):
        items = [{"id": i, "text": s} for i, s in records.items()]
    else:
        items = records
    for rec in items:
        labels = rec.get("labels") if labeled else None
        docs.append(RRDoc(str(rec["id"]), domain, list(rec["text"]), labels, split))
    return docs


def load_docs(root: str | None = None) -> list[RRDoc]:
    root = root or raw_dir()
    specs = [
        ("Train/CL_train.json", "CL", "train", True),
        ("Train/CL_dev.json", "CL", "dev", True),
        ("Train/IT_train.json", "IT", "train", True),
        ("Train/IT_dev.json", "IT", "dev", True),
        ("Test/CL_test.json", "CL", "test", False),
        ("Test/IT_test.json", "IT", "test", False),
    ]
    docs = []
    for name, domain, split, labeled in specs:
        docs.extend(_from_records(_read_json(os.path.join(root, name)), domain, split, labeled))
    return docs


def labeled_docs(docs: list[RRDoc]) -> list[RRDoc]:
    return [d for d in docs if d.split in ("train", "dev")]


def make_folds(docs: list[RRDoc], n_folds: int = 5, seed: int = 781) -> list[list[str]]:
    """9 CL + 9 IT per fold. Official dev documents are dealt one per fold first."""
    del seed  # assignment is deterministic; the seed is part of the recorded spec
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for domain in ("CL", "IT"):
        group = [d for d in docs if d.domain == domain and d.split in ("train", "dev")]
        devs = sorted((d for d in group if d.split == "dev"), key=lambda d: (-len(d.sentences), d.doc_id))
        trains = sorted((d for d in group if d.split == "train"), key=lambda d: (-len(d.sentences), d.doc_id))
        if len(devs) > n_folds:
            raise ValueError(f"{domain} has {len(devs)} dev docs, expected at most {n_folds}")
        for i, doc in enumerate(devs):
            folds[i % n_folds].append(doc.doc_id)
        load = [0] * n_folds
        taken = [0] * n_folds
        quota = len(trains) // n_folds
        for doc in trains:
            choices = [i for i in range(n_folds) if taken[i] < quota]
            if not choices:
                choices = list(range(n_folds))
            pick = min(choices, key=lambda i: (load[i], i))
            folds[pick].append(doc.doc_id)
            load[pick] += len(doc.sentences)
            taken[pick] += 1
    return folds


def ensure_folds(docs: list[RRDoc] | None = None, path: str | None = None) -> list[list[str]]:
    path = path or os.path.join(p3_dir(), "folds.json")
    if os.path.exists(path) and not os.environ.get("SAUL_RR_REBUILD_FOLDS"):
        return json.load(open(path, encoding="utf-8"))
    docs = labeled_docs(docs or load_docs())
    folds = make_folds(docs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    return folds


def truncate_ids(ids: list[int]) -> list[int]:
    """Keep the first 96 and last 32 content tokens. The marker is added by the caller.

    96 + 32 = 128 content tokens, which is the sentence budget in the plan. The marker
    sits in front of that slice.
    """
    if len(ids) <= HEAD_KEEP + TAIL_KEEP:
        return list(ids)
    return list(ids[:HEAD_KEEP]) + list(ids[-TAIL_KEEP:])


def encode_sentence(tok, text: str) -> list[int]:
    if callable(tok) and not hasattr(tok, "encode"):
        ids = list(tok(text))
    else:
        ids = [int(i) for i in tok.encode(text, add_special_tokens=False).ids]
    return truncate_ids(ids)


def _piece(content: list[int]) -> tuple[list[int], int, int, int]:
    """Return ids, marker position, content start, content end (end exclusive)."""
    ids = [SENT_ID] + list(content)
    marker = 0
    start = 1
    end = len(ids)
    if end == start:
        start = end = marker
    return ids, marker, start, end


def window_limit(context: str, override: int | None = None) -> int:
    if override:
        return int(override)
    if context == "echo":
        return ECHO_LIMIT
    return DENSE_LIMIT


def build_windows(doc: RRDoc, pieces: list[list[int]], max_tokens: int, sentence_only: bool = False) -> list[Window]:
    n = len(doc.sentences)
    if n == 0:
        return []
    spans = []
    for content in pieces:
        ids, marker, start, end = _piece(content)
        spans.append((ids, marker, start, end))
    windows: list[Window] = []
    i = 0
    guard = 0
    while i < n and guard <= n + 2:
        guard += 1
        blob: list[int] = []
        chosen: list[int] = []
        local_marker: list[int] = []
        local_start: list[int] = []
        local_end: list[int] = []
        j = i
        while j < n:
            ids, marker, start, end = spans[j]
            if sentence_only and chosen:
                break
            if blob and len(blob) + len(ids) > max_tokens:
                break
            shift = len(blob)
            use = ids if len(blob) + len(ids) <= max_tokens else ids[: max(1, max_tokens - len(blob))]
            blob.extend(use)
            chosen.append(j)
            local_marker.append(shift + marker)
            local_start.append(shift + start)
            local_end.append(min(shift + end, len(blob)))
            j += 1
            if sentence_only:
                break
            if len(use) < len(ids):
                break
        if not chosen:
            break
        m = len(chosen)
        labels = [-100] * m if doc.labels is None else [LABEL_ID[doc.labels[k]] for k in chosen]
        names = [] if doc.labels is None else [doc.labels[k] for k in chosen]
        rel = [chosen[t] / max(n - 1, 1) for t in range(m)]
        wpos = [t / max(m - 1, 1) for t in range(m)]
        windows.append(Window(
            doc_id=doc.doc_id,
            domain=doc.domain,
            input_ids=blob,
            sent_index=chosen,
            marker_pos=local_marker,
            tok_start=local_start,
            tok_end=local_end,
            labels=labels,
            rel_pos=rel,
            window_pos=wpos,
            rare=any(name in RARE_LABELS for name in names),
            split=doc.split,
        ))
        if j >= n:
            break
        mid = 0.5 * len(blob)
        acc = 0
        nxt = chosen[0] + 1
        for t, sidx in enumerate(chosen):
            if acc >= mid and sidx > chosen[0]:
                nxt = sidx
                break
            acc += local_end[t] - local_marker[t]
        if nxt <= i:
            nxt = i + 1
        i = nxt
    return windows


def windows_for(doc: RRDoc, tok=None, max_tokens: int = DENSE_LIMIT, sentence_only: bool = False,
                encode=None) -> list[Window]:
    encode = encode or (lambda text: encode_sentence(tok, text))
    pieces = [encode(s) for s in doc.sentences]
    return build_windows(doc, pieces, max_tokens, sentence_only)


def doc_windows(docs: list[RRDoc], tok=None, context: str = "bidirectional", sentence_only: bool = False,
                max_tokens: int | None = None) -> list[Window]:
    limit = window_limit(context, max_tokens)
    tok = tok or load_tokenizer()
    out = []
    for doc in docs:
        out.extend(windows_for(doc, tok, limit, sentence_only))
    return out


def epoch_windows(windows: list[Window], rng: np.random.Generator, rare_boost: bool = True) -> list[Window]:
    order = list(windows)
    if rare_boost:
        order.extend([w for w in windows if w.rare])
    if not order:
        return []
    perm = rng.permutation(len(order))
    return [order[int(i)] for i in perm]


def class_weights(docs: list[RRDoc]) -> torch.Tensor:
    counts = np.zeros(N_LABELS, dtype=np.float64)
    for doc in docs:
        if doc.labels is None:
            continue
        for lab in doc.labels:
            counts[LABEL_ID[lab]] += 1
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def collate(windows: list[Window]) -> dict[str, torch.Tensor]:
    width = max(len(w.input_ids) for w in windows)
    # flex_attention's compiled kernel has no backend for some lengths below one
    # block (E0 died at T=127). Round up to a multiple of 128 so short
    # sentence-only batches share one shape. Pad positions keep doc id -1.
    width = max(128, ((width + 127) // 128) * 128)
    sent = max(len(w.sent_index) for w in windows)
    b = len(windows)
    input_ids = torch.full((b, width), PAD_ID, dtype=torch.long)
    doc_ids = torch.full((b, width), -1, dtype=torch.long)
    marker = torch.zeros(b, sent, dtype=torch.long)
    tok_start = torch.zeros(b, sent, dtype=torch.long)
    tok_end = torch.zeros(b, sent, dtype=torch.long)
    labels = torch.full((b, sent), -100, dtype=torch.long)
    rel = torch.zeros(b, sent)
    wpos = torch.zeros(b, sent)
    mask = torch.zeros(b, sent, dtype=torch.bool)
    domain = torch.zeros(b, dtype=torch.long)
    teacher = torch.zeros(b, sent, N_LABELS)
    has_teacher = torch.zeros(b, sent, dtype=torch.bool)
    for i, w in enumerate(windows):
        n = len(w.input_ids)
        m = len(w.sent_index)
        input_ids[i, :n] = torch.tensor(w.input_ids, dtype=torch.long)
        doc_ids[i, :n] = 0
        marker[i, :m] = torch.tensor(w.marker_pos, dtype=torch.long)
        tok_start[i, :m] = torch.tensor(w.tok_start, dtype=torch.long)
        tok_end[i, :m] = torch.tensor(w.tok_end, dtype=torch.long)
        labels[i, :m] = torch.tensor(w.labels, dtype=torch.long)
        rel[i, :m] = torch.tensor(w.rel_pos, dtype=torch.float32)
        wpos[i, :m] = torch.tensor(w.window_pos, dtype=torch.float32)
        mask[i, :m] = True
        domain[i] = DOMAIN_ID[w.domain]
        if w.teacher is not None:
            teacher[i, :m] = torch.tensor(w.teacher, dtype=torch.float32)
            has_teacher[i, :m] = True
    return {
        "input_ids": input_ids,
        "doc_ids": doc_ids,
        "marker_pos": marker,
        "tok_start": tok_start,
        "tok_end": tok_end,
        "labels": labels,
        "rel_pos": rel,
        "window_pos": wpos,
        "sent_mask": mask,
        "domain": domain,
        "teacher": teacher,
        "has_teacher": has_teacher,
    }


def triangle(index: int, length: int) -> float:
    if length <= 1:
        return 1.0
    mid = (length - 1) / 2.0
    return 0.1 + 0.9 * (1.0 - abs(index - mid) / mid)
