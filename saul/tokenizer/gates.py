"""Tokenizer release gates (report_saul.md 6.6). All must pass before any pretraining run."""
from __future__ import annotations

import re
import statistics
from typing import Iterable

from tokenizers import Tokenizer

_DIGIT = re.compile(r"\d")


def gate_roundtrip(tok: Tokenizer, texts: Iterable[str]) -> dict:
    n = bad = 0
    examples = []
    for t in texts:
        n += 1
        ids = tok.encode(t, add_special_tokens=False).ids
        back = tok.decode(ids, skip_special_tokens=False)
        if back != t:
            bad += 1
            if len(examples) < 5:
                i = next((j for j, (x, y) in enumerate(zip(back, t)) if x != y), min(len(back), len(t)))
                examples.append({"orig": t[max(0, i - 40):i + 40], "back": back[max(0, i - 40):i + 40]})
    return {"n": n, "failures": bad, "pass": bad == 0, "examples": examples}


def token_boundaries(tok: Tokenizer, text: str) -> tuple[set[int], set[int]]:
    enc = tok.encode(text, add_special_tokens=False)
    starts, ends = set(), set()
    for s, e in enc.offsets:
        if e > s:
            starts.add(s)
            ends.add(e)
    return starts, ends


def gate_span_alignment(tok: Tokenizer, docs: dict[str, str], gold: dict[str, list], threshold: float = 0.995) -> dict:
    """Share of gold entity [start, end) offsets that coincide with token boundaries (raw text, offset_mapping)."""
    total = ok = 0
    per_label: dict[str, list[int]] = {}
    failures = []
    for did, spans in gold.items():
        text = docs.get(did)
        if text is None:
            continue
        starts, ends = token_boundaries(tok, text)
        for sp in spans:
            s, e, lab = int(sp[0]), int(sp[1]), sp[2]
            good = s in starts and e in ends
            total += 1
            ok += good
            pl = per_label.setdefault(lab, [0, 0])
            pl[0] += good
            pl[1] += 1
            if not good and len(failures) < 50:
                failures.append({"doc": did, "span": [s, e], "label": lab, "text": text[s:e],
                                 "context": text[max(0, s - 15):e + 15]})
    rate = ok / max(total, 1)
    return {"total": total, "aligned": ok, "rate": rate, "pass": rate > threshold,
            "per_label": {k: v[0] / max(v[1], 1) for k, v in per_label.items()}, "failures": failures}


def gate_compression(tok: Tokenizer, texts_by_source: dict[str, list[str]]) -> dict:
    out = {}
    for src, texts in texts_by_source.items():
        nb = nt = nw = 0
        for t in texts:
            nb += len(t.encode("utf-8"))
            nw += len(t.split())
            nt += len(tok.encode(t, add_special_tokens=False).ids)
        out[src] = {"bytes_per_token": nb / max(nt, 1), "tokens_per_word": nt / max(nw, 1), "tokens": nt}
    return out


def gate_continuation_length(tok: Tokenizer, truths: Iterable[str]) -> dict:
    lens = [len(tok.encode(" " + t, add_special_tokens=False).ids) for t in truths]
    lens.sort()
    if not lens:
        return {}
    p95 = lens[min(len(lens) - 1, int(0.95 * len(lens)))]
    return {"mean": statistics.mean(lens), "p95": p95, "max": lens[-1], "recommended_max_new_tokens": int(2 * p95)}


def gate_digits(tok: Tokenizer, n_special: int) -> dict:
    """No token may contain two digit characters (single-digit splitting)."""
    vocab = tok.get_vocab(with_added_tokens=False)
    bad = [s for s, i in vocab.items() if i >= n_special and len(_DIGIT.findall(s)) > 1]
    return {"multi_digit_tokens": bad[:20], "count": len(bad), "pass": not bad}


def gate_specials(tok: Tokenizer, specials: list[str], tags: list[str]) -> dict:
    wrong = [s for i, s in enumerate(specials) if tok.token_to_id(s) != i]
    split_tags = []
    for t in tags:
        ids = tok.encode(f"the {t} said", add_special_tokens=False).ids
        if tok.token_to_id(t) not in ids:
            split_tags.append(t)
    return {"wrong_ids": wrong, "tags_not_atomic": split_tags, "pass": not wrong and not split_tags}


def gate_no_unk(tok: Tokenizer, texts: Iterable[str]) -> dict:
    unk = tok.token_to_id("[UNK]")
    bad = 0
    for t in texts:
        if unk is not None and unk in tok.encode(t, add_special_tokens=False).ids:
            bad += 1
    return {"texts_with_unk": bad, "pass": bad == 0}
