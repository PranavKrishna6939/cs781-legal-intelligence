"""Windows, folds, truncation, markers."""
from __future__ import annotations

from collections import Counter

from saul.rr.data import (
    SENT_ID, RRDoc, build_windows, collate, make_folds, truncate_ids, windows_for,
)


def _doc(n: int, tokens: int = 10, domain: str = "CL", labels=None) -> RRDoc:
    sentences = [f"sentence {i} " + ("word " * 3) for i in range(n)]
    labs = labels or ["Fact"] * n
    return RRDoc("doc", domain, sentences, labs, "train")


def test_truncation_keeps_head_and_tail():
    ids = list(range(200))
    kept = truncate_ids(ids)
    assert kept[:96] == ids[:96]
    assert kept[-32:] == ids[-32:]
    assert len(kept) == 128
    assert len([SENT_ID] + kept) == 129


def test_short_sentence_is_unchanged():
    ids = [3, 4, 5]
    assert truncate_ids(ids) == ids


def test_windows_cover_every_sentence_about_twice():
    pieces = [[1] * 10 for _ in range(30)]
    doc = RRDoc("d", "CL", [str(i) for i in range(30)], ["Fact"] * 30, "train")
    windows = build_windows(doc, pieces, max_tokens=44)
    counts = Counter()
    for window in windows:
        counts.update(window.sent_index)
        for pos in window.marker_pos:
            assert window.input_ids[pos] == SENT_ID
    assert set(counts) == set(range(30))
    mean = sum(counts.values()) / 30
    assert 1.5 <= mean <= 2.5


def test_sentence_only_is_one_sentence():
    doc = _doc(5)
    windows = windows_for(doc, encode=lambda text: [1, 2, 3], max_tokens=2048, sentence_only=True)
    assert len(windows) == 5
    assert all(len(w.sent_index) == 1 for w in windows)


def test_rare_flag():
    doc = _doc(3, labels=["Fact", "Issue", "Fact"])
    windows = windows_for(doc, encode=lambda text: [1], max_tokens=2048)
    assert any(w.rare for w in windows)


def test_folds_are_balanced_and_split_dev():
    docs = []
    for domain in ("CL", "IT"):
        for i in range(5):
            docs.append(RRDoc(f"{domain}_dev_{i}", domain, ["s"] * (10 + i), ["Fact"] * (10 + i), "dev"))
        for i in range(40):
            docs.append(RRDoc(f"{domain}_train_{i}", domain, ["s"] * (20 + (i % 7)), ["Fact"] * (20 + (i % 7)), "train"))
    folds = make_folds(docs)
    assert len(folds) == 5
    for fold in folds:
        assert sum(1 for i in fold if i.startswith("CL")) == 9
        assert sum(1 for i in fold if i.startswith("IT")) == 9
    for domain in ("CL", "IT"):
        owners = []
        for k, fold in enumerate(folds):
            for i in range(5):
                if f"{domain}_dev_{i}" in fold:
                    owners.append(k)
        assert len(set(owners)) == 5


def test_collate_pads_with_negative_doc_ids():
    doc = _doc(2)
    windows = windows_for(doc, encode=lambda text: [8, 9], max_tokens=4, sentence_only=True)
    windows[1].input_ids = windows[1].input_ids + [7]
    batch = collate(windows)
    assert int(batch["doc_ids"][0, -1]) == -1
    assert int(batch["input_ids"][0, 0]) == SENT_ID
    assert batch["domain"].tolist() == [0, 0]
