"""N1, splits, views, and the training-pool sampler."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from saul.cjpe.const import BOS, CLS
from saul.cjpe.data import (
    assert_splits, assign_bucket, assign_overlap, case_year, make_batches, normalise,
    sample_indices, split_halves, subset_indices, text_hash, view_chunks, view_head_tail, view_tail,
)


def _counts():
    words = Counter({"special": 10, "the": 50, "court": 20, "appeal": 8, "number": 4, "company": 4})
    bigrams = Counter({("the", "court"): 15, ("the", "appeal"): 3})
    return words, bigrams


def test_normalise_is_idempotent_and_scoped():
    words, bigrams = _counts()
    text = "Speci\nal Companyrt and the\ncourt NUMBERICE.\nThe company number stays."
    once = normalise(text, words, bigrams)
    assert once == normalise(once, words, bigrams)
    assert "company" not in once.split()[0]
    assert "the court" in once
    assert "the company number stays" in once
    assert "\n" not in once


def test_join_follows_the_bigram_rule():
    # Two ordinary words stay apart even when the concatenation occurs in train.
    words = Counter({"been": 100, "committed": 50, "beencommitted": 8, "the": 100, "t": 4, "he": 4})
    assert normalise("been\ncommitted", words, Counter()) == "been committed"
    assert normalise("t\nhe", words, Counter()) == "the"
    legal = Counter({"a": 5000, "lien": 40, "alien": 80, "of": 8000, "course": 30, "ofcourse": 20})
    assert normalise("a\nlien", legal, Counter()) == "a lien"
    assert normalise("of\ncourse", legal, Counter()) == "of course"


def test_join_uses_the_train_counts():
    words, bigrams = _counts()
    joins = []
    got = normalise("speci\nal", words, bigrams, joins)
    assert got == "special"
    assert joins == ["speci|al->special"]
    assert normalise("the\ncourt", words, bigrams) == "the court"


def test_overlap_and_disjoint_pools():
    dev = [
        {"id": "2010_1", "text": "same"},
        {"id": "2010_2", "text": "only dev"},
    ]
    test_keys = {("2010_1", text_hash("same"))}
    ovl, sel = assign_overlap(dev, test_keys)
    assert [row["id"] for row in ovl] == ["2010_1"]
    assert [row["id"] for row in sel] == ["2010_2"]
    assert_splits(["2010_1"], [text_hash("same")], ["2010_2"], [text_hash("only dev")],
                  ["2010_1"], [text_hash("same")], ["2010_1"])
    with pytest.raises(AssertionError):
        assert_splits(["2010_2"], [text_hash("only dev")], ["2010_2"], [text_hash("only dev")],
                      ["2010_9"], ["other"], ["2010_2"])


def test_views_keep_the_final_token_and_chunks_are_end_aligned():
    ids = np.arange(100, 500)
    tail = view_tail(ids, 32, causal=False)
    assert tail[0] == BOS and tail[-1] == 499
    causal = view_tail(ids, 32, causal=True)
    assert causal[-2] == 499 and causal[-1] == CLS
    chunks = view_chunks(ids, size=64, max_chunks=3, causal=False)
    assert len(chunks) == 3
    assert chunks[-1][-1] == 499
    assert chunks[-1][1] == 500 - 64
    both = view_head_tail(ids, 4, 4, causal=False)
    assert both[0] == BOS and both[-1] == 499 and 4 in both


def test_sampler_weights_subset_and_buckets():
    sources = ["train", "ovl", "train"]
    rng = np.random.default_rng(0)
    picked = sample_indices(sources, rng, ovl_weight=2)
    assert picked.count(1) == 2
    assert sample_indices(sources, np.random.default_rng(0), no_ovl=True).count(1) == 0
    refit = sample_indices(["train", "sel"], np.random.default_rng(1), ovl_weight=3, refit=True)
    assert refit.count(1) == 3
    sources = ["train"] * 20 + ["ovl"] * 3
    keep = subset_indices(sources, n_train=5, seed=781)
    assert sum(1 for i in keep if sources[i] == "ovl") == 3
    assert sum(1 for i in keep if sources[i] == "train") == 5
    labels = np.asarray([0, 0, 1, 1])
    years = np.asarray([1951, 1952, 2011, 2012])
    halves = split_halves(labels, years, seed=781)
    assert halves.count("A") == 2 and halves.count("B") == 2
    assert assign_bucket(100) == 1024
    assert assign_bucket(2048) == 2048
    batches = make_batches([100, 3000, 200], token_budget=2048)
    widths = {width for width, _idx in batches}
    assert widths == {1024, 4096}
    assert case_year("2014_12") == 2014
