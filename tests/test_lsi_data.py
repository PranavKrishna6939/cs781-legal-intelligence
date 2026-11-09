import numpy as np

from saul.lsi.const import CHUNK, MAX_CHUNKS
from saul.lsi.data import (
    collate, covered_tokens, dedupe_labels, dev_groups, g0_report, groups_intact, iterative_stratify,
    make_batches, rare_weights, select_segment_ids, view_chunks,
)


def test_dedupe_labels_keeps_order():
    assert dedupe_labels([3, 3, 1, 3, 1]) == [3, 1]


def test_halves_are_stratified_and_groups_stay_together():
    rows = []
    for i in range(200):
        rows.append({"id": str(i), "text": f"unique {i}", "labels": [i % 10, (i * 3) % 10]})
    rows.append({"id": "dup-a", "text": "same fact", "labels": [1]})
    rows.append({"id": "dup-b", "text": "same fact", "labels": [1, 2]})
    half = iterative_stratify(dev_groups(rows))
    assert len(half) == len(rows)
    assert groups_intact(rows, half)
    report = g0_report(rows, half)
    assert report["groups_intact"]
    # Every label that occurs at least twice can be split. Gap stays small.
    assert report["max_gap"] <= 4


def test_chunks_cover_a_16k_case_and_keep_the_tail_token():
    tokens = np.arange(MAX_CHUNKS * (CHUNK - 1), dtype=np.int64)
    chunks = view_chunks(tokens)
    assert len(chunks) == MAX_CHUNKS
    body = [tok for chunk in chunks for tok in chunk[1:]]
    assert body[0] == 0
    assert body[-1] == int(tokens[-1])
    assert covered_tokens(tokens) == len(tokens)


def test_segment_sampler_keeps_both_ends_in_order():
    rng = np.random.default_rng(0)
    chosen = select_segment_ids(20, 4, train=True, rng=rng)
    assert chosen[0] == 0 and chosen[-1] == 19
    assert chosen == sorted(chosen) and len(chosen) == 4
    inferred = select_segment_ids(20, 4, train=False, rng=rng)
    assert inferred == [0, 1, 18, 19]
    assert select_segment_ids(3, 8, train=True, rng=rng) == [0, 1, 2]


def test_rare_weights_and_batches():
    freq = np.array([1000.0, 10.0])
    weights = rare_weights([[0], [1], [0, 1]], freq)
    assert weights[1] > weights[0]
    assert abs(weights.mean() - 1) < 1e-6
    lengths = [1, 2, 1, 4]
    widths = [512, 512, 2048, 2048]
    batches = make_batches(lengths, widths, token_budget=3000, rng=np.random.default_rng(0))
    for width, indexes in batches:
        assert all(widths[i] == width for i in indexes)
        assert sum(lengths[i] * width for i in indexes) <= 3000 or len(indexes) == 1


def test_collate_maps_chunks_back_to_cases():
    cases = [
        {"id": "a", "labels": [1, 1, 4], "chunks": [[1, 2, 3], [4, 5]]},
        {"id": "b", "labels": [0], "chunks": [[7, 8]]},
    ]
    batch = collate(cases, width=8)
    assert batch["input_ids"].shape == (3, 8)
    assert batch["labels"].shape == (2, 100)
    assert batch["labels"][0, 1] == 1 and batch["labels"][0, 4] == 1
    assert batch["case_index"].tolist() == [0, 0, 1]
    assert int((batch["doc_ids"][0] >= 0).sum()) == 3
