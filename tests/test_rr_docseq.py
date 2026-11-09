"""Stage-2 document sequence features, shape, and fold-safe augmentation."""
from __future__ import annotations

import numpy as np
import torch

from saul.rr.data import LABEL_ID, N_LABELS
from saul.rr.docseq import DocSeq, augment_fold, features, predict


def _probs(n: int, members: int, fill: float) -> list[np.ndarray]:
    return [np.full((n, N_LABELS), fill, dtype=np.float32) for _ in range(members)]


def test_feature_width_matches_members_position_and_domain():
    members = 3
    x = features(_probs(7, members, 1.0 / N_LABELS), "IT")
    assert x.shape == (7, 2 * N_LABELS * members + 9 + 2)
    assert x.dtype == np.float32


def test_docseq_predicts_label_distribution_per_sentence():
    n = 5
    model = DocSeq(2 * N_LABELS * 2 + 9 + 2)
    model.eval()
    out = predict(model, np.zeros((n, model.inp[1].in_features), dtype=np.float32))
    assert out.shape == (n, N_LABELS)
    assert torch.allclose(torch.tensor(out.sum(axis=1)), torch.ones(n), atol=1e-5)


def test_augmentation_draws_only_from_the_outer_training_ids():
    train = ["CCI_a", "COM_b", "HC_c"]
    held = "SC_held"
    ids = train + [held]
    probs, labels, domains = {}, {}, {}
    for j, doc_id in enumerate(ids):
        row = np.zeros((20, N_LABELS), dtype=np.float32)
        row[:, 0] = j + 1
        probs[doc_id] = [row]
        labels[doc_id] = np.zeros(20, dtype=np.int64)
        domains[doc_id] = "CL"
    seqs = augment_fold(probs, labels, domains, train, n=12, rng=np.random.default_rng(0))
    assert len(seqs) == 12
    train_marks = {1.0, 2.0, 3.0}
    for x, y, src in seqs:
        assert src in train and src != held
        assert src.split("_", 1)[0] in ("CCI", "COM")
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] > 20
        assert set(np.unique(x[:, 0])).issubset(train_marks)
        assert float(x[0, 0]) in (1.0, 2.0)
        assert np.all(y[20:] == LABEL_ID["Dissent"])
        assert np.all(y[:20] == 0)


def test_long_tail_stays_inside_the_outer_training_ids():
    train = ["CCI_a", "COM_long"]
    probs, labels, domains = {}, {}, {}
    for doc_id, n, mark in (("CCI_a", 20, 1.0), ("COM_long", 360, 2.0), ("SC_held", 300, 9.0)):
        row = np.zeros((n, N_LABELS), dtype=np.float32)
        row[:, 0] = mark
        probs[doc_id] = [row]
        labels[doc_id] = np.zeros(n, dtype=np.int64)
        domains[doc_id] = "CL"
    seqs = augment_fold(
        probs, labels, domains, train, n=16, rng=np.random.default_rng(1), long_tail=True,
    )
    assert seqs
    assert all(src in train for _, _, src in seqs)
    assert any(len(y) - len(labels[src]) >= 200 for _, y, src in seqs)
    assert all(9.0 not in set(np.unique(x[:, 0])) for x, _, _ in seqs)
