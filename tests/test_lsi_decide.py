import numpy as np

from saul.lsi.decide import Stacker, at_least_one, choose_temperature, fit_temperature, shrink_thresholds, stacker_oof
from saul.lsi.knn import vote


def test_stacker_oof_never_fits_the_scored_rows():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(20, 4))
    labels = (rng.random((20, 4)) > 0.7).astype(np.float32)
    seen = []
    original = Stacker.fit

    def spy(self, x, y):
        seen.append(len(x))
        return original(self, x, y)

    Stacker.fit = spy
    try:
        oof = stacker_oof(logits, labels, half=np.array(["A"] * 20), n_folds=5, seed=0)
    finally:
        Stacker.fit = original
    assert oof.shape == logits.shape
    assert seen and max(seen) <= 16


def test_shrinkage_moves_toward_the_global_threshold():
    raw = np.array([0.1, 0.9])
    out = shrink_thresholds(raw, global_t=0.5, counts=np.array([1.0, 1000.0]), kappa=50)
    assert abs(out[0] - 0.5) < abs(raw[0] - 0.5)
    assert abs(out[1] - raw[1]) < 0.05
    assert np.allclose(shrink_thresholds(raw, 0.4, np.array([10.0, 10.0]), None), 0.4)


def test_at_least_one_fills_an_empty_row():
    probs = np.array([[0.2, 0.1], [0.9, 0.1]])
    thr = np.array([0.5, 0.5])
    pred = at_least_one(probs, thr)
    assert pred[0].sum() == 1 and pred[0, 0]
    assert pred[1, 0] and not pred[1, 1]


def test_stacker_diagonal_stays_zero():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(30, 5))
    labels = (rng.random((30, 5)) > 0.6).astype(np.float32)
    model = Stacker(n_labels=5, steps=5).fit(logits, labels)
    assert np.allclose(np.diag(model.W), 0)


def test_temperature_is_fit_on_the_other_half():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(12, 3))
    labels = (rng.random((12, 3)) > 0.5).astype(np.float32)
    half = np.array(["A"] * 6 + ["B"] * 6)
    seen = []
    original = fit_temperature

    def spy(x, y):
        seen.append(len(x))
        return original(x, y)

    import saul.lsi.decide as decide
    decide.fit_temperature = spy
    try:
        info = choose_temperature(logits, labels, half)
    finally:
        decide.fit_temperature = original
    assert seen == [6, 6]
    assert 0.5 <= info["T"] <= 5.0


def test_knn_index_is_the_train_matrix_only():
    train = np.eye(4, dtype=np.float32)
    labels = np.eye(4, dtype=np.float32)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    probs = vote(query, train, labels, k=1)
    assert probs.shape == (1, 4)
    assert probs[0, 0] > 0.99
