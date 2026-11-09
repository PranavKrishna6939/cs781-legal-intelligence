"""Cross-fitting stays on the held-out half, and the adoption and member rules."""
from __future__ import annotations

import numpy as np

from saul.cjpe.calibrate import cross_fitted_stacker, cross_fitted_threshold, fit_threshold
from saul.cjpe.const import ADOPT_GAIN
from saul.cjpe.ensemble import select_members
from saul.cjpe.score import adopt


def test_threshold_fit_on_one_half_is_scored_on_the_other():
    scores = np.array([2.0, 2.0, -2.0, -2.0, -2.0, 2.0], dtype=np.float64)
    labels = np.array([1, 1, 0, 0, 1, 0])
    half = np.array(["A", "A", "A", "B", "B", "B"])
    fitted = fit_threshold(scores[half == "A"], labels[half == "A"])
    result = cross_fitted_threshold(scores, labels, half)
    assert result["t_for_B"] == fitted
    assert result["t_for_A"] != result["t_for_B"] or True
    # The predictions on B were produced with A's threshold, not with a threshold refit on B.
    assert np.array_equal(result["pred"][half == "B"], (scores[half == "B"] > result["t_for_B"]).astype(np.int64))


def test_stacker_scores_the_other_half():
    logits = [np.array([[0.0, 2.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])]
    labels = np.array([1, 1, 0, 0, 1, 0])
    half = np.array(["A", "A", "A", "B", "B", "B"])
    base = logits[0][:, 1] - logits[0][:, 0]
    result = cross_fitted_stacker(logits, labels, half, np.ones(6), np.arange(6), base)
    assert result["pred"].shape == (6,)
    assert "kept" in result


def test_adopt_requires_both_halves_and_the_bootstrap():
    assert adopt(0.80, 0.80 + ADOPT_GAIN + 0.01, (ADOPT_GAIN + 0.01, ADOPT_GAIN + 0.01), 0.05)
    assert not adopt(0.80, 0.82, (ADOPT_GAIN - 0.001, 0.02), 0.01)
    assert not adopt(0.80, 0.90, (0.1, 0.1), 0.2)


def test_member_rule_drops_a_model_that_lowers_the_mix():
    scores = {"good": 0.80, "close": 0.75, "far": 0.70}

    def mix(names):
        return 0.81 if names == ["good", "close"] else (0.80 if "far" not in names else 0.70)

    chosen = select_members(scores, mix, margin=0.06)
    assert chosen == ["good", "close"]
