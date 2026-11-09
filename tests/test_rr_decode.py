"""CRF marginals, frozen biases, nested collapse."""
from __future__ import annotations

import numpy as np
import torch

from saul.rr.data import LABEL_ID, RRDoc
from saul.rr.decode import should_collapse, transitions, tune_bias
from saul.rr.heads import ChainCRF


def test_marginals_sum_to_one_on_real_positions():
    crf = ChainCRF(13)
    emissions = torch.randn(2, 5, 13)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    marginals = crf.marginals(emissions, mask)
    assert torch.allclose(marginals[0, :4].sum(-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(marginals[1, :3].sum(-1), torch.ones(3), atol=1e-4)
    assert float(marginals[0, 4].abs().sum()) == 0.0


def test_dissent_bias_stays_zero():
    doc = RRDoc("d", "CL", ["a"] * 4, ["Fact"] * 4, "train")
    probs = {"d": np.full((4, 13), 1.0 / 13)}
    trans = transitions([doc])
    bias = tune_bias([doc], probs, trans, {"CL": 0.0, "IT": 0.0}, passes=1)
    assert bias[LABEL_ID["Dissent"]] == 0.0
    assert bias[LABEL_ID["PrecedentOverruled"]] == 0.0


def test_head_loss_is_finite():
    from saul.rr.data import collate, windows_for
    from saul.rr.heads import RRHeads

    doc = RRDoc("d", "IT", ["hello", "world"], ["Fact", "Issue"], "train")
    windows = windows_for(doc, encode=lambda text: [4, 5, 6], max_tokens=32)
    batch = collate(windows)
    head = RRHeads(32)
    head.set_weights(torch.ones(13))
    out = head(torch.randn(1, batch["input_ids"].shape[1], 32), batch)
    assert torch.isfinite(out["loss"])
    assert out["marginals"].shape == (1, 2, 13)
    assert torch.allclose(out["marginals"].sum(-1), torch.ones(1, 2), atol=1e-4)


def test_ia3_starts_as_identity():
    from saul.ner.peft import IA3Scale

    base = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    wrapped = IA3Scale(base, "output")
    x = torch.randn(2, 4)
    assert torch.allclose(wrapped(x), base(x))
    wrapped.merge_()
    assert torch.allclose(wrapped.scale, torch.ones(3))


def test_inprocess_score_matches_official_script():
    from saul.rr.score import macro_scores, official_macro

    docs = [
        RRDoc("c1", "CL", ["a", "b"], ["Fact", "Issue"], "dev"),
        RRDoc("i1", "IT", ["c"], ["None"], "dev"),
    ]
    pred = {"c1": ["Fact", "Fact"], "i1": ["None"]}
    gold = [0, 1, 12]
    got = [0, 0, 12]
    local = macro_scores(gold, got)["macro_f1"]
    assert abs(official_macro(docs, pred)["macro_f1"] - local) < 1e-6


def test_nested_collapse_on_a_large_gap():
    assert should_collapse(0.50, 0.80, 0.50)
    assert should_collapse(0.50, 0.52, 0.505)
    assert not should_collapse(0.50, 0.53, 0.52)
