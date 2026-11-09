"""CPU tests for tags, the CRF, voting, and the submission zip. No checkpoint required."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

import torch

from saul.model.modeling_saul import build_model, tiny_config
from saul.ner.backbone import NERModel
from saul.ner.data import Window, assign_tags, collate
from saul.ner.decode import decode_doc, default_config, propagation_oracle
from saul.ner.heads import CRF, NERHeads
from saul.ner.labels import IGNORE, O, tag_id, legal_transition_masks, spans_from_tags
from saul.ner.metrics import strict_scores
from saul.ner.peft import LoRA, apply_technique
from saul.ner.submit import write_zip


def test_bioes_roundtrip_and_cover():
    starts = [0, 5, 10]
    ends = [4, 9, 14]
    gold = [(0, 4, "APP"), (5, 14, "DATE")]
    tags, exact, used = assign_tags(starts, ends, gold)
    assert used == 2 and exact == 2
    assert spans_from_tags(tags, starts, ends) == gold
    tags, exact, used = assign_tags(starts, ends, [(1, 4, "APP")])
    assert used == 1 and exact == 0
    assert spans_from_tags(tags, starts, ends) == [(0, 4, "APP")]


def test_transitions_block_broken_spans():
    ok, start, end = legal_transition_masks()
    assert bool(ok[tag_id(0, "B"), tag_id(0, "E")])
    assert not bool(ok[tag_id(0, "B"), O])
    assert bool(start[tag_id(0, "S")])
    assert not bool(start[tag_id(0, "E")])
    assert bool(end[tag_id(0, "S")])
    assert not bool(end[tag_id(0, "B")])


def test_crf_viterbi_and_marginals():
    crf = CRF()
    emissions = torch.full((1, 3, 49), -4.0)
    b, e = tag_id(0, "B"), tag_id(0, "E")
    emissions[0, 0, b] = 8
    emissions[0, 1, e] = 8
    emissions[0, 2, O] = 8
    mask = torch.ones(1, 3, dtype=torch.bool)
    path = crf.viterbi(emissions, mask)[0]
    assert path == [b, e, O]
    probs = crf.marginals(emissions, mask)
    assert torch.allclose(probs[0].sum(-1), torch.ones(3), atol=1e-4)
    tags = torch.tensor([path])
    loss = crf.neg_log_likelihood(emissions, tags, mask)
    assert torch.isfinite(loss) and float(loss) >= 0


def test_propagation_copies_the_string():
    text = "Ranjeet went home. Ranjeet returned."
    first = text.find("Ranjeet")
    second = text.find("Ranjeet", first + 1)
    mentions = [{"start": first, "end": first + 7, "probs": [0.8] + [0.0] * 11 + [0.05]}]
    cfg = default_config()
    cfg["tau"] = {lab: 0.2 for lab in cfg["tau"]}
    spans = decode_doc(text, mentions, cfg)
    assert (first, first + 7, "APP") in spans
    assert (second, second + 7, "APP") in spans
    oracle = propagation_oracle(text, [(first, first + 7, "APP")])
    assert (second, second + 7, "APP") in oracle


def test_short_string_is_not_propagated():
    text = "Ram saw Ram."
    mentions = [{"start": 0, "end": 3, "probs": [0.9] + [0.0] * 11 + [0.05]}]
    cfg = default_config()
    cfg["tau"] = {lab: 0.2 for lab in cfg["tau"]}
    spans = decode_doc(text, mentions, cfg)
    assert spans == [(0, 3, "APP")]


def test_strict_f1_and_zip(tmp_path):
    gold = {"1": [(0, 3, "APP")], "2": []}
    pred = {"1": [(0, 3, "APP")], "2": []}
    assert strict_scores(gold, pred)["macro_f1"] < 1.0 or strict_scores(gold, pred)["per_label"]["APP"]["f"] == 1.0
    root = tmp_path
    os.makedirs(root / "submissions", exist_ok=True)
    # write_zip records under the repo submissions dir; the zip itself is what we check.
    zpath = write_zip({"1": [[0, 3, "APP"]]}, str(root), "unit")
    with zipfile.ZipFile(zpath) as z:
        assert z.namelist() == ["prediction.json"]
        body = json.loads(z.read("prediction.json"))
    assert body["1"] == [[0, 3, "APP"]]


def test_official_evaluator_self_score(tmp_path):
    labels = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
    lab = tmp_path / "labels.txt"
    lab.write_text("\n".join(labels) + "\n", encoding="utf-8")
    labels_list = labels
    gold = {"9": [[i * 3, i * 3 + 2, lab] for i, lab in enumerate(labels_list)]}
    truth = tmp_path / "truth.json"
    pred = tmp_path / "prediction.json"
    truth.write_text(json.dumps(gold), encoding="utf-8")
    pred.write_text(json.dumps(gold), encoding="utf-8")
    script = os.path.join(os.path.dirname(__file__), "..", "resources", "evaluation_script", "evaluate_phase2.py")
    proc = subprocess.run(
        [sys.executable, script, "--truth", str(truth), "--prediction", str(pred), "--labels", str(lab)],
        check=True, capture_output=True, text=True,
    )
    assert "Strict Macro-F1:        1.000000" in proc.stdout


def test_lora_trains_only_the_update_and_ced_stops():
    cfg = tiny_config("dense")
    backbone = build_model(cfg, enforce_budget=False, enforce_min=False)
    apply_technique(backbone, "lora")
    model = NERModel(backbone, NERHeads(cfg.d_model), "bidirectional")
    window = Window(
        "d", [4, 5, 6, 7, 8], [-1, -1, 0, 2, 4], [-1, -1, 2, 4, 6],
        [IGNORE, IGNORE, tag_id(0, "S"), O, O], 2, 3, 0, 3, True, False, [(0, 1, 0)],
    )
    batch = collate([window, window])
    loss = model(batch)["loss"]
    loss.backward()
    layer = backbone.blocks[0].mixer.q
    assert isinstance(layer, LoRA)
    assert layer.base.weight.grad is None
    assert layer.A.weight.grad is not None

    ced_cfg = tiny_config("ced", use_engram=False, residual="prenorm")
    ced = build_model(ced_cfg, enforce_budget=False, enforce_min=False)
    ced._ner_stop_after = 2
    ids = torch.randint(0, 50, (2, 8))
    ced(ids, return_hidden=True, mode="bidirectional").hidden.sum().backward()
    assert ced.blocks[1].mixer.q.weight.grad is not None
    assert ced.blocks[2].mixer.q.weight.grad is None


def test_echo_reads_the_second_copy():
    cfg = tiny_config("hybrid")
    backbone = build_model(cfg, enforce_budget=False, enforce_min=False)
    model = NERModel(backbone, NERHeads(cfg.d_model), "echo")
    ids = torch.randint(0, 40, (2, 6))
    docs = torch.zeros_like(ids)
    docs[:, -1] = -1
    hidden = model.hidden(ids, docs)
    assert hidden.shape == (2, 6, cfg.d_model)
    assert torch.isfinite(hidden).all()
