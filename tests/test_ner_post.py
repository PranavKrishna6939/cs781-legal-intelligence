"""Parser, boundary rules, role fallback, and the bias cache. No checkpoint and no fold-4 text."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from saul.model.modeling_saul import build_model, tiny_config
from saul.ner.backbone import NERModel
from saul.ner.data import Doc, Window, collate
from saul.ner.header import Party, counsel_cues, parse_header, witness_cues
from saul.ner.heads import NERHeads
from saul.ner.kd import _silver_tags, build_pool
from saul.ner.labels import IGNORE, LABELS, N_TAGS, O, tag_id
from saul.ner.post import _share_labels, default_post_config, post_process, tune_stack
from saul.ner.roles import N_FEATURES, ROLE_LABELS, Entity, apply_decisions, apply_rule_fallback, featurize
from saul.ner.rules import apply_rules
from saul.ner.tokdec import Decoder
from saul.ner.train import forwards_per_step


def _doc(text: str) -> Doc:
    ids, starts, ends = [], [], []
    for m in re.finditer(r"\S+", text):
        ids.append(1)
        starts.append(m.start())
        ends.append(m.end())
    return Doc("d", text, "t", ids, starts, ends, [O] * len(ids), [], {})


def test_header_patterns_from_the_plan():
    sci = (
        "PETITIONER: PARMAR KANAKSINH BHAGWANSINH (DEAD) BY L.R'S. Vs. "
        "RESPONDENT: 1. MAKWANA SHANABHAI BHIKHABHAI & 2. MAKWANA PRABATBHAI "
        "DATE OF JUDGMENT08/12/1994 BENCH: VENKATACHALA N. (J) BENCH: SAHAI, R.M. (J) CITATION: x"
    )
    header = parse_header(sci)
    names = {p.name for p in header.parties if p.kind == "party"}
    assert "PARMAR KANAKSINH BHAGWANSINH" in names
    assert "MAKWANA SHANABHAI BHIKHABHAI" in names
    assert "MAKWANA PRABATBHAI" in names
    assert any(p.kind == "judge" and "SAHAI" in p.name for p in header.parties)

    role = (
        "IN THE SUPREME COURT OF INDIA CRIMINAL APPELLATE JURISDICTION "
        "CRIMINAL APPEAL NO. 92/2015 JAGE RAM & ORS. ..Appellants Versus "
        "STATE OF HARYANA ..Respondent J U D G M E N T"
    )
    header = parse_header(role)
    sides = {(p.side, p.name) for p in header.parties if p.kind == "party"}
    assert ("A", "JAGE RAM") in sides
    assert ("R", "STATE OF HARYANA") in sides

    colon = (
        "Petitioner :- Sardar Gurmeet Singh And Another Respondent :- Smt. Raj Katyal "
        "Counsel for Petitioner :- Mohd. Aqueel Khan, Chandra Bhan Gupta "
        "Counsel for Respondent :- C.M.Rai Hon'ble J.J. Munir,J."
    )
    header = parse_header(colon)
    counsel = {p.name for p in header.parties if p.kind == "counsel"}
    assert "Mohd. Aqueel Khan" in counsel and "C.M.Rai" in counsel
    assert not any("Counsel for" in p.name for p in header.parties if p.kind == "party")

    ms = (
        "THE MANAGING DIRECTOR (SHRI GRISH BATRA) M/S.PADMINI INFRASTRUCTURE DEVELOPERS (I) LTD. "
        "… APPELLANT(S) VERSUS THE GENERAL SECRETARY (SHRI AMOL MAHAPATRA) "
        "ROYAL GARDEN RESDIENTS WELFARE ASSOCIATION … RESPONDENT(S) JUDGMENT"
    )
    header = parse_header(ms)
    assert any(p.name == "GRISH BATRA" and p.side == "A" for p in header.parties)
    assert any("PADMINI" in p.name for p in header.parties)

    text = "the complainant Jagdish (PW-5) and Dr. Pardeep Kumar (PW-7). Heard Mr. Shadan Farasat, learned counsel for the appellants."
    assert {c.name for c in witness_cues(text)} >= {"Jagdish", "Pardeep Kumar"}
    assert any(c.name == "Shadan Farasat" and c.side == "A" for c in counsel_cues(text))


def test_rule_fallback_relabels_a_body_repeat():
    text = (
        "IN THE SUPREME COURT OF INDIA CRIMINAL APPELLATE JURISDICTION "
        "CRIMINAL APPEAL NO. 92/2015 JAGE RAM & ORS. ..Appellants Versus "
        "STATE OF HARYANA ..Respondent\n1. Later Jage Ram spoke."
    )
    doc = _doc(text)
    at = text.rfind("Jage Ram")
    raw = [(at, at + len("Jage Ram"), "RESP", 0.9)]
    cfg = default_post_config()
    cfg["p2_rules"] = True
    cfg["reconcile"] = False
    cfg["propagate"] = {lab: False for lab in cfg["propagate"]}
    out = post_process(doc, raw, np.zeros((len(doc.ids), N_TAGS), np.float32), cfg)
    assert (at, at + len("Jage Ram"), "APP") in out


def test_p3_rules_one_each():
    text = "Criminal Case No. 3483 of 2017 ended."
    doc = _doc(text)
    start = text.find("No.")
    end = text.find("3483") + 4
    spans = apply_rules(doc, [(start, end, "CASENO")], {"p3_caseno": True})
    assert doc.text[spans[0][0]:spans[0][1]].startswith("Criminal")
    assert spans[0][1] > end

    text = "A v. B; C v. D stayed."
    doc = _doc(text)
    spans = apply_rules(doc, [(0, text.find("stayed") - 1, "PREC")], {"p3_prec_split": True})
    assert len([s for s in spans if s[2] == "PREC"]) == 2

    text = "See Khajani Devi's case here."
    doc = _doc(text)
    full = "Khajani Devi v. State"
    # The short-form rule reads the accepted PREC span, which is not in this sentence.
    # Put the full citation in the same document.
    text = "Khajani Devi v. State was followed. See Khajani Devi's case here."
    doc = _doc(text)
    src = (0, len(full), "PREC")
    spans = apply_rules(doc, [src], {"p3_prec_short": True})
    assert any(doc.text[s:e] == "Khajani Devi" and s > 0 for s, e, lab in spans if lab == "PREC")

    text = "Madras High Court and Gujarat High Court sat."
    doc = _doc(text)
    spans = apply_rules(doc, [(0, text.find(" sat"), "COURT")], {"p3_court_split": True})
    assert len(spans) == 2

    text = ".................... was the line."
    doc = _doc(text)
    spans = apply_rules(doc, [(0, 20, "JUDGE")], {"p3_junk": True})
    assert spans == []

    text = "Government of Assam notified it."
    doc = _doc(text)
    spans = apply_rules(doc, [(0, text.find(" notified"), "AUTH")], {"p3_auth_gov": True}, lexicon={"Government": "AUTH"})
    assert any(doc.text[s:e] == "Government" for s, e, lab in spans)


def test_bias_is_part_of_the_viterbi_cache_key():
    doc = _doc("Ann left.")
    logp = np.zeros((len(doc.ids), N_TAGS), np.float32)
    dec = Decoder([doc], {"d": logp})
    dec.spans(1.0e4, 0.0, {"APP": 0.0})
    dec.spans(1.0e4, 0.0, {"APP": 1.0})
    assert len(dec._cache) == 2


def test_feature_vector_has_60_entries():
    doc = _doc("Ann left the court.")
    ent = Entity("ann")
    ent.surfaces.add("Ann")
    ent.mentions.append((0, 3, "Ann"))
    flags = {k: False for k in ("a_party", "r_party", "a_counsel", "r_counsel", "judge", "witness", "has_a", "has_r")}
    flags.update({"cue_a": 0, "cue_r": 0, "header_end": 10})
    vec = featurize(doc, ent, [], None, flags, None)
    assert vec.shape == (N_FEATURES,)
    assert N_FEATURES == 60


def test_rule_fallback_skips_judges_and_uncovered_header_mentions():
    text = "SAHAI sat near Ram Lal versus. Later Ram Lal and SAHAI spoke."
    doc = _doc(text)
    header_end = text.index("Later")
    parties = [
        Party("SAHAI", "", "judge", text.find("SAHAI"), text.find("SAHAI") + 5, "F-SCI"),
        Party("Ram Lal", "A", "party", text.find("Ram Lal"), text.find("Ram Lal") + 7, "F-SCI"),
    ]
    cfg = {"p2_place_header": {lab: False for lab in ROLE_LABELS}}
    out = apply_rule_fallback(doc, [], parties, [], header_end, cfg)
    assert not any(lab == "JUDGE" or doc.text[s:e] == "SAHAI" for s, e, lab in out)
    body = text.rfind("Ram Lal")
    head = text.find("Ram Lal")
    assert (body, body + len("Ram Lal"), "APP") in out
    assert (head, head + len("Ram Lal"), "APP") not in out
    relabeled = apply_rule_fallback(
        doc, [(head, head + len("Ram Lal"), "RESP")], parties, [], header_end, cfg,
    )
    assert (head, head + len("Ram Lal"), "APP") in relabeled


def test_decisions_place_header_only_when_switched_or_already_predicted():
    text = "RAM LAL versus the bench. Later Ram Lal spoke."
    doc = _doc(text)
    header_end = text.index("Later")
    head = text.find("RAM LAL")
    body = text.find("Ram Lal")
    ent = Entity("ram lal")
    ent.surfaces.update({"RAM LAL", "Ram Lal"})
    ent.mentions = [(head, head + 7, "RAM LAL"), (body, body + 7, "Ram Lal")]
    classes = ["APP", "NONE"]
    proba = np.array([[0.9, 0.1]])
    cfg = {
        "theta_place": {"APP": 0.5},
        "theta_drop": 1.01,
        "p2_place_header": {"APP": False},
    }
    off = apply_decisions(doc, [], [ent], classes, proba, cfg, header_end)
    assert (body, body + 7, "APP") in off
    assert (head, head + 7, "APP") not in off
    cfg["p2_place_header"] = {"APP": True}
    on = apply_decisions(doc, [], [ent], classes, proba, cfg, header_end)
    assert (head, head + 7, "APP") in on
    cfg["p2_place_header"] = {"APP": False}
    predicted = apply_decisions(
        doc, [(head, head + 7, "RESP")], [ent], classes, proba, cfg, header_end,
    )
    assert (head, head + 7, "APP") in predicted


def test_shared_tune_collapses_per_label_switches():
    class _Dec:
        def tune(self, cfg=None, verbose=False):
            cfg = json.loads(json.dumps(cfg))
            cfg["propagate"] = {lab: i % 2 == 0 for i, lab in enumerate(LABELS)}
            cfg["lexicon"] = {lab: i % 2 == 1 for i, lab in enumerate(LABELS)}
            cfg["theta_place"] = {lab: 0.3 if i % 2 == 0 else 0.8 for i, lab in enumerate(ROLE_LABELS)}
            cfg["oof_f1"] = 0.9
            return cfg

        def score(self, cfg):
            mixed = len(set(cfg["propagate"].values())) > 1 or len(set(cfg["lexicon"].values())) > 1
            return {"macro_f1": 0.9 if mixed else 0.4}

    class _Stack:
        dec = _Dec()

        def score(self, cfg, bundle=None):
            return {"macro_f1": 0.4}

        def bind_oof(self, cfg, kind):
            return None

    out = tune_stack(_Stack(), shared=True)
    assert len(set(out["propagate"].values())) == 1
    assert len(set(out["lexicon"].values())) == 1
    assert len(set(out["theta_place"].values())) == 1
    # The helper itself must not keep a better per-label mix.
    mixed = default_post_config()
    mixed["propagate"] = {lab: i % 2 == 0 for i, lab in enumerate(LABELS)}
    shared = _share_labels(_Stack(), mixed)
    assert len(set(shared["propagate"].values())) == 1


def test_silver_low_margin_outside_tokens_are_ignore():
    doc = _doc("Ann left the court.")
    low = np.full((len(doc.ids), N_TAGS), 0.01, np.float32)
    low[:, O] = 0.4
    tags = _silver_tags(doc, [], low)
    assert tags == [IGNORE] * len(doc.ids)
    sure = np.full((len(doc.ids), N_TAGS), 0.01, np.float32)
    sure[:, O] = 0.95
    assert _silver_tags(doc, [], sure) == [O] * len(doc.ids)
    span = (0, doc.ends[0], "APP")
    tagged = _silver_tags(doc, [span], low)
    assert tagged[0] != IGNORE
    assert all(t == IGNORE for t in tagged[1:])


def test_ban_test_pool_is_separate_and_drops_fold3_grams(tmp_path, monkeypatch):
    words = " ".join(f"w{i}" for i in range(20))
    long_text = (words + " ") * 120
    seen = []

    class _Doc:
        text = long_text

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return long_text

    class _Store:
        n_docs = 1

        def doc(self, i):
            return np.array([10, 11, 12], dtype=np.int64)

    def _split(name, tok=None):
        seen.append(name)
        return [_Doc()] if name == "test" else []

    # kd.open_source is bound at import, so patch that name. The data-module
    # patch is not what build_pool calls.
    monkeypatch.setattr("saul.ner.kd.load_tokenizer", lambda: _Tok())
    monkeypatch.setattr("saul.ner.kd.open_source", lambda *a, **k: _Store())
    monkeypatch.setattr("saul.ner.data.load_split", _split)
    old = tmp_path / "pool.jsonl"
    old.write_text('{"id": "keep-me"}\n', encoding="utf-8")
    dest = build_pool(str(tmp_path), n_docs=None, min_tokens=1, ban_test=True)
    assert os.path.basename(dest) == "pool_v2.jsonl"
    assert dest != str(old)
    assert old.read_text(encoding="utf-8").startswith("{")
    assert os.path.getsize(dest) == 0
    assert "test" in seen
    meta = json.loads(open(dest + ".meta.json", encoding="utf-8").read())
    assert meta["ban_test"] is True and meta["n_docs"] == 1000

    kept = tmp_path / "kept"
    kept.mkdir()
    monkeypatch.setattr("saul.ner.data.load_split", lambda name, tok=None: [])
    plain = build_pool(str(kept), n_docs=1, min_tokens=1, ban_test=False)
    assert os.path.basename(plain) == "pool.jsonl"
    assert sum(1 for _ in open(plain, encoding="utf-8")) == 1


def test_default_train_step_is_one_forward():
    assert forwards_per_step(0.0, 0.0) == 1
    assert forwards_per_step(1.0, 0.0) == 2
    assert forwards_per_step(0.0, 1.0) == 2
    cfg = tiny_config("dense")
    backbone = build_model(cfg, enforce_budget=False, enforce_min=False)
    model = NERModel(backbone, NERHeads(cfg.d_model), "bidirectional")
    window = Window(
        "d", [4, 5, 6, 7, 8], [-1, -1, 0, 2, 4], [-1, -1, 2, 4, 6],
        [IGNORE, IGNORE, tag_id(0, "S"), O, O], 2, 3, 0, 3, True, False, [(0, 1, 0)],
    )
    batch = collate([window])
    calls = {"n": 0}
    orig = model.forward

    def wrapped(b):
        calls["n"] += 1
        return orig(b)

    model.forward = wrapped
    out = model(batch)
    assert calls["n"] == forwards_per_step(0.0, 0.0)
    assert "nll" in out and "ce" in out and "span" in out
    assert torch.isfinite(out["loss"])
