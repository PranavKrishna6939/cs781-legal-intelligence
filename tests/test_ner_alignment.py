"""Gold-span checks against the real Phase-2 files. Skips when the data is not on this machine."""
from __future__ import annotations

import os

import pytest

from saul.ner.data import load_split, load_tokenizer
from saul.ner.decode import propagation_oracle
from saul.ner.labels import spans_from_tags
from saul.ner.metrics import strict_scores


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.environ.get("SAUL_LNER_DIR", os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "Phase_2_LNER", "Train", "fold1-text.json"))),
    reason="Phase-2 folds are not on this machine",
)


def test_gold_tags_roundtrip_near_tokenizer_ceiling():
    tok = load_tokenizer()
    docs = load_split("train", tok)
    exact = used = 0
    extra = 0
    for doc in docs:
        pred = spans_from_tags(doc.tags, doc.starts, doc.ends)
        gold = {(a, b, c) for a, b, c in doc.gold}
        got = {(a, b, c) for a, b, c in pred}
        exact += len(gold & got)
        used += len(gold)
        # Every predicted span overlaps some gold span of the same label. No invented entities.
        for a, b, lab in pred:
            if (a, b, lab) in gold:
                continue
            if not any(lab == g[2] and not (b <= g[0] or a >= g[1]) for g in doc.gold):
                extra += 1
    rate = exact / used
    assert used > 8000
    assert 0.990 <= rate <= 0.996
    assert extra == 0


def test_propagation_oracle_matches_the_measured_ceiling():
    docs = load_split("train")
    gold = {d.doc_id: d.gold for d in docs}
    pred = {d.doc_id: propagation_oracle(d.text, d.gold) for d in docs}
    score = strict_scores(gold, pred)["macro_f1"]
    assert abs(score - 0.991) < 0.01


def test_unicode_offsets_survive_json(tmp_path):
    docs = load_split("train")
    doc = next((d for d in docs if any(ord(ch) > 127 for ch in d.text)), None)
    if doc is None:
        pytest.skip("no non-ascii training document")
    blob = [(a, b, c) for a, b, c in doc.gold]
    import json
    path = tmp_path / "p.json"
    path.write_text(json.dumps({doc.doc_id: blob}), encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    for a, b, lab in back[doc.doc_id]:
        assert doc.text[a:b]
        assert isinstance(a, int) and b > a
