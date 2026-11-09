"""prediction.json at the zip root, and the checks the evaluator will repeat."""
from __future__ import annotations

import json
import zipfile

import pytest

from saul.rr.data import RRDoc
from saul.rr.submit import check_prediction, write_zip


def _docs():
    return [
        RRDoc("a", "CL", ["one", "two"], None, "test"),
        RRDoc("b", "IT", ["three"], None, "test"),
    ]


def test_zip_contains_only_prediction_json(tmp_path):
    pred = {"a": ["Fact", "Issue"], "b": ["None"]}
    path = write_zip(pred, str(tmp_path), _docs())
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["prediction.json"]
        got = json.loads(z.read("prediction.json"))
    assert got == pred


def test_writer_rejects_bad_predictions():
    docs = _docs()
    with pytest.raises(ValueError):
        check_prediction({"a": ["Fact"]}, docs)
    with pytest.raises(ValueError):
        check_prediction({"a": ["Fact", "NotALabel"], "b": ["None"]}, docs)
    with pytest.raises(ValueError):
        check_prediction({"a": ["Fact", "Issue"]}, docs)
