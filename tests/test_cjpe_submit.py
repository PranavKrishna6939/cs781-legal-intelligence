"""Zip root, integer predictions, and no label field."""
from __future__ import annotations

import json
import zipfile

import pytest

from saul.cjpe.submit import check_prediction, predictions_from_logits, write_zip


def test_zip_lists_every_id_once_with_integer_preds(tmp_path):
    rows = [{"id": "b", "pred": 1}, {"id": "a", "pred": 0}]
    path = write_zip(rows, str(tmp_path), ["a", "b"])
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["prediction.json"]
        got = json.loads(archive.read("prediction.json"))
    assert got == rows
    assert all(type(row["pred"]) is int for row in got)


def test_writer_rejects_labels_bools_and_mismatched_ids():
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": 1, "label": 1}], ["a"])
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": True}], ["a"])
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": 0}, {"id": "a", "pred": 1}], ["a"])
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": 0}], ["a", "b"])


def test_predictions_come_from_logits_not_labels():
    rows = predictions_from_logits(["a", "b"], [[0.2, 0.1], [0.0, 3.0]])
    assert rows == [{"id": "a", "pred": 0}, {"id": "b", "pred": 1}]
    assert "label" not in rows[0]
