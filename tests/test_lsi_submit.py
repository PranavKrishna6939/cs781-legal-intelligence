import json
import os
import zipfile

import pytest

from saul.lsi.submit import check_prediction, write_zip


def test_zip_has_prediction_at_the_root(tmp_path):
    ids = ["a", "b"]
    rows = [{"id": "b", "pred": [1, 2]}, {"id": "a", "pred": [0]}]
    path = write_zip(rows, str(tmp_path), expected=ids)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["prediction.json"]
        payload = json.loads(archive.read("prediction.json"))
    assert [row["id"] for row in payload] == ids
    assert payload[0]["pred"] == [0]


def test_validator_rejects_bad_predictions():
    ids = ["a"]
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": [1, 1]}], ids)
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": ["1"]}], ids)
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": [100]}], ids)
    with pytest.raises(ValueError):
        check_prediction([{"id": "a", "pred": []}], ids, require_label=True)
    with pytest.raises(ValueError):
        check_prediction([{"id": "missing", "pred": [1]}], ids)


def test_validator_does_not_read_a_label_field():
    rows = [{"id": "a", "pred": [3], "labels": [0]}]
    check_prediction(rows, ["a"])
    assert "labels" not in json.dumps({k: rows[0][k] for k in ("id", "pred")})
