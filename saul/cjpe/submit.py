"""prediction.json at the zip root. Ids come from the current test file. Labels are never read."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile

from saul.cjpe import p4_dir, raw_dir


def test_ids(path: str | None = None) -> list[str]:
    path = path or os.path.join(raw_dir(), "Test", "test.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [row["id"] for row in rows]


def test_ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_test_hash(stop: bool = True) -> str:
    """Compare the current test file with the hash written at data-prep time."""
    ids = test_ids()
    digest = test_ids_digest(ids)
    cached = os.path.join(p4_dir(), "test_ids_sha256.txt")
    if os.path.exists(cached):
        previous = open(cached, encoding="utf-8").read().strip()
        if previous and previous != digest:
            message = f"test.json ids changed ({previous[:12]} -> {digest[:12]})"
            print(f"[submit] {message}", flush=True)
            if stop:
                raise SystemExit(message)
    return digest


def check_prediction(rows: list[dict], expected: list[str]) -> None:
    if not isinstance(rows, list):
        raise ValueError("prediction.json must be a list")
    ids = []
    for row in rows:
        if set(row) - {"id", "pred"}:
            raise ValueError(f"unexpected keys in {row.keys()}")
        if "label" in row:
            raise ValueError("prediction rows must not contain labels")
        pred = row["pred"]
        if type(pred) is not int or pred not in (0, 1):
            raise ValueError(f"pred must be the integer 0 or 1, got {pred!r}")
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids")
    if set(ids) != set(expected):
        raise ValueError(f"id mismatch missing={len(set(expected) - set(ids))} extra={len(set(ids) - set(expected))}")


def predictions_from_logits(ids: list[str], logits, threshold: float = 0.0) -> list[dict]:
    import numpy as np
    logits = np.asarray(logits)
    index = {case_id: i for i, case_id in enumerate(ids)}
    rows = []
    for case_id in ids:
        lo = float(logits[index[case_id], 1] - logits[index[case_id], 0])
        rows.append({"id": case_id, "pred": int(lo > threshold)})
    return rows


def write_zip(rows: list[dict], directory: str, expected: list[str] | None = None) -> str:
    expected = test_ids() if expected is None else expected
    check_prediction(rows, expected)
    os.makedirs(directory, exist_ok=True)
    json_path = os.path.join(directory, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    zpath = os.path.join(directory, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="prediction.json")
    print(f"[submit] {zpath}", flush=True)
    return zpath
