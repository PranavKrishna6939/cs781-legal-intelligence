"""prediction.json writer. Labels are never read."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile

from saul.lsi import p5_dir, raw_dir
from saul.lsi.const import N_LABELS

N_TEST = 13019


def test_ids(path: str | None = None) -> list[str]:
    path = path or os.path.join(raw_dir(), "Test", "test.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [str(row["id"]) for row in rows]


def test_ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def check_test_hash(stop: bool = True) -> str:
    digest = test_ids_digest(test_ids())
    path = os.path.join(p5_dir(), "test_ids_sha256.txt")
    if not os.path.exists(path):
        return digest
    saved = open(path, encoding="utf-8").read().strip()
    if saved != digest:
        message = f"test.json ids changed (saved {saved[:12]} now {digest[:12]})"
        if stop:
            raise SystemExit(message)
        print(message, flush=True)
    return digest


def check_prediction(rows: list[dict], expected: list[str], require_label: bool = False) -> None:
    got = [row.get("id") for row in rows]
    if got != expected and set(got) != set(expected):
        raise ValueError(f"id mismatch missing={len(set(expected)-set(got))} extra={len(set(got)-set(expected))}")
    if len(got) != len(set(got)):
        raise ValueError("duplicate ids")
    allowed = set(range(N_LABELS))
    by_id = {row["id"]: row for row in rows}
    for case_id in expected:
        pred = by_id[case_id].get("pred")
        if not isinstance(pred, list):
            raise ValueError(f"{case_id}: pred is not a list")
        if len(pred) != len(set(pred)):
            raise ValueError(f"{case_id}: duplicate statute ids")
        for lab in pred:
            if type(lab) is not int or lab not in allowed:
                raise ValueError(f"{case_id}: bad statute id {lab!r}")
        if require_label and not pred:
            raise ValueError(f"{case_id}: empty pred")


def rows_from_matrix(ids: list[str], pred) -> list[dict]:
    import numpy as np
    pred = np.asarray(pred)
    rows = []
    for index, case_id in enumerate(ids):
        labels = [int(i) for i in np.flatnonzero(pred[index] > 0)]
        rows.append({"id": case_id, "pred": labels})
    return rows


def write_zip(rows: list[dict], directory: str, expected: list[str] | None = None,
              require_label: bool = False) -> str:
    from_file = expected is None
    expected = expected if expected is not None else test_ids()
    if from_file and (len(expected) != N_TEST or len(set(expected)) != N_TEST):
        raise ValueError(f"expected {N_TEST} unique test ids, found {len(expected)}")
    check_prediction(rows, expected, require_label=require_label)
    by_id = {row["id"]: row for row in rows}
    ordered = [{"id": case_id, "pred": by_id[case_id]["pred"]} for case_id in expected]
    os.makedirs(directory, exist_ok=True)
    json_path = os.path.join(directory, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle)
    zpath = os.path.join(directory, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="prediction.json")
    print(f"[submit] {zpath}", flush=True)
    return zpath
