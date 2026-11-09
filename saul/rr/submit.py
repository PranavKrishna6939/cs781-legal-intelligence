"""Zip writer. prediction.json sits at the archive root. Nothing is appended to the score log."""
from __future__ import annotations

import json
import os
import zipfile

from saul.rr.data import LABELS


def check_prediction(pred: dict, test_docs) -> None:
    expected = {d.doc_id: d for d in test_docs}
    if set(pred) != set(expected):
        missing = set(expected) - set(pred)
        extra = set(pred) - set(expected)
        raise ValueError(f"ID mismatch missing={len(missing)} extra={len(extra)}")
    allowed = set(LABELS)
    for doc_id, labels in pred.items():
        doc = expected[doc_id]
        if not isinstance(labels, list) or len(labels) != len(doc.sentences):
            raise ValueError(f"{doc_id}: sentence-count mismatch")
        for lab in labels:
            if lab not in allowed:
                raise ValueError(f"{doc_id}: unknown label {lab}")


def write_zip(pred: dict, directory: str, test_docs) -> str:
    check_prediction(pred, test_docs)
    os.makedirs(directory, exist_ok=True)
    json_path = os.path.join(directory, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pred, f)
    zpath = os.path.join(directory, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="prediction.json")
    print(f"[submit] {zpath}", flush=True)
    return zpath
