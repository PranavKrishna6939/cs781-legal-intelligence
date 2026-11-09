"""Build Phase 4 zips from cached logits. Missing experiments are skipped."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saul.cjpe import p4_dir
from saul.cjpe.calibrate import apply_prior
from saul.cjpe.score import macro_scores, predict_logits
from saul.cjpe.submit import check_test_hash, predictions_from_logits, test_ids, write_zip
from saul.ner.registry import repo_root


SPECS = [
    ("P4_dense_lorma_tail2k", "exp/A1"),
    ("P4_dense_lorma_noleak", "exp/C1"),
    ("P4_dense_lorma_tapt", "exp/A4"),
    ("P4_dense_lorma_hier", "exp/A3"),
    ("P4_dense_mix", "mix/dense"),
    ("P4_hybrid_adapter_tail8k", "exp/A5"),
    ("P4_hybrid_adapter_noleak", "exp/C5"),
    ("P4_ced_partial_tail2k", "exp/A6"),
    ("P4_mix_all", "mix/all"),
    ("P4_conv_kd", "exp/A9"),
    ("P4_dense_full_tail2k", "exp/A7"),
    ("P4_mix_all_v2", "mix/all_v2"),
    ("P4_mix_all_refit", "mix/refit"),
    ("P4_mix_threshold", "mix/all"),
    ("P4_mix_stack", "mix/all_stack"),
]


def _load(directory: str, kind: str):
    path = os.path.join(directory, f"logits_{kind}.npz")
    if not os.path.exists(path):
        return None
    blob = np.load(path, allow_pickle=True)
    ids = [str(i) for i in blob["ids"].tolist()]
    logits = np.asarray(blob["logits"], dtype=np.float64)
    labels = np.asarray(blob["labels"]) if "labels" in blob.files else None
    return ids, logits, labels


def _ready(directory: str, name: str) -> tuple[np.ndarray, float] | None:
    test = _load(directory, "test")
    if test is None:
        print(f"[zips] skip {name}", flush=True)
        return None
    ids, logits, _labels = test
    calib_path = os.path.join(directory, "calib.json")
    done_path = os.path.join(directory, "done.json")
    threshold = 0.0
    if os.path.exists(calib_path):
        calib = json.load(open(calib_path, encoding="utf-8"))
        if name == "P4_mix_threshold" and not calib.get("threshold_kept"):
            print(f"[zips] skip {name}; threshold was not kept", flush=True)
            return None
        if name == "P4_mix_threshold":
            threshold = float(calib.get("threshold") or 0.0)
    elif os.path.exists(done_path):
        done = json.load(open(done_path, encoding="utf-8"))
        logits = apply_prior(logits, done.get("loss") == "ce")
    dsel = _load(directory, "dsel")
    dsel_f1 = None
    if dsel is not None and dsel[2] is not None:
        dsel_logits = dsel[1]
        if not os.path.exists(calib_path) and os.path.exists(done_path):
            done = json.load(open(done_path, encoding="utf-8"))
            dsel_logits = apply_prior(dsel_logits, done.get("loss") == "ce")
        dsel_f1 = macro_scores(dsel[2], predict_logits(dsel_logits, threshold))["f1"]
    order = {case_id: i for i, case_id in enumerate(ids)}
    wanted = test_ids()
    aligned = np.stack([logits[order[case_id]] for case_id in wanted])
    return aligned, threshold, dsel_f1, wanted


def main() -> None:
    check_test_hash(stop=True)
    root = os.path.join(repo_root(), "submissions", "phase4")
    for name, rel in SPECS:
        directory = os.path.join(p4_dir(), rel)
        got = _ready(directory, name)
        if got is None:
            continue
        aligned, threshold, dsel_f1, wanted = got
        rows = predictions_from_logits(wanted, aligned, threshold)
        dest = os.path.join(root, name)
        write_zip(rows, dest, wanted)
        print(f"[zips] {name} dsel={dsel_f1}", flush=True)


if __name__ == "__main__":
    main()
