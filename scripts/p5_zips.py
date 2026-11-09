"""Build Phase 5 submission zips from cached predictions. Skips a member that has no logits."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import KAPPAS
from saul.lsi.data import dev_half_array
from saul.lsi.decide import (
    choose_kappa, final_thresholds, load_logits, predict_matrix, sigmoid,
)
from saul.lsi.ensemble import assemble_cf2, build_mix
from saul.lsi.score import macro_f1
from saul.lsi.submit import rows_from_matrix, test_ids, write_zip
from saul.ner.registry import repo_root

# None is a real kappa (the global threshold). This marks "use the kappa the dev search picked".
_CHOSEN = object()


def _dev_score(exp: str) -> str:
    path = os.path.join(exp_dir(exp), "done.json")
    if not os.path.exists(path):
        return "na"
    blob = json.load(open(path, encoding="utf-8"))
    value = blob.get("dev_f1")
    return "na" if value is None else f"{float(value):.4f}"


def _zip_matrix(name: str, ids: list[str], pred, require_label: bool = False) -> None:
    directory = os.path.join(repo_root(), "submissions", "phase5", name)
    rows = rows_from_matrix(ids, pred)
    write_zip(rows, directory, expected=ids, require_label=require_label)
    print(f"[zips] {name} dev_cf={_dev_score(name)}", flush=True)


def _member_pred(exp: str, kappa, d2: str) -> tuple[list[str], np.ndarray] | None:
    test_path = os.path.join(exp_dir(exp), "test_logits.npz")
    dev_path = os.path.join(exp_dir(exp), "dev_logits.npz")
    if not (os.path.exists(test_path) and os.path.exists(dev_path)):
        print(f"[zips] skip {exp}, logits missing", flush=True)
        return None
    dev = load_logits(exp, "dev")
    test = load_logits(exp, "test")
    if "labels" not in dev:
        print(f"[zips] skip {exp}, no dev labels", flush=True)
        return None
    half = dev_half_array(dev["ids"], p5_dir())
    info = choose_kappa(sigmoid(dev["logits"]), dev["labels"], half)
    use = info["kappa"] if kappa is _CHOSEN else kappa
    thresholds = final_thresholds(sigmoid(dev["logits"]), dev["labels"], use)
    pred = predict_matrix(sigmoid(test["logits"]), thresholds, d2, None, "off")
    score = macro_f1(dev["labels"], predict_matrix(sigmoid(dev["logits"]), thresholds, d2, None, "off"))
    print(f"[zips] {exp} kappa={use} d2={d2} dev_macro={score:.4f}", flush=True)
    return test["ids"], pred


def _write_aligned(zip_name: str, ids: list[str], order: list[str], pred, require_label: bool) -> None:
    index = {case_id: i for i, case_id in enumerate(order)}
    if set(index) != set(ids):
        print(f"[zips] {zip_name} id set mismatch", flush=True)
        return
    write_zip(
        rows_from_matrix(ids, pred[[index[i] for i in ids]]),
        os.path.join(repo_root(), "submissions", "phase5", zip_name),
        expected=ids, require_label=require_label,
    )
    print(f"[zips] {zip_name}", flush=True)


def _fit_a4_stack() -> None:
    """Fit D3/D4 on the CF2 member. The zip is written only when one of them is kept."""
    from saul.lsi.data import load_raw, lookup_duplicates
    from saul.lsi.decide import _knn_matrix, run_decision

    dev = load_logits("A4", "dev")
    test = load_logits("A4", "test")
    if "labels" not in dev:
        print("[zips] A4 has no dev labels; stack zip skipped", flush=True)
        return
    raw = load_raw()
    dev_rows = {row["id"]: row for row in raw["dev"]}
    test_rows = {row["id"]: row for row in raw["test"]}
    knn_exp = "A1t" if os.path.exists(os.path.join(exp_dir("A1t"), "dev_knn.npz")) else "A1"
    have_knn = os.path.exists(os.path.join(exp_dir(knn_exp), "dev_knn.npz"))
    run_decision(
        dev["logits"], dev["labels"], dev["ids"], test["logits"], test["ids"],
        knn_dev=_knn_matrix(knn_exp, "dev", dev["ids"]) if have_knn else None,
        knn_test=_knn_matrix(knn_exp, "test", test["ids"]) if have_knn else None,
        duplicate_dev=lookup_duplicates([dev_rows[i] for i in dev["ids"]], raw["train"]),
        duplicate_test=lookup_duplicates([test_rows[i] for i in test["ids"]], raw["train"], raw["dev"]),
        name="A4stack",
    )


def _raw_half(exp: str, ids: list[str]) -> None:
    """Plain 0.5 thresholds, for the decision-layer contrast row."""
    test_path = os.path.join(exp_dir(exp), "test_logits.npz")
    if not os.path.exists(test_path):
        print(f"[zips] skip raw {exp}", flush=True)
        return
    test = load_logits(exp, "test")
    pred = sigmoid(test["logits"]) >= 0.5
    _write_aligned("P5_dense_lorma_raw", ids, test["ids"], pred, require_label=False)


def main() -> None:
    ids = test_ids()
    assemble_cf2()
    plan = [
        ("P5_tfidf_lr", "B1", "one"),
        ("P5_dense_lorma_laat", "A1", "one"),
        ("P5_dense_lorma_laat_tapt", "A1t", "one"),
        ("P5_dense_lorma_cf2", "A4", "one"),
        ("P5_hybrid_adapter_laat", "A5", "one"),
        ("P5_ced_partial_laat", "A6", "one"),
        ("P5_conv_kd", "A9", "one"),
        ("P5_dense_full", "A7", "one"),
    ]
    for zip_name, exp, d2 in plan:
        built = _member_pred(exp, _CHOSEN, d2)
        if built is None:
            continue
        order, pred = built
        _write_aligned(zip_name, ids, order, pred, require_label=(d2 != "off"))
        print(f"[zips] {zip_name} dev_cf={_dev_score(exp)}", flush=True)
    _raw_half("A4", ids)
    if os.path.exists(os.path.join(exp_dir("A4"), "dev_logits.npz")) and os.path.exists(
        os.path.join(exp_dir("B1"), "dev_logits.npz")
    ):
        dev = load_logits("A4", "dev")
        half = dev_half_array(dev["ids"], p5_dir())
        info = build_mix(["A4", "B1"], dev["ids"], "dev", dev["labels"], half, "mix_dense_tfidf")
        test = load_logits("A4", "test")
        build_mix(
            ["A4", "B1"], test["ids"], "test", None, None, "mix_dense_tfidf",
            temperatures=info.get("temperature"), weight_fit=info.get("weights"),
        )
        built = _member_pred("mix_dense_tfidf", _CHOSEN, "one")
        if built:
            order, pred = built
            _write_aligned("P5_mix_dense_tfidf", ids, order, pred, require_label=True)
    stack_report = os.path.join(exp_dir("A4stack"), "decide.json")
    if not os.path.exists(stack_report) and os.path.exists(os.path.join(exp_dir("A4"), "dev_logits.npz")):
        _fit_a4_stack()
    if os.path.exists(stack_report):
        kept = json.load(open(stack_report, encoding="utf-8"))
        pred_path = os.path.join(exp_dir("A4stack"), "test_pred.npz")
        if (kept.get("d3") or kept.get("d4")) and os.path.exists(pred_path):
            blob = np.load(pred_path, allow_pickle=True)
            _write_aligned(
                "P5_dense_cf2_stack", ids, [str(i) for i in blob["ids"]], blob["pred"], require_label=True,
            )
    mix = os.path.join(exp_dir("mix"), "test_logits.npz")
    if os.path.exists(mix):
        built = _member_pred("mix", _CHOSEN, "one")
        if built:
            order, pred = built
            index = {case_id: i for i, case_id in enumerate(order)}
            aligned = pred[[index[i] for i in ids]]
            write_zip(rows_from_matrix(ids, aligned),
                      os.path.join(repo_root(), "submissions", "phase5", "P5_mix_all"),
                      expected=ids, require_label=True)
        # Precision/recall neighbours of the chosen kappa, and the second-label rule.
        dev = load_logits("mix", "dev")
        half = dev_half_array(dev["ids"], p5_dir())
        chosen = choose_kappa(sigmoid(dev["logits"]), dev["labels"], half)["kappa"]
        for kappa in KAPPAS:
            if kappa == chosen:
                continue
            label = "global" if kappa is None else str(kappa)
            built = _member_pred("mix", kappa, "one")
            if not built:
                continue
            order, pred = built
            _write_aligned(f"P5_mix_kappa_{label}", ids, order, pred, require_label=True)
        built = _member_pred("mix", chosen, "two")
        if built:
            order, pred = built
            _write_aligned("P5_mix_all_d2k", ids, order, pred, require_label=True)
    else:
        print("[zips] mix logits not ready; mix zips skipped", flush=True)
    mix_v2 = os.path.join(exp_dir("mix_v2"), "test_logits.npz")
    if os.path.exists(mix_v2):
        built = _member_pred("mix_v2", _CHOSEN, "one")
        if built:
            order, pred = built
            _write_aligned("P5_mix_all_v2", ids, order, pred, require_label=True)


if __name__ == "__main__":
    main()
