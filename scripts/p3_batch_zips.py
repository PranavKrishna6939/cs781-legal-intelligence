"""Build S1–S6 from cached posteriors. Missing caches are skipped, not fatal."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saul.ner.registry import repo_root
from saul.rr.data import LABELS, labeled_docs, load_docs, p3_dir
from saul.rr.decode import (
    bias_array, decode_experiment, exact_oof, fit_decoder, load_split_probs,
    nested_stack_gain, predict_docs, save_cfg, transitions,
)
from saul.rr.ensemble import g2_note, save_mix_probs
from saul.rr.score import macro_scores
from saul.rr.submit import write_zip


RAW = {"CL": 0.0, "IT": 0.0}


def _test_docs():
    return [d for d in load_docs() if d.split == "test"]


def _zip(name: str, pred: dict) -> None:
    directory = os.path.join(repo_root(), "submissions", "phase3", name)
    write_zip(pred, directory, _test_docs())


def _require_complete(exp: str) -> None:
    """A submission needs the merged five-fold test cache, not a single fold's OOF."""
    if not os.path.exists(os.path.join(p3_dir(), exp, "test", "done.json")):
        raise FileNotFoundError(exp)


def _raw(exp: str) -> dict:
    _require_complete(exp)
    docs = load_docs()
    labeled = labeled_docs(docs)
    probs = load_split_probs(exp, "test")
    if not probs:
        raise FileNotFoundError(exp)
    return predict_docs(_test_docs(), probs, transitions(labeled), RAW, np.zeros(13))


def _tuned(exp: str) -> dict:
    _require_complete(exp)
    cfg_path = os.path.join(p3_dir(), exp, "decode.json")
    if not os.path.exists(cfg_path):
        cfg = fit_decoder(exp)
        _maybe_extras(exp, cfg)
        save_cfg(exp, cfg)
    return decode_experiment(exp, kind="test", use_exact=True)


def _maybe_extras(exp: str, cfg: dict) -> None:
    docs = labeled_docs(load_docs())
    probs = load_split_probs(exp, "oof")
    folds = json.load(open(os.path.join(p3_dir(), "folds.json"), encoding="utf-8"))
    base = cfg.get("nested_kept", cfg["nested"])
    gain = nested_stack_gain(docs, probs, folds, base)
    cfg["stack_gain"] = gain
    cfg["stack_kept"] = gain >= 0.01
    from saul.rr.decode import _score
    trans = transitions(docs)
    lam = cfg["lambda"]
    bias = bias_array(cfg)
    _, paths = _score(docs, probs, trans, lam, bias)
    pred = {doc_id: [LABELS[int(i)] for i in path] for doc_id, path in paths.items()}
    exact = exact_oof(docs, pred)
    cfg["exact_rate"] = exact["rate"]
    cfg["exact_kept"] = exact["kept"]
    dev_ids = {d.doc_id for d in docs if d.split == "dev"}
    dgold, dpred = [], []
    dom = {name: ([], []) for name in ("CL", "IT")}
    gold, got = [], []
    for doc in docs:
        if doc.doc_id not in paths:
            continue
        ids = [int(i) for i in paths[doc.doc_id]]
        gold.extend(doc.label_ids)
        got.extend(ids)
        dom[doc.domain][0].extend(doc.label_ids)
        dom[doc.domain][1].extend(ids)
        if doc.doc_id in dev_ids:
            dgold.extend(doc.label_ids)
            dpred.extend(ids)
    per = macro_scores(gold, got)["per_label"] if gold else {}
    cfg["per_label"] = {LABELS[i]: float(per[i]) for i in per}
    for name, (yg, yp) in dom.items():
        if yg:
            cfg[f"{name}_f1"] = macro_scores(yg, yp)["macro_f1"]
            print(f"[decode] {exp} {name} {cfg[f'{name}_f1']:.4f}")
    if per:
        print("[decode] per-label " + ", ".join(f"{LABELS[i]} {per[i]:.3f}" for i in range(len(LABELS))))
    if dgold:
        cfg["official_dev_f1"] = macro_scores(dgold, dpred)["macro_f1"]
        print(f"[decode] {exp} official-dev {cfg['official_dev_f1']:.4f} "
              f"kept {cfg.get('kept', cfg['tuned']):.4f} collapsed={cfg['collapsed']} "
              f"stack={cfg['stack_kept']} exact={cfg['exact_kept']}")


def build_s1_s6() -> None:
    jobs = [
        ("P3_E1_dense_lorma_raw", lambda: _raw("E1")),
        ("P3_E1_dense_lorma", lambda: _tuned("E1")),
        ("P3_E3_ced_ia3", lambda: _tuned("E3")),
        ("P3_E2_hybrid_rslora", lambda: _tuned("E2")),
        ("P3_mix_E1E2E3", _mix),
        ("P3_E0_dense_sentence_only", lambda: _raw("E0")),
    ]
    for name, fn in jobs:
        try:
            _zip(name, fn())
        except FileNotFoundError as exc:
            print(f"[batch] skip {name}: {exc}")


def _mix() -> dict:
    exps = ["E1", "E2", "E3"]
    for exp in exps:
        root = os.path.join(p3_dir(), exp)
        if not os.path.exists(os.path.join(root, "test", "done.json")):
            raise FileNotFoundError(exp)
        if not any(os.path.isdir(os.path.join(root, f"fold{k}", "oof")) for k in range(5)):
            raise FileNotFoundError(exp)
    for kind in ("oof", "test"):
        save_mix_probs("P3_mix_E1E2E3", exps, kind=kind)
    cfg = fit_decoder("P3_mix_E1E2E3")
    note = g2_note(exps, cfg.get("kept", cfg["tuned"]))
    cfg["g2"] = note
    _maybe_extras("P3_mix_E1E2E3", cfg)
    save_cfg("P3_mix_E1E2E3", cfg)
    print(f"[g2] {note}")
    return decode_experiment("P3_mix_E1E2E3", kind="test", use_exact=True)


def build_named(name: str, exp: str, tuned: bool = True) -> None:
    _zip(name, _tuned(exp) if tuned else _raw(exp))


def build_rule_free(name: str, exp: str) -> None:
    """Viterbi decoder only: no exact-match lookup and no dissent regex."""
    _require_complete(exp)
    cfg_path = os.path.join(p3_dir(), exp, "decode.json")
    if not os.path.exists(cfg_path):
        save_cfg(exp, fit_decoder(exp))
    oof = decode_experiment(exp, kind="oof", use_stack=False, use_exact=False, use_dissent=False)
    docs = labeled_docs(load_docs())
    gold, got = [], []
    for doc in docs:
        if doc.doc_id not in oof:
            continue
        gold.extend(doc.label_ids)
        got.extend(LABELS.index(lab) for lab in oof[doc.doc_id])
    scores = macro_scores(gold, got)
    dissent = LABELS.index("Dissent")
    print(f"[rule-free] {name} OOF macro-F1 {scores['macro_f1']:.4f} "
          f"Dissent {scores['per_label'][dissent]:.4f}")
    _zip(name, decode_experiment(exp, kind="test", use_stack=False, use_exact=False, use_dissent=False))


DISSENT_SET = [
    ("P3_mix_E1E2E3_dissent", "P3_mix_E1E2E3"),
    ("P3_E1_dense_lorma_dissent", "E1"),
    ("P3_E2_hybrid_rslora_dissent", "E2"),
    ("P3_E3_ced_ia3_dissent", "E3"),
]


def build_dissent(pairs=None) -> None:
    from saul.rr.decode import apply_dissent, dissent_oof, dissent_start

    docs = load_docs()
    labeled = labeled_docs(docs)
    for doc in _test_docs():
        start = dissent_start(doc)
        if start is not None:
            print(f"[dissent] test {doc.doc_id} from sentence {start} of {len(doc.sentences)}")
    for name, exp in pairs or DISSENT_SET:
        try:
            _require_complete(exp)
        except FileNotFoundError:
            print(f"[batch] skip {name}: {exp}")
            continue
        oof = decode_experiment(exp, kind="oof", use_stack=False, use_exact=False)
        gate = dissent_oof(labeled, oof)
        print(f"[dissent] {exp} OOF {gate['before']:.4f} -> {gate['after']:.4f} fired={len(gate['fired'])}")
        cfg_path = os.path.join(p3_dir(), exp, "decode.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["dissent_before"] = gate["before"]
        cfg["dissent_after"] = gate["after"]
        cfg["dissent_kept"] = gate["kept"]
        save_cfg(exp, cfg)
        if not gate["kept"]:
            print(f"[batch] skip {name}: rule lowers OOF")
            continue
        pred = decode_experiment(exp, kind="test", use_exact=True, use_dissent=True)
        _zip(name, pred)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dissent":
        build_dissent()
    elif len(sys.argv) > 3 and sys.argv[1] == "--rule-free":
        build_rule_free(sys.argv[2], sys.argv[3])
    else:
        build_s1_s6()
