"""Build a Codabench zip: prediction.json at the archive root, official checks applied."""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np
import torch

from saul.ner.backbone import load_ner, load_overlay
from saul.ner.data import load_folds, load_split
from saul.ner.infer import infer_docs
from saul.ner.labels import LABELS, N_TAGS
from saul.ner.registry import p2_dir, repo_root
from saul.ner.post import default_post_config, fit_test_model, post_process
from saul.ner.tokdec import Decoder, average_logps, build_lexicon, load_run_logps, oof_lexicons, raw_spans


def _check(pred: dict, test_ids: set[str]) -> None:
    if set(pred) != set(test_ids):
        missing = set(test_ids) - set(pred)
        extra = set(pred) - set(test_ids)
        raise ValueError(f"ID mismatch missing={len(missing)} extra={len(extra)}")
    allowed = set(LABELS)
    empty = {lab: 0 for lab in LABELS}
    for did, spans in pred.items():
        seen = set()
        for span in spans:
            if len(span) not in (3, 4):
                raise ValueError(f"{did}: bad span {span}")
            a, b, lab = span[:3]
            if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b <= a:
                raise ValueError(f"{did}: bad offsets {a, b}")
            if lab not in allowed:
                raise ValueError(f"{did}: bad label {lab}")
            key = (a, b, lab)
            if key in seen:
                raise ValueError(f"{did}: duplicate {key}")
            seen.add(key)
            empty[lab] += 1
    for lab, n in empty.items():
        if n == 0:
            print(f"[warn] label {lab} has no predictions")


def write_zip(pred: dict, directory: str, name: str) -> str:
    os.makedirs(directory, exist_ok=True)
    json_path = os.path.join(directory, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pred, f)
    zpath = os.path.join(directory, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="prediction.json")
    log_dir = os.path.join(repo_root(), "submissions")
    if os.path.abspath(directory).startswith(os.path.abspath(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "sub_p2.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name}\t{zpath}\n")
    print(f"[submit] {zpath}", flush=True)
    return zpath


def tune_tokens(exp_dirs: list[str], weights=None):
    """Tune the token decoder on the out-of-fold distributions of one or more experiments."""
    train_docs = load_split("train")
    sets = [load_run_logps([os.path.join(p2_dir(), e, f"fold{k}") for k in range(5)]) for e in exp_dirs]
    logps = sets[0] if len(sets) == 1 else average_logps(sets, weights)
    dec = Decoder(train_docs, logps, oof_lexicons(train_docs, load_folds(train_docs)))
    return dec.tune(verbose=True), train_docs


def test_logps(exp: str, device: str | None = None) -> dict[str, np.ndarray]:
    """Fold-averaged test distributions, cached under <exp>/test/."""
    out_dir = os.path.join(p2_dir(), exp, "test")
    test_docs = load_split("test")
    if os.path.exists(os.path.join(out_dir, "done.json")):
        return load_run_logps([out_dir])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    total = {d.doc_id: np.zeros((len(d.ids), N_TAGS), dtype=np.float64) for d in test_docs}
    n = 0
    for fold in range(5):
        run = os.path.join(p2_dir(), exp, f"fold{fold}")
        meta = json.load(open(os.path.join(run, "done.json"), encoding="utf-8"))
        model = load_ner(meta["model"], meta["technique"], context=meta["context"], device=device)
        load_overlay(model, torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
        model.to(device)
        result = infer_docs(
            model, test_docs, batch_size=4, device=device, mentions=False,
            header_mode=meta.get("header_mode", "raw"), preamble=bool(meta.get("preamble", False)),
        )
        for did in total:
            total[did] += np.exp(result[did]["tag_logprobs"].astype(np.float64))
        n += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for did, mass in total.items():
        out[did] = np.log(np.maximum(mass / n, 1e-8)).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{did}.npz"), tag_logprobs=out[did])
    with open(os.path.join(out_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": n}, f)
    return out


def decode_test(logps: dict[str, np.ndarray], cfg: dict, train_docs, bundle: dict | None = None) -> dict[str, list]:
    test_docs = load_split("test")
    lexicon = build_lexicon(train_docs)
    cfg = {**default_post_config(), **cfg}
    pred = {}
    for doc in test_docs:
        raw = raw_spans(
            doc.text, doc.starts, doc.ends, logps[doc.doc_id],
            cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"),
        )
        spans = post_process(doc, raw, logps[doc.doc_id], cfg, bundle, lexicon)
        pred[doc.doc_id] = [[int(s), int(e), lab] for s, e, lab in spans]
    _check(pred, {d.doc_id for d in test_docs})
    return pred


def _load_bundle(exp: str, cfg: dict, train_docs):
    if not cfg.get("use_p2_model"):
        return None
    path = os.path.join(p2_dir(), exp, "p2_model.joblib")
    if os.path.exists(path):
        import joblib
        return joblib.load(path)
    from saul.ner.data import load_folds
    from saul.ner.tokdec import load_run_logps as _load
    logps = _load([os.path.join(p2_dir(), exp, f"fold{k}") for k in range(5)])
    return fit_test_model(train_docs, logps, load_folds(train_docs), cfg)


def _tokdec_only(cfg: dict) -> dict:
    """Keep a saved token-decoder config and force the later stages off."""
    cfg = {**default_post_config(), **cfg}
    cfg["p2_rules"] = False
    cfg["use_p2_model"] = False
    for key in ("p3_caseno", "p3_prec_split", "p3_prec_cite", "p3_prec_short",
                "p3_auth_gov", "p3_auth_cand", "p3_court_split", "p3_junk"):
        cfg[key] = False
    return cfg


def submit_exp(exp: str, name: str | None = None, device: str | None = None, tokdec_only: bool = False) -> str:
    post_path = os.path.join(p2_dir(), exp, "post.json")
    tok_path = os.path.join(p2_dir(), exp, "tokdec.json")
    if tokdec_only and os.path.exists(tok_path):
        # Do not retune. A nested run may already have collapsed the per-label switches.
        train_docs = load_split("train")
        with open(tok_path, encoding="utf-8") as f:
            cfg = _tokdec_only(json.load(f))
        bundle = None
    elif tokdec_only or not os.path.exists(post_path):
        cfg, train_docs = tune_tokens([exp])
        cfg = _tokdec_only(cfg) if tokdec_only else {**default_post_config(), **cfg}
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        bundle = None
    else:
        train_docs = load_split("train")
        with open(post_path, encoding="utf-8") as f:
            cfg = json.load(f)
        bundle = _load_bundle(exp, cfg, train_docs)
    pred = decode_test(test_logps(exp, device), cfg, train_docs, bundle)
    name = name or exp
    dest = os.path.join(repo_root(), "submissions", "phase2", name)
    return write_zip(pred, dest, name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--tokdec-only", action="store_true",
                   help="S1 calibration zip: tokdec only, no P2 or P3")
    args = p.parse_args()
    submit_exp(args.exp, args.name, tokdec_only=args.tokdec_only)


if __name__ == "__main__":
    main()
