"""Probability-space mixtures of experiments. All models share the legal32k tokenizer, so
their per-token BIOES distributions line up. Weights are chosen on out-of-fold outputs."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from saul.ner.data import load_folds, load_split
from saul.ner.post import Stack, fit_test_model, tune_stack
from saul.ner.registry import p2_dir, repo_root
from saul.ner.tokdec import Decoder, average_logps, load_run_logps, oof_lexicons


def _oof(exp: str) -> dict[str, np.ndarray]:
    return load_run_logps([os.path.join(p2_dir(), exp, f"fold{k}") for k in range(5)])


def search_weights(exps: list[str], samples: int = 24, seed: int = 781) -> dict:
    docs = load_split("train")
    lex = oof_lexicons(docs, load_folds(docs))
    sets = [_oof(e) for e in exps]
    uniform = np.ones(len(exps)) / len(exps)
    cfg = Decoder(docs, average_logps(sets, uniform), lex).tune()
    rng = np.random.default_rng(seed)
    candidates = [uniform] + [rng.dirichlet(np.ones(len(exps))) for _ in range(samples)]
    for i in range(len(exps)):
        w = np.ones(len(exps))
        w[i] = 0.0
        if len(exps) > 1:
            candidates.append(w / w.sum())
    best = {"weights": uniform.tolist(), "f1": -1.0}
    for w in candidates:
        f1 = Decoder(docs, average_logps(sets, w), lex).score(cfg)["macro_f1"]
        if f1 > best["f1"]:
            best = {"weights": [float(x) for x in w], "f1": float(f1)}
    mixed = average_logps(sets, best["weights"])
    fold_of = {did: i for i, ids in enumerate(load_folds(docs)) for did in ids}
    stack = Stack(docs, mixed, lex, fold_of)
    cfg = tune_stack(stack, verbose=True)
    print(f"[ensemble] weights {best['weights']} post_f1={cfg['oof_f1']:.4f}", flush=True)
    return {"exps": exps, "weights": best["weights"], "cfg": cfg, "oof_f1": cfg["oof_f1"]}


def submit_mix(name: str, found: dict) -> str:
    from saul.ner.submit import decode_test, test_logps, write_zip
    sets = [test_logps(e) for e in found["exps"]]
    docs = load_split("train")
    bundle = fit_test_model(docs, average_logps([_oof(e) for e in found["exps"]], found["weights"]),
                            load_folds(docs), found["cfg"])
    pred = decode_test(average_logps(sets, found["weights"]), found["cfg"], docs, bundle)
    return write_zip(pred, os.path.join(repo_root(), "submissions", "phase2", name), name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exps", required=True, help="comma-separated experiment names")
    p.add_argument("--name", required=True)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    exps = [e.strip() for e in args.exps.split(",") if e.strip()]
    found = search_weights(exps, samples=args.samples)
    path = os.path.join(p2_dir(), "ensemble", args.name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(found, f, indent=2)
    print(f"[ensemble] wrote {path}")
    if args.submit:
        submit_mix(args.name, found)


if __name__ == "__main__":
    main()
