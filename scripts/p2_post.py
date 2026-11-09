"""Tune tokdec + the role resolver + boundary rules on out-of-fold outputs.

    python scripts/p2_post.py E1v2 --save E1v2
    python scripts/p2_post.py E1v2 --nested

When the nested score trails the full-OOF score by more than 2.5 points, the per-label
propagate, lexicon and placement thresholds are collapsed to one global value and the
reduced config is what gets saved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.data import load_folds, load_split  # noqa: E402
from saul.ner.post import Stack, fit_test_model, nested_macro, tune_stack  # noqa: E402
from saul.ner.registry import p2_dir  # noqa: E402
from saul.ner.tokdec import average_logps, load_run_logps, oof_lexicons  # noqa: E402


def _load(exps: list[str]):
    docs = load_split("train")
    folds = load_folds(docs)
    sets = [load_run_logps([os.path.join(p2_dir(), e, f"fold{i}") for i in range(5)]) for e in exps]
    logps = sets[0] if len(sets) == 1 else average_logps(sets)
    fold_of = {did: i for i, ids in enumerate(folds) for did in ids}
    return docs, folds, logps, fold_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--save", default=None)
    ap.add_argument("--nested", action="store_true")
    args = ap.parse_args()
    docs, folds, logps, fold_of = _load(args.exps)
    lex = oof_lexicons(docs, folds)
    stack = Stack(docs, logps, lex, fold_of)
    cfg = tune_stack(stack, verbose=True)
    nested = None
    if args.nested:
        mean, scores = nested_macro(docs, logps, folds, lex, shared=False)
        gap = cfg["oof_f1"] - mean
        print(f"[post] nested={mean:.4f} folds={[round(s, 4) for s in scores]} gap={gap:.4f}")
        if gap > 0.025:
            cfg = tune_stack(stack, shared=True, verbose=True)
            mean, scores = nested_macro(docs, logps, folds, lex, shared=True)
            print(f"[post] collapsed nested={mean:.4f} gap={cfg['oof_f1'] - mean:.4f}")
        nested = mean
        cfg["nested_f1"] = nested
    if args.save:
        dest = os.path.join(p2_dir(), args.save)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "post.json"), "w", encoding="utf-8") as f:
            json.dump({**cfg, "exps": args.exps}, f, indent=2)
        bundle = fit_test_model(docs, logps, folds, cfg)
        if bundle is not None:
            import joblib
            joblib.dump(bundle, os.path.join(dest, "p2_model.joblib"))
        print(f"[post] wrote {dest}/post.json")


if __name__ == "__main__":
    main()
