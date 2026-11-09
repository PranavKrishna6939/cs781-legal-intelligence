"""Tune the token-level decoder on out-of-fold outputs of one or more experiments.

    python scripts/p2_tokdec.py E1                # one experiment
    python scripts/p2_tokdec.py E1 E4 --save mix  # probability average of several
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.data import load_folds, load_split  # noqa: E402
from saul.ner.labels import LABELS  # noqa: E402
from saul.ner.registry import p2_dir  # noqa: E402
from saul.ner.tokdec import (  # noqa: E402
    Decoder, average_logps, build_lexicon, default_tok_config, load_run_logps, oof_lexicons,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--save", default=None, help="write the tuned config to <p2>/<save>/tokdec.json")
    ap.add_argument("--nested", action="store_true",
                    help="leave-one-fold tune; if the gap to full-OOF exceeds 2 points, collapse per-label switches")
    args = ap.parse_args()
    docs = load_split("train")
    folds = load_folds(docs)
    sets = [load_run_logps([os.path.join(p2_dir(), e, f"fold{i}") for i in range(5)]) for e in args.exps]
    logps = sets[0] if len(sets) == 1 else average_logps(sets)
    dec = Decoder(docs, logps, oof_lexicons(docs, folds))
    print(f"docs={len(dec.docs)}")
    base = dec.score(default_tok_config())
    print(f"default macro={base['macro_f1']:.4f}")
    cfg = dec.tune(verbose=True)
    if args.nested:
        held = []
        for ids in folds:
            held_ids = set(ids)
            train = [d for d in docs if d.doc_id not in held_ids and d.doc_id in logps]
            dev = [d for d in docs if d.doc_id in held_ids and d.doc_id in logps]
            if not train or not dev:
                continue
            train_ids = {d.doc_id for d in train}
            groups = []
            for fold_ids in folds:
                group = [i for i in fold_ids if i in train_ids]
                if group:
                    groups.append(group)
            train_lex = oof_lexicons(train, groups)
            dev_one = build_lexicon(train)
            dev_lex = {d.doc_id: dev_one for d in dev}
            inner = Decoder(train, logps, train_lex).tune()
            held.append(Decoder(dev, logps, dev_lex).score(inner)["macro_f1"])
        nested = sum(held) / len(held) if held else 0.0
        gap = cfg["oof_f1"] - nested
        print(f"[tokdec] nested={nested:.4f} full={cfg['oof_f1']:.4f} gap={gap:.4f}")
        if gap > 0.02:
            best = -1.0
            shared = None
            for prop in (False, True):
                for lex in (False, True):
                    trial = json.loads(json.dumps(cfg))
                    trial["propagate"] = {lab: prop for lab in LABELS}
                    trial["lexicon"] = {lab: lex for lab in LABELS}
                    score = dec.score(trial)["macro_f1"]
                    if shared is None or score > best + 1e-6:
                        best, shared = score, trial
            shared["oof_f1"] = best
            cfg = shared
            print(f"[tokdec] collapsed per-label switches, oof_f1={best:.4f}")
    sc = dec.score(cfg)
    print("  " + " ".join(f"{lab}={sc['per_label'][lab]['f']:.2f}" for lab in LABELS))
    print(json.dumps({k: v for k, v in cfg.items()}, indent=None))
    if args.save:
        path = os.path.join(p2_dir(), args.save, "tokdec.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**cfg, "exps": args.exps}, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
