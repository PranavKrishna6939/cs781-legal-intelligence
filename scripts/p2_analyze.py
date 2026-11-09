"""Out-of-fold error analysis for one or more Phase-2 runs.

    python scripts/p2_analyze.py E1 probe_ced --examples 6

Compares the saved decoder with a plain legal-BIOES Viterbi over the averaged token
distributions, then prints per-label scores and error samples.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.ner.data import load_split  # noqa: E402
from saul.ner.decode import decode_doc, default_config, load_mention_dir  # noqa: E402
from saul.ner.labels import LABELS, legal_transition_masks, spans_from_tags  # noqa: E402
from saul.ner.metrics import strict_scores  # noqa: E402
from saul.ner.registry import p2_dir  # noqa: E402


def legal_viterbi(logp: np.ndarray) -> list[int]:
    ok, start, end = (m.numpy() for m in legal_transition_masks())
    trans = np.where(ok, 0.0, -1e4)
    n, k = logp.shape
    score = np.where(start, 0.0, -1e4) + logp[0]
    back = np.zeros((n, k), dtype=np.int32)
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    score = score + np.where(end, 0.0, -1e4)
    tag = int(score.argmax())
    path = [tag]
    for t in range(n - 1, 0, -1):
        tag = int(back[t, tag])
        path.append(tag)
    return path[::-1]


def run_dirs(exp: str) -> list[str]:
    root = os.path.join(p2_dir(), exp)
    return sorted(os.path.join(root, d) for d in os.listdir(root) if d.startswith("fold"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--tokdec", action="store_true", help="also score the token decoder")
    ap.add_argument("--post", action="store_true", help="also score tokdec + P2 + P3")
    args = ap.parse_args()
    docs = {d.doc_id: d for d in load_split("train")}
    for exp in args.exps:
        gold, pred_cur, pred_vit = {}, {}, {}
        for run in run_dirs(exp):
            mentions = load_mention_dir(run)
            for name in os.listdir(run):
                if not name.endswith(".npz"):
                    continue
                did = name[:-4]
                doc = docs[did]
                z = np.load(os.path.join(run, name))
                logp = z["tag_logprobs"].astype(np.float64)
                gold[did] = list(doc.gold)
                pred_cur[did] = decode_doc(doc.text, mentions.get(did, []), default_config())
                path = legal_viterbi(logp)
                pred_vit[did] = spans_from_tags(path, doc.starts, doc.ends)
        print(f"=== {exp}: {len(gold)} docs")
        for name, pred in (("current", pred_cur), ("viterbi", pred_vit)):
            sc = strict_scores(gold, pred)
            per = " ".join(f"{lab}={sc['per_label'][lab]['f']:.2f}" for lab in LABELS)
            print(f"  {name:8s} macro={sc['macro_f1']:.4f} P={sc['macro_p']:.3f} R={sc['macro_r']:.3f}  {per}")
        sc = strict_scores(gold, pred_vit)
        for lab in LABELS:
            st = sc["per_label"][lab]
            print(f"  [vit] {lab:10s} tp={st['tp']:4d} fp={st['fp']:4d} fn={st['fn']:4d} f={st['f']:.3f}")
        if args.dump:
            os.makedirs(args.dump, exist_ok=True)
            with open(os.path.join(args.dump, f"{exp}_viterbi.json"), "w") as f:
                json.dump({d: [list(x) for x in v] for d, v in pred_vit.items()}, f)
        if args.tokdec or args.post:
            _extra(exp, docs, gold, args)
        if args.examples:
            _examples(docs, gold, pred_vit, args.examples)


def _extra(exp: str, docs, gold, args) -> None:
    from saul.ner.data import load_folds, load_split
    from saul.ner.post import Stack, default_post_config
    from saul.ner.tokdec import Decoder, load_run_logps, oof_lexicons
    train = load_split("train")
    folds = load_folds(train)
    logps = load_run_logps(run_dirs(exp))
    lex = oof_lexicons(train, folds)
    if args.tokdec:
        pred = Decoder(train, logps, lex).predict(default_post_config())
        sc = strict_scores(gold, {k: v for k, v in pred.items() if k in gold})
        print(f"  tokdec   macro={sc['macro_f1']:.4f}")
        _examples(docs, gold, pred, args.examples)
    if args.post:
        path = os.path.join(p2_dir(), exp, "post.json")
        cfg = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default_post_config()
        fold_of = {did: i for i, ids in enumerate(folds) for did in ids}
        pred = Stack(train, logps, lex, fold_of).predict(cfg)
        sc = strict_scores(gold, {k: v for k, v in pred.items() if k in gold})
        print(f"  post     macro={sc['macro_f1']:.4f}")
        _examples(docs, gold, pred, args.examples)


def _examples(docs, gold, pred, k: int) -> None:
    rng = np.random.default_rng(0)
    by_lab = collections.defaultdict(lambda: {"fn": [], "fp": [], "bnd": []})
    for did, gs in gold.items():
        text = docs[did].text
        g = {(a, b, c) for a, b, c in gs}
        p = {(a, b, c) for a, b, c in pred.get(did, [])}
        for a, b, c in g - p:
            over = [x for x in p if x[0] < b and a < x[1]]
            if over:
                x = over[0]
                by_lab[c]["bnd"].append(f"{did} gold[{text[a:b]!r}:{c}] pred[{text[x[0]:x[1]]!r}:{x[2]}]")
            else:
                ctx = text[max(0, a - 40):b + 40].replace("\n", " ")
                by_lab[c]["fn"].append(f"{did} miss {text[a:b]!r}  ctx={ctx!r}")
        for a, b, c in p - g:
            if not any(x[0] < b and a < x[1] for x in g):
                ctx = text[max(0, a - 40):b + 40].replace("\n", " ")
                by_lab[c]["fp"].append(f"{did} extra {text[a:b]!r}  ctx={ctx!r}")
    for lab in LABELS:
        print(f"--- {lab}: fn={len(by_lab[lab]['fn'])} fp={len(by_lab[lab]['fp'])} boundary/label={len(by_lab[lab]['bnd'])}")
        for kind in ("bnd", "fn", "fp"):
            items = by_lab[lab][kind]
            if not items:
                continue
            for i in rng.choice(len(items), size=min(k, len(items)), replace=False):
                print(f"    {kind}: {items[int(i)][:300]}")


if __name__ == "__main__":
    main()
