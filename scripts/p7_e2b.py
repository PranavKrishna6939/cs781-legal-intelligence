"""Second cross-encoder: quote-aligned positives, short training, max-pooled scores fused with E0."""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from p7_bm25 import micro_f1, write_zip
from p7_common import (
    OUT,
    apply_mask,
    check_submission,
    citation_contexts,
    decode,
    gold_of,
    halves,
    masks,
    passages,
    save_metrics,
    split_bundle,
    zrows,
)
from p7_e2 import GROUP, flatten, logits_of, overlap
from p7_model import build, checkpoint_every, cosine, gpu_mb, load_tokenizer

STEPS = 250


def grams(text: str) -> set:
    words = text.lower().split()
    return {tuple(words[i:i + 8]) for i in range(len(words) - 7)}


def build_groups() -> list[dict]:
    train = split_bundle("Train")
    saved = np.load(os.path.join(OUT, "e0", "train.npz"))
    q_index = {q: i for i, q in enumerate(saved["qids"].tolist())}
    cand = [str(c) for c in saved["cand"].tolist()]
    full = saved["full"]
    passage_of = {cid: passages(sents) for cid, sents in zip(train["cand"], train["sents"])}
    gold = gold_of(train["queries"])
    groups = []
    quoted = 0
    for q, sents in zip(train["qids"], train["q_sents"]):
        ctx = citation_contexts(sents) or ["context: " + " ".join(sents[:6])]
        ctx_grams = [(text, grams(text)) for text in ctx]
        negs = []
        for col in np.argsort(-full[q_index[q]]):
            cid = cand[col]
            if cid == q or cid in gold[q]:
                continue
            negs.append(cid)
            if len(negs) >= 15:
                break
        neg_passages = []
        for cid in negs:
            neg_passages.extend(passage_of.get(cid, [])[:2])
        if len(neg_passages) < GROUP - 1:
            continue
        for cid in list(gold[q])[:4]:
            options = passage_of.get(cid) or []
            best_context, best_passage, best_n = None, None, 0
            for text, cg in ctx_grams:
                if not cg:
                    continue
                for passage in options:
                    shared = len(cg & grams(passage))
                    if shared > best_n:
                        best_context, best_passage, best_n = text, passage, shared
            if best_passage is None:
                continue
            quoted += 1
            chosen = random.Random(hash(q + cid) & 0xFFFFFFFF).sample(neg_passages, GROUP - 1)
            groups.append({"context": best_context, "passages": [best_passage] + chosen})
    print(f"[e2b] quote groups {len(groups)} quoted {quoted}", flush=True)
    return groups


def train(groups, device, tok, bos, sep, pad):
    model = build("partial")
    scorer = nn.Linear(model.config.d_model, 1).to(device)
    checkpoint_every(model)
    model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": 2e-5}, {"params": scorer.parameters(), "lr": 1e-3}],
        weight_decay=0.01,
    )
    batch = 4
    print(f"[e2b] trainable {sum(p.numel() for p in trainable)}", flush=True)
    order = list(range(len(groups)))
    random.Random(781).shuffle(order)
    step = 0
    cursor = 0
    while step < STEPS:
        chunk = [groups[order[i % len(order)]] for i in range(cursor, cursor + batch)]
        cursor += batch
        contexts, texts = flatten(chunk)
        opt.zero_grad(set_to_none=True)
        raw = logits_of(model, scorer, tok, bos, sep, pad, contexts, texts, device).view(len(chunk), GROUP)
        loss = F.cross_entropy(raw, torch.zeros(len(chunk), dtype=torch.long, device=device))
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + list(scorer.parameters()), 1.0)
        for group, base in zip(opt.param_groups, (2e-5, 1e-3)):
            group["lr"] = cosine(step, STEPS, base)
        opt.step()
        step += 1
        if step % 20 == 0:
            print(f"[e2b] step {step}/{STEPS} loss {float(loss.detach()):.3f} gpu {gpu_mb()}MB", flush=True)
    os.makedirs(os.path.join(OUT, "e2b"), exist_ok=True)
    torch.save(
        {"model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
         "scorer": scorer.state_dict()},
        os.path.join(OUT, "e2b", "cross.pt"),
    )
    model.eval()
    return model, scorer


def max_scores(model, scorer, tok, bos, sep, pad, bundle, device, k: int = 40) -> np.ndarray:
    tag = bundle["split"].lower()
    saved = np.load(os.path.join(OUT, "e0", f"{tag}.npz"))
    names = ["full", "head", "context", "shingle"]
    weights = json.load(open(os.path.join(OUT, "e0", "metrics.json")))["weights"]
    lexical = sum(w * zrows(saved[name]) for w, name in zip(weights, names))
    q_index = {str(q): i for i, q in enumerate(saved["qids"].tolist())}
    cand = [str(c) for c in saved["cand"].tolist()]
    col_of = {c: i for i, c in enumerate(cand)}
    passage_of = {cid: passages(sents) for cid, sents in zip(bundle["cand"], bundle["sents"])}
    out = np.full((len(bundle["qids"]), len(cand)), -1e9, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for row, (q, sents) in enumerate(zip(bundle["qids"], bundle["q_sents"])):
            ctx = citation_contexts(sents)[:4] or ["context: " + " ".join(sents[:4])]
            short = []
            for col in np.argsort(-lexical[q_index[q]]):
                cid = cand[col]
                if cid != q:
                    short.append(cid)
                if len(short) >= k:
                    break
            contexts, texts, owners = [], [], []
            for cid in short:
                opts = passage_of.get(cid) or []
                best = sorted(opts, key=lambda text: max(overlap(text, c) for c in ctx), reverse=True)[:2]
                for passage in best:
                    for context in ctx:
                        contexts.append(context)
                        texts.append(passage)
                        owners.append(cid)
            totals = {cid: -1e9 for cid in short}
            for start in range(0, len(texts), 16):
                raw = logits_of(
                    model, scorer, tok, bos, sep, pad,
                    contexts[start:start + 16], texts[start:start + 16], device,
                ).float().cpu().numpy()
                for cid, value in zip(owners[start:start + 16], raw):
                    if value > totals[cid]:
                        totals[cid] = float(value)
            for cid, value in totals.items():
                out[row, col_of[cid]] = value
            if row % 20 == 0:
                print(f"[e2b] scored {bundle['split']} {row}", flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bos, pad = load_tokenizer()
    sep = int(tok.convert_tokens_to_ids("<|sep|>"))
    groups = build_groups()
    if len(groups) < 100:
        raise SystemExit(f"too few quote groups: {len(groups)}")
    model, scorer = train(groups, device, tok, bos, sep, pad)
    dev = split_bundle("Dev")
    ce = max_scores(model, scorer, tok, bos, sep, pad, dev, device)
    meta = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    saved = np.load(os.path.join(OUT, "e0", "dev.npz"))
    names = ["full", "head", "context", "shingle"]
    lexical = sum(w * zrows(saved[n]) for w, n in zip(meta["weights"], names))
    self_mask, after = masks(dev["qids"], dev["cand"], dev["q_sents"], dev["sents"])
    gold = gold_of(dev["queries"])
    a_ids, b_ids = halves(dev["qids"])
    base = meta["dev"]
    rows = {}
    best = None
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        z = zrows(ce)
        fused = apply_mask(np.where(ce > -1e8, lexical + lam * z, lexical), self_mask, after)
        pred = decode(dev["qids"], dev["cand"], fused, meta["decoder"], None, tuple(meta["mass_coef"]), meta["eps"])
        fa = micro_f1(gold, pred, [q for q in dev["qids"] if q in a_ids])["f1"]
        fb = micro_f1(gold, pred, [q for q in dev["qids"] if q in b_ids])["f1"]
        rows[str(lam)] = {"A": fa, "B": fb}
        print(f"[e2b] lam {lam} A {fa:.4f} B {fb:.4f}", flush=True)
        better = fa >= base["A"]["f1"] and fb >= base["B"]["f1"] and max(fa - base["A"]["f1"], fb - base["B"]["f1"]) >= 0.03
        if better and (best is None or fa + fb > best[0]):
            best = (fa + fb, lam, pred)
    save_metrics("e2b", {"sweep": rows, "e0": {"A": base["A"]["f1"], "B": base["B"]["f1"]}, "kept": best is not None})
    if best is None:
        print("[e2b] no lambda beat E0 on both halves", flush=True)
        return
    print(f"[e2b] keep lam {best[1]}", flush=True)
    test = split_bundle("Test")
    ce_t = max_scores(model, scorer, tok, bos, sep, pad, test, device)
    test_saved = np.load(os.path.join(OUT, "e0", "test.npz"))
    lex_t = sum(w * zrows(test_saved[n]) for w, n in zip(meta["weights"], names))
    tmask = masks(test["qids"], test["cand"], test["q_sents"], test["sents"])
    z_t = zrows(ce_t)
    fused_t = apply_mask(np.where(ce_t > -1e8, lex_t + best[1] * z_t, lex_t), *tmask)
    pred = decode(test["qids"], test["cand"], fused_t, meta["decoder"], None, tuple(meta["mass_coef"]), meta["eps"])
    check_submission(pred, test["qids"], test["cand"])
    write_zip("P7_E2b", pred)
    print("[e2b] wrote P7_E2b", flush=True)


if __name__ == "__main__":
    main()
