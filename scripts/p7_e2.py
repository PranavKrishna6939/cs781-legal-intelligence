"""E2: passage cross-encoder, top 6 blocks trained, listwise over one positive and 15 negatives."""
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

from p7_common import OUT, citation_contexts, gold_of, half_scores, passages, recall_at, save_metrics, split_bundle, zrows
from p7_model import Head, build, checkpoint_every, collate, cosine, encode_ids, forward_hidden, gpu_mb, load_tokenizer

GROUP = 16


def overlap(a: str, b: str) -> int:
    return len(set(a.lower().split()) & set(b.lower().split()))


def build_groups(limit_gold: int = 4) -> list[dict]:
    train = split_bundle("Train")
    saved = np.load(os.path.join(OUT, "e0", "train.npz"))
    q_index = {q: i for i, q in enumerate(saved["qids"].tolist())}
    cand = [str(c) for c in saved["cand"].tolist()]
    full = saved["full"]
    passage_of = {cid: passages(sents) for cid, sents in zip(train["cand"], train["sents"])}
    gold = gold_of(train["queries"])
    groups = []
    for q, sents in zip(train["qids"], train["q_sents"]):
        ctx = citation_contexts(sents) or ["context: " + " ".join(sents[:6])]
        scores = full[q_index[q]]
        negs = []
        for col in np.argsort(-scores):
            cid = cand[col]
            if cid == q or cid in gold[q]:
                continue
            negs.append(cid)
            if len(negs) >= 20:
                break
        neg_passages = []
        for cid in negs:
            neg_passages.extend(passage_of.get(cid, [])[:2])
        for cid in list(gold[q])[:limit_gold]:
            options = passage_of.get(cid) or []
            if not options or len(neg_passages) < GROUP - 1:
                continue
            context = max(ctx, key=lambda text: overlap(text, " ".join(options[:3])))
            positive = max(options, key=lambda text: overlap(text, context))
            chosen = random.Random(hash(q + cid) & 0xFFFFFFFF).sample(neg_passages, GROUP - 1)
            groups.append({"context": context, "passages": [positive] + chosen, "qid": q, "gold": cid})
    print(f"[e2] groups {len(groups)}", flush=True)
    return groups


def pair_batch(tok, bos, sep, pad, contexts, passages_, device):
    rows = []
    indexes = []
    for context, passage in zip(contexts, passages_):
        left, index = encode_ids(tok, context, bos, 160)
        right, _ = encode_ids(tok, passage, bos, 160)
        if index is None:
            index = min(len(left) - 1, 8)
        ids = left + [sep] + right[1:]
        if len(ids) > 384:
            ids = ids[:384]
            index = min(index, 383)
        rows.append(ids)
        indexes.append(index)
    return collate(rows, pad, device), indexes


def flatten(chunk):
    contexts = [group["context"] for group in chunk for _ in range(GROUP)]
    texts = [passage for group in chunk for passage in group["passages"]]
    return contexts, texts


def logits_of(model, scorer, tok, bos, sep, pad, contexts, passages_, device):
    ids, indexes = pair_batch(tok, bos, sep, pad, contexts, passages_, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        hidden = forward_hidden(model, ids)
    picked = hidden[torch.arange(hidden.shape[0], device=device), torch.tensor(indexes, device=device)]
    return scorer(picked.float()).squeeze(-1)


def main():
    os.makedirs(os.path.join(OUT, "e2"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bos, pad = load_tokenizer()
    sep = int(tok.convert_tokens_to_ids("<|sep|>"))
    groups = build_groups()
    if not groups:
        raise SystemExit("no training groups")
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
    while batch >= 1:
        try:
            sample = groups[:batch]
            contexts, texts = flatten(sample)
            opt.zero_grad(set_to_none=True)
            loss = logits_of(model, scorer, tok, bos, sep, pad, contexts, texts, device)
            loss = loss.view(batch, GROUP)
            loss = F.cross_entropy(loss, torch.zeros(batch, dtype=torch.long, device=device))
            loss.backward()
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[e2] oom at {batch}", flush=True)
            batch //= 2
            torch.cuda.empty_cache()
    if batch < 1:
        raise SystemExit("cross-encoder batch 1 does not fit")
    opt.zero_grad(set_to_none=True)
    open(os.path.join(OUT, "e2", "teacher.jsonl"), "w").close()
    epochs = 2
    total = epochs * ((len(groups) + batch - 1) // batch)
    step = 0
    teacher = []
    print(f"[e2] batch {batch} steps {total} trainable {sum(p.numel() for p in trainable)}", flush=True)
    for epoch in range(epochs):
        order = list(range(len(groups)))
        random.Random(781 + epoch).shuffle(order)
        for start in range(0, len(order), batch):
            chunk = [groups[i] for i in order[start:start + batch]]
            if len(chunk) != batch:
                continue
            contexts, texts = flatten(chunk)
            opt.zero_grad(set_to_none=True)
            raw = logits_of(model, scorer, tok, bos, sep, pad, contexts, texts, device).view(len(chunk), GROUP)
            loss = F.cross_entropy(raw, torch.zeros(len(chunk), dtype=torch.long, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(scorer.parameters()), 1.0)
            for group, base in zip(opt.param_groups, (2e-5, 1e-3)):
                group["lr"] = cosine(step, total, base)
            opt.step()
            if epoch == epochs - 1:
                teacher.append(raw.detach().float().cpu().numpy())
                with open(os.path.join(OUT, "e2", "teacher.jsonl"), "a", encoding="utf-8") as handle:
                    for group, row in zip(chunk, raw.detach().float().cpu().tolist()):
                        handle.write(json.dumps({
                            "context": group["context"], "passages": group["passages"], "logits": row,
                        }) + "\n")
            step += 1
            if step % 20 == 0:
                print(f"[e2] step {step}/{total} loss {float(loss):.3f} gpu {gpu_mb()}MB", flush=True)
    torch.save(
        {"model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
         "scorer": scorer.state_dict()},
        os.path.join(OUT, "e2", "cross.pt"),
    )
    if teacher:
        np.save(os.path.join(OUT, "e2", "teacher_logits.npy"), np.concatenate(teacher, axis=0))
    model.eval()
    dev = split_bundle("Dev")
    dev_scores = score_shortlist(model, scorer, tok, bos, sep, pad, dev, device)
    gold = gold_of(dev["queries"])
    pred = {q: dev_scores[q][:7] for q in dev["qids"]}
    report = half_scores(gold, pred, dev["qids"])
    ranked100 = {q: dev_scores[q] for q in dev["qids"]}
    recall = recall_at(gold, ranked100, dev["qids"], (7, 40))
    e1_path = os.path.join(OUT, "e1", "metrics.json")
    baseline = 0.0
    if os.path.exists(e1_path):
        baseline = json.load(open(e1_path))["neural"]["all"].get("recall@7", 0.0)
    gate = recall["recall@7"] >= baseline + 0.03
    save_metrics("e2", {"dev": report, "recall": recall, "baseline_recall@7": baseline, "gate": gate, "gpu_mb": gpu_mb()})
    print("[e2] recall", recall, "gate", gate, flush=True)
    if gate:
        test = split_bundle("Test")
        test_scores = score_shortlist(model, scorer, tok, bos, sep, pad, test, device)
        from p7_bm25 import write_zip
        from p7_common import check_submission
        pred = {q: test_scores[q][:7] for q in test["qids"]}
        check_submission(pred, test["qids"], test["cand"])
        write_zip("P7_E2", pred)
        print("[e2] wrote P7_E2", flush=True)


def score_shortlist(model, scorer, tok, bos, sep, pad, bundle, device, k: int = 40) -> dict:
    tag = bundle["split"].lower()
    saved = np.load(os.path.join(OUT, "e0", f"{tag}.npz"))
    names = ["full", "head", "context", "shingle"]
    weights = json.load(open(os.path.join(OUT, "e0", "metrics.json")))["weights"]
    lexical = sum(w * zrows(saved[name]) for w, name in zip(weights, names))
    q_index = {q: i for i, q in enumerate(saved["qids"].tolist())}
    cand = [str(c) for c in saved["cand"].tolist()]
    passage_of = {cid: passages(sents) for cid, sents in zip(bundle["cand"], bundle["sents"])}
    out = {}
    model.eval()
    with torch.no_grad():
        for row, (q, sents) in enumerate(zip(bundle["qids"], bundle["q_sents"])):
            ctx = citation_contexts(sents)[:8] or ["context: " + " ".join(sents[:4])]
            order = np.argsort(-lexical[q_index[q]])
            short = []
            for col in order:
                cid = cand[col]
                if cid != q:
                    short.append(cid)
                if len(short) >= k:
                    break
            contexts = []
            texts = []
            owners = []
            for cid in short:
                opts = passage_of.get(cid) or ["passage: "]
                best = sorted(opts, key=lambda text: max(overlap(text, c) for c in ctx), reverse=True)[:2]
                for passage in best:
                    for context in ctx[:4]:
                        contexts.append(context)
                        texts.append(passage)
                        owners.append(cid)
            if not texts:
                out[q] = short[:7]
                continue
            raw = []
            for start in range(0, len(texts), 32):
                raw.append(logits_of(
                    model, scorer, tok, bos, sep, pad,
                    contexts[start:start + 32], texts[start:start + 32], device,
                ).float().cpu())
            raw = torch.cat(raw).numpy()
            totals = {cid: [] for cid in short}
            for cid, value in zip(owners, raw):
                totals[cid].append(value)
            ranked = sorted(short, key=lambda cid: np.logaddexp.reduce(totals[cid]), reverse=True)
            out[q] = ranked
            if row % 20 == 0:
                print(f"[e2] shortlist {bundle['split']} {row}", flush=True)
    json.dump({q: ids[:15] for q, ids in out.items()}, open(os.path.join(OUT, "e2", f"{tag}_top15.json"), "w"))
    return out


if __name__ == "__main__":
    main()
