"""Cross-encoder reranker over the BM25 shortlist. One dense LoRMA epoch."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))
sys.path.insert(0, os.path.join(os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")), "scripts"))

from p7_bm25 import join_text, load_pool, load_queries, micro_f1, rank_split, top_ids
from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.peft import LoRMAPlus, wrap_lorma
from saul.tokenizer.train import load_hf_tokenizer

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
OUT = os.path.join(ROOT, "p7")
SUB = os.path.join(ROOT, "submissions", "phase7")
CKPT = os.path.join(ROOT, "checkpoints", "dense_v1")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizers", "legal32k")
SIDE = 384
TAIL = 256
SHORT = 100
NEG = 16
K = 7
EPOCHS = 2
# The 256-token run used 3.5 GB of the 23 GB A10G at batch 8. These fill the card.
TRAIN_BATCH = 24
EVAL_BATCH = 48


def checkpoint_every(model) -> None:
    import torch.utils.checkpoint as checkpoint

    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)

            def run(*inner):
                return _original(*inner, **kwargs)

            return checkpoint.checkpoint(run, *args, use_reentrant=False)

        block.forward = wrapped


def ids_of(tok, text: str, bos: int, n: int, tail: int = 0) -> list[int]:
    raw = tok.encode(text, add_special_tokens=False)
    if tail and len(raw) > n + tail:
        raw = raw[:n] + raw[-tail:]
    else:
        raw = raw[: n + tail]
    if not raw or raw[0] != bos:
        raw = [bos] + raw
    return raw


def pair_ids(q_ids: list[int], c_ids: list[int], sep: int) -> list[int]:
    return q_ids + [sep] + c_ids


def shortlist(split: str, queries: dict) -> dict[str, list[str]]:
    """Union of the top 100 from the full-text view and the first-40-sentences view."""
    pool_ids, pool_docs, pool_sents = load_pool(split)
    q1, c1, s1 = rank_split(pool_ids, pool_docs, queries, (1, 2), None, 300_000)
    q2, c2, s2 = rank_split(pool_ids, pool_sents, queries, (1, 3), 40, 300_000)
    a = top_ids(q1, c1, s1, SHORT)
    b = top_ids(q2, c2, s2, SHORT)
    # Keep BM25 full-text scores aligned to the union, for the blend.
    index = {cid: i for i, cid in enumerate(c1.tolist() if hasattr(c1, "tolist") else list(c1))}
    out = {}
    bm = {}
    for row, qid in enumerate(q1):
        ids = list(dict.fromkeys(a[qid] + b[qid]))
        out[qid] = ids
        bm[qid] = {cid: float(s1[row, index[cid]]) for cid in ids if cid in index}
    np.savez(os.path.join(OUT, f"rerank_bm25_{split.lower()}.npz"), qids=np.asarray(q1), cand_ids=c1, scores=s1.astype(np.float32))
    return out, bm


def collate(rows: list[list[int]], pad: int, device: str) -> torch.Tensor:
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out


def forward_logits(model, head, ids: torch.Tensor, pad: int) -> torch.Tensor:
    hidden = model(ids, mode="bidirectional", return_hidden=True).hidden
    mask = ids.ne(pad).unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return head(pooled).squeeze(-1)


def halves(qids: list[str]) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(781)
    ids = np.array(sorted(qids))
    rng.shuffle(ids)
    mid = len(ids) // 2
    return set(ids[:mid]), set(ids[mid:])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf_tokenizer(TOKENIZER)
    bos, pad = int(tok.bos_token_id), int(tok.pad_token_id)
    sep = int(tok.convert_tokens_to_ids("<|sep|>"))
    train = load_queries("Train")
    dev = load_queries("Dev")
    test = load_queries("Test")
    print("[rerank] building shortlists", flush=True)
    train_list, _train_bm = shortlist("Train", train)
    dev_list, dev_bm = shortlist("Dev", dev)
    test_list, test_bm = shortlist("Test", test)

    pool = json.load(open(os.path.join(ROOT, "data/raw/Phase_7_PCR/Train/train_candidates.json"), encoding="utf-8"))
    dev_pool = json.load(open(os.path.join(ROOT, "data/raw/Phase_7_PCR/Dev/dev_candidates.json"), encoding="utf-8"))
    test_pool = json.load(open(os.path.join(ROOT, "data/raw/Phase_7_PCR/Test/test_candidates.json"), encoding="utf-8"))
    text = {}
    for raw in (pool, dev_pool, test_pool):
        for cid, row in raw.items():
            text[cid] = join_text(row["text"])
    for qid, row in {**train, **dev, **test}.items():
        text[qid] = join_text(row["text"])

    cache: dict[tuple[str, str], list[int]] = {}

    def pair(qid: str, cid: str) -> list[int]:
        key = (qid, cid)
        if key not in cache:
            cache[key] = pair_ids(
                ids_of(tok, text[qid], bos, SIDE),
                ids_of(tok, text[cid], bos, SIDE, TAIL),
                sep,
            )
        return cache[key]

    examples = []
    for qid, row in train.items():
        gold = [c for c in (row.get("relevant_candidates") or []) if c in train_list.get(qid, []) or c in text]
        gold = [c for c in (row.get("relevant_candidates") or []) if c in text]
        negs = [c for c in train_list.get(qid, []) if c not in set(gold)][:NEG]
        for cid in gold:
            examples.append((qid, cid, 1.0))
        for cid in negs:
            examples.append((qid, cid, 0.0))
    print(f"[rerank] train pairs {len(examples)} positives {sum(y==1 for *_, y in examples)}", flush=True)

    model = SaulForCausalLM.from_pretrained(CKPT)
    wrap_lorma(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRMAPlus):
            for param in list(module.A.parameters()) + list(module.B.parameters()):
                param.requires_grad_(True)
    head = nn.Linear(model.config.d_model, 1)
    model.to(device).train()
    head.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    rng = np.random.default_rng(781)
    order = np.arange(len(examples))
    rng.shuffle(order)
    batch = TRAIN_BATCH
    steps = EPOCHS * math.ceil(len(order) / batch)
    step = 0
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        start = 0
        while start < len(order):
            chunk = [examples[int(i)] for i in order[start:start + batch]]
            ids = collate([pair(q, c) for q, c, _ in chunk], pad, device)
            labels = torch.tensor([y for *_, y in chunk], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = F.binary_cross_entropy_with_logits(forward_logits(model, head, ids, pad), labels)
                if not torch.isfinite(loss):
                    raise SystemExit("[rerank] non-finite loss")
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                print(f"[rerank] OOM, batch now {batch}", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            warm = max(1, int(0.05 * steps))
            scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
            for group in opt.param_groups:
                group["lr"] = 5e-4 * scale
            opt.step()
            if step % 20 == 0 or step + 1 == steps:
                gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0
                print(f"[rerank] epoch {epoch} step {step + 1}/{steps} loss {float(loss):.3f} batch {batch} peak {gb:.1f} GB", flush=True)
            step += 1
            start += len(chunk)

    torch.save({"overlay": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                "head": head.state_dict()}, os.path.join(OUT, "rerank.pt"))
    model.eval()
    head.eval()

    def score_split(queries: dict, lists: dict, bm: dict) -> dict[str, dict[str, float]]:
        jobs = [(qid, cid) for qid in queries for cid in lists[qid] if cid in text]
        flat = np.empty(len(jobs), dtype=np.float32)
        done = 0
        eval_batch = EVAL_BATCH
        while done < len(jobs):
            width = min(eval_batch, len(jobs) - done)
            chunk = jobs[done:done + width]
            ids = collate([pair(q, c) for q, c in chunk], pad, device)
            try:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    flat[done:done + width] = forward_logits(model, head, ids, pad).float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if device == "cuda":
                    torch.cuda.empty_cache()
                eval_batch = max(1, eval_batch // 2)
                print(f"[rerank] eval OOM, batch now {eval_batch}", flush=True)
                if width == 1:
                    raise
                continue
            done += width
            if done % (eval_batch * 20) == 0 or done == len(jobs):
                print(f"[rerank] scored pairs {done}/{len(jobs)}", flush=True)
        out = {qid: {} for qid in queries}
        for (qid, cid), logit in zip(jobs, flat):
            out[qid][cid] = (bm[qid].get(cid, 0.0), float(logit))
        return out

    def rank_of(scored: dict[str, dict[str, float]], lam: float) -> dict[str, list[str]]:
        pred = {}
        for qid, rows in scored.items():
            bms = np.array([b for b, _ in rows.values()])
            ces = np.array([c for _, c in rows.values()])
            bz = (bms - bms.mean()) / (bms.std() + 1e-8)
            cz = (ces - ces.mean()) / (ces.std() + 1e-8)
            mix = bz + lam * cz
            order = np.argsort(-mix)
            pred[qid] = [list(rows.keys())[i] for i in order[:15]]
        return pred

    dev_gold = {qid: [c for c in row.get("relevant_candidates") or [] if c] for qid, row in dev.items()}
    print("[rerank] scoring dev", flush=True)
    dev_scored = score_split(dev, dev_list, dev_bm)
    part_a, part_b = halves(list(dev))
    table = {}
    best_lam, best_a = 1.0, -1.0
    for lam in (0.5, 1.0, 2.0, 4.0):
        pred = {q: ids[:K] for q, ids in rank_of(dev_scored, lam).items()}
        fa = micro_f1(dev_gold, pred, sorted(part_a))["f1"]
        fb = micro_f1(dev_gold, pred, sorted(part_b))["f1"]
        table[str(lam)] = {"A": fa, "B": fb}
        print(f"[rerank] lambda {lam} A {fa:.4f} B {fb:.4f}", flush=True)
        if fa > best_a:
            best_lam, best_a = lam, fa
    kept = best_lam if table[str(best_lam)]["B"] >= 0.50 else None
    print(f"[rerank] chosen lambda {best_lam} kept {kept}", flush=True)

    print("[rerank] scoring test", flush=True)
    test_scored = score_split(test, test_list, test_bm)
    use = kept if kept is not None else best_lam
    ranked = rank_of(test_scored, use)
    ce_only = rank_of(test_scored, 1e6)
    lower = rank_of(test_scored, {0.5: 0.5, 1.0: 0.5, 2.0: 1.0, 4.0: 2.0}[use])

    def write(name: str, pred: dict[str, list[str]]) -> None:
        import zipfile
        dest = os.path.join(SUB, name)
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, "prediction.json")
        payload = {q: ids[:K] for q, ids in pred.items()}
        json.dump(payload, open(path, "w"))
        with zipfile.ZipFile(os.path.join(dest, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(path, arcname="prediction.json")
        print(f"[rerank] {name}", flush=True)

    write("P7_rerank", ranked)
    write("P7_rerank_ce", ce_only)
    write("P7_rerank_lam", lower)
    swap = {q: ids[:6] + [ids[7]] for q, ids in ranked.items()}
    write("P7_rerank_swap7", swap)
    json.dump({"lambda": use, "kept": kept, "dev": table}, open(os.path.join(OUT, "rerank.json"), "w"), indent=2)
    print("[rerank] done", flush=True)


if __name__ == "__main__":
    main()
