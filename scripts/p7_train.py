"""Dense LoRMA bi-encoder for Phase 7. One epoch, first 512 tokens, InfoNCE."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.peft import LoRMAPlus, wrap_lorma
from saul.tokenizer.train import load_hf_tokenizer

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
DATA = os.path.join(ROOT, "data", "raw", "Phase_7_PCR")
OUT = os.path.join(ROOT, "p7")
CKPT = os.path.join(ROOT, "checkpoints", "dense_v1")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizers", "legal32k")
LIMIT = 512
TEMP = 0.05


def join_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def load_split(name: str):
    queries = json.load(open(os.path.join(DATA, name, f"{name.lower()}.json"), encoding="utf-8"))
    pool = json.load(open(os.path.join(DATA, name, f"{name.lower()}_candidates.json"), encoding="utf-8"))
    return queries, pool


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


def encode_ids(tok, text: str, bos: int, limit: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids or ids[0] != bos:
        ids = [bos] + ids
    return ids[:limit]


def mean_pool(model, ids: torch.Tensor, pad: int) -> torch.Tensor:
    hidden = model(ids, mode="bidirectional", return_hidden=True).hidden
    mask = ids.ne(pad).unsqueeze(-1)
    summed = (hidden * mask).sum(1)
    return summed / mask.sum(1).clamp(min=1)


def collate(rows: list[list[int]], pad: int, device: str) -> torch.Tensor:
    width = max(len(row) for row in rows)
    out = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out


def save_encode(model, tok, pool, queries, split: str, bos: int, pad: int, device: str) -> None:
    model.eval()
    cand_ids = list(pool.keys())
    qids = list(queries.keys())

    def embed(texts: list[str]) -> np.ndarray:
        rows = [encode_ids(tok, text, bos, LIMIT) for text in texts]
        vecs = []
        for start in range(0, len(rows), 16):
            batch = collate(rows[start:start + 16], pad, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                vecs.append(F.normalize(mean_pool(model, batch, pad).float(), dim=-1).cpu().numpy())
        return np.concatenate(vecs, axis=0)

    cand = embed([join_text(pool[i]["text"]) for i in cand_ids])
    qry = embed([join_text(queries[i]["text"]) for i in qids])
    np.savez(
        os.path.join(OUT, f"dense_{split.lower()}.npz"),
        qids=np.asarray(qids), cand_ids=np.asarray(cand_ids),
        query=qry.astype(np.float32), cand=cand.astype(np.float32),
    )
    print(f"[train] encoded {split} queries {len(qids)} cands {len(cand_ids)}", flush=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf_tokenizer(TOKENIZER)
    bos = int(tok.bos_token_id)
    pad = int(tok.pad_token_id)
    train_q, train_pool = load_split("Train")
    hard = json.load(open(os.path.join(OUT, "bm25_train_hard.json"), encoding="utf-8"))
    tokens = {}
    for cid, row in train_pool.items():
        tokens[cid] = encode_ids(tok, join_text(row["text"]), bos, LIMIT)
    for qid, row in train_q.items():
        tokens[qid] = encode_ids(tok, join_text(row["text"]), bos, LIMIT)

    pairs = []
    for qid, row in train_q.items():
        gold = [c for c in (row.get("relevant_candidates") or []) if c in train_pool]
        negs = [c for c in hard.get(qid, []) if c in train_pool]
        if gold and negs:
            pairs.append((qid, gold, negs))
    print(f"[train] pairs {len(pairs)}", flush=True)

    model = SaulForCausalLM.from_pretrained(CKPT)
    wrap_lorma(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRMAPlus):
            for param in list(module.A.parameters()) + list(module.B.parameters()):
                param.requires_grad_(True)
    checkpoint_every(model)
    model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable {sum(p.numel() for p in trainable)}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
    rng = np.random.default_rng(781)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    batch = 4
    steps = math.ceil(len(order) / batch)
    step = 0
    for start in range(0, len(order), batch):
        chunk = [pairs[int(i)] for i in order[start:start + batch]]
        q_rows, p_rows, n_rows = [], [], []
        for qid, gold, negs in chunk:
            q_rows.append(tokens[qid])
            p_rows.append(tokens[gold[int(rng.integers(len(gold)))]])
            n_rows.append(tokens[negs[int(rng.integers(len(negs)))]])
        ids = collate(q_rows + p_rows + n_rows, pad, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            vec = F.normalize(mean_pool(model, ids, pad), dim=-1)
            b = len(chunk)
            query, pos, neg = vec[:b], vec[b:2 * b], vec[2 * b:]
            keys = torch.cat([pos, neg], dim=0)
            logits = query @ keys.T / TEMP
            loss = F.cross_entropy(logits, torch.arange(b, device=device))
        if not torch.isfinite(loss):
            raise SystemExit("[train] non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        warm = max(1, int(0.05 * steps))
        scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
        for group in opt.param_groups:
            group["lr"] = 5e-4 * scale
        opt.step()
        step += 1
        if step % 20 == 0 or step == steps:
            print(f"[train] step {step}/{steps} loss {float(loss):.3f}", flush=True)
    tensors = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(tensors, os.path.join(OUT, "overlay.pt"))
    model.eval()
    for name in ("Dev", "Test"):
        queries, pool = load_split(name)
        save_encode(model, tok, pool, queries, name, bos, pad, device)
    json.dump({"steps": step, "pairs": len(pairs), "limit": LIMIT}, open(os.path.join(OUT, "train_done.json"), "w"), indent=2)
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
