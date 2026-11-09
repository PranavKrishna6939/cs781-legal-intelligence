"""Short bidirectional MNTP adaptation before NER fine-tuning.

The Phase-1 trainer only enables MNTP for dense models. This loop calls the same
mask and the model's bidirectional mode directly, so it also covers the CED encoder
and the hybrid's attention layers (its Gated DeltaNet layers stay causal).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from saul.data.loader import mntp_mask
from saul.data.tokens import open_source
from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.data import load_split
from saul.ner.registry import p2_dir, repo_root, resolve_checkpoint


def _rows(pool: np.ndarray, batch: int, length: int, rng: np.random.Generator) -> torch.Tensor:
    n = pool.shape[0]
    if n <= length:
        reps = int(np.ceil(length / max(n, 1)))
        pool = np.tile(pool, reps)
        n = pool.shape[0]
    starts = rng.integers(0, n - length, size=batch)
    return torch.tensor(np.stack([pool[s:s + length] for s in starts]), dtype=torch.long)


def adapt(name: str, minutes: float = 20.0, token_budget: int = 20_000_000, seq_len: int = 512,
          batch_size: int = 4, lr: float = 1e-5, device: str | None = None) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(p2_dir(), "adapt", name)
    marker = os.path.join(out, "done.json")
    if os.path.exists(marker):
        print(f"[skip] adaptation {name}")
        return out
    os.makedirs(out, exist_ok=True)
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint(name)).to(device)
    model.train()
    if model.config.arch == "ced":
        model._ner_stop_after = int(model.config.n_encoder_layers)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    rng = np.random.default_rng(781)
    ner_ids = []
    for doc in load_split("train"):
        ner_ids.extend(doc.ids)
    pieces = [np.asarray(ner_ids, dtype=np.int64)]
    token_root = os.path.join(repo_root(), "data", "tokens", "legal32k")
    try:
        sc = open_source(token_root, "train", "supreme_court")
        take = min(sc.n_tokens, token_budget)
        pieces.append(np.asarray(sc.tokens[:take], dtype=np.int64))
    except FileNotFoundError:
        print("[adapt] supreme_court tokens not found; using the NER documents only")
    pool = np.concatenate(pieces)
    seen = 0
    deadline = time.time() + minutes * 60
    step = 0
    use_cuda = device == "cuda"
    while seen < token_budget and time.time() < deadline:
        ids = _rows(pool, batch_size, seq_len, rng)
        docs = torch.zeros_like(ids)
        masked, labels, loss_mask = mntp_mask(
            ids, docs, int(model.config.pad_token_id), int(model.config.mask_token_id),
            int(model.config.vocab_size), rng=rng, random_low=64,
        )
        masked = masked.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)
        docs = docs.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            out_m = model(masked, labels=labels, doc_ids=docs, loss_mask=loss_mask, mode="bidirectional")
            loss = out_m.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += int(ids.numel())
        step += 1
        if step % 20 == 0:
            print(f"[adapt] {name} step {step} tokens {seen} loss {float(loss):.3f}", flush=True)
    model._ner_stop_after = None
    model.save_pretrained(out)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f'{{"tokens": {seen}, "steps": {step}}}\n')
    print(f"[adapt] saved {out} tokens={seen}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--tokens", type=int, default=20_000_000)
    args = p.parse_args()
    adapt(args.model, minutes=args.minutes, token_budget=args.tokens)


if __name__ == "__main__":
    main()
