"""Masked next-token pretraining on normalised CJPE text (train + all of dev, or train only with --clean)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from saul.cjpe import p4_dir
from saul.cjpe.data import load_split
from saul.cjpe.model import enable_activation_checkpointing
from saul.data.loader import mntp_mask
from saul.ner.registry import resolve_checkpoint
from saul.utils.io import ensure_dir


def _corpus(clean: bool = False) -> np.ndarray:
    """All pool and D-sel text, or with `clean` only the train rows of the pool."""
    if clean:
        pool = load_split("pool")
        keep = [i for i, source in enumerate(pool.sources) if source == "train"]
        return np.concatenate([np.asarray(pool.tokens_of(i), dtype=np.int64) for i in keep])
    pieces = []
    for name in ("pool", "dsel"):
        split = load_split(name)
        pieces.append(np.asarray(split.tokens, dtype=np.int64))
    return np.concatenate(pieces)


def run_tapt(seq_len: int = 2048, lr: float = 1e-4, batch_size: int = 4, clean: bool = False) -> str:
    from saul.model.modeling_saul import SaulForCausalLM

    out = os.path.join(p4_dir(), "adapt", "dense_v1_clean" if clean else "dense_v1")
    marker = os.path.join(out, "done.json")
    if os.path.exists(marker):
        print(f"[tapt] reuse {out}", flush=True)
        return out
    ensure_dir(out)
    pool = _corpus(clean)
    if pool.shape[0] < seq_len:
        raise RuntimeError("CJPE token cache is shorter than one TAPT sequence")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint("dense_v1")).to(device)
    enable_activation_checkpointing(model, every=2)
    model.train()
    seen: set[int] = set()
    params = []
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n_seq = pool.shape[0] // seq_len
    steps = max(1, n_seq // batch_size)
    warm = max(1, int(steps * 0.05))
    rng = np.random.default_rng(781)
    cursor = 0
    order = rng.permutation(n_seq)
    use_cuda = device == "cuda"
    for step in range(steps):
        if cursor + batch_size > len(order):
            order = rng.permutation(n_seq)
            cursor = 0
        picks = order[cursor:cursor + batch_size]
        cursor += batch_size
        batch = np.stack([pool[int(i) * seq_len:(int(i) + 1) * seq_len] for i in picks])
        ids = torch.tensor(batch, dtype=torch.long)
        docs = torch.zeros_like(ids)
        masked, labels, loss_mask = mntp_mask(
            ids, docs, int(model.config.pad_token_id), int(model.config.mask_token_id),
            int(model.config.vocab_size), rng=rng, random_low=64,
        )
        masked, labels, loss_mask, docs = (t.to(device) for t in (masked, labels, loss_mask, docs))
        scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
        for group in opt.param_groups:
            group["lr"] = lr * scale
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            loss = model(masked, labels=labels, doc_ids=docs, loss_mask=loss_mask, mode="bidirectional").loss
        if not torch.isfinite(loss):
            raise SystemExit("[tapt] non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"[tapt] step {step}/{steps} loss {float(loss):.3f}", flush=True)
    model.save_pretrained(out)
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"steps": steps, "seq_len": seq_len, "tokens": int(steps * batch_size * seq_len), "clean": clean}, handle)
    print(f"[tapt] saved {out}", flush=True)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    run_tapt(clean=parser.parse_args().clean)


if __name__ == "__main__":
    main()
