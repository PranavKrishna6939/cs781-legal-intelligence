"""Held-out language-model metrics: mean next-token loss (nats) and bits per UTF-8 byte of the targets."""
from __future__ import annotations

import math

import numpy as np
import torch

from saul.data.loader import PackedLMDataset, collate_lm
from saul.model.lm_head import chunked_cross_entropy, next_token_targets


def fixed_val_batches(sources, weights, seq_len: int, batch_size: int, n_sequences: int, bos_id: int,
                      eos_id: int, pad_id: int, seed: int = 12345) -> list[dict]:
    """A deterministic list of packed validation batches (same every call, independent of workers)."""
    ds = PackedLMDataset(sources, weights, seq_len, bos_id, eos_id, pad_id, seed=seed, worker_override=(0, 1))
    it = iter(ds)
    batches = []
    for _ in range(max(1, n_sequences // batch_size)):
        b = collate_lm([next(it) for _ in range(batch_size)], pad_id=pad_id)
        b.pop("_wstate", None)
        batches.append(b)
    return batches


@torch.no_grad()
def evaluate_lm(model, batches: list[dict], token_bytes: np.ndarray | None, device, bf16: bool = True) -> dict:
    was_training = model.training
    model.eval()
    tb = None if token_bytes is None else torch.as_tensor(token_bytes, device=device, dtype=torch.float64)
    nats = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    nbytes = torch.zeros((), device=device, dtype=torch.float64)
    for b in batches:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and b["input_ids"].is_cuda):
            out = model(b["input_ids"], doc_ids=b["doc_ids"], position_ids=b["position_ids"],
                        cu_seqlens=b.get("cu_seqlens"))
        tgt = next_token_targets(b["labels"], b["doc_ids"])
        nats += chunked_cross_entropy(out.logits, tgt, reduction="sum").double()
        valid = tgt != -100
        count += valid.sum()
        if tb is not None:
            nbytes += tb[tgt.clamp(min=0)][valid].sum()
    if was_training:
        model.train()
    res = {"val_loss": float(nats / count.clamp(min=1))}
    if tb is not None and float(nbytes) > 0:
        res["val_bpb"] = float(nats / (math.log(2) * nbytes))
    return res
