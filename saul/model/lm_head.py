"""Tied LM head helpers: logit soft-cap, next-token targets, chunked fp32 cross-entropy without host syncs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    if not cap:
        return logits
    return cap * torch.tanh(logits / cap)


def next_token_targets(labels: torch.Tensor, doc_ids: torch.Tensor | None = None,
                       loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Targets aligned with the logits (no shifted copy of the logits is needed).

    tgt[:, t] = labels[:, t+1], ignored (-100) when
      * t is the last position,
      * t and t+1 are in different segments (no prediction across a document boundary),
      * `loss_mask` is given (MNTP) and the input at t+1 was not masked: the model predicts the masked token
        t+1 from position t, as in GPT-BERT.
    """
    tgt = torch.full_like(labels, -100)
    nxt = labels[:, 1:]
    keep = torch.ones_like(nxt, dtype=torch.bool)
    if doc_ids is not None:
        keep &= doc_ids[:, 1:] == doc_ids[:, :-1]
    if loss_mask is not None:
        keep &= loss_mask[:, 1:].to(torch.bool)
    tgt[:, :-1] = torch.where(keep, nxt, torch.full_like(nxt, -100))
    return tgt


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100,
                          chunk: int = 4096, reduction: str = "mean") -> torch.Tensor:
    """logits [B, T, V] (any dtype), targets [B, T]. Each chunk is upcast to fp32; no data-dependent control flow."""
    V = logits.shape[-1]
    flat_x = logits.reshape(-1, V)
    flat_y = targets.reshape(-1)
    total = flat_x.new_zeros((), dtype=torch.float32)
    for i in range(0, flat_x.shape[0], chunk):
        total = total + F.cross_entropy(flat_x[i:i + chunk].float(), flat_y[i:i + chunk],
                                        ignore_index=ignore_index, reduction="sum")
    if reduction == "sum":
        return total
    return total / (flat_y != ignore_index).sum().clamp(min=1)
