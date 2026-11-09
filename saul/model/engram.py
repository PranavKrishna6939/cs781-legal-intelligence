"""Hashed n-gram memory (Engram, scaled to ~tens of millions of parameters).

Token ids go through a compression map (case / leading-space folding, a persistent buffer set from the
tokenizer by `saul.tokenizer.compress`). Bigrams and trigrams that end at each position are hashed with
multi-head polynomial hashes into prime-sized tables. N-grams never reach across a document boundary: the
missing context is a reserved sentinel. The module returns a gated *delta* that the caller adds to the residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.model.norm import ZeroCenteredRMSNorm

_MULT = 1_000_003


def _primes(n: int, start: int) -> list[int]:
    out = []
    p = start | 1
    while len(out) < n:
        if all(p % d for d in range(3, int(p ** 0.5) + 1, 2)):
            out.append(p)
        p += 2
    return out


class Engram(nn.Module):
    def __init__(self, n_tables_per_order: int = 8, head_dim: int = 32, orders: tuple[int, ...] = (2, 3),
                 table_size: int = 100003, d_model: int = 1024, vocab_size: int = 32768):
        super().__init__()
        self.orders = tuple(orders)
        self.max_order = max(self.orders)
        self.n_heads = n_tables_per_order
        self.head_dim = head_dim
        self.tables = nn.ModuleDict()
        self.primes: dict[str, list[int]] = {}
        retrieved = 0
        for order in self.orders:
            primes = _primes(n_tables_per_order, table_size + 17 * order)
            self.primes[str(order)] = primes
            for h, p in enumerate(primes):
                self.tables[f"{order}_{h}"] = nn.Embedding(p, head_dim)
            retrieved += n_tables_per_order * head_dim
        self.retrieved = retrieved
        self.W_k = nn.Linear(retrieved, d_model, bias=False)
        self.W_v = nn.Linear(retrieved, d_model, bias=False)
        self.norm_h = ZeroCenteredRMSNorm(d_model)
        self.norm_k = ZeroCenteredRMSNorm(d_model)
        nn.init.zeros_(self.W_v.weight)
        self.register_buffer("compress", torch.arange(vocab_size, dtype=torch.long), persistent=True)

    @torch.no_grad()
    def set_compression(self, mapping: torch.Tensor) -> None:
        if mapping.shape != self.compress.shape:
            raise ValueError(f"compression map has shape {tuple(mapping.shape)}, expected {tuple(self.compress.shape)}")
        self.compress.copy_(mapping.to(self.compress))

    def windows(self, input_ids: torch.Tensor, doc_ids: torch.Tensor | None) -> torch.Tensor:
        """[B, T, max_order] of (compressed id + 1); column j is the token j steps back, 0 = outside the document."""
        c = self.compress[input_ids] + 1
        T = c.shape[1]
        cols = [c]
        for s in range(1, self.max_order):
            cs = F.pad(c, (s, 0))[:, :T]
            if doc_ids is not None:
                ds = F.pad(doc_ids, (s, 0), value=-(2 ** 30))[:, :T]
                cs = cs * (ds == doc_ids)
            cols.append(cs)
        return torch.stack(cols, dim=-1)

    def forward(self, h: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
        """h: [B, T, d] query stream, win: [B, T, max_order] from `windows`. Returns the delta to add to h."""
        parts = []
        for order in self.orders:
            acc = torch.zeros_like(win[..., 0])
            for j in range(order):
                acc = acc * _MULT + win[..., j]
            for head, prime in enumerate(self.primes[str(order)]):
                parts.append(self.tables[f"{order}_{head}"](acc % prime))
        # Tables train at a high LR without weight decay, so their scale drifts upward; normalising each retrieved
        # head keeps the value added to the residual stream bounded (unnormalised lookups diverged at ~1K steps).
        e = torch.cat([F.rms_norm(p.float(), (p.shape[-1],)) for p in parts], dim=-1).to(h.dtype)
        k = self.norm_k(self.W_k(e))
        hn = self.norm_h(h)
        alpha = torch.sigmoid((hn * k).sum(-1, keepdim=True) / (h.shape[-1] ** 0.5))
        return alpha * self.W_v(e)

    def serve_spec(self) -> dict:
        return {"hist": ("slot_fixed", (self.max_order - 1,), torch.long)}

    def serve_export_hist(self, input_ids: torch.Tensor, doc_ids: torch.Tensor | None) -> torch.Tensor:
        """[T, max_order-1] history after each packed token (column layout of `windows`)."""
        return self.windows(input_ids, doc_ids)[0, :, :self.max_order - 1]

    def serve_decode(self, h: torch.Tensor, ids: torch.Tensor, hist: torch.Tensor):
        """h: [Sb, 1, d], ids: [Sb], hist: [Sb, max_order-1] -> (delta [Sb, 1, d], new_hist)."""
        current = self.compress[ids] + 1
        win = torch.cat([current[:, None], hist], dim=-1)
        delta = self.forward(h, win[:, None, :])
        return delta, win[:, :self.max_order - 1]
