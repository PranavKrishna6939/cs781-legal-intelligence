"""Label-query decoder, retrieval memory, and the distribution-balanced loss for SAUL-LSI.

The decoder is the Query2Label / ML-Decoder head: one learned query per statute, label
self-attention, then cross-attention over every token of every segment. Retrieval slots and the
lexical prior are optional inputs, so the ablations turn them off without a second head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.lsi.const import N_LABELS


def cooc_bias(conditional: torch.Tensor) -> torch.Tensor:
    """Row-centred log P(j | i). A constant shift per row does not change the softmax."""
    logged = torch.log(conditional.clamp(min=0) + 1e-3)
    return logged - logged.mean(dim=1, keepdim=True)


def db_ntr_loss(logits: torch.Tensor, labels: torch.Tensor, freq: torch.Tensor,
                alpha: float = 0.1, beta: float = 10.0, mu: float = 0.9,
                lam: float = 2.0, kappa: float = 0.05) -> torch.Tensor:
    """Distribution-balanced loss with negative-tolerant regularisation (Huang et al., EMNLP 2021).

    With alpha=0.5, beta=0, lam=1 and kappa=0 this is exactly the mean binary cross-entropy.
    """
    counts = freq.to(device=logits.device, dtype=logits.dtype).clamp(min=1)
    inv = 1.0 / counts
    pos_mass = (labels * inv.view(1, -1)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
    reweight = inv.view(1, -1) / pos_mass
    weight = alpha + torch.sigmoid(beta * (reweight - mu))
    prior = (counts / counts.sum()).clamp(1e-4, 1 - 1e-4)
    shift = kappa * torch.log(prior / (1 - prior))
    adjusted = logits - shift.view(1, -1)
    positive = F.softplus(-adjusted)
    negative = F.softplus(lam * adjusted) / lam
    per_label = weight * (labels * positive + (1 - labels) * negative)
    return per_label.mean()


class _DecoderLayer(nn.Module):
    def __init__(self, width: int, n_heads: int, dropout: float):
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width {width} is not divisible by {n_heads} heads")
        self.heads = n_heads
        self.dh = width // n_heads
        self.ln_self = nn.LayerNorm(width)
        self.wq = nn.Linear(width, width)
        self.wk = nn.Linear(width, width)
        self.wv = nn.Linear(width, width)
        self.wo = nn.Linear(width, width)
        self.gamma = nn.Parameter(torch.zeros(n_heads))
        self.ln_cross = nn.LayerNorm(width)
        self.cq = nn.Linear(width, width)
        self.ck = nn.Linear(width, width)
        self.cv = nn.Linear(width, width)
        self.co = nn.Linear(width, width)
        self.ln_ff = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)
        self.drop = nn.Dropout(dropout)

    def _split(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return proj(x).view(batch, length, self.heads, self.dh).transpose(1, 2)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, mem_mask: torch.Tensor,
                bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.ln_self(queries)
        query = self._split(self.wq, hidden)
        key = self._split(self.wk, hidden)
        value = self._split(self.wv, hidden)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dh)
        if bias is not None:
            scores = scores + self.gamma.view(1, -1, 1, 1) * bias.view(1, 1, *bias.shape)
        attn = self.drop(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(attn, value).transpose(1, 2).contiguous().view(queries.shape[0], queries.shape[1], -1)
        queries = queries + self.drop(self.wo(mixed))

        hidden = self.ln_cross(queries)
        query = self._split(self.cq, hidden)
        key = self._split(self.ck, memory)
        value = self._split(self.cv, memory)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dh)
        scores = scores.masked_fill(~mem_mask[:, None, None, :], torch.finfo(scores.dtype).min / 4)
        cross = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(self.drop(cross), value).transpose(1, 2).contiguous().view(queries.shape[0], queries.shape[1], -1)
        queries = queries + self.drop(self.co(mixed))

        hidden = self.ln_ff(queries)
        queries = queries + self.drop(self.fc2(self.drop(F.gelu(self.fc1(hidden)))))
        return queries, cross


class LabelDecoderHead(nn.Module):
    """Two-layer label decoder. Lexical logits and neighbour slots are read from the batch."""

    def __init__(self, d_model: int, n_labels: int = N_LABELS, width: int = 512, n_heads: int = 8,
                 n_layers: int = 2, n_slots: int = 8, prior: torch.Tensor | None = None,
                 cooc: torch.Tensor | None = None, dropout: float = 0.1):
        super().__init__()
        self.n_labels = n_labels
        self.n_slots = n_slots
        self.wide_drop = 0.25
        self.slot_drop = 0.2
        self.mem_drop = 0.1
        self.proj = nn.Linear(d_model, width)
        self.tok_ln = nn.LayerNorm(width)
        self.seg_emb = nn.Embedding(64, d_model)
        nn.init.zeros_(self.seg_emb.weight)
        self.type_emb = nn.Parameter(torch.randn(2, width) * 0.02)
        self.label_emb = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.lex_dir = nn.Parameter(torch.randn(width) * 0.02)
        self.layers = nn.ModuleList(_DecoderLayer(width, n_heads, dropout) for _ in range(n_layers))
        self.out_w = nn.Parameter(torch.empty(n_labels, width))
        nn.init.xavier_uniform_(self.out_w)
        bias = torch.zeros(n_labels)
        if prior is not None:
            prior = prior.clamp(1e-4, 1 - 1e-4)
            bias = torch.log(prior / (1 - prior))
        self.out_bias = nn.Parameter(bias)
        self.gate = nn.Parameter(torch.full((n_labels,), 0.5))
        self.gate_bias = nn.Parameter(torch.zeros(n_labels))
        self.mem_emb = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.sim_dir = nn.Parameter(torch.randn(width) * 0.02)
        self.rank_emb = nn.Parameter(torch.randn(max(n_slots, 1), width) * 0.02)
        self.mem_ln = nn.LayerNorm(width)
        if cooc is not None:
            self.register_buffer("cooc", cooc_bias(cooc.float()))
        else:
            self.cooc = None

    def _slots(self, labels: torch.Tensor, sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(labels.shape[1], self.rank_emb.shape[0])
        labels = labels[:, :k].to(dtype=self.mem_emb.dtype)
        sim = sim[:, :k].to(dtype=self.mem_emb.dtype)
        count = labels.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mixture = torch.einsum("bkc,cw->bkw", labels / count, self.mem_emb)
        slots = mixture + sim.unsqueeze(-1) * self.sim_dir + self.rank_emb[:k].unsqueeze(0)
        slots = self.mem_ln(slots) + self.type_emb[1]
        mask = torch.ones(labels.shape[:2], dtype=torch.bool, device=labels.device)
        if self.training:
            drop_all = torch.rand(labels.shape[0], 1, device=labels.device) < self.mem_drop
            drop_one = torch.rand(labels.shape[0], k, device=labels.device) < self.slot_drop
            mask = ~(drop_all | drop_one)
        return slots, mask

    def forward(self, states: torch.Tensor, mask: torch.Tensor, batch: dict | None = None,
                return_extra: bool = False) -> dict:
        batch = batch or {}
        tokens = self.tok_ln(self.proj(states)) + self.type_emb[0]
        pieces = [tokens]
        masks = [mask]
        if self.n_slots and batch.get("nbr_labels") is not None:
            slots, slot_mask = self._slots(batch["nbr_labels"], batch["nbr_sim"])
            pieces.append(slots)
            masks.append(slot_mask)
        memory = torch.cat(pieces, dim=1)
        mem_mask = torch.cat(masks, dim=1)
        zlex = batch.get("zlex")
        if zlex is None:
            zlex = states.new_zeros(states.shape[0], self.n_labels)
        else:
            zlex = zlex.to(device=states.device, dtype=tokens.dtype)
            if self.training and self.wide_drop > 0:
                keep = (torch.rand(states.shape[0], 1, device=states.device) >= self.wide_drop).to(zlex.dtype)
                zlex = zlex * keep
        queries = self.label_emb.unsqueeze(0) + self.lex_dir.view(1, 1, -1) * zlex.unsqueeze(-1)
        cross = None
        for layer in self.layers:
            queries, cross = layer(queries, memory, mem_mask, self.cooc)
        z_deep = (queries * self.out_w.unsqueeze(0)).sum(-1) + self.out_bias
        fused = z_deep + self.gate * zlex + self.gate_bias
        out = {"logits": fused, "logits_deep": z_deep, "label_states": queries}
        if return_extra:
            out["alpha"] = cross
            out["mem_mask"] = mem_mask
        return out
