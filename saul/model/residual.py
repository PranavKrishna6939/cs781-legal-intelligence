"""Residual stream variants: pre-norm, static widened (AltUp), Qwen Gated Residual, DeepSeek mHC.

Every variant exposes `read(R) -> (x, ctx)` and `write(R, y, ctx) -> R`. `ctx` carries whatever the write needs
from the read (gate scales, mixing matrices), so modules hold no per-call state and compile cleanly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.model.norm import ZeroCenteredRMSNorm


class PreNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = ZeroCenteredRMSNorm(d_model)

    def read(self, h: torch.Tensor):
        return self.norm(h), None

    def write(self, h: torch.Tensor, y: torch.Tensor, ctx=None) -> torch.Tensor:
        return h + y


class StaticWidenedResidual(nn.Module):
    """Parallel streams. Read is a fixed round-robin one-hot; write adds y to every stream."""

    def __init__(self, d_model: int, n: int = 4, layer_idx: int = 0):
        super().__init__()
        self.n = n
        w = torch.zeros(n)
        w[layer_idx % n] = 1.0
        self.read_w = nn.Parameter(w)
        self.write_w = nn.Parameter(torch.ones(n))

    def read(self, R: torch.Tensor):
        w = self.read_w / self.read_w.sum().clamp(min=1e-6)
        return torch.einsum("n,btnd->btd", w, R), None

    def write(self, R: torch.Tensor, y: torch.Tensor, ctx=None) -> torch.Tensor:
        return R + y.unsqueeze(-2) * self.write_w.view(1, 1, -1, 1)


class GatedResidual(nn.Module):
    """Qwen3.8 Gated Residual. R: [B, T, n, d]. Replaces the block pre-norm."""

    def __init__(self, d_model: int, n: int = 4, rank: int = 64, eps: float = 1e-6):
        super().__init__()
        self.n = n
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(n, d_model))
        self.W_d = nn.Linear(n * d_model, rank, bias=False)
        self.W_u = nn.Linear(rank, n * d_model, bias=False)
        self.W_w = nn.Linear(n * d_model, n, bias=False)
        nn.init.zeros_(self.W_w.weight)

    def read(self, R: torch.Tensor):
        Rf = R.float()
        Rh = (Rf * Rf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * (1.0 + self.gamma.float())).to(R.dtype)
        flat = Rh.flatten(-2)
        gates = torch.sigmoid(self.W_u(F.silu(self.W_d(flat) / self.n))).view_as(Rh)
        s = 2 * torch.sigmoid(self.W_w(flat) / self.n)
        return (gates * Rh).mean(dim=2), s

    def write(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return R + s.unsqueeze(-1).to(y.dtype) * y.unsqueeze(-2)


def sinkhorn(logits: torch.Tensor, iters: int = 10) -> torch.Tensor:
    M = logits.float().softmax(dim=-1)
    for _ in range(iters):
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=1e-9)
    return M


class MHC(nn.Module):
    """Manifold-constrained hyper-connections, n residual streams.

    A (read, n), B (n x n, doubly stochastic), C (write, n) are predicted per token.
    """

    def __init__(self, d_model: int, n: int = 4, sinkhorn_iters: int = 10):
        super().__init__()
        self.n = n
        self.iters = sinkhorn_iters
        self.pred = nn.Linear(n * d_model, n + n * n + n, bias=False)
        nn.init.zeros_(self.pred.weight)

    def read(self, R: torch.Tensor):
        B, T, n, d = R.shape
        raw = self.pred(R.reshape(B, T, n * d))
        A = torch.softmax(raw[..., :n].float(), dim=-1).to(R.dtype)
        Bm = sinkhorn(raw[..., n:n + n * n].view(B, T, n, n), self.iters).to(R.dtype)
        C = (2 * torch.sigmoid(raw[..., n + n * n:].float())).to(R.dtype)
        return torch.einsum("btn,btnd->btd", A, R), (Bm, C)

    def write(self, R: torch.Tensor, y: torch.Tensor, ctx) -> torch.Tensor:
        Bm, C = ctx
        return torch.einsum("btij,btjd->btid", Bm, R) + C.unsqueeze(-1) * y.unsqueeze(-2)
