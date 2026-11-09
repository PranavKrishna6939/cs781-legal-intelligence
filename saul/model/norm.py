"""Zero-centred RMSNorm (Qwen3.8) and per-head QK norm."""
from __future__ import annotations

import torch
import torch.nn as nn


class ZeroCenteredRMSNorm(nn.Module):
    """y = x / rms(x) * (1 + gamma). gamma is initialised at 0 so the layer starts as plain RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * (1.0 + self.gamma.float())).to(dtype=x.dtype)


class QKNorm(nn.Module):
    """Per-head RMSNorm. x: [B, T, H, D]."""

    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(n_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * (1.0 + self.gamma.float())).to(dtype=x.dtype)
