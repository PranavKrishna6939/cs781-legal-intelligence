"""SwiGLU (clamped) and ReLU^2 feed-forwards. Output projection is zero-initialised."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, clamp: float = 10.0):
        super().__init__()
        self.clamp = clamp
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        if self.clamp:
            g = g.clamp(-self.clamp, self.clamp)
            u = u.clamp(-self.clamp, self.clamp)
        return self.down(F.silu(g) * u)


class ReLU2(nn.Module):
    def __init__(self, d_model: int, d_ff: int, clamp: float = 10.0):
        super().__init__()
        self.clamp = clamp
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x).clamp(-self.clamp, self.clamp)
        return self.down(h * h)
