"""Partial rotary embeddings. Only the first `rope_dims` of each head are rotated."""
from __future__ import annotations

import torch


def rope_cache(seq_len: int, rope_dims: int, base: float, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    half = rope_dims // 2
    inv = base ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rope_dims: int,
) -> torch.Tensor:
    """x: [B, T, H, D]. cos/sin: [max_pos, half]. position_ids: [B, T], each < max_pos."""
    if rope_dims <= 0:
        return x
    half = rope_dims // 2
    c = cos[position_ids][:, :, None, :]
    s = sin[position_ids][:, :, None, :]
    x_rot, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    a, b = x_rot[..., :half].float(), x_rot[..., half:].float()
    out = torch.cat([a * c - b * s, a * s + b * c], dim=-1)
    return torch.cat([out.to(dtype=x.dtype), x_pass], dim=-1)
