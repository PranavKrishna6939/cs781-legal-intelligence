"""Document-aware causal depthwise convolution and the double-gated short-conv block (LFM2 / Daedalus style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def doc_causal_conv(x: torch.Tensor, weight: torch.Tensor, doc_ids: torch.Tensor | None) -> torch.Tensor:
    """Depthwise causal conv that never mixes tokens from different documents.

    x: [B, T, C]; weight: [C, 1, K] (nn.Conv1d layout, tap K-1 is the current token); doc_ids: [B, T] or None.
    """
    K = weight.shape[-1]
    T = x.shape[1]
    w = weight[:, 0, :].to(x.dtype)
    y = x * w[:, K - 1]
    for s in range(1, K):
        xs = F.pad(x, (0, 0, s, 0))[:, :T]
        if doc_ids is not None:
            ds = F.pad(doc_ids, (s, 0), value=-(2 ** 30))[:, :T]
            xs = xs * (ds == doc_ids).unsqueeze(-1).to(x.dtype)
        y = y + xs * w[:, K - 1 - s]
    return y


def conv_tail(x: torch.Tensor, doc_ids: torch.Tensor | None, K: int) -> torch.Tensor:
    """Last K-1 inputs [B, K-1, C] for decoding, zeroed where they belong to a different document than the last token."""
    B, T, C = x.shape
    tail = x.new_zeros(B, K - 1, C)
    n = min(K - 1, T)
    if n == 0:
        return tail
    part = x[:, T - n:]
    if doc_ids is not None:
        keep = (doc_ids[:, T - n:] == doc_ids[:, -1:]).unsqueeze(-1).to(x.dtype)
        part = part * keep
    tail[:, K - 1 - n:] = part
    return tail


def conv_step(x: torch.Tensor, tail: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One decoding step. x: [B, 1, C], tail: [B, K-1, C] -> (y [B, 1, C], new tail)."""
    window = torch.cat([tail.to(x.dtype), x], dim=1)
    y = (window * weight[:, 0, :].t().to(x.dtype)[None]).sum(dim=1, keepdim=True)
    return y, window[:, 1:]


class ShortConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int = 3):
        super().__init__()
        self.kernel = kernel
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None, cache: dict | None = None,
                phase: str = "train") -> torch.Tensor:
        b, c, h = self.in_proj(x).chunk(3, dim=-1)
        u = b * h
        if phase == "step":
            y, cache["tail"] = conv_step(u, cache["tail"], self.conv.weight)
        else:
            y = doc_causal_conv(u, self.conv.weight, doc_ids)
            if phase == "prefill":
                cache["tail"] = conv_tail(u, doc_ids, self.kernel)
        return self.out_proj(c * y)

    def serve_spec(self) -> dict:
        return {"tail": ("slot_fixed", (self.kernel - 1, self.in_proj.in_features), None)}

    def serve_export(self, x, doc_ids):
        from saul.serve.packing import segment_spans

        b, c, h = self.in_proj(x).chunk(3, dim=-1)
        u = b * h
        y = doc_causal_conv(u, self.conv.weight, doc_ids)
        tails = []
        docs = doc_ids[0] if doc_ids is not None else torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
        for start, end, doc in segment_spans(docs):
            if doc < 0:
                continue
            tails.append(conv_tail(u[:, start:end], None, self.kernel)[0])
        exp = {"tail": torch.stack(tails) if tails else conv_tail(u[:, :0], None, self.kernel)}
        return self.out_proj(c * y), exp

    def serve_decode(self, x, st) -> torch.Tensor:
        b, c, h = self.in_proj(x).chunk(3, dim=-1)
        y, new_tail = conv_step(b * h, st["tail"], self.conv.weight)
        needs = st["needs"].view(x.shape[0], 1, 1)
        st["tail"].copy_(torch.where(needs, new_tail, st["tail"]))
        return self.out_proj(c * y)
