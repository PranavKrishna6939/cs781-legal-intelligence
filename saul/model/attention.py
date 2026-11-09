"""Gated grouped-query attention.

Layout is [B, T, heads, head_dim]. The backend is chosen by the type of mask object the model passes in:
a FlexAttention BlockMask, a dense bool tensor (SDPA), or `VarlenMeta` (FlashAttention-2 varlen). GQA is native
in all three (no repeated K/V copies). The sigmoid gate is applied to the attention output (Qwen G1), then the
zero-initialised output projection.

CED decoder layers attend over [local sliding-window keys ; global keys from the encoder memory] with one
softmax. Global keys get RoPE at their own positions, like local keys, so q.k is relative in both halves.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.model.masks import VarlenMeta, flex_fn
from saul.model.norm import QKNorm
from saul.model.rope import apply_partial_rope


@torch.compiler.disable
def _fa2(q, k, v, meta: VarlenMeta):
    from flash_attn import flash_attn_varlen_func

    B, T, H, D = q.shape
    window = (meta.window - 1, 0) if meta.window else (-1, -1)
    out = flash_attn_varlen_func(
        q.reshape(B * T, H, D), k.reshape(B * T, k.shape[2], D), v.reshape(B * T, v.shape[2], D),
        meta.cu_seqlens, meta.cu_seqlens, meta.max_seqlen, meta.max_seqlen,
        causal=meta.causal, window_size=window,
    )
    return out.view(B, T, H, D)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask) -> torch.Tensor:
    """q: [B, Tq, H, D], k/v: [B, Tk, KV, D], mask: BlockMask | bool tensor [B,1,Tq,Tk] | VarlenMeta."""
    if isinstance(mask, VarlenMeta):
        return _fa2(q, k, v, mask)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if torch.is_tensor(mask):
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, enable_gqa=True)
    else:
        out = flex_fn()(qt, kt, vt, block_mask=mask, enable_gqa=True)
    return out.transpose(1, 2)


class GatedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, rope_dims: int, window: int = 0,
                 gate: bool = True, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_dims = rope_dims
        self.window = window
        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.g = nn.Linear(d_model, n_heads * head_dim, bias=False) if gate else None
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = QKNorm(n_heads, head_dim) if qk_norm else nn.Identity()
        self.k_norm = QKNorm(n_kv_heads, head_dim) if qk_norm else nn.Identity()
        nn.init.zeros_(self.o.weight)

    def _kv(self, x, cos, sin, pos):
        B, T, _ = x.shape
        k = self.k_norm(self.k(x).view(B, T, self.n_kv_heads, self.head_dim))
        k = apply_partial_rope(k, cos, sin, pos, self.rope_dims)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim)
        return k, v

    def forward(self, x, cos, sin, pos, mask=None, memory=None, cache: dict | None = None,
                phase: str = "train", step_ctx=None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.n_heads, self.head_dim))
        q = apply_partial_rope(q, cos, sin, pos, self.rope_dims)
        k, v = self._kv(x, cos, sin, pos)
        kg = vg = None
        if memory is not None:
            kg, vg = self._kv(memory, cos, sin, pos)
        if phase == "step":
            out = self._step(q, k, v, kg, vg, cache, step_ctx)
        elif getattr(self, "_flash_causal", False) and x.is_cuda and memory is None:
            out = _flash_causal(q, k, v)
            self._export_kv = (k, v)
        else:
            if memory is not None:
                out = attend(q, torch.cat([k, kg], 1), torch.cat([v, vg], 1), mask)
            else:
                out = attend(q, k, v, mask)
            if phase == "prefill":
                self._fill(cache, step_ctx, k, v, kg, vg)
        if self.g is not None:
            out = out * torch.sigmoid(self.g(x)).view(B, T, self.n_heads, self.head_dim)
        return self.o(out.flatten(-2))

    @staticmethod
    def _buffer(cache: dict, name: str, like: torch.Tensor, max_len: int) -> torch.Tensor:
        if name not in cache:
            cache[name] = like.new_zeros(like.shape[0], max_len, *like.shape[2:])
        return cache[name]

    def _fill(self, cache, ctx, k, v, kg, vg):
        T = k.shape[1]
        for name, t in (("k", k), ("v", v), ("kg", kg), ("vg", vg)):
            if t is not None:
                self._buffer(cache, name, t, ctx.max_len)[:, :T] = t

    def _step(self, q, k, v, kg, vg, cache, ctx):
        """Decode one token. ctx.length = index being written; ctx.key_valid already marks it valid."""
        L = ctx.length
        cache["k"][:, L] = k[:, 0]
        cache["v"][:, L] = v[:, 0]
        valid = ctx.key_valid[:, :L + 1]
        local = valid
        if self.window:
            local = valid & (torch.arange(L + 1, device=q.device) > L - self.window)[None]
        keys, vals, m = cache["k"][:, :L + 1], cache["v"][:, :L + 1], local
        if kg is not None:
            cache["kg"][:, L] = kg[:, 0]
            cache["vg"][:, L] = vg[:, 0]
            keys = torch.cat([keys, cache["kg"][:, :L + 1]], 1)
            vals = torch.cat([vals, cache["vg"][:, :L + 1]], 1)
            m = torch.cat([local, valid], 1)
        return attend(q, keys, vals, m[:, None, None, :])

    def serve_spec(self, ced_global: bool = False) -> dict:
        kv = (self.n_kv_heads, self.head_dim)
        spec = {"K": ("slot_ctx", kv, None), "V": ("slot_ctx", kv, None)}
        if ced_global:
            spec["Kg"] = ("slot_ctx", kv, None)
            spec["Vg"] = ("slot_ctx", kv, None)
        return spec

    def serve_export(self, x, cos, sin, pos, mask, memory):
        """Same output as `forward(..., phase='train')`, plus the K/V that produced it (packed row, batch 0)."""
        y = self.forward(x, cos, sin, pos, mask, memory, phase="train")
        k, v = self._kv(x, cos, sin, pos)
        exp = {"K": k, "V": v}
        if memory is not None:
            kg, vg = self._kv(memory, cos, sin, pos)
            exp["Kg"] = kg
            exp["Vg"] = vg
        return y, exp

    def serve_decode(self, x, cos, sin, pos, st, memory=None) -> torch.Tensor:
        """One token for `Sb` slots. `st` holds the context cache plus `kv_index` [Sb]."""
        sb = x.shape[0]
        q = self.q_norm(self.q(x).view(sb, 1, self.n_heads, self.head_dim))
        q = apply_partial_rope(q, cos, sin, pos, self.rope_dims)
        k, v = self._kv(x, cos, sin, pos)
        at = st["plen_slot"] + st["kv_index"]
        _cache_write(st["K"], st["V"], st["idx"], at, k[:, 0], v[:, 0], st["needs"])
        kg = vg = None
        if memory is not None:
            kg, vg = self._kv(memory, cos, sin, pos)
            _cache_write(st["Kg"], st["Vg"], st["idx"], at, kg[:, 0], vg[:, 0], st["needs"])
        out = _cached_attention(q, st, self.head_dim, self.window, memory is not None)
        if self.g is not None:
            out = out * torch.sigmoid(self.g(x)).view(sb, 1, self.n_heads, self.head_dim)
        return self.o(out.flatten(-2))


def _cache_write(k_cache, v_cache, idx, at, k, v, needs) -> None:
    """Write the new token at absolute position `plen + generated_index`. Inactive slots keep the old value."""
    keep = needs.view(-1, 1, 1)
    k_cache.index_put_((idx, at), torch.where(keep, k, k_cache[idx, at]))
    v_cache.index_put_((idx, at), torch.where(keep, v, v_cache[idx, at]))


def _seqused(st) -> torch.Tensor:
    # plen + tokens generated so far, at least 1 so flash/SDPA always sees the current key.
    return (st["plen_slot"] + st["kv_index"] + 1).clamp(min=1, max=st["K"].shape[1])


def _cached_attention(q, st, head_dim: int, window: int, ced_global: bool) -> torch.Tensor:
    """q: [Sb, 1, H, D]. CUDA uses flash attention with per-row lengths (no dense mask)."""
    seqused = _seqused(st)
    if q.is_cuda and not ced_global:
        return _flash_decode(q, st["K"], st["V"], seqused, window)
    return _sdpa_decode(q, st, seqused, window, ced_global)


def _flash_causal(q, k, v) -> torch.Tensor:
    """Right-padded prefill. Causal flash: a real token never sees padding to its right."""
    out = torch.ops.aten._flash_attention_forward(
        q, k, v, None, None, q.shape[1], k.shape[1], 0.0, True, False, scale=q.shape[-1] ** -0.5,
    )[0]
    return out


def _flash_decode(q, k, v, seqused, window: int) -> torch.Tensor:
    """FlashAttention varlen. q is [B, 1, H, D]; k/v are [B, L, KV, D]."""
    from saul.model.flash_decode import flash_varlen_decode
    return flash_varlen_decode(q, k, v, seqused, window)


def _sdpa_decode(q, st, seqused, window: int, ced_global: bool) -> torch.Tensor:
    """CPU / CED path. Bool mask True means keep, same as `attend`."""
    parts_k = [st["K"]]
    parts_v = [st["V"]]
    masks = [_length_mask(st["K"], seqused, window)]
    if ced_global:
        parts_k.append(st["Kg"])
        parts_v.append(st["Vg"])
        masks.append(_length_mask(st["Kg"], seqused, 0))
    k = torch.cat(parts_k, dim=1).transpose(1, 2)
    v = torch.cat(parts_v, dim=1).transpose(1, 2)
    mask = torch.cat(masks, dim=-1)
    qh = q.transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, k, v, attn_mask=mask, enable_gqa=q.shape[2] != k.shape[1])
    return out.transpose(1, 2)


def _length_mask(cache, seqused, window: int) -> torch.Tensor:
    """[B, 1, 1, L] keep mask. Cache index equals absolute position."""
    length = cache.shape[1]
    ar = torch.arange(length, device=cache.device)
    keep = ar.view(1, 1, 1, -1) < seqused.view(-1, 1, 1, 1)
    if window:
        dist = (seqused.view(-1, 1, 1, 1) - 1) - ar.view(1, 1, 1, -1)
        keep = keep & (dist >= 0) & (dist < window)
    return keep
