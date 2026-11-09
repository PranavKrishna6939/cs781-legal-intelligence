"""Attention masks for packed documents. True means *keep*.

Mask kinds
  causal   causal and same document
  bidir    same document (MNTP batches)
  window   causal, same document, at most `window` tokens back
  ced      queries over [local keys ; global keys] (length 2T): the first half is `window`, the second `causal`

The model builds one mask object per kind per forward and hands it to every layer that needs it:
a FlexAttention BlockMask (`flex`), a dense bool tensor [B, 1, Tq, Tk] (`sdpa`), or varlen metadata (`fa2`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


def _same_doc(doc_q: torch.Tensor, doc_k: torch.Tensor) -> torch.Tensor:
    return doc_q[:, :, None] == doc_k[:, None, :]


def _with_self(mask: torch.Tensor) -> torch.Tensor:
    """A query with an all-False row makes softmax NaN. Always allow the diagonal."""
    tq, tk = mask.shape[-2], mask.shape[-1]
    eye = torch.eye(tq, tk, dtype=torch.bool, device=mask.device)
    return mask | eye.view(*([1] * (mask.ndim - 2)), tq, tk)


def document_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    T = doc_ids.shape[1]
    i = torch.arange(T, device=doc_ids.device)
    return _with_self(_same_doc(doc_ids, doc_ids) & (i[:, None] >= i[None, :])[None])


def bidirectional_document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    return _with_self(_same_doc(doc_ids, doc_ids))


def sliding_window_mask(doc_ids: torch.Tensor, window: int) -> torch.Tensor:
    T = doc_ids.shape[1]
    i = torch.arange(T, device=doc_ids.device)
    rel = i[:, None] - i[None, :]
    return _with_self(_same_doc(doc_ids, doc_ids) & ((rel >= 0) & (rel < window))[None])


def ced_mask(doc_ids: torch.Tensor, window: int) -> torch.Tensor:
    """[B, T, 2T]: local sliding-window keys, then global causal keys at the same positions."""
    return torch.cat([sliding_window_mask(doc_ids, window), document_causal_mask(doc_ids)], dim=-1)


def dense_mask(doc_ids: torch.Tensor, kind: str, window: int = 0) -> torch.Tensor:
    if kind == "causal":
        m = document_causal_mask(doc_ids)
    elif kind == "bidir":
        m = bidirectional_document_mask(doc_ids)
    elif kind == "window":
        m = sliding_window_mask(doc_ids, window)
    elif kind == "ced":
        m = ced_mask(doc_ids, window)
    else:
        raise ValueError(kind)
    return m[:, None]


def cu_seqlens_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """int32 segment boundaries over the flattened [B*T] batch. Host sync (nonzero); the loader precomputes it."""
    B, T = doc_ids.shape
    start = torch.ones_like(doc_ids, dtype=torch.bool)
    start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    idx = torch.nonzero(start.reshape(-1)).flatten()
    return torch.cat([idx, idx.new_tensor([B * T])]).to(torch.int32)


@dataclass
class VarlenMeta:
    """FlashAttention-2 varlen inputs. `max_seqlen` is the row length, an upper bound that needs no host sync."""
    cu_seqlens: torch.Tensor
    max_seqlen: int
    causal: bool
    window: int


@lru_cache(maxsize=1)
def flex_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def fa2_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        return True
    except Exception:
        return False


_FLEX = None


def flex_fn():
    global _FLEX
    if _FLEX is None:
        import torch._inductor.config as inductor_config
        backends = str(getattr(inductor_config, "max_autotune_gemm_backends", "TRITON"))
        if "ATEN" not in backends.upper():
            inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def block_mask(doc_ids: torch.Tensor, kind: str, window: int = 0):
    from torch.nn.attention.flex_attention import create_block_mask

    B, T = doc_ids.shape
    doc = doc_ids

    if kind == "causal":
        def mm(b, h, q, k):
            return (q >= k) & (doc[b, q] == doc[b, k])
        kv_len = T
    elif kind == "bidir":
        def mm(b, h, q, k):
            return doc[b, q] == doc[b, k]
        kv_len = T
    elif kind == "window":
        def mm(b, h, q, k):
            return (q >= k) & (q - k < window) & (doc[b, q] == doc[b, k])
        kv_len = T
    elif kind == "ced":
        def mm(b, h, q, k):
            is_g = k >= T
            kk = torch.where(is_g, k - T, k)
            return (q >= kk) & (doc[b, q] == doc[b, kk]) & (is_g | (q - kk < window))
        kv_len = 2 * T
    else:
        raise ValueError(kind)
    try:
        return create_block_mask(mm, B, None, T, kv_len, device=doc.device, _compile=True)
    except TypeError:
        return create_block_mask(mm, B, None, T, kv_len, device=doc.device)


def resolve_impl(requested: str, device: torch.device) -> str:
    """auto -> flex on CUDA when available, else sdpa. fa2 requires flash-attn; flex requires CUDA here."""
    if device.type != "cuda":
        return "sdpa"
    if requested == "fa2":
        if not fa2_available():
            raise RuntimeError("attn_impl=fa2 but flash_attn is not importable (build it for sm_120)")
        return "fa2"
    if requested in ("auto", "flex"):
        if flex_available():
            return "flex"
        if requested == "flex":
            raise RuntimeError("attn_impl=flex but torch.nn.attention.flex_attention is unavailable")
    return "sdpa"


def build_attn_masks(doc_ids: torch.Tensor, kinds: set[tuple[str, int]], impl: str,
                     cu_seqlens: torch.Tensor | None = None) -> dict:
    """One mask object per (kind, window). FA2 has no `ced` form; those layers fall back to flex/sdpa."""
    out: dict = {}
    T = doc_ids.shape[1]
    for kind, window in kinds:
        if impl == "fa2" and kind != "ced":
            if cu_seqlens is None:
                cu_seqlens = cu_seqlens_from_doc_ids(doc_ids)
            out[(kind, window)] = VarlenMeta(cu_seqlens.to(doc_ids.device, torch.int32), T,
                                             kind != "bidir", window if kind == "window" else 0)
        elif impl in ("flex", "fa2") and flex_available():
            out[(kind, window)] = block_mask(doc_ids, kind, window)
        else:
            out[(kind, window)] = dense_mask(doc_ids, kind, window)
    return out
