"""Variable-length flash attention for decode.

Batched flash attention ignores per-row lengths, so padding would be attended. This packs each row's real keys
into one buffer and calls flash varlen, which matches a sliced SDPA exactly and does not build a dense mask.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU tests never import the kernel
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _pack_kv(src, dst, cu, seq, stride_row, stride_tok, width, TOK: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        n = tl.load(seq + row)
        t0 = tile * TOK
        if t0 >= n:
            return
        dest = tl.load(cu + row)
        offs_t = t0 + tl.arange(0, TOK)
        offs_d = tl.arange(0, BLOCK)
        mask = (offs_t < n)[:, None] & (offs_d[None, :] < width)
        val = tl.load(src + row * stride_row + offs_t[:, None] * stride_tok + offs_d[None, :], mask=mask, other=0)
        tl.store(dst + (dest + offs_t)[:, None] * width + offs_d[None, :], val, mask=mask)


_BUFFERS: dict = {}


def _workspace(k: torch.Tensor):
    rows, length, nkv, dim = k.shape
    key = (str(k.device), rows, length, nkv, dim, k.dtype)
    buf = _BUFFERS.get(key)
    need = rows * length
    if buf is None or buf[0].shape[0] < need:
        buf = (
            k.new_empty(need, nkv, dim),
            k.new_empty(need, nkv, dim),
            torch.empty(rows + 1, dtype=torch.int32, device=k.device),
            torch.arange(rows + 1, dtype=torch.int32, device=k.device),
        )
        _BUFFERS[key] = buf
    return buf


@torch.compiler.disable
def flash_varlen_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused: torch.Tensor,
                        window: int = 0) -> torch.Tensor:
    """q: [B, 1, H, D], k/v: [B, L, KV, D], seqused: [B] valid key counts. Returns [B, 1, H, D]."""
    if triton is None:
        raise RuntimeError("varlen decode needs triton")
    rows, length, nkv, dim = k.shape
    seq = seqused.to(torch.int32).clamp(min=1, max=length)
    # Lockstep batches share one length. Flash can read that prefix in place; no pack, no dense mask.
    if not window and rows > 0 and bool(torch.equal(seq, seq[:1].expand_as(seq))):
        n = int(seq[0])
        out = torch.ops.aten._flash_attention_forward(
            q, k[:, :n], v[:, :n], None, None, 1, n, 0.0, False, False, scale=dim ** -0.5,
        )[0]
        return out.view(rows, 1, q.shape[2], dim)
    packed_k, packed_v, cu, cu_q = _workspace(k)
    cu[0] = 0
    torch.cumsum(seq, 0, out=cu[1:])
    width = nkv * dim
    tiles = triton.cdiv(length, 32)
    _pack_kv[(rows, tiles)](k, packed_k, cu, seq, k.stride(0), k.stride(1), width, TOK=32, BLOCK=256)
    _pack_kv[(rows, tiles)](v, packed_v, cu, seq, v.stride(0), v.stride(1), width, TOK=32, BLOCK=256)
    kwargs = {"scale": dim ** -0.5}
    if window:
        kwargs["window_size_left"] = int(window - 1)
        kwargs["window_size_right"] = 0
    out = torch.ops.aten._flash_attention_forward(
        q.reshape(rows, q.shape[2], dim), packed_k, packed_v, cu_q, cu, 1, length, 0.0, False, False, **kwargs,
    )[0]
    return out.view(rows, 1, q.shape[2], dim)
