#!/usr/bin/env python
"""Environment + kernel smoke test for the RTX PRO 6000 Blackwell (sm_120).

Checks: GPU/capability, torch/CUDA versions, and forward+backward timings for
  * PyTorch SDPA (causal, GQA)            * FlexAttention (document-causal block mask, GQA)
  * FlashAttention-2 varlen (sm_120 cubins) * flash-linear-attention chunk_gated_delta_rule (varlen)
Exit code 1 if a *required* backend fails (SDPA, FlexAttention). FA2 / fla failures are reported as warnings
because the pipeline has fallbacks, but the report's throughput numbers assume they work.

    python scripts/env_check.py --seq 2048 --batch 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def timed(fn, iters: int = 10) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=64)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA not available -> CPU only (fine for unit tests, not for training)")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  capability {cap}  mem {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    if cap != (12, 0):
        print(f"[warn] expected sm_120 (12, 0) for RTX PRO 6000 Blackwell, got {cap}")
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    B, T, H, KV, D = args.batch, args.seq, args.heads, args.kv_heads, args.head_dim
    ok = True

    # GEMM peak sanity
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    ms = timed(lambda: a @ a, 10)
    print(f"bf16 GEMM 8192^3: {2 * 8192**3 / ms / 1e9:.0f} TFLOPS")

    q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

    # 1) SDPA
    try:
        def sdpa():
            kk = k.repeat_interleave(H // KV, 1)
            vv = v.repeat_interleave(H // KV, 1)
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            o.sum().backward()
        print(f"SDPA causal fwd+bwd: {timed(sdpa):.2f} ms")
    except Exception as e:
        ok = False
        print(f"[FAIL] SDPA: {e}")

    # 2) FlexAttention with document mask
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        flex = torch.compile(flex_attention, dynamic=False)
        doc = torch.cumsum((torch.rand(B, T, device=dev) < 1 / 700).int(), dim=1)

        def mask_mod(b, h, qi, ki):
            return (qi >= ki) & (doc[b, qi] == doc[b, ki])

        bm = create_block_mask(mask_mod, B, None, T, T, device=dev)

        def fx():
            o = flex(q, k, v, block_mask=bm, enable_gqa=True)
            o.sum().backward()
        print(f"FlexAttention doc-causal fwd+bwd: {timed(fx):.2f} ms")
    except Exception as e:
        ok = False
        print(f"[FAIL] FlexAttention: {e}")

    # 3) FA2 varlen
    try:
        from flash_attn import flash_attn_varlen_func

        qv = q.detach().transpose(1, 2).reshape(B * T, H, D).requires_grad_()
        kv_ = k.detach().transpose(1, 2).reshape(B * T, KV, D).requires_grad_()
        vv_ = v.detach().transpose(1, 2).reshape(B * T, KV, D).requires_grad_()
        cu = torch.arange(0, B * T + 1, T, device=dev, dtype=torch.int32)

        def fa2():
            o = flash_attn_varlen_func(qv, kv_, vv_, cu, cu, T, T, causal=True)
            o.sum().backward()
        print(f"FA2 varlen causal fwd+bwd: {timed(fa2):.2f} ms")
    except Exception as e:
        print(f"[warn] FA2 unavailable ({e}). Build with TORCH_CUDA_ARCH_LIST=12.0; FlexAttention will be used.")

    # 4) fla GatedDeltaNet chunk kernel, varlen
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        Hg, Kd = 8, 128
        N = B * T
        qg = F.normalize(torch.randn(1, N, Hg, Kd, device=dev, dtype=torch.bfloat16), dim=-1).requires_grad_()
        kg = F.normalize(torch.randn(1, N, Hg, Kd, device=dev, dtype=torch.bfloat16), dim=-1).requires_grad_()
        vg = torch.randn(1, N, Hg, Kd, device=dev, dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(1, N, Hg, device=dev, dtype=torch.float32) * 0.1).requires_grad_()
        beta = torch.rand(1, N, Hg, device=dev, dtype=torch.bfloat16).requires_grad_()
        cu = torch.arange(0, N + 1, 700, device=dev, dtype=torch.long)
        if cu[-1] != N:
            cu = torch.cat([cu, torch.tensor([N], device=dev)])

        def gdn():
            o, _ = chunk_gated_delta_rule(qg, kg, vg, g, beta, cu_seqlens=cu)
            o.sum().backward()
        print(f"fla chunk_gated_delta_rule varlen (8x128, {N} tok) fwd+bwd: {timed(gdn):.2f} ms")
    except Exception as e:
        print(f"[warn] flash-linear-attention GDN unavailable ({e}); Saul-Hybrid will fall back to the slow reference path.")

    print("OK" if ok else "REQUIRED BACKEND FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
