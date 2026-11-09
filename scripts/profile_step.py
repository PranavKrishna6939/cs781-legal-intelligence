#!/usr/bin/env python
"""Profile one training micro-batch (forward + backward) of a model config on real packed data.

    python scripts/profile_step.py --model configs/models/hybrid_294m.yaml [--compile] [--batch 8 --seq 2048]

Prints the mean micro-batch time over `--iters` and the top CUDA kernels / CPU ops by self time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.data.loader import PackedLMDataset, collate_lm  # noqa: E402
from saul.data.tokens import open_split  # noqa: E402
from saul.model.factory import config_from_yaml  # noqa: E402
from saul.model.modeling_saul import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", default="data/tokens/legal32k")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--rows", type=int, default=25)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(config_from_yaml(a.model), enforce_min=False).cuda().train()
    if a.compile:
        for b in model.blocks:
            b.compile(dynamic=False)
    src = open_split(a.tokens, "train", ["supreme_court", "tribunals", "acts", "high_courts"])
    ds = PackedLMDataset(src, {s: 1.0 for s in src}, a.seq, 0, 1, 2, worker_override=(0, 1))
    it = iter(ds)
    batches = []
    for _ in range(a.iters + 2):
        b = collate_lm([next(it) for _ in range(a.batch)])
        b.pop("_wstate", None)
        cpu_cu = b["cu_seqlens"].long()
        b = {k: v.cuda() for k, v in b.items()}
        b["cu_seqlens_cpu"] = cpu_cu
        batches.append(b)

    def step(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(b["input_ids"], labels=b["labels"], doc_ids=b["doc_ids"], position_ids=b["position_ids"],
                         cu_seqlens=b["cu_seqlens"], cu_seqlens_cpu=b["cu_seqlens_cpu"]).loss
        loss.backward()

    for b in batches[:2]:
        step(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in batches[2:]:
        step(b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / a.iters
    ntok = a.batch * a.seq
    print(f"micro-batch fwd+bwd {dt*1000:.1f} ms  -> {ntok/dt:,.0f} tokens/s (compile={a.compile})")
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(batches[2])
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=a.rows))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
