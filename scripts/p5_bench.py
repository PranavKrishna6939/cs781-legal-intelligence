"""bf16 training throughput per backbone and technique on the current GPU."""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from saul.lsi import p5_dir
from saul.lsi.model import load_lsi, trainable_parameters
from saul.lsi.train import synthetic_batch


def bench(name: str, technique: str, width: int, cases: int, head: str, ckpt_every: int,
          context: str | None = None) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_lsi(name, technique, context, head=head, loss="bce", device="cuda", ckpt_every=ckpt_every,
                     attn="sdpa", compile_blocks=True)
    model.train()
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-5)
    batch = synthetic_batch(width, n_cases=cases, chunks_each=1, vocab=32000, device="cuda")
    tokens = int((batch["doc_ids"] >= 0).sum())
    try:
        for step in range(8):
            if step == 3:
                torch.cuda.synchronize()
                start = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
            out["loss"].backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tok_s = tokens * 5 / (time.time() - start)
        result = {"ok": True, "tok_s": round(tok_s), "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
    except torch.OutOfMemoryError:
        result = {"ok": False, "oom": True}
    del model, opt
    return {"model": name, "technique": technique, "context": context, "width": width, "cases": cases,
            "head": head, **result}


def main() -> None:
    rows = []
    for spec in sys.argv[1:]:
        name, technique, width, cases, head, ckpt, *context = spec.split(":")
        row = bench(name, technique, int(width), int(cases), head, int(ckpt), context[0] if context else None)
        print(json.dumps(row), flush=True)
        rows.append(row)
    os.makedirs(os.path.join(p5_dir(), "bench"), exist_ok=True)
    with open(os.path.join(p5_dir(), "bench", f"bf16_{int(time.time())}.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle)


if __name__ == "__main__":
    main()
