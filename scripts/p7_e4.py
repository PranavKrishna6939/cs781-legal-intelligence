"""E4: distill the E2 cross-encoder into a 6-block student bi-encoder."""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from p7_common import OUT, save_metrics
from p7_e1 import embed, info_nce, score_dev
from p7_e2 import GROUP
from p7_model import Head, build, checkpoint_every, cosine, gpu_mb, load_tokenizer


def load_groups(limit: int = 4000) -> list[dict]:
    path = os.path.join(OUT, "e2", "teacher.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main():
    os.makedirs(os.path.join(OUT, "e4"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bos, pad = load_tokenizer()
    groups = load_groups()
    model = build("student")
    head = Head(model.config.d_model).to(device)
    checkpoint_every(model)
    model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(list(trainable) + list(head.parameters()), lr=1e-4, weight_decay=0.01)
    print(f"[e4] trainable {sum(p.numel() for p in trainable)} groups {len(groups)}", flush=True)
    batch = 4
    epochs = 1
    total = epochs * ((len(groups) + batch - 1) // batch)
    step = 0
    started = time.time()
    peak = 0
    for epoch in range(epochs):
        order = list(range(len(groups)))
        random.Random(781 + epoch).shuffle(order)
        for start in range(0, len(order), batch):
            chunk = [groups[i] for i in order[start:start + batch]]
            opt.zero_grad(set_to_none=True)
            texts_ctx = []
            texts_pos = []
            for group in chunk:
                texts_ctx.append(group["context"])
                texts_pos.append(group["passages"][0])
            anchor = embed(model, head, tok, bos, pad, texts_ctx, device)
            positive = embed(model, head, tok, bos, pad, texts_pos, device, "mean")
            student_scores = []
            for group in chunk:
                ctx = embed(model, head, tok, bos, pad, [group["context"]], device)
                passages_ = embed(model, head, tok, bos, pad, group["passages"], device, "mean")
                student_scores.append((ctx @ passages_.T).view(-1))
            student = torch.stack(student_scores)
            teacher = torch.tensor([group["logits"] for group in chunk], device=device, dtype=torch.float32)
            # Margin-MSE on the positive against each negative, plus listwise KL.
            margin_s = student[:, :1] - student[:, 1:]
            margin_t = teacher[:, :1] - teacher[:, 1:]
            margin = F.mse_loss(margin_s, margin_t)
            kl = F.kl_div(
                F.log_softmax(student / 2.0, dim=-1),
                F.softmax(teacher / 2.0, dim=-1),
                reduction="batchmean",
            )
            contra = info_nce(anchor, positive)
            loss = margin + kl + 0.1 * contra
            if not torch.isfinite(loss):
                raise SystemExit("non-finite distillation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
            for group in opt.param_groups:
                group["lr"] = cosine(step, total, 1e-4)
            opt.step()
            step += 1
            peak = max(peak, gpu_mb())
            if step % 20 == 0:
                print(f"[e4] step {step}/{total} loss {float(loss):.3f}", flush=True)
    elapsed = time.time() - started
    torch.save(
        {"model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
         "head": head.state_dict()},
        os.path.join(OUT, "e4", "student.pt"),
    )
    # Reuse the E1 dev scorer. It writes P7_E4 when the E1 gate is not required:
    # score_dev only withholds the zip for metrics name "e1".
    t0 = time.time()
    score_dev(model, head, tok, bos, pad, device, "P7_E4", "e4")
    query_ms = (time.time() - t0) * 1000 / 118
    payload = json.load(open(os.path.join(OUT, "e4", "metrics.json")))
    payload.update({
        "parameters": int(sum(p.numel() for p in trainable) + sum(p.numel() for p in head.parameters())),
        "peak_gpu_mb": peak,
        "train_seconds": elapsed,
        "ms_per_dev_query": query_ms,
        "group": GROUP,
    })
    save_metrics("e4", payload)
    print("[e4] done", payload.get("dev", {}), flush=True)


if __name__ == "__main__":
    main()
