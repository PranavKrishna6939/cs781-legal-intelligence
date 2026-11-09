#!/usr/bin/env python
"""Stage 33: online distillation of Saul-Conv-Mini from a Saul-Dense checkpoint (the teacher is our own model)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.data.loader import PackedLMDataset, make_dataloader  # noqa: E402
from saul.data.stats import mixture_weights  # noqa: E402
from saul.data.tokens import open_split  # noqa: E402
from saul.model.factory import config_from_yaml  # noqa: E402
from saul.model.lm_head import next_token_targets  # noqa: E402
from saul.model.modeling_saul import SaulForCausalLM, build_model  # noqa: E402
from saul.train.distill import distillation_loss  # noqa: E402
from saul.train.optim import build_optimizers, optimizer_step  # noqa: E402
from saul.train.schedule import wsd_multiplier  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import init_run_dir, read_json  # noqa: E402
from saul.utils.params import count_params  # noqa: E402
from saul.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student_config", default="configs/models/conv_mini_99m.yaml")
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    cfg.run_dir = os.path.join(cfg.run_dir, "conv_mini_kd")
    init_run_dir(cfg.run_dir, cfg)
    seed_everything(int(cfg.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda"
    teacher = SaulForCausalLM.from_pretrained(a.teacher).to(device).eval()
    if bf16:
        teacher = teacher.to(torch.bfloat16)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_model(config_from_yaml(a.student_config), enforce_min=True).to(device)
    print(f"student params {count_params(student):,}")
    dcfg = load_config(cfg.data_config)
    shares = read_json(os.path.join(cfg.tokens, "token_shares.json"))["train"]
    weights = mixture_weights({s: shares[s]["tokens"] for s in shares}, dict(dcfg.mixture.stable))
    sources = open_split(cfg.tokens, "train", list(dcfg.sources))
    ds = PackedLMDataset(sources, weights, cfg.train.seq_len, student.config.bos_token_id,
                         student.config.eos_token_id, student.config.pad_token_id, seed=int(cfg.seed) + 1)
    loader = make_dataloader(ds, cfg.train.batch_size, num_workers=int(cfg.train.num_workers),
                             pad_id=student.config.pad_token_id)
    muon, adamw = build_optimizers(student, float(cfg.optim.muon_lr), float(cfg.optim.adamw_lr))
    data = iter(loader)
    accum = max(1, int(cfg.train.grad_accum))
    tokens_per_step = int(cfg.train.batch_size) * int(cfg.train.seq_len) * accum
    print(f"kd tokens/step {tokens_per_step:,} ({a.steps} steps = {a.steps * tokens_per_step / 1e9:.2f}B)", flush=True)
    for step in range(a.steps):
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _micro in range(accum):
            batch = next(data)
            batch.pop("_wstate", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            kw = dict(doc_ids=batch["doc_ids"], position_ids=batch["position_ids"], cu_seqlens=batch["cu_seqlens"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                with torch.no_grad():
                    t_logits = teacher(batch["input_ids"], **kw).logits
                s_logits = student(batch["input_ids"], **kw).logits
            tgt = next_token_targets(batch["labels"], batch["doc_ids"])
            loss = distillation_loss(s_logits, t_logits, tgt, k=a.topk)
            (loss / accum).backward()
            loss_acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer_step(muon, adamw, wsd_multiplier(step, a.steps))
        if step % 20 == 0:
            print(f"kd step {step} loss {loss_acc / accum:.4f}", flush=True)
    student.save_pretrained(os.path.join(cfg.run_dir, "student"))
    print("saved", os.path.join(cfg.run_dir, "student"))


if __name__ == "__main__":
    main()
