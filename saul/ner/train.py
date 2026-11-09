"""Fine-tune one Phase-1 checkpoint on one fold, or on every fold.

Checkpoints are chosen by strict macro-F1 on the held-out fold with the untuned token
decoder (saul.ner.tokdec defaults). Its thresholds and switches are fit later, on the
saved out-of-fold token distributions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from saul.ner.backbone import NERModel, load_ner, load_overlay, overlay_state, trainable_parameters
from saul.ner.data import (
    collate, docs_for_ids, epoch_windows, load_folds, load_split, span_weights, swap_role_strings,
    tag_weights, windows_for,
)
from saul.ner.infer import infer_docs, predict_tokens, save_inference
from saul.ner.metrics import gold_map, strict_scores
from saul.ner.registry import CONTEXT, p2_dir
from saul.train.ema import TailEMA
from saul.utils.io import MetricsLogger, ensure_dir


def _groups(model: NERModel, technique: str, base_lr: float, head_lr: float, adapter_lr: float, decay: float):
    groups = []
    head_params = [p for p in model.heads.parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    peft = technique in ("lorma", "lora", "lorma_pi", "adapter_bitfit")
    if peft:
        others = [p for p in model.backbone.parameters() if p.requires_grad]
        if others:
            groups.append({"params": others, "lr": adapter_lr})
        return groups
    n = len(model.backbone.blocks)
    buckets: dict[float, list] = {}
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            lr = base_lr * (decay ** (n - 1 - idx))
        elif name.startswith("embed") or name.startswith("backbone.embed"):
            lr = base_lr * (decay ** n)
        else:
            lr = base_lr
        buckets.setdefault(round(lr, 12), []).append(p)
    for lr, params in buckets.items():
        groups.append({"params": params, "lr": lr})
    return groups


def _lr_mult(step: int, total: int, warmup: float) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def forwards_per_step(rdrop: float, fgm: float) -> int:
    """Defaults (both zero) take a single forward. R-Drop and FGM each add one."""
    return 1 + int(rdrop > 0) + int(fgm > 0)


def _sym_kl(e1: torch.Tensor, e2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(e1.float(), dim=-1)
    log_q = F.log_softmax(e2.float(), dim=-1)
    p, q = log_p.exp(), log_q.exp()
    kl = 0.5 * ((p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1))
    denom = mask.to(kl.dtype).sum().clamp(min=1)
    return (kl * mask.to(kl.dtype)).sum() / denom


def reinit_top_blocks(model: NERModel, n: int) -> None:
    """Re-init the last n blocks the way Phase-1 does: N(0, 0.02) linears, zero output maps, zero norm gamma.

    ZeroCenteredRMSNorm is y = x/rms * (1 + gamma), so gamma = 0 is the Phase-1 unit gain.
    """
    if n <= 0:
        return
    import torch.nn as nn
    from saul.model.norm import QKNorm, ZeroCenteredRMSNorm
    output = {"o", "down", "o_proj", "out_proj"}
    for block in list(model.backbone.blocks)[-n:]:
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = name.rsplit(".", 1)[-1]
                if leaf in output:
                    nn.init.zeros_(mod.weight)
                else:
                    nn.init.normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, (ZeroCenteredRMSNorm, QKNorm)):
                nn.init.zeros_(mod.gamma)


def _dev_f1(model, dev_docs, batch_size: int, header: int | None, device: str) -> float:
    if not dev_docs:
        return 0.0
    was = model.training
    result = infer_docs(model, dev_docs, batch_size=batch_size, header=header, device=device, mentions=False)
    score = strict_scores(gold_map(dev_docs), predict_tokens(dev_docs, result))["macro_f1"]
    model.train(was)
    return float(score)


def train_one(exp: str, model_name: str, technique: str, fold: int, context: str | None = None,
              epochs: int = 10, batch_size: int = 16, accum: int = 1, header: int | None = None,
              base_lr: float = 3e-5, head_lr: float = 1e-3, adapter_lr: float = 5e-4,
              decay: float = 0.85, init: str | None = None, pseudo: str | None = None,
              device: str | None = None, seed: int = 781, adam_eps: float = 1e-8,
              rdrop: float = 0.0, fgm: float = 0.0, reinit_top: int = 0, hidden_dropout: float = 0.0,
              augment: str = "none", header_mode: str = "raw", preamble: bool = False,
              silver: str | None = None, silver_epochs: int = 1, train_frac: float = 1.0,
              full_data: bool = False) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    context = context or CONTEXT[model_name]
    run = os.path.join(p2_dir(), exp, "full" if full_data else f"fold{fold}")
    if os.path.exists(os.path.join(run, "done.json")):
        print(f"[skip] {run} already done")
        return run
    ensure_dir(run)
    docs = load_split("train")
    folds = load_folds(docs)
    if full_data:
        train_ids = [d.doc_id for d in docs]
        dev_ids: list[str] = []
    else:
        dev_ids = folds[fold]
        train_ids = [i for fi, ids in enumerate(folds) if fi != fold for i in ids]
    train_docs = docs_for_ids(docs, train_ids)
    dev_docs = docs_for_ids(docs, dev_ids)
    if train_frac < 1.0:
        pick = np.random.default_rng(seed + fold).permutation(len(train_docs))
        keep = max(1, int(round(len(train_docs) * train_frac)))
        train_docs = [train_docs[int(i)] for i in pick[:keep]]
    model = load_ner(model_name, technique, context=context, init=init, device=device)
    reinit_top_blocks(model, reinit_top)
    model.hidden_dropout_p = hidden_dropout
    model.header_mode = header_mode
    model.preamble = preamble
    model.heads.set_weights(tag_weights(train_docs).to(device), span_weights(train_docs).to(device))
    if pseudo:
        from saul.ner.kd import load_extra
        extra = load_extra(pseudo, context, header, header_mode=header_mode, preamble=preamble)
    else:
        extra = []
    windows = windows_for(train_docs, context, header, header_mode=header_mode, preamble=preamble)
    groups = _groups(model, technique, base_lr, head_lr, adapter_lr, decay)
    opt = torch.optim.AdamW(groups, weight_decay=0.01, eps=adam_eps)
    base_lrs = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(seed + fold)
    expected = max(len(windows), 1)
    steps_per_epoch = math.ceil((expected * 1.2) / batch_size)
    silver_windows = []
    silver_steps = 0
    if silver:
        from saul.ner.kd import load_extra, resolve_silver_path
        silver_windows = load_extra(
            resolve_silver_path(silver, fold), context, header,
            header_mode=header_mode, preamble=preamble,
        )
        silver_steps = math.ceil(max(len(silver_windows), 1) / batch_size) * max(silver_epochs, 0)
    total_steps = max(1, epochs * steps_per_epoch + silver_steps)
    ema_from = int(total_steps * 0.8)
    ema = TailEMA(model, decay=0.999)
    logger = MetricsLogger(run, {"enabled": True, "project": "saul-phase2", "name": f"{exp}-fold{fold}",
                                  "group": exp})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {exp} fold {fold} {model_name} {technique} {context} trainable={n_train} "
          f"windows={len(windows)} device={device}", flush=True)
    best = -1.0
    best_blob = None
    step = 0
    model.train()
    use_cuda = device == "cuda"
    metrics_path = os.path.join(run, "metrics.jsonl")

    def _once(chunk: list) -> dict:
        batch = {k: v.to(device, non_blocking=True) for k, v in collate(chunk).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            return model(batch)

    def _batch_loss(chunk: list) -> tuple[torch.Tensor, dict]:
        out = _once(chunk)
        loss = out["loss"]
        if rdrop > 0:
            out2 = _once(chunk)
            loss = 0.5 * (loss + out2["loss"]) + rdrop * _sym_kl(out["emissions"], out2["emissions"], out["mask"])
        return loss, out

    def _run_epochs(epoch_list, window_source, rare_boost: bool, flat_lr: float | None, do_dev: bool,
                    swap: bool) -> None:
        nonlocal step, best, best_blob
        for epoch in epoch_list:
            # Silver windows are already tagged. Swapping gold documents here would
            # throw those tags away. On the gold stage, rebuild windows from the
            # swapped text so a parsed header is parsed again.
            if swap and augment == "mention_swap":
                epoch_docs = []
                for doc in train_docs:
                    if rng.random() < 0.5:
                        epoch_docs.append(swap_role_strings(doc, train_docs, rng))
                    else:
                        epoch_docs.append(doc)
                order_src = windows_for(epoch_docs, context, header, header_mode=header_mode, preamble=preamble)
            else:
                order_src = window_source
            order = epoch_windows(order_src, rng, rare_boost=rare_boost)
            # Pseudo windows stay in their own batches. Mixing them into a gold batch would
            # drop the CRF for the gold rows, because an IGNORE tag makes the chain incomplete.
            pseudo_order = None
            if extra and flat_lr is None:
                pseudo_order = [extra[int(i)] for i in rng.integers(0, len(extra), size=len(order))]
            t0 = time.time()
            running = 0.0
            seen = 0
            opt.zero_grad(set_to_none=True)
            for start_i in range(0, len(order), batch_size):
                loss, out = _batch_loss(order[start_i:start_i + batch_size])
                if pseudo_order is not None:
                    extra_loss, _ = _batch_loss(pseudo_order[start_i:start_i + batch_size])
                    loss = 0.5 * loss + 0.5 * extra_loss
                (loss / accum).backward()
                if fgm > 0:
                    emb = model.backbone.embed.weight
                    grad = emb.grad
                    if grad is not None:
                        delta = fgm * grad / (grad.norm() + 1e-8)
                        emb.data.add_(delta)
                        ( _once(order[start_i:start_i + batch_size])["loss"] / accum).backward()
                        emb.data.sub_(delta)
                seen += 1
                running += float(loss.detach())
                if seen % accum != 0:
                    continue
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0))
                mult = _lr_mult(step, total_steps, 0.1)
                for group, lr0 in zip(opt.param_groups, base_lrs):
                    group["lr"] = (flat_lr if flat_lr is not None else lr0) * mult
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step >= ema_from and flat_lr is None:
                    ema.update(model)
                if step % 20 == 0:
                    rec = {"step": step, "loss": float(loss.detach()), "grad_norm": grad_norm,
                           "nll": float(out["nll"].detach()), "ce": float(out["ce"].detach()),
                           "span": float(out["span"].detach())}
                    with open(metrics_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                step += 1
            if seen % accum != 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            dev = None
            if do_dev and ((epoch % 2 == 1) or epoch == epoch_list[-1]):
                dev = _dev_f1(model, dev_docs, batch_size, header, device)
                if dev > best:
                    best = dev
                    best_blob = overlay_state(model)
                if ema.updates:
                    backup = ema.copy_to(model)
                    ema_dev = _dev_f1(model, dev_docs, batch_size, header, device)
                    if ema_dev > best:
                        best = ema_dev
                        best_blob = overlay_state(model)
                    ema.restore(model, backup)
                    dev = max(dev, ema_dev)
            logger.log({"epoch": epoch, "loss": running / max(seen, 1), "dev_f1": -1 if dev is None else dev,
                        "seconds": time.time() - t0}, step)
            print(f"[epoch] {epoch} loss={running / max(seen, 1):.4f} dev={dev} best={best:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if silver_windows and silver_epochs > 0:
        _run_epochs(range(silver_epochs), silver_windows, rare_boost=False, flat_lr=3e-5, do_dev=False,
                    swap=False)
    _run_epochs(range(epochs), windows, rare_boost=True, flat_lr=None, do_dev=not full_data, swap=True)
    if best_blob is None:
        best_blob = overlay_state(model)
    torch.save(best_blob, os.path.join(run, "best.pt"))
    load_overlay(model, best_blob)
    final_f1 = None
    per_label = {}
    if dev_docs:
        result = infer_docs(model, dev_docs, batch_size=batch_size, header=header, device=device)
        save_inference(run, dev_docs, result)
        final = strict_scores(gold_map(dev_docs), predict_tokens(dev_docs, result))
        final_f1 = final["macro_f1"]
        per_label = final["per_label"]
    with open(os.path.join(run, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"exp": exp, "fold": fold, "model": model_name, "technique": technique,
                   "context": context, "dev_f1": final_f1, "best_seen": best,
                   "per_label": per_label, "header_mode": header_mode, "preamble": preamble,
                   "full_data": full_data}, f, indent=2)
    print(f"[done] {run} dev_f1={final_f1}", flush=True)
    if logger._wandb is not None:
        logger._wandb.finish()
    return run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--technique", required=True)
    p.add_argument("--context", default=None)
    p.add_argument("--fold", default="0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--header", type=int, default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--pseudo", default=None)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=781)
    p.add_argument("--rdrop", type=float, default=0.0)
    p.add_argument("--fgm", type=float, default=0.0)
    p.add_argument("--reinit-top", type=int, default=0)
    p.add_argument("--hidden-dropout", type=float, default=0.0)
    p.add_argument("--augment", default="none", choices=("none", "mention_swap"))
    p.add_argument("--preamble-flag", action="store_true")
    p.add_argument("--header-mode", default="raw", choices=("raw", "parsed"))
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--silver", default=None)
    p.add_argument("--silver-epochs", type=int, default=1)
    args = p.parse_args()
    epochs = args.epochs if args.epochs is not None else (2 if args.technique == "probe" else 10)
    full_data = args.fold == "full"
    folds = [0] if full_data or args.fold == "all" else [int(x) for x in str(args.fold).split(",")]
    if args.fold == "all":
        folds = list(range(5))
    for fold in folds:
        train_one(args.exp, args.model, args.technique, fold, context=args.context, epochs=epochs,
                  batch_size=args.batch_size, accum=args.accum, header=args.header, base_lr=args.lr,
                  head_lr=args.head_lr, init=args.init, pseudo=args.pseudo, seed=args.seed,
                  adam_eps=args.adam_eps, rdrop=args.rdrop, fgm=args.fgm, reinit_top=args.reinit_top,
                  hidden_dropout=args.hidden_dropout, augment=args.augment, header_mode=args.header_mode,
                  preamble=args.preamble_flag, silver=args.silver, silver_epochs=args.silver_epochs,
                  train_frac=args.train_frac, full_data=full_data)


if __name__ == "__main__":
    main()
