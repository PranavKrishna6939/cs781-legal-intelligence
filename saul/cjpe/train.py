"""Fine-tune one Phase-1 backbone for binary judgment prediction."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from saul.cjpe import exp_dir
from saul.cjpe.const import A1_GATE, SEED
from saul.cjpe.data import (
    collate, holdout_indices, load_split, make_batches, make_view, sample_indices, subset_indices,
    training_split,
)
from saul.cjpe.calibrate import apply_prior
from saul.cjpe.model import load_cjpe, load_overlay, overlay_state, trainable_parameters
from saul.cjpe.score import macro_scores, predict_logits
from saul.utils.io import MetricsLogger, ensure_dir


def _take(params, seen: set[int]) -> list:
    """Drop tied duplicates. The LM head shares the embedding when weights are tied."""
    out = []
    for param in params:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        out.append(param)
    return out


def _groups(model, technique: str, head_lr: float, adapter_lr: float, base_lr: float, decay: float):
    groups = []
    seen: set[int] = set()
    head = _take(model.head.parameters(), seen)
    if head:
        groups.append({"params": head, "lr": head_lr})
    peft = technique in ("lorma", "lora", "lorma_pi", "adapter_bitfit", "ia3", "probe")
    if peft:
        others = _take(model.backbone.parameters(), seen)
        if others:
            groups.append({"params": others, "lr": adapter_lr})
        return groups
    n_blocks = len(model.backbone.blocks)
    buckets: dict[float, list] = {}
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if name.startswith("blocks."):
            index = int(name.split(".")[1])
            lr = base_lr * (decay ** (n_blocks - 1 - index))
        else:
            lr = base_lr * (decay ** n_blocks)
        buckets.setdefault(round(lr, 12), []).append(param)
    for lr, params in buckets.items():
        groups.append({"params": params, "lr": lr})
    return groups


def _lr_mult(step: int, total: int, warmup: float) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _move(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _teacher_lookup(table: dict[str, np.ndarray] | None, ids: list[str], device: str):
    if table is None:
        return None
    rows = [table[i] for i in ids]
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def load_teacher(path: str | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    blob = np.load(path, allow_pickle=True)
    ids = [str(i) for i in blob["ids"].tolist()]
    probs = np.asarray(blob["probs"], dtype=np.float32)
    return {i: row for i, row in zip(ids, probs)}


@torch.no_grad()
def predict_split(model, split, device: str, view: str, limit: int, head: int, tail: int,
                  token_budget: int) -> tuple[np.ndarray, list[str]]:
    model.eval()
    causal = model.context == "causal"
    sequences, labels, ids, lengths = [], [], [], []
    for i in range(len(split)):
        seq = make_view(split.tokens_of(i), view, limit, head, tail, causal)
        sequences.append(seq)
        labels.append(int(split.labels[i]))
        ids.append(split.ids[i])
        lengths.append(len(seq))
    order = []
    logits = np.zeros((len(split), 2), dtype=np.float32)
    for width, batch_idx in make_batches(lengths, token_budget):
        batch = collate([sequences[i] for i in batch_idx], [labels[i] for i in batch_idx],
                        [ids[i] for i in batch_idx], width)
        batch = _move(batch, device)
        batch.pop("labels", None)
        pred = model(batch)["logits"].float().cpu().numpy()
        for row, index in enumerate(batch_idx):
            logits[index] = pred[row]
        order.extend(batch_idx)
    _ = order
    return logits, ids


def _save_logits(path: str, ids: list[str], logits: np.ndarray, labels: np.ndarray | None = None) -> None:
    payload = {"ids": np.asarray(ids), "logits": logits}
    if labels is not None:
        payload["labels"] = labels
    np.savez(path, **payload)


def _eval_dsel(model, dsel, device, args, limit: int | None = None) -> dict:
    raw, ids = predict_split(
        model, dsel, device, args.view, args.limit if limit is None else limit,
        args.head, args.tail, args.batch_tokens,
    )
    logits = apply_prior(raw, model.loss_name == "ce")
    labels = dsel.labels
    pred = predict_logits(logits, 0.0)
    scores = macro_scores(labels, pred)
    halves = np.asarray(dsel.halves)
    for name in ("A", "B"):
        mask = halves == name
        scores[f"f1_{name.lower()}"] = macro_scores(labels[mask], pred[mask])["f1"] if mask.any() else 0.0
    return {"scores": scores, "logits": raw, "ids": ids}


def synthetic_batch(width: int, batch: int, vocab: int, device: str) -> dict:
    ids = torch.randint(10, vocab, (batch, width), device=device)
    docs = torch.zeros(batch, width, dtype=torch.long, device=device)
    return {
        "input_ids": ids,
        "doc_ids": docs,
        "labels": torch.randint(0, 2, (batch,), device=device),
        "lengths": torch.full((batch,), width, device=device),
        "ids": [str(i) for i in range(batch)],
    }


def run_bench(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_cjpe(
        args.model, args.technique, args.context, device=device, loss="ce",
        ckpt_every=args.ckpt_every, attn=args.attn,
    )
    model.train()
    width = args.limit
    result = {"model": args.model, "width": width, "attn": args.attn, "ok": False}
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-4)
    for batch_size in (8, 4, 2, 1):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            step_model = model
            if args.compile and device == "cuda":
                step_model = torch.compile(model)
            def once():
                batch = synthetic_batch(width, batch_size, int(model.backbone.config.vocab_size), device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = step_model(batch)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return loss
            for _ in range(3):
                once()
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            steps = 20
            last = None
            for _ in range(steps):
                last = once()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.time() - start, 1e-6)
            tokens = batch_size * width * steps
            result.update({
                "ok": True,
                "batch": batch_size,
                "tok_s": tokens / elapsed,
                "peak_gb": _peak_gb(),
                "loss": float(last.detach().float()),
                "finite": bool(torch.isfinite(last)),
            })
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
    out = os.path.join(args.bench_dir, f"bench_{args.model}_{width}_{args.attn or 'auto'}.json")
    ensure_dir(args.bench_dir)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"[bench] {result}", flush=True)
    return result


def train_one(args) -> None:
    directory = exp_dir(args.exp)
    ensure_dir(directory)
    done = os.path.join(directory, "done.json")
    if os.path.exists(done) and not args.force:
        print(f"[train] reuse {args.exp}", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bench:
        run_bench(args)
        return

    base_lr = 2e-5 if args.technique in ("full", "partial") else 5e-4
    adapter_lr = args.adapter_lr if args.adapter_lr is not None else base_lr
    model = load_cjpe(
        args.model, args.technique, args.context, init=args.init, device=device,
        loss=args.loss, ckpt_every=0 if args.smoke else args.ckpt_every, attn=args.attn,
    )
    causal = model.context == "causal"
    n_train = sum(p.numel() for p in trainable_parameters(model))
    print(f"[train] {args.exp} {args.model} {args.technique} {model.context} trainable={n_train}", flush=True)

    if args.smoke:
        _smoke(model, device, args, directory, n_train)
        return

    pool = training_split(refit=args.refit)
    dsel = load_split("dsel")
    test = load_split("test")
    indices = list(range(len(pool)))
    if args.holdout:
        fit, hold = holdout_indices(pool.sources, pool.labels, pool.years, seed=args.seed)
        indices = fit
        with open(os.path.join(directory, "holdout_ids.json"), "w", encoding="utf-8") as handle:
            json.dump([pool.ids[i] for i in hold], handle)
    elif args.subset:
        keep = set(subset_indices(pool.sources, args.subset, seed=SEED))
        indices = [i for i in indices if i in keep]
    teacher = load_teacher(args.teacher)
    rng = np.random.default_rng(args.seed)
    logger = MetricsLogger(directory, {"enabled": True, "project": "saul-phase4", "name": args.exp})
    groups = _groups(model, args.technique, args.head_lr, adapter_lr, base_lr if args.technique in ("full", "partial") else adapter_lr, args.decay)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    compiled: dict[int, torch.nn.Module] = {}
    best = -1.0
    best_path = os.path.join(directory, "overlay.pt")
    start = time.time()
    tokens_seen = 0
    step = 0
    micro = max(1, args.accum)
    history = []
    try:
        for epoch in range(args.epochs):
            limit = args.first_limit if epoch == 0 and args.first_limit else args.limit
            model.train()
            sources = [pool.sources[i] for i in indices]
            order = sample_indices(sources, rng, ovl_weight=args.ovl_weight, no_ovl=args.no_ovl, refit=args.refit)
            picked = [indices[i] for i in order]
            sequences, labels, ids, lengths = [], [], [], []
            for i in picked:
                seq = make_view(pool.tokens_of(i), args.view, limit, args.head, args.tail, causal)
                sequences.append(seq)
                labels.append(int(pool.labels[i]))
                ids.append(pool.ids[i])
                lengths.append(len(seq))
            batches = make_batches(lengths, args.batch_tokens, rng)
            opt.zero_grad(set_to_none=True)
            running = 0.0
            epoch_f1 = 0.0
            for batch_i, (width, batch_idx) in enumerate(batches):
                batch = collate(
                    [sequences[i] for i in batch_idx], [labels[i] for i in batch_idx],
                    [ids[i] for i in batch_idx], width,
                )
                batch = _move(batch, device)
                soft = _teacher_lookup(teacher, batch["ids"], device)
                module = model
                if args.compile and device == "cuda":
                    module = compiled.get(width)
                    if module is None:
                        module = torch.compile(model)
                        compiled[width] = module
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = module(batch, teacher=soft)["loss"] / micro
                if not torch.isfinite(loss):
                    _fail(directory, "non-finite loss", 2)
                loss.backward()
                running += float(loss.detach()) * micro
                tokens_seen += int(batch["lengths"].sum())
                if (batch_i + 1) % micro == 0 or batch_i + 1 == len(batches):
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                    elapsed_steps = max(step, 1)
                    total_steps = max(1, args.epochs * max(len(batches) // micro, 1))
                    scale = _lr_mult(step, total_steps, 0.05)
                    for group in opt.param_groups:
                        group["lr"] = group.get("base_lr", group["lr"]) * scale if "base_lr" in group else group["lr"]
                    if step == 0:
                        for group in opt.param_groups:
                            group["base_lr"] = group["lr"]
                    for group in opt.param_groups:
                        group["lr"] = group["base_lr"] * scale
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                quarter = max(1, len(batches) // 4)
                if (batch_i + 1) % quarter == 0 or batch_i + 1 == len(batches):
                    spent = max(time.time() - start, 1e-6)
                    ev = _eval_dsel(model, dsel, device, args, limit=limit)
                    score = ev["scores"]["f1"]
                    row = {
                        "epoch": epoch, "step": step, "limit": limit,
                        "loss": running / max(batch_i + 1, 1),
                        "dev_f1": score, "tok_s": tokens_seen / spent, "peak_gb": _peak_gb(),
                        **{f"dev_{k}": v for k, v in ev["scores"].items()},
                    }
                    history.append(row)
                    logger.log(row, step)
                    print(f"[train] {args.exp} epoch {epoch} batch {batch_i + 1}/{len(batches)} f1 {score:.4f}", flush=True)
                    epoch_f1 = score
                    # A short first epoch (G1 fallback) is only a sanity check. The checkpoint
                    # that is kept is one trained at the deployment length.
                    if limit == args.limit and score > best:
                        best = score
                        torch.save(overlay_state(model), best_path)
                        _save_logits(os.path.join(directory, "logits_dsel.npz"), ev["ids"], ev["logits"], dsel.labels)
                    if args.snapshots and limit == args.limit:
                        _save_snapshot(model, test, device, args, directory, step, ev, dsel.labels)
                    model.train()
            if args.gate is not None and epoch == 0 and epoch_f1 < args.gate:
                with open(os.path.join(directory, "gate_failed.json"), "w", encoding="utf-8") as handle:
                    json.dump({"dev_f1": epoch_f1, "gate": args.gate}, handle)
                logger.finish()
                raise SystemExit(3)
    except Exception as exc:
        if _is_oom(exc):
            _fail(directory, f"oom: {exc}", 4)
        raise
    if not os.path.exists(best_path):
        torch.save(overlay_state(model), best_path)
    load_overlay(model, torch.load(best_path, map_location="cpu", weights_only=False))
    model.to(device)
    ev = _eval_dsel(model, dsel, device, args)
    _save_logits(os.path.join(directory, "logits_dsel.npz"), ev["ids"], ev["logits"], dsel.labels)
    test_logits, test_ids = predict_split(
        model, test, device, args.view, args.limit, args.head, args.tail, args.batch_tokens,
    )
    _save_logits(os.path.join(directory, "logits_test.npz"), test_ids, test_logits)
    if args.pool_logits:
        pool_logits, pool_ids = predict_split(
            model, pool, device, args.view, args.limit, args.head, args.tail, args.batch_tokens,
        )
        _save_logits(os.path.join(directory, "logits_pool.npz"), pool_ids, pool_logits, pool.labels)
    snap = average_snapshots(directory, dsel) if args.snapshots else None
    spent = max(time.time() - start, 1e-6)
    blob = {
        "exp": args.exp,
        "model": args.model,
        "technique": args.technique,
        "context": model.context,
        "view": args.view,
        "limit": args.limit,
        "head": args.head,
        "tail": args.tail,
        "epochs": args.epochs,
        "loss": args.loss,
        "ovl_weight": 0 if args.no_ovl else args.ovl_weight,
        "refit": args.refit,
        "subset": args.subset,
        "init": args.init,
        "dev_f1": ev["scores"]["f1"],
        "dev_p": ev["scores"]["precision"],
        "dev_r": ev["scores"]["recall"],
        "dev_f1_a": ev["scores"]["f1_a"],
        "dev_f1_b": ev["scores"]["f1_b"],
        "peak_gb": _peak_gb(),
        "tok_s": tokens_seen / spent,
        "trainable": n_train,
        "hours": spent / 3600,
        "best_f1": best,
    }
    if snap is not None:
        blob["snap"] = snap
    with open(os.path.join(directory, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle)
    with open(done, "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    logger.finish()
    print(f"[train] {args.exp} dev_f1 {blob['dev_f1']:.4f}", flush=True)


def _save_snapshot(model, test, device, args, directory: str, step: int, ev: dict, labels) -> None:
    snaps = os.path.join(directory, "snaps")
    ensure_dir(snaps)
    test_logits, test_ids = predict_split(
        model, test, device, args.view, args.limit, args.head, args.tail, args.batch_tokens,
    )
    tag = f"{step:07d}"
    _save_logits(os.path.join(snaps, f"{tag}_dsel.npz"), ev["ids"], ev["logits"], labels)
    _save_logits(os.path.join(snaps, f"{tag}_test.npz"), test_ids, test_logits)


def average_snapshots(directory: str, dsel) -> dict | None:
    """Mean raw logits of every snapshot from the second half of training.

    The rule is fixed before any score is seen, so the average is not a pick
    over D-sel evaluations. Results go to the `<exp>_snap` experiment folder.
    """
    snaps = os.path.join(directory, "snaps")
    if not os.path.isdir(snaps):
        return None
    steps = sorted({int(name.split("_")[0]) for name in os.listdir(snaps) if name.endswith("_dsel.npz")})
    if not steps:
        return None
    keep = [s for s in steps if s >= 0.5 * steps[-1]]
    out_dir = directory.rstrip("/") + "_snap"
    ensure_dir(out_dir)
    for kind in ("dsel", "test"):
        blobs = [np.load(os.path.join(snaps, f"{s:07d}_{kind}.npz"), allow_pickle=True) for s in keep]
        ids = [str(i) for i in blobs[0]["ids"].tolist()]
        for blob in blobs[1:]:
            if [str(i) for i in blob["ids"].tolist()] != ids:
                raise RuntimeError("snapshot id order changed")
        mean = np.mean([np.asarray(b["logits"], dtype=np.float32) for b in blobs], axis=0)
        _save_logits(os.path.join(out_dir, f"logits_{kind}.npz"), ids, mean,
                     dsel.labels if kind == "dsel" else None)
    blob = np.load(os.path.join(out_dir, "logits_dsel.npz"), allow_pickle=True)
    pred = predict_logits(np.asarray(blob["logits"]), 0.0)
    halves = np.asarray(dsel.halves)
    scores = macro_scores(dsel.labels, pred)
    result = {
        "steps": keep,
        "dev_f1": scores["f1"],
        "dev_p": scores["precision"],
        "dev_r": scores["recall"],
        "dev_f1_a": macro_scores(dsel.labels[halves == "A"], pred[halves == "A"])["f1"],
        "dev_f1_b": macro_scores(dsel.labels[halves == "B"], pred[halves == "B"])["f1"],
    }
    with open(os.path.join(out_dir, "done.json"), "w", encoding="utf-8") as handle:
        json.dump({"exp": os.path.basename(out_dir), "loss": "logit_adjust", "source": os.path.basename(directory), **result}, handle, indent=2)
    print(f"[train] snapshot mean of {len(keep)} dev_f1 {result['dev_f1']:.4f}", flush=True)
    return result


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _fail(directory: str, reason: str, code: int) -> None:
    with open(os.path.join(directory, "failed.json"), "w", encoding="utf-8") as handle:
        json.dump({"reason": reason}, handle)
    print(f"[train] {reason}", flush=True)
    raise SystemExit(code)


def _smoke(model, device: str, args, directory: str, n_train: int) -> None:
    model.train()
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-4)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last = None
    width = min(args.limit, 256)
    for _ in range(args.steps):
        batch = synthetic_batch(width, 2, int(model.backbone.config.vocab_size), device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            last = model(batch)["loss"]
        if not torch.isfinite(last):
            _fail(directory, "non-finite smoke loss", 2)
        opt.zero_grad(set_to_none=True)
        last.backward()
        opt.step()
    blob = {
        "exp": args.exp, "smoke": True, "finite": True, "loss": float(last.detach().float()),
        "peak_gb": _peak_gb(), "trainable": n_train, "context": model.context, "model": args.model,
    }
    with open(os.path.join(directory, "done.json"), "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    print(f"[smoke] {blob}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--technique", default="lorma")
    parser.add_argument("--context", default=None)
    parser.add_argument("--view", default="tail", choices=["tail", "head", "head_tail"])
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--first-limit", type=int, default=0)
    parser.add_argument("--head", type=int, default=512)
    parser.add_argument("--tail", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--ovl-weight", type=int, default=2)
    parser.add_argument("--no-ovl", action="store_true")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--init", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--loss", default="logit_adjust", choices=["logit_adjust", "ce", "kd"])
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=2)
    parser.add_argument("--attn", default=None)
    parser.add_argument("--gate", type=float, default=None)
    parser.add_argument("--pool-logits", action="store_true", default=True)
    parser.add_argument("--no-pool-logits", action="store_false", dest="pool_logits")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--snapshots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from saul.cjpe import p4_dir
    if args.bench_dir is None:
        args.bench_dir = os.path.join(p4_dir(), "bench")
    if args.gate is None and args.exp == "A1":
        args.gate = A1_GATE
    try:
        train_one(args)
    except SystemExit:
        raise
    except Exception as exc:
        if _is_oom(exc):
            _fail(exp_dir(args.exp), f"oom: {exc}", 4)
        raise
    sys.stdout.flush()


if __name__ == "__main__":
    main()
