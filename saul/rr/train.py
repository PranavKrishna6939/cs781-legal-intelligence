"""Fine-tune one Phase-1 checkpoint for rhetorical roles.

The best epoch is the one with the highest fold-dev macro-F1 on window-averaged
marginals (argmax, no transition bias). Test marginals are cached after that
checkpoint is reloaded, and done.json is written last.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from saul.ner.registry import CONTEXT
from saul.rr.data import (
    SENT_ID, class_weights, collate, doc_windows, ensure_folds, epoch_windows, labeled_docs, load_docs,
    p3_dir,
)
from saul.rr.infer import accumulate, merge_test, write_cache
from saul.rr.model import RRModel, load_overlay, load_rr, overlay_state, trainable_parameters
from saul.rr.score import macro_scores
from saul.train.ema import TailEMA
from saul.utils.io import MetricsLogger, ensure_dir


def _groups(model: RRModel, technique: str, head_lr: float, adapter_lr: float, base_lr: float, decay: float):
    groups = []
    head = [p for p in model.heads.parameters() if p.requires_grad]
    if head:
        groups.append({"params": head, "lr": head_lr})
    peft = technique in ("lorma", "lora", "lorma_pi", "adapter_bitfit", "ia3")
    if peft or technique == "probe":
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
        else:
            lr = base_lr * (decay ** n)
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


def _default_batch(context: str) -> tuple[int, int]:
    if context == "echo":
        return 2, 8
    return 8, 2


def _predict(packed, docs) -> tuple[list[int], list[int]]:
    gold, pred = [], []
    by_id = {d.doc_id: d for d in docs}
    for doc_id, (mass, weight) in packed.items():
        doc = by_id[doc_id]
        if doc.labels is None:
            continue
        denom = np.maximum(weight, 1e-8)[:, None]
        probs = mass / denom
        pred.extend(probs.argmax(1).tolist())
        gold.extend(doc.label_ids)
    return gold, pred


def _check_markers(windows) -> None:
    bad = 0
    for window in windows:
        for pos in window.marker_pos:
            if pos < 0 or pos >= len(window.input_ids) or window.input_ids[pos] != SENT_ID:
                bad += 1
    if bad:
        raise SystemExit(f"[rr] {bad} marker positions are not token id {SENT_ID}")


def _attach_teacher(windows, table: dict[str, np.ndarray] | None):
    if not table:
        return windows
    out = []
    for window in windows:
        probs = table.get(window.doc_id)
        if probs is None:
            out.append(window)
            continue
        rows = np.stack([probs[i] for i in window.sent_index])
        window.teacher = rows
        out.append(window)
    return out


def _load_teacher_table(directory: str | None) -> dict[str, np.ndarray] | None:
    if not directory or not os.path.isdir(directory):
        return None
    from saul.rr.infer import load_probs
    table = {}
    for name in os.listdir(directory):
        if name.endswith(".npz"):
            table[name[:-4]] = load_probs(os.path.join(directory, name))
    return table or None


def train_one(exp: str, model_name: str, technique: str, fold: int, context: str | None = None,
              epochs: int = 10, batch_size: int | None = None, accum: int | None = None,
              head_lr: float = 1e-3, adapter_lr: float | None = None, base_lr: float = 3e-5,
              decay: float = 0.85, init: str | None = None, seed: int = 781,
              sentence_only: bool = False, window: int | None = None, full_data: bool = False,
              g1_min: float | None = None, teacher_dir: str | None = None, kd_weight: float = 0.0,
              pool_jsonl: str | None = None, silver_jsonl: str | None = None, silver_epochs: int = 1,
              smoke: bool = False, device: str | None = None) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    context = context or CONTEXT[model_name]
    if batch_size is None or accum is None:
        default_bs, default_acc = _default_batch(context)
        batch_size = default_bs if batch_size is None else batch_size
        accum = default_acc if accum is None else accum
    if adapter_lr is None:
        adapter_lr = 3e-3 if technique == "ia3" else 5e-4
    run = os.path.join(p3_dir(), exp, "full" if full_data else f"fold{fold}")
    if os.path.exists(os.path.join(run, "done.json")):
        print(f"[skip] {run} already done")
        return run
    ensure_dir(run)
    docs = load_docs()
    labeled = labeled_docs(docs)
    folds = ensure_folds(labeled)
    by_id = {d.doc_id: d for d in labeled}
    if full_data:
        train_docs = labeled
        dev_docs = []
    else:
        dev_ids = set(folds[fold])
        dev_docs = [by_id[i] for i in folds[fold]]
        train_docs = [d for d in labeled if d.doc_id not in dev_ids]
    test_docs = [d for d in docs if d.split == "test"]
    tok_windows = doc_windows(train_docs, context=context, sentence_only=sentence_only, max_tokens=window)
    dev_windows = doc_windows(dev_docs, context=context, sentence_only=sentence_only, max_tokens=window) if dev_docs else []
    test_windows = doc_windows(test_docs, context=context, sentence_only=sentence_only, max_tokens=window)
    _check_markers(tok_windows + dev_windows + test_windows)
    teacher = _load_teacher_table(teacher_dir)
    tok_windows = _attach_teacher(tok_windows, teacher)
    pool_windows = []
    if pool_jsonl:
        from saul.rr.kd import pool_windows_from
        pool_windows = _attach_teacher(pool_windows_from(pool_jsonl, context, window, sentence_only), teacher)
        pool_windows = [w for w in pool_windows if w.teacher is not None]
    silver_windows = []
    if silver_jsonl:
        from saul.rr.kd import silver_windows_from
        # Fold k's silver comes from fold k's teachers, which never saw fold k's dev documents.
        silver_windows = silver_windows_from(silver_jsonl.format(fold=fold), context, window, sentence_only)
    # The pool is ~11x the gold set. Each pool pass samples at most this many windows,
    # so one KD epoch costs about two gold epochs instead of twelve.
    pool_cap = min(len(pool_windows), len(tok_windows))
    silver_cap = min(len(silver_windows), 3 * len(tok_windows))

    model = load_rr(model_name, technique, context=context, init=init, device=device)
    model.kd_weight = kd_weight if teacher else 0.0
    model.heads.set_weights(class_weights(train_docs).to(device))
    groups = _groups(model, technique, head_lr, adapter_lr, base_lr, decay)
    if not groups:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(seed + fold)
    rare_n = sum(1 for w in tok_windows if w.rare)
    steps_per = math.ceil(max(len(tok_windows) + rare_n, 1) / batch_size)
    pool_steps = math.ceil(pool_cap / batch_size) if pool_cap else 0
    # Silver is a prefix stage. The gold recipe keeps its own cosine and TailEMA.
    total_steps = max(1, epochs * (steps_per + pool_steps))
    ema_from = int(total_steps * 0.8)
    ema = TailEMA(model, decay=0.999)
    logger = MetricsLogger(run, {"enabled": True, "project": "saul-phase3", "name": f"{exp}-fold{fold}", "group": exp})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {exp} fold {fold} {model_name} {technique} {context} trainable={n_train} "
          f"windows={len(tok_windows)} device={device}", flush=True)
    best = -1.0
    best_blob = None
    step = 0
    model.train()
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    seen_sent = 0

    def _loss(chunk, kd_only: bool = False):
        batch = {k: v.to(device) for k, v in collate(chunk).items()}
        if kd_only:
            batch["kd_only"] = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            return model(batch)

    def _sample(source, cap: int):
        if cap >= len(source):
            return source
        pick = rng.choice(len(source), size=cap, replace=False)
        return [source[int(i)] for i in pick]

    def _epoch(epoch: int, source, rare: bool, do_dev: bool, advance_schedule: bool = True,
               kd_only: bool = False) -> float | None:
        nonlocal step, best, best_blob, seen_sent
        order = epoch_windows(source, rng, rare_boost=rare)
        running = 0.0
        seen = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        def _apply_step(out_loss) -> None:
            nonlocal step
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
            mult = _lr_mult(step, total_steps, 0.06) if advance_schedule else 1.0
            for group, lr0 in zip(opt.param_groups, base_lrs):
                group["lr"] = lr0 * mult
            opt.step()
            opt.zero_grad(set_to_none=True)
            if not advance_schedule:
                return
            if step >= ema_from:
                ema.update(model)
            if step % 20 == 0:
                rec = {"step": step, "loss": float(out_loss.detach()), "nll": float(out["nll"]),
                       "ce": float(out["ce"]), "lsp": float(out["lsp"])}
                with open(os.path.join(run, "metrics.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            step += 1

        for start_i in range(0, len(order), batch_size):
            chunk = order[start_i:start_i + batch_size]
            if kd_only and not any(w.teacher is not None for w in chunk):
                continue
            out = _loss(chunk, kd_only=kd_only)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise SystemExit("[rr] non-finite loss")
            (loss / accum).backward()
            seen += 1
            running += float(loss.detach())
            seen_sent += sum(len(w.sent_index) for w in chunk)
            if seen % accum != 0:
                continue
            _apply_step(loss)
        if seen % accum != 0:
            _apply_step(loss)
        avg = running / max(seen, 1)
        dev = None
        if do_dev and dev_windows:
            dev = _dev(model, dev_docs, dev_windows, batch_size, device)
            if dev > best:
                best = dev
                best_blob = overlay_state(model)
            if ema.updates:
                backup = ema.copy_to(model)
                ema_dev = _dev(model, dev_docs, dev_windows, batch_size, device)
                if ema_dev > best:
                    best = ema_dev
                    best_blob = overlay_state(model)
                ema.restore(model, backup)
                dev = max(dev, ema_dev)
        logger.log({"epoch": epoch, "loss": avg, "dev_f1": -1 if dev is None else dev,
                    "seconds": time.time() - t0}, step)
        print(f"[epoch] {epoch} loss={avg:.4f} dev={dev} best={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if smoke and epoch == 0 and (not math.isfinite(avg) or not (0.1 <= avg <= 20.0)):
            raise SystemExit(f"[smoke] loss {avg} is outside 0.1..20")
        # One check at the logged epoch 2. Stop only if no epoch so far has cleared the bar.
        if g1_min is not None and fold == 0 and epoch == 2 and dev is not None and best < g1_min:
            with open(os.path.join(run, "g1_failed.json"), "w", encoding="utf-8") as f:
                json.dump({"dev": dev, "best": best, "epoch": epoch, "min": g1_min}, f)
            raise SystemExit(f"[g1] best {best:.4f} < {g1_min} after epoch {epoch}")
        return dev

    if silver_windows and silver_epochs > 0:
        for epoch in range(silver_epochs):
            _epoch(epoch, _sample(silver_windows, silver_cap), rare=False, do_dev=False, advance_schedule=False)
    for epoch in range(epochs):
        _epoch(epoch, tok_windows, rare=True, do_dev=not full_data)
        if pool_cap:
            _epoch(epoch, _sample(pool_windows, pool_cap), rare=False, do_dev=False, kd_only=True)
    if best_blob is None:
        best_blob = overlay_state(model)
    torch.save(best_blob, os.path.join(run, "best.pt"))
    load_overlay(model, best_blob)
    final = None
    if dev_docs:
        packed = accumulate(model, dev_windows, {d.doc_id: len(d.sentences) for d in dev_docs}, batch_size, device)
        write_cache(os.path.join(run, "oof"), packed)
        gold, pred = _predict(packed, dev_docs)
        final = macro_scores(gold, pred)["macro_f1"]
    test_packed = accumulate(model, test_windows, {d.doc_id: len(d.sentences) for d in test_docs}, batch_size, device)
    write_cache(os.path.join(run, "test"), test_packed)
    minutes = (time.time() - t_start) / 60.0
    peak = 0.0
    if use_cuda:
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    rate = seen_sent / max(time.time() - t_start, 1e-6)
    # done.json has to exist before the merge. The merge requires every fold's
    # done.json, and this fold is the last one that can satisfy that check.
    with open(os.path.join(run, "done.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exp": exp, "fold": fold, "model": model_name, "technique": technique, "context": context,
            "window": window or (1024 if context == "echo" else 2048), "sentence_only": sentence_only,
            "dev_f1": final, "best_seen": best, "trainable": n_train, "peak_gb": peak,
            "minutes": minutes, "sentences_per_s": rate, "seed": seed, "full_data": full_data,
            "init": init,
        }, f, indent=2)
    _maybe_merge_test(exp, test_docs)
    print(f"[done] {run} dev_f1={final} peak_gb={peak:.1f}", flush=True)
    logger.finish()
    return run


def _dev(model, docs, windows, batch_size, device) -> float:
    packed = accumulate(model, windows, {d.doc_id: len(d.sentences) for d in docs}, batch_size, device)
    gold, pred = _predict(packed, docs)
    return macro_scores(gold, pred)["macro_f1"]


def _maybe_merge_test(exp: str, test_docs) -> None:
    root = os.path.join(p3_dir(), exp)
    fold_dirs = []
    for k in range(5):
        path = os.path.join(root, f"fold{k}", "test")
        done = os.path.join(root, f"fold{k}", "done.json")
        if not (os.path.isdir(path) and os.path.exists(done)):
            return
        fold_dirs.append(path)
    dest = os.path.join(root, "test")
    merge_test(fold_dirs, dest, [d.doc_id for d in test_docs])
    with open(os.path.join(dest, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": 5}, f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--technique", required=True)
    p.add_argument("--context", default=None)
    p.add_argument("--fold", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--adapter-lr", type=float, default=None)
    p.add_argument("--init", default=None)
    p.add_argument("--seed", type=int, default=781)
    p.add_argument("--sentence-only", action="store_true")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--g1-min", type=float, default=None)
    p.add_argument("--teacher-dir", default=None)
    p.add_argument("--kd-weight", type=float, default=0.0)
    p.add_argument("--pool", default=None)
    p.add_argument("--silver", default=None)
    p.add_argument("--silver-epochs", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    full = args.fold == "full"
    folds = [0] if full else list(range(5)) if args.fold == "all" else [int(x) for x in str(args.fold).split(",")]
    epochs = 1 if args.smoke and args.epochs == 10 else args.epochs
    for fold in folds:
        train_one(
            args.exp, args.model, args.technique, fold, context=args.context, epochs=epochs,
            batch_size=args.batch_size, accum=args.accum, head_lr=args.head_lr, adapter_lr=args.adapter_lr,
            init=args.init, seed=args.seed, sentence_only=args.sentence_only, window=args.window,
            full_data=full, g1_min=args.g1_min, teacher_dir=args.teacher_dir, kd_weight=args.kd_weight,
            pool_jsonl=args.pool, silver_jsonl=args.silver, silver_epochs=args.silver_epochs,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
