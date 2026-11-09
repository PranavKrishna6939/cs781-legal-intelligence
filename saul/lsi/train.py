"""Fine-tune one Phase-1 backbone for statute identification."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import CHUNK, MAX_CHUNKS, N_LABELS, SEED
from saul.lsi.data import (
    case_record, collate, dev_half_array, load_token_split, make_batches, multi_hot, rare_weights,
    resample_record, subset_indices,
)
from saul.lsi.decide import quick_macro
from saul.lsi.model import load_lsi, load_overlay, lr_mult, overlay_state, param_groups, prune_adalora, trainable_parameters


def _n_trainable(model) -> int:
    return int(sum(param.numel() for param in trainable_parameters(model)))
from saul.lsi.score import micro_f1, scores, subset_accuracy
from saul.utils.io import MetricsLogger, ensure_dir


def _move(batch: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


VIEW = {"view": "chunks", "head": 766, "tail": 256}


def _records(split: dict, indexes: np.ndarray, limit: int, max_chunks: int, train: bool = False) -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [case_record(split, int(i), limit=limit, max_chunks=max_chunks, train=train, rng=rng, **VIEW)
            for i in indexes]


def _attach_side(records: list[dict], indexes: np.ndarray, split: str, lexical: bool, retrieval: bool, k: int) -> None:
    if not lexical and not retrieval:
        return
    feats = os.path.join(p5_dir(), "feats")
    rows = np.asarray(indexes, dtype=np.int64)
    zlex = np.load(os.path.join(feats, f"zlex_{split}.npy"), mmap_mode="r") if lexical else None
    if retrieval:
        # Load once. Indexing an npz member inside the loop returns a view of a freshly
        # decompressed array, and each record would keep that whole array alive.
        with np.load(os.path.join(feats, f"knn_{split}.npz")) as knn:
            neighbour_index = np.array(knn["idx"])
            neighbour_sim = np.array(knn["sim"])
        neighbour_labels = np.load(os.path.join(feats, "knn_labels.npy"), mmap_mode="r")
        chosen = neighbour_index[rows, :k]
        label_rows = np.array(neighbour_labels[chosen], dtype=np.float32)
        sim_rows = np.array(neighbour_sim[rows, :k], dtype=np.float32)
    else:
        label_rows = sim_rows = None
    zlex_rows = np.array(zlex[rows], dtype=np.float32) if zlex is not None else None
    for position, record in enumerate(records):
        if zlex_rows is not None:
            record["zlex"] = zlex_rows[position]
        if label_rows is not None:
            record["nbr_labels"] = label_rows[position]
            record["nbr_sim"] = sim_rows[position]


class _EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.ready = False
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }

    def update(self, model) -> None:
        self.ready = True
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self, model) -> dict:
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup: dict) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])


def _save_logits(path: str, ids: list[str], logits: np.ndarray, labels: np.ndarray | None = None) -> None:
    payload = {"ids": np.asarray(ids), "logits": logits.astype(np.float32)}
    if labels is not None:
        payload["labels"] = labels.astype(np.float32)
    np.savez(path, **payload)


def predict_records(model, records: list[dict], device: str, batch_tokens: int,
                    sink: dict | None = None, states_path: str | None = None) -> tuple[list[str], np.ndarray]:
    model.eval()
    lengths = [len(record["chunks"]) for record in records]
    widths = [record["width"] for record in records]
    ids: list[str] = []
    rows = []
    deep_rows = []
    states_mm = None
    cursor = 0
    if states_path is not None:
        if not hasattr(model.head, "label_emb"):
            raise RuntimeError("--train-states needs the decoder head")
        shape = (len(records), int(model.head.n_labels), int(model.head.label_emb.shape[-1]))
        states_mm = np.memmap(states_path, dtype=np.float16, mode="w+", shape=shape)
    use_cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for width, indexes in make_batches(lengths, widths, batch_tokens):
            batch = _move(collate([records[i] for i in indexes], width), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                out = model(batch)
            logits = out["logits"].float().cpu().numpy()
            deep = out["logits_deep"].float().cpu().numpy() if "logits_deep" in out else None
            states = out["label_states"].detach().float().cpu().numpy() if states_mm is not None else None
            for local, index in enumerate(indexes):
                ids.append(records[index]["id"])
                rows.append(logits[local])
                if deep is not None:
                    deep_rows.append(deep[local])
                if states is not None:
                    states_mm[cursor] = states[local]
                    cursor += 1
    if states_mm is not None:
        states_mm.flush()
        with open(states_path + ".json", "w", encoding="utf-8") as handle:
            json.dump({"shape": list(states_mm.shape), "ids": ids}, handle)
    if sink is not None and deep_rows:
        sink["ids"] = list(ids)
        sink["deep"] = np.stack(deep_rows)
    return ids, np.stack(rows)


def _align(ids: list[str], matrix: np.ndarray, records: list[dict]) -> np.ndarray:
    order = {case_id: i for i, case_id in enumerate(ids)}
    return matrix[[order[record["id"]] for record in records]]


def _eval_dev(model, dev_records, dev_labels, half, device, batch_tokens) -> dict:
    sink: dict = {}
    ids, logits = predict_records(model, dev_records, device, batch_tokens, sink=sink)
    logits = _align(ids, logits, dev_records)
    scored = quick_macro(logits, dev_labels, half)
    pred = scored["pred"]
    result = {
        "macro_f1": scored["macro_f1"],
        "half": scored["half"],
        "micro_f1": micro_f1(dev_labels, pred),
        "subset_accuracy": subset_accuracy(dev_labels, pred),
        "logits": logits,
    }
    if "deep" in sink:
        deep = _align(sink["ids"], sink["deep"], dev_records)
        result["deep_macro_f1"] = quick_macro(deep, dev_labels, half)["macro_f1"]
        result["deep_logits"] = deep
    return result


def synthetic_batch(width: int, n_cases: int, chunks_each: int, vocab: int, device: str) -> dict:
    records = []
    for case in range(n_cases):
        chunks = [list(np.random.randint(4, vocab, size=width)) for _ in range(chunks_each)]
        records.append({"id": str(case), "labels": [case % N_LABELS], "chunks": chunks, "width": width})
    return _move(collate(records, width), device)


def run_bench(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lsi(args.model, args.technique, args.context, attn=args.attn, ckpt_every=args.ckpt_every,
                     head=args.head, loss=args.loss, device=device)
    model.train()
    width = args.limit
    batch = synthetic_batch(width, n_cases=2, chunks_each=2, vocab=32000, device=device)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-4)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        out = model(batch)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    tokens = int((batch["doc_ids"] >= 0).sum())
    start = time.time()
    steps = 5
    for _ in range(steps):
        out = model(batch)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    elapsed = max(time.time() - start, 1e-6)
    result = {
        "model": args.model, "width": width, "attn": args.attn, "ok": True,
        "tok_s": tokens * steps / elapsed, "peak_gb": _peak_gb(),
        "loss": float(out["loss"].detach()), "finite": bool(torch.isfinite(out["loss"])),
    }
    directory = os.path.join(p5_dir(), "bench")
    ensure_dir(directory)
    path = os.path.join(directory, f"bench_{args.model}_{width}_{args.attn or 'auto'}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    print(json.dumps(result), flush=True)
    return result


def run_steps(model, batches: list[dict], opt, device: str, technique: str = "full") -> float:
    """A few optimiser steps. Used by the CPU smoke test and by --smoke."""
    model.train()
    last = 0.0
    for step, batch in enumerate(batches):
        moved = _move(batch, device) if "cpu" not in str(next(model.parameters()).device) else batch
        out = model(moved if torch.is_tensor(batch["input_ids"]) and batch["input_ids"].device.type == device else _move(batch, device))
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise SystemExit("[train] non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in trainable_parameters(model):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        opt.step()
        last = float(loss.detach())
        if technique == "adalora" and step == max(1, len(batches) // 2):
            prune_adalora(model.backbone, keep=8)
    return last


def _fail(directory: str, reason: str, code: int) -> None:
    ensure_dir(directory)
    with open(os.path.join(directory, "failed.json"), "w", encoding="utf-8") as handle:
        json.dump({"reason": reason}, handle)
    raise SystemExit(code)


def train_one(args) -> None:
    directory = exp_dir(args.exp)
    if os.path.exists(os.path.join(directory, "done.json")) and not args.force:
        print(f"[train] reuse {directory}", flush=True)
        return
    ensure_dir(directory)
    VIEW.update(view=args.view, head=args.head_tokens, tail=args.tail_tokens)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = p5_dir()
    train_split = load_token_split("train", os.path.join(tok, "tok"))
    dev_split = load_token_split("dev", os.path.join(tok, "tok"))
    test_split = load_token_split("test", os.path.join(tok, "tok"))
    stats = json.load(open(os.path.join(tok, "label_stats.json"), encoding="utf-8"))
    prior = torch.tensor(stats["freq"], dtype=torch.float32) / max(stats["n_train"], 1)
    dev_ids = [row["id"] for row in dev_split["meta"]]
    half = dev_half_array(dev_ids, tok)
    train_index = np.arange(len(train_split["meta"]))
    if args.subset:
        train_index = subset_indices(len(train_index), args.subset, args.seed)
    if args.cf2_fold:
        extra = np.flatnonzero(half == args.cf2_fold) + len(train_split["meta"])
        train_split_fit = _concat_splits(train_split, dev_split)
        train_index = np.concatenate([train_index, extra])
        eval_dev = np.flatnonzero(half != args.cf2_fold)
    else:
        train_split_fit = train_split
        eval_dev = np.arange(len(dev_split["meta"]))
    if args.cf2_fold and (args.lexical or args.retrieval):
        _fail(directory, "cf2 fold is not combined with lexical or retrieval inputs", 2)
    infer_chunks = 1 if args.max_chunks_infer else (args.infer_chunks or args.max_chunks)
    train_records = _records(train_split_fit, train_index, args.limit, args.max_chunks, train=args.view == "segments")
    dev_records = _records(dev_split, eval_dev, args.limit, infer_chunks)
    _attach_side(train_records, train_index, "train", args.lexical, args.retrieval, args.knn)
    _attach_side(dev_records, eval_dev, "dev", args.lexical, args.retrieval, args.knn)
    dev_labels = np.stack([multi_hot(dev_split["meta"][int(i)]["labels"]) for i in eval_dev])
    eval_half = half[eval_dev] if not args.cf2_fold else _monitor_half(len(eval_dev), args.seed)
    freq = np.asarray(stats["freq"])
    weights = None
    if args.sampling == "sqrt":
        weights = rare_weights([record["labels"] for record in train_records], freq)
    cooc = None
    if args.cooc:
        cooc = torch.tensor(np.load(os.path.join(tok, "pair_conditional.npy")), dtype=torch.float32)
    model = load_lsi(
        args.model, args.technique, args.context, init=args.init, head=args.head, loss=args.loss,
        device=device, ckpt_every=args.ckpt_every, attn=args.attn, prior=prior,
        compile_blocks=args.compile_blocks and device == "cuda", cooc=cooc,
    )
    if args.init_from:
        load_overlay(model, torch.load(args.init_from, map_location="cpu", weights_only=False))
        print(f"[train] warm start {args.init_from}", flush=True)
    model.label_freq = torch.tensor(stats["freq"], dtype=torch.float32, device=device)
    model.deep_sup = args.deep_sup
    model.db_fallback = False
    if hasattr(model.head, "wide_drop"):
        model.head.wide_drop = args.wide_dropout
    if args.compile and device == "cuda":
        model.chunk_hidden = torch.compile(model.chunk_hidden)
    if args.loss == "pair":
        model.pair_conditional = torch.tensor(
            np.load(os.path.join(tok, "pair_conditional.npy")), dtype=torch.float32, device=device,
        )
    teacher_table = None
    teacher_states = None
    teacher_state_index = None
    if args.teacher:
        teacher_blob = np.load(args.teacher)
        payload = teacher_blob["logits"] if "logits" in teacher_blob.files else teacher_blob["probs"]
        teacher_table = {str(i): row for i, row in zip(teacher_blob["ids"], payload)}
    if args.teacher_states:
        with open(args.teacher_states + ".json", encoding="utf-8") as handle:
            state_meta = json.load(handle)
        teacher_states = np.memmap(args.teacher_states, dtype=np.float16, mode="r", shape=tuple(state_meta["shape"]))
        teacher_state_index = {str(case_id): row for row, case_id in enumerate(state_meta["ids"])}
    groups = param_groups(
        model, args.technique, args.head_lr, args.adapter_lr, args.base_lr, args.decay, embed_lr=args.embed_lr,
    )
    opt = torch.optim.AdamW(groups, weight_decay=0.01, fused=device == "cuda")
    base_lrs = [group["lr"] for group in opt.param_groups]
    logger = MetricsLogger(directory, {"enabled": not args.no_wandb, "project": "saul-phase5", "name": args.exp})
    lengths = [len(record["chunks"]) for record in train_records]
    widths = [record["width"] for record in train_records]
    best = -1.0
    best_blob = None
    history = []
    start = time.time()
    tokens_seen = 0
    steps_per_epoch = max(1, len(make_batches(lengths, widths, args.batch_tokens)))
    total_steps = steps_per_epoch * args.epochs
    accum = max(args.accum, 1)
    opt_total = max(1, (total_steps + accum - 1) // accum)
    step = 0
    opt_step = 0
    ema = _EMA(model, args.ema) if args.ema > 0 else None
    warm_steps = max(1, int(opt_total * args.warmup))
    try:
        for epoch in range(args.epochs):
            rng = np.random.default_rng(args.seed + epoch)
            if args.view == "segments":
                for record in train_records:
                    resample_record(record, rng)
            batches = make_batches(lengths, widths, args.batch_tokens, rng, weights)
            model.train()
            for batch_i, (width, indexes) in enumerate(batches):
                batch = _move(collate([train_records[i] for i in indexes], width), device)
                teacher = None
                if teacher_table is not None:
                    missing = [case_id for case_id in batch["ids"] if case_id not in teacher_table]
                    if missing:
                        _fail(directory, f"teacher missing {len(missing)} train ids", 2)
                    teacher = torch.tensor(
                        np.stack([teacher_table[case_id] for case_id in batch["ids"]]),
                        dtype=torch.float32, device=device,
                    )
                if teacher_states is not None:
                    missing = [case_id for case_id in batch["ids"] if case_id not in teacher_state_index]
                    if missing:
                        _fail(directory, f"teacher states missing {len(missing)} train ids", 2)
                    rows = [teacher_state_index[case_id] for case_id in batch["ids"]]
                    batch["teacher_states"] = torch.from_numpy(np.asarray(teacher_states[rows])).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = model(batch, teacher=teacher)
                loss_value = float(out["loss"].detach())
                if (not math.isfinite(loss_value)) or loss_value > 50:
                    if args.loss == "db" and not model.db_fallback:
                        print("[train] DB-NTR unstable, switching to ASL", flush=True)
                        model.db_fallback = True
                        opt.zero_grad(set_to_none=True)
                        continue
                    _fail(directory, "non-finite loss", 2)
                (out["loss"] / accum).backward()
                if (step + 1) % accum == 0:
                    for param in trainable_parameters(model):
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                    mult = lr_mult(opt_step, opt_total, args.warmup, args.lr_floor)
                    for group, base in zip(opt.param_groups, base_lrs):
                        group["lr"] = base * mult
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    opt_step += 1
                    if ema is not None and opt_step >= warm_steps:
                        ema.update(model)
                tokens_seen += int((batch["doc_ids"] >= 0).sum())
                step += 1
                if args.technique == "adalora" and step == total_steps // 2:
                    prune_adalora(model.backbone, keep=8)
                halfway = batch_i + 1 == max(1, len(batches) // 2) or batch_i + 1 == len(batches)
                if args.eval_per_epoch:
                    halfway = batch_i + 1 == len(batches)
                if batch_i % 200 == 0:
                    rate = tokens_seen / max(time.time() - start, 1e-6)
                    print(f"[train] {args.exp} epoch {epoch} batch {batch_i}/{len(batches)} "
                          f"loss {loss_value:.4f} tok/s {rate:.0f}", flush=True)
                if halfway:
                    backup = ema.apply(model) if ema is not None and ema.ready else None
                    scored = _eval_dev(model, dev_records, dev_labels, eval_half, device, args.batch_tokens)
                    if backup is not None:
                        ema.restore(model, backup)
                    model.train()
                    row = {"epoch": epoch, "step": step, "macro_f1": scored["macro_f1"], "half": scored["half"],
                           "micro_f1": scored["micro_f1"], "subset_accuracy": scored["subset_accuracy"]}
                    if "deep_macro_f1" in scored:
                        row["deep_macro_f1"] = scored["deep_macro_f1"]
                    history.append(row)
                    logger.log({"dev_macro_f1": scored["macro_f1"], "loss": loss_value}, step)
                    deep_note = f" deep {scored['deep_macro_f1']:.4f}" if "deep_macro_f1" in scored else ""
                    print(f"[train] {args.exp} epoch {epoch} macro {scored['macro_f1']:.4f}{deep_note}", flush=True)
                    if scored["macro_f1"] > best:
                        best = scored["macro_f1"]
                        saved = ema.apply(model) if ema is not None and ema.ready else None
                        best_blob = overlay_state(model)
                        if saved is not None:
                            ema.restore(model, saved)
                        torch.save(best_blob, os.path.join(directory, "best.pt"))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"[train] oom: {exc}", flush=True)
            _fail(directory, f"oom: {exc}", 4)
        raise
    if args.gate and best < args.gate:
        with open(os.path.join(directory, "gate_failed.json"), "w", encoding="utf-8") as handle:
            json.dump({"best": best, "gate": args.gate}, handle)
        raise SystemExit(3)
    if best_blob is not None:
        load_overlay(model, best_blob)
    dev_sink: dict = {}
    dev_ids_out, dev_logits = predict_records(model, dev_records, device, args.batch_tokens, sink=dev_sink)
    test_ids, test_logits, test_sink = [], None, {}
    if not args.skip_test:
        test_index = np.arange(len(test_split["meta"]))
        test_records = _records(test_split, test_index, args.limit, infer_chunks)
        _attach_side(test_records, test_index, "test", args.lexical, args.retrieval, args.knn)
        test_ids, test_logits = predict_records(model, test_records, device, args.batch_tokens, sink=test_sink)
    dev_label = {record["id"]: record["labels"] for record in dev_records}
    dev_labels_out = np.stack([multi_hot(dev_label[case_id]) for case_id in dev_ids_out])
    kind = "dev" if not args.cf2_fold else f"dev_{args.cf2_fold}_oof"
    _save_logits(os.path.join(directory, f"{kind}_logits.npz"), dev_ids_out, dev_logits, dev_labels_out)
    if test_logits is not None:
        _save_logits(os.path.join(directory, "test_logits.npz"), test_ids, test_logits)
    if "deep" in dev_sink:
        _save_logits(os.path.join(directory, f"{kind}_logits_deep.npz"), dev_sink["ids"], dev_sink["deep"], dev_labels_out)
    if test_sink.get("deep") is not None:
        _save_logits(os.path.join(directory, "test_logits_deep.npz"), test_sink["ids"], test_sink["deep"])
    if args.train_logits or args.train_states:
        states_path = os.path.join(directory, "train_states.f16") if args.train_states else None
        tr_ids, tr_logits = predict_records(model, train_records, device, args.batch_tokens, states_path=states_path)
        if args.train_logits:
            train_label = {record["id"]: record["labels"] for record in train_records}
            tr_labels = np.stack([multi_hot(train_label[case_id]) for case_id in tr_ids])
            _save_logits(os.path.join(directory, "train_logits.npz"), tr_ids, tr_logits, tr_labels)
    by_id = {case_id: i for i, case_id in enumerate(dev_ids_out)}
    ordered = dev_logits[[by_id[record["id"]] for record in dev_records]]
    final_scores = quick_macro(ordered, dev_labels, eval_half)
    elapsed = time.time() - start
    done = {
        "exp": args.exp, "model": args.model, "technique": args.technique, "head": args.head,
        "loss": args.loss, "context": model.context, "dev_f1": final_scores["macro_f1"],
        "half": final_scores["half"],
        "micro_f1": micro_f1(dev_labels, final_scores["pred"]),
        "subset_accuracy": subset_accuracy(dev_labels, final_scores["pred"]),
        "peak_gb": _peak_gb(), "tok_s": tokens_seen / max(elapsed, 1e-6),
        "trainable": _n_trainable(model), "hours": elapsed / 3600,
        "cf2_fold": args.cf2_fold, "history": history,
        "lexical": args.lexical, "retrieval": args.retrieval, "cooc": args.cooc,
        "db_fallback": bool(getattr(model, "db_fallback", False)),
        "deep_macro_f1": history[-1].get("deep_macro_f1") if history else None,
    }
    with open(os.path.join(directory, "done.json"), "w", encoding="utf-8") as handle:
        json.dump(done, handle)
    logger.finish()
    print(json.dumps({k: done[k] for k in ("exp", "dev_f1", "hours", "tok_s")}), flush=True)


def _concat_splits(left: dict, right: dict) -> dict:
    flat = np.concatenate([left["flat"], right["flat"]])
    offsets = np.concatenate([left["offsets"], right["offsets"][1:] + left["offsets"][-1]])
    return {"flat": flat, "offsets": offsets, "meta": left["meta"] + right["meta"]}


def _monitor_half(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = np.array(["A"] * n)
    half[rng.permutation(n)[: n // 2]] = "B"
    return half


def _smoke(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lsi(args.model, args.technique, args.context, head=args.head, loss="bce",
                     device=device, attn=args.attn, ckpt_every=0)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=1e-4)
    batches = [synthetic_batch(64, 2, 1, 32000, device) for _ in range(args.smoke_steps)]
    # synthetic ids are inside vocab only when vocab is the real one; clamp was randint to 32000.
    last = run_steps(model, batches, opt, device, args.technique)
    directory = exp_dir(args.exp)
    ensure_dir(directory)
    with open(os.path.join(directory, "done.json"), "w", encoding="utf-8") as handle:
        json.dump({"smoke": True, "loss": last, "finite": True, "peak_gb": _peak_gb(),
                   "trainable": _n_trainable(model)}, handle)
    print(json.dumps({"smoke": args.exp, "loss": last}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="smoke")
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--technique", default="lorma")
    parser.add_argument("--context", default=None)
    parser.add_argument("--head", default="laat", choices=("laat", "pooled", "decoder"))
    parser.add_argument("--loss", default="bce", choices=("bce", "asl", "focal", "pair", "kd", "db"))
    parser.add_argument("--cooc", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--deep-sup", type=float, default=0.0)
    parser.add_argument("--wide-dropout", type=float, default=0.25)
    parser.add_argument("--ema", type=float, default=0.0)
    parser.add_argument("--infer-chunks", type=int, default=0)
    parser.add_argument("--embed-lr", type=float, default=1e-3)
    parser.add_argument("--lr-floor", type=float, default=0.1)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--train-states", action="store_true")
    parser.add_argument("--teacher-states", default=None)
    parser.add_argument("--sampling", default="uniform", choices=("uniform", "sqrt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--cf2-fold", default=None, choices=(None, "A", "B"))
    parser.add_argument("--init", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--view", default="chunks", choices=("chunks", "head", "headtail", "segments"))
    parser.add_argument("--head-tokens", type=int, default=766)
    parser.add_argument("--tail-tokens", type=int, default=256)
    parser.add_argument("--eval-per-epoch", action="store_true")
    parser.add_argument("--train-logits", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=2)
    parser.add_argument("--limit", type=int, default=CHUNK)
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    parser.add_argument("--max-chunks-infer", action="store_true")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-lr", type=float, default=5e-4)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bench:
        run_bench(args)
    elif args.smoke:
        _smoke(args)
    else:
        train_one(args)


if __name__ == "__main__":
    main()
