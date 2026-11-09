"""Phase 6 trainer. Full fine-tune, LoRMA, or Pfeiffer, from a Phase 1 checkpoint.

Checkpoints store the trainable tensors (`latest.pt` for resume, `best.pt` for the EMA
weights that won the holdout). Decoding rebuilds the wrappers from `meta.json` and the
Phase 1 base, then copies those tensors in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from saul.cjpe.model import enable_activation_checkpointing
from saul.ner.peft import freeze_all, unfreeze_adapters, wrap_lorma, wrap_pfeiffer
from saul.ner.registry import resolve_checkpoint
from saul.summ.data import (
    bucket_of,
    build_sequence,
    caption_is_safe,
    collate,
    holdout_ids,
    load_rows,
    p6_dir,
    pack_sequences,
    reserved_length_ids,
    strip_caption,
)
from saul.summ.length import fit_ridge
from saul.tokenizer.train import load_hf_tokenizer


def _tokenizer():
    path = os.environ.get("SAUL_TOKENIZER", os.path.expanduser("~/cs781/artifacts/tokenizers/legal32k"))
    return load_hf_tokenizer(path)


def _ids(tok) -> tuple[int, int, int]:
    bos, eos = int(tok.bos_token_id), int(tok.eos_token_id)
    summary = int(tok.convert_tokens_to_ids("<|summary|>"))
    if summary < 0:
        raise RuntimeError("<|summary|> is missing")
    return bos, eos, summary


def init_length_rows(model, summary_id: int, len_ids: list[int]) -> None:
    with torch.no_grad():
        ref = model.embed.weight[summary_id]
        scale = 0.02 * ref.detach().float().norm().clamp(min=1e-6)
        noise = torch.randn(len(len_ids), ref.shape[0], device=ref.device, dtype=ref.dtype)
        rows = ref.unsqueeze(0) + noise * scale.to(ref.dtype)
        model.embed.weight.index_copy_(0, torch.tensor(len_ids, device=ref.device), rows)


def _hook_length_rows(param: torch.nn.Parameter, len_ids: list[int]) -> None:
    index = torch.tensor(len_ids, device=param.device)

    def hook(grad):
        kept = grad.index_select(0, index.to(grad.device))
        out = torch.zeros_like(grad)
        return out.index_copy(0, index.to(grad.device), kept)

    param.register_hook(hook)


def attach(model, technique: str, rank: int, bottleneck: int, len_ids: list[int]) -> None:
    if technique == "full":
        return
    if technique == "lorma":
        wrap_lorma(model, rank=rank)
    elif technique == "pfeiffer":
        wrap_pfeiffer(model, bottleneck=bottleneck)
    else:
        raise ValueError(technique)
    freeze_all(model)
    unfreeze_adapters(model)
    if technique == "pfeiffer":
        for name, param in model.named_parameters():
            if name.endswith("gamma"):
                param.requires_grad_(True)
    model.embed.weight.requires_grad_(True)
    _hook_length_rows(model.embed.weight, len_ids)


def param_groups(model, base_lr: float, fast_lr: float, embed_lr: float) -> list[dict]:
    buckets: dict[tuple, list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim < 2 or "embed" in name or name.endswith("gamma")
        if "embed" in name:
            lr = embed_lr
        elif name.endswith("mixer.A") or name.endswith("mixer.W_a.bias"):
            lr = fast_lr
        else:
            lr = base_lr
        key = (lr, 0.0 if no_decay else 0.01)
        buckets.setdefault(key, []).append(param)
    return [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in buckets.items()]


def lr_mult(step: int, total: int, warmup: float = 0.03, floor: float = 0.1) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return floor + (1.0 - floor) * cosine


class _EMA:
    def __init__(self, params: list[torch.nn.Parameter], decay: float = 0.999):
        self.decay = decay
        self.params = params
        self.shadow = [p.detach().float().clone() for p in params]

    def update(self) -> None:
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    def copy_to(self) -> list[torch.Tensor]:
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for shadow, param in zip(self.shadow, self.params):
                param.copy_(shadow.to(dtype=param.dtype))
        return backup

    def restore(self, backup: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for saved, param in zip(backup, self.params):
                param.copy_(saved)


def _trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def save_tensors(model, path: str) -> None:
    tensors = {name: p.detach().float().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(tensors, path)


def load_tensors(model, path: str) -> None:
    tensors = torch.load(path, map_location="cpu", weights_only=False)
    owned = dict(model.named_parameters())
    missing = [name for name in tensors if name not in owned]
    if missing:
        raise RuntimeError(f"checkpoint keys missing from model: {missing[:5]}")
    with torch.no_grad():
        for name, value in tensors.items():
            owned[name].copy_(value)


def load_run(run: str, device: str):
    """Rebuild the Phase 1 model, re-apply the technique, and load `best.pt` (else `latest.pt`)."""
    from saul.model.modeling_saul import SaulForCausalLM

    meta = json.load(open(os.path.join(run, "meta.json"), encoding="utf-8"))
    tok = _tokenizer()
    model = SaulForCausalLM.from_pretrained(meta["base"])
    attach(model, meta["technique"], meta.get("rank", 32), meta.get("bottleneck", 128), meta["len_ids"])
    weights = os.path.join(run, "best.pt" if os.path.exists(os.path.join(run, "best.pt")) else "latest.pt")
    load_tensors(model, weights)
    model.to(device).eval()
    return model, tok, meta


def _pairs(tok, rows, len_ids, lmax, strip, bos, eos, summary_id, summary_of) -> list[tuple[list[int], list[int]]]:
    pairs = []
    for row in rows:
        text = strip_caption(row["document"]) if strip else row["document"]
        doc = tok.encode(text, add_special_tokens=False)
        summary = summary_of(row)
        words = len(summary.split()) if isinstance(summary, str) else 0
        ids = tok.encode(summary, add_special_tokens=False)
        seq, mask = build_sequence(doc, ids, bos, summary_id, len_ids[bucket_of(words)], eos, lmax)
        pairs.append((seq, mask))
    return pairs


def _forward(model, row: dict, pad_id: int, device: str) -> torch.Tensor:
    batch = collate([row], pad_id)
    ids = torch.from_numpy(batch["input_ids"]).to(device)
    docs = torch.from_numpy(batch["doc_ids"]).to(device)
    pos = torch.from_numpy(batch["position_ids"]).to(device)
    mask = torch.from_numpy(batch["loss_mask"]).to(device)
    cu = torch.from_numpy(batch["cu_seqlens"])
    enabled = device == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
        return model(
            ids, labels=ids, doc_ids=docs, position_ids=pos, loss_mask=mask,
            cu_seqlens=cu.to(device), cu_seqlens_cpu=cu, mode="causal",
        ).loss


def _grouped(pairs: list, budget: int) -> list[list[int]]:
    groups, cur, cost = [], [], 0
    for i, (seq, _mask) in enumerate(pairs):
        if cur and cost + len(seq) > budget:
            groups.append(cur)
            cur, cost = [], 0
        cur.append(i)
        cost += len(seq)
    if cur:
        groups.append(cur)
    return groups


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_b")
    ap.add_argument("--technique", default="full", choices=("full", "lorma", "pfeiffer"))
    ap.add_argument("--lmax", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fast-lr", type=float, default=1e-4)
    ap.add_argument("--embed-lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--pack-tokens", type=int, default=16384)
    ap.add_argument("--step-tokens", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--name", default="E1")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    tok = _tokenizer()
    bos, eos, summary_id = _ids(tok)
    len_ids = reserved_length_ids(tok)
    train_rows = load_rows("train")
    held = holdout_ids()
    fit_rows = [row for row in train_rows if row["id"] not in held]
    val_rows = [row for row in train_rows if row["id"] in held]
    if args.limit:
        fit_rows = fit_rows[: args.limit]
        val_rows = val_rows[: min(4, len(val_rows))]
    strip = caption_is_safe(fit_rows)
    print(f"[p6] strip_caption {strip} fit {len(fit_rows)} val {len(val_rows)}", flush=True)

    teacher = {}
    if args.teacher:
        for row in json.load(open(args.teacher, encoding="utf-8")):
            teacher[row["id"]] = row["summary"]

    from saul.model.modeling_saul import SaulForCausalLM

    base = resolve_checkpoint(args.model)
    model = SaulForCausalLM.from_pretrained(base)
    lmax = min(args.lmax, int(model.config.max_position))
    init_length_rows(model, summary_id, len_ids)
    attach(model, args.technique, args.rank, args.bottleneck, len_ids)
    enable_activation_checkpointing(model, every=1)
    model.to(device).train()
    params = _trainable(model)
    print(f"[p6] trainable {sum(p.numel() for p in params)} lmax {lmax}", flush=True)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.fast_lr, args.embed_lr), lr=args.lr)
    ema = _EMA(params)
    ridge = fit_ridge(fit_rows).tolist()

    run = os.path.join(p6_dir(), "exp", args.name)
    os.makedirs(run, exist_ok=True)
    meta = {
        "base": base, "model": args.model, "technique": args.technique, "rank": args.rank,
        "bottleneck": args.bottleneck, "lmax": lmax, "strip": strip, "len_ids": len_ids,
        "ridge": ridge, "length_method": "ridge", "summary_id": summary_id,
    }
    json.dump(meta, open(os.path.join(run, "meta.json"), "w"), indent=2)

    def pairs_for(rows, summary_of):
        return _pairs(tok, rows, len_ids, lmax, strip, bos, eos, summary_id, summary_of)

    gold_of = lambda row: row["summary"]
    gold = pairs_for(fit_rows, gold_of)
    teach = pairs_for(fit_rows, lambda row: teacher.get(row["id"], row["summary"])) if teacher else None
    pad = int(tok.pad_token_id)
    if teach is None:
        groups = _grouped(gold, args.pack_tokens)
    else:
        # The same documents in each group, sized by the longer of the two targets.
        sized = [(list(range(max(len(g[0]), len(t[0])))), [1]) for g, t in zip(gold, teach)]
        groups = _grouped(sized, args.pack_tokens)
    order = list(range(len(groups)))
    tokens_per_epoch = sum(len(seq) for seq, _mask in gold)
    if teach is not None:
        tokens_per_epoch += sum(len(seq) for seq, _mask in teach)
    total_steps = max(1, math.ceil(args.epochs * tokens_per_epoch / args.step_tokens))
    print(f"[p6] groups {len(groups)} steps/epoch-scale {total_steps} tokens {tokens_per_epoch}", flush=True)

    state_path = os.path.join(run, "state.json")
    step = 0
    epoch = 0
    cursor = 0
    best = -1.0
    if os.path.exists(state_path) and os.path.exists(os.path.join(run, "latest.pt")):
        state = json.load(open(state_path, encoding="utf-8"))
        load_tensors(model, os.path.join(run, "latest.pt"))
        step, epoch, cursor = state["step"], state["epoch"], state["cursor"]
        best = state["best"]
        order = state["order"]
        print(f"[p6] resume step {step} epoch {epoch}", flush=True)

    rng = np.random.default_rng(781)
    started = time.time()
    step_t = started
    seen = 0
    opt.zero_grad(set_to_none=True)
    pending = 0

    def take_step(micro_tokens: int) -> None:
        nonlocal step, pending, step_t
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale = lr_mult(step, total_steps)
        for group, spec in zip(opt.param_groups, param_groups(model, args.lr, args.fast_lr, args.embed_lr)):
            group["lr"] = spec["lr"] * scale
        opt.step()
        ema.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        now = time.time()
        rate = micro_tokens / max(1e-6, now - step_t)
        step_t = now
        pending = 0
        if step <= 3 or step % 10 == 0 or args.smoke:
            print(f"[p6] step {step} step_tok/s {rate:.0f} lr {scale:.3f}", flush=True)

    while epoch < args.epochs:
        if cursor == 0:
            rng.shuffle(order)
        while cursor < len(order):
            idx = groups[order[cursor]]
            cursor += 1
            try:
                loss_g = _forward(model, pack_sequences([gold[i] for i in idx], args.pack_tokens)[0], pad, device)
                if not torch.isfinite(loss_g):
                    raise SystemExit("[p6] non-finite loss")
                ntok = sum(len(gold[i][0]) for i in idx)
                scale = ntok / args.step_tokens
                weight = 0.5 if teach is not None else 1.0
                (loss_g * weight * scale).backward()
                if teach is not None:
                    loss_t = _forward(model, pack_sequences([teach[i] for i in idx], args.pack_tokens)[0], pad, device)
                    ttok = sum(len(teach[i][0]) for i in idx)
                    (loss_t * 0.5 * (ttok / args.step_tokens)).backward()
                    ntok += ttok
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[p6] skip OOM group of {len(idx)}", flush=True)
                continue
            seen += ntok
            pending += ntok
            if pending >= args.step_tokens:
                take_step(pending)
                if step % 50 == 0:
                    save_tensors(model, os.path.join(run, "latest.pt"))
                    json.dump({"step": step, "epoch": epoch, "cursor": cursor, "best": best, "order": order},
                              open(state_path, "w"))
                if args.max_steps and step >= args.max_steps:
                    epoch = args.epochs
                    cursor = len(order)
                    break
            marks = {max(1, len(order) // 4), max(1, len(order) // 2), max(1, 3 * len(order) // 4)}
            if not args.smoke and epoch >= 1 and cursor in marks:
                best = _select(model, ema, tok, val_rows, meta, run, device, best, args)
        epoch += 1
        cursor = 0
        print(f"[p6] epoch {epoch} done step {step}", flush=True)
        if not args.smoke and epoch >= 1 and epoch <= args.epochs:
            best = _select(model, ema, tok, val_rows, meta, run, device, best, args)

    save_tensors(model, os.path.join(run, "latest.pt"))
    backup = ema.copy_to()
    if not os.path.exists(os.path.join(run, "best.pt")):
        save_tensors(model, os.path.join(run, "best.pt"))
    if args.smoke and val_rows:
        from saul.summ.decode import generate, lengths_for
        from saul.summ.length import predict_ridge
        one = val_rows[:1]
        pred = predict_ridge(np.array(meta["ridge"], dtype=np.float64), one[0]["document"])
        targets, forced = lengths_for([pred], 1.0, len_ids)
        text = generate(model, tok, one, targets, forced, lmax, strip, device, n=1, slots=1)
        print(f"[p6] smoke summary words {len(text[0].split())}", flush=True)
    ema.restore(backup)
    json.dump({"step": step, "epoch": epoch, "best": best, "tokens_per_s": seen / max(1e-6, time.time() - started)},
              open(os.path.join(run, "done.json"), "w"), indent=2)
    print(f"[p6] train done step {step} best {best}", flush=True)


def _select(model, ema, tok, val_rows, meta, run, device, best, args) -> float:
    from saul.summ.decode import generate, lengths_for, predicted_words
    from saul.summ.score import score_texts

    backup = ema.copy_to()
    model.eval()
    try:
        ridge = np.array(meta["ridge"], dtype=np.float64)
        words = predicted_words(
            model, tok, val_rows, meta["len_ids"], meta["lmax"], meta["strip"], device,
            meta["length_method"], ridge,
        )
        if "length_chosen" not in meta:
            gold = [len(row["summary"].split()) for row in val_rows]
            self_words = predicted_words(
                model, tok, val_rows, meta["len_ids"], meta["lmax"], meta["strip"], device, "self", None,
            )
            ridge_err = float(np.mean([abs(w - g) for w, g in zip(words, gold)]))
            self_err = float(np.mean([abs(w - g) for w, g in zip(self_words, gold)]))
            meta["length_method"] = "self" if self_err < ridge_err else "ridge"
            meta["length_chosen"] = True
            meta["length_mae"] = {"ridge": ridge_err, "self": self_err}
            json.dump(meta, open(os.path.join(run, "meta.json"), "w"), indent=2)
            print(f"[p6] length method {meta['length_method']} mae {meta['length_mae']}", flush=True)
            if meta["length_method"] == "self":
                words = self_words
        targets, forced = lengths_for(words, 1.0, meta["len_ids"])
        slots = 16 if device == "cuda" else 2
        texts = generate(
            model, tok, val_rows, targets, forced, meta["lmax"], meta["strip"], device, n=1, slots=slots,
        )
        scored = score_texts(texts, [row["summary"] for row in val_rows], device=device if device == "cuda" else "cpu")
    finally:
        ema.restore(backup)
        model.train()
    print(f"[p6] holdout bert {scored['bert']:.4f} r1 {scored['rouge1']:.4f}", flush=True)
    if scored["bert"] > best:
        best = scored["bert"]
        backup = ema.copy_to()
        save_tensors(model, os.path.join(run, "best.pt"))
        ema.restore(backup)
        json.dump({"bert": scored["bert"], "rouge1": scored["rouge1"], "rougeL": scored["rougeL"]},
                  open(os.path.join(run, "val.json"), "w"), indent=2)
    return best


if __name__ == "__main__":
    train()
