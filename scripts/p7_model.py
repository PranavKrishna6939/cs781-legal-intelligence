"""Phase 1 encoder helpers for Saul-CiteRank. The cosine schedule uses the real step count."""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.peft import IA3Scale, LoRMAPlus, apply_partial, apply_technique, merge_adapters
from saul.tokenizer.train import load_hf_tokenizer

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
CKPT = os.path.join(ROOT, "checkpoints", "dense_v1")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizers", "legal32k")


class Head(nn.Module):
    def __init__(self, d_model: int, width: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(hidden.float()), dim=-1)


def load_tokenizer():
    tok = load_hf_tokenizer(TOKENIZER)
    return tok, int(tok.bos_token_id), int(tok.pad_token_id)


def marker_piece(tok) -> list[int]:
    return tok.encode("CITATION", add_special_tokens=False)


def encode_ids(tok, text: str, bos: int, limit: int) -> tuple[list[int], int | None]:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids or ids[0] != bos:
        ids = [bos] + ids
    piece = marker_piece(tok)
    index = None
    if piece:
        for i in range(len(ids) - len(piece) + 1):
            if ids[i:i + len(piece)] == piece:
                index = i + len(piece) - 1
                break
    if len(ids) > limit:
        if index is not None and index >= limit:
            start = min(index - limit // 2, len(ids) - limit)
            start = max(0, start)
            index -= start
            ids = ids[start:start + limit]
        else:
            ids = ids[:limit]
            if index is not None and index >= limit:
                index = None
    return ids, index


def collate(rows: list[list[int]], pad: int, device: str) -> torch.Tensor:
    width = max(len(row) for row in rows)
    out = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out


def checkpoint_every(model) -> None:
    import torch.utils.checkpoint as checkpoint

    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)

            def run(*inner):
                return _original(*inner, **kwargs)

            return checkpoint.checkpoint(run, *args, use_reentrant=False)

        block.forward = wrapped


def pool(hidden: torch.Tensor, ids: torch.Tensor, pad: int, indexes: list[int | None]) -> torch.Tensor:
    rows = []
    for batch, index in enumerate(indexes):
        if index is not None and index < hidden.shape[1]:
            rows.append(hidden[batch, index])
        else:
            mask = ids[batch].ne(pad).unsqueeze(-1)
            rows.append((hidden[batch] * mask).sum(0) / mask.sum().clamp(min=1))
    return torch.stack(rows)


def forward_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    return model(ids, mode="bidirectional", return_hidden=True).hidden


def build(technique: str, train_blocks: bool = False) -> SaulForCausalLM:
    model = SaulForCausalLM.from_pretrained(CKPT)
    if technique == "partial":
        n = len(model.blocks)
        apply_partial(model, n - 6, n)
    elif technique == "student":
        for param in model.parameters():
            param.requires_grad_(False)
        model._ner_stop_after = 6
        for index, block in enumerate(model.blocks):
            if index < 6:
                for param in block.parameters():
                    param.requires_grad_(True)
    else:
        apply_technique(model, technique)
        for param in model.parameters():
            param.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, LoRMAPlus):
                for param in list(module.A.parameters()) + list(module.B.parameters()):
                    param.requires_grad_(True)
            elif isinstance(module, IA3Scale):
                module.scale.requires_grad_(True)
    if train_blocks:
        pass
    return model


def unwrap_lorma(model) -> None:
    merge_adapters(model)
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRMAPlus):
            parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, module.base)


def cosine(step: int, total: int, base: float) -> float:
    warm = max(1, int(0.05 * total))
    if step < warm:
        return base * (step + 1) / warm
    progress = (step - warm) / max(total - warm, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(opt, step: int, total: int, bases: list[float]) -> None:
    for group, base in zip(opt.param_groups, bases):
        group["lr"] = cosine(step, total, base)


def gpu_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
