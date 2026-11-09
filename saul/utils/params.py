"""Parameter counting with the competition's strictest reading: every parameter counts, tied weights once.

`nn.Module.named_parameters()` already de-duplicates shared tensors, so a tied embedding/LM head counts once.
Buffers (e.g. RoPE caches, Engram compression map) are not parameters and do not count.
"""
from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn

MAX_PARAMS = 299_000_000  # safety margin under the 300M rule


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for _, p in model.named_parameters())


def param_breakdown(model: nn.Module, depth: int = 2) -> "OrderedDict[str, int]":
    out: "OrderedDict[str, int]" = OrderedDict()
    for name, p in model.named_parameters():
        key = ".".join(name.split(".")[:depth])
        out[key] = out.get(key, 0) + p.numel()
    return out


def assert_param_budget(model: nn.Module, max_params: int = MAX_PARAMS, min_params: int = 50_000_000,
                        enforce_min: bool = True) -> int:
    n = count_params(model)
    if n > max_params:
        raise AssertionError(f"model has {n:,} params > budget {max_params:,} (competition cap is 300M)")
    if enforce_min and n < min_params:
        raise AssertionError(f"model has {n:,} params < 50M competition minimum")
    return n


def format_breakdown(model: nn.Module, depth: int = 2) -> str:
    bd = param_breakdown(model, depth)
    total = sum(bd.values())
    lines = [f"{'module':<48}{'params':>14}{'share':>8}"]
    for k, v in bd.items():
        lines.append(f"{k:<48}{v:>14,}{100 * v / max(total, 1):>7.2f}%")
    lines.append(f"{'TOTAL':<48}{total:>14,}")
    return "\n".join(lines)
