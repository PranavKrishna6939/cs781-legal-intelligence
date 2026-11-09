"""Parameter-efficient adapters for Saul blocks.

Targets are the mixer and FFN linears only: attention q/k/v/g/o, SwiGLU gate_up/down,
GDN projections, and the conv block's in/out projections. Depthwise convs, the narrow
GDN gates, and the residual bottlenecks stay frozen.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_LEAVES = {
    "q", "k", "v", "g", "o",
    "gate_up", "down",
    "q_proj", "k_proj", "v_proj", "z_proj", "o_proj",
    "in_proj", "out_proj",
}


class LoRMAPlus(nn.Module):
    """h = W0 x + (alpha/r) B A (W0 x), with B initialised at 0 so the block starts as W0."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        d_out = base.out_features
        self.A = nn.Linear(d_out, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.scale * self.B(self.A(y))

    @torch.no_grad()
    def merge_(self) -> None:
        eye = torch.eye(self.B.weight.shape[0], device=self.B.weight.device, dtype=self.base.weight.dtype)
        ba = (self.B.weight @ self.A.weight).to(self.base.weight.dtype)
        self.base.weight.copy_((eye + self.scale * ba) @ self.base.weight)


class LoRA(nn.Module):
    """rsLoRA: scale is alpha / sqrt(rank). B starts at 0."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / math.sqrt(rank)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))

    @torch.no_grad()
    def merge_(self) -> None:
        delta = (self.B.weight @ self.A.weight).to(self.base.weight.dtype)
        self.base.weight.add_(self.scale * delta)


class LoRMApi(nn.Module):
    """Permutation rank inflation. Identity at init: I_pi(BA) = I, so the block starts as W0.

    The permutation cycles row i by i positions. BA is formed in the forward (the paper's
    cost note); this variant is the E9 ablation, not a default run.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        d_out = base.out_features
        self.A = nn.Parameter(torch.zeros(rank, d_out))
        self.B = nn.Parameter(torch.randn(d_out, rank) * 0.02)
        with torch.no_grad():
            self.B[:, 0] = 1.0
            self.A[0, 0] = 1.0
            self.A[0, 1:] = 0.0

    def _inflate(self, ba: torch.Tensor) -> torch.Tensor:
        # ba: [d, d]. Row i is rotated left by i.
        d = ba.shape[0]
        idx = (torch.arange(d, device=ba.device).unsqueeze(1) + torch.arange(d, device=ba.device).unsqueeze(0)) % d
        return ba.gather(1, idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        ba = self.B @ self.A
        mixed = self._inflate(ba)
        return torch.nn.functional.linear(y, self.scale * mixed)


IA3_OUTPUT = {"k", "v", "k_proj", "v_proj", "in_proj"}
IA3_INPUT = {"down"}


class IA3Scale(nn.Module):
    """Learned rescaling, initialised at 1 so the wrapped linear starts unchanged.

    `output` multiplies the linear's output (keys, values, conv in-projection).
    `input` multiplies the linear's input (the FFN down projection).
    """

    def __init__(self, base: nn.Linear, on: str = "output"):
        super().__init__()
        if on not in ("output", "input"):
            raise ValueError(on)
        self.base = base
        self.on = on
        for p in base.parameters():
            p.requires_grad_(False)
        width = base.out_features if on == "output" else base.in_features
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.on == "input":
            return self.base(x * self.scale.to(dtype=x.dtype))
        return self.base(x) * self.scale.to(dtype=x.dtype)

    @torch.no_grad()
    def merge_(self) -> None:
        scale = self.scale.to(dtype=self.base.weight.dtype)
        if self.on == "output":
            self.base.weight.mul_(scale.view(-1, 1))
            if self.base.bias is not None:
                self.base.bias.mul_(scale)
        else:
            self.base.weight.mul_(scale.view(1, -1))
        self.scale.fill_(1.0)


class Pfeiffer(nn.Module):
    """Bottleneck after the FFN. The down projection starts at 0, so the block starts unchanged."""

    def __init__(self, ffn: nn.Module, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.ffn = ffn
        self.up = nn.Linear(d_model, bottleneck)
        self.down = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ffn(x)
        return y + self.down(F.gelu(self.up(y)))


def _replace(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _target_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    found = []
    for name, mod in model.named_modules():
        if ".blocks." not in f".{name}." and not name.startswith("blocks."):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_LEAVES and isinstance(mod, nn.Linear):
            found.append((name, mod))
    return found


def wrap_lorma(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    pairs = _target_linears(model)
    for name, mod in pairs:
        _replace(model, name, LoRMAPlus(mod, rank, alpha))
    return len(pairs)


def wrap_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    pairs = _target_linears(model)
    for name, mod in pairs:
        _replace(model, name, LoRA(mod, rank, alpha))
    return len(pairs)


def wrap_lorma_pi(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    pairs = _target_linears(model)
    for name, mod in pairs:
        _replace(model, name, LoRMApi(mod, rank, alpha))
    return len(pairs)


def wrap_ia3(model: nn.Module) -> int:
    freeze_all(model)
    n = 0
    for name, mod in _target_linears(model):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in IA3_OUTPUT:
            _replace(model, name, IA3Scale(mod, "output"))
            n += 1
        elif leaf in IA3_INPUT:
            _replace(model, name, IA3Scale(mod, "input"))
            n += 1
    return n


def wrap_pfeiffer(model: nn.Module, bottleneck: int = 64) -> int:
    n = 0
    for block in model.blocks:
        if not isinstance(block.ffn, Pfeiffer):
            block.ffn = Pfeiffer(block.ffn, model.config.d_model, bottleneck)
            n += 1
    return n


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def apply_bitfit(model: nn.Module) -> None:
    """Train norm gains and the few biases. Call after adapters are inserted, then unfreeze those."""
    freeze_all(model)
    for name, p in model.named_parameters():
        if name.endswith("gamma") or name.endswith("bias"):
            p.requires_grad_(True)


def unfreeze_adapters(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, LoRMApi):
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)
        elif isinstance(mod, (LoRMAPlus, LoRA)):
            for p in list(mod.A.parameters()) + list(mod.B.parameters()):
                p.requires_grad_(True)
        elif isinstance(mod, IA3Scale):
            mod.scale.requires_grad_(True)
        elif isinstance(mod, Pfeiffer):
            for p in mod.up.parameters():
                p.requires_grad_(True)
            for p in mod.down.parameters():
                p.requires_grad_(True)


def apply_partial(model: nn.Module, start: int, end: int) -> None:
    freeze_all(model)
    for i, block in enumerate(model.blocks):
        if start <= i < end:
            for p in block.parameters():
                p.requires_grad_(True)
    for p in model.final_norm.parameters():
        p.requires_grad_(True)


def apply_technique(model: nn.Module, technique: str, partial: tuple[int, int] | None = None) -> str:
    """Attach adapters / freeze masks. Heads are not part of `model` and stay trainable."""
    technique = technique.lower()
    if technique in ("full", "probe"):
        for p in model.parameters():
            p.requires_grad_(technique == "full")
        return technique
    if technique == "lorma":
        wrap_lorma(model)
        return technique
    if technique == "lora":
        wrap_lora(model)
        return technique
    if technique == "lorma_pi":
        wrap_lorma_pi(model)
        return technique
    if technique == "adapter_bitfit":
        wrap_pfeiffer(model)
        apply_bitfit(model)
        unfreeze_adapters(model)
        return technique
    if technique == "ia3":
        wrap_ia3(model)
        return technique
    if technique == "partial":
        if partial is None:
            n = len(model.blocks)
            partial = (n // 2, n)
        apply_partial(model, partial[0], partial[1])
        return technique
    raise ValueError(f"unknown technique {technique}")


def merge_adapters(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, (LoRMAPlus, LoRA, IA3Scale)):
            mod.merge_()
