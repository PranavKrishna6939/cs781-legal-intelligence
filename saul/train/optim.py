"""Muon (Newton–Schulz orthogonalised updates) for 2-D maps, AdamW for the rest.

Q/K/V matrices are orthogonalised per head and the fused SwiGLU `gate_up` matrix per half. Every block of the
same shape (across heads *and* layers) is orthogonalised in one batched Newton–Schulz call.

Each orthogonalised block is scaled by `0.2 * sqrt(max(rows, cols))` (Moonlight). NS output has singular values
near 1, so its RMS is about `1/sqrt(max(rows, cols))` and the scaled update has RMS about 0.2, the same as a typical
AdamW update. Muon therefore uses an AdamW-scale learning rate (~1e-3 to 3e-3), not the 0.02 of the original Muon.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.optim import AdamW, Optimizer

from saul.model.attention import GatedAttention
from saul.model.mlp import SwiGLU

MUON_RMS = 0.2


def zeropower_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton–Schulz (Jordan et al.) on the last two dims; leading dims are a batch.

    Each matrix is normalised by its Frobenius norm so singular values start in [0, 1]. The iteration runs in
    bfloat16 like the reference implementation (these coefficients overshoot in fp32 by design).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.float()


class Muon(Optimizer):
    """Nesterov-momentum Muon with cautious decoupled weight decay.

    `blocks` maps a parameter to the number of equal row-blocks it is split into before orthogonalisation
    (heads for Q/K/V, 2 for fused gate/up). It lives on the optimizer, not in `param_groups`, so it survives
    `load_state_dict` (param ids differ across processes).
    """

    def __init__(self, params, lr: float = 2e-3, momentum: float = 0.95, weight_decay: float = 0.1,
                 ns_steps: int = 5, cautious_wd: bool = True, blocks: dict | None = None):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                       ns_steps=ns_steps, cautious_wd=cautious_wd))
        self.blocks = blocks or {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            buckets: dict[tuple, list] = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "mom" not in state:
                    state["mom"] = torch.zeros_like(p.grad, dtype=torch.float32)
                mom = state["mom"]
                g = p.grad.float()
                mom.mul_(group["momentum"]).add_(g)
                g_use = g.add(mom, alpha=group["momentum"])
                n = self.blocks.get(p, 1)
                rows, cols = g_use.shape
                if rows % n:
                    raise ValueError(f"cannot split {tuple(g_use.shape)} into {n} row blocks")
                view = g_use.view(n, rows // n, cols)
                buckets[tuple(view.shape[1:])].append((p, view))
            for shape, items in buckets.items():
                stacked = torch.cat([v for _, v in items], dim=0)
                upd = zeropower_newtonschulz(stacked, group["ns_steps"]) * (MUON_RMS * max(shape) ** 0.5)
                offset = 0
                for p, view in items:
                    u = upd[offset:offset + view.shape[0]].reshape(p.shape).to(p.dtype)
                    offset += view.shape[0]
                    if group["weight_decay"]:
                        if group["cautious_wd"]:
                            mask = (u * p) > 0
                            p.mul_(1 - group["lr"] * group["weight_decay"] * mask)
                        else:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(u, alpha=-group["lr"])
        return loss


def build_optimizers(model, muon_lr: float, adamw_lr: float, muon_wd: float = 0.1, ns_steps: int = 5,
                     embed_lr: float | None = None, engram_lr_mult: float = 5.0, use_muon: bool = True):
    """Return (muon, adamw). AdamW groups are named: embed, norm, engram, other (+ matrix when use_muon=False).

    With `use_muon=False` (the AdamW baseline) Muon is None and the 2-D matrices Muon would take go to an AdamW
    group at `muon_lr` with decoupled weight decay `muon_wd`.
    """
    blocks: dict = {}
    muon_params = []
    seen = set()

    def take_muon(p, n_blocks: int = 1):
        if p is None or id(p) in seen or p.ndim != 2:
            return
        seen.add(id(p))
        muon_params.append(p)
        if n_blocks > 1:
            blocks[p] = n_blocks

    for mod in model.modules():
        if isinstance(mod, GatedAttention):
            take_muon(mod.q.weight, mod.n_heads)
            take_muon(mod.k.weight, mod.n_kv_heads)
            take_muon(mod.v.weight, mod.n_kv_heads)
            take_muon(mod.g.weight if mod.g is not None else None)
            take_muon(mod.o.weight)
        elif isinstance(mod, SwiGLU):
            take_muon(mod.gate_up.weight, 2)
            take_muon(mod.down.weight)
        elif hasattr(mod, "q_proj") and hasattr(mod, "A"):  # GatedDeltaNet
            take_muon(mod.q_proj.weight, mod.n_heads)
            take_muon(mod.k_proj.weight, mod.n_heads)
            take_muon(mod.v_proj.weight, mod.n_heads)
            take_muon(mod.z_proj.weight, mod.n_heads)
            take_muon(mod.o_proj.weight)
        elif hasattr(mod, "in_proj") and hasattr(mod, "conv") and isinstance(mod.in_proj, torch.nn.Linear):
            take_muon(mod.in_proj.weight, 3)
            take_muon(mod.out_proj.weight)

    embed, norm, engram, other = [], [], [], []
    for name, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        lname = name.lower()
        if "embed" in lname or name.endswith("lm_head.weight"):
            embed.append(p)
        elif "tables" in lname:
            engram.append(p)
        elif p.ndim == 1 or "gamma" in lname or "norm" in lname:
            norm.append(p)
        else:
            other.append(p)
    adam_groups = []
    if use_muon:
        muon = Muon(muon_params, lr=muon_lr, weight_decay=muon_wd, ns_steps=ns_steps, blocks=blocks)
    else:
        muon = None
        adam_groups.append({"name": "matrix", "params": muon_params, "lr": muon_lr, "weight_decay": muon_wd})
    if embed:
        adam_groups.append({"name": "embed", "params": embed, "lr": embed_lr or adamw_lr, "weight_decay": 0.0})
    if norm:
        adam_groups.append({"name": "norm", "params": norm, "lr": adamw_lr, "weight_decay": 0.1})
    if engram:
        adam_groups.append({"name": "engram", "params": engram, "lr": adamw_lr * engram_lr_mult, "weight_decay": 0.0})
    if other:
        adam_groups.append({"name": "other", "params": other, "lr": adamw_lr, "weight_decay": 0.0})
    all_cuda = all(p.is_cuda for g in adam_groups for p in g["params"])
    adamw = AdamW(adam_groups, betas=(0.9, 0.95), eps=1e-10, fused=all_cuda or None)
    return muon, adamw


def optimizer_step(muon: Muon | None, adamw: AdamW | None, lr_mult: float) -> None:
    for opt in (muon, adamw):
        if opt is None:
            continue
        for g in opt.param_groups:
            if "_base_lr" not in g:
                g["_base_lr"] = g["lr"]
            g["lr"] = g["_base_lr"] * lr_mult
        opt.step()
