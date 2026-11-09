"""Tail EMA of parameters, updated only during the decay phase.

The fp32 shadow lives on the parameters' device (1.2 GB for 300M params) and is updated with one fused
`_foreach_lerp_`, so an update costs a few milliseconds instead of a device-to-host copy of the whole model.
It is allocated at the first update, so the stable phase pays nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TailEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.names = [n for n, _ in model.named_parameters()]
        self.shadow: list[torch.Tensor] | None = None
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        params = [p.detach() for _, p in model.named_parameters()]
        if self.shadow is None:
            self.shadow = [p.float().clone() for p in params]
        else:
            torch._foreach_lerp_(self.shadow, [p.float() for p in params], 1.0 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict:
        shadow = None if self.shadow is None else {n: t.cpu() for n, t in zip(self.names, self.shadow)}
        return {"decay": self.decay, "updates": self.updates, "shadow": shadow}

    def load_state_dict(self, st: dict, device=None) -> None:
        self.decay = st["decay"]
        self.updates = st["updates"]
        sh = st.get("shadow")
        self.shadow = None if sh is None else [sh[n].to(device) if device else sh[n] for n in self.names]

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> list[torch.Tensor]:
        """Swap EMA weights in. Returns the live weights so the caller can restore them."""
        if self.shadow is None:
            raise RuntimeError("EMA has no updates yet")
        backup = []
        for p, s in zip(model.parameters(), self.shadow):
            backup.append(p.detach().clone())
            p.copy_(s.to(device=p.device, dtype=p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: list[torch.Tensor]) -> None:
        for p, b in zip(model.parameters(), backup):
            p.copy_(b)
