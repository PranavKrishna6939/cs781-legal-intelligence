"""Warmup–stable–decay schedule. The decay tail is `1 - sqrt(progress)` down to 0 (MiniCPM-style WSD)."""
from __future__ import annotations


def wsd_multiplier(step: int, total_steps: int, warmup_frac: float = 0.02, decay_frac: float = 0.20) -> float:
    if total_steps <= 1:
        return 1.0
    warm = max(1, int(total_steps * warmup_frac))
    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < warm:
        return (step + 1) / warm
    if step < decay_start:
        return 1.0
    span = max(1, total_steps - decay_start)
    progress = min(1.0, (step - decay_start + 1) / span)
    return max(0.0, 1.0 - progress ** 0.5)


def in_decay(step: int, total_steps: int, decay_frac: float = 0.20) -> bool:
    return step >= int(total_steps * (1.0 - decay_frac))
