"""Seeding and RNG-state capture/restore (for exact resume)."""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    import torch

    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict[str, Any]) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def stable_hash(s: str, salt: str = "") -> int:
    """Process-independent 64-bit hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b((salt + s).encode("utf-8"), digest_size=8).digest(), "little")


def stable_unit(s: str, salt: str = "") -> float:
    """Deterministic float in [0, 1) derived from a string."""
    return stable_hash(s, salt) / 2**64
