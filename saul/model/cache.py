"""Incremental-decoding state shared by all Saul architectures.

Prompts are left-padded. `key_valid[:, :length]` marks real (non-pad) positions; attention layers keep K/V
buffers of `max_len`, recurrent layers keep their state, convolutions keep their last `kernel-1` inputs, and
Engram keeps the last token ids it needs for its n-gram hashes.
"""
from __future__ import annotations

import torch


class SaulCache:
    def __init__(self, n_layers: int, batch: int, max_len: int, device):
        self.layers: list[dict] = [dict() for _ in range(n_layers)]
        self.engram: dict = {}
        self.max_len = int(max_len)
        self.length = 0
        self.key_valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
        self.next_pos = torch.zeros(batch, dtype=torch.long, device=device)

    @property
    def batch(self) -> int:
        return int(self.key_valid.shape[0])

    def expand(self, n: int) -> "SaulCache":
        """Repeat every row `n` times (row i becomes rows i*n .. i*n+n-1). Used to fan one prefill out to n samples."""
        if n == 1:
            return self

        def rep(x):
            return x.repeat_interleave(n, dim=0) if torch.is_tensor(x) else x

        for d in [*self.layers, self.engram]:
            for k in list(d):
                d[k] = rep(d[k])
        self.key_valid = rep(self.key_valid)
        self.next_pos = rep(self.next_pos)
        return self
