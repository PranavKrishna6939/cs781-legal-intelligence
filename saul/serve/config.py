"""Engine configuration and a startup memory estimate.

One engine serves every Saul architecture. Later checkpoints (hybrid, CED, conv, long context) only change
the model; the pool sizes come from `max_slots`, `block_size` and the length caps.
"""
from __future__ import annotations

from dataclasses import dataclass


def _cover(buckets: tuple[int, ...], cap: int) -> tuple[int, ...]:
    """Keep buckets that fit in `cap`, and make sure the largest one is exactly `cap`."""
    if cap < 1:
        raise ValueError("length caps must be positive")
    kept = tuple(b for b in buckets if 1 <= b <= cap)
    if not kept or kept[-1] != cap:
        kept = tuple(sorted(set(kept + (cap,))))
    return kept


@dataclass
class EngineConfig:
    max_slots: int = 512
    block_size: int = 16
    max_prompt_len: int = 1024
    max_new_tokens: int = 1024
    suffix_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024)
    prompt_buckets: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    slot_buckets: tuple[int, ...] | None = None
    prefill_token_budget: int = 65536
    admit_batch_blocks: int = 8
    dtype: str = "bfloat16"
    cuda_graphs: bool = True
    seed: int = 781
    device: str = "cuda"
    mbr_batch_groups: int = 32
    rouge_workers: int = 6
    truncate_left: bool = False

    def __post_init__(self) -> None:
        if self.max_slots < 1 or self.block_size < 1:
            raise ValueError("max_slots and block_size must be positive")
        if self.max_slots % self.block_size:
            raise ValueError("max_slots must be a multiple of block_size")
        if self.slot_buckets is None:
            b = self.block_size
            raw = [b, 4 * b, 8 * b, self.max_slots]
            self.slot_buckets = tuple(sorted({min(self.max_slots, max(b, x)) for x in raw}))
        self.suffix_buckets = _cover(self.suffix_buckets, self.max_new_tokens)
        self.prompt_buckets = _cover(self.prompt_buckets, self.max_prompt_len)
        self.slot_buckets = _cover(self.slot_buckets, self.max_slots)
        if any(s % self.block_size for s in self.slot_buckets):
            raise ValueError("slot buckets must be multiples of block_size")
        if self.prefill_token_budget < self.max_prompt_len:
            raise ValueError("prefill_token_budget must be at least max_prompt_len")
        # A dense SDPA mask is quadratic. 8k tokens is a few hundred MB; 64k is several GB and a slow build.
        self.prefill_token_budget = min(self.prefill_token_budget, max(self.max_prompt_len, 8192))

    @property
    def n_blocks(self) -> int:
        return self.max_slots // self.block_size

    def bucket(self, value: int, buckets: tuple[int, ...]) -> int:
        for b in buckets:
            if b >= value:
                return b
        raise ValueError(f"{value} exceeds buckets {buckets}")

    def estimate_bytes(self, n_layers: int, n_kv: int, head_dim: int, vocab: int, n_gdn: int = 0,
                       gdn_heads: int = 0, gdn_dim: int = 0) -> int:
        """Rough pool + logit buffer size. Weights are extra (the loaded model)."""
        g, s = self.n_blocks, self.max_slots
        lp, ls = self.max_prompt_len, self.max_new_tokens
        kv = n_layers * 2 * n_kv * head_dim * 2  # bf16 K and V
        prompt = g * lp * kv
        suffix = s * ls * kv
        gdn = n_gdn * s * gdn_heads * gdn_dim * gdn_dim * 4
        logits = s * vocab * 4
        return prompt + suffix + gdn + logits


def memory_ok(cfg: EngineConfig, needed: int, free: int | None) -> None:
    if free is not None and needed > int(0.9 * free):
        raise RuntimeError(f"engine pools need {needed / 2**30:.1f} GiB but only {free / 2**30:.1f} GiB is free")
