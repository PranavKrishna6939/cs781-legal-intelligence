"""Batched sampling. Processors match saul.decode.processors and run as tensor ops over the live slots.

n-gram blocking uses exact packed int64 keys. Each token id is 16 bits. For vocabularies up to 2^16 the packing
is collision-free. The sign bit (token ids >= 2^15 in the high slot) is handled with a logical shift, so a 49k
vocab still matches the brute-force ban.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from saul.decode.processors import NEG

TRUNC = {"none": 0, "min_p": 1, "top_p": 2, "eta": 3}


@dataclass
class MBRParams:
    enabled: bool = False
    utility: str = "mixed"
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.utility not in ("rouge", "embed", "mixed"):
            raise ValueError(self.utility)
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.8
    truncation: str = "min_p"
    min_p: float = 0.05
    top_p: float = 0.9
    eta: float = 3e-3
    no_repeat_ngram: int = 4
    frequency_penalty: float = 0.3
    frequency_window: int = 64
    target_words: int | None = 128
    max_new_tokens: int = 1024
    stop_token_ids: tuple[int, ...] = ()
    suppress_special: bool = True
    seed: int | None = None
    mbr: MBRParams = field(default_factory=MBRParams)
    # Test-only teacher forcing. One sequence is applied to every slot of the request.
    forced_tokens: tuple[int, ...] | None = None
    record_logits: bool = False
    max_ngram_occurrences: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.truncation not in TRUNC:
            raise ValueError(self.truncation)
        if not 0 <= self.min_p <= 1 or not 0 <= self.top_p <= 1:
            raise ValueError("min_p and top_p must be in [0, 1]")
        if self.no_repeat_ngram < 0 or self.no_repeat_ngram > 4:
            raise ValueError("no_repeat_ngram must be 0..4")
        if type(self.max_ngram_occurrences) is not int or not 1 <= self.max_ngram_occurrences <= 32767:
            raise ValueError("max_ngram_occurrences must be an integer in 1..32767")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.target_words is not None and self.target_words < 1:
            raise ValueError("target_words must be >= 1 or None")


def pack_key(tokens: torch.Tensor) -> torch.Tensor:
    """tokens [..., n] int64 -> one int64. n <= 4 and each id must fit in 16 bits."""
    key = torch.zeros(tokens.shape[:-1], dtype=torch.long, device=tokens.device)
    for i in range(tokens.shape[-1]):
        key = (key << 16) | tokens[..., i]
    return key


def _logical_rshift(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return x
    return torch.bitwise_and(torch.bitwise_right_shift(x, bits), (1 << (64 - bits)) - 1)


def ngram_ban_dense(logits: torch.Tensor, generated: torch.Tensor, n: int) -> torch.Tensor:
    """Same banned positions as GeneratedNoRepeatNGram, via packed keys. generated: [R, L]."""
    rows, length = generated.shape
    if n <= 1 or length < n:
        return logits
    windows = generated.unfold(1, n, 1)
    keys = pack_key(windows)
    prefix = pack_key(generated[:, length - n + 1:])
    match = keys[:, :-1] == prefix[:, None] if keys.shape[1] > 1 else keys.new_zeros(rows, 0, dtype=torch.bool)
    # keys[:, :-1] are n-grams that end strictly before the last token; the last window's prefix is
    # tokens[L-n:L-1], while the current prefix is tokens[L-n+1:L]. Include every stored window.
    match = _logical_rshift(keys, 16) == prefix[:, None]
    banned_tok = torch.bitwise_and(keys, 65535)
    banned = torch.zeros_like(logits, dtype=torch.bool)
    if match.any():
        r, c = match.nonzero(as_tuple=True)
        banned[r, banned_tok[r, c]] = True
    return logits.masked_fill(banned, NEG)


def ngram_ban_pool(logits: torch.Tensor, pools, idx: torch.Tensor, levels: tuple[int, ...] = (2, 3, 4)) -> torch.Tensor:
    """Ban from ng_keys. No host sync: scatters a dense match mask instead of nonzero()."""
    if not idx.numel() or not levels:
        return logits
    out = logits
    device = logits.device
    keys = pools.ng_keys[idx]
    positions = torch.arange(keys.shape[1], device=device).view(1, -1)
    sl = pools.slen[idx]
    ns = pools.ngram_n[idx]
    for n in levels:
        if n < 2:
            continue
        cols = []
        for back in range(n - 1, 0, -1):
            pos = (sl - back).clamp(min=0)
            tok = pools.out_tokens[idx, pos]
            tok = torch.where(sl >= back, tok, torch.zeros_like(tok))
            cols.append(tok)
        prefix = pack_key(torch.stack(cols, dim=1))
        prefix_of = _logical_rshift(keys, 16)
        valid = (keys >= 0) & (positions < sl.view(-1, 1)) & (ns == n).view(-1, 1) & (sl >= n).view(-1, 1)
        match = valid & (prefix_of == prefix.view(-1, 1))
        banned_tok = torch.bitwise_and(keys, 65535).clamp(0, out.shape[-1] - 1)
        hits = torch.zeros_like(out)
        hits.scatter_add_(1, torch.where(match, banned_tok, torch.zeros_like(banned_tok)), match.to(out.dtype))
        out = out.masked_fill(hits >= pools.ngram_max[idx, None], NEG)
    return out


def store_ngram_keys(pools, idx: torch.Tensor, levels: tuple[int, ...] = (2, 3, 4)) -> None:
    """After a token is written and slen has been incremented, store the n-gram ending at slen-1."""
    if not idx.numel():
        return
    pos = pools.slen[idx] - 1
    ns = pools.ngram_n[idx]
    for n in levels:
        if n < 2:
            continue
        cols = []
        for back in range(n - 1, -1, -1):
            p = (pos - back).clamp(min=0)
            tok = torch.where(pos >= back, pools.out_tokens[idx, p], torch.zeros_like(pos))
            cols.append(tok)
        key = pack_key(torch.stack(cols, dim=1))
        ready = (pos >= n - 1) & (ns == n)
        key = torch.where(ready, key, torch.full_like(key, -1))
        dest = pos.clamp(min=0)
        prev = pools.ng_keys[idx, dest]
        pools.ng_keys[idx, dest] = torch.where(ready, key, prev)


def update_freq(pools, idx: torch.Tensor, nxt: torch.Tensor, keep: torch.Tensor | None = None) -> None:
    """`keep` is true for slots that actually appended `nxt`. Stop-token slots pass keep=False."""
    if not idx.numel():
        return
    sl = pools.slen[idx]
    win = pools.freq_win[idx]
    leave_pos = sl - win - 1
    leaving = leave_pos >= 0
    if keep is not None:
        leaving = leaving & keep
    old = pools.out_tokens[idx, leave_pos.clamp(min=0)]
    dec = leaving.to(pools.freq.dtype)
    pools.freq[idx, old] = (pools.freq[idx, old] - dec).clamp(min=0)
    add = torch.ones(idx.shape[0], dtype=pools.freq.dtype, device=idx.device) if keep is None else keep.to(pools.freq.dtype)
    pools.freq[idx, nxt] = pools.freq[idx, nxt] + add


def apply_penalties(logits: torch.Tensor, pools, idx: torch.Tensor, suppress: torch.Tensor | None,
                    levels: tuple[int, ...] = (2, 3, 4)) -> torch.Tensor:
    """suppress, n-gram, frequency. Temperature and truncation happen in `sample_from`."""
    out = logits.clone()
    if suppress is not None:
        out = out.masked_fill(suppress.view(1, -1), NEG)
    out = ngram_ban_pool(out, pools, idx, levels)
    pen = pools.freq_pen[idx].unsqueeze(1)
    out = out - pen * pools.freq[idx].to(out.dtype)
    return out


def _truncate(logits: torch.Tensor, kind: torch.Tensor, min_p: torch.Tensor, top_p: torch.Tensor,
              eta: torch.Tensor, do_top_p: bool = False, do_eta: bool = False) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    thresh = min_p.unsqueeze(1) * probs.max(dim=-1, keepdim=True).values
    out = logits.masked_fill((kind == 1).unsqueeze(1) & (probs < thresh), NEG)
    if do_top_p:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        sp = torch.softmax(sorted_logits.float(), dim=-1)
        drop = (torch.cumsum(sp, dim=-1) - sp) >= top_p.unsqueeze(1)
        drop[:, 0] = False
        kept = sorted_logits.masked_fill(drop, NEG)
        restored = torch.empty_like(logits).scatter_(1, order, kept)
        out = torch.where((kind == 2).unsqueeze(1), restored, out)
    if do_eta:
        logp = torch.log_softmax(logits.float(), dim=-1)
        pr = logp.exp()
        H = -(pr * logp).sum(-1, keepdim=True)
        eps = torch.minimum(eta.unsqueeze(1).expand_as(H), (eta.unsqueeze(1) ** 0.5) * torch.exp(-H))
        eta_out = logits.masked_fill(pr < eps, NEG)
        out = torch.where((kind == 3).unsqueeze(1), eta_out, out)
    return out


def sample_from(logits: torch.Tensor, pools, idx: torch.Tensor, suppress: torch.Tensor | None,
                generator: torch.Generator | None = None, levels: tuple[int, ...] = (2, 3, 4),
                do_top_p: bool = False, do_eta: bool = False, all_greedy: bool = False) -> torch.Tensor:
    """logits [R, V] -> next ids [R]. Greedy slots skip temperature and truncation."""
    processed = apply_penalties(logits, pools, idx, suppress, levels)
    if all_greedy:
        return processed.argmax(dim=-1)
    greedy = pools.greedy[idx]
    temp = pools.temp[idx].clamp(min=1e-4).unsqueeze(1)
    scaled = torch.where(greedy.unsqueeze(1), processed, processed / temp)
    trunc = _truncate(scaled, pools.trunc[idx], pools.min_p[idx], pools.top_p[idx], pools.eta[idx], do_top_p, do_eta)
    scaled = torch.where(greedy.unsqueeze(1), processed, trunc)
    # rows that truncation emptied fall back to the penalised logits (argmax is finite)
    empty = ~torch.isfinite(scaled).any(dim=-1, keepdim=True)
    scaled = torch.where(empty, processed, scaled)
    probs = torch.softmax(scaled.float(), dim=-1)
    drawn = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(greedy, scaled.argmax(dim=-1), drawn)


def sample_seeded(processed_logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Exponential-race (Gumbel-max) sample so a per-request seed does not depend on other rows."""
    probs = torch.softmax(processed_logits.float(), dim=-1).clamp(min=1e-20)
    u = torch.rand(probs.shape, generator=generator, device=probs.device).clamp(1e-8, 1 - 1e-8)
    return (probs.log() - (-u.log()).log()).argmax(dim=-1)
