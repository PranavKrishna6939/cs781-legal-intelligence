"""Logits processors. Repetition controls look only at *generated* tokens, never the prompt.

Penalising prompt tokens would suppress copying parties, sections and citations, which is exactly what a legal
continuation should do. The generation engine passes only the new tokens ([rows, t]) to every processor, so
`prompt_len` defaults to 0. All processors are vectorised over rows (no per-row Python loops).
"""
from __future__ import annotations

import torch

NEG = -1e9


class GeneratedNoRepeatNGram:
    def __init__(self, n: int, prompt_len: int = 0):
        self.n = n
        self.prompt_len = prompt_len

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        gen = generated[:, self.prompt_len:]
        n, L = self.n, gen.shape[1]
        if n <= 0 or L < n:
            return logits
        windows = gen.unfold(1, n, 1)                      # [R, L-n+1, n]
        prefix = gen[:, L - n + 1:]                         # [R, n-1]
        match = (windows[:, :, :-1] == prefix[:, None, :]).all(-1)
        hits = torch.zeros_like(logits).scatter_add_(1, windows[:, :, -1], match.to(logits.dtype))
        return logits.masked_fill(hits > 0, NEG)


class WindowedFrequencyPenalty:
    def __init__(self, window: int, penalty: float, prompt_len: int = 0):
        self.window = window
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        gen = generated[:, self.prompt_len:]
        if self.penalty == 0 or gen.shape[1] == 0:
            return logits
        tail = gen[:, -self.window:]
        counts = torch.zeros_like(logits).scatter_add_(1, tail, torch.ones_like(tail, dtype=logits.dtype))
        return logits - self.penalty * counts


class SuppressTokens:
    """Never emit these ids (control tokens, anonymization tags, reserved ids). They never occur in test text."""

    def __init__(self, ids: list[int]):
        self.ids = sorted(set(int(i) for i in ids))
        self._t: torch.Tensor | None = None

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        if not self.ids:
            return logits
        if self._t is None or self._t.device != logits.device:
            self._t = torch.tensor(self.ids, device=logits.device, dtype=torch.long)
        return logits.index_fill(1, self._t, NEG)


class Temperature:
    def __init__(self, t: float):
        self.t = t

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        return logits if self.t in (0, 1) else logits / self.t


class MinP:
    def __init__(self, p_base: float):
        self.p_base = p_base

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        thresh = self.p_base * probs.max(dim=-1, keepdim=True).values
        return logits.masked_fill(probs < thresh, NEG)


class TopP:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits.float(), dim=-1).cumsum(-1)
        drop = (cum - torch.softmax(sorted_logits.float(), dim=-1)) >= self.p
        drop[..., 0] = False
        return logits.scatter(-1, idx, sorted_logits.masked_fill(drop, NEG))


class EtaSampling:
    """Hewitt et al. (2022): drop tokens with p < min(eta, sqrt(eta) * exp(-H(p)))."""

    def __init__(self, eta: float = 3e-3):
        self.eta = eta

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        H = -(p * logp).sum(-1, keepdim=True)
        eps = torch.minimum(torch.full_like(H, self.eta), (self.eta ** 0.5) * torch.exp(-H))
        return logits.masked_fill(p < eps, NEG)


def sample_from_logits(logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    """logits: [R, V] -> [R]. Temperature is a processor, so sampling is from softmax(logits)."""
    if greedy:
        return logits.argmax(dim=-1)
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), num_samples=1)[:, 0]


def compose(*processors):
    def fn(logits, generated):
        for p in processors:
            if p is not None:
                logits = p(logits, generated)
        return logits
    return fn


def build_logits_fn(cfg: dict, suppress_ids: list[int], temperature: float | None = None):
    """Order: suppress specials -> generated-only repetition penalties -> temperature -> truncation."""
    temp = float(cfg.get("temperature", 0.8) if temperature is None else temperature)
    trunc = cfg.get("truncation", "min_p")
    head = {
        "min_p": lambda: MinP(float(cfg.get("min_p", 0.05))),
        "top_p": lambda: TopP(float(cfg.get("top_p", 0.9))),
        "eta": lambda: EtaSampling(float(cfg.get("eta", 3e-3))),
        "none": lambda: None,
    }[trunc]()
    return compose(
        SuppressTokens(suppress_ids),
        GeneratedNoRepeatNGram(int(cfg.get("no_repeat_ngram", 4))),
        WindowedFrequencyPenalty(int(cfg.get("frequency_window", 64)), float(cfg.get("frequency_penalty", 0.3))),
        Temperature(temp) if temp > 0 else None,
        head if temp > 0 else None,
    )
