"""Batched continuation with a KV / recurrent-state cache.

Prompts are left-padded and prefilled once; for n samples per prompt the cache is fanned out (no repeated
prefill). Each step decodes one token for every row. Special tokens are never sampled (they never occur in the
test text), so generation stops on length: a per-token table counts whitespace-separated words incrementally
(no detokenisation per step) and a row is done once its continuation has `target_words` words. The count is
verified on the decoded text before stopping.
"""
from __future__ import annotations

import torch

from saul.decode.postprocess import to_128_words
from saul.decode.processors import sample_from_logits
from saul.tokenizer.specials import N_SPECIAL


class WordCounter:
    """Per-token whitespace structure, so word counts update with three table lookups per step."""

    def __init__(self, tokenizer, vocab_size: int, device):
        starts = torch.zeros(vocab_size, dtype=torch.long)
        first_nonws = torch.zeros(vocab_size, dtype=torch.bool)
        last_ws = torch.zeros(vocab_size, dtype=torch.bool)
        empty = torch.ones(vocab_size, dtype=torch.bool)
        special = set(suppressed_ids(tokenizer, vocab_size))
        for i in range(vocab_size):
            if i in special:
                continue
            s = tokenizer.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not s:
                continue
            empty[i] = False
            first_nonws[i] = not s[0].isspace()
            last_ws[i] = s[-1].isspace()
            starts[i] = sum(1 for a, b in zip(s, s[1:]) if a.isspace() and not b.isspace())
        self.starts, self.first_nonws = starts.to(device), first_nonws.to(device)
        self.last_ws, self.empty = last_ws.to(device), empty.to(device)

    def init(self, rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(rows, dtype=torch.long, device=device), torch.ones(rows, dtype=torch.bool, device=device)

    def update(self, ids: torch.Tensor, words: torch.Tensor, prev_ws: torch.Tensor):
        words = words + self.starts[ids] + (self.first_nonws[ids] & prev_ws).long()
        prev_ws = torch.where(self.empty[ids], prev_ws, self.last_ws[ids])
        return words, prev_ws


def suppressed_ids(tokenizer, vocab_size: int) -> list[int]:
    """The 64 reserved ids, the tokenizer's named specials, and every special added token (e.g. `<|fill_k|>`)."""
    ids = set(range(min(N_SPECIAL, vocab_size)))
    ids |= {int(i) for i in getattr(tokenizer, "all_special_ids", []) if i is not None}
    ids |= {int(i) for i, t in getattr(tokenizer, "added_tokens_decoder", {}).items() if getattr(t, "special", False)}
    return sorted(i for i in ids if i < vocab_size)


def _encode(tokenizer, prompts: list[str], prepend_bos: bool, device):
    bos, pad = tokenizer.bos_token_id, tokenizer.pad_token_id
    enc = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if prepend_bos and (not ids or ids[0] != bos):
            ids = [bos] + ids
        enc.append(ids)
    T = max(len(x) for x in enc)
    ids = torch.full((len(enc), T), pad, dtype=torch.long)
    valid = torch.zeros((len(enc), T), dtype=torch.bool)
    for i, x in enumerate(enc):
        ids[i, T - len(x):] = torch.tensor(x, dtype=torch.long)
        valid[i, T - len(x):] = True
    return ids.to(device), valid.to(device)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int = 1,
    max_new_tokens: int = 400,
    target_words: int = 128,
    logits_fn=None,
    greedy: bool = False,
    prepend_bos: bool = True,
    device: str | None = None,
    counter: WordCounter | None = None,
) -> list[list[str]]:
    """Returns, for each prompt, `n_samples` continuations.

    Sampling is the engine's batched processors (suppress specials, generated-only n-gram block, windowed frequency
    penalty, temperature, min-p), which is what `build_logits_fn` applies by default. `logits_fn` and `counter`
    are accepted so older callers keep working; a custom `logits_fn` is not replayed token by token.
    """
    del logits_fn, counter
    from saul.serve.api import continuations
    device = device or str(next(model.parameters()).device)
    return continuations(model, tokenizer, prompts, n_samples, max_new_tokens, target_words, greedy, device,
                         prepend_bos)


def _decode(tokenizer, new: torch.Tensor) -> list[str]:
    return tokenizer.batch_decode(new.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)


def continue_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, target_words: int = 128,
                   logits_fn=None, greedy: bool = False, prepend_bos: bool = True, device: str | None = None) -> list[str]:
    """One continuation per prompt (kept for callers of the old API)."""
    res = generate(model, tokenizer, prompts, 1, max_new_tokens, target_words, logits_fn, greedy, prepend_bos, device)
    return [r[0] for r in res]
