"""Whitespace word counting for early stop, and the stop rule.

A slot stops for length when word `target + 1` has started, or the token that completed word `target` ended in
whitespace. Stopping when word `target` merely starts would keep a half-written last word.
"""
from __future__ import annotations

import torch

from saul.tokenizer.specials import N_SPECIAL


def suppressed_ids(tokenizer, vocab_size: int) -> list[int]:
    ids = set(range(min(N_SPECIAL, vocab_size)))
    ids |= {int(i) for i in getattr(tokenizer, "all_special_ids", []) if i is not None}
    ids |= {int(i) for i, t in getattr(tokenizer, "added_tokens_decoder", {}).items() if getattr(t, "special", False)}
    return sorted(i for i in ids if i < vocab_size)


class WordCounter:
    def __init__(self, tokenizer, vocab_size: int, device):
        starts = torch.zeros(vocab_size, dtype=torch.long)
        first_nonws = torch.zeros(vocab_size, dtype=torch.bool)
        last_ws = torch.zeros(vocab_size, dtype=torch.bool)
        empty = torch.ones(vocab_size, dtype=torch.bool)
        special = set(suppressed_ids(tokenizer, vocab_size))
        decode = tokenizer.decode
        for i in range(vocab_size):
            if i in special:
                continue
            try:
                s = decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                s = decode([i])
            if not s:
                continue
            empty[i] = False
            first_nonws[i] = not s[0].isspace()
            last_ws[i] = s[-1].isspace()
            starts[i] = sum(1 for a, b in zip(s, s[1:]) if a.isspace() and not b.isspace())
        self.starts = starts.to(device)
        self.first_nonws = first_nonws.to(device)
        self.last_ws = last_ws.to(device)
        self.empty = empty.to(device)

    def update(self, ids: torch.Tensor, words: torch.Tensor, prev_ws: torch.Tensor):
        words = words + self.starts[ids] + (self.first_nonws[ids] & prev_ws).long()
        prev_ws = torch.where(self.empty[ids], prev_ws, self.last_ws[ids])
        return words, prev_ws


# finish codes: 1 words, 2 length, 3 stop
def stop_flags(words, prev_ws, slen, nxt, target, max_new, stop_ids: torch.Tensor):
    word_done = (target >= 0) & ((words > target) | ((words == target) & prev_ws))
    length_done = slen >= max_new
    stop_done = (nxt[:, None] == stop_ids.view(1, -1)).any(dim=1) if stop_ids.numel() else torch.zeros_like(word_done)
    done = word_done | length_done | stop_done
    reason = torch.zeros_like(slen, dtype=torch.int8)
    reason = torch.where(word_done, torch.ones_like(reason), reason)
    reason = torch.where(length_done & ~word_done, torch.full_like(reason, 2), reason)
    reason = torch.where(stop_done & ~word_done & ~length_done, torch.full_like(reason, 3), reason)
    return done, reason
