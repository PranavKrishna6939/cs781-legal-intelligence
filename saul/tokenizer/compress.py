"""Engram tokenizer compression: map every token id to a canonical id for its NFKC-lowercased, left-stripped text.

"The", " the" and "THE" share one Engram key, so n-gram tables are not split across surface variants. Special
tokens (ids below `n_special`) map to themselves.
"""
from __future__ import annotations

import unicodedata

import torch


def build_compression_map(tokenizer, vocab_size: int, n_special: int = 64) -> torch.Tensor:
    canon: dict[str, int] = {}
    out = torch.arange(vocab_size, dtype=torch.long)
    for i in range(n_special, vocab_size):
        text = tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        key = unicodedata.normalize("NFKC", text).lower().lstrip()
        if not key or "\ufffd" in text:
            continue
        out[i] = canon.setdefault(key, i)
    return out
