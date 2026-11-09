"""Turn token ids into the evaluator's 128-whitespace-word continuation."""
from __future__ import annotations


def to_128_words(text: str, n: int = 128) -> str:
    words = text.split()
    if len(words) < n:
        raise ValueError(f"generation has {len(words)} words, need {n}")
    return " ".join(words[:n])


def strip_specials(text: str, specials: list[str]) -> str:
    for s in specials:
        text = text.replace(s, "")
    return text.strip()
