"""Special-token layout, frozen for all phases (report_saul.md 6.5).

ids 0..63 are special: 11 named control/task markers, then the IL-TUR anonymization placeholders exactly as they
appear in the downstream files, then `<|reserved_k|>` fillers. Reserving them now avoids resizing the tied
embedding later (which would break the "start from your Phase-1 model" requirement).
"""
from __future__ import annotations

NAMED_SPECIALS = [
    "<|bos|>",      # 0  document start
    "<|eos|>",      # 1  document end
    "<|pad|>",      # 2
    "<|mask|>",     # 3  MNTP mask (used in pretraining)
    "<|sep|>",      # 4
    "<|cls|>",      # 5  pooled classification (Phases 3-5, 7)
    "<|doc|>",      # 6
    "<|summary|>",  # 7  Phase 6 prompt marker
    "<|query|>",    # 8  Phase 7
    "<|cand|>",     # 9  Phase 7
    "<|sent|>",     # 10 Phase 3 sentence marker
]
N_SPECIAL = 64
BOS, EOS, PAD, MASK = "<|bos|>", "<|eos|>", "<|pad|>", "<|mask|>"


def build_special_tokens(anon_tags: list[str]) -> list[str]:
    tags = [t for t in dict.fromkeys(anon_tags) if t not in NAMED_SPECIALS]
    out = NAMED_SPECIALS + tags
    if len(out) > N_SPECIAL:
        raise ValueError(f"{len(out)} specials exceed the reserved block of {N_SPECIAL}")
    out += [f"<|reserved_{i}|>" for i in range(N_SPECIAL - len(out))]
    assert len(out) == N_SPECIAL
    return out


def zero_byte_specials(specials: list[str]) -> set[str]:
    """Specials that do not correspond to text bytes (control markers); anonymization tags *are* text."""
    return {s for s in specials if s.startswith("<|")}
