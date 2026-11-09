"""Label and BIOES tag layout. Tag 0 is O; then four tags (B, I, E, S) per label."""
from __future__ import annotations

import torch

LABELS = [
    "APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
    "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO",
]
RARE_LABELS = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT")
PERSON_LABELS = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT")
SUBTOKEN_LABELS = ("STAT", "AUTH", "COURT")
N_LABELS = len(LABELS)
NONE = N_LABELS  # span-head "not a span" class
N_SPAN = N_LABELS + 1
O = 0
N_TAGS = 1 + N_LABELS * 4  # 49
IGNORE = -100
KINDS = ("B", "I", "E", "S")
SEP_ID = 4
PAD_ID = 2


def tag_id(label_index: int, kind: str) -> int:
    return 1 + label_index * 4 + KINDS.index(kind)


def tag_parts(tag: int) -> tuple[str, int | None]:
    if tag <= 0:
        return "O", None
    return KINDS[(tag - 1) % 4], (tag - 1) // 4


def label_id(name: str) -> int:
    return LABELS.index(name)


def legal_transition_masks(n_labels: int = N_LABELS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boolean masks: transitions[prev, next], start[tag], end[tag]."""
    k = 1 + n_labels * 4
    ok = torch.zeros(k, k, dtype=torch.bool)
    start = torch.zeros(k, dtype=torch.bool)
    end = torch.zeros(k, dtype=torch.bool)

    def kind_of(t: int) -> str:
        return "O" if t == 0 else KINDS[(t - 1) % 4]

    def lab_of(t: int) -> int:
        return -1 if t == 0 else (t - 1) // 4

    for t in range(k):
        kd = kind_of(t)
        start[t] = kd in ("O", "B", "S")
        end[t] = kd in ("O", "E", "S")
    for i in range(k):
        ki, li = kind_of(i), lab_of(i)
        for j in range(k):
            kj, lj = kind_of(j), lab_of(j)
            if ki in ("O", "E", "S"):
                ok[i, j] = kj in ("O", "B", "S")
            else:  # B or I must continue the same label
                ok[i, j] = lj == li and kj in ("I", "E")
    return ok, start, end


def spans_from_tags(tags: list[int], starts: list[int], ends: list[int]) -> list[tuple[int, int, str]]:
    """Decode a BIOES tag sequence into character spans. Illegal tags are skipped."""
    out = []
    i = 0
    n = len(tags)
    while i < n:
        kind, lab = tag_parts(int(tags[i]))
        if kind == "S" and lab is not None and starts[i] >= 0:
            out.append((int(starts[i]), int(ends[i]), LABELS[lab]))
            i += 1
            continue
        if kind == "B" and lab is not None and starts[i] >= 0:
            j = i + 1
            closed = False
            while j < n:
                kj, lj = tag_parts(int(tags[j]))
                if lj != lab or kj not in ("I", "E"):
                    break
                if kj == "E":
                    out.append((int(starts[i]), int(ends[j]), LABELS[lab]))
                    i = j + 1
                    closed = True
                    break
                j += 1
            if not closed:
                i += 1
            continue
        i += 1
    return out
