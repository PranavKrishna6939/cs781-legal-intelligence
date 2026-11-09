"""Pack prompts into one row and describe the resulting segments. No model imports (gdn calls this too)."""
from __future__ import annotations


def segment_spans(doc_ids) -> list[tuple[int, int, int]]:
    """doc_ids: 1-D tensor or list. Returns (start, end, doc_id) for each contiguous run."""
    if hasattr(doc_ids, "tolist"):
        docs = [int(x) for x in doc_ids.tolist()]
    else:
        docs = [int(x) for x in doc_ids]
    if not docs:
        return []
    spans = []
    start = 0
    for i in range(1, len(docs) + 1):
        if i == len(docs) or docs[i] != docs[start]:
            spans.append((start, i, docs[start]))
            start = i
    return spans


def pack_prompts(token_rows: list[list[int]], pad_id: int, multiple: int = 128) -> dict:
    """One packed row. A trailing pad segment (doc id -1) brings the length up to a multiple of `multiple`."""
    ids: list[int] = []
    docs: list[int] = []
    pos: list[int] = []
    for i, row in enumerate(token_rows):
        ids.extend(row)
        docs.extend([i] * len(row))
        pos.extend(range(len(row)))
    if multiple > 1:
        extra = (-len(ids)) % multiple
        if extra:
            ids.extend([pad_id] * extra)
            docs.extend([-1] * extra)
            pos.extend(range(extra))
    return {"ids": ids, "docs": docs, "pos": pos}
