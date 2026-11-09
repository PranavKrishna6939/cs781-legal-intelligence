"""Phase 6 sequences: whole-document prompt, length-bucket token, packed rows.

Sequence: `<|bos|> document <|summary|> <|len_b|> summary <|eos|>`.
The loss mask is 1 on the length token, the summary and `<|eos|>` only.
`next_token_targets` trains a position when the mask is 1 on the *next* token.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

# Fifteen edges, sixteen buckets: below 150, then each interval, then 3300 and above.
BUCKET_EDGES = (150, 200, 260, 330, 420, 520, 640, 780, 950, 1150, 1400, 1700, 2100, 2600, 3300)
_CAPTION = re.compile(r"(?i)(?:the judgment of the court was delivered by|\bjudgment\b)")
_WORD = re.compile(r"[a-z0-9]+")


def p6_dir() -> str:
    env = os.environ.get("SAUL_P6_DIR")
    if env:
        return env
    for path in ("/data/cs781-p6", os.path.expanduser("~/cs781/p6")):
        if os.path.isdir(path):
            return path
    return "/data/cs781-p6"


def data_dir() -> str:
    return os.environ.get("SAUL_P6_DATA", os.path.expanduser("~/cs781/data/raw/Phase_6_Summ"))


def bucket_of(n_words: int) -> int:
    return int(np.digitize(np.int64(n_words), BUCKET_EDGES))


def bucket_midpoints() -> list[float]:
    """One representative word count per bucket, for turning a bucket id into a stop length."""
    edges = [0, *BUCKET_EDGES, int(BUCKET_EDGES[-1] * 1.4)]
    return [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]


def reserved_length_ids(tok) -> list[int]:
    """First 16 `<|reserved_k|>` ids that are actually in the tokenizer. Fails closed."""
    vocab = tok.get_vocab()
    found = []
    for i in range(64):
        name = f"<|reserved_{i}|>"
        if name in vocab:
            found.append(int(vocab[name]))
    if len(found) < 16:
        raise RuntimeError(f"need 16 <|reserved_k|> ids, found {len(found)}")
    return found[:16]


def caption_cut(text: str) -> int:
    """Words to drop: through the first judgment marker inside the first 300 words, else 120."""
    words = text.split()
    if not words:
        return 0
    head = " ".join(words[:300])
    match = _CAPTION.search(head)
    if match is None:
        return min(120, len(words))
    return min(len(words), len(head[: match.end()].split()))


def strip_caption(text: str) -> str:
    words = text.split()
    cut = caption_cut(text)
    if cut <= 0 or cut >= len(words):
        return text
    return " ".join(words[cut:])


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.split()) > 3]


def caption_is_safe(rows: list[dict], n: int = 20) -> bool:
    """True when no summary sentence is mostly copied from the dropped caption (first `n` docs)."""
    for row in rows[:n]:
        words = row["document"].split()
        cut = caption_cut(row["document"])
        caption = set(_WORD.findall(" ".join(words[:cut]).lower()))
        if len(caption) < 5:
            continue
        for sent in _sentences(row.get("summary", "")):
            toks = set(_WORD.findall(sent.lower()))
            if len(toks) >= 5 and len(toks & caption) / len(toks) > 0.5:
                return False
    return True


def fit_document(doc_ids: list[int], budget: int) -> list[int]:
    """Keep the document when it fits. Otherwise the first 65% and last 35% of the budget."""
    if budget < 1:
        return []
    if len(doc_ids) <= budget:
        return list(doc_ids)
    head = int(budget * 0.65)
    tail = budget - head
    if tail <= 0:
        return list(doc_ids[:budget])
    return list(doc_ids[:head]) + list(doc_ids[-tail:])


def build_sequence(
    doc_ids: list[int],
    summary_ids: list[int],
    bos: int,
    summary_id: int,
    len_id: int,
    eos: int,
    lmax: int,
) -> tuple[list[int], list[int]]:
    """Training row. The mask is 1 from the length token through `<|eos|>`.

    A summary longer than the budget is cut from the end. The document then fills what remains,
    keeping its head and tail when it still does not fit.
    """
    summary_ids = list(summary_ids[: max(1, lmax - 4)])
    body = fit_document(doc_ids, lmax - 4 - len(summary_ids))
    seq = [bos, *body, summary_id, len_id, *summary_ids, eos]
    prompt = 1 + len(body) + 1  # bos + document + summary marker
    mask = [0] * prompt + [1] * (len(seq) - prompt)
    if len(seq) > lmax:
        raise RuntimeError(f"sequence {len(seq)} exceeds lmax {lmax}")
    return seq, mask


def prompt_ids(doc_ids: list[int], summary_id: int, lmax: int, room: int) -> list[int]:
    """Decode prompt without BOS. The engine prepends BOS. `room` is tokens reserved for the summary."""
    # bos (engine) + length token + eos, plus the summary room
    body = fit_document(doc_ids, lmax - (3 + room))
    return [*body, summary_id]


def pack_sequences(pairs: list[tuple[list[int], list[int]]], token_budget: int) -> list[dict]:
    """Pack variable-length rows into sequences. `doc_ids` restart so documents cannot attend across."""
    rows: list[dict] = []
    cur: list[tuple[list[int], list[int]]] = []
    cost = 0
    for pair in pairs:
        if cur and cost + len(pair[0]) > token_budget:
            rows.append(_pack(cur))
            cur, cost = [], 0
        cur.append(pair)
        cost += len(pair[0])
    if cur:
        rows.append(_pack(cur))
    return rows


def _pack(parts: list[tuple[list[int], list[int]]]) -> dict:
    ids: list[int] = []
    mask: list[int] = []
    docs: list[int] = []
    pos: list[int] = []
    for i, (seq, loss) in enumerate(parts):
        ids.extend(seq)
        mask.extend(loss)
        docs.extend([i] * len(seq))
        pos.extend(range(len(seq)))
    return {"input_ids": ids, "loss_mask": mask, "doc_ids": docs, "position_ids": pos}


def collate(rows: list[dict], pad_id: int) -> dict:
    """Pad packed rows and build `cu_seqlens`. Pad positions get doc id -1 so they are not a segment."""
    from saul.data.loader import cu_seqlens_np

    width = max(len(row["input_ids"]) for row in rows)
    batch = len(rows)
    ids = np.full((batch, width), pad_id, dtype=np.int64)
    mask = np.zeros((batch, width), dtype=np.int64)
    docs = np.full((batch, width), -1, dtype=np.int64)
    pos = np.zeros((batch, width), dtype=np.int64)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = row["input_ids"]
        mask[i, :n] = row["loss_mask"]
        docs[i, :n] = row["doc_ids"]
        pos[i, :n] = row["position_ids"]
    return {
        "input_ids": ids,
        "loss_mask": mask,
        "doc_ids": docs,
        "position_ids": pos,
        "cu_seqlens": cu_seqlens_np(docs),
    }


def load_rows(split: str) -> list[dict]:
    name = "Train" if split == "train" else "Test"
    path = os.path.join(data_dir(), name, f"{split}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def holdout_ids() -> set[str]:
    path = os.path.join(p6_dir(), "split.json")
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle)["holdout_ids"])


def norm_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def gram_index(texts: list[str], n: int = 13, step: int = 7) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for j, text in enumerate(texts):
        words = norm_words(text)
        for i in range(0, max(0, len(words) - n), step):
            index[" ".join(words[i : i + n])].add(j)
    return index


def matching_query_ids(
    queries: list[dict],
    index: dict[str, set[int]],
    n: int = 13,
    min_hits: int = 10,
) -> list[str]:
    """Query ids whose text shares at least `min_hits` sampled n-grams with the index."""
    hit = []
    for row in queries:
        counts: dict[int, int] = defaultdict(int)
        words = norm_words(row["document"])
        for i in range(0, max(0, len(words) - n), max(1, (len(words) - n) // 40 or 1)):
            for j in index.get(" ".join(words[i : i + n]), ()):
                counts[j] += 1
        if counts and max(counts.values()) >= min_hits:
            hit.append(row["id"])
    return hit


def write_test_overlap(sc_path: str, out_path: str, min_hits: int = 10) -> list[str]:
    """Ids of Phase 6 test documents found in the Phase 1 Supreme Court file. Does not read headnotes."""
    test = load_rows("test")
    records = json.load(open(sc_path, encoding="utf-8"))
    texts = [row.get("text") or "" for row in records]
    found = matching_query_ids(test, gram_index(texts), min_hits=min_hits)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"ids": found, "n": len(found), "min_hits": min_hits}, handle, indent=2)
    return found


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--sc", default="/data/cs781-archive/recovery_20261005/official_raw/Phase_1_Pretraining/Train/train_supreme_court.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.overlap:
        raise SystemExit("pass --overlap")
    dest = args.out or os.path.join(p6_dir(), "sc_test_overlap.json")
    found = write_test_overlap(args.sc, dest)
    print(f"[p6] test overlap {len(found)} -> {dest}")


def repeat_rate(text: str) -> float:
    words = norm_words(text)
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


if __name__ == "__main__":
    main()
