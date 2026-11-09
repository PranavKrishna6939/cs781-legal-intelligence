"""Phase 5 cases, stratified halves, token views, and batches."""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

from saul.lsi import exp_dir, p5_dir, raw_dir
from saul.lsi.const import BOS_ID, BUCKETS, CHUNK, MAX_CHUNKS, N_LABELS, PAD_ID, SEED, SUBSET
from saul.utils.io import write_json


def text_hash(text: str) -> str:
    folded = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.md5(folded.encode()).hexdigest()


def dedupe_labels(labels) -> list[int]:
    seen = []
    for lab in labels or []:
        lab = int(lab)
        if lab not in seen:
            seen.append(lab)
    return seen


def _read_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    out = []
    for row in rows:
        item = {"id": str(row["id"]), "text": row["text"]}
        if "labels" in row and row["labels"] is not None:
            item["labels"] = dedupe_labels(row["labels"])
        out.append(item)
    return out


def load_raw() -> dict[str, list[dict]]:
    root = raw_dir()
    return {
        "train": _read_json(os.path.join(root, "Train", "train.json")),
        "dev": _read_json(os.path.join(root, "Train", "dev.json")),
        "test": _read_json(os.path.join(root, "Test", "test.json")),
    }


def duplicate_labels(query: list[dict], train: list[dict]) -> list[list[int] | None]:
    """Train labels of an exact text match, or None when the query text is not in train."""
    return lookup_duplicates(query, train)


def lookup_duplicates(query: list[dict], *corpora: list[dict]) -> list[list[int] | None]:
    """Union of labels from every corpus whose text matches. None when nothing matches.

    Dev scoring passes train only. Test scoring may also pass dev: those labels are known.
    """
    tables: list[dict[str, list[int]]] = []
    for corpus in corpora:
        table: dict[str, list[int]] = {}
        for row in corpus:
            labels = table.setdefault(text_hash(row["text"]), [])
            for lab in row.get("labels", []):
                lab = int(lab)
                if lab not in labels:
                    labels.append(lab)
        tables.append(table)
    out: list[list[int] | None] = []
    for row in query:
        found: list[int] = []
        digest = text_hash(row["text"])
        hit = False
        for table in tables:
            labels = table.get(digest)
            if not labels:
                continue
            hit = True
            for lab in labels:
                if lab not in found:
                    found.append(lab)
        out.append(sorted(found) if hit else None)
    return out


def hash_groups(rows: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(text_hash(row["text"]), []).append(index)
    return groups


def overlap_stats(splits: dict[str, list[dict]]) -> dict:
    hashed = {name: {} for name in splits}
    for name, rows in splits.items():
        for row in rows:
            hashed[name].setdefault(text_hash(row["text"]), row)
    ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}

    def n_overlap(a: str, b: str) -> int:
        return len(set(hashed[a]) & set(hashed[b]))

    dev_sets: dict[str, list[list[int]]] = {}
    for row in splits["dev"]:
        dev_sets.setdefault(text_hash(row["text"]), []).append(sorted(row.get("labels", [])))
    conflicts = 0
    for row in splits["train"]:
        options = dev_sets.get(text_hash(row["text"]))
        if options and sorted(row.get("labels", [])) not in options:
            conflicts += 1
    within = {name: len(rows) - len(hashed[name]) for name, rows in splits.items()}
    return {
        "n": {name: len(rows) for name, rows in splits.items()},
        "id_overlap": {
            "train_dev": len(ids["train"] & ids["dev"]),
            "train_test": len(ids["train"] & ids["test"]),
            "dev_test": len(ids["dev"] & ids["test"]),
        },
        "text_overlap": {
            "train_dev": n_overlap("train", "dev"),
            "train_test": n_overlap("train", "test"),
            "dev_test": n_overlap("dev", "test"),
        },
        "within_split_duplicate_texts": within,
        "train_dev_label_conflicts": conflicts,
    }


class _Group:
    def __init__(self, indexes: list[int], counts: np.ndarray):
        self.indexes = indexes
        self.counts = counts
        self.labels = set(int(i) for i in np.flatnonzero(counts))
        self.size = len(indexes)


def iterative_stratify(groups: list[_Group], n_labels: int = N_LABELS, seed: int = SEED) -> list[str]:
    """Sechidis iterative stratification. Returns 'A' or 'B' per original row index."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    remaining = set(range(len(groups)))
    sides: list[list[int]] = [[], []]
    counts = [np.zeros(n_labels, dtype=np.int64), np.zeros(n_labels, dtype=np.int64)]
    sizes = [0, 0]
    while remaining:
        present = np.zeros(n_labels, dtype=np.int64)
        for index in remaining:
            present += groups[index].counts
        owned = np.flatnonzero(present > 0)
        if owned.size == 0:
            choice = list(remaining)
        else:
            label = int(owned[np.argmin(present[owned])])
            owners = [index for index in remaining if label in groups[index].labels]
            choice = [max(owners, key=lambda index: groups[index].size)]
            if counts[0][label] < counts[1][label] or (
                counts[0][label] == counts[1][label] and sizes[0] <= sizes[1]
            ):
                side = 0
            else:
                side = 1
        if owned.size == 0:
            side = 0 if sizes[0] <= sizes[1] else 1
        for index in choice:
            sides[side].append(index)
            sizes[side] += groups[index].size
            counts[side] += groups[index].counts
            remaining.remove(index)
    half = {}
    for side, name in ((0, "A"), (1, "B")):
        for index in sides[side]:
            for row in groups[index].indexes:
                half[row] = name
    return [half[i] for i in range(sum(group.size for group in groups))]


def dev_groups(rows: list[dict]) -> list[_Group]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(text_hash(row["text"]), []).append(index)
    groups = []
    for indexes in grouped.values():
        counts = np.zeros(N_LABELS, dtype=np.int64)
        for index in indexes:
            for lab in rows[index].get("labels", []):
                counts[int(lab)] += 1
        groups.append(_Group(indexes, counts))
    return groups


def half_balance(rows: list[dict], half: list[str]) -> dict:
    counts = {"A": np.zeros(N_LABELS, dtype=np.int64), "B": np.zeros(N_LABELS, dtype=np.int64)}
    for row, name in zip(rows, half):
        for lab in row.get("labels", []):
            counts[name][lab] += 1
    gap = np.abs(counts["A"] - counts["B"])
    return {
        "max_gap": int(gap.max()) if gap.size else 0,
        "labels_over_1": int((gap > 1).sum()),
        "labels_in_A": int((counts["A"] > 0).sum()),
        "labels_in_B": int((counts["B"] > 0).sum()),
        "n_A": int(sum(name == "A" for name in half)),
        "n_B": int(sum(name == "B" for name in half)),
    }


def groups_intact(rows: list[dict], half: list[str]) -> bool:
    seen: dict[str, str] = {}
    for row, name in zip(rows, half):
        digest = text_hash(row["text"])
        if digest in seen and seen[digest] != name:
            return False
        seen[digest] = name
    return True


def g0_report(rows: list[dict], half: list[str]) -> dict:
    balance = half_balance(rows, half)
    intact = groups_intact(rows, half)
    passed = (
        balance["max_gap"] <= 1
        and intact
        and balance["labels_in_A"] == N_LABELS
        and balance["labels_in_B"] == N_LABELS
    )
    return {"pass": passed, "groups_intact": intact, **balance}


def _even_indices(n: int, k: int) -> list[int]:
    """k distinct positions spread across ``0 .. n-1``. Rounding a linspace can collide."""
    if k >= n:
        return list(range(n))
    raw = [int(round(float(x))) for x in np.linspace(0, n - 1, k)]
    picked: list[int] = []
    used: set[int] = set()
    for idx in raw:
        if idx not in used:
            picked.append(idx)
            used.add(idx)
    if len(picked) < k:
        for idx in range(n):
            if idx not in used:
                picked.append(idx)
                used.add(idx)
            if len(picked) == k:
                break
        picked.sort()
    return picked


def view_chunks(token_ids: np.ndarray, size: int = CHUNK, max_chunks: int = MAX_CHUNKS,
                bos: int = BOS_ID) -> list[list[int]]:
    """Non-overlapping chunks. Each chunk starts with bos. Beyond max_chunks, keep the first half
    and an even sample of the rest."""
    content = max(1, size - 1)
    ids = [int(x) for x in token_ids]
    pieces = [ids[start:start + content] for start in range(0, len(ids), content)] or [[]]
    if len(pieces) > max_chunks:
        head_n = max_chunks // 2
        rest = pieces[head_n:]
        pick = _even_indices(len(rest), max_chunks - head_n)
        pieces = pieces[:head_n] + [rest[i] for i in pick]
    return [[bos, *piece] for piece in pieces]


def view_head(token_ids: np.ndarray, limit: int, bos: int = BOS_ID) -> list[int]:
    return [bos, *[int(x) for x in token_ids[: max(0, limit - 1)]]]


def covered_tokens(token_ids: np.ndarray, size: int = CHUNK, max_chunks: int = MAX_CHUNKS) -> int:
    content = max(1, size - 1)
    return min(len(token_ids), content * max_chunks)


def assign_bucket(length: int, buckets: tuple[int, ...] = BUCKETS) -> int:
    for width in buckets:
        if length <= width:
            return width
    return buckets[-1]


def multi_hot(labels: list[int], n_labels: int = N_LABELS) -> np.ndarray:
    out = np.zeros(n_labels, dtype=np.float32)
    for lab in labels:
        if 0 <= int(lab) < n_labels:
            out[int(lab)] = 1.0
    return out


def label_frequencies(rows: list[dict]) -> np.ndarray:
    freq = np.zeros(N_LABELS, dtype=np.float64)
    for row in rows:
        for lab in row.get("labels", []):
            freq[int(lab)] += 1
    return freq


def rare_weights(label_sets: list[list[int]], freq: np.ndarray) -> np.ndarray:
    """Mean of freq^-0.5 over a case's labels, normalised to mean 1."""
    scale = np.power(np.maximum(freq, 1.0), -0.5)
    weights = np.ones(len(label_sets), dtype=np.float64)
    for index, labels in enumerate(label_sets):
        if labels:
            weights[index] = float(np.mean([scale[int(lab)] for lab in labels]))
    return weights / max(weights.mean(), 1e-8)


def cooccurrence(rows: list[dict], min_count: int = 200) -> dict:
    freq = label_frequencies(rows)
    pair = np.zeros((N_LABELS, N_LABELS), dtype=np.int64)
    for row in rows:
        labels = sorted(set(int(lab) for lab in row.get("labels", [])))
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                pair[left, right] += 1
                pair[right, left] += 1
    conditional = np.zeros((N_LABELS, N_LABELS), dtype=np.float32)
    for left in range(N_LABELS):
        if freq[left] <= 0:
            continue
        for right in range(N_LABELS):
            if left != right and pair[left, right] >= min_count:
                conditional[left, right] = pair[left, right] / freq[left]
    top = []
    for left in range(N_LABELS):
        for right in range(left + 1, N_LABELS):
            if pair[left, right] > 0:
                top.append((int(pair[left, right]), left, right))
    top.sort(reverse=True)
    return {
        "top_pairs": [{"count": count, "labels": [left, right]} for count, left, right in top[:30]],
        "conditional": conditional,
        "freq": freq,
    }


def subset_indices(n: int, k: int = SUBSET, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, n)
    return np.sort(rng.choice(n, size=k, replace=False))


def cf2_mask(sources: list[str], halves: list[str | None], fold: str) -> np.ndarray:
    """Train plus one dev half. `fold` is the half that is trained on."""
    keep = []
    for index, source in enumerate(sources):
        if source == "train":
            keep.append(True)
        elif source == "dev" and halves[index] == fold:
            keep.append(True)
        else:
            keep.append(False)
    return np.asarray(keep, dtype=bool)


def make_batches(lengths: list[int], widths: list[int], token_budget: int,
                 rng: np.random.Generator | None = None, weights: np.ndarray | None = None,
                 ) -> list[tuple[int, list[int]]]:
    """Group cases that share a chunk-width bucket. A batch stays inside `token_budget` tokens."""
    rng = rng or np.random.default_rng(SEED)
    by_width: dict[int, list[int]] = {}
    for index, width in enumerate(widths):
        by_width.setdefault(width, []).append(index)
    batches = []
    for width, indexes in by_width.items():
        indexes = list(indexes)
        if weights is None:
            rng.shuffle(indexes)
        else:
            prob = np.asarray(weights)[indexes]
            prob = prob / prob.sum()
            indexes = list(rng.choice(indexes, size=len(indexes), replace=True, p=prob))
        bucket: list[int] = []
        tokens = 0
        for index in indexes:
            need = int(lengths[index]) * width
            if bucket and tokens + need > token_budget:
                batches.append((width, bucket))
                bucket, tokens = [], 0
            bucket.append(index)
            tokens += need
        if bucket:
            batches.append((width, bucket))
    rng.shuffle(batches)
    return batches


def collate(cases: list[dict], width: int) -> dict:
    """Stack every chunk of the batch. `case_index[k]` says which case chunk k belongs to."""
    chunks = []
    case_index = []
    ids = []
    labels = []
    for case_i, case in enumerate(cases):
        ids.append(case["id"])
        labels.append(case["labels"])
        for chunk in case["chunks"]:
            chunks.append(chunk[:width])
            case_index.append(case_i)
    if not chunks:
        raise ValueError("empty batch")
    input_ids = np.full((len(chunks), width), PAD_ID, dtype=np.int64)
    doc_ids = np.full((len(chunks), width), -1, dtype=np.int64)
    for row, chunk in enumerate(chunks):
        n = min(len(chunk), width)
        input_ids[row, :n] = chunk[:n]
        doc_ids[row, :n] = 0
    import torch
    batch = {
        "input_ids": torch.from_numpy(input_ids),
        "doc_ids": torch.from_numpy(doc_ids),
        "case_index": torch.tensor(case_index, dtype=torch.long),
        "labels": torch.from_numpy(np.stack([multi_hot(lab) for lab in labels])),
        "ids": ids,
        "n_cases": len(cases),
    }
    if cases and "seg_ids" in cases[0]:
        seg_index = []
        for case in cases:
            segs = case.get("seg_ids") or [0] * len(case["chunks"])
            for seg in segs:
                seg_index.append(int(seg))
        batch["seg_index"] = torch.tensor(seg_index, dtype=torch.long)
    if cases and "zlex" in cases[0]:
        batch["zlex"] = torch.from_numpy(np.stack([case["zlex"] for case in cases]).astype(np.float32))
    if cases and "nbr_labels" in cases[0]:
        batch["nbr_labels"] = torch.from_numpy(np.stack([case["nbr_labels"] for case in cases]).astype(np.float32))
        batch["nbr_sim"] = torch.from_numpy(np.stack([case["nbr_sim"] for case in cases]).astype(np.float32))
    return batch


def _write_tokens(directory: str, name: str, rows: list[dict], sequences: list[list[int]]) -> None:
    os.makedirs(directory, exist_ok=True)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    total = sum(len(seq) for seq in sequences)
    flat = np.zeros(total, dtype=np.uint16)
    cursor = 0
    for index, seq in enumerate(sequences):
        flat[cursor:cursor + len(seq)] = seq
        cursor += len(seq)
        offsets[index + 1] = cursor
    np.save(os.path.join(directory, f"{name}.npy"), flat)
    np.save(os.path.join(directory, f"{name}.offsets.npy"), offsets)
    meta = [{"id": row["id"], "labels": row.get("labels", []), "hash": text_hash(row["text"])} for row in rows]
    with open(os.path.join(directory, f"{name}.meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def load_token_split(name: str, directory: str | None = None) -> dict:
    directory = directory or os.path.join(p5_dir(), "tok")
    flat = np.load(os.path.join(directory, f"{name}.npy"))
    offsets = np.load(os.path.join(directory, f"{name}.offsets.npy"))
    with open(os.path.join(directory, f"{name}.meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"flat": flat, "offsets": offsets, "meta": meta}


def case_tokens(split: dict, index: int) -> np.ndarray:
    start, end = int(split["offsets"][index]), int(split["offsets"][index + 1])
    return split["flat"][start:end]


def view_headtail(token_ids: np.ndarray, head: int, tail: int) -> np.ndarray:
    """First `head` and last `tail` tokens. Short cases are kept whole."""
    if len(token_ids) <= head + tail:
        return token_ids
    return np.concatenate([token_ids[:head], token_ids[len(token_ids) - tail:]]) if tail else token_ids[:head]


def segment_count(token_ids: np.ndarray, size: int) -> int:
    content = max(1, size - 1)
    return max(1, (len(token_ids) + content - 1) // content)


def select_segment_ids(n_segments: int, max_chunks: int, train: bool, rng: np.random.Generator) -> list[int]:
    """Training keeps the first segment, the last, and a random sample of the middle.

    Inference keeps the first half of the budget and the last half, in document order.
    """
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if n_segments <= max_chunks:
        return list(range(n_segments))
    if train and max_chunks >= 2:
        middle = list(range(1, n_segments - 1))
        rng.shuffle(middle)
        chosen = [0, n_segments - 1] + middle[: max_chunks - 2]
        chosen.sort()
        return chosen
    head_n = max_chunks // 2
    tail_n = max_chunks - head_n
    return sorted(set(list(range(head_n)) + list(range(n_segments - tail_n, n_segments))))


def materialise_segments(token_ids: np.ndarray, size: int, seg_ids: list[int], bos: int = BOS_ID) -> list[list[int]]:
    content = max(1, size - 1)
    chunks = []
    for seg in seg_ids:
        piece = token_ids[int(seg) * content: int(seg) * content + content]
        chunks.append([bos, *[int(tok) for tok in piece]])
    return chunks or [[bos]]


def resample_record(record: dict, rng: np.random.Generator) -> None:
    """Draw a new middle-segment sample. The chunk count stays the same, so batches stay valid."""
    tokens = record.get("tokens")
    if tokens is None:
        return
    limit = int(record["segment_limit"])
    cap = int(record["segment_cap"])
    seg_ids = select_segment_ids(segment_count(tokens, limit), cap, train=True, rng=rng)
    record["chunks"] = materialise_segments(tokens, limit, seg_ids)
    record["seg_ids"] = seg_ids


def case_record(split: dict, index: int, view: str = "chunks", limit: int = CHUNK,
                max_chunks: int = MAX_CHUNKS, head: int = 766, tail: int = 256,
                train: bool = False, rng: np.random.Generator | None = None) -> dict:
    tokens = case_tokens(split, index)
    meta = split["meta"][index]
    seg_ids = None
    if view == "head":
        chunks = [view_head(tokens, limit)]
    elif view == "headtail":
        chunks = view_chunks(view_headtail(tokens, head, tail), size=limit, max_chunks=max_chunks)
    elif view == "segments":
        rng = rng or np.random.default_rng(SEED)
        seg_ids = select_segment_ids(segment_count(tokens, limit), max_chunks, train=train, rng=rng)
        chunks = materialise_segments(tokens, limit, seg_ids)
    else:
        chunks = view_chunks(tokens, size=limit, max_chunks=max_chunks)
    width = assign_bucket(max(len(chunk) for chunk in chunks))
    record = {
        "id": meta["id"],
        "labels": meta.get("labels", []),
        "hash": meta["hash"],
        "chunks": chunks,
        "width": width,
        "n_tokens": int(len(tokens)),
    }
    if seg_ids is not None:
        record["seg_ids"] = seg_ids
    if view == "segments" and train:
        record["tokens"] = tokens
        record["segment_limit"] = limit
        record["segment_cap"] = max_chunks
    return record


def prepare(workers: int = 4) -> dict:
    directory = p5_dir()
    os.makedirs(directory, exist_ok=True)
    splits = load_raw()
    stats = overlap_stats(splits)
    groups = dev_groups(splits["dev"])
    half = iterative_stratify(groups)
    # iterative_stratify returns one entry per row only if groups cover each row once.
    if len(half) != len(splits["dev"]):
        raise RuntimeError(f"half length {len(half)} != dev {len(splits['dev'])}")
    report = g0_report(splits["dev"], half)
    write_json(report, os.path.join(directory, "g0.json"))
    write_json(stats, os.path.join(directory, "stats.json"))
    occ = cooccurrence(splits["train"])
    write_json(
        {"freq": occ["freq"].tolist(), "top_pairs": occ["top_pairs"],
         "n_train": len(splits["train"])},
        os.path.join(directory, "label_stats.json"),
    )
    np.save(os.path.join(directory, "pair_conditional.npy"), occ["conditional"])
    dev_ids = [row["id"] for row in splits["dev"]]
    write_json({"dev_half": {row_id: name for row_id, name in zip(dev_ids, half)}},
               os.path.join(directory, "splits.json"))
    from saul.cjpe.data import encode_texts
    tok_dir = os.path.join(directory, "tok")
    length_stats = {}
    for name, rows in splits.items():
        cached = os.path.join(tok_dir, f"{name}.npy")
        if os.path.exists(cached) and os.path.exists(os.path.join(tok_dir, f"{name}.offsets.npy")):
            offsets = np.load(os.path.join(tok_dir, f"{name}.offsets.npy"))
            lengths = sorted(int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1))
        else:
            sequences = encode_texts([row["text"] for row in rows], workers=workers)
            _write_tokens(tok_dir, name, rows, sequences)
            lengths = sorted(len(seq) for seq in sequences)
        length_stats[name] = {
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * len(lengths))],
            "p99": lengths[int(0.99 * len(lengths))],
            "max": lengths[-1],
        }
    digest = hashlib.sha256("\n".join(row["id"] for row in splits["test"]).encode()).hexdigest()
    with open(os.path.join(directory, "test_ids_sha256.txt"), "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    summary = {"g0": report, "lengths": length_stats, "stats": stats}
    write_json(summary, os.path.join(directory, "prepare.json"))
    print(json.dumps({"g0_pass": report["pass"], "max_gap": report["max_gap"]}), flush=True)
    return summary


def dev_half_array(ids: list[str], directory: str | None = None) -> np.ndarray:
    path = os.path.join(directory or p5_dir(), "splits.json")
    mapping = json.load(open(path, encoding="utf-8"))["dev_half"]
    return np.asarray([mapping[i] for i in ids])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare(workers=args.workers)


if __name__ == "__main__":
    main()
