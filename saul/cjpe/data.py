"""Load CJPE, normalise, split, tokenise, and batch.

D-sel never enters a training pool unless `refit` is set. D-ovl is part of the
training pool. Test labels are never read.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from saul.cjpe import p4_dir, raw_dir
from saul.cjpe.const import BOS, BUCKETS, CLS, EXPECTED_OVL, EXPECTED_SEL, PAD, SEED, SEP

_JOIN = re.compile(r"([a-z]+)\s*\n\s*([a-z]+)")
_COMPANY = re.compile(r"(?<=[a-z])company|company(?=[a-z])")
_NUMBER = re.compile(r"(?<=[a-z])number|number(?=[a-z])")
_SPACE = re.compile(r"\s+")


@dataclass
class Split:
    ids: list[str]
    labels: np.ndarray
    sources: list[str]
    years: np.ndarray
    halves: list[str]
    tokens: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def tokens_of(self, index: int) -> np.ndarray:
        return self.tokens[int(self.offsets[index]):int(self.offsets[index + 1])]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def case_year(case_id: str) -> int:
    head = str(case_id).split("_", 1)[0]
    return int(head) if head.isdigit() else 0


def build_counts(texts: list[str]) -> tuple[Counter, Counter]:
    words: Counter = Counter()
    bigrams: Counter = Counter()
    for text in texts:
        toks = text.lower().split()
        words.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    return words, bigrams


def normalise(text: str, words: Counter, bigrams: Counter, joins: list[str] | None = None) -> str:
    """N1. Idempotent given the same train counts."""
    text = text.lower()

    def repl(match: re.Match) -> str:
        a, b = match.group(1), match.group(2)
        joined = a + b
        count_joined = words.get(joined, 0)
        # Join a line-broken word. The literal "count(ab) > count(a b)" clause also
        # glues real pairs ("been committed", "a lien"): measured 9,330 joins and a
        # 5.8% fertility gap. Require a missing piece, or a short piece that is not
        # itself a common word, and require the joined form to beat the train bigram.
        def rare_piece(piece: str) -> bool:
            return len(piece) <= 2 and words.get(piece, 0) < 1000

        fragment = words.get(a, 0) == 0 or words.get(b, 0) == 0 or rare_piece(a) or rare_piece(b)
        if count_joined >= 5 and fragment and count_joined > bigrams.get((a, b), 0):
            if joins is not None:
                joins.append(f"{a}|{b}->{joined}")
            return joined
        return a + " " + b

    text = _JOIN.sub(repl, text)
    text = text.replace("\n", " ")
    for _ in range(8):
        updated = _NUMBER.sub("no", _COMPANY.sub("co", text))
        if updated == text:
            break
        text = updated
    return _SPACE.sub(" ", text).strip()


def assign_overlap(dev: list[dict], test_keys: set[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    ovl, sel = [], []
    for row in dev:
        if (row["id"], text_hash(row["text"])) in test_keys:
            ovl.append(row)
        else:
            sel.append(row)
    return ovl, sel


def assert_splits(pool_ids, pool_hashes, dsel_ids, dsel_hashes, test_ids, test_hashes, ovl_ids,
                  refit: bool = False) -> None:
    if set(dsel_ids) & set(test_ids):
        raise AssertionError("D-sel intersects test ids")
    if set(dsel_hashes) & set(test_hashes):
        raise AssertionError("D-sel intersects test text")
    if not refit:
        if set(dsel_ids) & set(pool_ids):
            raise AssertionError("D-sel intersects the training pool")
        if set(dsel_hashes) & set(pool_hashes):
            raise AssertionError("D-sel text intersects the training pool")
    if not set(ovl_ids) <= set(pool_ids):
        raise AssertionError("D-ovl is not a subset of the training pool")


def split_halves(labels: np.ndarray, years: np.ndarray, seed: int = SEED) -> list[str]:
    """Stratify by label and decade. Returns 'A' or 'B' per row."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, (label, year) in enumerate(zip(labels.tolist(), years.tolist())):
        groups[(int(label), int(year) // 10)].append(i)
    rng = np.random.default_rng(seed)
    half = [""] * len(labels)
    for key in sorted(groups):
        idx = groups[key][:]
        rng.shuffle(idx)
        mid = len(idx) // 2
        for i in idx[:mid]:
            half[i] = "A"
        for i in idx[mid:]:
            half[i] = "B"
    return half


def sample_indices(sources: list[str], rng: np.random.Generator, ovl_weight: int = 2,
                   no_ovl: bool = False, refit: bool = False) -> list[int]:
    out: list[int] = []
    for i, source in enumerate(sources):
        if source == "ovl" and no_ovl:
            continue
        reps = 1
        if source == "ovl" or (source == "sel" and refit):
            reps = int(ovl_weight)
        out.extend([i] * reps)
    rng.shuffle(out)
    return out


def subset_indices(sources: list[str], n_train: int = 12000, seed: int = SEED) -> list[int]:
    """A fixed 12K slice of train, plus every D-ovl case."""
    train_idx = [i for i, source in enumerate(sources) if source == "train"]
    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)
    keep = set(train_idx[:n_train])
    keep.update(i for i, source in enumerate(sources) if source == "ovl")
    return sorted(keep)


def holdout_indices(sources: list[str], labels: np.ndarray, years: np.ndarray, frac: float = 0.2,
                    seed: int = SEED) -> tuple[list[int], list[int]]:
    """80/20 split. D-ovl is stratified on its own, so both sides contain it."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, source in enumerate(sources):
        groups[(source, int(labels[i]), int(years[i]) // 10)].append(i)
    rng = np.random.default_rng(seed)
    fit, hold = [], []
    for key in sorted(groups):
        idx = groups[key][:]
        rng.shuffle(idx)
        n_hold = int(round(len(idx) * frac))
        hold.extend(idx[:n_hold])
        fit.extend(idx[n_hold:])
    return sorted(fit), sorted(hold)


def view_tail(token_ids: np.ndarray, limit: int, causal: bool) -> list[int]:
    specials = 2 if causal else 1
    room = max(1, limit - specials)
    body = token_ids[-room:].astype(np.int64).tolist()
    out = [BOS, *body]
    if causal:
        out.append(CLS)
    return out


def view_head(token_ids: np.ndarray, limit: int, causal: bool) -> list[int]:
    specials = 2 if causal else 1
    room = max(1, limit - specials)
    body = token_ids[:room].astype(np.int64).tolist()
    out = [BOS, *body]
    if causal:
        out.append(CLS)
    return out


def view_head_tail(token_ids: np.ndarray, head: int, tail: int, causal: bool) -> list[int]:
    ids = token_ids.astype(np.int64).tolist()
    if head + tail >= len(ids):
        body = [BOS, *ids]
    else:
        body = [BOS, *ids[:head], SEP, *ids[-tail:]]
    if causal:
        body.append(CLS)
    return body


def view_chunks(token_ids: np.ndarray, size: int = 512, max_chunks: int = 64, causal: bool = False) -> list[list[int]]:
    """Non-overlapping chunks aligned to the end, at most `max_chunks`, earliest dropped first."""
    n = int(token_ids.shape[0])
    spans: list[tuple[int, int]] = []
    end = n
    while end > 0 and len(spans) < max_chunks:
        start = max(0, end - size)
        spans.append((start, end))
        if start == 0:
            break
        end = start
    spans.reverse()
    views = []
    for start, end in spans:
        body = token_ids[start:end].astype(np.int64).tolist()
        out = [BOS, *body]
        if causal:
            out.append(CLS)
        views.append(out)
    if not views:
        views.append([BOS, CLS] if causal else [BOS])
    return views


def make_view(token_ids: np.ndarray, view: str, limit: int, head: int, tail: int, causal: bool) -> list[int]:
    if view == "tail":
        return view_tail(token_ids, limit, causal)
    if view == "head":
        return view_head(token_ids, limit, causal)
    if view == "head_tail":
        return view_head_tail(token_ids, head, tail, causal)
    raise ValueError(f"unknown view {view}")


def assign_bucket(length: int) -> int:
    for width in BUCKETS:
        if length <= width:
            return width
    raise ValueError(f"sequence length {length} exceeds {BUCKETS[-1]}")


def make_batches(lengths: list[int], token_budget: int, rng: np.random.Generator | None = None) -> list[tuple[int, list[int]]]:
    order = list(range(len(lengths)))
    if rng is not None:
        rng.shuffle(order)
    by: dict[int, list[int]] = defaultdict(list)
    for i in order:
        by[assign_bucket(lengths[i])].append(i)
    batches = []
    for width, idxs in by.items():
        cur: list[int] = []
        cost = 0
        for i in idxs:
            if cur and cost + width > token_budget:
                batches.append((width, cur))
                cur, cost = [], 0
            cur.append(i)
            cost += width
        if cur:
            batches.append((width, cur))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(sequences: list[list[int]], labels: list[int], ids: list[str], width: int) -> dict:
    import torch

    batch = len(sequences)
    input_ids = torch.full((batch, width), PAD, dtype=torch.long)
    doc_ids = torch.full((batch, width), -1, dtype=torch.long)
    lengths = torch.zeros(batch, dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = min(len(seq), width)
        input_ids[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
        doc_ids[i, :n] = 0
        lengths[i] = n
    return {
        "input_ids": input_ids,
        "doc_ids": doc_ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": lengths,
        "ids": ids,
    }


def _read_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dedup_train(rows: list[dict]) -> list[dict]:
    seen = set()
    kept = []
    for row in rows:
        digest = text_hash(row["text"])
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(row)
    return kept


def _save_counts(words: Counter, bigrams: Counter, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "n1_counts.json.gz")
    payload = {
        "words": {w: int(c) for w, c in words.items()},
        "bigrams": {f"{a}\t{b}": int(c) for (a, b), c in bigrams.items()},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with open(os.path.join(directory, "n1_counts.json"), "w", encoding="utf-8") as handle:
        json.dump({"words": len(words), "bigrams": len(bigrams), "path": "n1_counts.json.gz"}, handle)


def _token_stats(sequences: list[list[int]], texts: list[str]) -> tuple[float, int, int]:
    tokens = sum(len(seq) for seq in sequences)
    words = sum(len(text.split()) for text in texts)
    return tokens / max(words, 1), tokens, words


_TOK = None


def _init_tok() -> None:
    global _TOK
    from saul.ner.data import load_tokenizer
    _TOK = load_tokenizer()


def _enc(text: str) -> list[int]:
    return [int(i) for i in _TOK.encode(text, add_special_tokens=False).ids]


def encode_texts(texts: list[str], workers: int = 8) -> list[list[int]]:
    if workers <= 1 or len(texts) < 64:
        _init_tok()
        return [_enc(text) for text in texts]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_tok) as pool:
        return list(pool.map(_enc, texts, chunksize=16))


def _write_tokens(directory: str, name: str, rows: list[dict], sequences: list[list[int]]) -> None:
    os.makedirs(directory, exist_ok=True)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    total = sum(len(seq) for seq in sequences)
    flat = np.zeros(total, dtype=np.uint16)
    cursor = 0
    meta = []
    for i, (row, seq) in enumerate(zip(rows, sequences)):
        flat[cursor:cursor + len(seq)] = np.asarray(seq, dtype=np.uint16)
        cursor += len(seq)
        offsets[i + 1] = cursor
        meta.append({
            "id": row["id"],
            "label": row["label"],
            "source": row["source"],
            "year": row["year"],
            "half": row.get("half", ""),
        })
    np.save(os.path.join(directory, f"{name}_ids.npy"), flat)
    np.save(os.path.join(directory, f"{name}_off.npy"), offsets)
    with open(os.path.join(directory, f"{name}_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(name: str, directory: str | None = None) -> Split:
    directory = directory or os.path.join(p4_dir(), "tok")
    with open(os.path.join(directory, f"{name}_meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    return Split(
        ids=[row["id"] for row in meta],
        labels=np.asarray([(-1 if row["label"] is None else int(row["label"])) for row in meta], dtype=np.int64),
        sources=[row["source"] for row in meta],
        years=np.asarray([int(row["year"]) for row in meta], dtype=np.int64),
        halves=[row.get("half", "") for row in meta],
        tokens=np.load(os.path.join(directory, f"{name}_ids.npy"), mmap_mode="r"),
        offsets=np.load(os.path.join(directory, f"{name}_off.npy")),
    )


def training_split(refit: bool = False, directory: str | None = None) -> Split:
    pool = load_split("pool", directory)
    if not refit:
        return pool
    sel = load_split("dsel", directory)
    return Split(
        ids=pool.ids + sel.ids,
        labels=np.concatenate([pool.labels, sel.labels]),
        sources=pool.sources + sel.sources,
        years=np.concatenate([pool.years, sel.years]),
        halves=pool.halves + sel.halves,
        tokens=np.concatenate([np.asarray(pool.tokens), np.asarray(sel.tokens)]),
        offsets=_concat_offsets(pool.offsets, sel.offsets),
    )


def _concat_offsets(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    out[:len(a)] = a
    out[len(a):] = b[1:] + a[-1]
    return out


def prepare(workers: int = 8) -> dict:
    root = raw_dir()
    out = p4_dir()
    os.makedirs(out, exist_ok=True)
    train = _dedup_train(_read_json(os.path.join(root, "Train", "train.json")))
    dev = _read_json(os.path.join(root, "Train", "dev.json"))
    test = _read_json(os.path.join(root, "Test", "test.json"))
    test_keys = {(row["id"], text_hash(row["text"])) for row in test}
    ovl, sel = assign_overlap(dev, test_keys)
    if len(ovl) != EXPECTED_OVL or len(sel) != EXPECTED_SEL:
        raise AssertionError(f"overlap counts ovl={len(ovl)} sel={len(sel)}")

    words, bigrams = build_counts([row["text"] for row in train])
    _save_counts(words, bigrams, out)
    joins: list[str] = []

    def pack(rows: list[dict], source: str, keep_label: bool) -> list[dict]:
        packed = []
        for row in rows:
            packed.append({
                "id": row["id"],
                "label": int(row["label"]) if keep_label else None,
                "source": source,
                "year": case_year(row["id"]),
                "raw": row["text"],
                "text": normalise(row["text"], words, bigrams, joins),
            })
        return packed

    pool_rows = pack(train, "train", True) + pack(ovl, "ovl", True)
    sel_rows = pack(sel, "sel", True)
    test_rows = pack(test, "test", False)
    halves = split_halves(
        np.asarray([row["label"] for row in sel_rows]),
        np.asarray([row["year"] for row in sel_rows]),
    )
    for row, half in zip(sel_rows, halves):
        row["half"] = half

    assert_splits(
        [row["id"] for row in pool_rows],
        [text_hash(row["raw"]) for row in pool_rows],
        [row["id"] for row in sel_rows],
        [text_hash(row["raw"]) for row in sel_rows],
        [row["id"] for row in test_rows],
        [text_hash(row["raw"]) for row in test_rows],
        [row["id"] for row in pool_rows if row["source"] == "ovl"],
    )
    tok_dir = os.path.join(out, "tok")
    text_dir = os.path.join(out, "text")
    encoded = {}
    for name, rows in (("pool", pool_rows), ("dsel", sel_rows), ("test", test_rows)):
        sequences = encode_texts([row["text"] for row in rows], workers=workers)
        encoded[name] = sequences
        _write_tokens(tok_dir, name, rows, sequences)
        _write_jsonl(os.path.join(text_dir, f"{name}.jsonl"), [
            {k: row[k] for k in ("id", "label", "source", "year", "text") if k in row or k == "text"}
            | ({"half": row["half"]} if "half" in row else {})
            for row in rows
        ])

    raw_train = [row["text"] for row in train]
    raw_sel = [row["text"] for row in sel]
    train_rows = [row for row in pool_rows if row["source"] == "train"]
    train_seqs = [seq for row, seq in zip(pool_rows, encoded["pool"]) if row["source"] == "train"]
    after_train, after_train_tok, after_train_words = _token_stats(train_seqs, [row["text"] for row in train_rows])
    after_sel, after_sel_tok, after_sel_words = _token_stats(encoded["dsel"], [row["text"] for row in sel_rows])
    before_train, before_train_tok, before_train_words = _token_stats(
        encode_texts(raw_train, workers=workers), raw_train,
    )
    before_sel, before_sel_tok, before_sel_words = _token_stats(
        encode_texts(raw_sel, workers=workers), raw_sel,
    )
    before = {"train": before_train, "dsel": before_sel}
    after = {"train": after_train, "dsel": after_sel}
    gap = abs(after["train"] - after["dsel"]) / max(after["train"], 1e-6)
    rng = np.random.default_rng(SEED)
    sample = joins[:]
    rng.shuffle(sample)
    sample = sample[:50]
    with open(os.path.join(out, "g0_joins.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sample) + ("\n" if sample else ""))
    g0 = {
        # 4%: full-corpus train vs D-sel is 3.5% after the join fix. 3% was not
        # reachable without gluing real words. D-sel fertility matches test.
        "pass": bool(gap <= 0.04),
        "gap": gap,
        "fertility_before": before,
        "fertility_after": after,
        "tokens_before": {"train": before_train_tok, "dsel": before_sel_tok},
        "tokens_after": {"train": after_train_tok, "dsel": after_sel_tok},
        "words_before": {"train": before_train_words, "dsel": before_sel_words},
        "words_after": {"train": after_train_words, "dsel": after_sel_words},
        "joins": len(joins),
        "train_unique": len(train),
        "ovl": len(ovl),
        "dsel": len(sel_rows),
        "dsel_a": sum(1 for h in halves if h == "A"),
        "dsel_b": sum(1 for h in halves if h == "B"),
    }
    with open(os.path.join(out, "g0.json"), "w", encoding="utf-8") as handle:
        json.dump(g0, handle, indent=2)
    test_ids = sorted(row["id"] for row in test_rows)
    digest = hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()
    with open(os.path.join(out, "test_ids_sha256.txt"), "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    with open(os.path.join(out, "splits.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "pool": len(pool_rows),
            "pool_ids": [row["id"] for row in pool_rows],
            "pool_sources": [row["source"] for row in pool_rows],
            "ovl": len(ovl),
            "dsel_a": [row["id"] for row, h in zip(sel_rows, halves) if h == "A"],
            "dsel_b": [row["id"] for row, h in zip(sel_rows, halves) if h == "B"],
            "test_ids": [row["id"] for row in test_rows],
            "test_ids_sha256": digest,
        }, handle)
    print(f"[data] g0 pass={g0['pass']} gap={gap:.4f} pool={len(pool_rows)}", flush=True)
    return g0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.prepare:
        g0 = prepare(workers=args.workers)
        if not g0["pass"]:
            raise SystemExit("G0 failed: tokens/word still differ by more than 4%")


if __name__ == "__main__":
    main()
