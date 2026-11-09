#!/usr/bin/env python
"""Stage 11: tokenize train/val splits into uint16 memmaps (+ token-byte table, token shares, rare-token audit).

    python scripts/11_tokenize_corpus.py --tokenizer artifacts/tokenizers/legal32k --out data/tokens/legal32k
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from saul.tokenizer.train import load_raw_tokenizer, token_byte_lengths  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ensure_dir, list_shards, read_json, write_json  # noqa: E402

_T: dict = {}


def _init(tok_dir: str, bos: int, eos: int, vocab: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _T.update(tok=load_raw_tokenizer(tok_dir), bos=bos, eos=eos, vocab=vocab)


def _encode_shard(args: tuple[str, str]) -> tuple[str, int, np.ndarray, np.ndarray]:
    shard, part_path = args
    tok, bos, eos = _T["tok"], _T["bos"], _T["eos"]
    texts, anon = [], []
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            anon.append(int(bool(r["meta"].get("anonymized", False))))
    idx = np.zeros((len(texts), 4), dtype=np.int64)
    counts = np.zeros(_T["vocab"], dtype=np.int64)
    pos = 0
    with open(part_path, "wb") as out:
        for b in range(0, len(texts), 256):
            encs = tok.encode_batch(texts[b:b + 256], add_special_tokens=False)
            batch = []
            for j, e in enumerate(encs):
                ids = np.asarray([bos] + e.ids + [eos], dtype=np.uint16)
                batch.append(ids)
                i = b + j
                idx[i] = (pos, len(ids), len(texts[i].encode("utf-8")), anon[i])
                pos += len(ids)
            cat = np.concatenate(batch)
            out.write(cat.tobytes())
            counts += np.bincount(cat, minlength=_T["vocab"])
    return part_path, pos, idx, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_config", default="configs/data.yaml")
    ap.add_argument("--tokenizer", default="artifacts/tokenizers/legal32k")
    ap.add_argument("--out", default=None, help="default: data/tokens/<tokenizer name>")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rare_min_count", type=int, default=50)
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="tokenize every k-th train shard per source (vocab ablations need ~1B tokens, not all)")
    a = ap.parse_args()
    cfg = load_config(a.data_config)
    tok = load_raw_tokenizer(a.tokenizer)
    meta = read_json(os.path.join(a.tokenizer, "saul_tokenizer_meta.json"))
    V = tok.get_vocab_size(True)
    assert V <= 65536, "uint16 storage requires vocab <= 65536"
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    out_root = ensure_dir(a.out or os.path.join(cfg.paths.tokens, os.path.basename(os.path.normpath(a.tokenizer))))
    np.save(os.path.join(out_root, "token_bytes.npy"), np.asarray(token_byte_lengths(tok, meta["specials"]), np.int32))
    shutil.copyfile(os.path.join(a.tokenizer, "saul_tokenizer_meta.json"), os.path.join(out_root, "tokenizer_meta.json"))

    shares: dict = {}
    train_counts = np.zeros(V, dtype=np.int64)
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.tokenizer, bos, eos, V)) as pool:
        for split in a.splits:
            shares[split] = {}
            for src in cfg.sources:
                shards = list_shards(os.path.join(cfg.paths.final, split, src))
                if split == "train" and a.train_fraction < 1.0 and shards:
                    stride = max(1, round(1.0 / a.train_fraction))
                    shards = shards[::stride]
                if not shards:
                    continue
                t0 = time.time()
                d = ensure_dir(os.path.join(out_root, split, src))
                jobs = [(s, os.path.join(d, f"part_{i:05d}.bin")) for i, s in enumerate(shards)]
                total = 0
                indices = []
                with open(os.path.join(d, "tokens.bin"), "wb") as fout:
                    for part, n, idx, counts in pool.imap(_encode_shard, jobs):  # ordered
                        idx[:, 0] += total
                        indices.append(idx)
                        total += n
                        with open(part, "rb") as fp:
                            shutil.copyfileobj(fp, fout, length=64 << 20)
                        os.remove(part)
                        if split == "train":
                            train_counts += counts
                index = np.concatenate(indices) if indices else np.zeros((0, 4), np.int64)
                np.save(os.path.join(d, "index.npy"), index)
                assert os.path.getsize(os.path.join(d, "tokens.bin")) == 2 * total
                shares[split][src] = {"docs": int(len(index)), "tokens": int(total),
                                      "bytes": int(index[:, 2].sum()), "anonymized_docs": int(index[:, 3].sum())}
                print(f"[{split}/{src}] {shares[split][src]} ({time.time()-t0:.0f}s)", flush=True)
    write_json(shares, os.path.join(out_root, "token_shares.json"))
    np.save(os.path.join(out_root, "train_token_counts.npy"), train_counts)
    specials = set(range(len(meta["specials"])))
    rare = [int(i) for i in np.nonzero(train_counts < a.rare_min_count)[0] if int(i) not in specials]
    write_json({"min_count": a.rare_min_count, "rare_ids": rare,
                "rare_tokens": [tok.id_to_token(i) for i in rare[:500]]},
               os.path.join(a.tokenizer, "rare_tokens.json"))
    print(f"rare tokens (<{a.rare_min_count} occurrences): {len(rare)} -> {a.tokenizer}/rare_tokens.json")


if __name__ == "__main__":
    main()
