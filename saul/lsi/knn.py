"""Train-only nearest-neighbour label votes from LAAT representations."""
from __future__ import annotations

import os

import numpy as np

from saul.lsi import exp_dir
from saul.lsi.const import DECISION_GAIN, SEED
from saul.lsi.score import macro_f1

LAMBDAS = (0.0, 0.1, 0.2, 0.3)


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def vote(query: np.ndarray, train_emb: np.ndarray, train_labels: np.ndarray, k: int = 32,
         temperature: float = 0.05, batch: int = 256) -> np.ndarray:
    """Similarity-weighted label share of the k nearest training cases. The index is train only."""
    query = l2_normalise(np.asarray(query, dtype=np.float32))
    train_emb = l2_normalise(np.asarray(train_emb, dtype=np.float32))
    train_labels = np.asarray(train_labels, dtype=np.float32)
    out = np.zeros((len(query), train_labels.shape[1]), dtype=np.float32)
    k = min(k, len(train_emb))
    for start in range(0, len(query), batch):
        q = query[start:start + batch]
        sims = q @ train_emb.T
        index = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top = np.take_along_axis(sims, index, axis=1)
        weights = np.exp((top - top.max(axis=1, keepdims=True)) / temperature)
        weights /= weights.sum(axis=1, keepdims=True)
        gathered = train_labels[index]
        out[start:start + len(q)] = np.einsum("bk,bkl->bl", weights, gathered)
    return out


def choose_lambda(model_probs: np.ndarray, knn_probs: np.ndarray, labels: np.ndarray, half) -> dict:
    half = np.asarray(half)
    gains = []
    picked = {}
    for fit_name, score_name in (("A", "B"), ("B", "A")):
        fit, score = half == fit_name, half == score_name

        def quality(lam: float, rows) -> float:
            mixed = (1 - lam) * model_probs[rows] + lam * knn_probs[rows]
            return macro_f1(labels[rows], mixed >= 0.5)

        lam = max(LAMBDAS, key=lambda value: quality(value, fit))
        picked[fit_name] = lam
        gains.append(quality(lam, score) - quality(0.0, score))
    gain = float(np.mean(gains))
    lam = picked["A"] if gains[0] >= gains[1] else picked["B"]
    keep = gain >= DECISION_GAIN and lam > 0
    return {
        "lambda": float(lam if keep else 0.0),
        "gain": gain,
        "keep": keep,
        "half_gain": gains,
        "by_half": {name: float(value if keep else 0.0) for name, value in picked.items()},
    }


def save_votes(name: str, kind: str, ids: list[str], probs: np.ndarray) -> str:
    directory = exp_dir(name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{kind}_knn.npz")
    np.savez(path, ids=np.asarray(ids), probs=probs)
    return path


def dump_label_means(model, records: list[dict], device: str, batch_tokens: int = 32768) -> np.ndarray:
    """Mean LAAT label representation per case, L2-normalised."""
    import torch
    from saul.lsi.data import collate, make_batches

    model.eval()
    width = int(model.head.weight.shape[1])
    means = np.zeros((len(records), width), dtype=np.float32)
    lengths = [len(record["chunks"]) for record in records]
    widths = [record["width"] for record in records]
    batches = make_batches(lengths, widths, batch_tokens, np.random.default_rng(SEED))
    with torch.no_grad():
        for bucket, indexes in batches:
            batch = collate([records[i] for i in indexes], bucket)
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            reps = model(moved, return_extra=True)["label_reps"].mean(dim=1)
            means[list(indexes)] = reps.float().cpu().numpy()
    return l2_normalise(means)


def main() -> None:
    """Dump LAAT label means for train, dev and test, then vote from the train index only."""
    import argparse
    import torch

    from saul.lsi.data import load_token_split, multi_hot
    from saul.lsi.model import load_lsi, load_overlay
    from saul.lsi.train import _records
    from saul.lsi import p5_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="A1")
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--technique", default="lorma")
    parser.add_argument("--context", default="bidirectional")
    parser.add_argument("--limit", type=int, default=2048)
    args = parser.parse_args()
    overlay = os.path.join(exp_dir(args.exp), "best.pt")
    if not os.path.exists(overlay):
        raise SystemExit(f"[knn] no overlay at {overlay}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lsi(args.model, args.technique, args.context, device=device)
    load_overlay(model, torch.load(overlay, map_location=device))
    tok = os.path.join(p5_dir(), "tok")
    embeddings = {}
    metas = {}
    for name in ("train", "dev", "test"):
        split = load_token_split(name, tok)
        metas[name] = split["meta"]
        records = _records(split, np.arange(len(split["meta"])), args.limit, 8)
        embeddings[name] = dump_label_means(model, records, device)
        np.savez(os.path.join(exp_dir(args.exp), f"emb_{name}.npz"),
                 ids=np.asarray([row["id"] for row in split["meta"]]), emb=embeddings[name])
    train_labels = np.stack([multi_hot(row.get("labels", [])) for row in metas["train"]])
    for name in ("dev", "test"):
        probs = vote(embeddings[name], embeddings["train"], train_labels)
        save_votes(args.exp, name, [row["id"] for row in metas[name]], probs)
    print(f"[knn] votes written under {exp_dir(args.exp)}", flush=True)


if __name__ == "__main__":
    main()
