"""Windowed inference. Sentence marginals are averaged with a triangle weight."""
from __future__ import annotations

import os

import numpy as np
import torch

from saul.rr.data import N_LABELS, Window, collate, triangle
from saul.rr.model import RRModel


@torch.no_grad()
def accumulate(model: RRModel, windows: list[Window], n_sent: dict[str, int], batch_size: int,
               device: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mass = {i: np.zeros((n, N_LABELS), dtype=np.float64) for i, n in n_sent.items()}
    weight = {i: np.zeros(n, dtype=np.float64) for i, n in n_sent.items()}
    was = model.training
    model.eval()
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk).items()}
        out = model(batch)
        marg = out["marginals"].float().cpu().numpy()
        for row, window in enumerate(chunk):
            m = len(window.sent_index)
            for t in range(m):
                w = triangle(t, m)
                idx = window.sent_index[t]
                mass[window.doc_id][idx] += w * marg[row, t]
                weight[window.doc_id][idx] += w
    model.train(was)
    return {i: (mass[i], weight[i]) for i in mass}


def save_npz(path: str, mass: np.ndarray, weight: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, marginals=mass.astype(np.float32), weight=weight.astype(np.float32))


def load_probs(path: str) -> np.ndarray:
    blob = np.load(path)
    mass = blob["marginals"].astype(np.float64)
    weight = blob["weight"].astype(np.float64)
    denom = np.maximum(weight, 1e-8)[:, None]
    probs = mass / denom
    probs = np.clip(probs, 1e-8, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def write_cache(directory: str, packed: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for doc_id, (mass, weight) in packed.items():
        save_npz(os.path.join(directory, f"{doc_id}.npz"), mass, weight)


def merge_test(fold_dirs: list[str], dest: str, doc_ids: list[str]) -> None:
    os.makedirs(dest, exist_ok=True)
    n = len(fold_dirs)
    for doc_id in doc_ids:
        acc = None
        for fold_dir in fold_dirs:
            probs = load_probs(os.path.join(fold_dir, f"{doc_id}.npz"))
            acc = probs.astype(np.float64) if acc is None else acc + probs
        merged = acc / n
        weight = np.full(merged.shape[0], float(n), dtype=np.float32)
        save_npz(os.path.join(dest, f"{doc_id}.npz"), merged * weight[:, None], weight)
