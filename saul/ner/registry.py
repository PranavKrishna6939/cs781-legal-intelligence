"""Phase-1 checkpoints and the context mode each one uses for NER."""
from __future__ import annotations

import os

# Paths are on the GPU server. Relative paths are under the repo root.
CHECKPOINTS = {
    "dense_v1": "runs/dense_286m/anneal/ckpt_final_ema",
    "dense_v2": "runs/dense_286m/anneal_v2/ckpt_final_ema",
    "dense_d5": "runs/dense_286m/anneal_d5_clm_testmix/ckpt_final_ema",
    "hybrid_b": "/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema",
    "hybrid_h5": "/data/cs781-archive/gpu2/hybrid_294m/anneal_h5_testmix/ckpt_final_ema",
    "hybrid_h7": "/data/cs781-archive/gpu3/hybrid_294m/anneal_h7_sc55/ckpt_final_ema",
    "ced": "/data/cs781-archive/gpu3/ced_290m/wsd/ckpt_final_ema",
    "conv_mini": "/data/cs781-archive/gpu3/conv_mini/conv_mini_kd/student",
    "dense_52m": "/data/cs781-archive/gpu3/ablations/a07_mntp12/ckpt_final_ema",
}

# Default context. Echo for models whose token mixer is causal.
CONTEXT = {
    "dense_v1": "bidirectional",
    "dense_v2": "bidirectional",
    "dense_d5": "bidirectional",
    "dense_52m": "bidirectional",
    "hybrid_b": "echo",
    "hybrid_h5": "echo",
    "hybrid_h7": "echo",
    "conv_mini": "echo",
    "ced": "encoder",
}

# Partial fine-tune trains these encoder layers (CED has 11, indices 0..10).
PARTIAL_LAYERS = {
    "ced": (5, 11),
}


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def p2_dir() -> str:
    return os.environ.get("SAUL_P2_DIR", "/data/cs781-p2")


def resolve_checkpoint(name: str) -> str:
    root = os.environ.get("SAUL_CKPT_ROOT")
    if root and os.path.isdir(os.path.join(root, name)):
        return os.path.join(root, name)
    path = CHECKPOINTS[name]
    if not os.path.isabs(path):
        path = os.path.join(repo_root(), path)
    return path
