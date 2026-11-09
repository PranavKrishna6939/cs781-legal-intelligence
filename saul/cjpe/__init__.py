"""Phase 4 Court Judgment Prediction."""
from __future__ import annotations

import os

from saul.ner.registry import repo_root


def p4_dir() -> str:
    return os.environ.get("SAUL_P4_DIR", "/data/cs781-p4")


def raw_dir() -> str:
    return os.environ.get(
        "SAUL_P4_RAW", os.path.join(repo_root(), "data", "raw", "Phase_4_CJPE")
    )


def exp_dir(name: str) -> str:
    return os.path.join(p4_dir(), "exp", name)
