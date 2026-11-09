"""Phase 5 Legal Statute Identification."""
from __future__ import annotations

import os

from saul.ner.registry import repo_root


def p5_dir() -> str:
    return os.environ.get("SAUL_P5_DIR", "/data/cs781-p5")


def raw_dir() -> str:
    return os.environ.get(
        "SAUL_P5_RAW", os.path.join(repo_root(), "data", "raw", "Phase_5_LSI")
    )


def exp_dir(name: str) -> str:
    return os.path.join(p5_dir(), "exp", name)
