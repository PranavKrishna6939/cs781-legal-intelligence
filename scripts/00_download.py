#!/usr/bin/env python
"""Download the gated Phase-1 corpus, the Phase-1 test prompts, and small downstream files.

    python scripts/00_download.py --config configs/data.yaml [--skip_phase1]

Requires access to the gated dataset and a token in HF_TOKEN or ~/.cache/huggingface/token.
Verifies the Phase-1 file sizes against the HF tree.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download  # noqa: E402

from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import ensure_dir  # noqa: E402


def read_token(path: str) -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = os.path.expanduser(path)
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def fetch(repo: str, filename: str, raw_dir: str, token: str | None) -> str:
    dst = os.path.join(raw_dir, filename)
    if os.path.exists(dst):
        print(f"[skip] {filename}")
        return dst
    ensure_dir(os.path.dirname(dst))
    src = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", token=token)
    # copy (not symlink) so the raw dir is self-contained and the HF cache can be cleared
    shutil.copyfile(src, dst)
    print(f"[ok]   {filename} ({os.path.getsize(dst)/2**20:.1f} MiB)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--skip_phase1", action="store_true", help="only test + downstream files")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    token = read_token(cfg.hf.token_path)
    raw = ensure_dir(cfg.paths.raw)

    fetch(cfg.hf.repo, cfg.hf.test_file, raw, token)
    for f in cfg.hf.downstream_files:
        try:
            fetch(cfg.hf.repo, f, raw, token)
        except Exception as e:  # downstream files are optional for Phase 1
            print(f"[warn] could not fetch {f}: {e}")
    if a.skip_phase1:
        return
    bad = []
    for src, f in cfg.hf.phase1_files.items():
        p = fetch(cfg.hf.repo, f, raw, token)
        exp = cfg.hf.expected_sizes.get(src)
        got = os.path.getsize(p)
        if exp and got != exp:
            bad.append((f, got, exp))
    if bad:
        for f, got, exp in bad:
            print(f"[ERROR] size mismatch {f}: {got} != {exp} (partial download or dataset update)")
        sys.exit(1)
    print("all Phase-1 files present with expected sizes")


if __name__ == "__main__":
    main()
