#!/usr/bin/env python
"""Serve a Saul checkpoint. SSH tunnel: ssh -L 8000:localhost:8000 the GPU server

    python scripts/52_serve.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema \
        --tokenizer artifacts/tokenizers/legal32k --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import uvicorn

    from saul.serve.api import LLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_slots", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--truncate_left", action="store_true", help="Explicit dev overflow policy; keep the last prompt tokens")
    a = ap.parse_args()
    llm = LLM(checkpoint=a.checkpoint, tokenizer_path=a.tokenizer, max_slots=a.max_slots, block_size=a.block_size,
              max_prompt_len=a.max_prompt_len, max_new_tokens=a.max_new_tokens, device="cuda",
              truncate_left=a.truncate_left)
    from saul.serve.server import build_app
    uvicorn.run(build_app(llm), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
