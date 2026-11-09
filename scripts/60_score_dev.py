#!/usr/bin/env python
"""Generate one dev-set recipe and score it with the official Phase-1 evaluator.

    python scripts/60_score_dev.py --name H3 --checkpoint runs/hybrid_294m/anneal/ckpt_final_ema \
        --mbr_n 16 --temperature 0.9 --out runs/eval/dev_full/H3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.eval.official import phase1_score  # noqa: E402
from saul.serve.api import LLM  # noqa: E402
from saul.serve.sampling import MBRParams, SamplingParams  # noqa: E402
from saul.utils.config import load_config, to_plain  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizers/legal32k")
    ap.add_argument("--prompts", default="data/final/dev_prompts.json")
    ap.add_argument("--truth", default="data/final/dev_truth.json")
    ap.add_argument("--decode", default="configs/decode/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mbr_n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--truncation", default=None)
    ap.add_argument("--frequency_penalty", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=781)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = to_plain(load_config(a.decode))
    n = 1 if a.greedy else max(1, a.mbr_n)
    temp = 0.0 if a.greedy else (a.temperature if a.temperature is not None else (0.9 if n > 1 else 0.8))
    target = int(cfg.get("target_words", 128))
    prompts = read_json(a.prompts)
    order = list(prompts)
    pred_path = os.path.join(a.out, "prediction.json")
    if not os.path.exists(pred_path):
        block = max(1, n)
        slots = (512 // block) * block
        llm = LLM(checkpoint=a.checkpoint, tokenizer_path=a.tokenizer, max_slots=slots, block_size=block,
                  max_prompt_len=2048, max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
                  prefill_token_budget=65536, admit_batch_blocks=8, truncate_left=True,
                  cuda_graphs=torch.cuda.is_available(), device="cuda" if torch.cuda.is_available() else "cpu",
                  dtype="bfloat16", rouge_workers=6)
        trunc = "none" if temp <= 0 else (a.truncation or cfg.get("truncation", "min_p"))
        freq = a.frequency_penalty if a.frequency_penalty is not None else float(cfg.get("frequency_penalty", 0.3))
        params = SamplingParams(
            n=n, temperature=temp, truncation=trunc,
            min_p=float(cfg.get("min_p", 0.05)), top_p=float(cfg.get("top_p", 0.9)), eta=float(cfg.get("eta", 3e-3)),
            no_repeat_ngram=int(cfg.get("no_repeat_ngram", 4)),
            frequency_penalty=freq,
            frequency_window=int(cfg.get("frequency_window", 64)), target_words=target,
            max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
            mbr=MBRParams(enabled=n > 1, utility=(cfg.get("mbr") or {}).get("utility", "mixed"),
                          alpha=float((cfg.get("mbr") or {}).get("alpha", 0.5))),
        )
        t0 = time.time()
        outs = llm.generate([prompts[k]["text"] for k in order], params, use_tqdm=False,
                            prepend_bos=bool(cfg.get("prepend_bos", True)))
        ordered = {k: {"text": o.text} for k, o in zip(order, outs)}
        os.makedirs(a.out, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=True)
        gen_s = round(time.time() - t0, 1)
        llm.close()
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        gen_s = None
    scores = phase1_score(a.truth, pred_path)
    words = [len(r["text"].split()) for r in read_json(pred_path).values()]
    row = {"name": a.name, "checkpoint": os.path.abspath(a.checkpoint), "mbr_n": n, "temperature": temp,
           "greedy": a.greedy, "gen_seconds": gen_s, "n": len(words), "exact_128": sum(w == target for w in words),
           **scores}
    write_json(row, os.path.join(a.out, "scores.json"))
    print(a.name, scores, flush=True)


if __name__ == "__main__":
    main()
