#!/usr/bin/env python
"""Stage 40: generate dev continuations for every checkpoint of a run, score them, write selection.json.

Generation uses the frozen decoder config without MBR (checkpoint ranking, not final numbers) on the fast dev
subset by default. Scoring uses the official evaluate_phase1.py. Existing prediction.json files are reused.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.eval.official import phase1_score  # noqa: E402
from saul.serve.api import LLM  # noqa: E402
from saul.serve.sampling import SamplingParams  # noqa: E402
from saul.utils.config import load_config  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="data/final/dev_prompts_fast.json")
    ap.add_argument("--truth", default="data/final/dev_truth_fast.json")
    ap.add_argument("--decode", default="configs/decode/default.yaml")
    ap.add_argument("--last", type=int, default=6, help="score the newest N numbered checkpoints plus stable/final")
    ap.add_argument("--batch_rows", type=int, default=128)
    a = ap.parse_args()
    torch.manual_seed(781)
    cfg = load_config(a.decode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = list(read_json(a.prompts).items())
    numbered = sorted((d for d in glob.glob(os.path.join(a.run_dir, "**", "ckpt_*"), recursive=True)
                       if os.path.basename(d)[5:].isdigit()), key=lambda d: int(os.path.basename(d)[5:]))
    named = [d for d in glob.glob(os.path.join(a.run_dir, "**", "ckpt_*"), recursive=True)
             if os.path.basename(d)[5:] in ("stable", "final", "final_ema")]
    ckpts = numbered[-a.last:] + sorted(named)
    rows = []
    params = SamplingParams(
        n=1, temperature=float(cfg.get("temperature", 0.8)), truncation=cfg.get("truncation", "min_p"),
        min_p=float(cfg.get("min_p", 0.05)), top_p=float(cfg.get("top_p", 0.9)), eta=float(cfg.get("eta", 3e-3)),
        no_repeat_ngram=int(cfg.get("no_repeat_ngram", 4)), frequency_penalty=float(cfg.get("frequency_penalty", 0.3)),
        frequency_window=int(cfg.get("frequency_window", 64)), target_words=int(cfg.get("target_words", 128)),
        max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
    )
    for ck in ckpts:
        name = os.path.relpath(ck, a.run_dir).replace(os.sep, "__")
        pred_path = os.path.join(a.run_dir, "dev_predictions", name, "prediction.json")
        if not os.path.exists(pred_path):
            llm = LLM(checkpoint=ck, tokenizer_path=a.tokenizer, max_slots=min(128, max(1, a.batch_rows)),
                      block_size=1, max_prompt_len=2048, max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
                      device=device, dtype="bfloat16" if device == "cuda" else "float32",
                      cuda_graphs=device == "cuda", rouge_workers=1)
            outs = llm.generate([r["text"] for _, r in prompts], params, prepend_bos=bool(cfg.get("prepend_bos", True)))
            write_json({k: {"text": o.text} for (k, _), o in zip(prompts, outs)}, pred_path)
            llm.close()
            del llm
            if device == "cuda":
                torch.cuda.empty_cache()
        scores = phase1_score(a.truth, pred_path)
        rows.append({"checkpoint": name, **scores})
        print(name, scores, flush=True)
    if not rows:
        raise SystemExit(f"no checkpoints under {a.run_dir}")
    best = max(rows, key=lambda r: r.get("bertscore_f1", -1))
    write_json({"rows": rows, "best": best["checkpoint"]}, os.path.join(a.run_dir, "selection.json"))
    print("best", best["checkpoint"])


if __name__ == "__main__":
    main()
