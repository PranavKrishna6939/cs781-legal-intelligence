#!/usr/bin/env python
"""Stage 50: decoding sweep on the fast dev subset.

Pass 1 scores every (temperature, truncation, n-gram block, frequency penalty) cell with one sample per prompt.
Pass 2 adds MBR (16 samples, mixed own-model/ROUGE-L utility) to the best `--mbr_top` cells only.
The winner is written to configs/decode/final.yaml. roberta is used only inside the official evaluator.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from saul.eval.official import phase1_score  # noqa: E402
from saul.serve.api import LLM  # noqa: E402
from saul.serve.sampling import MBRParams, SamplingParams  # noqa: E402
from saul.utils.config import save_config  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402


def run_cell(llm, items, cell: dict) -> dict:
    n = int(cell["mbr"]["n"]) if cell["mbr"]["enabled"] else 1
    temp = float(cell["temperature"])
    params = SamplingParams(
        n=n, temperature=temp, truncation="none" if temp <= 0 else cell["truncation"],
        min_p=float(cell.get("min_p", 0.05)), top_p=float(cell.get("top_p", 0.9)),
        eta=float(cell.get("eta", 3e-3)), no_repeat_ngram=int(cell["no_repeat_ngram"]),
        frequency_penalty=float(cell["frequency_penalty"]), frequency_window=int(cell.get("frequency_window", 64)),
        target_words=int(cell["target_words"]), max_new_tokens=int(cell["max_new_tokens"]),
        mbr=MBRParams(enabled=n > 1, utility=cell["mbr"].get("utility", "mixed"), alpha=float(cell["mbr"].get("alpha", 0.5))),
    )
    prompts = [r["text"] for _, r in items]
    outs = llm.generate(prompts, params, prepend_bos=bool(cell.get("prepend_bos", True)))
    return {k: {"text": o.text} for (k, _), o in zip(items, outs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="data/final/dev_prompts_fast.json")
    ap.add_argument("--truth", default="data/final/dev_truth_fast.json")
    ap.add_argument("--out", default="runs/decode_sweep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_prompts", type=int, default=64, help="rows per generation batch (prompts x samples)")
    ap.add_argument("--mbr_top", type=int, default=3)
    a = ap.parse_args()
    items = list(read_json(a.prompts).items())
    if a.limit:
        items = items[:a.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = LLM(checkpoint=a.checkpoint, tokenizer_path=a.tokenizer, max_slots=256, block_size=16,
              max_prompt_len=2048, max_new_tokens=1024, truncate_left=True, device=device,
              dtype="bfloat16" if device == "cuda" else "float32", cuda_graphs=device == "cuda", rouge_workers=6)
    os.makedirs(a.out, exist_ok=True)

    base = {"target_words": 128, "max_new_tokens": 1024, "prepend_bos": True, "min_p": 0.05, "top_p": 0.9,
            "eta": 3e-3, "frequency_window": 64}
    rows = []

    def score(cell: dict, tag: str) -> dict:
        path = os.path.join(a.out, tag, "prediction.json")
        write_json(run_cell(llm, items, cell), path)
        row = {**{k: cell[k] for k in ("temperature", "truncation", "no_repeat_ngram", "frequency_penalty")},
               "mbr": cell["mbr"]["enabled"], **phase1_score(a.truth, path)}
        rows.append(row)
        print(row, flush=True)
        return row

    grid = list(itertools.product([0.5, 0.8], ["min_p", "top_p", "eta"], [0, 4], [0.0, 0.3]))
    cells = []
    for i, (temp, trunc, ngram, freq) in enumerate(grid):
        cell = {**base, "temperature": temp, "truncation": trunc, "no_repeat_ngram": ngram,
                "frequency_penalty": freq, "mbr": {"enabled": False, "n": 1, "utility": "mixed", "alpha": 0.5}}
        cells.append((score(cell, f"cell_{i:03d}"), cell))
    cells.sort(key=lambda rc: rc[0].get("bertscore_f1", -1), reverse=True)
    for j, (_, cell) in enumerate(cells[:a.mbr_top]):
        mcell = {**cell, "mbr": {"enabled": True, "n": 16, "utility": "mixed", "alpha": 0.5}}
        score(mcell, f"mbr_{j:02d}")
        cells.append((rows[-1], mcell))
    best_row, best = max(cells, key=lambda rc: rc[0].get("bertscore_f1", -1))
    write_json({"rows": rows, "best": best_row}, os.path.join(a.out, "sweep.json"))
    save_config(best, "configs/decode/final.yaml")
    print("best", best_row)


if __name__ == "__main__":
    main()
