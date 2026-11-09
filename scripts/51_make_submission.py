#!/usr/bin/env python
"""Stage 51: generate prediction.json for the official test prompts and zip it at the archive root.

The output mirrors resources/starter_code/submission.zip exactly: one `prediction.json` at the zip root, a JSON object
{"1": {"text": ...}, ..., "1000": {"text": ...}} in the test file's key order, 2-space indent, ASCII-only (non-ASCII
characters are \\u-escaped), no trailing newline, deflate compression. The file is re-read and checked with the
official evaluator's own rules before the zip is written.

    python scripts/51_make_submission.py --checkpoint runs/.../ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k \
        --out submissions/dense_v2_ema_mbr16 [--mbr_n 16] [--temperature 0.8] [--greedy] [--limit 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.serve.api import LLM  # noqa: E402
from saul.serve.sampling import MBRParams, SamplingParams  # noqa: E402
from saul.utils.config import load_config, to_plain  # noqa: E402
from saul.utils.io import read_json, write_json  # noqa: E402

SPECIAL_MARKERS = ("<|", "|>", "<NAME>", "<ORG>", "<SECTION>", "<ENTITY>", "<ACT>", "<CITATION>")


def check_prediction(path: str, test: dict, target_words: int) -> dict:
    """The evaluator's validate(): non-empty dict, str ids, each record has a non-empty str `text`; ids == test ids."""
    def no_dups(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON keys")
        return dict(pairs)

    data = json.load(open(path, encoding="utf-8"), object_pairs_hook=no_dups)
    if not isinstance(data, dict) or not data:
        raise ValueError("prediction must be a non-empty JSON object")
    for i, rec in data.items():
        if not isinstance(i, str) or not i.strip() or not isinstance(rec, dict) or set(rec) != {"text"} \
                or not isinstance(rec["text"], str) or not rec["text"].strip():
            raise ValueError(f"invalid record for id {i!r}")
    if list(data) != list(test):
        raise ValueError("ids or id order differ from the test file")
    words = [len(r["text"].split()) for r in data.values()]
    leaks = [i for i, r in data.items() if any(m in r["text"] for m in SPECIAL_MARKERS)]
    raw = open(path, "rb").read()
    return {"n": len(data), "min_words": min(words), "max_words": max(words), "exact_128": sum(w == target_words for w in words),
            "special_token_leaks": len(leaks), "ascii_only": all(b < 128 for b in raw), "bytes": len(raw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test", default="data/raw/Phase_1_Pretraining/Test/test.json")
    ap.add_argument("--decode", default="configs/decode/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mbr_n", type=int, default=None, help="candidates per prompt for MBR; 1 disables MBR")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test; no zip)")
    ap.add_argument("--batch_rows", type=int, default=512, help="prompts x samples decoded together")
    ap.add_argument("--max_prompts", type=int, default=64, help="prompts per prefill (dense [B,T,T] mask memory)")
    ap.add_argument("--seed", type=int, default=781)
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    cfg = to_plain(load_config(a.decode))
    mbr = dict(cfg.get("mbr") or {})
    n = int(mbr.get("n", 16)) if mbr.get("enabled", False) else 1
    if a.mbr_n is not None:
        n = max(1, a.mbr_n)
    temp = float(mbr.get("temperature", cfg.get("temperature", 0.8))) if n > 1 else float(cfg.get("temperature", 0.8))
    if a.temperature is not None:
        temp = a.temperature
    if a.greedy:
        temp, n = 0.0, 1
    target = int(cfg.get("target_words", 128))
    test = read_json(a.test)
    order = list(test)
    if a.limit:
        order = order[:a.limit]
    block = max(1, n)
    slots = (512 // block) * block
    llm = LLM(checkpoint=a.checkpoint, tokenizer_path=a.tokenizer, max_slots=slots, block_size=block,
              max_prompt_len=2048, max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
              prefill_token_budget=65536, admit_batch_blocks=min(8, max(1, a.max_prompts)),
              cuda_graphs=torch.cuda.is_available(), device="cuda" if torch.cuda.is_available() else "cpu",
              dtype="bfloat16", rouge_workers=6)
    params = SamplingParams(
        n=n, temperature=temp, truncation="none" if temp <= 0 else cfg.get("truncation", "min_p"),
        min_p=float(cfg.get("min_p", 0.05)), top_p=float(cfg.get("top_p", 0.9)), eta=float(cfg.get("eta", 3e-3)),
        no_repeat_ngram=int(cfg.get("no_repeat_ngram", 4)), frequency_penalty=float(cfg.get("frequency_penalty", 0.3)),
        frequency_window=int(cfg.get("frequency_window", 64)), target_words=target,
        max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
        mbr=MBRParams(enabled=n > 1, utility=mbr.get("utility", "mixed"), alpha=float(mbr.get("alpha", 0.5))),
    )
    t0 = time.time()
    prompts = [test[k]["text"] for k in order]
    outs = llm.generate(prompts, params, use_tqdm=False, prepend_bos=bool(cfg.get("prepend_bos", True)))
    preds = {str(k): {"text": o.text} for k, o in zip(order, outs)}
    reasons = {}
    for o in outs:
        for r in o.finish_reasons:
            reasons[r] = reasons.get(r, 0) + 1
    print(f"{len(preds)}/{len(order)}  {time.time() - t0:.0f}s", flush=True)

    ordered = {k: preds[k] for k in order}
    os.makedirs(a.out, exist_ok=True)
    pred_path = os.path.join(a.out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=True)
    info = {"checkpoint": os.path.abspath(a.checkpoint), "decode_config": a.decode, "temperature": temp,
            "mbr_n": n, "truncation": cfg.get("truncation"), "no_repeat_ngram": cfg.get("no_repeat_ngram"),
            "frequency_penalty": cfg.get("frequency_penalty"), "seed": a.seed, "seconds": round(time.time() - t0, 1),
            "finish_reasons": reasons, "engine": llm.stats()}
    if a.limit:
        print(json.dumps(info))
        return
    check = check_prediction(pred_path, test, target)
    if check["n"] != len(test) or check["special_token_leaks"] or check["min_words"] < 1:
        raise SystemExit(f"prediction check failed: {check}")
    if check["exact_128"] != len(test):
        print(f"[note] {len(test) - check['exact_128']} continuation(s) shorter than {target} words", flush=True)
    zpath = os.path.join(a.out, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        if names != ["prediction.json"] or z.read("prediction.json") != open(pred_path, "rb").read():
            raise SystemExit(f"zip layout check failed: {names}")
    info.update(check, sha256=hashlib.sha256(open(pred_path, "rb").read()).hexdigest(), zip=zpath)
    write_json(info, os.path.join(a.out, "manifest.json"))
    print(json.dumps(info), flush=True)
    print("wrote", zpath)


if __name__ == "__main__":
    main()
