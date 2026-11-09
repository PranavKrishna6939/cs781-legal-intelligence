#!/usr/bin/env python
"""Time one decode step and a short generate. Prints milliseconds and tokens/s."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from saul.serve.api import LLM
from saul.serve.sampling import SamplingParams


def step_ms(llm, slots, suffix, steps=20, graphs=True):
    eng = llm.engine
    eng.runner.use_graphs = graphs
    p = eng.pools
    b = p.B
    sb = eng.cfg.bucket(slots, eng.cfg.slot_buckets)
    lp = eng.cfg.bucket(1024, eng.cfg.prompt_buckets)
    ls = eng.cfg.bucket(suffix, eng.cfg.suffix_buckets)
    groups = sb // b
    p.needs_decode[:sb] = True
    p.slen[:sb] = max(1, suffix // 2)
    p.plen[:groups] = min(1024, lp)
    p.plen_slot[:sb] = p.plen[(torch.arange(sb, device=p.device) // b)]
    p.pos[:sb] = p.plen_slot[:sb] + p.slen[:sb] - 1
    p.token_ids[:sb] = 4
    p.layers  # touch
    # write a real key so SDPA is not all-masked
    for layer in p.layers:
        if "K" in layer:
            layer["K"][:sb].normal_()
            layer["V"][:sb].normal_()
    for _ in range(2):
        eng.runner.decode(sb, lp, ls)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.runner.decode(sb, lp, ls)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1000
    p.needs_decode[:sb] = False
    return sb, lp, ls, ms


def main():
    llm = LLM(checkpoint="runs/dense_286m/anneal_v2/ckpt_final_ema", tokenizer_path="artifacts/tokenizers/legal32k",
              max_slots=512, block_size=1, max_prompt_len=1024, max_new_tokens=256, device="cuda",
              dtype="bfloat16", cuda_graphs=False, rouge_workers=1, prefill_token_budget=8192)
    print("compiled", llm.engine.runner.decode_fn is not llm.model.serve_decode, "graphs", llm.engine.runner.use_graphs, flush=True)
    for slots in (16, 128, 512):
        sb, lp, ls, ms = step_ms(llm, slots, suffix=128, graphs=False)
        print(f"slots={sb} {ms:.2f} ms  {1000/ms*sb:.0f} tok/s", flush=True)
    import json
    data = json.load(open("data/raw/Phase_1_Pretraining/Test/test.json"))
    texts = [v["text"] if isinstance(v, dict) else v for v in list(data.values())[:256]]
    llm.engine.pools.zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(texts, SamplingParams(n=1, temperature=0.0, truncation="none", no_repeat_ngram=0,
                                              frequency_penalty=0.0, target_words=None, max_new_tokens=32))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = sum(sum(o.gen_tokens) for o in outs)
    print(f"e2e {len(texts)} prompts x 32 tok: {dt:.2f}s  {gen/dt:.0f} tok/s", flush=True)
    # one prompt, teacher-forced from the greedy trace, against a full forward
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained("artifacts/tokenizers/legal32k")
    ids = tok.encode(texts[0], add_special_tokens=False)
    bos = tok.bos_token_id
    if bos is not None:
        ids = [bos] + ids
    ids = ids[:1024]
    req = llm.engine.submit_ids(ids, SamplingParams(
        n=1, temperature=0.0, truncation="none", no_repeat_ngram=0, frequency_penalty=0.0,
        target_words=None, max_new_tokens=8, record_logits=True))
    llm.engine.run_until_complete()
    forced = [int(row.argmax()) for row in req.traces[0]]
    full = llm.model(torch.tensor([ids + forced], device="cuda")).logits[0].float()
    agree = 0
    for t, row in enumerate(req.traces[0]):
        agree += int(row.argmax() == full[len(ids) - 1 + t].argmax())
    print(f"argmax agree {agree}/{len(forced)}", flush=True)


if __name__ == "__main__":
    main()
