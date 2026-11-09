#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
.venv/bin/python -u - << 'PY'
import json, os, time, torch
from saul.serve.api import LLM
from saul.serve.sampling import MBRParams, SamplingParams

data = json.load(open("data/raw/Phase_1_Pretraining/Test/test.json"))
texts = [item["text"] if isinstance(item, dict) else item for item in data.values()]
llm = LLM(checkpoint="runs/dense_286m/anneal_v2/ckpt_final_ema", tokenizer_path="artifacts/tokenizers/legal32k",
          max_slots=512, block_size=16, max_prompt_len=2048, max_new_tokens=1024, device="cuda",
          dtype="bfloat16", cuda_graphs=True, rouge_workers=6, mbr_batch_groups=8)
torch.cuda.reset_peak_memory_stats()
params = SamplingParams(n=16, temperature=0.9, truncation="min_p", target_words=128, max_new_tokens=1024,
                        mbr=MBRParams(enabled=True, utility="mixed", alpha=0.5))
t0 = time.perf_counter()
outs = llm.generate(texts, params)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gen = sum(sum(o.gen_tokens) for o in outs)
row = {"seconds": round(dt, 2), "prompts": len(texts), "generated_tokens": gen,
       "tok_s": round(gen / dt, 1), "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
print("n16_mbr", row, flush=True)
os.makedirs("artifacts/inference", exist_ok=True)
e2e = {
    "n1_minp": {"seconds": 111.57, "tok_s": 1464.7, "prompts": 1000},
    "n1_greedy": {"seconds": 121.48, "tok_s": 1374.1, "prompts": 1000},
    "n16_mbr": row,
}
json.dump(e2e, open("artifacts/inference/e2e.json", "w"), indent=2)
llm.close()
PY
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S1_dense_v2ema_minp --mbr_n 1 --temperature 0.8
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S2_dense_v2ema_mbr16 --mbr_n 16 --temperature 0.9
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S3_dense_v1ema_mbr16 --mbr_n 16 --temperature 0.9
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S4_dense_v2ema_greedy --greedy
echo ALL_DONE
