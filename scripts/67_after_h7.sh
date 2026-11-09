#!/bin/bash
# After H7 finishes: score H7 and the hybrid-late soup, then the fast-dev decode sweep
# on the current best checkpoint (hybrid anneal EMA).
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
CK=runs/hybrid_294m/anneal_h7_sc55
$PY -u scripts/41_eval_val.py \
  --checkpoints $CK/ckpt_final $CK/ckpt_final_ema \
  --sources supreme_court \
  --out $CK/sc_bpb.json
$PY -u scripts/60_score_dev.py --name H7 \
  --checkpoint $CK/ckpt_final_ema \
  --mbr_n 16 --temperature 0.9 \
  --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json \
  --out runs/eval/dev_full/H7
$PY -u scripts/60_score_dev.py --name soup_late \
  --checkpoint runs/eval/soup_hybrid_late \
  --mbr_n 16 --temperature 0.9 \
  --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json \
  --out runs/eval/dev_full/soup_late
echo SCORE_H7_DONE
$PY -u scripts/50_decode_sweep.py \
  --checkpoint runs/eval/hybrid_anneal/ckpt_final_ema \
  --tokenizer artifacts/tokenizers/legal32k \
  --prompts data/final/dev_prompts_fast.json \
  --truth data/final/dev_truth_fast.json \
  --out runs/decode_sweep/hybrid_anneal_ema
echo SWEEP_DONE
