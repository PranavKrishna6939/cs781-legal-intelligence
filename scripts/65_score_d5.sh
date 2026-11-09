#!/bin/bash
# After D5 finishes: Supreme Court bits/byte, then the same MBR-16 dev recipe as S3.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
CK=runs/dense_286m/anneal_d5_clm_testmix
$PY -u scripts/41_eval_val.py \
  --checkpoints $CK/ckpt_final $CK/ckpt_final_ema \
  --sources supreme_court \
  --out $CK/sc_bpb.json
$PY -u scripts/60_score_dev.py --name D5 \
  --checkpoint $CK/ckpt_final_ema \
  --mbr_n 16 --temperature 0.9 \
  --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json \
  --out runs/eval/dev_full/D5
echo SCORE_D5_DONE
