#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
TOK=artifacts/tokenizers/legal32k
CK=runs/ced_290m/wsd/ckpt_final_ema

$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/CED1_ema_minp --mbr_n 1 --temperature 0.8
$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/CED2_ema_mbr16 --mbr_n 16 --temperature 0.9
$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/CED3_ema_greedy --greedy
echo ALL_DONE
