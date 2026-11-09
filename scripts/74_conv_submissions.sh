#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
TOK=artifacts/tokenizers/legal32k
CK=runs/conv_mini/conv_mini_kd/student

$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/C1_conv_minp --mbr_n 1 --temperature 0.8
$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/C2_conv_mbr16 --mbr_n 16 --temperature 0.9
$PY -u scripts/51_make_submission.py --checkpoint "$CK" --tokenizer "$TOK" \
  --out submissions/C3_conv_greedy --greedy
echo ALL_DONE
