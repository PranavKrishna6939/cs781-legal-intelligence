#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
TOK=artifacts/tokenizers/legal32k
EXT=runs/hybrid_294m/extend/ckpt_final_ema
ANN=runs/hybrid_294m/anneal/ckpt_final_ema

$PY -u scripts/51_make_submission.py --checkpoint "$EXT" --tokenizer "$TOK" \
  --out submissions/H1_hybrid_extendema_minp --mbr_n 1 --temperature 0.8
$PY -u scripts/51_make_submission.py --checkpoint "$EXT" --tokenizer "$TOK" \
  --out submissions/H2_hybrid_extendema_mbr16 --mbr_n 16 --temperature 0.9
$PY -u scripts/51_make_submission.py --checkpoint "$ANN" --tokenizer "$TOK" \
  --out submissions/H3_hybrid_annealema_mbr16 --mbr_n 16 --temperature 0.9
$PY -u scripts/51_make_submission.py --checkpoint "$EXT" --tokenizer "$TOK" \
  --out submissions/H4_hybrid_extendema_greedy --greedy
echo ALL_DONE
