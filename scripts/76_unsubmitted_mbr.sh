#!/bin/bash
# Usage: 76_unsubmitted_mbr.sh CKPT OUT [CKPT OUT ...]
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
TOK=artifacts/tokenizers/legal32k
while [[ $# -ge 2 ]]; do
  $PY -u scripts/51_make_submission.py --checkpoint "$1" --tokenizer "$TOK" \
    --out "$2" --mbr_n 16 --temperature 0.9
  shift 2
done
echo ALL_DONE
