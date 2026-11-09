#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
.venv/bin/python -u scripts/53_bench_inference.py --mode after_verify
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S1_dense_v2ema_minp --mbr_n 1 --temperature 0.8
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S2_dense_v2ema_mbr16 --mbr_n 16 --temperature 0.9
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S3_dense_v1ema_mbr16 --mbr_n 16 --temperature 0.9
.venv/bin/python -u scripts/51_make_submission.py --checkpoint runs/dense_286m/anneal_v2/ckpt_final_ema --tokenizer artifacts/tokenizers/legal32k --out submissions/engine/S4_dense_v2ema_greedy --greedy
echo ALL_DONE
