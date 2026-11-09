#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
root=runs/recovery_20261005
checkpoint=runs/dense_286m/anneal/ckpt_final_ema
.venv/bin/python -u scripts/90_recovery_dev.py
for cell in '4 0.3 baseline' '4 0 frequency_off' '0 0.3 ngram_off' '0 0 both_off'; do
    read -r ngram frequency name <<< "$cell"
    .venv/bin/python -u scripts/91_recovery_decode.py \
        --prompts "$root/dev/screen_prompts.json" --checkpoint "$checkpoint" \
        --out "$root/dense_d1/$name" --ngram "$ngram" --frequency "$frequency"
done
.venv/bin/python -u scripts/92_recovery_score.py \
    --truth "$root/dev/screen_truth.json" --meta "$root/dev/screen_meta.json" \
    --runs "$root/dense_d1/baseline" "$root/dense_d1/frequency_off" \
    "$root/dense_d1/ngram_off" "$root/dense_d1/both_off"
