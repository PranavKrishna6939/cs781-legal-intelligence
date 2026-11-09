#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
root=runs/recovery_20261005
checkpoint=runs/dense_286m/anneal/ckpt_final_ema
for cell in '4 baseline' '0 ngram_off'; do
    read -r ngram name <<< "$cell"
    if [[ -f "$root/dense_confirmation/$name/prediction.json" ]]; then
        continue
    fi
    .venv/bin/python -u scripts/91_recovery_decode.py \
        --prompts "$root/dev/confirmation_prompts.json" --checkpoint "$checkpoint" \
        --out "$root/dense_confirmation/$name" --ngram "$ngram" --frequency 0.3 --allow_incomplete
done
.venv/bin/python -u scripts/92_recovery_score.py \
    --truth "$root/dev/confirmation_truth.json" --meta "$root/dev/confirmation_meta.json" \
    --runs "$root/dense_confirmation/baseline" "$root/dense_confirmation/ngram_off"
.venv/bin/python -u scripts/97_recovery_pool_audit.py \
    --run "$root/dense_d1/ngram_off" --truth "$root/dev/screen_truth.json" \
    --out "$root/dense_d1/ngram_off_rouge"
.venv/bin/python -u scripts/92_recovery_score.py \
    --truth "$root/dev/screen_truth.json" --meta "$root/dev/screen_meta.json" \
    --runs "$root/dense_d1/baseline" "$root/dense_d1/ngram_off" "$root/dense_d1/ngram_off_rouge"
