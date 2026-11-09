#!/bin/bash
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
# The previous server must have been stopped after the confirmation/audit completed.
.venv/bin/python -u - <<'PY'
import runpy
q=runpy.run_path('scripts/99_recovery_training_queue.py',run_name='queue_helpers')
R=q['RUN']; PY=q['PY']; run=q['run']
checkpoint='runs/dense_286m/anneal/ckpt_final_ema'
q['state']('decoder_allow_two')
with q['server'](checkpoint,'allow_two'):
    run(PY,'-u','scripts/91_recovery_decode.py','--checkpoint',checkpoint,
        '--prompts',R/'dev/screen_prompts.json','--out',R/'dense_d1/allow_two',
        '--max_ngram_occurrences','2')
run(PY,'-u','scripts/92_recovery_score.py','--truth',R/'dev/screen_truth.json',
    '--meta',R/'dev/screen_meta.json','--runs',R/'dense_d1/baseline',
    R/'dense_d1/ngram_off',R/'dense_d1/allow_two')
PY
exec .venv/bin/python -u scripts/99_recovery_training_queue.py
