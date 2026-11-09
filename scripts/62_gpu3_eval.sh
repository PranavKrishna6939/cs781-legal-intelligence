#!/bin/bash
# Workstream A on gpu_3, then the hybrid-late soup, then H7.
# Checkpoints are expected under runs/eval/ (weights) and runs/hybrid_294m/stable/ckpt_stable (resumable).
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
TOK=artifacts/tokenizers/legal32k
EV=runs/eval

echo "=== A1 supreme-court bits/byte ==="
$PY -u scripts/41_eval_val.py \
  --checkpoints \
    $EV/hybrid_anneal/ckpt_7500 \
    $EV/hybrid_anneal/ckpt_8000 \
    $EV/hybrid_anneal/ckpt_8500 \
    $EV/hybrid_anneal/ckpt_9000 \
    $EV/hybrid_anneal/ckpt_final \
    $EV/hybrid_anneal/ckpt_final_ema \
    $EV/hybrid_extend/ckpt_final \
    $EV/hybrid_extend/ckpt_final_ema \
    $EV/hybrid_stable/ckpt_stable \
    runs/ablations/a03_gr/ckpt_final \
  --sources supreme_court \
  --out $EV/a1_sc_bpb.json

echo "=== soup: hybrid-late ==="
$PY -u scripts/59_soup.py --out $EV/soup_hybrid_late \
  $EV/hybrid_anneal/ckpt_8500 \
  $EV/hybrid_anneal/ckpt_9000 \
  $EV/hybrid_anneal/ckpt_final_ema

echo "=== A5 greedy dev (bits/byte slope companions) ==="
$PY -u scripts/60_score_dev.py --name a03_greedy --checkpoint runs/ablations/a03_gr/ckpt_final \
  --greedy --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json --out $EV/dev_full/a03_greedy
$PY -u scripts/60_score_dev.py --name hybrid_stable_greedy --checkpoint $EV/hybrid_stable/ckpt_stable \
  --greedy --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json --out $EV/dev_full/hybrid_stable_greedy
$PY -u scripts/60_score_dev.py --name hybrid_anneal_ema_greedy --checkpoint $EV/hybrid_anneal/ckpt_final_ema \
  --greedy --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json --out $EV/dev_full/hybrid_anneal_ema_greedy

echo "=== A2 eight submitted recipes on the full dev set ==="
score() {
  $PY -u scripts/60_score_dev.py --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json "$@"
}
score --name H3 --checkpoint $EV/hybrid_anneal/ckpt_final_ema --mbr_n 16 --temperature 0.9 --out $EV/dev_full/H3
score --name S3 --checkpoint $EV/dense_v1_ema --mbr_n 16 --temperature 0.9 --out $EV/dev_full/S3
score --name H2 --checkpoint $EV/hybrid_extend/ckpt_final_ema --mbr_n 16 --temperature 0.9 --out $EV/dev_full/H2
score --name S2 --checkpoint $EV/dense_v2_ema --mbr_n 16 --temperature 0.9 --out $EV/dev_full/S2
score --name H4 --checkpoint $EV/hybrid_extend/ckpt_final_ema --greedy --out $EV/dev_full/H4
score --name H1 --checkpoint $EV/hybrid_extend/ckpt_final_ema --mbr_n 1 --temperature 0.8 --out $EV/dev_full/H1
score --name S4 --checkpoint $EV/dense_v2_ema --greedy --out $EV/dev_full/S4
score --name S1 --checkpoint $EV/dense_v2_ema --mbr_n 1 --temperature 0.8 --out $EV/dev_full/S1

$PY - << 'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("runs/eval/dev_full/*/scores.json")):
    rows.append(json.load(open(p)))
rows.sort(key=lambda r: r.get("bertscore_f1", -1), reverse=True)
json.dump(rows, open("runs/eval/dev_full/summary.json", "w"), indent=2)
print("DEV_RANK")
for r in rows:
    print(f"{r['name']:28} bert {r.get('bertscore_f1')} rouge {r.get('rouge_l_f1')} bleu {r.get('bleu_4')}")
PY

echo "=== H7 hybrid 55% Supreme Court anneal ==="
if ! $PY -c "import fla; print('fla', getattr(fla, '__version__', 'ok'))"; then
  echo "H7_SKIPPED no fla"
  exit 0
fi
if [[ ! -f runs/hybrid_294m/stable/ckpt_stable/trainer.pt ]]; then
  echo "H7_SKIPPED no resumable ckpt_stable"
  exit 0
fi
$PY -u scripts/30_train.py --config configs/train/hybrid.yaml \
  run_dir=runs/hybrid_294m/anneal_h7_sc55 \
  train.mixture=anneal data_config=configs/data_anneal_sc55.yaml \
  --resume runs/hybrid_294m/stable/ckpt_stable
echo ALL_DONE
