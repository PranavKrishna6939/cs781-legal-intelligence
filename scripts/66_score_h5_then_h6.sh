#!/bin/bash
# After H5 finishes: score it, then the 600-step low-LR polish (H6) at 2K context.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
CK=runs/hybrid_294m/anneal_h5_testmix
$PY -u scripts/41_eval_val.py \
  --checkpoints $CK/ckpt_final $CK/ckpt_final_ema \
  --sources supreme_court \
  --out $CK/sc_bpb.json
$PY -u scripts/60_score_dev.py --name H5 \
  --checkpoint $CK/ckpt_final_ema \
  --mbr_n 16 --temperature 0.9 \
  --prompts data/final/dev_prompts.json --truth data/final/dev_truth.json \
  --out runs/eval/dev_full/H5
echo SCORE_H5_DONE
$PY -u scripts/30_train.py --config configs/train/hybrid.yaml \
  run_dir=runs/hybrid_294m/anneal_h6_polish \
  train.mixture=anneal data_config=configs/data_anneal_h6.yaml \
  train.seq_len=2048 train.long_doc_prob=0 train.mntp=false \
  train.total_steps=600 train.eval_every=200 train.ckpt_every=200 \
  optim.muon_lr=0.0005 optim.adamw_lr=0.0005 optim.embed_lr=0.0005 \
  optim.warmup_frac=0.05 optim.decay_frac=0.3 \
  --init_from $CK/ckpt_final
echo H6_DONE
