#!/usr/bin/env bash
# Phase 5 v2 GPU queue on the A10G. One job at a time; waits for the PID given as $1 first.
set -u
cd "$HOME/cs781"
export SAUL_P5_DIR=$HOME/cs781/p5 SAUL_CKPT_ROOT=$HOME/cs781/checkpoints WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
COMMON="--head pooled --limit 512 --compile-blocks --eval-per-epoch --no-wandb --batch-tokens 16384 --accum 2"

if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

run() {
  local exp=$1; shift
  if [ -f "p5/exp/$exp/done.json" ]; then echo "[queue] skip $exp"; return; fi
  echo "[queue] start $exp $(date -u +%H:%M)"
  $PY -m saul.lsi.train --exp "$exp" "$@" > "logs/p5_$exp.log" 2>&1
  echo "[queue] end $exp rc=$? $(date -u +%H:%M)"
}

run E1_dense_lorma --model dense_v1 --technique lorma --context bidirectional --loss bce \
  --view headtail --head-tokens 766 --tail-tokens 256 --max-chunks 2 --ckpt-every 1 --epochs 2 \
  --adapter-lr 5e-4 --head-lr 1e-3 $COMMON
run E3_hybrid_adapter --model hybrid_b --technique adapter --context causal --loss asl \
  --view headtail --head-tokens 383 --tail-tokens 128 --max-chunks 1 --ckpt-every 1 --epochs 2 \
  --adapter-lr 5e-4 --head-lr 1e-3 $COMMON
run E5_dense_full --model dense_v1 --technique full --context bidirectional --loss bce \
  --view headtail --head-tokens 766 --tail-tokens 256 --max-chunks 2 --ckpt-every 1 --epochs 2 \
  --base-lr 4e-5 --head-lr 1e-3 --train-logits $COMMON
echo "[queue] all done $(date -u +%H:%M)"
