#!/usr/bin/env bash
# One-shot status of the proxy ablation box (run on gpu_3): queue log, current run, summary, vocab prep, GPU.
cd "${SAUL_ROOT:-$HOME/cs781}"
qlog=$(ls -t logs/32_ablations*.log | head -1)
echo "== queue ($qlog)"; grep -E "^\[(run|done|failed|wait)" "$qlog" | tail -6 | cut -c1-220
cur=$(grep -E "^\[run\]" "$qlog" | tail -1 | sed -E 's/^\[run\] ([^:]+):.*/\1/')
if [ -n "$cur" ] && [ -f "runs/ablations/$cur/metrics.jsonl" ]; then
  echo "== current: $cur"
  python3 - "$cur" << 'PY'
import json, sys
rows = [json.loads(l) for l in open(f"runs/ablations/{sys.argv[1]}/metrics.jsonl") if l.strip()]
tr = [r for r in rows if "loss" in r]
ev = [r for r in rows if "val_loss" in r]
if tr:
    t = tr[-1]
    tail = tr[-10:]
    print(f"step {t['step']}/7600  loss(last10 mean) {sum(r['loss'] for r in tail)/len(tail):.4f}  "
          f"tok/s {t['tokens_per_s']:,.0f}  gnorm {t['gnorm']:.3f}  mem {t['gpu_gb']:.1f}G")
    eta_h = (7600 - t['step']) * 131072 / max(t['tokens_per_s'], 1) / 3600
    print(f"eta this run {eta_h:.2f} h")
for e in ev[-3:]:
    print("eval", {k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()})
PY
  grep -m3 -E "Traceback|Error|recompile_limit|nan" "runs/ablations/$cur/train.log" | cut -c1-200
fi
echo "== summary"; [ -f artifacts/ablations/summary.md ] && cat artifacts/ablations/summary.md
echo "== vocab prep"; tail -2 logs/vocab_prep.log | cut -c1-200
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; df -h / | tail -1
