#!/bin/bash
# Build the in-training dev set, score the eight submitted recipes on it, fit the
# calibration weight, then distill Conv-Mini from the dense v1 EMA.
set -euo pipefail
cd "${SAUL_ROOT:-$HOME/cs781}"
PY=.venv/bin/python
$PY -u scripts/58_make_dev_intrain.py
P=data/final/dev_intrain_prompts.json
T=data/final/dev_intrain_truth.json
score() { $PY -u scripts/60_score_dev.py --prompts "$P" --truth "$T" "$@"; }
score --name H3 --checkpoint runs/eval/hybrid_anneal/ckpt_final_ema --mbr_n 16 --temperature 0.9 --out runs/eval/dev_intrain/H3
score --name S3 --checkpoint runs/eval/dense_v1_ema --mbr_n 16 --temperature 0.9 --out runs/eval/dev_intrain/S3
score --name H2 --checkpoint runs/eval/hybrid_extend/ckpt_final_ema --mbr_n 16 --temperature 0.9 --out runs/eval/dev_intrain/H2
score --name S2 --checkpoint runs/eval/dense_v2_ema --mbr_n 16 --temperature 0.9 --out runs/eval/dev_intrain/S2
score --name H4 --checkpoint runs/eval/hybrid_extend/ckpt_final_ema --greedy --out runs/eval/dev_intrain/H4
score --name H1 --checkpoint runs/eval/hybrid_extend/ckpt_final_ema --mbr_n 1 --temperature 0.8 --out runs/eval/dev_intrain/H1
score --name S4 --checkpoint runs/eval/dense_v2_ema --greedy --out runs/eval/dev_intrain/S4
score --name S1 --checkpoint runs/eval/dense_v2_ema --mbr_n 1 --temperature 0.8 --out runs/eval/dev_intrain/S1
$PY - << 'PY'
import json, glob
coda = {
    "H3": (0.849464, 0.234969, 0.117418),
    "S3": (0.849266, 0.231938, 0.110233),
    "H2": (0.849250, 0.231960, 0.109850),
    "S2": (0.848465, 0.228758, 0.105745),
    "H4": (0.844124, 0.222836, 0.102763),
    "H1": (0.843041, 0.213367, 0.092086),
    "S4": (0.842879, 0.220758, 0.101602),
    "S1": (0.842582, 0.211227, 0.090514),
}
keys = ("bertscore_f1", "rouge_l_f1", "bleu_4")
held, intra = {}, {}
for kind, dest in (("dev_full", held), ("dev_intrain", intra)):
    for p in glob.glob(f"runs/eval/{kind}/*/scores.json"):
        r = json.load(open(p))
        if r["name"] in coda:
            dest[r["name"]] = tuple(r[k] for k in keys)
names = [n for n in coda if n in held and n in intra]
best, best_w = None, None
for i in range(21):
    w = i / 20
    err = 0.0
    for n in names:
        pred = tuple(w * intra[n][j] + (1 - w) * held[n][j] for j in range(3))
        err += sum((pred[j] - coda[n][j]) ** 2 for j in range(3))
    if best is None or err < best:
        best, best_w = err, w
out = {"w_intrain": best_w, "mse": best, "n": len(names), "names": names}
json.dump(out, open("runs/eval/calibration.json", "w"), indent=2)
print("CALIBRATION", out, flush=True)
PY
echo CALIBRATION_DONE
$PY -u scripts/33_distill.py --teacher runs/eval/dense_v1_ema --steps 2000 \
  run_dir=runs/conv_mini train.mntp=false
echo KD_DONE
