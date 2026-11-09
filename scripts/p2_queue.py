"""Obsolete: this queue must not be run. The live queue is scripts/p2_queue2.py, and only after the score lane in reports/phase2_plan.md.

Run the Phase-2 experiment order on the one GPU. Each step is a subprocess.

Finished folds are skipped by the trainer (done.json), so the queue can be restarted.
Gates follow reports/phase2_plan.md: echo versus adapted attention, full hybrid
fine-tuning only when LoRA is close, and a second self-training round only when
the first round gains.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P2 = os.environ.get("SAUL_P2_DIR", "/data/cs781-p2")
PY = sys.executable


def run(module: str, args: list[str]) -> None:
    cmd = [PY, "-m", module, *args]
    print("EVENT START " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f"EVENT ERROR rc={rc} {' '.join(cmd)}", flush=True)
        sys.exit(rc)
    print("EVENT TRAIN_DONE " + " ".join(args), flush=True)


def dev(exp: str, fold: int = 0) -> float:
    path = os.path.join(P2, exp, f"fold{fold}", "done.json")
    return float(json.load(open(path, encoding="utf-8"))["dev_f1"])


def mean_dev(exp: str) -> float:
    return sum(dev(exp, k) for k in range(5)) / 5


def main() -> None:
    raise SystemExit(
        "scripts/p2_queue.py is obsolete and must not be run. Use scripts/p2_queue2.py."
    )
    os.environ["SAUL_REQUIRE_DATA"] = "1"
    run("pytest", ["tests/test_ner_decode.py", "tests/test_ner_alignment.py", "-q"])
    probes = [
        ("probe_dense", "dense_v1", "bidirectional"),
        ("probe_hybrid", "hybrid_b", "echo"),
        ("probe_ced", "ced", "encoder"),
        ("probe_conv", "conv_mini", "echo"),
        ("probe_52m", "dense_52m", "bidirectional"),
    ]
    for exp, model, context in probes:
        run("saul.ner.train", ["--exp", exp, "--model", model, "--technique", "probe",
                               "--context", context, "--fold", "0", "--batch-size", "8"])
    # Full backward through a 250M+ model OOMs at batch 16 on the 96GB card.
    # Echo doubles the sequence, so those runs use a smaller batch. Effective batch stays 16.
    dense = ["--batch-size", "8", "--accum", "2"]
    echo = ["--batch-size", "2", "--accum", "8"]
    run("saul.ner.train", ["--exp", "E1smoke", "--model", "dense_v1", "--technique", "full",
                           "--context", "bidirectional", "--fold", "0", "--epochs", "2", *dense])
    run("saul.ner.train", ["--exp", "E1", "--model", "dense_v1", "--technique", "full",
                           "--context", "bidirectional", "--fold", "all", *dense])
    run("saul.ner.submit", ["--exp", "E1", "--name", "P2_E1_dense_full"])
    for header, exp in ((0, "E1h0"), (384, "E1h384")):
        run("saul.ner.train", ["--exp", exp, "--model", "dense_v1", "--technique", "full",
                               "--context", "bidirectional", "--fold", "0", "--epochs", "8",
                               "--header", str(header), *dense])
    run("saul.ner.adapt", ["--model", "ced", "--minutes", "20"])
    run("saul.ner.adapt", ["--model", "hybrid_b", "--minutes", "20"])
    run("saul.ner.train", ["--exp", "E2", "--model", "dense_v1", "--technique", "lorma",
                           "--context", "bidirectional", "--fold", "all", *dense])
    run("saul.ner.submit", ["--exp", "E2", "--name", "P2_E2_dense_lorma"])
    run("saul.ner.train", ["--exp", "E3", "--model", "hybrid_b", "--technique", "lora",
                           "--context", "echo", "--fold", "0", *echo])
    run("saul.ner.train", ["--exp", "E3bidir", "--model", "hybrid_b", "--technique", "lora",
                           "--context", "bidirectional", "--fold", "0", *dense,
                           "--init", os.path.join(P2, "adapt", "hybrid_b")])
    if dev("E3bidir") >= dev("E3") + 0.01:
        print("EVENT GATE hybrid bidirectional attention wins", flush=True)
        hybrid_exp = "E3b"
        run("saul.ner.train", ["--exp", "E3b", "--model", "hybrid_b", "--technique", "lora",
                               "--context", "bidirectional", "--fold", "all", *dense,
                               "--init", os.path.join(P2, "adapt", "hybrid_b")])
    else:
        print("EVENT GATE hybrid echo kept", flush=True)
        hybrid_exp = "E3"
        run("saul.ner.train", ["--exp", "E3", "--model", "hybrid_b", "--technique", "lora",
                               "--context", "echo", "--fold", "all", *echo])
    run("saul.ner.submit", ["--exp", hybrid_exp, "--name", "P2_E3_hybrid_lora"])
    run("saul.ner.train", ["--exp", "E4", "--model", "ced", "--technique", "partial",
                           "--context", "encoder", "--fold", "all", *dense,
                           "--init", os.path.join(P2, "adapt", "ced")])
    run("saul.ner.submit", ["--exp", "E4", "--name", "P2_E4_ced_partial"])
    run("saul.ner.train", ["--exp", "E5", "--model", "hybrid_b", "--technique", "adapter_bitfit",
                           "--context", "echo", "--fold", "all", *echo])
    run("saul.ner.submit", ["--exp", "E5", "--name", "P2_E5_hybrid_adapter"])
    run("saul.ner.train", ["--exp", "E3h5", "--model", "hybrid_h5", "--technique", "lora",
                           "--context", "echo", "--fold", "all", *echo])
    if dev(hybrid_exp) >= dev("E1") - 0.02:
        print("EVENT GATE E3-full", flush=True)
        run("saul.ner.train", ["--exp", "E3full", "--model", "hybrid_b", "--technique", "full",
                               "--context", "echo", "--fold", "all", *echo])
    teachers = f"E1,{hybrid_exp},E3h5"
    run("saul.ner.kd", ["--student", "conv_mini", "--teachers", teachers, "--exp", "E6",
                        "--batch-size", "4"])
    run("saul.ner.train", ["--exp", "E6gold", "--model", "conv_mini", "--technique", "full",
                           "--context", "echo", "--fold", "0", "--epochs", "8", "--batch-size", "4", "--accum", "2"])
    run("saul.ner.submit", ["--exp", "E6", "--name", "P2_E6_conv_kd"])
    run("saul.ner.kd", ["--student", "dense_52m", "--teachers", teachers, "--exp", "E6b",
                        "--batch-size", "8"])
    run("saul.ner.train", ["--exp", "E6bgold", "--model", "dense_52m", "--technique", "full",
                           "--context", "bidirectional", "--fold", "0", "--epochs", "8"])
    run("saul.ner.submit", ["--exp", "E6b", "--name", "P2_E6b_dense52_kd"])
    run("saul.ner.kd", ["--student", "dense_v1", "--teachers", teachers, "--dump-only",
                        "--hard-out", os.path.join(P2, "pool", "hard.jsonl")])
    run("saul.ner.train", ["--exp", "E7", "--model", "dense_v1", "--technique", "full",
                           "--context", "bidirectional", "--fold", "all", *dense,
                           "--pseudo", os.path.join(P2, "pool", "hard.jsonl")])
    if mean_dev("E7") >= mean_dev("E1") + 0.01:
        print("EVENT GATE self-training round 2", flush=True)
        winner = "hybrid_b" if mean_dev(hybrid_exp) > mean_dev("E1") else "dense_v1"
        context = "echo" if winner == "hybrid_b" else "bidirectional"
        run("saul.ner.kd", ["--student", winner, "--teachers", f"E1,{hybrid_exp},E7", "--dump-only",
                            "--teacher-dir", os.path.join(P2, "pool", "teacher_r2"),
                            "--hard-out", os.path.join(P2, "pool", "hard_r2.jsonl")])
        run("saul.ner.train", ["--exp", "E7r2", "--model", winner, "--technique", "full",
                               "--context", context, "--fold", "all", *(echo if context == "echo" else dense),
                               "--pseudo", os.path.join(P2, "pool", "hard_r2.jsonl")])
    run("saul.ner.submit", ["--exp", "E7", "--name", "P2_E7_dense_selftrain"])
    members = [e for e in ("E1", "E2", hybrid_exp, "E7", "E5", "E3full") if os.path.exists(
        os.path.join(P2, e, "fold4", "done.json"))]
    # Hybrid leaves the mixture when it trails dense by more than 5 points. It stays submitted.
    if hybrid_exp in members and mean_dev(hybrid_exp) < mean_dev("E1") - 0.05:
        members = [e for e in members if e != hybrid_exp]
        print("EVENT GATE hybrid dropped from the ensemble", flush=True)
    run("saul.ner.ensemble", ["--exps", ",".join(members), "--name", "E8", "--submit"])
    print("EVENT QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
