"""Phase-2 compliance queue. Do not start it before the score lane in reports/phase2_plan.md.

E1v2 is trained by hand and is not launched here. Order:

  E4v2  CED 253M       partial FT (enc 5-10)     encoder, MNTP-adapted init
  E2v2  Dense 286M     LoRMA+                    bidirectional
  E3v2  Hybrid 294M    LoRA                      echo
  E5v2  Hybrid 294M    adapters + BitFit         echo
  E6v2  Conv-Mini 99M  KD from E1v2+E4v2+E3v2    echo   (+ E6gold control, fold 0)
  E8    probability mixture, including E5v2 when that run has finished
  E7v2  Dense 286M     self-training on hard teacher tags (if time remains)

SAUL_P2_RECIPE is split and appended to every train command, and to the KD student.
SAUL_P2_EPOCHS defaults to 10. Submits and ensembles are optional.
Finished folds are skipped (done.json), so the queue can be restarted.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P2 = os.environ.get("SAUL_P2_DIR", "/data/cs781-p2")
PY = sys.executable
EPOCHS = os.environ.get("SAUL_P2_EPOCHS", "10")
RECIPE = os.environ.get("SAUL_P2_RECIPE", "").split()


def run(module: str, args: list[str], optional: bool = False) -> bool:
    cmd = [PY, "-m", module, *args]
    print("EVENT START " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f"EVENT ERROR rc={rc} {' '.join(cmd)}", flush=True)
        if not optional:
            sys.exit(rc)
        return False
    print("EVENT DONE " + " ".join(args), flush=True)
    return True


def done(exp: str) -> bool:
    return all(os.path.exists(os.path.join(P2, exp, f"fold{k}", "done.json")) for k in range(5))


def main() -> None:
    os.environ["SAUL_REQUIRE_DATA"] = "1"
    run("pytest", ["tests/test_ner_decode.py", "tests/test_ner_alignment.py", "-q"])
    dense = ["--batch-size", "8", "--accum", "2", "--epochs", EPOCHS, *RECIPE]
    echo = ["--batch-size", "2", "--accum", "8", "--epochs", EPOCHS, *RECIPE]
    # E1v2 is trained by hand before this queue (score lane). Do not train it here.

    run("saul.ner.adapt", ["--model", "ced", "--minutes", "20"])
    run("saul.ner.train", ["--exp", "E4v2", "--model", "ced", "--technique", "partial",
                           "--context", "encoder", "--fold", "all", *dense,
                           "--init", os.path.join(P2, "adapt", "ced")])
    run("saul.ner.submit", ["--exp", "E4v2", "--name", "P2_E4_ced_partial"], optional=True)

    run("saul.ner.train", ["--exp", "E2v2", "--model", "dense_v1", "--technique", "lorma",
                           "--context", "bidirectional", "--fold", "all", *dense])
    run("saul.ner.submit", ["--exp", "E2v2", "--name", "P2_E2_dense_lorma"], optional=True)

    run("saul.ner.train", ["--exp", "E3v2", "--model", "hybrid_b", "--technique", "lora",
                           "--context", "echo", "--fold", "all", *echo])
    run("saul.ner.submit", ["--exp", "E3v2", "--name", "P2_E3_hybrid_lora"], optional=True)

    run("saul.ner.train", ["--exp", "E5v2", "--model", "hybrid_b", "--technique", "adapter_bitfit",
                           "--context", "echo", "--fold", "all", *echo])
    run("saul.ner.submit", ["--exp", "E5v2", "--name", "P2_E5_hybrid_adapter"], optional=True)

    teachers = ",".join(e for e in ("E1v2", "E4v2", "E3v2") if done(e))
    run("saul.ner.kd", ["--student", "conv_mini", "--teachers", teachers, "--exp", "E6v2",
                        "--epochs", EPOCHS, "--batch-size", "4", *RECIPE])
    run("saul.ner.submit", ["--exp", "E6v2", "--name", "P2_E6_conv_kd"], optional=True)
    run("saul.ner.train", ["--exp", "E6gold", "--model", "conv_mini", "--technique", "full",
                           "--context", "echo", "--fold", "0", "--batch-size", "4", "--accum", "4",
                           "--epochs", EPOCHS, *RECIPE], optional=True)

    members = [e for e in ("E1v2", "E2v2", "E4v2", "E3v2", "E5v2", "E6v2") if done(e)]
    run("saul.ner.ensemble", ["--exps", ",".join(members), "--name", "P2_E8_mix", "--submit"], optional=True)

    run("saul.ner.kd", ["--student", "dense_v1", "--teachers", teachers, "--dump-only",
                        "--hard-out", os.path.join(P2, "pool", "hard.jsonl")], optional=True)
    if os.path.exists(os.path.join(P2, "pool", "hard.jsonl")):
        run("saul.ner.train", ["--exp", "E7v2", "--model", "dense_v1", "--technique", "full",
                               "--context", "bidirectional", "--fold", "all", *dense,
                               "--pseudo", os.path.join(P2, "pool", "hard.jsonl")], optional=True)
        if done("E7v2"):
            run("saul.ner.submit", ["--exp", "E7v2", "--name", "P2_E7_dense_selftrain"], optional=True)
            members = [e for e in ("E1v2", "E2v2", "E4v2", "E3v2", "E5v2", "E6v2", "E7v2") if done(e)]
            run("saul.ner.ensemble", ["--exps", ",".join(members), "--name", "P2_E8_mix2", "--submit"],
                optional=True)
    print("EVENT QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
