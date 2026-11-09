"""Recipe screens R0–R7 on two folds. A launcher: this file is not started by the code pass.

Adoption (plan §7.2 and §S.7): mean macro gain >= +1.0 and no fold worse by more than 1.5,
or the mean of the six role labels gains >= +3.0.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable

SCREENS = [
    ("R0", []),
    ("R1", ["--lr", "2e-5", "--head-lr", "3e-4", "--adam-eps", "1e-6"]),
    ("R2", ["--hidden-dropout", "0.1", "--rdrop", "1.0"]),
    ("R3", ["--fgm", "1.0"]),
    ("R4", ["--reinit-top", "2"]),
    ("R5", ["--augment", "mention_swap"]),
    ("R6-384", ["--header", "384"]),
    ("R6-0", ["--header", "0"]),
    ("R7", ["--preamble-flag"]),
]


def main() -> None:
    fold = "0,1"
    if "--fold" in sys.argv:
        fold = sys.argv[sys.argv.index("--fold") + 1]
    base = ["--model", "dense_v1", "--technique", "full", "--context", "bidirectional",
            "--fold", fold, "--batch-size", "8", "--accum", "2", "--epochs", "10"]
    for name, extra in SCREENS:
        cmd = [PY, "-m", "saul.ner.train", "--exp", name, *base, *extra]
        print("EVENT START " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"EVENT ERROR rc={rc} {name}", flush=True)
            sys.exit(rc)
    print("Adoption is computed from done.json dev_f1 after the runs: "
          "mean gain >= +1.0 and no fold drop > 1.5, or role-label mean >= +3.0.")


if __name__ == "__main__":
    main()
