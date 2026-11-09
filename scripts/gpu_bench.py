#!/usr/bin/env python
"""Record RTX PRO 6000 kernel availability. Delegates to scripts/env_check.py and stores the transcript.

On a machine without the GPU this still runs and writes artifacts/gpu_bench.json with status cpu_only,
so the training box can be checked with the same command:

    python scripts/gpu_bench.py
    python scripts/30_train.py --config configs/train/base.yaml --bench 50
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "env_check.py")],
                          capture_output=True, text=True)
    out = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "ok" if proc.returncode == 0 else "failed_or_cpu",
    }
    path = os.path.join(ROOT, "artifacts", "gpu_bench.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print(f"wrote {path}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
