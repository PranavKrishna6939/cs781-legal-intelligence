"""Call the organiser's Phase-1 evaluator. BERTScore is the metric; nothing here loads a model for generation."""
from __future__ import annotations

import os
import subprocess
import sys


def phase1_score(truth: str, prediction: str, script: str | None = None) -> dict[str, float]:
    script = script or os.path.join(os.path.dirname(__file__), "..", "..", "resources", "evaluation_script",
                                    "evaluate_phase1.py")
    script = os.path.abspath(script)
    proc = subprocess.run([sys.executable, script, "--truth", truth, "--prediction", prediction],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout)
    out = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            out[key] = float(v.strip())
        except ValueError:
            continue
    return out
