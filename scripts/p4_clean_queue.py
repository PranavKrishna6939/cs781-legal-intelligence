"""Train the promising recipes on train.json only.

D-ovl, the 1,517 dev rows that are also in test.json, is not a training
row. D-sel stays the selection set. A step with done.json is skipped.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.cjpe import exp_dir

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOGS = os.path.join(ROOT, "logs")


def _log(name: str) -> str:
    os.makedirs(LOGS, exist_ok=True)
    return os.path.join(LOGS, name)


def run(args: list[str], log_name: str) -> int:
    print(f"[clean] {log_name}: {' '.join(args)}", flush=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with open(_log(log_name), "ab") as handle:
        code = subprocess.Popen(
            [PY, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        ).wait()
    print(f"[clean] {log_name} exit {code}", flush=True)
    if code != 0:
        raise SystemExit(f"{log_name} failed with {code}; see {_log(log_name)}")
    return code


def done(name: str) -> bool:
    return os.path.exists(os.path.join(exp_dir(name), "done.json"))


def main() -> None:
    # Same recipe as A1, without the overlapping dev cases. The saved TAPT
    # checkpoint was fit on dev text as well, so this run starts from the
    # original dense checkpoint.
    if not done("C1"):
        run([
            "-m", "saul.cjpe.train", "--exp", "C1", "--model", "dense_v1",
            "--technique", "lorma", "--context", "bidirectional", "--view", "tail",
            "--limit", "2048", "--epochs", "2", "--no-ovl", "--batch-tokens", "16384",
            "--accum", "4", "--compile", "--attn", "sdpa", "--gate", "0.74",
            "--first-limit", "1024",
        ], "p4_C1.log")
    # Same recipe as the Hybrid fallback that was leading with the overlap
    # still in the pool. Tail 8192 did not fit; tail 4096 did.
    if not done("C5"):
        run([
            "-m", "saul.cjpe.train", "--exp", "C5", "--model", "hybrid_b",
            "--technique", "adapter_bitfit", "--context", "causal", "--view", "tail",
            "--limit", "4096", "--epochs", "2", "--no-ovl", "--batch-tokens", "16384",
            "--attn", "sdpa",
        ], "p4_C5.log")
    print("[clean] lane finished", flush=True)


if __name__ == "__main__":
    main()
