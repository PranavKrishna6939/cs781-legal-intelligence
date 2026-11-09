"""Phase 3 score lane, second half (plan §12.2).

Started after E1, E2 and E3 finished. Waits for the TAPT process passed as argv[1],
then runs the revised order. Finished folds are skipped by the trainer, so this
script can be restarted.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3_queue import P3, run  # noqa: E402


def wait_pid(pid: int) -> None:
    print(f"[queue2] waiting for pid {pid}", flush=True)
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(30)
    print(f"[queue2] pid {pid} gone", flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        wait_pid(int(sys.argv[1]))
    train = ["-m", "saul.rr.train"]
    dense = ["--model", "dense_v1", "--technique", "lorma", "--context", "bidirectional",
             "--fold", "all", "--epochs", "10"]
    adapt = os.path.join(P3, "adapt", "dense_v1")
    if not os.path.exists(os.path.join(adapt, "done.json")):
        run(["-m", "saul.rr.kd", "--tapt"], "p3_tapt.log", fatal=False)
    if os.path.exists(os.path.join(adapt, "done.json")):
        run(train + ["--exp", "E1tapt", *dense, "--init", adapt], "p3_E1tapt.log", fatal=False)
    run(["-m", "saul.rr.kd", "--dump", "E1,E2,E3"], "p3_dump.log", fatal=False)
    if os.path.exists(os.path.join(P3, "kd", "teacher", "done.json")):
        run(["-m", "saul.rr.kd", "--student", "--teachers", "E1,E2,E3"], "p3_E6.log", fatal=False)
    run(train + ["--exp", "E1_seed782", *dense, "--seed", "782"], "p3_E1_seed782.log", fatal=False)
    run(["-m", "saul.rr.kd", "--silver", "--teachers", "E1,E2,E3"], "p3_silver.log", fatal=False)
    silver = os.path.join(P3, "silver", "pool_fold{fold}.jsonl")
    if all(os.path.exists(silver.format(fold=k)) for k in range(5)):
        run(train + ["--exp", "E1silver", *dense, "--silver", silver, "--silver-epochs", "1"],
            "p3_E1silver.log", fatal=False)
    run(["-m", "saul.rr.kd", "--gold-control"], "p3_E6gold.log", fatal=False)
    run(train + ["--exp", "E0", "--model", "dense_v1", "--technique", "lorma", "--context", "bidirectional",
                 "--sentence-only", "--fold", "0", "--epochs", "10"], "p3_E0.log", fatal=False)
    run(train + ["--exp", "E2_seed782", "--model", "hybrid_b", "--technique", "lora", "--context", "echo",
                 "--fold", "all", "--epochs", "10", "--seed", "782"], "p3_E2_seed782.log", fatal=False)
    run(["scripts/p3_analyze.py"], "p3_analyze.log", fatal=False)
    print("[queue2] lane finished", flush=True)


if __name__ == "__main__":
    main()
