"""Phase 3 score lane.

Run on the GPU server from ~/cs781 after rsync. A fold that already has done.json is
skipped by the trainer, so restarting this script continues the lane.

Children are started in their own session. Logs are absolute paths under ~/cs781/logs.
"""
from __future__ import annotations

import json
import os
import subprocess

_OPEN = []

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOGS = os.path.join(ROOT, "logs")
P3 = os.environ.get("SAUL_P3_DIR", "/data/cs781-p3")


def _log(name: str) -> str:
    os.makedirs(LOGS, exist_ok=True)
    return os.path.join(LOGS, name)


def launch(args: list[str], log_name: str) -> subprocess.Popen:
    log_path = _log(log_name)
    print(f"[queue] {log_name}: {' '.join(args)}", flush=True)
    handle = open(log_path, "ab")
    _OPEN.append(handle)
    return subprocess.Popen(
        [PY, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
    )


def wait(proc: subprocess.Popen, log_name: str, fatal: bool) -> int:
    code = proc.wait()
    print(f"[queue] {log_name} exit {code}", flush=True)
    if code != 0 and fatal:
        raise SystemExit(f"{log_name} failed with {code}; see { _log(log_name) }")
    return code


def run(args: list[str], log_name: str, fatal: bool = True) -> int:
    return wait(launch(args, log_name), log_name, fatal)


def run_pair(jobs: list[tuple[list[str], str, bool]], one_lane: bool) -> None:
    if one_lane or len(jobs) == 1:
        for args, log_name, fatal in jobs:
            run(args, log_name, fatal)
        return
    procs = [(launch(args, log_name), log_name, fatal) for args, log_name, fatal in jobs]
    failed = None
    for proc, log_name, fatal in procs:
        code = proc.wait()
        print(f"[queue] {log_name} exit {code}", flush=True)
        if code != 0 and fatal and failed is None:
            failed = log_name
    if failed:
        raise SystemExit(f"{failed} failed; see {_log(failed)}")


def mean_dev(exp: str) -> float | None:
    scores = []
    for k in range(5):
        path = os.path.join(P3, exp, f"fold{k}", "done.json")
        if not os.path.exists(path):
            return None
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("dev_f1") is None:
            return None
        scores.append(blob["dev_f1"])
    return sum(scores) / len(scores)


def smoke_peak() -> float:
    path = os.path.join(P3, "E1smoke", "fold0", "done.json")
    if not os.path.exists(path):
        return 99.0
    return float(json.load(open(path, encoding="utf-8")).get("peak_gb") or 99.0)


def best_single(exps: list[str]) -> str | None:
    scored = [(mean_dev(exp), exp) for exp in exps]
    scored = [(s, e) for s, e in scored if s is not None]
    if not scored:
        return None
    return max(scored)[1]


def main() -> None:
    train = ["-m", "saul.rr.train"]
    pool = launch(["-m", "saul.rr.kd", "--pool"], "p3_pool.log")
    run(train + ["--exp", "E1smoke", "--model", "dense_v1", "--technique", "lorma",
                 "--context", "bidirectional", "--fold", "0", "--smoke"], "p3_E1smoke.log")
    one_lane = smoke_peak() > 50.0
    print(f"[queue] smoke peak {smoke_peak():.1f} GB; one_lane={one_lane}", flush=True)
    e1 = train + ["--exp", "E1", "--model", "dense_v1", "--technique", "lorma",
                  "--context", "bidirectional", "--fold", "all", "--epochs", "10", "--g1-min", "0.415"]
    e3 = train + ["--exp", "E3", "--model", "ced", "--technique", "ia3",
                  "--context", "encoder", "--fold", "all", "--epochs", "10"]
    if not one_lane:
        # §6: paired Dense/CED jobs use batch 4 × accumulation 4 so each stays near 40 GB.
        e1 += ["--batch-size", "4", "--accum", "4"]
        e3 += ["--batch-size", "4", "--accum", "4"]
    run_pair([(e1, "p3_E1.log", True), (e3, "p3_E3.log", False)], one_lane)
    run(train + ["--exp", "E0", "--model", "dense_v1", "--technique", "lorma",
                 "--context", "bidirectional", "--sentence-only", "--fold", "all", "--epochs", "10"],
        "p3_E0.log", fatal=False)
    run(train + ["--exp", "E2", "--model", "hybrid_b", "--technique", "lora",
                 "--context", "echo", "--fold", "all", "--epochs", "10"], "p3_E2.log", fatal=False)
    pool_code = pool.wait()
    print(f"[queue] pool exit {pool_code}", flush=True)
    if pool_code == 0:
        run(["-m", "saul.rr.kd", "--tapt"], "p3_tapt.log", fatal=False)
        adapt = os.path.join(P3, "adapt", "dense_v1")
        if os.path.exists(os.path.join(adapt, "done.json")):
            run(train + ["--exp", "E1tapt", "--model", "dense_v1", "--technique", "lorma",
                         "--context", "bidirectional", "--fold", "all", "--epochs", "10",
                         "--init", adapt], "p3_E1tapt.log", fatal=False)
        run(["-m", "saul.rr.kd", "--dump", "E1,E2,E3"], "p3_dump.log", fatal=False)
        if os.path.exists(os.path.join(P3, "kd", "teacher", "done.json")):
            run(["-m", "saul.rr.kd", "--student", "--teachers", "E1,E2,E3"], "p3_E6.log", fatal=False)
        run(["-m", "saul.rr.kd", "--silver", "--teachers", "E1,E2,E3"], "p3_silver.log", fatal=False)
        silver = os.path.join(P3, "silver", "pool.jsonl")
        if os.path.exists(silver):
            run(train + ["--exp", "E1silver", "--model", "dense_v1", "--technique", "lorma",
                         "--context", "bidirectional", "--fold", "all", "--epochs", "10",
                         "--silver", silver, "--silver-epochs", "1"], "p3_E1silver.log", fatal=False)
    run(train + ["--exp", "E4", "--model", "dense_v1", "--technique", "ia3",
                 "--context", "bidirectional", "--fold", "all", "--epochs", "10"], "p3_E4.log", fatal=False)
    e1_dev, e2_dev = mean_dev("E1"), mean_dev("E2")
    if e1_dev is not None and e2_dev is not None and e2_dev >= e1_dev - 0.08:
        run(train + ["--exp", "E5", "--model", "hybrid_b", "--technique", "lorma",
                     "--context", "echo", "--fold", "all", "--epochs", "10"], "p3_E5.log", fatal=False)
    else:
        print(f"[queue] skip E5; E2 dev {e2_dev} vs E1 {e1_dev}", flush=True)
    run(["-m", "saul.rr.kd", "--gold-control"], "p3_E6gold.log", fatal=False)
    leader = best_single(["E1", "E2", "E3", "E4", "E1tapt"])
    if leader:
        spec = {
            "E1": ("dense_v1", "lorma", "bidirectional"),
            "E2": ("hybrid_b", "lora", "echo"),
            "E3": ("ced", "ia3", "encoder"),
            "E4": ("dense_v1", "ia3", "bidirectional"),
            "E1tapt": ("dense_v1", "lorma", "bidirectional"),
        }[leader]
        init = ["--init", os.path.join(P3, "adapt", "dense_v1")] if leader == "E1tapt" else []
        run(train + ["--exp", f"{leader}_seed782", "--model", spec[0], "--technique", spec[1],
                     "--context", spec[2], "--fold", "all", "--epochs", "10", "--seed", "782", *init],
            "p3_seed782.log", fatal=False)
    code = run(train + ["--exp", "E1w4096", "--model", "dense_v1", "--technique", "lorma",
                        "--context", "bidirectional", "--window", "4096", "--fold", "all", "--epochs", "10"],
               "p3_E1w4096.log", fatal=False)
    if code != 0:
        with open(os.path.join(P3, "E1w4096", "failed.json"), "w", encoding="utf-8") as f:
            json.dump({"reason": "nonzero exit, likely OOM or NaN", "code": code}, f)
    run(["scripts/p3_analyze.py"], "p3_analyze.log", fatal=False)
    print("[queue] lane finished. Build zips with: .venv/bin/python scripts/p3_batch_zips.py", flush=True)
    print("[queue] frozen probe (report only): .venv/bin/python -m saul.rr.train "
          "--exp E1probe --model dense_v1 --technique probe --context bidirectional --fold 0 --epochs 10",
          flush=True)


if __name__ == "__main__":
    main()
