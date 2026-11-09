"""Phase 5 lane. One GPU job at a time. --dry-run prints the plan and launches nothing.

The first GPU step waits until the GPU has no compute process. It never kills another job.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
if not os.path.isdir(ROOT):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable
LOGS = os.path.join(ROOT, "logs")
P5 = os.environ.get("SAUL_P5_DIR", "/data/cs781-p5")

_OPEN = []


def _log(name: str) -> str:
    os.makedirs(LOGS, exist_ok=True)
    return os.path.join(LOGS, name)


def gpu_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


def _ps() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""


def phase4_running() -> bool:
    out = _ps()
    return "scripts/p4_" in out or "saul.cjpe" in out


def phase4_released() -> bool:
    """Phase 4's last training step is C5, or the user has said that lane is finished."""
    if os.environ.get("P5_PHASE4_DONE") == "1":
        return True
    path = os.path.join(os.environ.get("SAUL_P4_DIR", "/data/cs781-p4"), "exp", "C5", "done.json")
    return os.path.exists(path)


def tfidf_running() -> bool:
    return "p5_tfidf.py" in _ps() or "scripts/p5_tfidf.py" in _ps()


def wait_for_gpu() -> None:
    if os.environ.get("P5_FORCE_GPU") == "1":
        print("[queue] P5_FORCE_GPU=1, not waiting", flush=True)
        return
    while True:
        reasons = []
        if gpu_busy():
            reasons.append("GPU busy")
        if phase4_running():
            reasons.append("Phase 4 process")
        if not phase4_released():
            reasons.append("Phase 4 C5 done.json missing (set P5_PHASE4_DONE=1 if the user says it is finished)")
        if not reasons:
            print("[queue] GPU free and Phase 4 training is finished", flush=True)
            return
        print(f"[queue] {', '.join(reasons)}; sleeping 5 min", flush=True)
        time.sleep(300)


def _env(cpu: bool) -> dict:
    env = os.environ.copy()
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        env["NUMEXPR_NUM_THREADS"] = "4"
    return env


def launch(args: list[str], log_name: str, cpu: bool = False) -> subprocess.Popen:
    log_path = _log(log_name)
    cmd = [PY, *args]
    if cpu:
        cmd = ["nice", "-n", "19", *cmd]
    print(f"[queue] {log_name}: {' '.join(cmd)}", flush=True)
    handle = open(log_path, "ab")
    _OPEN.append(handle)
    return subprocess.Popen(
        cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, env=_env(cpu),
    )


def run(args: list[str], log_name: str, fatal: bool = True, cpu: bool = False) -> int:
    proc = launch(args, log_name, cpu=cpu)
    code = proc.wait()
    print(f"[queue] {log_name} exit {code}", flush=True)
    if code != 0 and fatal:
        raise SystemExit(f"{log_name} failed with {code}")
    return code


def done(exp: str) -> bool:
    return os.path.exists(os.path.join(P5, "exp", exp, "done.json"))


def read_done(exp: str) -> dict:
    path = os.path.join(P5, "exp", exp, "done.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def disk_ok() -> bool:
    try:
        stats = os.statvfs("/data")
    except OSError:
        return True
    free_gb = stats.f_bavail * stats.f_frsize / (1024 ** 3)
    print(f"[queue] /data free {free_gb:.1f} GB", flush=True)
    return free_gb > 15


def train(exp: str, model: str, technique: str, **flags) -> list[str]:
    args = ["-m", "saul.lsi.train", "--exp", exp, "--model", model, "--technique", technique]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        else:
            args += [flag, str(value)]
    return args


def g1() -> dict:
    path = os.path.join(P5, "g1.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    bench = os.path.join(P5, "bench", "bench_dense_v1_2048_sdpa.json")
    if not os.path.exists(bench):
        return {"ok": False, "reason": "no bench"}
    blob = json.load(open(bench, encoding="utf-8"))
    ok = blob.get("tok_s", 0) >= 25000 and blob.get("peak_gb", 99) < 80
    decision = {"ok": ok, "tok_s": blob.get("tok_s"), "peak_gb": blob.get("peak_gb"),
                "hybrid_limit": 2048 if ok else 1024, "reduced_probe": not ok}
    os.makedirs(P5, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle)
    return decision


def started_at() -> float:
    path = os.path.join(P5, "queue_start.json")
    os.makedirs(P5, exist_ok=True)
    if os.path.exists(path):
        return float(json.load(open(path, encoding="utf-8"))["start"])
    now = time.time()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"start": now}, handle)
    return now


def slipped(hours: float) -> bool:
    return (time.time() - started_at()) / 3600 > hours


def probe_names(reduced: bool, late: bool) -> list[str]:
    if late:
        return ["P1", "P3", "P5"]
    if reduced:
        return ["P1", "P2", "P3", "P5"]
    return ["P1", "P2", "P3", "P4", "P5", "P6"]


PROBE_SPEC = {
    "P1": dict(head="laat", loss="bce", sampling="uniform"),
    "P2": dict(head="pooled", loss="bce", sampling="uniform"),
    "P3": dict(head="laat", loss="asl", sampling="uniform"),
    "P4": dict(head="laat", loss="focal", sampling="uniform"),
    "P5": dict(head="laat", loss="bce", sampling="sqrt"),
    "P6": dict(head="laat", loss="pair", sampling="uniform"),
}


def _scored(exp: str):
    from saul.lsi.data import dev_half_array
    from saul.lsi.decide import load_logits, quick_macro
    blob = load_logits(exp, "dev")
    if "labels" not in blob:
        return None
    half = dev_half_array(blob["ids"])
    scored = quick_macro(blob["logits"], blob["labels"], half)
    return blob, half, scored


def variant_beats(base: str, new: str) -> bool:
    """+0.7 on both halves and a paired bootstrap p < 0.1, from the saved dev logits."""
    from saul.lsi.decide import quick_macro
    from saul.lsi.score import adopt, paired_bootstrap
    left = _scored(base)
    right = _scored(new)
    if left is None or right is None:
        return False
    blob_b, half, base_scores = left
    blob_n, _, _ = right
    index = {case_id: row for row, case_id in enumerate(blob_n["ids"])}
    if any(case_id not in index for case_id in blob_b["ids"]):
        return False
    ordered = blob_n["logits"][[index[case_id] for case_id in blob_b["ids"]]]
    new_scores = quick_macro(ordered, blob_b["labels"], half)
    gains = (
        new_scores["half"]["A"] - base_scores["half"]["A"],
        new_scores["half"]["B"] - base_scores["half"]["B"],
    )
    p_value = paired_bootstrap(blob_b["labels"], new_scores["pred"], base_scores["pred"])
    return adopt(base_scores["macro_f1"], new_scores["macro_f1"], gains, p_value)


def adopt_tapt() -> bool:
    if not done("A1") or not done("A1t"):
        return False
    return variant_beats("A1", "A1t")


def chosen_recipe() -> dict:
    init = os.path.join(P5, "adapt", "dense_v1")
    recipe = {
        "init": init if adopt_tapt() else None,
        "head": "laat",
        "loss": "bce",
        "sampling": "uniform",
    }
    if not done("P1"):
        return recipe
    if "P2" in probe_names(False, False) and done("P2") and variant_beats("P1", "P2"):
        recipe["head"] = "pooled"
    best_loss, best_score = None, -1.0
    for name, loss in (("P3", "asl"), ("P4", "focal"), ("P6", "pair")):
        if done(name) and variant_beats("P1", name):
            scored = _scored(name)
            score = scored[2]["macro_f1"] if scored else -1.0
            if score > best_score:
                best_score, best_loss = score, loss
    if best_loss:
        recipe["loss"] = best_loss
    if done("P5") and variant_beats("P1", "P5"):
        recipe["sampling"] = "sqrt"
    return recipe


def a5_close_to_dense() -> bool:
    a5 = read_done("A5").get("dev_f1")
    dense = [read_done(name).get("dev_f1") for name in ("A1", "A1t")]
    dense = [float(value) for value in dense if value is not None]
    if a5 is None or not dense:
        return False
    return float(a5) >= max(dense) - 0.02


def steps(reduced: bool, hybrid_limit: int, init: str | None, hours: float) -> list[tuple]:
    """(argv, log, fatal, skip_if_done exp or None)."""
    common = dict(attn="sdpa", head="laat", loss="bce", epochs=5)
    jobs = []
    jobs.append((["-m", "saul.lsi.data", "--prepare", "--workers", "4"], "p5_prepare.log", True, None))
    jobs.append((["scripts/p5_tfidf.py"], "p5_tfidf.log", False, "B1"))
    for model, technique, context, limit in (
        ("dense_v1", "lorma", "bidirectional", 2048),
        ("hybrid_b", "adapter", "echo", hybrid_limit),
        ("ced", "partial", "encoder", 2048),
        ("conv_mini", "full", "echo", 2048),
    ):
        jobs.append((train(
            f"smoke_{model}", model, technique, context=context, limit=256, smoke=True, ckpt_every=0,
        ), f"p5_smoke_{model}.log", False, f"smoke_{model}"))
        jobs.append((train(
            f"bench_{model}", model, technique, context=context, limit=limit, bench=True,
        ), f"p5_bench_{model}.log", model == "dense_v1", None))
    a1_flags = dict(common, context="bidirectional", gate=0.42)
    jobs.append((train("A1", "dense_v1", "lorma", **a1_flags), "p5_A1.log", True, "A1"))
    jobs.append((["-m", "saul.lsi.tapt"], "p5_tapt.log", False, None))
    jobs.append((train("A1t", "dense_v1", "lorma", init=os.path.join(P5, "adapt", "dense_v1"), **common,
                       context="bidirectional"), "p5_A1t.log", False, "A1t"))
    late = hours >= 8
    for name in probe_names(reduced, late):
        jobs.append((train(name, "dense_v1", "lorma", epochs=2, subset=15000, context="bidirectional",
                           init=init, **PROBE_SPEC[name]), f"p5_{name}.log", False, name))
    best_init = init or ""
    for fold in ("A", "B"):
        flags = dict(common, context="bidirectional", cf2_fold=fold)
        if best_init:
            flags["init"] = best_init
        jobs.append((train(f"A4{fold}", "dense_v1", "lorma", **flags), f"p5_A4{fold}.log", False, f"A4{fold}"))
    knn_exp = "A1t" if init else "A1"
    jobs.append((["-m", "saul.lsi.knn", "--exp", knn_exp, "--model", "dense_v1", "--technique", "lorma"],
                 "p5_knn.log", False, None))
    jobs.append((train("A5", "hybrid_b", "adapter", context="echo", limit=hybrid_limit, **common),
                 "p5_A5.log", False, "A5"))
    jobs.append((train("A6", "ced", "partial", context="encoder", **common), "p5_A6.log", False, "A6"))
    jobs.append((["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A4A,A4B", "--tag", "A4"],
                 "p5_cf2.log", False, None))
    jobs.append((["-m", "saul.lsi.ensemble"], "p5_mix.log", False, None))
    jobs.append((["-m", "saul.lsi.kd", "--from-mix", "mix"], "p5_teacher.log", False, None))
    jobs.append((train("A9", "conv_mini", "full", context="echo", loss="kd", epochs=4,
                       teacher=os.path.join(P5, "kd", "teacher.npz")), "p5_A9.log", False, "A9"))
    if hours < 30:
        jobs.append((train("A7", "dense_v1", "full", context="bidirectional", epochs=3, base_lr=2e-5,
                           head="laat", loss="bce", attn="sdpa", init=init), "p5_A7.log", False, "A7"))
    jobs.append((["-m", "saul.lsi.ensemble"], "p5_mix_final.log", False, None))
    jobs.append((["-m", "saul.lsi.decide", "--exp", "mix", "--knn-exp", knn_exp], "p5_decide.log", False, None))
    jobs.append((["scripts/p5_zips.py"], "p5_zips.log", False, None))
    # Reserve. The live queue skips these after 24 Oct 12:00 IST (06:30 UTC).
    for fold in ("A", "B"):
        jobs.append((train(f"A5cf{fold}", "hybrid_b", "adapter", context="echo", limit=hybrid_limit,
                           cf2_fold=fold, epochs=5, head="laat", loss="bce", attn="sdpa"),
                     f"p5_A5cf{fold}.log", False, f"A5cf{fold}"))
        jobs.append((train(f"A4seed{fold}", "dense_v1", "lorma", context="bidirectional", seed=782,
                           cf2_fold=fold, **common), f"p5_A4seed{fold}.log", False, f"A4seed{fold}"))
    jobs.append((["-m", "saul.lsi.tapt", "--model", "hybrid_b", "--context", "echo"],
                 "p5_tapt_hybrid.log", False, None))
    jobs.append((["-m", "saul.lsi.sections", "--exp", "A1", "--model", "dense_v1", "--technique", "lorma"],
                 "p5_sections.log", False, None))
    jobs.append((train("A1probe", "dense_v1", "probe", context="bidirectional", epochs=2, head="laat",
                       loss="bce", attn="sdpa"), "p5_probe.log", False, "A1probe"))
    jobs.append((train("A9gold", "conv_mini", "full", context="echo", loss="bce", epochs=2, head="laat",
                       attn="sdpa"), "p5_A9gold.log", False, "A9gold"))
    jobs.append((train("A13", "dense_v1", "adalora", context="bidirectional", epochs=2, head="laat",
                       loss="bce", attn="sdpa"), "p5_adalora.log", False, "A13"))
    jobs.append((["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A5cfA,A5cfB", "--tag", "A5cf"],
                 "p5_cf2_hybrid.log", False, None))
    jobs.append((["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A4seedA,A4seedB", "--tag", "A4seed"],
                 "p5_cf2_seed.log", False, None))
    jobs.append((["-m", "saul.lsi.ensemble", "--out", "mix_v2"], "p5_mix_v2.log", False, None))
    jobs.append((["scripts/p5_zips.py"], "p5_zips_reserve.log", False, None))
    jobs.append((["scripts/p5_analyze.py"], "p5_analyze.log", False, None))
    return jobs


def reserve_closed() -> bool:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = datetime.datetime(2026, 10, 24, 6, 30, tzinfo=datetime.timezone.utc)
    return now >= cutoff


def main() -> None:
    dry = "--dry-run" in sys.argv
    decision = {"ok": True, "hybrid_limit": 2048, "reduced_probe": False}
    if not dry and os.path.exists(os.path.join(P5, "g1.json")):
        decision = g1()
    init = os.path.join(P5, "adapt", "dense_v1") if os.path.exists(os.path.join(P5, "adapt", "dense_v1", "done.json")) else None
    jobs = steps(decision.get("reduced_probe", False), decision.get("hybrid_limit", 2048), init, hours=0)
    if dry:
        for args, log_name, fatal, skip in jobs:
            mark = "skip-done" if skip and done(skip) else "run"
            print(f"{mark}\tfatal={fatal}\t{log_name}\t{' '.join(args)}")
        print(f"[dry-run] {len(jobs)} steps, nothing launched")
        return
    os.makedirs(P5, exist_ok=True)
    if not os.path.exists(os.path.join(P5, "g0.json")):
        run(["-m", "saul.lsi.data", "--prepare", "--workers", "4"], "p5_prepare.log", cpu=True)
    if not done("B1"):
        if tfidf_running():
            print("[queue] p5_tfidf.py is already running; waiting for B1/done.json", flush=True)
            while not done("B1"):
                time.sleep(60)
        else:
            run(["scripts/p5_tfidf.py"], "p5_tfidf.log", fatal=False, cpu=True)
    wait_for_gpu()
    started_at()
    for model, technique, context, limit in (
        ("dense_v1", "lorma", "bidirectional", 2048),
        ("hybrid_b", "adapter", "echo", 2048),
        ("ced", "partial", "encoder", 2048),
        ("conv_mini", "full", "echo", 2048),
    ):
        if not done(f"smoke_{model}"):
            run(train(f"smoke_{model}", model, technique, context=context, smoke=True, ckpt_every=0),
                f"p5_smoke_{model}.log", fatal=False)
        if model == "dense_v1" or not os.path.exists(os.path.join(P5, "bench", f"bench_{model}_{limit}_sdpa.json")):
            if disk_ok():
                run(train(f"bench_{model}", model, technique, context=context, limit=limit, bench=True),
                    f"p5_bench_{model}.log", fatal=model == "dense_v1")
    decision = g1()
    hybrid_limit = decision.get("hybrid_limit", 2048)
    if not done("A1"):
        run(train("A1", "dense_v1", "lorma", context="bidirectional", gate=0.42, epochs=5, head="laat",
                  loss="bce", attn="sdpa"), "p5_A1.log", fatal=True)
    if not os.path.exists(os.path.join(P5, "adapt", "dense_v1", "done.json")):
        run(["-m", "saul.lsi.tapt"], "p5_tapt.log", fatal=False)
    if os.path.exists(os.path.join(P5, "adapt", "dense_v1", "done.json")) and not done("A1t"):
        run(train("A1t", "dense_v1", "lorma", context="bidirectional", epochs=5, head="laat", loss="bce",
                  attn="sdpa", init=os.path.join(P5, "adapt", "dense_v1")), "p5_A1t.log", fatal=False)
    tapt_init = os.path.join(P5, "adapt", "dense_v1") if adopt_tapt() else None
    late = slipped(8)
    for name in probe_names(bool(decision.get("reduced_probe")), late):
        if done(name):
            continue
        flags = dict(PROBE_SPEC[name])
        run(train(name, "dense_v1", "lorma", epochs=2, subset=15000, context="bidirectional", attn="sdpa",
                  init=tapt_init, **flags), f"p5_{name}.log", fatal=False)
    chosen = chosen_recipe()
    with open(os.path.join(P5, "chosen.json"), "w", encoding="utf-8") as handle:
        json.dump(chosen, handle)
    print(f"[queue] chosen recipe {json.dumps(chosen)}", flush=True)
    init_path = chosen.get("init")
    for fold in ("A", "B"):
        if not done(f"A4{fold}"):
            run(train(f"A4{fold}", "dense_v1", "lorma", context="bidirectional", epochs=5, head=chosen["head"],
                      loss=chosen["loss"], sampling=chosen["sampling"], attn="sdpa", cf2_fold=fold,
                      init=init_path), f"p5_A4{fold}.log", fatal=False)
    knn_exp = "A1t" if tapt_init and os.path.exists(os.path.join(P5, "exp", "A1t", "best.pt")) else "A1"
    knn_dir = os.path.join(P5, "exp", knn_exp)
    if os.path.exists(os.path.join(knn_dir, "best.pt")) and not os.path.exists(os.path.join(knn_dir, "dev_knn.npz")):
        run(["-m", "saul.lsi.knn", "--exp", knn_exp, "--model", "dense_v1", "--technique", "lorma"],
            "p5_knn.log", fatal=False)
    if not done("A5"):
        run(train("A5", "hybrid_b", "adapter", context="echo", limit=hybrid_limit, epochs=5, head="laat",
                  loss=chosen["loss"], sampling=chosen["sampling"], attn="sdpa"), "p5_A5.log", fatal=False)
    if not done("A6"):
        run(train("A6", "ced", "partial", context="encoder", epochs=5, head="laat", loss="bce",
                  sampling="uniform", attn="sdpa"), "p5_A6.log", fatal=False)
    run(["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A4A,A4B", "--tag", "A4"],
        "p5_cf2.log", fatal=False, cpu=True)
    run(["-m", "saul.lsi.ensemble"], "p5_mix.log", fatal=False, cpu=True)
    if os.path.exists(os.path.join(P5, "exp", "mix", "mix.json")) and not done("A9"):
        run(["-m", "saul.lsi.kd", "--from-mix", "mix"], "p5_teacher.log", fatal=False, cpu=True)
        if os.path.exists(os.path.join(P5, "kd", "teacher.npz")):
            run(train("A9", "conv_mini", "full", context="echo", loss="kd", epochs=4, head="laat", attn="sdpa",
                      teacher=os.path.join(P5, "kd", "teacher.npz")), "p5_A9.log", fatal=False)
    if not reserve_closed() and not slipped(30) and not done("A7"):
        run(train("A7", "dense_v1", "full", context="bidirectional", epochs=3, head="laat", loss="bce",
                  attn="sdpa", init=init_path), "p5_A7.log", fatal=False)
    run(["-m", "saul.lsi.ensemble"], "p5_mix_final.log", fatal=False, cpu=True)
    if os.path.exists(os.path.join(P5, "exp", "mix", "dev_logits.npz")):
        knn_flag = ["--knn-exp", knn_exp] if os.path.exists(os.path.join(knn_dir, "dev_knn.npz")) else []
        run(["-m", "saul.lsi.decide", "--exp", "mix", *knn_flag], "p5_decide.log", fatal=False, cpu=True)
    run(["scripts/p5_zips.py"], "p5_zips.log", fatal=False, cpu=True)
    if not reserve_closed():
        if not done("A7"):
            run(train("A7", "dense_v1", "full", context="bidirectional", epochs=3, head="laat", loss="bce",
                      attn="sdpa", init=init_path), "p5_A7.log", fatal=False)
        if a5_close_to_dense():
            for fold in ("A", "B"):
                if not done(f"A5cf{fold}"):
                    run(train(f"A5cf{fold}", "hybrid_b", "adapter", context="echo", limit=hybrid_limit,
                              cf2_fold=fold, epochs=5, head="laat", loss=chosen["loss"],
                              sampling=chosen["sampling"], attn="sdpa"), f"p5_A5cf{fold}.log", fatal=False)
        for fold in ("A", "B"):
            if not done(f"A4seed{fold}"):
                run(train(f"A4seed{fold}", "dense_v1", "lorma", context="bidirectional", seed=782, cf2_fold=fold,
                          epochs=5, head=chosen["head"], loss=chosen["loss"], sampling=chosen["sampling"],
                          attn="sdpa", init=init_path), f"p5_A4seed{fold}.log", fatal=False)
        if not os.path.exists(os.path.join(P5, "adapt", "hybrid_b", "done.json")):
            run(["-m", "saul.lsi.tapt", "--model", "hybrid_b", "--context", "echo"],
                "p5_tapt_hybrid.log", fatal=False)
        if not done("A1probe"):
            run(train("A1probe", "dense_v1", "probe", context="bidirectional", epochs=2, head="laat", loss="bce",
                      attn="sdpa"), "p5_probe.log", fatal=False)
        run(["-m", "saul.lsi.sections", "--exp", knn_exp, "--model", "dense_v1", "--technique", "lorma"],
            "p5_sections.log", fatal=False)
        if not done("A9gold"):
            run(train("A9gold", "conv_mini", "full", context="echo", loss="bce", epochs=2, head="laat", attn="sdpa"),
                "p5_A9gold.log", fatal=False)
        if not done("A13"):
            run(train("A13", "dense_v1", "adalora", context="bidirectional", epochs=2, head="laat", loss="bce",
                      attn="sdpa"), "p5_adalora.log", fatal=False)
        run(["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A5cfA,A5cfB", "--tag", "A5cf"],
            "p5_cf2_hybrid.log", fatal=False, cpu=True)
        run(["-m", "saul.lsi.ensemble", "--assemble-cf2", "--folds", "A4seedA,A4seedB", "--tag", "A4seed"],
            "p5_cf2_seed.log", fatal=False, cpu=True)
        run(["-m", "saul.lsi.ensemble", "--out", "mix_v2"], "p5_mix_v2.log", fatal=False, cpu=True)
        run(["scripts/p5_zips.py"], "p5_zips_reserve.log", fatal=False, cpu=True)
    run(["scripts/p5_analyze.py"], "p5_analyze.log", fatal=False, cpu=True)
    print("[queue] lane finished", flush=True)


if __name__ == "__main__":
    main()
