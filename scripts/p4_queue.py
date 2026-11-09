"""Phase 4 lane. A step with done.json is skipped, so a restart continues the lane.

Logs are absolute paths under ~/cs781/logs. Children get their own session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.const import ADOPT_GAIN, G1_FAST, G1_SLOW
from saul.cjpe.score import adopt, half_gains, macro_scores, paired_bootstrap, predict_logits

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOGS = os.path.join(ROOT, "logs")
_OPEN = []


def _log(name: str) -> str:
    os.makedirs(LOGS, exist_ok=True)
    return os.path.join(LOGS, name)


def _env(cpu: bool) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def launch(args: list[str], log_name: str, cpu: bool = False) -> subprocess.Popen:
    print(f"[queue] {log_name}: {' '.join(args)}", flush=True)
    handle = open(_log(log_name), "ab")
    _OPEN.append(handle)
    return subprocess.Popen(
        [PY, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True, env=_env(cpu),
    )


def run(args: list[str], log_name: str, fatal: bool = True, cpu: bool = False) -> int:
    code = launch(args, log_name, cpu).wait()
    print(f"[queue] {log_name} exit {code}", flush=True)
    if code != 0 and fatal:
        raise SystemExit(f"{log_name} failed with {code}; see {_log(log_name)}")
    return code


def done(name: str) -> bool:
    return os.path.exists(os.path.join(exp_dir(name), "done.json"))


def read_done(name: str) -> dict:
    path = os.path.join(exp_dir(name), "done.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def started_at() -> float:
    path = os.path.join(p4_dir(), "queue_start.json")
    os.makedirs(p4_dir(), exist_ok=True)
    if os.path.exists(path):
        return float(json.load(open(path, encoding="utf-8"))["start"])
    now = time.time()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"start": now}, handle)
    return now


def slipped(hours: float) -> bool:
    return (time.time() - started_at()) / 3600 > hours


def _preds(exp: str):
    import numpy as np
    from saul.cjpe.calibrate import apply_prior
    blob = np.load(os.path.join(exp_dir(exp), "logits_dsel.npz"), allow_pickle=True)
    logits = apply_prior(np.asarray(blob["logits"], dtype=np.float64), read_done(exp).get("loss") == "ce")
    labels = np.asarray(blob["labels"])
    from saul.cjpe.data import load_split
    halves = np.asarray(load_split("dsel").halves)
    return labels, predict_logits(logits), halves, macro_scores(labels, predict_logits(logits))["f1"]


def beats(base: str, new: str) -> bool:
    if not (done(base) and done(new)):
        return False
    y, pred_b, halves, f1_b = _preds(base)
    _y, pred_n, _h, f1_n = _preds(new)
    gains = half_gains(y, pred_n, pred_b, halves)
    p_value = paired_bootstrap(y, pred_n, pred_b)
    kept = adopt(f1_b, f1_n, gains, p_value, ADOPT_GAIN)
    print(f"[queue] {new} vs {base}: gain {f1_n - f1_b:.4f} halves {gains} p {p_value:.3f} keep {kept}", flush=True)
    return kept


def write_chosen(payload: dict) -> None:
    path = os.path.join(p4_dir(), "chosen.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[queue] chosen {payload}", flush=True)


def read_chosen() -> dict:
    path = os.path.join(p4_dir(), "chosen.json")
    if not os.path.exists(path):
        return {"view": "tail", "limit": 2048, "head": 512, "tail": 1536, "ovl_weight": 2, "init": None, "attn": "sdpa"}
    return json.load(open(path, encoding="utf-8"))


def a1_recipe(attn: str) -> dict:
    """The flagship run: dense LoRMA, bidirectional, tail 2048, D-ovl weight 2."""
    return {
        "view": "tail", "limit": 2048, "head": 512, "tail": 1536,
        "ovl_weight": 2, "init": None, "attn": attn,
    }


def best_epochs(exp: str, default: int = 2) -> int:
    """Epoch count at which `exp` recorded its best D-sel score. Epochs in the log are 0-based."""
    path = os.path.join(exp_dir(exp), "history.json")
    if not os.path.exists(path):
        return default
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    if not rows:
        return default
    deploy = int(read_done(exp).get("limit") or 0)
    # A 1,024-token first epoch is a G1 sanity check. Trim to the best epoch at the deployment length.
    usable = [row for row in rows if not deploy or int(row.get("limit", deploy)) == deploy]
    best = max(usable or rows, key=lambda row: float(row.get("dev_f1", -1.0)))
    return max(1, int(best.get("epoch", 0)) + 1)


def dense_args(exp: str, chosen: dict, epochs: int, extra: list[str] | None = None) -> list[str]:
    # 16,384 tokens is 8 cases at bucket 2048. Accumulating 4 steps is effective batch 32.
    args = [
        "-m", "saul.cjpe.train", "--exp", exp, "--model", "dense_v1", "--technique", "lorma",
        "--context", "bidirectional", "--view", chosen.get("view", "tail"),
        "--limit", str(chosen.get("limit", 2048)), "--head", str(chosen.get("head", 512)),
        "--tail", str(chosen.get("tail", 1536)), "--epochs", str(epochs),
        "--ovl-weight", str(chosen.get("ovl_weight", 2)),
        "--batch-tokens", "16384", "--accum", "4", "--compile",
        "--attn", chosen.get("attn", "sdpa"),
    ]
    if chosen.get("init"):
        args += ["--init", chosen["init"]]
    if extra:
        args += extra
    return args


def encode_args(exp: str, model: str, technique: str, context: str, attn: str, init: str | None = None) -> list[str]:
    cmd = [
        "-m", "saul.cjpe.hier", "--encode", "--exp", exp, "--model", model,
        "--technique", technique, "--context", context, "--attn", attn,
    ]
    if init:
        cmd += ["--init", init]
    return cmd


def g1_decision(rows: list[dict]) -> dict:
    """G1 is the dense LoRMA SDPA measurement at 2,048. Flex is only a fallback if SDPA did not run."""
    def usable(row: dict, attn: str | None = None) -> bool:
        good = (
            row.get("model") == "dense_v1" and row.get("width") == 2048
            and row.get("ok") and float(row.get("peak_gb", 99)) < 80
        )
        if attn is not None:
            good = good and row.get("attn") == attn
        return good

    sdpa = [row for row in rows if usable(row, "sdpa")]
    pool = sdpa or [row for row in rows if usable(row)]
    if not pool:
        return {"decision": "stop", "tok_s": 0, "first_limit": 0, "attn": "sdpa"}
    best = max(pool, key=lambda row: row["tok_s"])
    tok = float(best["tok_s"])
    if tok >= G1_FAST:
        decision, first = "ok", 0
    elif tok >= G1_SLOW:
        decision, first = "tail1024", 1024
    else:
        decision, first = "stop", 0
    return {"decision": decision, "tok_s": tok, "first_limit": first, "attn": best.get("attn") or "sdpa", "peak_gb": best.get("peak_gb")}


def main() -> None:
    os.makedirs(p4_dir(), exist_ok=True)
    started_at()
    g0_path = os.path.join(p4_dir(), "g0.json")
    if not os.path.exists(g0_path):
        run(["-m", "saul.cjpe.data", "--prepare"], "p4_data.log", fatal=True, cpu=True)
    if os.path.exists(g0_path):
        g0 = json.load(open(g0_path, encoding="utf-8"))
        if not g0.get("pass"):
            raise SystemExit(f"G0 failed: tokens/word gap {g0.get('gap')}")
    if not done("B0"):
        run(["scripts/p4_baseline.py"], "p4_b0.log", fatal=False, cpu=True)

    bench_dir = os.path.join(p4_dir(), "bench")
    os.makedirs(bench_dir, exist_ok=True)
    benches = [
        ("smoke_dense", ["--exp", "smoke_dense", "--model", "dense_v1", "--technique", "lorma", "--context", "bidirectional", "--smoke"]),
        ("smoke_hybrid", ["--exp", "smoke_hybrid", "--model", "hybrid_b", "--technique", "adapter_bitfit", "--context", "causal", "--smoke"]),
        ("smoke_ced", ["--exp", "smoke_ced", "--model", "ced", "--technique", "partial", "--context", "encoder", "--smoke"]),
        ("smoke_conv", ["--exp", "smoke_conv", "--model", "conv_mini", "--technique", "full", "--context", "causal", "--smoke"]),
    ]
    for exp, args in benches:
        if not done(exp):
            run(["-m", "saul.cjpe.train", *args], f"p4_{exp}.log", fatal=False)
    specs = [
        ("dense_v1", "lorma", "bidirectional", 2048, "sdpa"),
        ("dense_v1", "lorma", "bidirectional", 2048, "flex"),
        ("hybrid_b", "adapter_bitfit", "causal", 8192, "sdpa"),
        ("ced", "partial", "encoder", 2048, "sdpa"),
        ("conv_mini", "full", "causal", 4096, "sdpa"),
    ]
    rows = []
    for model, technique, context, width, attn in specs:
        path = os.path.join(bench_dir, f"bench_{model}_{width}_{attn}.json")
        if not os.path.exists(path):
            run(["-m", "saul.cjpe.train", "--exp", f"bench_{model}", "--model", model, "--technique", technique,
                 "--context", context, "--limit", str(width), "--attn", attn, "--bench", "--compile"],
                f"p4_bench_{model}_{width}_{attn}.log", fatal=False)
        if os.path.exists(path):
            rows.append(json.load(open(path, encoding="utf-8")))
    decision = g1_decision(rows)
    with open(os.path.join(p4_dir(), "g1.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    print(f"[queue] G1 {decision}", flush=True)
    if decision["decision"] == "stop":
        raise SystemExit("G1 below 15K tokens/s")

    chosen = read_chosen()
    attn = decision.get("attn") or "sdpa"
    chosen["attn"] = attn
    recipe = a1_recipe(attn)
    if not done("A1"):
        extra = ["--gate", "0.74"]
        if decision["first_limit"]:
            extra += ["--first-limit", str(decision["first_limit"])]
        run(dense_args("A1", recipe, 2, extra), "p4_A1.log", fatal=True)
    if not done("A3"):
        if not os.path.exists(os.path.join(exp_dir("A1"), "chunks", "done.json")):
            run(encode_args("A1", "dense_v1", "lorma", "bidirectional", attn), "p4_A3_encode.log", fatal=False)
        if os.path.exists(os.path.join(exp_dir("A1"), "chunks", "done.json")):
            run(["-m", "saul.cjpe.hier", "--train", "--vectors", "A1", "--name", "A3", "--reference", "A1"],
                "p4_A3.log", fatal=False)

    probes = [
        ("A2_tail512", ["--view", "tail", "--limit", "512", "--ovl-weight", "2"]),
        ("A2_tail2048", ["--view", "tail", "--limit", "2048", "--ovl-weight", "2"]),
        ("A2_tail4096", ["--view", "tail", "--limit", "4096", "--ovl-weight", "2"]),
        ("A2_tail8192", ["--view", "tail", "--limit", "8192", "--ovl-weight", "2"]),
        ("A2_ht", ["--view", "head_tail", "--head", "512", "--tail", "1536", "--ovl-weight", "2"]),
        ("A2_head", ["--view", "head", "--limit", "2048", "--ovl-weight", "2"]),
        ("A2_ovl1", ["--view", "tail", "--limit", "2048", "--ovl-weight", "1"]),
        ("A2_ovl4", ["--view", "tail", "--limit", "2048", "--ovl-weight", "4"]),
    ]
    if slipped(8):
        probes = [item for item in probes if item[0] in {"A2_tail512", "A2_tail4096", "A2_ht"}]
    for exp, extra in probes:
        if done(exp):
            continue
        run(dense_args(exp, recipe, 1, ["--subset", "12000", "--no-pool-logits", *extra]), f"p4_{exp}.log", fatal=False)
    view, limit, head, tail = "tail", 2048, 512, 1536
    kept_views = []
    # Only a longer tail, or head+tail, can replace tail-2048. tail-512 and head-only are probes.
    for exp, spec in (
        ("A2_tail4096", ("tail", 4096, 512, 1536)),
        ("A2_tail8192", ("tail", 8192, 512, 1536)),
        ("A2_ht", ("head_tail", 2048, 512, 1536)),
    ):
        if beats("A2_tail2048", exp):
            kept_views.append((read_done(exp).get("dev_f1", 0.0), spec))
    if kept_views:
        view, limit, head, tail = max(kept_views)[1]
    ovl = 2
    best_ovl = read_done("A2_tail2048").get("dev_f1", 0.0)
    for exp, weight in (("A2_ovl1", 1), ("A2_ovl4", 4)):
        score = read_done(exp).get("dev_f1", 0.0)
        if beats("A2_tail2048", exp) and score > best_ovl:
            ovl, best_ovl = weight, score
    chosen.update({"view": view, "limit": limit, "head": head, "tail": tail, "ovl_weight": ovl})
    write_chosen(chosen)

    adapt = os.path.join(p4_dir(), "adapt", "dense_v1")
    if not os.path.exists(os.path.join(adapt, "done.json")):
        run(["-m", "saul.cjpe.tapt"], "p4_tapt.log", fatal=False)
    if os.path.exists(os.path.join(adapt, "done.json")) and not done("A4"):
        run(dense_args("A4", {**recipe, "init": adapt}, best_epochs("A1")), "p4_A4.log", fatal=False)
    if beats("A1", "A4"):
        chosen["init"] = adapt
        write_chosen(chosen)
        if not done("A3tapt"):
            run(encode_args("A4", "dense_v1", "lorma", "bidirectional", attn, adapt), "p4_A3tapt_encode.log", fatal=False)
            if os.path.exists(os.path.join(exp_dir("A4"), "chunks", "done.json")):
                run(["-m", "saul.cjpe.hier", "--train", "--vectors", "A4", "--name", "A3tapt", "--reference", "A4",
                     "--ovl-weight", str(chosen.get("ovl_weight", 2))],
                    "p4_A3tapt.log", fatal=False)

    a5_view = chosen.get("view", "tail")
    a5_limit = int(chosen.get("limit", 2048))
    # A5's own default is causal tail 8192. Use another view only when A2 adopted one.
    if a5_view == "tail" and a5_limit == 2048:
        a5_view, a5_limit = "tail", 8192
    if not done("A5"):
        code = run(["-m", "saul.cjpe.train", "--exp", "A5", "--model", "hybrid_b", "--technique", "adapter_bitfit",
                    "--context", "causal", "--view", a5_view, "--limit", str(a5_limit),
                    "--head", str(chosen.get("head", 512)), "--tail", str(chosen.get("tail", 1536)),
                    "--epochs", "2", "--ovl-weight", str(chosen["ovl_weight"]),
                    "--batch-tokens", "16384", "--attn", "sdpa"],
                   "p4_A5.log", fatal=False)
        if code in (4, 137, -9) and not done("A5") and a5_limit != 4096:
            run(["-m", "saul.cjpe.train", "--exp", "A5", "--model", "hybrid_b", "--technique", "adapter_bitfit",
                 "--context", "causal", "--view", "tail", "--limit", "4096", "--epochs", "2",
                 "--ovl-weight", str(chosen["ovl_weight"]), "--batch-tokens", "16384", "--attn", "sdpa", "--force"],
                "p4_A5_4096.log", fatal=False)
    if done("A5") and not done("A3hyb"):
        if not os.path.exists(os.path.join(exp_dir("A5"), "chunks", "done.json")):
            run(encode_args("A5", "hybrid_b", "adapter_bitfit", "causal", "sdpa"), "p4_A5_chunks.log", fatal=False)
        if os.path.exists(os.path.join(exp_dir("A5"), "chunks", "done.json")):
            run(["-m", "saul.cjpe.hier", "--train", "--vectors", "A5", "--name", "A3hyb", "--reference", "A5",
                 "--ovl-weight", str(chosen["ovl_weight"])], "p4_A3hyb.log", fatal=False)
    if not done("A6"):
        run(["-m", "saul.cjpe.train", "--exp", "A6", "--model", "ced", "--technique", "partial",
             "--context", "encoder", "--view", "tail", "--limit", "2048", "--epochs", "2",
             "--ovl-weight", str(chosen["ovl_weight"]), "--attn", chosen["attn"]],
            "p4_A6.log", fatal=False)

    members = [exp for exp in ("A1", "A4", "A3", "A3tapt", "A3hyb", "A5", "A6") if done(exp) and os.path.exists(os.path.join(exp_dir(exp), "logits_dsel.npz"))]
    if members and not os.path.exists(os.path.join(p4_dir(), "mix", "all", "calib.json")):
        run(["-m", "saul.cjpe.ensemble", "--exps", ",".join(members), "--name", "all"], "p4_mix_all.log", fatal=False)
    dense_members = [exp for exp in ("A1", "A4", "A3", "A3tapt") if exp in members]
    if dense_members and not os.path.exists(os.path.join(p4_dir(), "mix", "dense", "calib.json")):
        run(["-m", "saul.cjpe.ensemble", "--exps", ",".join(dense_members), "--name", "dense"], "p4_mix_dense.log", fatal=False)
    run(["scripts/p4_zips.py"], "p4_zips_day1.log", fatal=False, cpu=True)

    calib_path = os.path.join(p4_dir(), "mix", "all", "calib.json")
    if os.path.exists(calib_path):
        teacher_members = list(json.load(open(calib_path, encoding="utf-8")).get("members", []))
    else:
        teacher_members = list(members)
    if chosen.get("init") and "A4" not in teacher_members and done("A4"):
        teacher_members.append("A4")
    teacher_members = [exp for exp in teacher_members if os.path.exists(os.path.join(exp_dir(exp), "logits_pool.npz"))]
    teacher_path = os.path.join(p4_dir(), "kd", "teacher.npz")
    if not done("A9"):
        if teacher_members and not os.path.exists(teacher_path):
            run(["-m", "saul.cjpe.kd", "--teacher", "--members", ",".join(teacher_members)], "p4_teacher.log", fatal=False)
        if os.path.exists(teacher_path):
            run(["-m", "saul.cjpe.kd", "--student"], "p4_A9.log", fatal=False)
        else:
            print("[queue] A9 not started: no teacher probabilities", flush=True)
    if not done("A7"):
        run(["-m", "saul.cjpe.train", "--exp", "A7", "--model", "dense_v1", "--technique", "full",
             "--context", "bidirectional", "--view", "tail", "--limit", "2048", "--epochs", "1",
             "--ovl-weight", "2", "--batch-tokens", "16384", "--accum", "4", "--attn", attn],
            "p4_A7.log", fatal=False)

    dense_epochs = best_epochs("A1") if chosen.get("init") else 2
    if not slipped(30):
        if not done("A11"):
            run(dense_args("A11", chosen, dense_epochs, ["--seed", "782"]), "p4_A11.log", fatal=False)
        if not done("A12"):
            run(dense_args("A12", chosen, dense_epochs, ["--refit"]), "p4_A12.log", fatal=False)
    else:
        print("[queue] skipping A11 and A12; the lane is past 30 hours", flush=True)
    if done("A11") and not done("A3s2"):
        if not os.path.exists(os.path.join(exp_dir("A11"), "chunks", "done.json")):
            run(encode_args("A11", "dense_v1", "lorma", "bidirectional", attn, chosen.get("init")),
                "p4_A11_chunks.log", fatal=False)
        if os.path.exists(os.path.join(exp_dir("A11"), "chunks", "done.json")):
            run(["-m", "saul.cjpe.hier", "--train", "--vectors", "A11", "--name", "A3s2", "--reference", "A11",
                 "--ovl-weight", str(chosen.get("ovl_weight", 2))],
                "p4_A3s2.log", fatal=False)

    later = [exp for exp in ("A1", "A4", "A3", "A3tapt", "A3hyb", "A3s2", "A5", "A6", "A7", "A9", "A11", "A12") if done(exp)]
    if later:
        run(["-m", "saul.cjpe.ensemble", "--exps", ",".join(later), "--name", "all_v2"], "p4_mix_v2.log", fatal=False)
    if done("A12"):
        run(["-m", "saul.cjpe.ensemble", "--exps", ",".join(later), "--name", "refit"], "p4_mix_refit.log", fatal=False)

    if not slipped(36):
        if not done("A1_noovl"):
            run(dense_args("A1_noovl", recipe, 2, ["--no-ovl"]), "p4_A1_noovl.log", fatal=False)
        if not done("A1_ce"):
            run(dense_args("A1_ce", recipe, 1, ["--subset", "12000", "--loss", "ce", "--no-pool-logits"]),
                "p4_A1_ce.log", fatal=False)
        if not done("A1probe"):
            run(["-m", "saul.cjpe.train", "--exp", "A1probe", "--model", "dense_v1", "--technique", "probe",
                 "--context", "bidirectional", "--epochs", "1", "--limit", "2048", "--no-pool-logits"],
                "p4_A1probe.log", fatal=False)
        if read_done("A3").get("holdout_needed") and not done("A3hold"):
            if not done("A1_holdout"):
                run(dense_args("A1_holdout", recipe, 2, ["--holdout"]), "p4_A1_holdout.log", fatal=False)
            if done("A1_holdout") and not os.path.exists(os.path.join(exp_dir("A1_holdout"), "chunks", "done.json")):
                run(encode_args("A1_holdout", "dense_v1", "lorma", "bidirectional", attn), "p4_holdout_encode.log", fatal=False)
            fit = os.path.join(exp_dir("A1_holdout"), "holdout_ids.json")
            if os.path.exists(os.path.join(exp_dir("A1_holdout"), "chunks", "done.json")):
                run(["-m", "saul.cjpe.hier", "--train", "--vectors", "A1_holdout", "--name", "A3hold",
                     "--reference", "A1", "--fit-ids", fit], "p4_A3hold.log", fatal=False)
        if not done("A9gold"):
            run(["-m", "saul.cjpe.kd", "--gold"], "p4_A9gold.log", fatal=False)
    run(["scripts/p4_zips.py"], "p4_zips_final.log", fatal=False, cpu=True)
    run(["scripts/p4_analyze.py"], "p4_analyze.log", fatal=False, cpu=True)
    print("[queue] lane finished", flush=True)


if __name__ == "__main__":
    main()
