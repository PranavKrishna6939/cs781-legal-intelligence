"""Second train-only lane: more members, snapshot averages, longer Hybrid context.

Nothing here trains on, or pretrains on, the 1,517 dev rows that are also in
test.json. D-sel selects. A step with done.json is skipped, so a restart
continues the lane. After every step the equal-weight mix and its zip are
rebuilt from whatever members exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.data import load_split
from saul.cjpe.ensemble import _align, mix_logits, select_members
from saul.cjpe.score import macro_scores, predict_logits
from saul.cjpe.submit import check_test_hash, predictions_from_logits, test_ids, write_zip

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOGS = os.path.join(ROOT, "logs")
ZIPS = os.path.join(ROOT, "submissions", "phase4")
VIEW_GAIN = 0.003


def run(args: list[str], log_name: str) -> int:
    print(f"[r2] {log_name}: {' '.join(args)}", flush=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, log_name), "ab") as handle:
        code = subprocess.Popen(
            [PY, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        ).wait()
    print(f"[r2] {log_name} exit {code}", flush=True)
    return code


def done(name: str) -> bool:
    return os.path.exists(os.path.join(exp_dir(name), "done.json"))


def info(name: str) -> dict:
    return json.load(open(os.path.join(exp_dir(name), "done.json"), encoding="utf-8"))


def train(exp: str, extra: list[str], log_name: str | None = None) -> int:
    if done(exp):
        return 0
    return run(["-m", "saul.cjpe.train", "--exp", exp, "--no-ovl", "--attn", "sdpa", "--snapshots", *extra],
               log_name or f"p4_{exp}.log")


def hybrid(exp: str, limit: int, seed: int, batch_tokens: int) -> list[str]:
    return ["--model", "hybrid_b", "--technique", "adapter_bitfit", "--context", "causal", "--view", "tail",
            "--limit", str(limit), "--epochs", "1", "--batch-tokens", str(batch_tokens), "--seed", str(seed)]


def dense(seed: int, init: str | None = None) -> list[str]:
    out = ["--model", "dense_v1", "--technique", "lorma", "--context", "bidirectional", "--view", "tail",
           "--limit", "2048", "--epochs", "2", "--batch-tokens", "16384", "--accum", "4", "--compile",
           "--gate", "0.74", "--first-limit", "1024", "--seed", str(seed)]
    if init:
        out += ["--init", init]
    return out


def _scores(logits: np.ndarray, dsel) -> dict:
    pred = predict_logits(logits, 0.0)
    halves = np.asarray(dsel.halves)
    return {
        "f1": macro_scores(dsel.labels, pred)["f1"],
        "a": macro_scores(dsel.labels[halves == "A"], pred[halves == "A"])["f1"],
        "b": macro_scores(dsel.labels[halves == "B"], pred[halves == "B"])["f1"],
    }


def member_logits(name: str, kind: str, order: list[str]) -> np.ndarray:
    plain = info(name).get("loss") == "ce"
    return _align(name, kind, order, plain)


def _passes(new: dict, base: dict) -> bool:
    return new["f1"] - base["f1"] >= VIEW_GAIN and new["a"] > base["a"] and new["b"] > base["b"]


def with_views(base: str, views: list[str], dsel, test_order: list[str]) -> dict:
    """Pick one representation of a run: the base, one longer view, or their average.

    A candidate is kept only when it gains at least VIEW_GAIN overall and on both
    D-sel halves. Views of the same run are alternatives. When two of them pass,
    the one that wins on both halves is used; otherwise the higher macro-F1 wins.
    """
    order = list(dsel.ids)
    d = member_logits(base, "dsel", order)
    t = member_logits(base, "test", test_order)
    base_s = _scores(d, dsel)
    chosen = {"dsel": d, "test": t, "scores": base_s, "used": [base]}
    by_view: dict[str, dict] = {}
    for view in views:
        if not done(view):
            continue
        vd = member_logits(view, "dsel", order)
        vt = member_logits(view, "test", test_order)
        options = {
            "avg": (mix_logits([d, vd]), mix_logits([t, vt]), [base, view]),
            "replace": (vd, vt, [view]),
        }
        best = None
        for kind, (od, ot, used) in options.items():
            scores = _scores(od, dsel)
            keep = _passes(scores, base_s)
            print(f"[r2] view {view} {kind} on {base}: {scores['f1']:.4f} "
                  f"halves {scores['a']:.4f}/{scores['b']:.4f} keep {keep}", flush=True)
            if keep and (best is None or scores["f1"] > best["scores"]["f1"]):
                best = {"dsel": od, "test": ot, "scores": scores, "used": used, "kind": kind}
        if best is not None:
            by_view[view] = best
    if not by_view:
        return chosen
    names = list(by_view)
    winner = names[0]
    for other in names[1:]:
        cur, nxt = by_view[winner]["scores"], by_view[other]["scores"]
        if nxt["a"] > cur["a"] and nxt["b"] > cur["b"]:
            winner = other
        elif not (cur["a"] > nxt["a"] and cur["b"] > nxt["b"]) and nxt["f1"] > cur["f1"]:
            winner = other
    chosen = by_view[winner]
    print(f"[r2] {base} uses {'+'.join(chosen['used'])} ({chosen['kind']}) {chosen['scores']['f1']:.4f}", flush=True)
    return chosen


MEMBERS = [
    # (member name, base run, re-scored views of it; views are alternatives)
    ("dense", "C1", ["C1_t4096", "C1_t8192"]),
    ("hybrid", "C5", ["C5_t8192"]),
    ("ced", "C7_snap", []),
    ("hybrid8k", "C6_snap", ["C6_t16384"]),
    ("hybrid_s2", "C8_snap", []),
    ("dense_tapt", "C10_snap", ["C10_t4096"]),
    ("dense_s2", "C9_snap", ["C9_t4096"]),
]


def build_mix() -> None:
    check_test_hash(stop=True)
    dsel = load_split("dsel")
    wanted = test_ids()
    pool = {}
    for name, base, views in MEMBERS:
        if not done(base):
            continue
        pool[name] = with_views(base, views, dsel, wanted)
    if not pool:
        return
    if pool.get("ced") and pool["ced"]["scores"]["f1"] < 0.80:
        print(f"[r2] drop ced {pool['ced']['scores']['f1']:.4f} < 0.80", flush=True)
        del pool["ced"]
    for name, member in pool.items():
        s = member["scores"]
        print(f"[r2] member {name} ({'+'.join(member['used'])}) dsel {s['f1']:.4f} halves {s['a']:.4f}/{s['b']:.4f}", flush=True)

    def mix_f1(names: list[str]) -> float:
        return _scores(mix_logits([pool[n]["dsel"] for n in names]), dsel)["f1"]

    chosen = select_members({n: m["scores"]["f1"] for n, m in pool.items()}, mix_f1)
    d = mix_logits([pool[n]["dsel"] for n in chosen])
    t = mix_logits([pool[n]["test"] for n in chosen])
    s = _scores(d, dsel)
    out = os.path.join(p4_dir(), "mix", "noleak")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "logits_dsel.npz"), ids=np.asarray(dsel.ids), logits=d, labels=dsel.labels)
    np.savez(os.path.join(out, "logits_test.npz"), ids=np.asarray(wanted), logits=t)
    blob = {"members": chosen, "used": {n: pool[n]["used"] for n in chosen}, "dev_f1": s["f1"],
            "dev_f1_a": s["a"], "dev_f1_b": s["b"]}
    with open(os.path.join(out, "mix.json"), "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    tag = f"{int(round(s['f1'] * 10000)):04d}"
    print(f"[r2] mix {'+'.join(chosen)} dsel {s['f1']:.4f} halves {s['a']:.4f}/{s['b']:.4f}", flush=True)
    # The score is part of the directory name, so a later mix cannot overwrite this zip.
    write_zip(predictions_from_logits(wanted, t, 0.0),
              os.path.join(ZIPS, "P4_mix_noleak_" + "_".join(chosen) + "_" + tag), wanted)
    for name, member in pool.items():
        if name in ("dense", "hybrid") and len(member["used"]) == 1:
            continue
        label = f"{name}_views" if name in ("dense", "hybrid") else name
        member_tag = f"{int(round(member['scores']['f1'] * 10000)):04d}"
        write_zip(predictions_from_logits(wanted, member["test"], 0.0),
                  os.path.join(ZIPS, f"P4_{label}_noleak_{member_tag}"), wanted)


def infer(source: str, name: str, limit: int) -> int:
    if done(name):
        return 0
    return run(["-m", "saul.cjpe.infer", "--source", source, "--name", name,
                "--limit", str(limit), "--batch-tokens", "16384"], f"p4_{name}.log")


def main() -> None:
    build_mix()

    # Longer views of the two finished runs, scored without retraining.
    if not done("C5_t8192"):
        run(["-m", "saul.cjpe.infer", "--source", "C5", "--name", "C5_t8192", "--limit", "8192",
             "--batch-tokens", "16384"], "p4_C5_t8192.log")
    if not done("C1_t4096"):
        run(["-m", "saul.cjpe.infer", "--source", "C1", "--name", "C1_t4096", "--limit", "4096",
             "--batch-tokens", "16384"], "p4_C1_t4096.log")
    infer("C1", "C1_t8192", 8192)
    build_mix()

    # CED partial encoder: fast, and a different backbone from both members.
    train("C7", ["--model", "ced", "--technique", "partial", "--context", "encoder", "--view", "tail",
                 "--limit", "2048", "--epochs", "2", "--batch-tokens", "32768"])
    build_mix()

    # Hybrid trained at tail 8192, one sequence per batch. A shorter tail would still pad to
    # the 8192 bucket, so the memory fallback is checkpointing every block instead.
    if not done("C6"):
        code = train("C6", hybrid("C6", 8192, 782, 8192))
        if code == 4:
            train("C6", hybrid("C6", 8192, 782, 8192) + ["--ckpt-every", "1", "--force"], "p4_C6_ckpt1.log")
    build_mix()

    # Second Hybrid seed at the C5 recipe, one epoch (C5's best checkpoint was the end of epoch 0).
    train("C8", hybrid("C8", 4096, 783, 16384))
    build_mix()

    # TAPT on train.json text only, then the dense recipe from it.
    clean = os.path.join(p4_dir(), "adapt", "dense_v1_clean")
    if not os.path.exists(os.path.join(clean, "done.json")):
        run(["-m", "saul.cjpe.tapt", "--clean"], "p4_tapt_clean.log")
    if os.path.exists(os.path.join(clean, "done.json")):
        train("C10", dense(781, clean))
    infer("C10", "C10_t4096", 4096)
    build_mix()

    # Second dense seed.
    train("C9", dense(783))
    infer("C9", "C9_t4096", 4096)
    build_mix()

    # Longer read of the long-context Hybrid, only after the training steps.
    infer("C6", "C6_t16384", 16384)
    build_mix()
    print("[r2] lane finished", flush=True)


if __name__ == "__main__":
    main()
