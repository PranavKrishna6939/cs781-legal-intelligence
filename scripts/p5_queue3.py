"""SAUL-LSI ablations, then E2 / E1 / E3 / E4. Run this on the A10G box, one GPU job at a time."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saul.lsi import exp_dir, p5_dir

GAIN = 0.003


def _done(exp: str) -> dict | None:
    path = os.path.join(exp_dir(exp), "done.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _pair(exp: str) -> tuple[float, float, float]:
    done = _done(exp)
    if done is None:
        raise SystemExit(f"[queue] {exp} has no done.json")
    half = done["half"]
    return float(half["A"]), float(half["B"]), float(done["dev_f1"])


def _beats(new: str, old: str) -> bool:
    left, right = _pair(new), _pair(old)
    ok = left[0] >= right[0] + GAIN and left[1] >= right[1] + GAIN
    print(f"[queue] {new} {left[2]:.4f} (A {left[0]:.4f} B {left[1]:.4f}) vs "
          f"{old} {right[2]:.4f} (A {right[0]:.4f} B {right[1]:.4f}) -> {'keep' if ok else 'drop'}",
          flush=True)
    return ok


def _call(cmd: list[str]) -> int:
    print("[queue]", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code not in (4, -9):
        return code
    print(f"[queue] killed ({code}), retrying with half the micro-batch", flush=True)
    retry = list(cmd)
    batch_at = retry.index("--batch-tokens")
    accum_at = retry.index("--accum")
    retry[batch_at + 1] = str(max(4096, int(retry[batch_at + 1]) // 2))
    retry[accum_at + 1] = str(int(retry[accum_at + 1]) * 2)
    print("[queue]", " ".join(retry), flush=True)
    return subprocess.call(retry)


def _run(cmd: list[str], exp: str) -> None:
    if _done(exp) is not None:
        print(f"[queue] reuse {exp}", flush=True)
        return
    code = _call(cmd)
    if code != 0 or _done(exp) is None:
        raise SystemExit(f"[queue] {exp} failed with code {code}")


def _cmd(exp: str, model: str, technique: str, epochs: int, train_s: int, infer_s: int,
         accum: int, base_lr: str, ckpt: int, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-m", "saul.lsi.train",
        "--exp", exp, "--model", model, "--technique", technique,
        "--head", "decoder", "--view", "segments", "--limit", "512",
        "--compile-blocks", "--batch-tokens", "16384", "--accum", str(accum),
        "--epochs", str(epochs), "--eval-per-epoch", "--no-wandb",
        "--max-chunks", str(train_s), "--infer-chunks", str(infer_s),
        "--head-lr", "5e-4", "--embed-lr", "1e-3", "--adapter-lr", "5e-4",
        "--base-lr", base_lr, "--lr-floor", "0.1", "--decay", "0.85",
        "--attn", "sdpa", "--ckpt-every", str(ckpt),
    ]
    return cmd + extra


def _arch(kept: dict) -> list[str]:
    extra = ["--loss", "db", "--deep-sup", "0.3"] if kept["db"] else ["--loss", "bce"]
    if kept["cooc"]:
        extra.append("--cooc")
    if kept["retrieval"]:
        extra += ["--retrieval", "--knn", "8"]
    if kept["lexical"]:
        extra += ["--lexical", "--wide-dropout", "0.25", "--ema", "0.999"]
    return extra


def _wait_feats() -> None:
    feats = os.path.join(p5_dir(), "feats")
    needed = ("zlex_train.npy", "zlex_dev.npy", "zlex_test.npy", "knn_train.npz", "knn_dev.npz",
              "knn_test.npz", "knn_labels.npy")
    for _ in range(120):
        missing = [name for name in needed if not os.path.exists(os.path.join(feats, name))]
        if not missing:
            return
        print(f"[queue] waiting for features: {missing[0]}", flush=True)
        time.sleep(30)
    raise SystemExit("[queue] features were not ready after an hour")


def main() -> None:
    ced = dict(model="ced", technique="full", epochs=1, train_s=3, infer_s=3, accum=8,
               base_lr="4e-5", ckpt=0)
    _run(_cmd("A1_decoder", extra=["--loss", "bce", "--skip-test"], **ced), "A1_decoder")
    if _pair("A1_decoder")[2] < 0.28:
        raise SystemExit("[queue] A1 macro-F1 is below 0.28. The decoder is not learning; not starting the rest.")
    _run(_cmd("A2_cooc", extra=["--loss", "bce", "--cooc", "--skip-test"], **ced), "A2_cooc")
    _run(_cmd("A3_db", extra=["--loss", "db", "--deep-sup", "0.3", "--cooc", "--skip-test"], **ced), "A3_db")
    _wait_feats()
    _run(_cmd("A4_retrieval", extra=["--loss", "db", "--deep-sup", "0.3", "--cooc", "--retrieval", "--knn", "8",
                                    "--skip-test"], **ced), "A4_retrieval")
    _run(_cmd("A5_full", extra=["--loss", "db", "--deep-sup", "0.3", "--cooc", "--retrieval", "--knn", "8",
                               "--lexical", "--wide-dropout", "0.25", "--ema", "0.999", "--skip-test"], **ced),
         "A5_full")

    kept = {
        "cooc": _beats("A2_cooc", "A1_decoder"),
        "db": _beats("A3_db", "A2_cooc"),
        "retrieval": _beats("A4_retrieval", "A3_db"),
        "lexical": _beats("A5_full", "A4_retrieval"),
    }
    # A component that lost its own comparison stays out, even if a later row was higher for another reason.
    print("[queue] kept", json.dumps(kept), flush=True)
    arch = _arch(kept)
    full = all(kept.values())
    e2_extra = list(arch) + ["--train-logits", "--train-states"]
    e2 = dict(model="ced", technique="full", epochs=2 if full else 3, train_s=4, infer_s=8, accum=8,
              base_lr="4e-5", ckpt=0)
    if full:
        e2_extra += ["--init-from", os.path.join(exp_dir("A5_full"), "best.pt")]
    _run(_cmd("E2_ced", extra=e2_extra, **e2), "E2_ced")
    _run(_cmd("E1_dense_lorma", extra=arch, model="dense_v1", technique="lorma", epochs=2, train_s=2,
              infer_s=6, accum=4, base_lr="3e-5", ckpt=2), "E1_dense_lorma")
    _run(_cmd("E3_hybrid_adapter", extra=arch + ["--context", "causal"], model="hybrid_b", technique="adapter",
              epochs=2, train_s=2, infer_s=4, accum=4, base_lr="3e-5", ckpt=2), "E3_hybrid_adapter")
    teacher = os.path.join(exp_dir("E2_ced"), "train_logits.npz")
    states = os.path.join(exp_dir("E2_ced"), "train_states.f16")
    e4_extra = arch + ["--context", "causal", "--teacher", teacher, "--teacher-states", states]
    _run(_cmd("E4_conv_kd", extra=e4_extra, model="conv_mini", technique="full", epochs=3, train_s=3,
              infer_s=8, accum=6, base_lr="3e-5", ckpt=0), "E4_conv_kd")
    print("[queue] finished", flush=True)


if __name__ == "__main__":
    main()
