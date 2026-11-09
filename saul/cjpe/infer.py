"""Re-score a trained overlay on D-sel and test with a different view or length."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from saul.cjpe import exp_dir
from saul.cjpe.calibrate import apply_prior
from saul.cjpe.data import load_split
from saul.cjpe.model import load_cjpe, load_overlay
from saul.cjpe.score import macro_scores, predict_logits
from saul.cjpe.train import _save_logits, predict_split
from saul.utils.io import ensure_dir

LENGTH_BINS = ((0, 2048), (2048, 4096), (4096, 8192), (8192, 10 ** 9))


def length_report(labels: np.ndarray, pred: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in LENGTH_BINS:
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() >= 30:
            out[f"{lo}-{hi}"] = macro_scores(labels[mask], pred[mask])["f1"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--view", default=None)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()

    out = exp_dir(args.name)
    marker = os.path.join(out, "done.json")
    if os.path.exists(marker):
        print(f"[infer] reuse {args.name}", flush=True)
        return
    ensure_dir(out)
    src = json.load(open(os.path.join(exp_dir(args.source), "done.json"), encoding="utf-8"))
    view = args.view or src["view"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_cjpe(src["model"], src["technique"], src["context"], init=src.get("init"),
                      device=device, loss=src["loss"], ckpt_every=0, attn=args.attn)
    blob = torch.load(os.path.join(exp_dir(args.source), "overlay.pt"), map_location="cpu", weights_only=False)
    load_overlay(model, blob)
    model.to(device)

    dsel = load_split("dsel")
    test = load_split("test")
    # Same precision as the scoring pass inside train.py, so scores are comparable.
    raw, ids = predict_split(model, dsel, device, view, args.limit, src["head"], src["tail"], args.batch_tokens)
    test_raw, test_ids = predict_split(model, test, device, view, args.limit, src["head"], src["tail"], args.batch_tokens)
    _save_logits(os.path.join(out, "logits_dsel.npz"), ids, raw, dsel.labels)
    _save_logits(os.path.join(out, "logits_test.npz"), test_ids, test_raw)

    pred = predict_logits(apply_prior(raw, src["loss"] == "ce"), 0.0)
    scores = macro_scores(dsel.labels, pred)
    halves = np.asarray(dsel.halves)
    lengths = np.diff(np.asarray(dsel.offsets))
    blob = {
        **{k: src[k] for k in ("model", "technique", "context", "head", "tail", "loss", "init")},
        "exp": args.name, "source": args.source, "view": view, "limit": args.limit,
        "dev_f1": scores["f1"], "dev_p": scores["precision"], "dev_r": scores["recall"],
        "dev_f1_a": macro_scores(dsel.labels[halves == "A"], pred[halves == "A"])["f1"],
        "dev_f1_b": macro_scores(dsel.labels[halves == "B"], pred[halves == "B"])["f1"],
        "by_length": length_report(dsel.labels, pred, lengths),
    }
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    print(f"[infer] {args.name} dev_f1 {blob['dev_f1']:.4f} halves {blob['dev_f1_a']:.4f}/{blob['dev_f1_b']:.4f} "
          f"by_length {json.dumps({k: round(v, 3) for k, v in blob['by_length'].items()})}", flush=True)


if __name__ == "__main__":
    main()
