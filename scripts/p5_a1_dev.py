"""Score dev with the saved A1 checkpoint and measure the blend with the TF-IDF scores."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from saul.lsi import exp_dir, p5_dir
from saul.lsi.const import CHUNK, MAX_CHUNKS
from saul.lsi.data import case_record, load_token_split, multi_hot
from saul.lsi.decide import quick_macro
from saul.lsi.model import load_lsi, load_overlay
from saul.lsi.train import predict_records


def main() -> None:
    directory = exp_dir("A1")
    out = os.path.join(directory, "dev_logits.npz")
    tok = os.path.join(p5_dir(), "tok")
    dev = load_token_split("dev", tok)
    if not os.path.exists(out):
        model = load_lsi("dense_v1", "lorma", "bidirectional", head="laat", loss="bce", device="cuda",
                         ckpt_every=0, attn="sdpa")
        load_overlay(model, torch.load(os.path.join(directory, "best.pt"), map_location="cuda"))
        records = [case_record(dev, i, limit=CHUNK, max_chunks=MAX_CHUNKS) for i in range(len(dev["meta"]))]
        start = time.time()
        ids, logits = predict_records(model, records, "cuda", 32768)
        print(f"[a1] dev scored in {time.time() - start:.0f}s", flush=True)
        labels = np.stack([multi_hot(dev["meta"][i]["labels"]) for i in range(len(dev["meta"]))])
        order = {case_id: i for i, case_id in enumerate(ids)}
        logits = logits[[order[row["id"]] for row in dev["meta"]]]
        np.savez(out, ids=np.asarray([row["id"] for row in dev["meta"]]), logits=logits, labels=labels)
    a1 = np.load(out)
    b1 = np.load(os.path.join(exp_dir("B1"), "dev_logits.npz"))
    order = {case_id: i for i, case_id in enumerate(b1["ids"])}
    b1_logits = b1["logits"][[order[case_id] for case_id in a1["ids"]]]
    labels = a1["labels"]

    def z(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    half = np.array(["A"] * len(labels))
    half[np.random.default_rng(781).permutation(len(labels))[: len(labels) // 2]] = "B"
    result = {}
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
        mixed = w * z(a1["logits"]) + (1 - w) * z(b1_logits)
        scored = quick_macro(mixed, labels, half)
        result[str(w)] = {"macro": round(scored["macro_f1"], 4), "half": scored["half"]}
        print(f"[a1] weight on A1 {w}: macro {scored['macro_f1']:.4f}", flush=True)
    with open(os.path.join(directory, "blend_b1.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main()
