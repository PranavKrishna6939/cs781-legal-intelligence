"""Where LAAT looks, and a first-chunk ablation that does not retrain."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from saul.lsi import exp_dir, p5_dir
from saul.lsi.data import case_record, collate, load_token_split
from saul.lsi.model import load_lsi, load_overlay
from saul.lsi.decide import quick_macro
from saul.lsi.score import macro_f1, sigmoid
from saul.utils.io import write_json


def _batches(records, budget):
    from saul.lsi.data import make_batches
    lengths = [len(record["chunks"]) for record in records]
    widths = [record["width"] for record in records]
    return make_batches(lengths, widths, budget, np.random.default_rng(0))


def attention_by_decile(model, records, device: str, budget: int = 32768) -> dict:
    model.eval()
    mass = np.zeros(10, dtype=np.float64)
    seen = 0
    with torch.no_grad():
        for width, indexes in _batches(records, budget):
            batch = collate([records[i] for i in indexes], width)
            moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            alpha = model(moved, return_extra=True)["alpha"]
            weight = alpha.mean(dim=1).float().cpu().numpy()
            for row in weight:
                total = row.sum() + 1e-8
                for decile, positions in enumerate(np.array_split(np.arange(len(row)), 10)):
                    mass[decile] += row[positions].sum() / total
                seen += 1
    mass /= max(seen, 1)
    return {"decile_mass": mass.tolist(), "cases": seen}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--technique", default="lorma")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lsi(args.model, args.technique, head="laat", device=device, ckpt_every=0)
    blob = torch.load(os.path.join(exp_dir(args.exp), "best.pt"), map_location="cpu", weights_only=False)
    load_overlay(model, blob)
    dev = load_token_split("dev", os.path.join(p5_dir(), "tok"))
    n_cases = len(dev["meta"]) if args.limit <= 0 else min(len(dev["meta"]), args.limit)
    records = [case_record(dev, i, limit=2048) for i in range(n_cases)]
    report = attention_by_decile(model, records, device)
    from saul.lsi.data import dev_half_array, multi_hot
    labels = np.stack([multi_hot(dev["meta"][i]["labels"]) for i in range(len(records))])
    from saul.lsi.train import predict_records
    full_ids, full = predict_records(model, records, device, 32768)
    first = []
    for source in records:
        record = dict(source)
        record["chunks"] = source["chunks"][:1]
        record["width"] = source["width"]
        first.append(record)
    head_ids, head = predict_records(model, first, device, 32768)
    half = dev_half_array([record["id"] for record in records])
    full_pos = {case_id: i for i, case_id in enumerate(full_ids)}
    head_pos = {case_id: i for i, case_id in enumerate(head_ids)}
    order = [record["id"] for record in records]
    full_m = full[[full_pos[i] for i in order]]
    head_m = head[[head_pos[i] for i in order]]
    def _macro(logits: np.ndarray) -> float:
        if len(set(np.asarray(half).tolist())) < 2:
            return macro_f1(labels, sigmoid(logits) >= 0.5)
        return quick_macro(logits, labels, half)["macro_f1"]

    report["full_macro"] = _macro(full_m)
    report["first_chunk_macro"] = _macro(head_m)
    write_json(report, os.path.join(exp_dir(args.exp), "sections.json"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
