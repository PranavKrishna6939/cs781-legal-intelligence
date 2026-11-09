"""Rescore the distilled student with its own passage vectors. The first E4 number reused LoRMA's."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from p7_common import OUT, apply_mask, citation_contexts, gold_of, half_scores, masks, passages, save_metrics, split_bundle, zrows
from p7_e1 import SLOT_TAU, embed, encode_split, scatter_max
from p7_model import Head, build, checkpoint_every, load_tokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bos, pad = load_tokenizer()
    model = build("student")
    head = Head(model.config.d_model)
    blob = torch.load(os.path.join(OUT, "e4", "student.pt"), map_location="cpu")
    model.load_state_dict(blob["model"], strict=False)
    head.load_state_dict(blob["head"])
    checkpoint_every(model)
    model.to(device).eval()
    head.to(device).eval()
    dev = split_bundle("Dev")
    path = os.path.join(OUT, "e4", "dev_passages.npz")
    encode_split(model, head, tok, bos, pad, dev, device, path)
    saved = np.load(path)
    cand_vec = torch.tensor(saved["vec"].astype(np.float32), device=device)
    owner = saved["owner"]
    n_cand = len(dev["cand"])
    local = np.zeros((len(dev["qids"]), n_cand), dtype=np.float32)
    with torch.no_grad():
        for row, sents in enumerate(dev["q_sents"]):
            ctx = citation_contexts(sents)[:12] or ["context: " + " ".join(sents[:6])]
            ctx_v = embed(model, head, tok, bos, pad, ctx, device).float()
            per = scatter_max(ctx_v @ cand_vec.T, owner, n_cand)
            local[row] = (SLOT_TAU * torch.logsumexp(per / SLOT_TAU, dim=0)).float().cpu().numpy()
            if row % 40 == 0:
                print(f"[e4] scored {row}", flush=True)
    meta = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    e0 = np.load(os.path.join(OUT, "e0", "dev.npz"))
    names = ["full", "head", "context", "shingle"]
    lexical = sum(w * zrows(e0[n]) for w, n in zip(meta["weights"], names))
    self_mask, after = masks(dev["qids"], dev["cand"], dev["q_sents"], dev["sents"])
    gold = gold_of(dev["queries"])
    neural = apply_mask(zrows(local), self_mask, after)
    from p7_common import decode
    pred_n = decode(dev["qids"], dev["cand"], neural, "column", tau=0.5)
    fused = apply_mask(lexical + zrows(local), self_mask, after)
    pred_f = decode(dev["qids"], dev["cand"], fused, meta["decoder"], None, tuple(meta["mass_coef"]), meta["eps"])
    report_n = half_scores(gold, pred_n, dev["qids"])
    report_f = half_scores(gold, pred_f, dev["qids"])
    base = meta["dev"]
    kept = report_f["A"]["f1"] >= base["A"]["f1"] and report_f["B"]["f1"] >= base["B"]["f1"]
    previous = {}
    prev_path = os.path.join(OUT, "e4", "metrics.json")
    if os.path.exists(prev_path):
        previous = json.load(open(prev_path))
    previous.update({"neural": report_n, "fused": report_f, "kept": kept, "note": "rescored with the student's own passages"})
    save_metrics("e4", previous)
    print("[e4] neural", {k: round(report_n[k]["f1"], 4) for k in ("all", "A", "B")}, flush=True)
    print("[e4] fused", {k: round(report_f[k]["f1"], 4) for k in ("all", "A", "B")}, "kept", kept, flush=True)


if __name__ == "__main__":
    main()
