"""E5: train-only fusion of every score that exists, then Sinkhorn, with and without citator edges."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p7_bm25 import write_zip
from p7_common import (
    OUT,
    apply_mask,
    both_halves_at_least,
    check_submission,
    citation_contexts,
    decode,
    gold_of,
    half_scores,
    masks,
    save_metrics,
    split_bundle,
    year_of,
    zrows,
)


def load_matrix(path, key, qids, cand):
    if not os.path.exists(path):
        return None
    saved = np.load(path, allow_pickle=True)
    if key not in saved.files:
        return None
    saved_q = [str(q) for q in saved["qids"].tolist()]
    saved_c = [str(c) for c in saved["cand"].tolist()]
    if saved_q != list(qids) or saved_c != list(cand):
        return None
    return saved[key].astype(np.float32)


def feature_list(tag, bundle, use_citator: bool):
    e0 = np.load(os.path.join(OUT, "e0", f"{tag}.npz"))
    names = ["full", "head", "context", "shingle"]
    feats = [zrows(e0[name]) for name in names]
    labels = list(names)
    neural = os.path.join(OUT, "e1", f"{tag}_neural.npz")
    for key in ("L", "G"):
        matrix = load_matrix(neural, key, bundle["qids"], bundle["cand"])
        if matrix is not None:
            feats.append(zrows(matrix))
            labels.append(key)
    if use_citator:
        feats.append(zrows(e0["citator"]))
        labels.append("citator")
    qy = np.array([year_of(sents)[0] or 0 for sents in bundle["q_sents"]])
    cy = np.array([year_of(sents)[0] or 0 for sents in bundle["sents"]])
    gap = np.clip(qy[:, None] - cy[None, :], -50, 50).astype(np.float32)
    feats.append(zrows(gap))
    labels.append("year_gap")
    return np.stack(feats, axis=0), labels


def fit_weights(features, gold, qids, cand, self_mask, after):
    # features: F, Q, C. Learn a positive weight per feature by listwise CE on train.
    weights = torch.nn.Parameter(torch.zeros(features.shape[0]))
    opt = torch.optim.Adam([weights], lr=0.05)
    gold_sets = [set(gold[q]) for q in qids]
    cand = list(cand)
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        score = torch.einsum("f,fqc->qc", torch.softmax(weights, dim=0), torch.tensor(features))
        loss = 0.0
        used = 0
        for row, gold_ids in enumerate(gold_sets):
            cols = [i for i, cid in enumerate(cand) if cid in gold_ids]
            if not cols:
                continue
            row_score = score[row].clone()
            row_score[self_mask[row]] = -1e4
            row_score[after[row]] = -1e4
            top = torch.topk(row_score, k=min(100, row_score.shape[0])).indices
            keep = torch.tensor(sorted(set(top.tolist()) | set(cols)), dtype=torch.long)
            chosen = row_score[keep]
            labels = torch.tensor([cand[int(i)] in gold_ids for i in keep], dtype=torch.bool)
            if not labels.any():
                continue
            loss = loss - torch.logsumexp(chosen[labels], 0) + torch.logsumexp(chosen, 0)
            used += 1
        if used == 0:
            break
        (loss / used).backward()
        opt.step()
        if step % 50 == 0:
            print(f"[e5] fusion step {step} loss {float(loss) / used:.3f}", flush=True)
    final = torch.softmax(weights.detach(), dim=0).numpy()
    return final


def predict(features, weights, bundle, self_mask, after, meta):
    score = np.tensordot(weights, features, axes=(0, 0))
    score = apply_mask(score, self_mask, after)
    counts = np.array([len(citation_contexts(sents)) or 1 for sents in bundle["q_sents"]])
    return decode(
        bundle["qids"], bundle["cand"], score, meta["decoder"], counts,
        tuple(meta["mass_coef"]), meta["eps"], 0.5,
    )


def main():
    meta_path = os.path.join(OUT, "e0", "metrics.json")
    if not os.path.exists(meta_path):
        raise SystemExit("E0 metrics are required before fusion")
    meta = json.load(open(meta_path))
    train = split_bundle("Train")
    dev = split_bundle("Dev")
    test = split_bundle("Test")
    gold = gold_of(train["queries"])
    dev_gold = gold_of(dev["queries"])
    train_mask = masks(train["qids"], train["cand"], train["q_sents"], train["sents"])
    dev_mask = masks(dev["qids"], dev["cand"], dev["q_sents"], dev["sents"])
    test_mask = masks(test["qids"], test["cand"], test["q_sents"], test["sents"])
    reports = {}
    for use_citator, name in ((False, "P7_E5"), (True, "P7_E5_citator")):
        train_f, labels = feature_list("train", train, use_citator)
        weights = fit_weights(train_f, gold, train["qids"], train["cand"], *train_mask)
        dev_f, _ = feature_list("dev", dev, use_citator)
        pred = predict(dev_f, weights, dev, *dev_mask, meta)
        report = half_scores(dev_gold, pred, dev["qids"])
        base = meta["dev"]
        kept = report["A"]["f1"] > base["A"]["f1"] and report["B"]["f1"] > base["B"]["f1"]
        reports[name] = {"weights": dict(zip(labels, [float(w) for w in weights])), "dev": report, "kept": kept}
        print(name, {k: round(report[k]["f1"], 4) for k in ("all", "A", "B")}, "kept", kept, flush=True)
        if not kept:
            continue
        test_f, _ = feature_list("test", test, use_citator)
        test_pred = predict(test_f, weights, test, *test_mask, meta)
        check_submission(test_pred, test["qids"], test["cand"])
        write_zip(name, test_pred)
    save_metrics("e5", reports)
    print("[e5] done", flush=True)


if __name__ == "__main__":
    main()
