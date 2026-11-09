"""R3: LoRMA residual scorer.

Positives are passage pairs from a train gold link that share an 8-word span. The span is
masked on the query side. Negatives are passages from the 4-gram ranker's top non-gold.
The score is one feature in the R2 ranker and is kept only when both dev halves improve.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from p7_common import OUT, apply_mask, citation_contexts, gold_of, half_scores, masks, passages, save_metrics, split_bundle, zrows
from p7_e1 import info_nce, load_run, save_run
from p7_model import Head, build, checkpoint_every, cosine, encode_ids, forward_hidden, gpu_mb, load_tokenizer, pool
from p7_r1 import named_zip
from p7_r2 import (
    apply_decoder,
    build as build_ranker,
    decode_grid,
    feature_names,
    fill_scores,
    fit_ranker,
    rows_of,
)

LIMIT = 192
N_NEG = 7
TEMP = 0.05


def gram_set(words: list[str], n: int = 6) -> set[int]:
    found = set()
    for i in range(0, max(len(words) - n + 1, 0)):
        found.add(hash(tuple(words[i:i + n])) & 0xFFFFFFFF)
    return found


def mask_shared(query: str, passage: str, n: int = 6) -> str:
    qwords = query.split()
    pwords = passage.split()
    target = {tuple(pwords[i:i + n]) for i in range(max(len(pwords) - n + 1, 0))}
    i = 0
    while i <= len(qwords) - n:
        if tuple(qwords[i:i + n]) in target:
            qwords[i:i + n] = ["mask"] * n
            i += n
        else:
            i += 1
    return " ".join(qwords)


def mine_pairs() -> list[dict]:
    bundle = split_bundle("Train")
    gold = gold_of(bundle["queries"])
    cand_index = {c: i for i, c in enumerate(bundle["cand"])}
    cand_passages = [passages(sents) for sents in bundle["sents"]]
    cand_sets = [[gram_set(p.split()) for p in group] for group in cand_passages]
    self_mask, after = masks(bundle["qids"], bundle["cand"], bundle["q_sents"], bundle["sents"])
    stored = np.load(os.path.join(OUT, "r1", "train.npz"))
    full4 = apply_mask(stored["full4"], self_mask, after)
    rows = []
    for q_i, q in enumerate(bundle["qids"]):
        ctx = citation_contexts(bundle["q_sents"][q_i])
        q_passages = (ctx + passages(bundle["q_sents"][q_i]))[:20]
        q_sets = [gram_set(p.split()) for p in q_passages]
        gset = {str(c) for c in gold[q]}
        ranked = np.argsort(-full4[q_i])
        negatives = [bundle["cand"][j] for j in ranked if bundle["cand"][j] not in gset][:N_NEG]
        neg_text = []
        for cid in negatives:
            group = cand_passages[cand_index[cid]]
            neg_text.append(group[0] if group else "")
        for cid in gset:
            if cid not in cand_index:
                continue
            best = (0, None, None)
            for qi, qset in enumerate(q_sets):
                if not qset:
                    continue
                for pi, cset in enumerate(cand_sets[cand_index[cid]]):
                    overlap = len(qset & cset)
                    if overlap > best[0]:
                        best = (overlap, qi, pi)
            if best[1] is None:
                continue
            query = mask_shared(q_passages[best[1]], cand_passages[cand_index[cid]][best[2]])
            rows.append({
                "q": q,
                "c": cid,
                "query": query,
                "positive": cand_passages[cand_index[cid]][best[2]],
                "negatives": neg_text,
            })
        if q_i % 50 == 0:
            print(f"[r3] mine {q_i}/{len(bundle['qids'])} pairs {len(rows)}", flush=True)
    path = os.path.join(OUT, "r3", "pairs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[r3] pairs {len(rows)} -> {path}", flush=True)
    return rows


def load_pairs() -> list[dict]:
    path = os.path.join(OUT, "r3", "pairs.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pad_to(tok, texts, bos, pad, device) -> tuple[torch.Tensor, list[int | None]]:
    encoded = [encode_ids(tok, text, bos, LIMIT) for text in texts]
    ids = torch.full((len(texts), LIMIT), pad, dtype=torch.long, device=device)
    for i, (row, _) in enumerate(encoded):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return ids, [index for _, index in encoded]


def embed(model, head, tok, bos, pad, texts, device) -> torch.Tensor:
    ids, indexes = pad_to(tok, texts, bos, pad, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = forward_hidden(model, ids)
    return head(pool(hidden, ids, pad, indexes))


def step_loss(model, head, tok, bos, pad, rows, device) -> torch.Tensor:
    queries, positives, negatives = [], [], []
    for row in rows:
        queries.append(row["query"])
        positives.append(row["positive"])
        negatives.extend(row["negatives"][:N_NEG])
    vectors = embed(model, head, tok, bos, pad, queries + positives + negatives, device)
    n = len(rows)
    query = vectors[:n]
    positive = vectors[n:n + n]
    negative = vectors[n + n:].view(n, N_NEG, -1)
    cands = torch.cat([positive.unsqueeze(1), negative], dim=1)
    scores = torch.einsum("bd,bkd->bk", query, cands) / TEMP
    hard = F.cross_entropy(scores, torch.zeros(n, dtype=torch.long, device=device))
    return hard + 0.1 * info_nce(query, positive)


def train(steps: int) -> None:
    rows = load_pairs()
    rows = [row for row in rows if len(row["negatives"]) >= N_NEG and row["query"] and row["positive"]]
    print(f"[r3] trainable pairs {len(rows)}", flush=True)
    device = "cuda"
    model = build("lorma").to(device)
    head = Head(model.config.d_model).to(device)
    pretrain = os.path.join(OUT, "e1", "pretrain.pt")
    if os.path.exists(pretrain):
        load_run(pretrain, model, head)
        print("[r3] loaded e1 pretrain", flush=True)
    checkpoint_every(model)
    model.train()
    head.train()
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 5e-4},
        {"params": list(head.parameters()), "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    bases = [5e-4, 1e-3]
    tok, bos, pad = load_tokenizer()
    batch = 16
    rng = random.Random(781)
    path = os.path.join(OUT, "r3", "lorma.pt")
    for step in range(steps):
        chunk = [rows[rng.randrange(len(rows))] for _ in range(batch)]
        opt.zero_grad(set_to_none=True)
        try:
            loss = step_loss(model, head, tok, bos, pad, chunk, device)
        except torch.cuda.OutOfMemoryError:
            batch = max(4, batch // 2)
            print(f"[r3] oom, batch now {batch}", flush=True)
            torch.cuda.empty_cache()
            continue
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
        for group, base in zip(opt.param_groups, bases):
            group["lr"] = cosine(step, steps, base)
        opt.step()
        if step % 20 == 0 or step + 1 == steps:
            print(f"[r3] step {step + 1}/{steps} loss {float(loss.detach()):.3f} batch {batch} gpu {gpu_mb()}MB", flush=True)
        if (step + 1) % 200 == 0 or step + 1 == steps:
            save_run(path, model, head, {"stage": "r3", "step": step + 1})
    print(f"[r3] saved {path}", flush=True)


def load_model(device):
    model = build("lorma").to(device)
    head = Head(model.config.d_model).to(device)
    load_run(os.path.join(OUT, "r3", "lorma.pt"), model, head)
    model.eval()
    head.eval()
    return model, head


def encode_texts(model, head, tok, bos, pad, texts, device, batch: int = 32) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            vec = embed(model, head, tok, bos, pad, texts[start:start + batch], device)
            out.append(vec.float().cpu().numpy())
            done = min(start + batch, len(texts))
            if done == batch or done % (batch * 50) == 0 or done == len(texts):
                print(f"[r3] encoded {done}/{len(texts)}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 256), dtype=np.float32)


def score_split(split: str, model, head, tok, bos, pad, device) -> None:
    name = {"Train": "train", "Dev": "dev", "Test": "test"}[split]
    bundle = split_bundle(split)
    stored = np.load(os.path.join(OUT, "r1", f"{name}.npz"))
    self_mask, after = masks(bundle["qids"], bundle["cand"], bundle["q_sents"], bundle["sents"])
    if [str(q) for q in bundle["qids"]] != [str(x) for x in stored["qids"]]:
        raise SystemExit(f"{split} id order differs from R1")
    full4 = apply_mask(stored["full4"], self_mask, after)
    cand_passages = [passages(sents) for sents in bundle["sents"]]
    cand_sets = [[gram_set(p.split()) for p in group] for group in cand_passages]
    chosen_text = []
    chosen_at = []
    for q_i, q in enumerate(bundle["qids"]):
        q_passages = citation_contexts(bundle["q_sents"][q_i]) or passages(bundle["q_sents"][q_i])[:2]
        q_sets = [gram_set(p.split()) for p in q_passages]
        top = np.argpartition(-full4[q_i], min(99, full4.shape[1] - 1))[:100]
        for j in top:
            best = (0, 0)
            for qset in q_sets:
                for pi, cset in enumerate(cand_sets[j]):
                    overlap = len(qset & cset)
                    if overlap > best[0]:
                        best = (overlap, pi)
            text = cand_passages[j][best[1]] if cand_passages[j] else ""
            chosen_text.append(text)
            chosen_at.append((q_i, int(j), len(chosen_text) - 1))
        if q_i % 40 == 0:
            print(f"[r3] pick {split} {q_i}/{len(bundle['qids'])}", flush=True)
    q_texts = []
    q_spans = []
    for sents in bundle["q_sents"]:
        ctx = citation_contexts(sents) or passages(sents)[:2]
        q_spans.append((len(q_texts), len(q_texts) + len(ctx)))
        q_texts.extend(ctx)
    print(f"[r3] encode {split} queries {len(q_texts)} passages {len(chosen_text)}", flush=True)
    q_vec = encode_texts(model, head, tok, bos, pad, q_texts, device)
    c_vec = encode_texts(model, head, tok, bos, pad, chosen_text, device)
    scores = np.full((len(bundle["qids"]), len(bundle["cand"])), -1.0, dtype=np.float32)
    for q_i, j, slot in chosen_at:
        lo, hi = q_spans[q_i]
        if hi > lo:
            scores[q_i, j] = float((q_vec[lo:hi] @ c_vec[slot]).max())
    np.save(os.path.join(OUT, "r3", f"{name}_L.npy"), scores)
    print(f"[r3] scored {split}", flush=True)


def fuse() -> None:
    e0 = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    r1 = json.load(open(os.path.join(OUT, "r1", "metrics.json")))
    r2 = json.load(open(os.path.join(OUT, "r2", "metrics.json")))
    k1, b = r1["params"]["full4"]["k1"], r1["params"]["full4"]["b"]
    train_gold_raw = gold_of(split_bundle("Train")["queries"])
    train_gold_raw = {str(q): {str(c) for c in ids} for q, ids in train_gold_raw.items()}
    dev_gold = {str(q): [str(c) for c in ids] for q, ids in gold_of(split_bundle("Dev")["queries"]).items()}
    built = {}
    for split, leave in (("train", True), ("dev", False), ("test", False)):
        built[split] = build_ranker(split, e0, train_gold_raw, k1, b, leave)
        neural = np.load(os.path.join(OUT, "r3", f"{split}_L.npy"))
        masked = apply_mask(neural, built[split]["self_mask"], built[split]["after"])
        built[split]["cols"]["neural"] = zrows(masked)
        built[split]["cols"]["neural_col"] = zrows(masked.T).T
    train_gold = {q: train_gold_raw.get(q, set()) & set(built["train"]["cand"]) for q in built["train"]["qids"]}
    names = feature_names(with_citator=True) + ["neural", "neural_col"]
    params = r2["params"]
    x, y, groups, _ = rows_of(built["train"], names, train_gold)
    model = fit_ranker(x, y, groups, params, params["rounds"])
    scored = {}
    for split in ("train", "dev", "test"):
        xx, _, _, keep = rows_of(built[split], names, None)
        pred = model.predict(xx)
        scored[split] = fill_scores(len(built[split]["qids"]), len(built[split]["cand"]), built[split]["short"], keep, pred)
    choice = decode_grid(
        built["train"]["qids"], built["train"]["cand"], scored["train"], built["train"]["marker_counts"],
        train_gold, built["train"]["self_mask"], built["train"]["after"],
    )
    dev_pred = apply_decoder(built["dev"], scored["dev"], built["dev"]["marker_counts"], choice)
    report = half_scores(dev_gold, dev_pred, built["dev"]["qids"])
    ref = r2["results"]["citator"]["dev"]
    ok = report["A"]["f1"] > ref["A"]["f1"] and report["B"]["f1"] > ref["B"]["f1"]
    print("[r3] dev", {k: round(report[k]["f1"], 4) for k in ("all", "A", "B")}, "beats R2 both halves", ok, flush=True)
    if ok:
        from p7_common import check_submission
        test_pred = apply_decoder(built["test"], scored["test"], built["test"]["marker_counts"], choice)
        check_submission(test_pred, built["test"]["qids"], built["test"]["cand"])
        named_zip("P7_R3_ltr_lorma_citator", test_pred)
    else:
        print("[r3] gate failed, no zip", flush=True)
    save_metrics("r3", {"dev": report, "ref": {k: ref[k]["f1"] for k in ("all", "A", "B")}, "gate": ok,
                        "decoder": {"kind": choice[1], "knob": choice[2]}})


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("mine", "all"):
        mine_pairs()
    if stage in ("train", "all"):
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1200)
    if stage in ("score", "all"):
        device = "cuda"
        model, head = load_model(device)
        tok, bos, pad = load_tokenizer()
        for split in ("Train", "Dev", "Test"):
            score_split(split, model, head, tok, bos, pad, device)
        fuse()


if __name__ == "__main__":
    main()
