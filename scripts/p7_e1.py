"""E1 LoRMA (and E3 IA3) bi-encoder: rare-span pretraining, then multi-positive InfoNCE."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781")))

from p7_bm25 import micro_f1
from p7_common import (
    OUT,
    apply_mask,
    citation_contexts,
    decode,
    gold_of,
    half_scores,
    masks,
    passages,
    save_metrics,
    split_bundle,
    zrows,
)
from p7_model import (
    Head,
    build,
    checkpoint_every,
    collate,
    cosine,
    encode_ids,
    forward_hidden,
    gpu_mb,
    load_tokenizer,
    pool,
    unwrap_lorma,
)

TEMP = 0.05
SLOT_TAU = 0.1


def gram_hash(words, start: int) -> int:
    value = 1469598103934665603
    for word in words[start:start + 12]:
        value ^= hash(word) & 0xFFFFFFFFFFFFFFFF
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def mine(limit: int = 30000) -> list[tuple[str, str]]:
    docs = []
    for split in ("Train", "Dev"):
        bundle = split_bundle(split)
        for sents in bundle["sents"]:
            docs.append(" ".join(sents).lower().split())
    buckets = 1 << 22
    mask = buckets - 1
    table = np.zeros(buckets, dtype=np.uint8)
    for doc_i, words in enumerate(docs):
        seen = set()
        for start in range(0, len(words) - 11):
            key = gram_hash(words, start) & mask
            if key in seen or table[key] >= 5:
                continue
            seen.add(key)
            table[key] += 1
        if doc_i % 500 == 0:
            print(f"[e1] mine count {doc_i}/{len(docs)}", flush=True)
    locs: dict[int, list[tuple[int, int]]] = {}
    for doc_i, words in enumerate(docs):
        seen = set()
        for start in range(0, len(words) - 11):
            key = gram_hash(words, start) & mask
            if key in seen or not (2 <= table[key] <= 4):
                continue
            seen.add(key)
            bucket = locs.setdefault(key, [])
            if len(bucket) < 2:
                bucket.append((doc_i, start))
    rare = [key for key, places in locs.items() if len(places) >= 2]
    rng = random.Random(781)
    rng.shuffle(rare)
    pairs = []
    for key in rare:
        (d1, i1), (d2, i2) = locs[key][:2]
        left = docs[d1][max(0, i1 - 40):i1] + ["[MASK]"] + docs[d1][i1 + 12:i1 + 60]
        right = docs[d2][max(0, i2 - 30):i2 + 110]
        pairs.append(("context: " + " ".join(left), "passage: " + " ".join(right)))
        if len(pairs) >= limit:
            break
    crops = []
    for words in docs:
        if len(words) < 80:
            continue
        for _ in range(2):
            a = rng.randrange(0, max(1, len(words) - 60))
            b = rng.randrange(0, max(1, len(words) - 60))
            crops.append((
                "passage: " + " ".join(words[a:a + 100]),
                "passage: " + " ".join(words[b:b + 100]),
            ))
            if len(crops) >= limit:
                break
        if len(crops) >= limit:
            break
    print(f"[e1] pairs {len(pairs)} crops {len(crops)}", flush=True)
    return pairs + crops


def prepare(bundle, topn: int) -> list[dict]:
    path = os.path.join(OUT, "e0", "train.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; E0 has to finish first")
    saved = np.load(path, allow_pickle=True)
    q_index = {q: i for i, q in enumerate(saved["qids"].tolist())}
    c_index = {c: i for i, c in enumerate(saved["cand"].tolist())}
    full = saved["full"]
    gold = gold_of(bundle["queries"])
    cited = {}
    for q, ids in gold.items():
        for cid in ids:
            cited.setdefault(cid, set()).add(q)
    cand_passages = {cid: passages(sents) for cid, sents in zip(bundle["cand"], bundle["sents"])}
    q_sents = {q: sents for q, sents in zip(bundle["qids"], bundle["q_sents"])}
    rows = []
    for q in bundle["qids"]:
        ctx = citation_contexts(q_sents[q]) or ["context: " + " ".join(q_sents[q][:6])]
        q_pass = passages(q_sents[q])[:8]
        scores = full[q_index[q]]
        order = np.argsort(-scores)
        gold_ids = [c for c in gold[q] if c in cand_passages and c != q]
        negs = []
        for col in order:
            cid = bundle["cand"][col] if False else saved["cand"][col]
            cid = str(cid)
            if cid == q or cid in gold_ids or cid not in cand_passages:
                continue
            negs.append(cid)
            if len(negs) >= 12:
                break
        mates = [c for c, users in cited.items() if q not in users and c in cand_passages and c not in gold_ids]
        rng = random.Random(hash(q) & 0xFFFFFFFF)
        rng.shuffle(mates)
        rows.append({
            "qid": q,
            "contexts": ctx[:12],
            "q_passages": q_pass,
            "gold": gold_ids,
            "negs": (negs[:8] + mates[:8])[:12],
            "passages": cand_passages,
        })
    return rows


def chosen_passages(row, topn: int) -> list[str]:
    """Distant evidence: passages of a candidate that share the most words with the citation contexts."""
    ctx_words = set()
    for text in row["contexts"]:
        ctx_words.update(text.lower().split())
    out = {}
    cands = row["gold"][:4] + row["negs"]
    for cid in cands:
        ranked = sorted(
            row["passages"][cid],
            key=lambda text: len(ctx_words & set(text.lower().split())),
            reverse=True,
        )
        out[cid] = ranked[:topn] or ranked[:1]
    return out


def evidence_rate(rows, topn: int) -> float:
    hit = tot = 0
    for row in rows[:200]:
        ctx = set()
        for text in row["contexts"]:
            words = text.lower().split()
            ctx.update(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
        picked = chosen_passages(row, topn)
        for cid in row["gold"][:3]:
            tot += 1
            ok = False
            for text in picked.get(cid, []):
                words = text.lower().split()
                if any(tuple(words[i:i + 8]) in ctx for i in range(max(0, len(words) - 7))):
                    ok = True
                    break
            hit += int(ok)
    return hit / max(tot, 1)


def embed(model, head, tok, bos, pad, texts, device, indexes_mode="marker"):
    encoded = [encode_ids(tok, text, bos, 160) for text in texts]
    ids = collate([row for row, _ in encoded], pad, device)
    indexes = [index if indexes_mode == "marker" else None for _, index in encoded]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        hidden = forward_hidden(model, ids)
    return head(pool(hidden, ids, pad, indexes))


def info_nce(anchor, positive) -> torch.Tensor:
    logits = anchor @ positive.T / TEMP
    return F.cross_entropy(logits, torch.arange(anchor.shape[0], device=anchor.device))


def supervised_loss(model, head, tok, bos, pad, batch, device, topn: int) -> torch.Tensor:
    losses = []
    for row in batch:
        picked = row.get("picked") or chosen_passages(row, topn)
        cands = [c for c in row["gold"][:4] + row["negs"] if c in picked]
        if not cands or not any(c in row["gold"] for c in cands):
            continue
        texts = row["contexts"] + row["q_passages"]
        splits = [len(row["contexts"]), len(row["contexts"]) + len(row["q_passages"])]
        groups = []
        for cid in cands:
            groups.append(picked[cid])
            texts.extend(picked[cid])
        vec = embed(model, head, tok, bos, pad, texts, device)
        ctx = vec[:splits[0]]
        qpass = vec[splits[0]:splits[1]]
        offset = splits[1]
        cand_vecs = []
        for group in groups:
            cand_vecs.append(vec[offset:offset + len(group)].mean(0))
            offset += len(group)
        cand = torch.stack(cand_vecs)
        # Passage-mean is the training surrogate for max-over-passages: the two
        # passages were already selected as the evidence, so both should match.
        local = torch.logsumexp(ctx @ cand.T / SLOT_TAU, dim=0) * SLOT_TAU if len(ctx) else cand.new_zeros(len(cands))
        glob = (qpass @ cand.T).mean(0) if len(qpass) else cand.new_zeros(len(cands))
        scores = local + glob
        n_pos = sum(c in row["gold"] for c in cands)
        losses.append(-torch.logsumexp(scores[:n_pos] / TEMP, 0) + torch.logsumexp(scores / TEMP, 0))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def fit_batch(fn, start: int) -> int:
    batch = start
    while batch >= 1:
        try:
            torch.cuda.empty_cache()
            loss = fn(batch)
            loss.backward()
            return batch
        except torch.cuda.OutOfMemoryError:
            print(f"[e1] oom at batch {batch}", flush=True)
            batch //= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise SystemExit("batch 1 does not fit")


def run_epochs(model, head, opt, bases, steps_of, n_items, epochs, device, step_fn, save_path=None, save_meta=None):
    batch = steps_of
    # `steps_of` here is the probe batch size found by the caller; the schedule
    # is built only after that size is known.
    total = epochs * math_ceil(n_items, batch)
    step = 0
    for epoch in range(epochs):
        order = list(range(n_items))
        random.Random(781 + epoch).shuffle(order)
        for start in range(0, n_items, batch):
            chunk = order[start:start + batch]
            # A short final batch changes the flex-attention shape and the
            # inductor compile on this GPU has no kernel for it.
            if len(chunk) != batch:
                continue
            opt.zero_grad(set_to_none=True)
            loss = step_fn(chunk)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            for group, base in zip(opt.param_groups, bases):
                group["lr"] = cosine(step, total, base)
            opt.step()
            step += 1
            if step % 20 == 0 or step == total:
                print(f"[e1] step {step}/{total} loss {float(loss.detach()):.3f} gpu {gpu_mb()}MB", flush=True)
            if save_path and step % 200 == 0:
                save_run(save_path, model, head, save_meta or {})
    return step


def math_ceil(n, batch):
    return (n + batch - 1) // batch


def save_run(path, model, head, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save({"model": tensors, "head": head.state_dict(), **meta}, path)


def load_run(path, model, head, merge_then: str | None = None):
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["model"], strict=False)
    if merge_then:
        unwrap_lorma(model)
        # unwrap leaves the merged base; the caller rebuilds adapters after this
    if head is not None and "head" in blob and merge_then is None:
        head.load_state_dict(blob["head"])
    return blob


def encode_split(model, head, tok, bos, pad, bundle, device, path):
    model.eval()
    head.eval()
    owners = []
    vectors = []
    with torch.no_grad():
        texts = []
        for cand_i, sents in enumerate(bundle["sents"]):
            for passage in passages(sents):
                owners.append(cand_i)
                texts.append(passage)
                if len(texts) >= 4:
                    vectors.append(embed(model, head, tok, bos, pad, texts, device, "mean").float().cpu())
                    texts = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if texts:
            vectors.append(embed(model, head, tok, bos, pad, texts, device, "mean").float().cpu())
    np.savez(
        path,
        vec=torch.cat(vectors).numpy().astype(np.float16),
        owner=np.asarray(owners, dtype=np.int32),
        cand=np.asarray(bundle["cand"]),
    )
    print(f"[e1] encoded {bundle['split']} passages {len(owners)}", flush=True)


def scatter_max(sims: torch.Tensor, owner, n_cand: int) -> torch.Tensor:
    per = torch.full((sims.shape[0], n_cand), -1e4, device=sims.device, dtype=sims.dtype)
    index = torch.as_tensor(owner, device=sims.device, dtype=torch.long)
    per.scatter_reduce_(1, index.view(1, -1).expand(sims.shape[0], -1), sims, reduce="amax", include_self=True)
    return per


def fill_neural(model, head, tok, bos, pad, bundle, device, passage_path):
    if not os.path.exists(passage_path):
        encode_split(model, head, tok, bos, pad, bundle, device, passage_path)
    saved = np.load(passage_path)
    cand_vec = torch.tensor(saved["vec"].astype(np.float32), device=device)
    owner = saved["owner"]
    qids = bundle["qids"]
    n_cand = len(bundle["cand"])
    local = np.zeros((len(qids), n_cand), dtype=np.float32)
    glob = np.zeros_like(local)
    with torch.no_grad():
        for row, sents in enumerate(bundle["q_sents"]):
            ctx = citation_contexts(sents)[:12] or ["context: " + " ".join(sents[:6])]
            qpass = passages(sents)[:8]
            ctx_v = embed(model, head, tok, bos, pad, ctx, device).float()
            q_v = embed(model, head, tok, bos, pad, qpass, device, "mean").float()
            per = scatter_max(ctx_v @ cand_vec.T, owner, n_cand)
            local[row] = (SLOT_TAU * torch.logsumexp(per / SLOT_TAU, dim=0)).float().cpu().numpy()
            glob[row] = scatter_max(q_v @ cand_vec.T, owner, n_cand).mean(0).float().cpu().numpy()
            if row % 40 == 0:
                print(f"[e1] scored {bundle['split']} {row}/{len(qids)}", flush=True)
    np.savez(
        os.path.join(OUT, "e1", f"{bundle['split'].lower()}_neural.npz"),
        L=local, G=glob, qids=np.asarray(qids), cand=np.asarray(bundle["cand"]),
    )
    return local, glob


def score_dev(model, head, tok, bos, pad, device, zip_name: str, metrics_name: str):
    dev = split_bundle("Dev")
    model.eval()
    head.eval()
    L, G = fill_neural(model, head, tok, bos, pad, dev, device, os.path.join(OUT, "e1", "dev_passages.npz"))
    gold = gold_of(dev["queries"])
    qids = dev["qids"]
    self_mask, after = masks(qids, dev["cand"], dev["q_sents"], dev["sents"])
    neural = apply_mask(zrows(L) + zrows(G), self_mask, after)
    e0 = np.load(os.path.join(OUT, "e0", "dev.npz"))
    weights = json.load(open(os.path.join(OUT, "e0", "metrics.json")))["weights"]
    names = ["full", "head", "context", "shingle"]
    lexical = sum(w * zrows(e0[name]) for w, name in zip(weights, names))
    fused = apply_mask(lexical + zrows(L) + zrows(G), self_mask, after)
    kind = json.load(open(os.path.join(OUT, "e0", "metrics.json")))
    pred = decode(qids, dev["cand"], fused, kind["decoder"], None, tuple(kind["mass_coef"]), kind["eps"], 0.5)
    report = half_scores(gold, pred, qids)
    neural_pred = decode(qids, dev["cand"], neural, "column", tau=0.5)
    neural_report = half_scores(gold, neural_pred, qids)
    e0_report = kind["dev"]
    gate = (
        report["A"]["f1"] >= e0_report["A"]["f1"] + 0.02
        and report["B"]["f1"] >= e0_report["B"]["f1"] + 0.02
    )
    save_metrics(metrics_name, {"dev": report, "neural": neural_report, "gate": gate, "gpu_mb": gpu_mb()})
    print(f"[{metrics_name}] dev", {k: round(report[k]["f1"], 4) for k in ("all", "A", "B")}, "gate", gate, flush=True)
    if not gate and metrics_name in ("e1", "e3", "e4"):
        return gate
    test = split_bundle("Test")
    Lt, Gt = fill_neural(model, head, tok, bos, pad, test, device, os.path.join(OUT, "e1", "test_passages.npz"))
    tmask = masks(test["qids"], test["cand"], test["q_sents"], test["sents"])
    lexical_file = np.load(os.path.join(OUT, "e0", "test.npz"))
    lexical_t = sum(w * zrows(lexical_file[name]) for w, name in zip(weights, names))
    fused_t = apply_mask(lexical_t + zrows(Lt) + zrows(Gt), *tmask)
    test_pred = decode(test["qids"], test["cand"], fused_t, kind["decoder"], None, tuple(kind["mass_coef"]), kind["eps"], 0.5)
    from p7_common import check_submission
    from p7_bm25 import write_zip
    check_submission(test_pred, test["qids"], test["cand"])
    write_zip(zip_name, test_pred)
    print(f"[{metrics_name}] wrote {zip_name}", flush=True)
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pretrain", "supervised", "score", "both"), default="both")
    parser.add_argument("--technique", choices=("lorma", "ia3"), default="lorma")
    parser.add_argument("--init", default="")
    parser.add_argument("--merge-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrain-limit", type=int, default=20000)
    args = parser.parse_args()
    os.makedirs(os.path.join(OUT, "e1"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bos, pad = load_tokenizer()
    model = build(args.technique)
    head = Head(model.config.d_model).to(device)
    if args.init and args.merge_init:
        donor = build("lorma")
        blob = load_run(args.init, donor, None, merge_then="ia3")
        merged = donor.state_dict()
        own = model.state_dict()
        for key, value in merged.items():
            target = key
            if target not in own:
                target = key.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
            if target in own and own[target].shape == value.shape:
                own[target] = value
        model.load_state_dict(own)
        if "head" in blob:
            head.load_state_dict(blob["head"])
    elif args.init:
        load_run(args.init, model, head)
    checkpoint_every(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[e1] technique {args.technique} trainable {sum(p.numel() for p in trainable)}", flush=True)
    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": 5e-4}, {"params": list(head.parameters()), "lr": 1e-3}],
        weight_decay=0.01,
    )
    bases = [5e-4, 1e-3]
    if args.stage == "score":
        model.eval()
        head.eval()
        zip_name = "P7_E1" if args.technique == "lorma" else "P7_E3"
        metrics_name = "e1" if args.technique == "lorma" else "e3"
        score_dev(model, head, tok, bos, pad, device, zip_name, metrics_name)
        return
    if args.stage in ("pretrain", "both"):
        pairs = mine(args.pretrain_limit)
        texts = [(a, b) for a, b in pairs]

        def probe(batch):
            chunk = texts[:batch]
            opt.zero_grad(set_to_none=True)
            anchor = embed(model, head, tok, bos, pad, [a for a, _ in chunk], device, "mean")
            positive = embed(model, head, tok, bos, pad, [b for _, b in chunk], device, "mean")
            return info_nce(anchor, positive)

        batch = fit_batch(probe, 64)
        opt.zero_grad(set_to_none=True)
        print(f"[e1] pretrain batch {batch}", flush=True)

        def step_fn(idxs):
            chunk = [texts[i] for i in idxs]
            anchor = embed(model, head, tok, bos, pad, [a for a, _ in chunk], device, "mean")
            positive = embed(model, head, tok, bos, pad, [b for _, b in chunk], device, "mean")
            return info_nce(anchor, positive)

        run_epochs(
            model, head, opt, bases, batch, len(texts), args.epochs, device, step_fn,
            os.path.join(OUT, "e1", "pretrain.pt"), {"stage": "pretrain", "technique": args.technique},
        )
        save_run(os.path.join(OUT, "e1", "pretrain.pt"), model, head, {"stage": "pretrain", "technique": args.technique})
        print("[e1] pretrain saved", flush=True)
    if args.stage in ("supervised", "both"):
        train = split_bundle("Train")
        rows = [row for row in prepare(train, 2) if row["gold"] and row["negs"]]
        rate = evidence_rate(rows, 2)
        topn = 2 if rate >= 0.60 else 4
        print(f"[e1] evidence@{2} {rate:.3f} using top {topn}", flush=True)
        for row in rows:
            row["picked"] = chosen_passages(row, topn)
            row.pop("passages", None)

        def probe(batch):
            opt.zero_grad(set_to_none=True)
            return supervised_loss(model, head, tok, bos, pad, rows[:batch], device, topn)

        batch = fit_batch(probe, 4 if args.technique == "lorma" else 4)
        opt.zero_grad(set_to_none=True)
        print(f"[e1] supervised batch {batch} gpu {gpu_mb()}MB", flush=True)

        def step_fn(idxs):
            return supervised_loss(model, head, tok, bos, pad, [rows[i] for i in idxs], device, topn)

        run_epochs(model, head, opt, bases, batch, len(rows), args.epochs, device, step_fn)
        name = "lorma.pt" if args.technique == "lorma" else "ia3.pt"
        save_run(os.path.join(OUT, "e1", name), model, head, {"stage": "supervised", "technique": args.technique, "topn": topn})
        zip_name = "P7_E1" if args.technique == "lorma" else "P7_E3"
        metrics_name = "e1" if args.technique == "lorma" else "e3"
        score_dev(model, head, tok, bos, pad, device, zip_name, metrics_name)
        print("[e1] supervised done", flush=True)


if __name__ == "__main__":
    main()
