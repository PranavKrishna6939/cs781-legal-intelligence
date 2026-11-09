"""Chunk vectors and a document transformer over the whole judgment."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from saul.cjpe import exp_dir, p4_dir
from saul.cjpe.const import SEED
from saul.cjpe.data import collate, load_split, view_chunks
from saul.cjpe.model import load_cjpe, load_overlay, logit_adjusted_ce
from saul.cjpe.score import macro_scores, predict_logits
from saul.utils.io import ensure_dir


def fourier_features(x: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    freqs = torch.arange(n_freq, device=x.device, dtype=torch.float32)
    ang = x.float().unsqueeze(-1) * (2.0 ** freqs) * math_pi()
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def math_pi() -> float:
    return 3.141592653589793


class DocModel(nn.Module):
    def __init__(self, in_dim: int = 2048, d_model: int = 256, n_layers: int = 2, n_heads: int = 4,
                 max_chunks: int = 64, dropout: float = 0.2, n_freq: int = 8):
        super().__init__()
        self.max_chunks = max_chunks
        self.n_freq = n_freq
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(max_chunks, d_model)
        self.freq = nn.Linear(n_freq * 2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.score = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, vectors: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.proj(vectors)
        lengths = mask.long().sum(1)
        index = torch.arange(vectors.shape[1], device=vectors.device)[None, :]
        from_end = (lengths[:, None] - 1 - index).clamp(min=0, max=self.max_chunks - 1)
        hidden = hidden + self.pos(from_end)
        hidden = hidden + self.freq(fourier_features(from_end.float() / max(vectors.shape[1], 1), self.n_freq))
        hidden = self.enc(hidden, src_key_padding_mask=~mask)
        weights = self.score(hidden).squeeze(-1).float()
        weights = weights.masked_fill(~mask, torch.finfo(torch.float32).min / 4)
        alpha = torch.softmax(weights, dim=-1)
        pooled = torch.einsum("bc,bcd->bd", alpha.to(hidden.dtype), hidden)
        return self.fc(self.drop(pooled)), alpha


def section_mask(mask: torch.Tensor, kind: str, k: int) -> torch.Tensor:
    """Keep a subset of the real chunks. `mask` is True on real chunks."""
    out = torch.zeros_like(mask)
    for row in range(mask.shape[0]):
        idx = mask[row].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if kind == "first":
            keep = idx[:k]
        elif kind == "last":
            keep = idx[-k:]
        elif kind == "middle":
            start = max(0, (idx.numel() - k) // 2)
            keep = idx[start:start + k]
        elif kind == "every_other":
            keep = idx[::2]
        else:
            raise ValueError(kind)
        out[row, keep] = True
    return out


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def encode_exp(exp: str, model_name: str, technique: str, context: str | None, init: str | None,
               batch_tokens: int = 32768, attn: str | None = None) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_cjpe(model_name, technique, context, init=init, device=device, ckpt_every=0, attn=attn)
    blob = torch.load(os.path.join(exp_dir(exp), "overlay.pt"), map_location="cpu", weights_only=False)
    load_overlay(model, blob)
    model.to(device)
    model.eval()
    dest = os.path.join(exp_dir(exp), "chunks")
    ensure_dir(dest)
    causal = model.context == "causal"
    for name in ("pool", "dsel", "test"):
        split = load_split(name)
        sequences, owners, lengths = [], [], []
        counts = []
        for i in range(len(split)):
            chunks = view_chunks(split.tokens_of(i), 512, 64, causal)
            counts.append(len(chunks))
            for chunk in chunks:
                sequences.append(chunk)
                owners.append(i)
                lengths.append(len(chunk))
        dim = None
        pieces = [[] for _ in range(len(split))]
        # Chunks are at most 514 tokens (512 body + bos, + cls when causal). The
        # training buckets start at 1024, which would pad every chunk to 1024.
        width = 576
        if lengths and max(lengths) > width:
            raise ValueError(f"chunk length {max(lengths)} exceeds encode width {width}")
        step = max(1, batch_tokens // width)
        batches = [list(range(i, min(i + step, len(sequences)))) for i in range(0, len(sequences), step)]
        for batch_idx in batches:
            batch = collate(
                [sequences[i] for i in batch_idx], [0] * len(batch_idx), [str(i) for i in batch_idx], width,
            )
            batch = _move(batch, device)
            hidden = model.hidden(batch["input_ids"], batch["doc_ids"])
            attn, mean, _last = model.head.parts(hidden, batch["doc_ids"])
            vec = torch.cat([attn, mean], dim=-1).float().cpu().numpy()
            dim = vec.shape[-1]
            for row, index in enumerate(batch_idx):
                pieces[owners[index]].append(vec[row])
        flat = []
        offsets = [0]
        for rows in pieces:
            stacked = np.stack(rows).astype(np.float16) if rows else np.zeros((0, dim or 1), dtype=np.float16)
            flat.append(stacked)
            offsets.append(offsets[-1] + stacked.shape[0])
        vectors = np.concatenate(flat) if flat else np.zeros((0, dim or 1), dtype=np.float16)
        np.savez(
            os.path.join(dest, f"{name}.npz"),
            vectors=vectors,
            offsets=np.asarray(offsets, dtype=np.int64),
            ids=np.asarray(split.ids),
            labels=split.labels,
            sources=np.asarray(split.sources),
            years=split.years,
            halves=np.asarray(split.halves),
        )
        print(f"[hier] encoded {name} chunks={offsets[-1]} dim={dim}", flush=True)
    with open(os.path.join(dest, "done.json"), "w", encoding="utf-8") as handle:
        json.dump({"exp": exp}, handle)
    return dest


def _load_vectors(path: str) -> dict:
    blob = np.load(path, allow_pickle=True)
    offsets = blob["offsets"]
    vectors = blob["vectors"]
    rows = [vectors[int(offsets[i]):int(offsets[i + 1])] for i in range(len(offsets) - 1)]
    return {
        "rows": rows,
        "ids": [str(i) for i in blob["ids"].tolist()],
        "labels": np.asarray(blob["labels"]),
        "sources": [str(s) for s in blob["sources"].tolist()],
        "years": np.asarray(blob["years"]),
        "halves": [str(h) for h in blob["halves"].tolist()],
    }


def _pad(rows: list[np.ndarray], indexes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    picked = [rows[i] for i in indexes]
    width = max(1, max(r.shape[0] for r in picked))
    dim = picked[0].shape[1]
    data = torch.zeros(len(picked), width, dim)
    mask = torch.zeros(len(picked), width, dtype=torch.bool)
    for i, row in enumerate(picked):
        n = row.shape[0]
        if n:
            data[i, :n] = torch.from_numpy(row.astype(np.float32))
            mask[i, :n] = True
    return data, mask


def _run_epoch(model, opt, rows, labels, sources, indexes, ovl_weight: int, train: bool, device: str, seed: int = SEED):
    model.train(train)
    order = indexes[:]
    rng = np.random.default_rng(seed)
    if train:
        expanded = []
        for i in order:
            reps = ovl_weight if sources[i] == "ovl" else 1
            expanded.extend([i] * reps)
        rng.shuffle(expanded)
        order = expanded
    total = correct = 0
    losses = []
    pred_logits = {}
    batch = 16
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        data, mask = _pad(rows, chunk)
        data, mask = data.to(device), mask.to(device)
        y = torch.tensor([int(labels[i]) for i in chunk], device=device)
        with torch.set_grad_enabled(train):
            logits, _alpha = model(data, mask)
            # D-ovl is already repeated `ovl_weight` times above. Do not weight the loss again.
            loss = logit_adjusted_ce(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        losses.append(float(loss.detach()))
        got = logits.argmax(-1).detach().cpu().numpy()
        gold = y.cpu().numpy()
        correct += int((got == gold).sum())
        total += len(chunk)
        for i, row in enumerate(chunk):
            pred_logits[row] = logits[i].detach().float().cpu().numpy()
    return float(np.mean(losses)) if losses else 0.0, correct / max(total, 1), pred_logits


def train_doc(vectors: str, name: str, reference: str | None, fit_ids: str | None, epochs: int = 15,
              seeds: int = 5, ovl_weight: int = 2) -> dict:
    pool = _load_vectors(os.path.join(exp_dir(vectors), "chunks", "pool.npz"))
    dsel = _load_vectors(os.path.join(exp_dir(vectors), "chunks", "dsel.npz"))
    test = _load_vectors(os.path.join(exp_dir(vectors), "chunks", "test.npz"))
    fit = list(range(len(pool["ids"])))
    if fit_ids:
        allowed = set(json.load(open(fit_ids, encoding="utf-8")))
        fit = [i for i, case_id in enumerate(pool["ids"]) if case_id in allowed]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = int(pool["rows"][0].shape[1])
    directory = exp_dir(name)
    ensure_dir(directory)
    accums, test_accums, pool_accums = [], [], []
    train_acc = 0.0
    dev_scores = []
    last_state = None
    for seed in range(seeds):
        torch.manual_seed(SEED + seed)
        model = DocModel(in_dim=in_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        best, best_state, best_dev = -1.0, None, None
        for epoch in range(epochs):
            _run_epoch(model, opt, pool["rows"], pool["labels"], pool["sources"], fit, ovl_weight, True, device, SEED + seed * 100 + epoch)
            _loss, _acc, dev_map = _run_epoch(
                model, opt, dsel["rows"], dsel["labels"], dsel["sources"], list(range(len(dsel["ids"]))),
                1, False, device,
            )
            logits = np.stack([dev_map[i] for i in range(len(dsel["ids"]))])
            score = macro_scores(dsel["labels"], predict_logits(logits))["f1"]
            if score > best:
                best = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_dev = logits
        model.load_state_dict(best_state)
        last_state = best_state
        _loss, acc, _train_map = _run_epoch(
            model, opt, pool["rows"], pool["labels"], pool["sources"], fit, 1, False, device,
        )
        train_acc += acc
        dev_scores.append(best)
        accums.append(np.asarray(best_dev, dtype=np.float64))
        _loss, _acc, test_map = _run_epoch(
            model, opt, test["rows"], np.zeros(len(test["ids"]), dtype=np.int64), test["sources"],
            list(range(len(test["ids"]))), 1, False, device,
        )
        test_accums.append(np.stack([test_map[i] for i in range(len(test["ids"]))]).astype(np.float64))
        _loss, _acc, pool_map = _run_epoch(
            model, opt, pool["rows"], pool["labels"], pool["sources"], list(range(len(pool["ids"]))), 1, False, device,
        )
        pool_accums.append(np.stack([pool_map[i] for i in range(len(pool["ids"]))]).astype(np.float64))
        print(f"[hier] {name} seed {seed} dev {best:.4f}", flush=True)
    mean_logits = np.mean(accums, axis=0)
    dev_f1 = macro_scores(dsel["labels"], predict_logits(mean_logits))["f1"]
    halves = np.asarray(dsel["halves"])
    preds = predict_logits(mean_logits)
    f1_a = macro_scores(dsel["labels"][halves == "A"], preds[halves == "A"])["f1"]
    f1_b = macro_scores(dsel["labels"][halves == "B"], preds[halves == "B"])["f1"]
    test_logits = np.mean(test_accums, axis=0)
    pool_logits = np.mean(pool_accums, axis=0)
    if last_state is not None:
        torch.save(last_state, os.path.join(directory, "doc_model.pt"))
    np.savez(os.path.join(directory, "logits_dsel.npz"), ids=np.asarray(dsel["ids"]), logits=mean_logits, labels=dsel["labels"])
    np.savez(os.path.join(directory, "logits_test.npz"), ids=np.asarray(test["ids"]), logits=test_logits)
    np.savez(os.path.join(directory, "logits_pool.npz"), ids=np.asarray(pool["ids"]), logits=pool_logits, labels=pool["labels"])
    ref = None
    ref_path = os.path.join(exp_dir(reference), "done.json") if reference else ""
    if reference and os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding="utf-8")).get("dev_f1")
    train_acc /= max(seeds, 1)
    holdout_needed = bool(train_acc > 0.97 and ref is not None and dev_f1 < ref)
    blob = {
        "exp": name, "dev_f1": dev_f1, "dev_f1_a": f1_a, "dev_f1_b": f1_b,
        "train_acc": train_acc, "seeds": seeds, "holdout_needed": holdout_needed,
        "reference_f1": ref, "loss": "logit_adjust",
    }
    with open(os.path.join(directory, "done.json"), "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)
    _section_study(vectors, name, in_dim, device)
    print(f"[hier] {name} dev {dev_f1:.4f} holdout_needed={holdout_needed}", flush=True)
    return blob


def _section_study(vectors: str, name: str, in_dim: int, device: str) -> None:
    dsel = _load_vectors(os.path.join(exp_dir(vectors), "chunks", "dsel.npz"))
    pool = _load_vectors(os.path.join(exp_dir(vectors), "chunks", "pool.npz"))
    report = {}
    attention = _mean_attention(os.path.join(exp_dir(name), "doc_model.pt"), dsel, in_dim, device)
    if attention is None:
        attention = []
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(pool["ids"]), size=min(4000, len(pool["ids"])), replace=False)
    pool = {
        key: ([pool[key][i] for i in take] if isinstance(pool[key], list) else pool[key][take])
        for key in pool
    }
    for kind in ("first", "last", "middle", "every_other"):
        widths = (1,) if kind == "every_other" else (1, 2, 4, 8, 16)
        for k in widths:
            torch.manual_seed(SEED)
            model = DocModel(in_dim=in_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            score = 0.0
            for _epoch in range(8):
                _train_masked(model, opt, pool, kind, k, device)
                score = _eval_masked(model, dsel, kind, k, device)
            report[f"{kind}_{k}"] = score
    with open(os.path.join(exp_dir(name), "sections.json"), "w", encoding="utf-8") as handle:
        json.dump({"scores": report, "attention_from_end": attention}, handle, indent=2)


def _train_masked(model, opt, pool, kind, k, device):
    indexes = list(range(len(pool["ids"])))
    data, mask = _pad(pool["rows"], indexes[:256] if len(indexes) > 256 else indexes)
    # A cheap single-batch update is not the study. Walk the pool in batches and mask.
    for start in range(0, len(indexes), 16):
        chunk = indexes[start:start + 16]
        data, mask = _pad(pool["rows"], chunk)
        mask = section_mask(mask, kind, k)
        if not mask.any():
            continue
        data, mask = data.to(device), mask.to(device)
        y = torch.tensor([int(pool["labels"][i]) for i in chunk], device=device)
        opt.zero_grad(set_to_none=True)
        logits, _alpha = model(data, mask)
        loss = logit_adjusted_ce(logits, y)
        loss.backward()
        opt.step()


def _eval_masked(model, dsel, kind, k, device) -> float:
    model.eval()
    got = []
    with torch.no_grad():
        indexes = list(range(len(dsel["ids"])))
        for start in range(0, len(indexes), 16):
            chunk = indexes[start:start + 16]
            data, mask = _pad(dsel["rows"], chunk)
            mask = section_mask(mask, kind, k)
            data, mask = data.to(device), mask.to(device)
            logits, _alpha = model(data, mask)
            got.append(logits.float().cpu().numpy())
    logits = np.concatenate(got)
    return macro_scores(dsel["labels"], predict_logits(logits))["f1"]


def _mean_attention(path, dsel, in_dim, device):
    if not os.path.exists(path):
        return None
    model = DocModel(in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    totals = np.zeros(64, dtype=np.float64)
    counts = np.zeros(64, dtype=np.float64)
    with torch.no_grad():
        indexes = list(range(len(dsel["ids"])))
        for start in range(0, len(indexes), 16):
            chunk = indexes[start:start + 16]
            data, mask = _pad(dsel["rows"], chunk)
            data, mask = data.to(device), mask.to(device)
            _logits, alpha = model(data, mask)
            alpha = alpha.float().cpu().numpy()
            real = mask.cpu().numpy()
            for row in range(alpha.shape[0]):
                idx = np.flatnonzero(real[row])
                for pos, j in enumerate(idx):
                    from_end = len(idx) - 1 - pos
                    if from_end < 64:
                        totals[from_end] += alpha[row, j]
                        counts[from_end] += 1
    return (totals / np.maximum(counts, 1)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--exp", default="A1")
    parser.add_argument("--model", default="dense_v1")
    parser.add_argument("--technique", default="lorma")
    parser.add_argument("--context", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--vectors", default="A1")
    parser.add_argument("--name", default="A3")
    parser.add_argument("--reference", default="A1")
    parser.add_argument("--fit-ids", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ovl-weight", type=int, default=2)
    parser.add_argument("--attn", default=None)
    args = parser.parse_args()
    if args.encode:
        encode_exp(args.exp, args.model, args.technique, args.context, args.init, attn=args.attn)
    if args.train:
        train_doc(args.vectors, args.name, args.reference, args.fit_ids, args.epochs, args.seeds, args.ovl_weight)


if __name__ == "__main__":
    main()
