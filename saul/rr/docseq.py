"""Document-level sequence layer over cached window posteriors (stage 2).

The window tagger sees about 45 sentences. Structure that spans a whole document,
such as a separate member's order appended after the majority's ruling, is out of
its reach. This layer runs a BiGRU over every sentence of a document. Its inputs are
the members' posteriors, document position and domain only: no text rules.

Training uses out-of-fold posteriors only, so the inputs are never memorised. The
outer 5 folds are the stage-1 folds. Fold k's stage-2 model never sees fold k's
documents, and neither did the stage-1 models that produced its inputs.

Test inputs are per-fold stage-1 posteriors (the same kind of input as training),
and the predictions are averaged over stage-1 folds and stage-2 models.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from saul.rr.data import LABEL_ID, LABELS, N_LABELS, labeled_docs, load_docs, p3_dir
from saul.rr.infer import load_probs, save_npz
from saul.rr.score import macro_scores

N_FREQ = 4


def _position(n: int) -> np.ndarray:
    rel = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    cols = [rel[:, None]]
    for k in range(N_FREQ):
        cols.append(np.sin(np.pi * (2 ** k) * rel)[:, None])
        cols.append(np.cos(np.pi * (2 ** k) * rel)[:, None])
    return np.concatenate(cols, axis=1)


def features(probs: list[np.ndarray], domain: str, running: bool = False) -> np.ndarray:
    n = probs[0].shape[0]
    dom = np.zeros((n, 2))
    dom[:, 0 if domain == "CL" else 1] = 1.0
    logs = [np.log(np.clip(p, 1e-6, 1.0)) / 6.0 for p in probs]
    cols = list(probs) + logs + [_position(n), dom]
    if running:
        mean = np.mean(np.stack(probs, axis=0), axis=0)
        ruling = mean[:, LABEL_ID["RulingByPresentCourt"]]
        seen = np.maximum.accumulate(np.concatenate([[0.0], ruling[:-1]]))
        remain = 1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)
        cols.append(np.stack([seen, remain], axis=1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def _prefix(doc_id: str) -> str:
    return doc_id.split("_", 1)[0]


def augment_fold(probs: dict[str, list[np.ndarray]], labels: dict[str, np.ndarray],
                 domains: dict[str, str], train_ids: list[str], n: int, rng: np.random.Generator,
                 running: bool = False, long_tail: bool = False, long_prob: float = 0.5
                 ) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Synthetic appended orders, drawn only from this outer fold's training ids.

    Document A is a CCI or COM training document and keeps its gold labels. The
    first 10–60% of a different training document B is appended and labelled Dissent.
    With long_tail, half of the tails are 200–400 sentences from a long training
    document, because the real dissents are not all short.
    """
    train = list(train_ids)
    hosts = [i for i in train if _prefix(i) in ("CCI", "COM")]
    if len(hosts) == 0 or len(train) < 2 or n <= 0:
        return []
    dissent = LABEL_ID["Dissent"]
    long_ids = [i for i in train if probs[i][0].shape[0] >= 200]
    out = []
    for _ in range(n):
        a = hosts[int(rng.integers(len(hosts)))]
        others = [i for i in train if i != a]
        b = others[int(rng.integers(len(others)))]
        body = probs[b][0].shape[0]
        use_long = long_tail and long_ids and float(rng.random()) < long_prob
        if use_long:
            pool = [i for i in long_ids if i != a] or [i for i in long_ids]
            b = pool[int(rng.integers(len(pool)))]
            body = probs[b][0].shape[0]
            lo = min(200, body)
            hi = min(body, 400)
            take = int(rng.integers(lo, hi + 1))
        else:
            take = max(1, int(round(body * float(rng.uniform(0.1, 0.6)))))
            take = min(take, body)
        joined = [np.concatenate([p_a, p_b[:take]], axis=0) for p_a, p_b in zip(probs[a], probs[b])]
        y = np.concatenate([labels[a], np.full(take, dissent, dtype=np.int64)])
        out.append((features(joined, domains[a], running=running), y, a))
    return out


class DocSeq(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.2,
                 n_types: int = 0, type_dim: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, type_dim) if n_types else None
        width = d_in + (type_dim if n_types else 0)
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, N_LABELS))

    def forward(self, x: torch.Tensor, type_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.type_emb is not None:
            extra = self.type_emb(type_id)[:, None, :].expand(-1, x.shape[1], -1)
            x = torch.cat([x, extra], dim=-1)
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def _weights(docs) -> torch.Tensor:
    counts = np.ones(N_LABELS)
    for doc in docs:
        for lab in doc.label_ids:
            counts[lab] += 1
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(xs: list[np.ndarray], ys: list[np.ndarray], weights: torch.Tensor, seed: int,
                epochs: int = 60, lr: float = 2e-3, device: torch.device | None = None,
                type_ids: list[int] | None = None, n_types: int = 0) -> DocSeq:
    device = device or _device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DocSeq(xs[0].shape[1], n_types=n_types).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    model.train()
    last = None
    for epoch in range(epochs):
        for i in rng.permutation(len(xs)):
            x = torch.from_numpy(xs[i])[None].to(device)
            y = torch.from_numpy(ys[i]).to(device)
            tid = None if type_ids is None else torch.tensor([type_ids[i]], device=device)
            loss = loss_fn(model(x, tid)[0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = loss
        if epoch % 20 == 0 or epoch + 1 == epochs:
            print(f"[docseq] seed-epoch {epoch} loss {float(last.detach()):.4f} device {device}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model: DocSeq, x: np.ndarray, type_id: int = 0) -> np.ndarray:
    device = next(model.parameters()).device
    tid = None if model.type_emb is None else torch.tensor([type_id], device=device)
    logits = model(torch.from_numpy(x)[None].to(device), tid)[0]
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _oof_tables(exps: list[str]) -> dict[str, list[np.ndarray]]:
    tables = {}
    for exp in exps:
        for k in range(5):
            folder = os.path.join(p3_dir(), exp, f"fold{k}", "oof")
            for name in os.listdir(folder):
                if name.endswith(".npz"):
                    tables.setdefault(name[:-4], []).append(load_probs(os.path.join(folder, name)))
    return {k: v for k, v in tables.items() if len(v) == len(exps)}


def _fold_test(exps: list[str], k: int, doc_id: str) -> list[np.ndarray]:
    return [load_probs(os.path.join(p3_dir(), exp, f"fold{k}", "test", f"{doc_id}.npz")) for exp in exps]


def _prefix_index(doc_ids: list[str]) -> dict[str, int]:
    prefixes = sorted({_prefix(i) for i in doc_ids})
    return {p: i + 1 for i, p in enumerate(prefixes)}


def run(exps: list[str], name: str, seeds: int = 3, epochs: int = 60, doc_type: bool = False,
        running: bool = False, augment: int = 0, long_tail: bool = False, long_prob: float = 0.5) -> dict:
    docs = load_docs()
    labeled = labeled_docs(docs)
    by_id = {d.doc_id: d for d in labeled}
    test_docs = [d for d in docs if d.split == "test"]
    folds = json.load(open(os.path.join(p3_dir(), "folds.json"), encoding="utf-8"))
    table = _oof_tables(exps)
    feats = {i: features(table[i], by_id[i].domain, running=running) for i in table if i in by_id}
    labels = {i: np.array(by_id[i].label_ids, dtype=np.int64) for i in feats}
    domains = {i: by_id[i].domain for i in feats}
    prefix_of = _prefix_index(list(by_id))
    n_types = (max(prefix_of.values()) + 1) if doc_type else 0

    def type_id(doc_id: str) -> int:
        return prefix_of.get(_prefix(doc_id), 0)

    root = os.path.join(p3_dir(), name)
    oof_probs: dict[str, np.ndarray] = {}
    test_acc = {d.doc_id: None for d in test_docs}
    n_test = 0
    for k, held in enumerate(folds):
        held = set(held)
        train_ids = [i for i in feats if i not in held]
        print(f"[docseq] outer fold {k} training on {len(train_ids)} documents", flush=True)
        weights = _weights([by_id[i] for i in train_ids])
        xs = [feats[i] for i in train_ids]
        ys = [labels[i] for i in train_ids]
        type_ids = [type_id(i) for i in train_ids]
        if augment:
            extra = augment_fold(table, labels, domains, train_ids, augment,
                                 np.random.default_rng(781 + k), running=running,
                                 long_tail=long_tail, long_prob=long_prob)
            for x, y, src in extra:
                xs.append(x)
                ys.append(y)
                type_ids.append(type_id(src))
            print(f"[docseq] outer fold {k} added {len(extra)} synthetic documents", flush=True)
        models = [
            train_model(xs, ys, weights, seed=781 + 17 * k + s, epochs=epochs,
                        type_ids=type_ids if doc_type else None, n_types=n_types)
            for s in range(seeds)
        ]
        for i in held:
            if i in feats:
                oof_probs[i] = np.mean([predict(m, feats[i], type_id(i)) for m in models], axis=0)
        for doc in test_docs:
            for kk in range(5):
                x = features(_fold_test(exps, kk, doc.doc_id), doc.domain, running=running)
                p = np.mean([predict(m, x, type_id(doc.doc_id)) for m in models], axis=0)
                test_acc[doc.doc_id] = p if test_acc[doc.doc_id] is None else test_acc[doc.doc_id] + p
        n_test += 5
        print(f"[docseq] outer fold {k} done", flush=True)
    oof_dir = os.path.join(root, "fold0", "oof")
    test_dir = os.path.join(root, "test")
    os.makedirs(oof_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for i, p in oof_probs.items():
        save_npz(os.path.join(oof_dir, f"{i}.npz"), p, np.ones(p.shape[0]))
    for i, acc in test_acc.items():
        p = acc / n_test
        save_npz(os.path.join(test_dir, f"{i}.npz"), p, np.ones(p.shape[0]))
    with open(os.path.join(test_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"members": exps, "seeds": seeds, "epochs": epochs, "doc_type": doc_type,
                   "running": running, "augment": augment, "long_tail": long_tail}, f)
    gold, pred, base = [], [], []
    for i, p in oof_probs.items():
        gold.extend(labels[i].tolist())
        pred.extend(p.argmax(1).tolist())
        base.extend(np.mean(table[i], axis=0).argmax(1).tolist())
    after = macro_scores(gold, pred)
    before = macro_scores(gold, base)
    summary = {
        "members": exps,
        "oof_mix_argmax": before["macro_f1"],
        "oof_docseq_argmax": after["macro_f1"],
        "doc_type": doc_type,
        "running": running,
        "augment": augment,
        "long_tail": long_tail,
        "per_label": {LABELS[c]: after["per_label"][c] for c in range(N_LABELS)},
        "per_label_mix": {LABELS[c]: before["per_label"][c] for c in range(N_LABELS)},
    }
    with open(os.path.join(root, "docseq.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[docseq] OOF argmax mix {before['macro_f1']:.4f} -> docseq {after['macro_f1']:.4f}", flush=True)
    for c in range(N_LABELS):
        print(f"[docseq]   {LABELS[c]:24s} {before['per_label'][c]:.3f} -> {after['per_label'][c]:.3f}")
    d = LABEL_ID["Dissent"]
    for doc in test_docs:
        p = test_acc[doc.doc_id] / n_test
        hits = int((p.argmax(1) == d).sum())
        if hits:
            print(f"[docseq] test {doc.doc_id[:60]} Dissent argmax on {hits} of {p.shape[0]}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members", default="E1,E2,E3")
    p.add_argument("--name", default="P3_docseq_E1E2E3")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--doc-type", action="store_true")
    p.add_argument("--running", action="store_true")
    p.add_argument("--augment", type=int, default=0)
    p.add_argument("--long-tail", action="store_true")
    p.add_argument("--long-prob", type=float, default=0.5)
    args = p.parse_args()
    torch.set_num_threads(8)
    run([x for x in args.members.split(",") if x], args.name, args.seeds, args.epochs,
        doc_type=args.doc_type, running=args.running, augment=args.augment,
        long_tail=args.long_tail, long_prob=args.long_prob)


if __name__ == "__main__":
    main()
