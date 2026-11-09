"""Domain pool, teacher dump, Conv-Mini distillation, TAPT, and silver labels.

The pool is not the Phase-2 pool. It bans 13-grams from all 100 rhetorical-role
documents and refuses to train unless its meta says ban_rr.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from saul.data.loader import mntp_mask
from saul.data.tokens import open_source
from saul.ner.data import load_tokenizer
from saul.ner.kd import _grams
from saul.ner.registry import repo_root, resolve_checkpoint
from saul.rr.data import LABELS, RRDoc, doc_windows, load_docs, p3_dir, windows_for
from saul.rr.infer import accumulate, load_probs, save_npz

KEYWORDS = {
    "CL": ("CCI", "COMPAT", "NCLAT", "Competition Act"),
    "IT": ("ITAT", "Income-tax", "Income Tax"),
}


def pool_dir() -> str:
    return os.path.join(p3_dir(), "pool")


def pool_path() -> str:
    return os.path.join(pool_dir(), "pool_v2.jsonl")


def split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_agreement(docs=None) -> float:
    docs = docs or [d for d in load_docs() if d.split in ("train", "dev")]
    if not docs:
        return 0.0
    ok = 0
    for doc in docs:
        again = split_sentences(" ".join(doc.sentences))
        if again == doc.sentences:
            ok += 1
    rate = ok / len(docs)
    print(f"[split] exact rejoin agreement {ok}/{len(docs)} = {rate:.3f}")
    return rate


def _ban_grams() -> set[tuple]:
    banned: set[tuple] = set()
    for doc in load_docs():
        banned |= _grams(" ".join(doc.sentences))
    return banned


def _domain_of(text: str) -> str | None:
    scores = {dom: sum(text.count(word) for word in words) for dom, words in KEYWORDS.items()}
    if scores["CL"] == 0 and scores["IT"] == 0:
        return None
    if scores["CL"] == scores["IT"]:
        return None
    return "CL" if scores["CL"] > scores["IT"] else "IT"


def build_pool(n_docs: int = 800, max_scan: int = 20000) -> str:
    dest = pool_path()
    meta_path = dest + ".meta.json"
    os.makedirs(pool_dir(), exist_ok=True)
    want = {"ban_rr": True, "n_docs": int(n_docs)}
    if os.path.exists(dest) and os.path.exists(meta_path):
        prev = json.load(open(meta_path, encoding="utf-8"))
        if prev.get("ban_rr") and prev.get("n_docs", 0) >= n_docs and os.path.getsize(dest) > 0:
            print(f"[pool] reuse {dest}")
            return dest
    split_agreement()
    tok = load_tokenizer()
    banned = _ban_grams()
    token_root = os.path.join(repo_root(), "data", "tokens", "legal32k")
    picked = 0
    scanned = 0
    with open(dest, "w", encoding="utf-8") as f:
        for source in ("tribunals", "supreme_court", "high_courts"):
            try:
                store = open_source(token_root, "train", source)
            except FileNotFoundError:
                print(f"[pool] missing {source}")
                continue
            for i in range(store.n_docs):
                if picked >= n_docs or scanned >= max_scan:
                    break
                scanned += 1
                raw = store.doc(i).astype(np.int64)
                ids = raw
                if ids.shape[0] and int(ids[0]) == 0:
                    ids = ids[1:]
                if ids.shape[0] and int(ids[-1]) == 1:
                    ids = ids[:-1]
                if ids.shape[0] < 64:
                    continue
                text = tok.decode(ids.tolist(), skip_special_tokens=True)
                domain = _domain_of(text)
                if domain is None:
                    continue
                sentences = split_sentences(text)[:400]
                if len(sentences) < 4:
                    continue
                if _grams(" ".join(sentences)) & banned:
                    continue
                rec = {"id": f"{source}-{i}", "domain": domain, "sentences": sentences}
                f.write(json.dumps(rec) + "\n")
                picked += 1
                if picked % 50 == 0:
                    print(f"[pool] {picked} kept, {scanned} scanned", flush=True)
            if picked >= n_docs or scanned >= max_scan:
                break
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**want, "picked": picked, "scanned": scanned, "keywords": KEYWORDS}, f, indent=2)
    print(f"[pool] wrote {picked} documents to {dest}", flush=True)
    return dest


def require_pool(path: str | None = None) -> str:
    path = path or pool_path()
    meta = path + ".meta.json"
    if not os.path.exists(meta):
        raise RuntimeError(f"missing pool meta {meta}")
    blob = json.load(open(meta, encoding="utf-8"))
    if not blob.get("ban_rr"):
        raise RuntimeError("pool meta does not say ban_rr; refusing to train")
    return path


def _records(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pool_windows_from(path: str, context: str, window: int | None, sentence_only: bool):
    require_pool(path)
    docs = [RRDoc(r["id"], r["domain"], r["sentences"], None, "pool") for r in _records(path)]
    return doc_windows(docs, context=context, sentence_only=sentence_only, max_tokens=window)


def silver_windows_from(path: str, context: str, window: int | None, sentence_only: bool):
    tok = load_tokenizer()
    limit = window or (1024 if context == "echo" else 2048)
    windows = []

    def _flush(rec, sentences, labels, indices):
        if not sentences:
            return
        doc = RRDoc(rec["id"], rec["domain"], sentences, labels, "silver")
        made = windows_for(doc, tok=tok, max_tokens=limit, sentence_only=sentence_only)
        denom = max(len(rec["sentences"]) - 1, 1)
        for piece in made:
            piece.rel_pos = [indices[j] / denom for j in piece.sent_index]
        windows.extend(made)

    for rec in _records(path):
        sentences, labels, indices = [], [], []
        for i, (sent, lab) in enumerate(zip(rec["sentences"], rec["labels"])):
            if lab is None:
                _flush(rec, sentences, labels, indices)
                sentences, labels, indices = [], [], []
                continue
            sentences.append(sent)
            labels.append(lab)
            indices.append(i)
        _flush(rec, sentences, labels, indices)
    return windows


def _teacher_dirs(exps: list[str], fold: int) -> list[str]:
    return [os.path.join(p3_dir(), "kd", "teacher", exp, f"fold{fold}") for exp in exps]


def dump_teachers(exps: list[str], batch_size: int = 4) -> str:
    """Cache each fold model on gold documents and on the pool. done.json is last."""
    from saul.rr.model import load_overlay, load_rr
    path = require_pool()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gold = [d for d in load_docs() if d.split != "test"]
    pool_docs = [RRDoc(r["id"], r["domain"], r["sentences"], None, "pool") for r in _records(path)]
    out_root = os.path.join(p3_dir(), "kd", "teacher")
    for exp in exps:
        for fold in range(5):
            run = os.path.join(p3_dir(), exp, f"fold{fold}")
            done = os.path.join(run, "done.json")
            dest = os.path.join(out_root, exp, f"fold{fold}")
            marker = os.path.join(dest, "done.json")
            if os.path.exists(marker):
                print(f"[dump] reuse {dest}")
                continue
            if not os.path.exists(done):
                raise FileNotFoundError(done)
            meta = json.load(open(done, encoding="utf-8"))
            model = load_rr(meta["model"], meta["technique"], context=meta["context"], device=device)
            load_overlay(model, torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
            model.to(device)
            docs = gold + pool_docs
            windows = doc_windows(docs, context=meta["context"], max_tokens=meta.get("window"),
                                   sentence_only=bool(meta.get("sentence_only")))
            packed = accumulate(model, windows, {d.doc_id: len(d.sentences) for d in docs}, batch_size, device)
            os.makedirs(dest, exist_ok=True)
            for doc_id, (mass, weight) in packed.items():
                save_npz(os.path.join(dest, f"{doc_id}.npz"), mass, weight)
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"exp": exp, "fold": fold, "docs": len(docs)}, f)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[dump] {exp} fold {fold}", flush=True)
    marker = os.path.join(out_root, "done.json")
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"exps": exps}, f)
    return out_root


def average_fold(exps: list[str], fold: int, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    ids = None
    tables = []
    for folder in _teacher_dirs(exps, fold):
        names = {n[:-4] for n in os.listdir(folder) if n.endswith(".npz")}
        ids = names if ids is None else ids & names
        tables.append(folder)
    for doc_id in ids or []:
        acc = None
        for folder in tables:
            probs = load_probs(os.path.join(folder, f"{doc_id}.npz"))
            acc = probs.astype(np.float64) if acc is None else acc + probs
        acc /= len(tables)
        weight = np.ones(acc.shape[0], dtype=np.float32)
        save_npz(os.path.join(dest, f"{doc_id}.npz"), acc * weight[:, None], weight)
    return dest


def train_student(exp: str = "E6", epochs: int = 10, teachers: list[str] | None = None) -> None:
    from saul.rr.train import train_one
    teachers = teachers or ["E1", "E2", "E3"]
    path = require_pool()
    for fold in range(5):
        dest = os.path.join(p3_dir(), "kd", "avg", f"fold{fold}")
        if not os.path.isdir(dest) or not os.listdir(dest):
            average_fold(teachers, fold, dest)
        train_one(
            exp, "conv_mini", "full", fold, context="echo", epochs=epochs, window=1024,
            teacher_dir=dest, kd_weight=0.5, pool_jsonl=path,
        )


def train_gold_control(epochs: int = 10) -> None:
    from saul.rr.train import train_one
    train_one("E6gold", "conv_mini", "full", 0, context="echo", epochs=epochs, window=1024)


def write_silver(exps: list[str], threshold: float = 0.9, fold: int = 0) -> str:
    """Pseudo-labels on the pool from one fold's averaged teachers. Low-confidence sentences are null."""
    path = require_pool()
    dest = os.path.join(p3_dir(), "silver", f"pool_fold{fold}.jsonl")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    avg = os.path.join(p3_dir(), "kd", "avg", f"fold{fold}")
    if not os.path.isdir(avg):
        average_fold(exps, fold, avg)
    with open(dest, "w", encoding="utf-8") as f:
        for rec in _records(path):
            probs_path = os.path.join(avg, f"{rec['id']}.npz")
            if not os.path.exists(probs_path):
                continue
            probs = load_probs(probs_path)
            labels = []
            for row in probs:
                if float(row.max()) >= threshold:
                    labels.append(LABELS[int(row.argmax())])
                else:
                    labels.append(None)
            f.write(json.dumps({"id": rec["id"], "domain": rec["domain"],
                                 "sentences": rec["sentences"], "labels": labels}) + "\n")
    print(f"[silver] {dest}", flush=True)
    return dest


def run_tapt(token_budget: int = 15_000_000, epochs: int = 3, seq_len: int = 512, batch_size: int = 4) -> str:
    """MNTP on dense_v1 over the domain pool. Same step as saul.ner.adapt, different text."""
    from saul.model.modeling_saul import SaulForCausalLM
    path = require_pool()
    out = os.path.join(p3_dir(), "adapt", "dense_v1")
    marker = os.path.join(out, "done.json")
    if os.path.exists(marker):
        print(f"[tapt] reuse {out}")
        return out
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer()
    ids = []
    for rec in _records(path):
        for sent in rec["sentences"]:
            ids.extend(int(i) for i in tok.encode(sent, add_special_tokens=False).ids)
    if len(ids) < seq_len:
        raise RuntimeError("domain pool is shorter than one TAPT sequence")
    pool = np.asarray(ids, dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint("dense_v1")).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rng = np.random.default_rng(781)
    seen = 0
    step = 0
    target = token_budget * max(epochs, 1)
    use_cuda = device == "cuda"
    while seen < target:
        starts = rng.integers(0, pool.shape[0] - seq_len, size=batch_size)
        batch = torch.tensor(np.stack([pool[s:s + seq_len] for s in starts]), dtype=torch.long)
        docs = torch.zeros_like(batch)
        masked, labels, loss_mask = mntp_mask(
            batch, docs, int(model.config.pad_token_id), int(model.config.mask_token_id),
            int(model.config.vocab_size), rng=rng, random_low=64,
        )
        masked, labels, loss_mask, docs = (t.to(device) for t in (masked, labels, loss_mask, docs))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            loss = model(masked, labels=labels, doc_ids=docs, loss_mask=loss_mask, mode="bidirectional").loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += int(batch.numel())
        step += 1
        if step % 20 == 0:
            print(f"[tapt] step {step} tokens {seen} loss {float(loss):.3f}", flush=True)
        if not torch.isfinite(loss):
            raise SystemExit("[tapt] non-finite loss")
    model.save_pretrained(out)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"tokens": seen, "steps": step, "epochs": epochs}, f)
    print(f"[tapt] saved {out}", flush=True)
    return out


def kd_member(mix_exps: list[str], student: str = "E6") -> bool:
    """True when adding the student raises the equal-mix OOF argmax score."""
    from saul.rr.ensemble import oof_argmax_f1
    from saul.rr.decode import average_marginals
    from saul.rr.score import macro_scores
    from saul.rr.data import labeled_docs, load_docs
    base = [e for e in mix_exps if oof_argmax_f1(e) is not None]
    if oof_argmax_f1(student) is None or not base:
        return False
    docs = {d.doc_id: d for d in labeled_docs(load_docs())}

    def _f1(exps):
        table = average_marginals(exps, kind="oof")
        gold, pred = [], []
        for doc_id, probs in table.items():
            doc = docs.get(doc_id)
            if doc is None or doc.labels is None:
                continue
            gold.extend(doc.label_ids)
            pred.extend(probs.argmax(1).tolist())
        return macro_scores(gold, pred)["macro_f1"]

    return _f1(base + [student]) > _f1(base)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pool", action="store_true")
    p.add_argument("--dump", default="")
    p.add_argument("--student", action="store_true")
    p.add_argument("--gold-control", action="store_true")
    p.add_argument("--tapt", action="store_true")
    p.add_argument("--silver", action="store_true")
    p.add_argument("--teachers", default="E1,E2,E3")
    args = p.parse_args()
    teachers = [x for x in args.teachers.split(",") if x]
    if args.pool:
        build_pool()
    if args.dump:
        dump_teachers([x for x in args.dump.split(",") if x])
    if args.tapt:
        run_tapt()
    if args.silver:
        for fold in range(5):
            write_silver(teachers, fold=fold)
    if args.student:
        train_student(teachers=teachers)
    if args.gold_control:
        train_gold_control()


if __name__ == "__main__":
    main()
