"""Pseudo-label pool, teacher dump, hard labels for self-training, and the KD students.

Teachers are fine-tuned fold checkpoints. Their token distributions are averaged into
one memmap. A student then trains on gold plus that memmap (knowledge distillation) or
on gold plus high-confidence hard tags (self-training).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from saul.data.tokens import open_source
from saul.ner.backbone import load_ner, load_overlay
from saul.ner.data import Doc, build_windows, load_tokenizer, tokenize_doc, window_shape
from saul.ner.infer import infer_docs
from saul.ner.labels import IGNORE, N_TAGS
from saul.ner.registry import CONTEXT, p2_dir, repo_root
from saul.ner.train import train_one


def _grams(text: str, k: int = 13) -> set[tuple]:
    words = text.split()
    return {tuple(words[i:i + k]) for i in range(max(len(words) - k + 1, 0))}


def _pool_dest(out_dir: str, ban_test: bool) -> str:
    """The test-banned pool is a different file from the older KD pool."""
    if ban_test:
        return os.path.join(out_dir, "pool_v2.jsonl")
    return os.path.join(out_dir, "pool.jsonl")


def build_pool(out_dir: str | None = None, n_docs: int | None = None, min_tokens: int = 3000,
               ban_test: bool = False) -> str:
    # T2 asks for 1,000 documents once fold3 is banned. The older KD pool stays at 800.
    if n_docs is None:
        n_docs = 1000 if ban_test else 800
    out_dir = out_dir or os.path.join(p2_dir(), "pool")
    os.makedirs(out_dir, exist_ok=True)
    dest = _pool_dest(out_dir, ban_test)
    if ban_test and os.path.basename(dest) != "pool_v2.jsonl":
        raise RuntimeError("ban_test must write pool_v2.jsonl, not pool.jsonl")
    meta_path = dest + ".meta.json"
    want = {"ban_test": bool(ban_test), "n_docs": int(n_docs), "min_tokens": int(min_tokens)}
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and os.path.exists(meta_path):
        prev = json.load(open(meta_path, encoding="utf-8"))
        if all(prev.get(k) == v for k, v in want.items()):
            print(f"[pool] reuse {dest}")
            return dest
    tok = load_tokenizer()
    from saul.ner.data import load_split
    banned: set[tuple] = set()
    for doc in load_split("train", tok):
        banned |= _grams(doc.text)
    if ban_test:
        for doc in load_split("test", tok):
            banned |= _grams(doc.text)
    token_root = os.path.join(repo_root(), "data", "tokens", "legal32k")
    picked = 0
    with open(dest, "w", encoding="utf-8") as f:
        for source in ("supreme_court", "high_courts"):
            try:
                store = open_source(token_root, "train", source)
            except FileNotFoundError:
                print(f"[pool] missing {source}")
                continue
            for i in range(store.n_docs):
                if picked >= n_docs:
                    break
                raw = store.doc(i).astype(np.int64)
                if raw.shape[0] < min_tokens:
                    continue
                ids = raw
                if ids[0] == 0:
                    ids = ids[1:]
                if ids.shape[0] and ids[-1] == 1:
                    ids = ids[:-1]
                text = tok.decode(ids.tolist(), skip_special_tokens=True)
                if len(text) < 2000 or _grams(text) & banned:
                    continue
                f.write(json.dumps({"id": f"{source}-{i}", "text": text}) + "\n")
                picked += 1
            if picked >= n_docs:
                break
    with open(meta_path, "w", encoding="utf-8") as meta:
        json.dump(want, meta)
    print(f"[pool] wrote {picked} documents to {dest}", flush=True)
    return dest


def _pool_docs(path: str) -> list[Doc]:
    tok = load_tokenizer()
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids, starts, ends = tokenize_doc(tok, rec["text"])
            if not ids:
                continue
            docs.append(Doc(rec["id"], rec["text"], "pool", ids, starts, ends, [0] * len(ids), [], {}))
    return docs


def _teacher_runs(spec: str) -> list[tuple[str, str, str, str]]:
    """spec 'E1,E3' -> (run_dir, model_name, technique, context) per fold, from done.json."""
    runs = []
    for exp in spec.split(","):
        exp = exp.strip()
        if not exp:
            continue
        root = os.path.join(p2_dir(), exp)
        for fold in range(5):
            done = os.path.join(root, f"fold{fold}", "done.json")
            if not os.path.exists(done):
                continue
            meta = json.load(open(done, encoding="utf-8"))
            runs.append((os.path.join(root, f"fold{fold}"), meta["model"], meta["technique"], meta["context"]))
    if not runs:
        raise FileNotFoundError(f"no finished folds for teachers {spec}")
    return runs


def dump_teachers(teachers: str, pool_path: str, out_dir: str | None = None, batch_size: int = 4,
                  device: str | None = None) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or os.path.join(p2_dir(), "pool", "teacher")
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "done.json")
    if os.path.exists(done):
        prev = json.load(open(done, encoding="utf-8"))
        if prev.get("spec") == teachers:
            print(f"[dump] reuse {out_dir}")
            return out_dir
        raise FileNotFoundError(f"{out_dir} was dumped for {prev.get('spec')}, not {teachers}")
    docs = _pool_docs(pool_path)
    offset_of = {}
    cursor = 0
    for doc in docs:
        offset_of[doc.doc_id] = cursor
        cursor += len(doc.ids)
    mm = np.memmap(os.path.join(out_dir, "probs.dat"), dtype=np.float32, mode="w+", shape=(cursor, N_TAGS))
    mm[:] = 0
    runs = _teacher_runs(teachers)
    for run, model_name, technique, context in runs:
        model = load_ner(model_name, technique, context=context, device=device)
        load_overlay(model, torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
        model.to(device)
        print(f"[dump] {run}", flush=True)
        for start in range(0, len(docs), 4):
            chunk = docs[start:start + 4]
            result = infer_docs(model, chunk, batch_size=batch_size, device=device, mentions=False)
            for doc in chunk:
                n = len(doc.ids)
                probs = np.exp(result[doc.doc_id]["tag_logprobs"])
                off = offset_of[doc.doc_id]
                mm[off:off + n] += probs[:n]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mm.flush()
    index = [{"id": d.doc_id, "text": d.text, "offset": offset_of[d.doc_id], "n": len(d.ids)} for d in docs]
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"teachers": len(runs), "docs": index}, f)
    with open(done, "w", encoding="utf-8") as f:
        json.dump({"teachers": len(runs), "tokens": int(cursor), "spec": teachers}, f)
    print(f"[dump] {len(runs)} teachers, {cursor} tokens", flush=True)
    return out_dir


def write_hard(teacher_dir: str, dest: str | None = None, threshold: float = 0.9) -> str:
    dest = dest or os.path.join(p2_dir(), "pool", "hard.jsonl")
    meta = json.load(open(os.path.join(teacher_dir, "index.json"), encoding="utf-8"))
    n_teachers = max(int(meta["teachers"]), 1)
    total = meta["docs"][-1]["offset"] + meta["docs"][-1]["n"] if meta["docs"] else 0
    mm = np.memmap(os.path.join(teacher_dir, "probs.dat"), dtype=np.float32, mode="r", shape=(total, N_TAGS))
    with open(dest, "w", encoding="utf-8") as f:
        for rec in meta["docs"]:
            probs = np.asarray(mm[rec["offset"]:rec["offset"] + rec["n"]]) / n_teachers
            tags = []
            for row in probs:
                k = int(row.argmax())
                tags.append(k if float(row[k]) >= threshold else IGNORE)
            f.write(json.dumps({"id": rec["id"], "text": rec["text"], "tags": tags}) + "\n")
    print(f"[hard] {dest}", flush=True)
    return dest


def resolve_silver_path(path: str, fold: int) -> str:
    """Use fold{k}.jsonl when `path` is the silver directory written by write_silver."""
    if path and os.path.isdir(path):
        candidate = os.path.join(path, f"fold{fold}.jsonl")
        if os.path.exists(candidate):
            return candidate
    return path


def load_extra(path: str, context: str, header: int | None = None, header_mode: str = "raw",
               preamble: bool = False) -> list:
    """Windows from a hard-tag jsonl or a teacher dump directory."""
    if os.path.isdir(path):
        return _teacher_windows(path, context, header, header_mode, preamble)
    return _hard_windows(path, context, header, header_mode, preamble)


def _make_windows(doc: Doc, context: str, header: int | None, probs: np.ndarray | None, supervise: bool,
                  header_mode: str = "raw", preamble: bool = False, tok=None) -> list:
    h, b, s = window_shape(context, header)
    out = []
    for window in build_windows(doc, h, b, s, header_mode=header_mode, tok=tok, preamble=preamble):
        window.supervise_none = supervise
        if probs is not None:
            window.teacher_probs = probs[window.body_tok0:window.body_tok0 + window.body_len]
        out.append(window)
    return out


def _hard_windows(path: str, context: str, header: int | None, header_mode: str = "raw",
                  preamble: bool = False) -> list:
    tok = load_tokenizer()
    windows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids, starts, ends = tokenize_doc(tok, rec["text"])
            tags = [int(t) for t in rec["tags"]]
            if len(tags) != len(ids):
                continue
            doc = Doc(rec["id"], rec["text"], "pseudo", ids, starts, ends, tags, [], {})
            windows.extend(_make_windows(doc, context, header, None, False, header_mode, preamble, tok))
    return windows


def _teacher_windows(directory: str, context: str, header: int | None, header_mode: str = "raw",
                     preamble: bool = False) -> list:
    meta = json.load(open(os.path.join(directory, "index.json"), encoding="utf-8"))
    n_teachers = max(int(meta["teachers"]), 1)
    docs = meta["docs"]
    total = docs[-1]["offset"] + docs[-1]["n"] if docs else 0
    mm = np.memmap(os.path.join(directory, "probs.dat"), dtype=np.float32, mode="r", shape=(total, N_TAGS))
    tok = load_tokenizer()
    windows = []
    for rec in docs:
        ids, starts, ends = tokenize_doc(tok, rec["text"])
        if len(ids) != rec["n"]:
            continue
        tags = [IGNORE] * len(ids)
        doc = Doc(rec["id"], rec["text"], "pseudo", ids, starts, ends, tags, [], {})
        probs = np.asarray(mm[rec["offset"]:rec["offset"] + rec["n"]], dtype=np.float32) / n_teachers
        windows.extend(_make_windows(doc, context, header, probs, False, header_mode, preamble, tok))
    return windows


def train_student(exp: str, student: str, teachers: str, epochs: int = 20, batch_size: int = 8,
                  train_kw: dict | None = None, pool: str | None = None) -> None:
    pool = pool or build_pool()
    dumped = dump_teachers(teachers, pool)
    context = CONTEXT[student]
    # Train through the normal loop. The extra windows carry teacher distributions,
    # and the head adds token KL when those are present. Gold stays half the batches.
    # A gold-only control is a separate --exp with no pseudo, run by the queue.
    for fold in range(5):
        # Point --pseudo at the teacher directory. train_one calls load_extra.
        os.environ["SAUL_KD_DIR"] = dumped
        train_one(exp, student, "full", fold, context=context, epochs=epochs,
                  batch_size=batch_size, pseudo=dumped, **(train_kw or {}))


def _silver_stack(cfg: dict, post_config: str, teacher_exp: str):
    """The tuned post stack for silver labels.

    Lexicon strings for fold k exclude that fold, matching the teacher, which has
    not seen it. The role classifier is the saved test model, or one fit on the
    teacher's out-of-fold features when the file is not beside the config.
    """
    from saul.ner.data import load_folds, load_split
    from saul.ner.post import fit_test_model
    from saul.ner.tokdec import load_run_logps, oof_lexicons
    bundle = None
    if cfg.get("use_p2_model"):
        path = os.path.join(os.path.dirname(os.path.abspath(post_config)), "p2_model.joblib")
        if os.path.exists(path):
            import joblib
            bundle = joblib.load(path)
        else:
            docs = load_split("train")
            runs = [os.path.join(p2_dir(), teacher_exp, f"fold{k}") for k in range(5)]
            bundle = fit_test_model(docs, load_run_logps(runs), load_folds(docs), cfg)
    lex_for_fold: dict[int, dict] = {}
    if any((cfg.get("lexicon") or {}).values()) or cfg.get("p3_auth_gov"):
        docs = load_split("train")
        folds = load_folds(docs)
        by_doc = oof_lexicons(docs, folds)
        for i, ids in enumerate(folds):
            if ids and ids[0] in by_doc:
                lex_for_fold[i] = by_doc[ids[0]]
    return bundle, lex_for_fold


def write_silver(teacher_exp: str, pool_path: str, post_config: str, out_dir: str | None = None,
                 device: str | None = None) -> str:
    """One teacher fold per student fold. Spans come from post_process; low-margin outside tokens are IGNORE."""
    from saul.ner.post import post_process
    from saul.ner.tokdec import raw_spans
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or os.path.join(p2_dir(), "silver")
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(post_config, encoding="utf-8"))
    bundle, lex_for_fold = _silver_stack(cfg, post_config, teacher_exp)
    docs = _pool_docs(pool_path)
    for fold in range(5):
        dest = os.path.join(out_dir, f"fold{fold}.jsonl")
        run = os.path.join(p2_dir(), teacher_exp, f"fold{fold}")
        meta = json.load(open(os.path.join(run, "done.json"), encoding="utf-8"))
        model = load_ner(meta["model"], meta["technique"], context=meta["context"], device=device)
        load_overlay(model, torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
        model.to(device)
        model.header_mode = meta.get("header_mode", "raw")
        model.preamble = bool(meta.get("preamble", False))
        print(f"[silver] teacher {run}", flush=True)
        lexicon = lex_for_fold.get(fold)
        with open(dest, "w", encoding="utf-8") as f:
            for start in range(0, len(docs), 4):
                chunk = docs[start:start + 4]
                result = infer_docs(model, chunk, batch_size=4, device=device, mentions=False,
                                    header_mode=model.header_mode, preamble=model.preamble)
                for doc in chunk:
                    logp = result[doc.doc_id]["tag_logprobs"]
                    raw = raw_spans(doc.text, doc.starts, doc.ends, logp, cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
                    spans = post_process(doc, raw, logp, cfg, bundle, lexicon)
                    tags = _silver_tags(doc, spans, np.exp(logp))
                    f.write(json.dumps({"id": doc.doc_id, "text": doc.text, "tags": tags}) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[silver] wrote {out_dir}", flush=True)
    return out_dir


def _silver_tags(doc, spans, probs: np.ndarray) -> list[int]:
    from saul.ner.labels import O, label_id, tag_id
    tags = [O] * len(doc.ids)
    for s, e, lab in spans:
        idxs = [i for i, (a, b) in enumerate(zip(doc.starts, doc.ends)) if a < e and b > s]
        if not idxs:
            continue
        li = label_id(lab)
        if len(idxs) == 1:
            tags[idxs[0]] = tag_id(li, "S")
        else:
            tags[idxs[0]] = tag_id(li, "B")
            for i in idxs[1:-1]:
                tags[i] = tag_id(li, "I")
            tags[idxs[-1]] = tag_id(li, "E")
    for i, row in enumerate(probs):
        if tags[i] == O and float(row.max()) < 0.9:
            tags[i] = IGNORE
    return tags


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teachers", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pool-only", action="store_true")
    p.add_argument("--dump-only", action="store_true")
    p.add_argument("--hard-out", default=None)
    p.add_argument("--teacher-dir", default=None)
    p.add_argument("--n-docs", type=int, default=None,
                   help="default 800, or 1000 when the fold3 ban is on")
    p.add_argument("--ban-test", action="store_true", help="drop pool documents that share a 13-gram with fold3")
    p.add_argument("--silver", action="store_true")
    p.add_argument("--teacher-per-fold", action="store_true")
    p.add_argument("--post-config", default=None)
    # The queue appends SAUL_P2_RECIPE. These are the train.py flags that recipe can carry.
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--adam-eps", type=float, default=None)
    p.add_argument("--rdrop", type=float, default=None)
    p.add_argument("--fgm", type=float, default=None)
    p.add_argument("--reinit-top", type=int, default=None)
    p.add_argument("--hidden-dropout", type=float, default=None)
    p.add_argument("--augment", default=None)
    p.add_argument("--preamble-flag", action="store_true")
    p.add_argument("--header-mode", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)
    args = p.parse_args()
    banned = args.ban_test or args.silver or args.teacher_per_fold
    if args.pool_only:
        build_pool(n_docs=args.n_docs, ban_test=args.ban_test)
        return
    pool = build_pool(n_docs=args.n_docs, ban_test=banned)
    if args.silver or args.teacher_per_fold:
        if not args.post_config:
            raise SystemExit("--silver requires --post-config")
        write_silver(args.teachers, pool, args.post_config)
        return
    if args.dump_only:
        dumped = dump_teachers(args.teachers, pool, out_dir=args.teacher_dir)
        if args.hard_out:
            write_hard(dumped, args.hard_out)
        return
    recipe = {}
    if args.lr is not None:
        recipe["base_lr"] = args.lr
    if args.head_lr is not None:
        recipe["head_lr"] = args.head_lr
    if args.adam_eps is not None:
        recipe["adam_eps"] = args.adam_eps
    if args.rdrop is not None:
        recipe["rdrop"] = args.rdrop
    if args.fgm is not None:
        recipe["fgm"] = args.fgm
    if args.reinit_top is not None:
        recipe["reinit_top"] = args.reinit_top
    if args.hidden_dropout is not None:
        recipe["hidden_dropout"] = args.hidden_dropout
    if args.augment is not None:
        recipe["augment"] = args.augment
    if args.preamble_flag:
        recipe["preamble"] = True
    if args.header_mode is not None:
        recipe["header_mode"] = args.header_mode
    if args.seed is not None:
        recipe["seed"] = args.seed
    if args.train_frac is not None:
        recipe["train_frac"] = args.train_frac
    train_student(args.exp, args.student, args.teachers, epochs=args.epochs,
                  batch_size=args.batch_size, train_kw=recipe, pool=pool)


if __name__ == "__main__":
    main()
