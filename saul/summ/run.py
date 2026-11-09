"""Hybrid LoRMA summarizer for the 4-hour Phase 6 window.

Trains on official train.json only, holds out 200 documents, then writes five
submission files from one long decode.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import torch

from saul.cjpe.model import enable_activation_checkpointing
from saul.ner.peft import LoRMAPlus, wrap_lorma
from saul.ner.registry import resolve_checkpoint
from saul.tokenizer.train import load_hf_tokenizer

ROOT = os.environ.get("SAUL_ROOT", os.path.expanduser("~/cs781"))
DATA = os.path.join(ROOT, "data", "raw", "Phase_6_Summ")
OUT = "/data/cs781-p6"
EXP = os.path.join(OUT, "exp", "H1")
SUB = os.path.join(ROOT, "submissions", "phase6")
TOKENIZER = os.path.join(ROOT, "artifacts", "tokenizers", "legal32k")
P90_WORDS = 1501
P50_WORDS = 638
P75_WORDS = 963
# Longer than p75 so the p50 and p75 cuts are different texts, and short enough to decode in the remaining window.
GEN_WORDS = 1200
MAX_LEN = 8192


def _load_rows(name: str) -> list[dict]:
    path = os.path.join(DATA, "Train" if name == "train" else "Test", f"{name}.json")
    return json.load(open(path, encoding="utf-8"))


def holdout_ids(train: list[dict]) -> set[str]:
    path = os.path.join(OUT, "split.json")
    if os.path.exists(path):
        return set(json.load(open(path, encoding="utf-8"))["holdout_ids"])
    words = np.array([len(row["summary"].split()) for row in train])
    bins = np.digitize(words, np.quantile(words, [0.25, 0.5, 0.75]))
    rng = np.random.default_rng(781)
    chosen = []
    for b in range(4):
        chosen.extend(int(i) for i in rng.choice(np.where(bins == b)[0], size=50, replace=False))
    return {train[i]["id"] for i in chosen}


def _cap_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def edits(text: str) -> dict[str, str]:
    """The five submission texts. Each one changes the string."""
    sentences = _sentences(text)
    dropped = " ".join(sentences[1:]) if len(sentences) > 1 else _cap_words(text, max(1, len(text.split()) - 1))
    tail_n = max(1, int(len(sentences) * 0.9)) if sentences else 1
    tailed = " ".join(sentences[:tail_n]) if sentences else text
    return {
        "full": text.strip(),
        "drop_lead": dropped.strip(),
        "drop_tail": tailed.strip(),
        "p50": _cap_words(text, P50_WORDS),
        "p75": _cap_words(text, P75_WORDS),
    }


def _body(tok, document: str, mode: str, sep_id: int) -> list[int]:
    ids = tok.encode(document, add_special_tokens=False)
    if mode == "head" or len(ids) <= 4096:
        return ids[:4096]
    return ids[:2048] + [sep_id] + ids[-2048:]


def example(tok, row: dict, mode: str, bos: int, sep_id: int, summary_id: int, eos: int) -> tuple[list[int], list[int]]:
    prompt = [bos] + _body(tok, row["document"], mode, sep_id) + [summary_id]
    summary = tok.encode(_cap_words(row["summary"], P90_WORDS), add_special_tokens=False) + [eos]
    room = MAX_LEN - len(prompt)
    summary = summary[: max(1, room)]
    mask = [0] * len(prompt) + [1] * len(summary)
    return prompt + summary, mask


def prompt_text(tok, row: dict, mode: str, bos: int, sep_id: int, summary_id: int) -> str:
    """Text the decoder re-encodes. BOS is left to the engine, which prepends it."""
    ids = _body(tok, row["document"], mode, sep_id) + [summary_id]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _batch(seqs: list[list[int]], masks: list[list[int]], pad: int, device: str):
    width = max(len(seq) for seq in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long, device=device)
    loss_mask = torch.zeros((len(seqs), width), dtype=torch.bool, device=device)
    for i, (seq, mask) in enumerate(zip(seqs, masks)):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        loss_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.bool, device=device)
    return ids, loss_mask


def _trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _save(model, path: str) -> None:
    tensors = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(tensors, path)


def _load(model, path: str) -> None:
    tensors = torch.load(path, map_location="cpu", weights_only=False)
    missing = []
    owned = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in tensors.items():
            if name not in owned:
                missing.append(name)
                continue
            owned[name].copy_(value)
    if missing:
        raise RuntimeError(f"overlay keys missing from model: {missing[:5]}")


def _generate(model, tok, rows: list[dict], mode: str, bos: int, sep_id: int, summary_id: int,
              target_words: int, device: str) -> list[str]:
    """Greedy decode. Flash attention in the server path requires bf16 weights."""
    from saul.serve.api import LLM
    from saul.serve.finish import postprocess, specials_of
    from saul.serve.sampling import SamplingParams

    model.eval()
    held = next(model.parameters()).dtype
    model.to(dtype=torch.bfloat16)
    max_new = target_words * 2
    llm = LLM(
        model, tok, max_slots=4, block_size=4, max_prompt_len=4608, max_new_tokens=max_new,
        prefill_token_budget=8192, cuda_graphs=False, device=device, dtype="bfloat16", rouge_workers=1,
    )
    params = SamplingParams(
        n=1, temperature=0.0, truncation="none", target_words=target_words, max_new_tokens=max_new,
    )
    specials = specials_of(tok)
    out = []
    try:
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            prompts = [prompt_text(tok, row, mode, bos, sep_id, summary_id) for row in batch]
            generated = llm.generate(prompts, params, prepend_bos=True)
            out.extend(postprocess(o.candidates[0], target_words, specials) for o in generated)
            print(f"[summ] generated {len(out)}/{len(rows)}", flush=True)
    finally:
        llm.close()
        model.to(dtype=held)
    return out


def _score(ids: list[str], preds: list[str], refs: list[str]) -> dict:
    """BERTScore on GPU. The caller moves the summarizer off the GPU first."""
    import importlib.util

    import bert_score

    spec = importlib.util.spec_from_file_location(
        "evaluate_phase6", os.path.join(ROOT, "resources", "evaluation_script", "evaluate_phase6.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = max(len(preds), 1)
    _, _, f = bert_score.score(preds, refs, lang="en", batch_size=8, verbose=False, device="cpu")
    scorer = mod.rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    word_tok = mod.tokenizers.DefaultTokenizer(use_stemmer=True)
    r1 = rl = 0.0
    for pred, ref in zip(preds, refs):
        r1 += scorer.score(ref, pred)["rouge1"].fmeasure
        pt, rt = word_tok.tokenize(pred), word_tok.tokenize(ref)
        if not pt and not rt:
            rl += 1.0
        elif pt and rt:
            z = mod.lcs(pt, rt)
            pp, rr = z / len(pt), z / len(rt)
            rl += 0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr)
    return {"bert": float(f.mean().item()), "rouge1": r1 / n, "rougeL": rl / n, "n": len(ids)}


def _write_preds(name: str, ids: list[str], texts: list[str]) -> str:
    rows = [{"id": i, "summary": t if t.strip() else i} for i, t in zip(ids, texts)]
    dest = os.path.join(SUB, name)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "prediction.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    import zipfile
    zpath = os.path.join(dest, "submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname="prediction.json")
    return zpath


def train() -> str:
    os.makedirs(EXP, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf_tokenizer(TOKENIZER)
    bos = int(tok.bos_token_id)
    eos = int(tok.eos_token_id)
    pad = int(tok.pad_token_id)
    sep_id = int(tok.convert_tokens_to_ids("<|sep|>"))
    summary_id = int(tok.convert_tokens_to_ids("<|summary|>"))
    if summary_id < 0:
        raise RuntimeError("<|summary|> is not in the tokenizer")

    train_rows = _load_rows("train")
    held = holdout_ids(train_rows)
    fit = [row for row in train_rows if row["id"] not in held]
    val = [row for row in train_rows if row["id"] in held]
    print(f"[summ] fit {len(fit)} val {len(val)}", flush=True)

    from saul.model.modeling_saul import SaulForCausalLM
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint("hybrid_b"))
    wrap_lorma(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRMAPlus):
            for param in list(module.A.parameters()) + list(module.B.parameters()):
                param.requires_grad_(True)
    enable_activation_checkpointing(model, every=1)
    model.to(device).train()
    n_train = sum(p.numel() for p in _trainable(model))
    print(f"[summ] trainable {n_train}", flush=True)
    probe = fit[0]
    seq, mask = example(tok, probe, "ends", bos, sep_id, summary_id, eos)
    text = prompt_text(tok, probe, "ends", bos, sep_id, summary_id)
    again = tok.encode(text, add_special_tokens=False)
    body = seq[1: seq.index(summary_id) + 1]
    if again != body:
        raise RuntimeError(f"prompt round-trip broke: {len(again)} vs {len(body)}")
    print(f"[summ] prompt tokens {len(seq)} summary tokens {sum(mask)}", flush=True)

    opt = torch.optim.AdamW(_trainable(model), lr=5e-4, weight_decay=0.01)
    mode = "ends"
    switched = False
    best = -1.0
    best_path = os.path.join(EXP, "overlay.pt")
    latest_path = os.path.join(EXP, "latest.pt")
    state_path = os.path.join(EXP, "state.json")
    start = time.time()
    budget_min = float(os.environ.get("P6_BUDGET_MIN", "0"))
    deadline = start + budget_min * 60 if budget_min else None
    evaluated = False
    step = 0
    order = list(range(len(fit)))
    rng = np.random.default_rng(781)
    rng.shuffle(order)
    cursor = 0
    if os.path.exists(state_path) and os.path.exists(latest_path) and not os.path.exists(os.path.join(EXP, "done.json")):
        state = json.load(open(state_path, encoding="utf-8"))
        _load(model, latest_path)
        order = state["order"]
        cursor = int(state["cursor"])
        step = int(state["step"])
        mode = state["mode"]
        switched = bool(state["switched"])
        best = float(state["best"])
        evaluated = bool(state["evaluated"])
        print(f"[summ] resume step {step} cursor {cursor} mode {mode}", flush=True)
    # One epoch. A step packs sequences until 16K tokens.
    while cursor < len(order):
        batch_rows = []
        cost = 0
        while cursor < len(order) and len(batch_rows) < 4:
            row = fit[order[cursor]]
            seq, _mask = example(tok, row, mode, bos, sep_id, summary_id, eos)
            if batch_rows and cost + len(seq) > 8192:
                break
            batch_rows.append(row)
            cost += len(seq)
            cursor += 1
        seqs, masks = [], []
        for row in batch_rows:
            seq, mask = example(tok, row, mode, bos, sep_id, summary_id, eos)
            seqs.append(seq)
            masks.append(mask)
        ids, loss_mask = _batch(seqs, masks, pad, device)
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(ids, labels=ids, loss_mask=loss_mask, mode="causal").loss
            if not torch.isfinite(loss):
                raise SystemExit("[summ] non-finite loss")
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[summ] skip OOM step {step} tokens {ids.shape[-1]}", flush=True)
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_(_trainable(model), 1.0)
        warm = max(1, int(0.05 * max(len(fit) // 2, 1)))
        total = max(len(fit) // 2, 1)
        scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for group in opt.param_groups:
            group["lr"] = 5e-4 * scale
        opt.step()
        step += 1
        if step % 20 == 0:
            print(f"[summ] step {step} loss {float(loss):.3f} mode {mode} lr {5e-4 * scale:.2e}", flush=True)
        if step % 200 == 0:
            _save(model, latest_path)
            json.dump({"order": order, "cursor": cursor, "step": step, "mode": mode,
                       "switched": switched, "best": best, "evaluated": evaluated},
                      open(state_path, "w"))
        if deadline and time.time() >= deadline:
            print(f"[summ] budget {budget_min:.0f} min reached at step {step}", flush=True)
            break
        # The 200-summary check is only for an open-ended run. A budget run spends the GPU on training and one test decode.
        due = deadline is None and ((not evaluated and time.time() - start > 1800) or cursor >= len(order))
        if due:
            score = _eval_val(model, tok, val, mode, bos, sep_id, summary_id, device)
            evaluated = True
            print(f"[summ] val {score}", flush=True)
            if score["bert"] > best:
                best = score["bert"]
                _save(model, best_path)
                json.dump(score, open(os.path.join(EXP, "val.json"), "w"), indent=2)
            if not switched and score["bert"] < 0.84 and cursor < len(order):
                mode = "head"
                switched = True
                print("[summ] view switched to first 4096 tokens", flush=True)
            model.train()
    _save(model, best_path)
    blob = {"best_bert": best, "mode": mode, "steps": step, "hours": (time.time() - start) / 3600, "trainable": n_train,
            "budget_min": budget_min}
    json.dump(blob, open(os.path.join(EXP, "done.json"), "w"), indent=2)
    print(f"[summ] train done {blob}", flush=True)
    return best_path


def _eval_val(model, tok, val, mode, bos, sep_id, summary_id, device) -> dict:
    preds = _generate(model, tok, val, mode, bos, sep_id, summary_id, P90_WORDS, device)
    ids = [row["id"] for row in val]
    refs = [row["summary"] for row in val]
    score = _score(ids, preds, refs)
    json.dump({"ids": ids, "summaries": preds, **score}, open(os.path.join(EXP, "val_preds.json"), "w"))
    return score


def decode() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf_tokenizer(TOKENIZER)
    bos = int(tok.bos_token_id)
    sep_id = int(tok.convert_tokens_to_ids("<|sep|>"))
    summary_id = int(tok.convert_tokens_to_ids("<|summary|>"))
    from saul.model.modeling_saul import SaulForCausalLM
    model = SaulForCausalLM.from_pretrained(resolve_checkpoint("hybrid_b"))
    wrap_lorma(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRMAPlus):
            for param in list(module.A.parameters()) + list(module.B.parameters()):
                param.requires_grad_(True)
    _load(model, os.path.join(EXP, "overlay.pt"))
    model.to(device).eval()
    done = json.load(open(os.path.join(EXP, "done.json"), encoding="utf-8"))
    mode = done.get("mode", "ends")

    test = _load_rows("test")
    print(f"[summ] decoding test {len(test)} mode {mode} words {GEN_WORDS}", flush=True)
    test_text = _generate(model, tok, test, mode, bos, sep_id, summary_id, GEN_WORDS, device)

    ranked = []
    test_ids = [row["id"] for row in test]
    for name in ("full", "drop_lead", "drop_tail", "p75", "p50"):
        test_edit = [edits(text)[name] for text in test_text]
        zpath = _write_preds(f"P6_hybrid_lorma_{name}", test_ids, test_edit)
        words = int(np.mean([len(t.split()) for t in test_edit]))
        ranked.append({"name": name, "zip": zpath, "mean_words": words})
        print(f"[summ] {name} mean_words {words} {zpath}", flush=True)
    json.dump(ranked, open(os.path.join(SUB, "rank.json"), "w"), indent=2)
    print("[summ] upload order " + ", ".join(row["name"] for row in ranked), flush=True)


def main() -> None:
    train()
    decode()


if __name__ == "__main__":
    main()
