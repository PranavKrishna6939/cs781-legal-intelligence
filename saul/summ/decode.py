"""Greedy and MBR decoding for Phase 6.

The engine stops on `target_words` and always suppresses ids 0..63. The length token is forced,
so it is still emitted. `no_repeat_ngram` stays at 4, which is the sampler's maximum.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from saul.summ.data import bucket_of, holdout_ids, load_rows, p6_dir, prompt_ids, strip_caption
from saul.summ.length import predict_ridge, stop_words, words_from_probs
from saul.summ.score import score_texts
from saul.summ.submit import check_rows, write_zip


def render_prompt(tok, ids: list[int]) -> str:
    text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    again = tok.encode(text, add_special_tokens=False)
    if list(again) != list(ids):
        raise RuntimeError(f"prompt round-trip broke: {len(again)} vs {len(ids)}")
    return text


def _summary_id(tok) -> int:
    tid = tok.convert_tokens_to_ids("<|summary|>")
    if tid is None or int(tid) < 0:
        raise RuntimeError("<|summary|> is not in the tokenizer")
    return int(tid)


@torch.no_grad()
def bucket_probs(model, prompts: list[list[int]], len_ids: list[int], device: str) -> list[torch.Tensor]:
    """Distribution over the 16 length ids at the last prompt position (the summary marker)."""
    was = model.training
    model.eval()
    index = torch.tensor(len_ids, device=device)
    out = []
    try:
        for row in prompts:
            ids = torch.tensor([row], dtype=torch.long, device=device)
            pos = torch.arange(len(row), device=device)[None]
            docs = torch.zeros_like(ids)
            enabled = device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                logits = model(ids, position_ids=pos, doc_ids=docs).logits[0, -1]
            out.append(torch.softmax(logits.index_select(0, index).float(), dim=0).cpu())
    finally:
        model.train(was)
    return out


def prompt_rows(tok, rows: list[dict], lmax: int, rooms: list[int] | int, strip: bool) -> list[list[int]]:
    marker = _summary_id(tok)
    if isinstance(rooms, int):
        rooms = [rooms] * len(rows)
    built = []
    for row, room in zip(rows, rooms):
        text = strip_caption(row["document"]) if strip else row["document"]
        doc = tok.encode(text, add_special_tokens=False)
        built.append(prompt_ids(doc, marker, lmax, room))
    return built


def _params(n: int, temperature: float, target: int, forced: int, alpha: float, max_new: int):
    from saul.serve.sampling import MBRParams, SamplingParams

    return SamplingParams(
        n=n,
        temperature=temperature,
        truncation="none" if temperature == 0 else "min_p",
        min_p=0.05,
        no_repeat_ngram=4,
        max_ngram_occurrences=2,
        frequency_penalty=0.1,
        frequency_window=64,
        target_words=target,
        max_new_tokens=max_new,
        forced_tokens=(forced,),
        mbr=MBRParams(enabled=n > 1, utility="mixed", alpha=alpha),
    )


def generate(
    model,
    tok,
    rows: list[dict],
    targets: list[int],
    forced: list[int],
    lmax: int,
    strip: bool,
    device: str,
    n: int = 1,
    temperature: float = 0.0,
    alpha: float = 0.6,
    slots: int = 16,
) -> list[str]:
    """One string per row. `forced[i]` is the length-token id for that row."""
    from saul.serve.api import LLM

    rooms = [max(8, target * 2) for target in targets]
    prompts = [row for row in prompt_rows(tok, rows, lmax, rooms, strip)]
    bos = int(tok.bos_token_id)
    texts = []
    use_ids = False
    for row in prompts:
        try:
            texts.append(render_prompt(tok, row))
        except RuntimeError:
            texts.append("")
            use_ids = True
    id_rows = [[bos, *row] for row in prompts]
    max_new = min(6144, max(16, max(targets) * 2))
    max_prompt = max(len(row) for row in prompts) + 1
    block = n if n > 1 else 1
    slots = max(block, slots - (slots % block))
    held = next(model.parameters()).dtype
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
    llm = LLM(
        model, tok,
        max_slots=slots, block_size=block, max_prompt_len=max_prompt, max_new_tokens=max_new,
        prefill_token_budget=max_prompt, cuda_graphs=device == "cuda", device=device,
        dtype="bfloat16" if device == "cuda" else "float32", rouge_workers=1,
    )
    params = [_params(n, temperature, t, f, alpha, max_new) for t, f in zip(targets, forced)]
    try:
        outputs = []
        # The engine queue holds 4096 requests. The training set is larger than that.
        for start in range(0, len(prompts), 512):
            end = start + 512
            if use_ids:
                reqs = [llm.engine.submit_ids(ids, par) for ids, par in zip(id_rows[start:end], params[start:end])]
                llm.engine.run_until_complete()
                for req in reqs:
                    if req.error is not None:
                        raise req.error
                    if req.output is None:
                        raise RuntimeError("decode produced no output")
                    outputs.append(req.output)
            else:
                outputs.extend(llm.generate(texts[start:end], params[start:end], prepend_bos=True))
            print(f"[p6] decoded {len(outputs)}/{len(prompts)}", flush=True)
    finally:
        llm.close()
        if device == "cuda":
            model.to(dtype=held)
            torch.cuda.empty_cache()
    return [out.text if out.text.strip() else " " for out in outputs]


def predicted_words(model, tok, rows, len_ids, lmax, strip, device, method: str, ridge) -> list[float]:
    if method == "ridge":
        return [predict_ridge(ridge, row["document"]) for row in rows]
    room = 2048
    prompts = prompt_rows(tok, rows, lmax, room, strip)
    # The prompt forward does not include BOS. Prepend it so the position matches training.
    bos = int(tok.bos_token_id)
    prompts = [[bos, *row] for row in prompts]
    probs = bucket_probs(model, prompts, len_ids, device)
    return [words_from_probs(p) for p in probs]


def lengths_for(method_words: list[float], lam: float, len_ids: list[int]) -> tuple[list[int], list[int]]:
    """Stop lengths and the forced bucket id (the bucket of the unscaled prediction)."""
    from saul.summ.data import bucket_of

    targets = [stop_words(w, lam) for w in method_words]
    forced = [len_ids[bucket_of(int(round(w)))] for w in method_words]
    return targets, forced


def sweep_lambda(model, tok, rows, words, len_ids, lmax, strip, device, slots: int) -> dict:
    """Decode once at lambda 1.2 and score prefixes. Greedy decoding makes a prefix the shorter run.

    Lambda 1.0 is the baseline. Another length replaces it only when its BERTScore gain is at least
    0.002 and the paired interval excludes zero. ROUGE-1 breaks a tie.
    """
    from saul.summ.score import paired_bootstrap

    targets, forced = lengths_for(words, 1.2, len_ids)
    full = generate(model, tok, rows, targets, forced, lmax, strip, device, n=1, slots=slots)
    refs = [row["summary"] for row in rows]
    dev = device if device == "cuda" else "cpu"
    scored = {}
    table = []
    for lam in (0.8, 0.9, 1.0, 1.1, 1.2):
        cut = [" ".join(text.split()[: stop_words(w, lam)]) for text, w in zip(full, words)]
        one = score_texts(cut, refs, device=dev)
        scored[lam] = one
        table.append({"lambda": lam, "bert": one["bert"], "rouge1": one["rouge1"], "rougeL": one["rougeL"]})
        print(f"[p6] lambda {lam} bert {one['bert']:.4f} r1 {one['rouge1']:.4f}", flush=True)
    winner = 1.0
    for lam in (0.8, 0.9, 1.1, 1.2):
        gain = paired_bootstrap(scored[lam]["per_bert"], scored[winner]["per_bert"])
        better_rouge = scored[lam]["rouge1"] > scored[winner]["rouge1"]
        if gain["keep"] or (gain["mean"] > 0 and better_rouge and scored[lam]["bert"] >= scored[winner]["bert"]):
            if gain["keep"]:
                winner = lam
    return {"lambda": winner, "table": table}


def write_submission(name, ids, texts, earlier=None) -> str:
    rows = [{"id": i, "summary": t if str(t).strip() else i} for i, t in zip(ids, texts)]
    report = check_rows(rows, ids, earlier)
    path = write_zip(name, rows, report)
    print(f"[p6] {name} {report} {path}", flush=True)
    return path


def load_meta(run: str) -> dict:
    with open(os.path.join(run, "meta.json"), encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    from saul.summ.train import load_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=("test", "holdout", "train"))
    ap.add_argument("--mode", default="greedy", choices=("greedy", "mbr"))
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--name", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_run(args.run, device)
    rows = load_rows("train" if args.split in ("holdout", "train") else "test")
    if args.split == "holdout":
        held = holdout_ids()
        rows = [row for row in rows if row["id"] in held]
    elif args.split == "train":
        held = holdout_ids()
        rows = [row for row in rows if row["id"] not in held]
    if args.limit:
        rows = rows[: args.limit]
    method = meta.get("length_method", "ridge")
    import numpy as np

    ridge = np.array(meta["ridge"], dtype=np.float64) if meta.get("ridge") is not None else None
    words = predicted_words(model, tok, rows, meta["len_ids"], meta["lmax"], meta["strip"], device, method, ridge)
    if args.sweep:
        result = sweep_lambda(model, tok, rows, words, meta["len_ids"], meta["lmax"], meta["strip"], device, args.slots)
        dest = os.path.join(args.run, "lambda.json")
        json.dump(result, open(dest, "w"), indent=2)
        print(f"[p6] sweep -> {dest} lambda {result['lambda']}", flush=True)
        return
    n = 1 if args.mode == "greedy" else args.n
    temperature = 0.0 if args.mode == "greedy" else args.temperature
    targets, forced = lengths_for(words, args.lam, meta["len_ids"])
    texts = generate(
        model, tok, rows, targets, forced, meta["lmax"], meta["strip"], device,
        n=n, temperature=temperature, alpha=args.alpha, slots=args.slots,
    )
    if args.split == "test" and args.name:
        write_submission(args.name, [row["id"] for row in rows], texts)
    else:
        dest = os.path.join(p6_dir(), f"{args.split}_{args.mode}.json")
        json.dump([{"id": r["id"], "summary": t} for r, t in zip(rows, texts)], open(dest, "w"))
        if args.split == "holdout":
            scored = score_texts(texts, [r["summary"] for r in rows], device=device if device == "cuda" else "cpu")
            print(f"[p6] holdout bert {scored['bert']:.4f} r1 {scored['rouge1']:.4f}", flush=True)
        print(f"[p6] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
