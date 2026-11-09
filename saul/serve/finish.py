"""Batched MBR. Embedding scores come from one packed forward; ROUGE-L is the existing bit-parallel LCS."""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from saul.decode.mbr import mbr_pick, rouge_matrix
from saul.decode.postprocess import to_128_words
from saul.serve.packing import pack_prompts, segment_spans

REASON = {0: "none", 1: "words", 2: "length", 3: "stop"}


def filter_candidates(texts: list[str], reasons: list[str], target: int | None) -> list[str]:
    """Keep finished-on-words candidates of exactly `target` words. If none qualify, keep the longest."""
    if target is None:
        return list(texts) or [""]
    full = [" ".join(t.split()[:target]) for t, r in zip(texts, reasons)
            if r == "words" and len(t.split()) >= target]
    if full:
        return full
    if not texts:
        return [""]
    return [max(texts, key=lambda t: (len(t.split()), len(t)))]


def postprocess(text: str, target: int | None, specials: list[str]) -> str:
    for s in specials:
        if s:
            text = text.replace(s, "")
    text = text.strip()
    if target and len(text.split()) >= target:
        return to_128_words(text, target)
    return text.strip()


def _rouge_one(cands: list[str]) -> np.ndarray:
    if len(cands) <= 1:
        return np.zeros((len(cands), len(cands)))
    return rouge_matrix(cands)


def rouge_groups(groups: list[list[str]], workers: int) -> list[np.ndarray]:
    if len(groups) == 0:
        return []
    if workers <= 1 or len(groups) == 1:
        return [_rouge_one(g) for g in groups]
    with Pool(min(workers, len(groups))) as pool:
        return pool.map(_rouge_one, groups)


@torch.no_grad()
def embed_groups(model, tokenizer, groups: list[list[str]], device) -> list[np.ndarray]:
    """Greedy-match F1, one packed forward for every candidate in the batch. Matches `embedding_matrix`."""
    flat: list[list[int]] = []
    bounds: list[tuple[int, int]] = []
    pad = tokenizer.pad_token_id
    for cands in groups:
        start = len(flat)
        for c in cands:
            ids = tokenizer.encode(c, add_special_tokens=False)[:512] or [pad]
            flat.append(ids)
        bounds.append((start, len(flat)))
    if not flat:
        return []
    # Flex's dense fallback materialises [heads, T, T]. Keep each forward under a few thousand tokens and use
    # SDPA, which does not allocate that score matrix. Groups are never split across forwards.
    mode = "bidirectional" if getattr(model.config, "arch", "dense") == "dense" else "causal"
    dev = device
    lengths = [len(row) for row in flat]
    group_tokens = [sum(lengths[a:b]) for a, b in bounds]
    hidden_parts: list[torch.Tensor | None] = [None] * len(flat)
    order = list(range(len(bounds)))
    prev_impl = model.config.attn_impl
    model.config.attn_impl = "sdpa"
    try:
        i = 0
        while i < len(order):
            tok = 0
            j = i
            while j < len(order) and (j == i or tok + group_tokens[order[j]] <= 2048):
                tok += group_tokens[order[j]]
                j += 1
            chunk_groups = order[i:j]
            i = j
            rows = []
            for g in chunk_groups:
                a, b = bounds[g]
                rows.extend(flat[a:b])
            packed = pack_prompts(rows, pad, multiple=1)
            ids = torch.tensor([packed["ids"]], dtype=torch.long, device=dev)
            docs = torch.tensor([packed["docs"]], dtype=torch.long, device=dev)
            pos = torch.tensor([packed["pos"]], dtype=torch.long, device=dev)
            hidden = model(ids, doc_ids=docs, position_ids=pos, mode=mode, return_hidden=True).hidden[0]
            hidden = F.normalize(hidden.float(), dim=-1)
            spans = [sp for sp in segment_spans(packed["docs"]) if sp[2] >= 0]
            cursor = 0
            for g in chunk_groups:
                a, b = bounds[g]
                for k in range(a, b):
                    s, e, _ = spans[cursor]
                    hidden_parts[k] = hidden[s:e]
                    cursor += 1
    finally:
        model.config.attn_impl = prev_impl
    out = []
    for start, end in bounds:
        out.append(_f1([hidden_parts[k] for k in range(start, end)]))
    return out


def _f1(segs: list[torch.Tensor]) -> np.ndarray:
    n = len(segs)
    if n <= 1:
        return np.zeros((n, n))
    t = max(s.shape[0] for s in segs)
    d = segs[0].shape[-1]
    h = segs[0].new_zeros(n, t, d)
    valid = torch.zeros(n, t, dtype=torch.bool, device=h.device)
    for i, s in enumerate(segs):
        h[i, :s.shape[0]] = s
        valid[i, :s.shape[0]] = True
    sim = torch.einsum("itd,jsd->ijts", h, h)
    sim = sim.masked_fill(~valid[None, :, None, :], -2.0)
    best = sim.max(dim=-1).values
    prec = (best * valid[:, None, :].to(best.dtype)).sum(-1) / valid.sum(-1).clamp(min=1)[:, None]
    rec = prec.T
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
    return f1.cpu().numpy()


class Finisher:
    def __init__(self, model, tokenizer, specials: list[str], workers: int = 1):
        self.model = model
        self.tok = tokenizer
        self.specials = specials
        self.workers = workers
        self.device = next(model.parameters()).device

    def choose(self, groups: list[dict]) -> list[str]:
        """Each group: texts, reasons, target, utility, alpha, mbr. Returns one string per group."""
        picked_inputs = []
        meta = []
        for g in groups:
            kept = filter_candidates(g["texts"], g["reasons"], g["target"])
            picked_inputs.append(kept)
            meta.append(g)
        utilities = [g["utility"] if g["mbr"] and len(picked_inputs[i]) > 1 else "rouge" for i, g in enumerate(meta)]
        need_embed = [i for i, u in enumerate(utilities) if u in ("embed", "mixed") and len(picked_inputs[i]) > 1]
        need_rouge = [i for i, u in enumerate(utilities) if u in ("rouge", "mixed") and len(picked_inputs[i]) > 1]
        embed = {}
        if need_embed:
            mats = embed_groups(self.model, self.tok, [picked_inputs[i] for i in need_embed], self.device)
            embed = dict(zip(need_embed, mats))
        rouge = {}
        if need_rouge:
            mats = rouge_groups([picked_inputs[i] for i in need_rouge], self.workers)
            rouge = dict(zip(need_rouge, mats))
        out = []
        for i, kept in enumerate(picked_inputs):
            if len(kept) == 1:
                choice = kept[0]
            else:
                u = utilities[i]
                if u == "embed":
                    M = embed[i]
                elif u == "rouge":
                    M = rouge[i]
                else:
                    a = meta[i]["alpha"]
                    M = a * embed[i] + (1 - a) * rouge[i]
                choice = kept[mbr_pick(M)]
            out.append(postprocess(choice, meta[i]["target"], self.specials))
        return out


def specials_of(tokenizer) -> list[str]:
    found = []
    for name in ("all_special_tokens",):
        found.extend(list(getattr(tokenizer, name, []) or []))
    found.extend(["<|bos|>", "<|eos|>", "<|pad|>", "<|mask|>", "<NAME>", "<ORG>", "<SECTION>", "<ENTITY>", "<ACT>", "<CITATION>"])
    # de-duplicate, keep order
    seen = set()
    out = []
    for s in found:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def workers_or_one() -> int:
    return 1 if os.environ.get("SAUL_SERVE_ROUGE_SERIAL") == "1" else 6
