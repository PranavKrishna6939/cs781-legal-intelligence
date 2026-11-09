"""Minimum-Bayes-risk selection: the candidate closest to the others under a utility.

Utilities allowed by the competition are lexical (ROUGE-L) or this model's own hidden states.
A pretrained metric model such as roberta-large must not be called at inference.

`utility_matrix` scores all candidate pairs at once: ROUGE-L via bit-parallel LCS (O(n) big-int ops per pair)
and a BERTScore-style greedy-match F1 over this model's hidden states from one batched forward.
"""
from __future__ import annotations

import numpy as np
import torch


def _lcs(a: list, b: list) -> int:
    """LCS length, bit-parallel (Allison–Dix / Hyyrö): V' = (V + (V & M)) | (V & ~M)."""
    if not a or not b:
        return 0
    masks: dict = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    V = full
    for y in b:
        U = V & masks.get(y, 0)
        V = ((V + U) | (V - U)) & full
    return len(a) - bin(V).count("1")


def _lcs_dp(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[-1], prev[j + 1]))
        prev = cur
    return prev[-1]


def _f1(l: int, na: int, nb: int) -> float:
    if l == 0:
        return 0.0
    p, r = l / na, l / nb
    return 2 * p * r / (p + r)


def rouge_l(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    return _f1(_lcs(ta, tb), len(ta), len(tb))


def rouge_matrix(cands: list[str]) -> np.ndarray:
    vocab: dict[str, int] = {}
    toks = [[vocab.setdefault(w, len(vocab)) for w in c.split()] for c in cands]
    n = len(cands)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            U[i, j] = U[j, i] = _f1(_lcs(toks[i], toks[j]), max(len(toks[i]), 1), max(len(toks[j]), 1))
    return U


@torch.no_grad()
def embedding_matrix(model, tokenizer, cands: list[str], device, bidirectional: bool | None = None,
                     max_tokens: int = 512) -> np.ndarray:
    """Greedy-match F1 on L2-normalised last-layer states of this model. One padded forward for all candidates."""
    if bidirectional is None:
        bidirectional = getattr(model.config, "arch", "dense") == "dense"
    enc = [tokenizer.encode(c, add_special_tokens=False)[:max_tokens] or [tokenizer.pad_token_id] for c in cands]
    T = max(len(e) for e in enc)
    n = len(enc)
    ids = torch.full((n, T), tokenizer.pad_token_id, dtype=torch.long, device=device)
    doc = torch.full((n, T), -1, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, device=device)
        doc[i, :len(e)] = 0
    pos = torch.arange(T, device=device)[None].expand(n, -1)
    out = model(ids, doc_ids=doc, position_ids=pos, mode="bidirectional" if bidirectional else "causal",
                return_hidden=True)
    h = torch.nn.functional.normalize(out.hidden.float(), dim=-1)          # [n, T, d]
    valid = doc >= 0
    sim = torch.einsum("itd,jsd->ijts", h, h)                               # [n, n, T, T]
    sim = sim.masked_fill(~valid[None, :, None, :], -2.0)
    best = sim.max(dim=-1).values                                           # [n, n, T]: token t of i vs j
    prec = (best * valid[:, None, :]).sum(-1) / valid.sum(-1)[:, None]     # P[i, j]: hyp i matched into ref j
    rec = prec.T
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
    return f1.cpu().numpy()


def mbr_pick(U: np.ndarray) -> int:
    n = U.shape[0]
    if n == 1:
        return 0
    scores = (U.sum(1) - np.diag(U)) / (n - 1)
    return int(np.argmax(scores))


def mbr_select(candidates: list[str], utility) -> str:
    """Generic form with a pairwise utility callable (tests, small n)."""
    n = len(candidates)
    if n == 1:
        return candidates[0]
    U = np.array([[utility(a, b) if i != j else 0.0 for j, b in enumerate(candidates)]
                  for i, a in enumerate(candidates)])
    return candidates[mbr_pick(U)]


def mbr_choose(candidates: list[str], utility: str = "mixed", alpha: float = 0.5, model=None, tokenizer=None,
               device=None) -> str:
    """utility: 'rouge' | 'embed' | 'mixed' (alpha * embed + (1 - alpha) * rouge)."""
    if len(candidates) == 1:
        return candidates[0]
    if utility == "rouge":
        U = rouge_matrix(candidates)
    elif utility == "embed":
        U = embedding_matrix(model, tokenizer, candidates, device)
    else:
        U = alpha * embedding_matrix(model, tokenizer, candidates, device) + (1 - alpha) * rouge_matrix(candidates)
    return candidates[mbr_pick(U)]
