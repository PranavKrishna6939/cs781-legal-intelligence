"""Windowed inference. Token distributions are averaged across overlapping windows,
then turned into mention probabilities for the offline decoder.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from saul.ner.data import collate, windows_for
from saul.ner.labels import N_LABELS, N_TAGS, O
from saul.ner.decode import decode_doc, default_config


def _tri(index: int, length: int) -> float:
    if length <= 1:
        return 1.0
    mid = (length - 1) / 2.0
    return 0.1 + 0.9 * (1.0 - abs(index - mid) / mid)


def _span_dist_from_marginal(marg: torch.Tensor) -> torch.Tensor:
    """Mean token marginal -> 13-way (12 labels + O as none)."""
    piece = marg.mean(0)
    dist = piece.new_zeros(N_LABELS + 1)
    dist[N_LABELS] = piece[O]
    for lab in range(N_LABELS):
        dist[lab] = piece[1 + lab * 4:1 + (lab + 1) * 4].sum()
    return dist / dist.sum().clamp(min=1e-8)


class _Acc:
    def __init__(self, n_tokens: int):
        self.mass = np.zeros((n_tokens, N_TAGS), dtype=np.float64)
        self.weight = np.zeros(n_tokens, dtype=np.float64)
        self.spans: dict[tuple[int, int], list] = {}

    def add_token(self, index: int, probs: np.ndarray, weight: float) -> None:
        self.mass[index] += probs * weight
        self.weight[index] += weight

    def add_span(self, start: int, end: int, probs: np.ndarray, weight: float) -> None:
        self.spans.setdefault((start, end), []).append((probs, weight))

    def mentions(self) -> tuple[np.ndarray, list[dict]]:
        w = np.maximum(self.weight, 1e-8)
        avg = self.mass / w[:, None]
        avg = avg / np.maximum(avg.sum(-1, keepdims=True), 1e-8)
        logp = np.log(np.maximum(avg, 1e-8))
        mentions = []
        for (s, e), items in self.spans.items():
            num = np.zeros(len(items[0][0]), dtype=np.float64)
            den = 0.0
            for probs, weight in items:
                num += probs * weight
                den += weight
            mentions.append({"start": int(s), "end": int(e), "probs": (num / max(den, 1e-8)).tolist()})
        return logp.astype(np.float32), mentions


@torch.no_grad()
def infer_docs(model, docs, batch_size: int = 8, header: int | None = None, device: str = "cpu",
               mentions: bool = True, header_mode: str | None = None, preamble: bool | None = None) -> dict[str, dict]:
    """Return doc_id -> {tag_logprobs, tok_starts, tok_ends, mentions}. mentions=False skips the
    span-head candidates and the document Viterbi (the token decoder needs neither)."""
    model.eval()
    context = model.context
    if header_mode is None:
        header_mode = getattr(model, "header_mode", "raw")
    if preamble is None:
        preamble = bool(getattr(model, "preamble", False))
    windows = windows_for(docs, context, header, header_mode=header_mode, preamble=preamble)
    acc = {d.doc_id: _Acc(len(d.ids)) for d in docs}
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk).items()}
        hidden = model.hidden(batch["input_ids"], batch["doc_ids"])
        packed, mask, lengths = model.heads._pack(hidden, batch["body_mask"])
        emissions = model.heads.emit(packed).float()
        if model.heads.probe or model.heads.crf is None:
            marginal = torch.softmax(emissions, dim=-1)
        else:
            marginal = model.heads.crf.marginals(emissions, mask)
        span_rows = []
        if mentions and model.heads.span is not None:
            span_rows = model.heads.span.candidates(packed, lengths)
        else:
            span_rows = [[] for _ in chunk]
        for i, window in enumerate(chunk):
            n = int(lengths[i])
            store = acc[window.doc_id]
            row = marginal[i, :n].detach().float().cpu().numpy()
            for t in range(n):
                store.add_token(window.body_tok0 + t, row[t], _tri(t, n))
            for s, e, probs in span_rows[i]:
                if e > n:
                    continue
                cs = window.char_starts[window.body_start + s]
                ce = window.char_ends[window.body_start + e - 1]
                if cs < 0 or ce < 0:
                    continue
                store.add_span(cs, ce, np.asarray(probs, dtype=np.float64), _tri((s + e) // 2, n))
    out = {}
    for doc in docs:
        logp, _span_only = acc[doc.doc_id].mentions()
        if mentions and model.heads.crf is not None and not model.heads.probe:
            _add_document_spans(acc[doc.doc_id], model, logp, doc, device)
        _, rows = acc[doc.doc_id].mentions()
        out[doc.doc_id] = {
            "tag_logprobs": logp,
            "tok_starts": np.asarray(doc.starts, dtype=np.int32),
            "tok_ends": np.asarray(doc.ends, dtype=np.int32),
            "mentions": rows,
        }
    return out


def _add_document_spans(store: _Acc, model, logp: np.ndarray, doc, device: str) -> None:
    """One Viterbi pass over the merged token distributions, added to the span pool."""
    from saul.ner.labels import spans_from_tags
    n = logp.shape[0]
    if n == 0:
        return
    emissions = torch.tensor(logp, device=device).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    path = model.heads.crf.viterbi(emissions, mask)[0]
    token_probs = torch.tensor(np.exp(logp), dtype=torch.float32)
    start_at = {int(s): i for i, s in enumerate(doc.starts)}
    end_at = {int(e): i for i, e in enumerate(doc.ends)}
    for s, e, _lab in spans_from_tags(path, doc.starts, doc.ends):
        if s not in start_at or e not in end_at:
            continue
        tok_s, tok_e = start_at[s], end_at[e]
        dist = _span_dist_from_marginal(token_probs[tok_s:tok_e + 1]).numpy()
        store.add_span(s, e, dist, 1.0)


def save_inference(directory: str, docs, result: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    mention_path = os.path.join(directory, "oof_mentions.jsonl")
    with open(mention_path, "w", encoding="utf-8") as f:
        for doc in docs:
            rec = result[doc.doc_id]
            np.savez_compressed(
                os.path.join(directory, f"{doc.doc_id}.npz"),
                tag_logprobs=rec["tag_logprobs"],
                tok_starts=rec["tok_starts"],
                tok_ends=rec["tok_ends"],
                span_s=np.asarray([m["start"] for m in rec["mentions"]], dtype=np.int32),
                span_e=np.asarray([m["end"] for m in rec["mentions"]], dtype=np.int32),
                span_probs=np.asarray([m["probs"] for m in rec["mentions"]], dtype=np.float32)
                if rec["mentions"] else np.zeros((0, N_LABELS + 1), dtype=np.float32),
            )
            for m in rec["mentions"]:
                f.write(json.dumps({"doc": doc.doc_id, **m}) + "\n")


def predict_docs(docs, result: dict, cfg: dict | None = None) -> dict[str, list]:
    cfg = cfg or default_config()
    pred = {}
    for doc in docs:
        pred[doc.doc_id] = decode_doc(doc.text, result[doc.doc_id]["mentions"], cfg)
    return pred


def predict_tokens(docs, result: dict, cfg: dict | None = None, lexicons: dict | None = None) -> dict[str, list]:
    from saul.ner.tokdec import default_tok_config, finish, raw_spans
    cfg = cfg or default_tok_config()
    pred = {}
    for doc in docs:
        raw = raw_spans(doc.text, doc.starts, doc.ends, result[doc.doc_id]["tag_logprobs"],
                        cfg["penalty_begin"], cfg["penalty_end"], cfg.get("bias"))
        pred[doc.doc_id] = finish(doc.text, raw, cfg, (lexicons or {}).get(doc.doc_id))
    return pred
