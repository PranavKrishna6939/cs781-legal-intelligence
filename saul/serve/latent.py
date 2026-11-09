"""Bounded frozen-backbone features for the J09 utility-witness pilot.

These are raw last-block states, exactly the states returned by return_hidden,
not input embeddings or a pretrained external encoder. No generation lives here.
"""
from __future__ import annotations

import hashlib
import json
import torch
import torch.nn.functional as F

from saul.serve.packing import pack_prompts, segment_spans


def make_bank(body, dimension, device):
    center = torch.as_tensor(body['center'], dtype=torch.float32, device=device)
    anchors = torch.as_tensor(body['anchors'], dtype=torch.float32, device=device)
    tau = float(body.get('tau', .1))
    if center.shape != (dimension,) or anchors.ndim != 2 or anchors.shape[1] != dimension:
        raise ValueError('Bank dimension does not match served backbone')
    if not 2 <= anchors.shape[0] <= 256 or not .01 <= tau <= 1:
        raise ValueError('Bank needs 2..256 anchors and tau in [.01,1]')
    if not torch.isfinite(center).all() or not torch.isfinite(anchors).all():
        raise ValueError('Nonfinite bank')
    if (anchors.norm(dim=-1) < 1e-6).any():
        raise ValueError('Zero anchor')
    identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return dict(center=center, anchors=F.normalize(anchors, dim=-1), tau=tau, id=identity)


def witnesses(hidden, bank):
    normalized = F.normalize(hidden.float() - bank['center'], dim=-1)
    similarity = normalized @ bank['anchors'].T
    support = similarity.clamp(min=0).amax(dim=0)
    mass = (similarity / bank['tau']).softmax(dim=-1).mean(dim=0)
    return support, mass


@torch.inference_mode()
def latent_features(model, tokenizer, texts, *, kind='future', sample_tokens=0, bank=None,
                    token_budget=2048):
    if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
        raise ValueError('texts must have 1..64 entries')
    if kind not in ('future', 'prefix') or not 0 <= sample_tokens <= 32:
        raise ValueError('Invalid kind or sample_tokens (0..32)')
    if not 512 <= token_budget <= 2048:
        raise ValueError('token_budget must be 512..2048')
    device = next(model.parameters()).device
    specials = set(getattr(tokenizer, 'all_special_ids', []) or [])
    rows, clipped = [], []
    for text in texts:
        if not isinstance(text, str) or len(text) > 100000:
            raise ValueError('Text must be a string of at most 100000 characters')
        ids = tokenizer.encode(text, add_special_tokens=False)
        if kind == 'prefix':
            bos = getattr(tokenizer, 'bos_token_id', None)
            if bos is not None and (not ids or ids[0] != bos):
                ids = [bos] + ids
            clipped.append(len(ids) > 2048)
            ids = ids[-2048:]
        else:
            clipped.append(len(ids) > 512)
            ids = ids[:512]
        if not ids or not any(i not in specials for i in ids):
            raise ValueError('No nonspecial tokens')
        rows.append(ids)
    output = []
    previous = model.config.attn_impl
    model.config.attn_impl = 'sdpa'
    mode = 'bidirectional' if kind == 'future' and model.config.arch == 'dense' else 'causal'
    try:
        start = 0
        while start < len(rows):
            end, size = start, 0
            while end < len(rows) and (end == start or size + len(rows[end]) <= token_budget):
                size += len(rows[end]); end += 1
            packed = pack_prompts(rows[start:end], tokenizer.pad_token_id, multiple=1)
            ids = torch.tensor([packed['ids']], device=device, dtype=torch.long)
            docs = torch.tensor([packed['docs']], device=device, dtype=torch.long)
            positions = torch.tensor([packed['pos']], device=device, dtype=torch.long)
            # Same backbone path as forward(return_hidden=True), without unused vocabulary logits.
            hidden = model._backbone(ids, positions, docs, mode)[0]
            spans = [span for span in segment_spans(packed['docs']) if span[2] >= 0]
            for offset, (left, right, _) in enumerate(spans):
                keep = torch.tensor([i not in specials for i in rows[start+offset]], device=device)
                h = hidden[left:right][keep].float()
                item = dict(tokens=len(h), truncated=clipped[start+offset])
                if kind == 'prefix':
                    item['pooled'] = torch.cat([h[-1], h[-64:].mean(0), h[-256:].mean(0), h.mean(0)]).cpu().tolist()
                if sample_tokens:
                    indices = torch.linspace(0, len(h)-1, min(sample_tokens, len(h)), device=device).long()
                    item['samples'] = h[indices].cpu().tolist()
                if bank is not None:
                    support, mass = witnesses(h, bank)
                    item.update(support=support.cpu().tolist(), mass=mass.cpu().tolist())
                output.append(item)
            start = end
    finally:
        model.config.attn_impl = previous
    return output
