"""Load a Phase-1 checkpoint and run it in one of the NER context modes.

bidirectional  Dense and the 52M proxy. Attention without a window sees the whole window.
echo           Hybrid and Conv-Mini. The window is fed twice, back to back, with no
               padding between the copies, so the causal mixers carry state into the
               second copy. Hidden states are read from that second copy.
encoder        CED. `_ner_stop_after` returns the encoder output; layers 0-1 keep the
               pretrained sliding window and layers 2-10 are bidirectional.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from saul.model.modeling_saul import SaulForCausalLM
from saul.ner.heads import NERHeads
from saul.ner.peft import apply_technique
from saul.ner.registry import CONTEXT, PARTIAL_LAYERS, resolve_checkpoint


class NERModel(nn.Module):
    def __init__(self, backbone: SaulForCausalLM, heads: NERHeads, context: str):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.context = context

    def hidden(self, input_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        return hidden_states(self.backbone, input_ids, doc_ids, self.context)

    def forward(self, batch: dict) -> dict:
        h = self.hidden(batch["input_ids"], batch["doc_ids"])
        p = float(getattr(self, "hidden_dropout_p", 0.0) or 0.0)
        if self.training and p > 0:
            h = torch.nn.functional.dropout(h, p=p, training=True)
        return self.heads(h, batch)


def hidden_states(model: SaulForCausalLM, input_ids: torch.Tensor, doc_ids: torch.Tensor, context: str) -> torch.Tensor:
    if context == "echo":
        return _echo(model, input_ids, doc_ids)
    stop = None
    if context == "encoder":
        stop = int(model.config.n_encoder_layers)
        model._ner_stop_after = stop
    try:
        return _normed(model, input_ids, doc_ids, "bidirectional")
    finally:
        if stop is not None:
            model._ner_stop_after = None


def _normed(model: SaulForCausalLM, input_ids: torch.Tensor, doc_ids: torch.Tensor, mode: str) -> torch.Tensor:
    """Final-normed hidden states without the LM head. The raw residual stream is too large
    in scale for a freshly initialised head, and the 32k-way logits are pure overhead here."""
    b, t = input_ids.shape
    pos = torch.arange(t, device=input_ids.device)[None].expand(b, -1)
    hidden = model._backbone(input_ids, pos, doc_ids, mode)
    return model.final_norm(hidden)


def _echo(model: SaulForCausalLM, input_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """Repeat each real (non-pad) span so the second copy has seen the first. Pads stay at the end."""
    b, t = input_ids.shape
    lengths = (doc_ids >= 0).sum(1)
    width = int(lengths.max()) * 2
    ids = input_ids.new_full((b, width), int(model.config.pad_token_id))
    docs = doc_ids.new_full((b, width), -1)
    for i in range(b):
        n = int(lengths[i])
        real = input_ids[i, :n]
        ids[i, :n] = real
        ids[i, n:n * 2] = real
        docs[i, :n * 2] = 0
    hidden = _normed(model, ids, docs, "causal")
    back = hidden.new_zeros(b, t, hidden.shape[-1])
    for i in range(b):
        n = int(lengths[i])
        back[i, :n] = hidden[i, n:n * 2]
    return back


def load_ner(name: str, technique: str, context: str | None = None, init: str | None = None,
             device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> NERModel:
    path = init or resolve_checkpoint(name)
    backbone = SaulForCausalLM.from_pretrained(path)
    context = context or CONTEXT[name]
    partial = PARTIAL_LAYERS.get(name) if technique == "partial" else None
    apply_technique(backbone, technique, partial)
    if dtype is not None:
        backbone = backbone.to(dtype=dtype)
    backbone.to(device)
    heads = NERHeads(backbone.config.d_model, probe=(technique == "probe")).to(device)
    return NERModel(backbone, heads, context)


def trainable_parameters(model: NERModel) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def overlay_state(model: NERModel) -> dict:
    """Weights that training changed: every trainable tensor, plus the head."""
    keep = {name for name, p in model.named_parameters() if p.requires_grad}
    # Heads are always stored in full so a reload does not depend on requires_grad.
    blob = {"context": model.context}
    blob["tensors"] = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep or k.startswith("heads.")}
    return blob


def load_overlay(model: NERModel, blob: dict) -> None:
    missing, unexpected = model.load_state_dict(blob["tensors"], strict=False)
    bad = [k for k in unexpected if k in blob["tensors"]]
    if bad:
        raise RuntimeError(f"unexpected overlay keys: {bad[:8]}")
    # Frozen base weights are absent from the overlay on purpose.
    _ = missing
