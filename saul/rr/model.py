"""Load a Phase-1 backbone and read final-normed states. No LM logits."""
from __future__ import annotations

import torch
import torch.nn as nn

from saul.ner.backbone import hidden_states
from saul.ner.peft import apply_technique
from saul.ner.registry import CONTEXT, resolve_checkpoint
from saul.rr.heads import RRHeads


class RRModel(nn.Module):
    def __init__(self, backbone, heads: RRHeads, context: str):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.context = context
        self.hidden_dropout_p = 0.0
        self.kd_weight = 0.0

    def forward(self, batch: dict) -> dict:
        hidden = hidden_states(self.backbone, batch["input_ids"], batch["doc_ids"], self.context)
        p = float(self.hidden_dropout_p or 0.0)
        if self.training and p > 0:
            hidden = torch.nn.functional.dropout(hidden, p=p, training=True)
        kd_only = bool(batch.get("kd_only", False))
        if kd_only:
            batch = {**batch, "kd_only": torch.tensor(True)}
        return self.heads(hidden, batch, kd_weight=self.kd_weight)


def load_rr(name: str, technique: str, context: str | None = None, init: str | None = None,
            device: str | torch.device = "cpu", dropout: float = 0.1) -> RRModel:
    from saul.model.modeling_saul import SaulForCausalLM

    path = init or resolve_checkpoint(name)
    backbone = SaulForCausalLM.from_pretrained(path)
    context = context or CONTEXT[name]
    apply_technique(backbone, technique)
    backbone.to(device)
    heads = RRHeads(backbone.config.d_model, dropout=dropout).to(device)
    return RRModel(backbone, heads, context)


def trainable_parameters(model: RRModel) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def overlay_state(model: RRModel) -> dict:
    keep = {name for name, p in model.named_parameters() if p.requires_grad}
    tensors = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in keep or k.startswith("heads.")
    }
    return {"context": model.context, "tensors": tensors}


def load_overlay(model: RRModel, blob: dict) -> None:
    missing, unexpected = model.load_state_dict(blob["tensors"], strict=False)
    bad = [k for k in unexpected if k in blob["tensors"]]
    if bad:
        raise RuntimeError(f"unexpected overlay keys: {bad[:8]}")
    _ = missing
