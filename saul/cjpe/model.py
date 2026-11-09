"""Phase-4 classifier: a Phase-1 backbone, a pooling head, and logit-adjusted loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.cjpe.const import PRIOR0, PRIOR1
from saul.ner.backbone import _normed
from saul.ner.peft import apply_technique
from saul.ner.registry import PARTIAL_LAYERS, resolve_checkpoint

CJPE_CONTEXT = {
    "dense_v1": "bidirectional",
    "dense_v2": "bidirectional",
    "hybrid_b": "causal",
    "hybrid_h5": "causal",
    "hybrid_h7": "causal",
    "ced": "encoder",
    "conv_mini": "causal",
}


class PoolHead(nn.Module):
    """Attention pool concatenated with a masked mean, plus the final token for causal models."""

    def __init__(self, d_model: int, dropout: float = 0.1, use_cls: bool = False):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.use_cls = use_cls
        width = d_model * (3 if use_cls else 2)
        self.fc = nn.Linear(width, 2)

    def parts(self, hidden: torch.Tensor, doc_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = doc_ids >= 0
        weights = self.score(hidden).squeeze(-1).float()
        weights = weights.masked_fill(~mask, torch.finfo(torch.float32).min / 4)
        alpha = torch.softmax(weights, dim=-1).to(hidden.dtype)
        attn = torch.einsum("bt,btd->bd", alpha, hidden)
        counted = mask.float().unsqueeze(-1)
        mean = (hidden * counted).sum(1) / counted.sum(1).clamp(min=1.0)
        last = hidden.new_zeros(hidden.shape[0], hidden.shape[-1])
        if self.use_cls:
            index = mask.long().sum(1).clamp(min=1) - 1
            last = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
        return attn, mean, last

    def forward(self, hidden: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        attn, mean, last = self.parts(hidden, doc_ids)
        pieces = [attn, mean, last] if self.use_cls else [attn, mean]
        return self.fc(self.drop(torch.cat(pieces, dim=-1)))


class CJPEModel(nn.Module):
    def __init__(self, backbone, head: PoolHead, context: str, loss: str = "logit_adjust"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.context = context
        self.loss_name = loss
        self.kd_temperature = 2.0

    def hidden(self, input_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        stop = None
        if self.context == "encoder":
            stop = int(self.backbone.config.n_encoder_layers)
            self.backbone._ner_stop_after = stop
        mode = "causal" if self.context == "causal" else "bidirectional"
        try:
            return _normed(self.backbone, input_ids, doc_ids, mode)
        finally:
            if stop is not None:
                self.backbone._ner_stop_after = None

    def forward(self, batch: dict, teacher: torch.Tensor | None = None) -> dict:
        hidden = self.hidden(batch["input_ids"], batch["doc_ids"])
        logits = self.head(hidden, batch["doc_ids"])
        out = {"logits": logits}
        labels = batch.get("labels")
        if labels is None:
            return out
        if self.loss_name == "kd":
            if teacher is None:
                raise RuntimeError("kd loss requires teacher probabilities")
            out["loss"] = kd_loss(logits, teacher, labels, self.kd_temperature)
        elif self.loss_name == "ce":
            out["loss"] = F.cross_entropy(logits, labels)
        else:
            out["loss"] = logit_adjusted_ce(logits, labels)
        return out


def logit_adjusted_ce(logits: torch.Tensor, labels: torch.Tensor,
                      prior: tuple[float, float] = (PRIOR0, PRIOR1)) -> torch.Tensor:
    """Menon et al. 2021. Adding log prior during training targets the balanced error."""
    log_prior = torch.log(logits.new_tensor(prior))
    return F.cross_entropy(logits + log_prior, labels)


def kd_loss(logits: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """0.5 * logit-adjusted CE + 0.5 * T^2 * KL(teacher || student). `teacher` is already softened."""
    hard = logit_adjusted_ce(logits, labels)
    log_student = F.log_softmax(logits / temperature, dim=-1)
    soft = teacher.clamp(min=1e-8)
    kl = (soft * (soft.log() - log_student)).sum(-1).mean() * (temperature ** 2)
    return 0.5 * hard + 0.5 * kl


def enable_activation_checkpointing(model: nn.Module, every: int = 2) -> int:
    """Checkpoint every Nth block. The original forward signature is unchanged."""
    if every <= 0:
        return 0
    import torch.utils.checkpoint as checkpoint

    wrapped = 0
    for index, block in enumerate(model.blocks):
        if index % every != 0:
            continue
        original = block.forward

        def wrapped_forward(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)

            def run(*inner):
                return _original(*inner, **kwargs)

            return checkpoint.checkpoint(run, *args, use_reentrant=False)

        block.forward = wrapped_forward
        wrapped += 1
    return wrapped


def set_attention(model: nn.Module, impl: str | None) -> None:
    if impl:
        model.config.attn_impl = impl


def load_cjpe(name: str, technique: str, context: str | None = None, init: str | None = None,
              device: str | torch.device = "cpu", dropout: float = 0.1, loss: str = "logit_adjust",
              ckpt_every: int = 2, attn: str | None = None) -> CJPEModel:
    from saul.model.modeling_saul import SaulForCausalLM

    path = init or resolve_checkpoint(name)
    backbone = SaulForCausalLM.from_pretrained(path)
    context = context or CJPE_CONTEXT[name]
    partial = PARTIAL_LAYERS.get(name) if technique == "partial" else None
    apply_technique(backbone, technique, partial)
    set_attention(backbone, attn)
    if ckpt_every:
        enable_activation_checkpointing(backbone, ckpt_every)
    backbone.to(device)
    head = PoolHead(backbone.config.d_model, dropout=dropout, use_cls=(context == "causal")).to(device)
    return CJPEModel(backbone, head, context, loss=loss)


def trainable_parameters(model: CJPEModel) -> list[nn.Parameter]:
    """Unique trainable parameters. Tied embeddings are registered twice."""
    seen: set[int] = set()
    out = []
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        out.append(param)
    return out


def overlay_state(model: CJPEModel) -> dict:
    keep = {name for name, param in model.named_parameters() if param.requires_grad}
    tensors = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in keep or key.startswith("head.")
    }
    return {
        "context": model.context,
        "loss": model.loss_name,
        "tensors": tensors,
    }


def load_overlay(model: CJPEModel, blob: dict) -> None:
    missing, unexpected = model.load_state_dict(blob["tensors"], strict=False)
    bad = [key for key in unexpected if key in blob["tensors"]]
    if bad:
        raise RuntimeError(f"unexpected overlay keys: {bad[:8]}")
    _ = missing


def corrected_logits(logits: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Plain CE still carries the train prior. Logit-adjusted training does not."""
    if loss_name == "ce":
        shift = torch.log(logits.new_tensor(PRIOR0 / PRIOR1))
        out = logits.clone()
        out[:, 1] = out[:, 1] + shift
        return out
    return logits
