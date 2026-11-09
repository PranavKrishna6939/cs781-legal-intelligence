"""Phase 5 multi-label model: chunked backbone, LAAT or pooled head, multi-label losses."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from saul.cjpe.model import enable_activation_checkpointing, set_attention
from saul.lsi.const import N_LABELS
from saul.lsi.decoder import LabelDecoderHead, db_ntr_loss
from saul.ner.backbone import _echo, _normed
from saul.ner.peft import (
    LoRA, _replace, _target_linears, apply_partial, apply_technique, freeze_all, unfreeze_adapters,
    wrap_lorma, wrap_pfeiffer,
)
from saul.ner.registry import PARTIAL_LAYERS, resolve_checkpoint

LSI_CONTEXT = {
    "dense_v1": "bidirectional",
    "dense_v2": "bidirectional",
    "hybrid_b": "echo",
    "ced": "encoder",
    "conv_mini": "echo",
}


class AdaLoRA(LoRA):
    """LoRA with a per-rank gate. `prune_` drops the lowest-importance ranks, which is the
    budget step of AdaLoRA without editing the shared PEFT module."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__(base, rank, alpha)
        self.rank_gate = nn.Parameter(torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.A(x) * self.rank_gate
        return self.base(x) + self.scale * self.B(hidden)

    @torch.no_grad()
    def prune_(self, keep: int = 8) -> int:
        score = self.A.weight.norm(dim=1) * self.B.weight.norm(dim=0) * self.rank_gate.abs()
        order = torch.argsort(score)
        drop = order[: max(0, score.numel() - keep)]
        self.rank_gate.data[drop] = 0
        return int(drop.numel())


def wrap_adalora(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    pairs = _target_linears(model)
    for name, mod in pairs:
        _replace(model, name, AdaLoRA(mod, rank, alpha))
    return len(pairs)


def prune_adalora(model: nn.Module, keep: int = 8) -> int:
    dropped = 0
    for mod in model.modules():
        if isinstance(mod, AdaLoRA):
            dropped += mod.prune_(keep)
    return dropped


def apply_adapter(model: nn.Module, bottleneck: int = 64) -> None:
    """Pfeiffer adapters plus norm gains. Biases stay frozen: BitFit is not a Phase 5 technique."""
    wrap_pfeiffer(model, bottleneck)
    freeze_all(model)
    unfreeze_adapters(model)
    for name, param in model.named_parameters():
        if name.endswith("gamma"):
            param.requires_grad_(True)


def apply_lsi_technique(model: nn.Module, technique: str, partial: tuple[int, int] | None = None) -> str:
    technique = technique.lower()
    if technique == "adapter":
        apply_adapter(model)
        return technique
    if technique == "adalora":
        freeze_all(model)
        wrap_adalora(model)
        return technique
    if technique == "lorma":
        # LoRMA on the block linears, plus the token embeddings. Everything else stays frozen.
        wrap_lorma(model)
        freeze_all(model)
        unfreeze_adapters(model)
        model.get_input_embeddings().weight.requires_grad_(True)
        return technique
    return apply_technique(model, technique, partial)


class LAATHead(nn.Module):
    """Label-wise attention over every token of a case (CAML / LAAT)."""

    def __init__(self, d_model: int, n_labels: int = N_LABELS, bottleneck: int = 512,
                 prior: torch.Tensor | None = None):
        super().__init__()
        self.proj = nn.Linear(d_model, bottleneck)
        self.query = nn.Parameter(torch.randn(n_labels, bottleneck) * 0.02)
        self.weight = nn.Parameter(torch.zeros(n_labels, d_model))
        nn.init.xavier_uniform_(self.weight)
        bias = torch.zeros(n_labels)
        if prior is not None:
            p = prior.clamp(1e-4, 1 - 1e-4)
            bias = torch.log(p / (1 - p))
        self.bias = nn.Parameter(bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, return_extra: bool = False) -> dict:
        u = torch.tanh(self.proj(hidden))
        scores = torch.einsum("bth,lh->blt", u, self.query)
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min / 4)
        alpha = torch.softmax(scores, dim=-1)
        z = torch.einsum("blt,btd->bld", alpha, hidden)
        logits = (z * self.weight).sum(-1) + self.bias
        out = {"logits": logits}
        if return_extra:
            out["alpha"] = alpha
            out["label_reps"] = z
        return out


class PooledMaxHead(nn.Module):
    """Attention-pool and mean-pool each chunk, then take the max over chunks of the case."""

    def __init__(self, d_model: int, n_labels: int = N_LABELS):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model * 2, n_labels)

    def chunk_vectors(self, hidden: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        mask = doc_ids >= 0
        weights = self.score(hidden).squeeze(-1)
        weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min / 4)
        alpha = torch.softmax(weights, dim=-1)
        attn = torch.einsum("nw,nwd->nd", alpha, hidden)
        counted = mask.float().unsqueeze(-1)
        mean = (hidden * counted).sum(1) / counted.sum(1).clamp(min=1.0)
        return torch.cat([attn, mean], dim=-1)

    def forward(self, chunk_hidden: torch.Tensor, doc_ids: torch.Tensor, case_index: torch.Tensor,
                n_cases: int) -> dict:
        pooled = self.chunk_vectors(chunk_hidden, doc_ids)
        index = case_index.to(pooled.device)[:, None].expand_as(pooled)
        packed = pooled.new_full((n_cases, pooled.shape[-1]), float("-inf"))
        packed = packed.scatter_reduce(0, index, pooled, reduce="amax", include_self=True)
        return {"logits": self.fc(packed)}


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def asl_loss(logits: torch.Tensor, labels: torch.Tensor, gamma_neg: float = 4.0,
             gamma_pos: float = 0.0, clip: float = 0.05) -> torch.Tensor:
    """Asymmetric loss (Ridnik et al. 2021)."""
    xs_pos = torch.sigmoid(logits)
    xs_neg = (1.0 - xs_pos + clip).clamp(max=1.0)
    loss_pos = labels * torch.log(xs_pos.clamp(min=1e-8)) * (1 - xs_pos).pow(gamma_pos)
    loss_neg = (1 - labels) * torch.log(xs_neg.clamp(min=1e-8)) * (1 - xs_neg).pow(gamma_neg)
    return -(loss_pos + loss_neg).mean()


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pt = labels * prob + (1 - labels) * (1 - prob)
    base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (base * (1 - pt).pow(gamma)).mean()


def pair_penalty(logits: torch.Tensor, labels: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
    """Pull probabilities toward P(label | gold positives) for frequent pairs."""
    target = labels @ conditional
    return F.mse_loss(torch.sigmoid(logits), target.clamp(0, 1))


def task_loss(logits: torch.Tensor, labels: torch.Tensor, name: str,
              conditional: torch.Tensor | None = None) -> torch.Tensor:
    if name == "asl":
        return asl_loss(logits, labels)
    if name == "focal":
        return focal_loss(logits, labels)
    if name == "pair":
        base = bce_loss(logits, labels)
        if conditional is None:
            return base
        return base + 0.1 * pair_penalty(logits, labels, conditional)
    return bce_loss(logits, labels)


def kd_loss(logits: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor,
            temperature: float = 2.0) -> torch.Tensor:
    hard = bce_loss(logits, labels)
    soft = F.binary_cross_entropy_with_logits(logits / temperature, teacher) * (temperature ** 2)
    return 0.5 * hard + 0.5 * soft


def regroup_cases(hidden: torch.Tensor, doc_ids: torch.Tensor, case_index: torch.Tensor,
                  n_cases: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the real tokens of each case. Returns [B, T, d] and a bool mask."""
    mask = doc_ids >= 0
    pieces: list[torch.Tensor] = []
    for case in range(n_cases):
        rows = torch.nonzero(case_index == case, as_tuple=False).flatten()
        parts = [hidden[int(row), mask[int(row)]] for row in rows]
        parts = [part for part in parts if part.numel()]
        if parts:
            pieces.append(torch.cat(parts, dim=0))
        else:
            pieces.append(hidden.new_zeros(1, hidden.shape[-1]))
    width = max(piece.shape[0] for piece in pieces)
    out = hidden.new_zeros(n_cases, width, hidden.shape[-1])
    out_mask = torch.zeros(n_cases, width, dtype=torch.bool, device=hidden.device)
    for case, piece in enumerate(pieces):
        out[case, : piece.shape[0]] = piece
        out_mask[case, : piece.shape[0]] = True
    return out, out_mask


class LSIModel(nn.Module):
    def __init__(self, backbone, head: nn.Module, context: str, head_name: str, loss_name: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.context = context
        self.head_name = head_name
        self.loss_name = loss_name
        self.kd_temperature = 2.0
        self.pair_conditional: torch.Tensor | None = None

    def chunk_hidden(self, input_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        if self.context == "echo":
            return _echo(self.backbone, input_ids, doc_ids)
        if self.context == "causal":
            return _normed(self.backbone, input_ids, doc_ids, "causal")
        stop = None
        if self.context == "encoder":
            stop = int(self.backbone.config.n_encoder_layers)
            self.backbone._ner_stop_after = stop
        try:
            return _normed(self.backbone, input_ids, doc_ids, "bidirectional")
        finally:
            if stop is not None:
                self.backbone._ner_stop_after = None

    def forward(self, batch: dict, teacher: torch.Tensor | None = None, return_extra: bool = False) -> dict:
        hidden = self.chunk_hidden(batch["input_ids"], batch["doc_ids"])
        if self.head_name == "decoder" and "seg_index" in batch:
            index = batch["seg_index"].to(hidden.device).clamp(0, self.head.seg_emb.num_embeddings - 1)
            hidden = hidden + self.head.seg_emb(index).unsqueeze(1).to(dtype=hidden.dtype)
        n_cases = int(batch["n_cases"])
        if self.head_name == "pooled":
            out = self.head(hidden, batch["doc_ids"], batch["case_index"], n_cases)
        else:
            states, mask = regroup_cases(hidden, batch["doc_ids"], batch["case_index"], n_cases)
            if self.head_name == "decoder":
                out = self.head(states, mask, batch, return_extra=return_extra)
            else:
                out = self.head(states, mask, return_extra=return_extra)
        labels = batch.get("labels")
        if labels is None:
            return out
        out["loss"] = self._loss(out, labels, teacher, batch.get("teacher_states"))
        return out

    def _loss(self, out: dict, labels: torch.Tensor, teacher: torch.Tensor | None,
              teacher_states: torch.Tensor | None) -> torch.Tensor:
        logits = out["logits"]
        fallback = bool(getattr(self, "db_fallback", False))
        if self.loss_name == "db" and not fallback:
            freq = getattr(self, "label_freq", None)
            if freq is None:
                raise RuntimeError("db loss requires model.label_freq")
            loss = db_ntr_loss(logits, labels, freq)
            deep_sup = float(getattr(self, "deep_sup", 0.0))
            if teacher is None and deep_sup and "logits_deep" in out:
                loss = loss + deep_sup * bce_loss(out["logits_deep"], labels)
        elif self.loss_name == "kd":
            if teacher is None:
                raise RuntimeError("kd loss requires teacher probabilities")
            loss = kd_loss(logits, labels, teacher, self.kd_temperature)
        elif fallback:
            loss = asl_loss(logits, labels)
        else:
            conditional = self.pair_conditional
            if conditional is not None and conditional.device != logits.device:
                conditional = conditional.to(logits.device)
            loss = task_loss(logits, labels, self.loss_name, conditional)
        if teacher is not None and self.head_name == "decoder":
            temperature = self.kd_temperature
            student = torch.sigmoid(logits / temperature).clamp(1e-6, 1 - 1e-6)
            target = torch.sigmoid(teacher.to(dtype=logits.dtype) / temperature).clamp(1e-6, 1 - 1e-6)
            loss = loss + (temperature ** 2) * F.binary_cross_entropy(student, target.detach())
            if teacher_states is not None and "label_states" in out:
                width = out["label_states"].shape[-1]
                predicted = F.layer_norm(out["label_states"].float(), (width,))
                gold = F.layer_norm(teacher_states.to(device=predicted.device, dtype=predicted.dtype), (width,))
                loss = loss + 0.5 * F.mse_loss(predicted, gold.detach())
        return loss


def load_lsi(name: str, technique: str, context: str | None = None, init: str | None = None,
             head: str = "laat", loss: str = "bce", device: str | torch.device = "cpu",
             ckpt_every: int = 2, attn: str | None = None, prior: torch.Tensor | None = None,
             n_labels: int = N_LABELS, compile_blocks: bool = False,
             cooc: torch.Tensor | None = None) -> LSIModel:
    from saul.model.modeling_saul import SaulForCausalLM

    path = init or resolve_checkpoint(name)
    backbone = SaulForCausalLM.from_pretrained(path)
    context = context or LSI_CONTEXT[name]
    partial = PARTIAL_LAYERS.get(name) if technique == "partial" else None
    apply_lsi_technique(backbone, technique, partial)
    if technique == "lorma":
        backbone.get_input_embeddings().weight.requires_grad_(True)
    set_attention(backbone, attn)
    if ckpt_every:
        enable_activation_checkpointing(backbone, ckpt_every)
    if compile_blocks:
        # In-place compile keeps state_dict keys free of the `_orig_mod.` prefix, so overlays stay portable.
        for block in backbone.blocks:
            block.compile(dynamic=None)
    backbone.to(device)
    d_model = backbone.config.d_model
    if head == "pooled":
        task_head = PooledMaxHead(d_model, n_labels)
    elif head == "decoder":
        width = 512 if d_model >= 512 else max(32, d_model)
        n_heads = 8 if width % 8 == 0 else 4
        task_head = LabelDecoderHead(d_model, n_labels, width=width, n_heads=n_heads, prior=prior, cooc=cooc)
    else:
        task_head = LAATHead(d_model, n_labels, prior=prior)
    task_head = task_head.to(device)
    return LSIModel(backbone, task_head, context, head, loss)


def trainable_parameters(model: LSIModel) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def overlay_state(model: LSIModel) -> dict:
    keep = {name for name, param in model.named_parameters() if param.requires_grad}
    tensors = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in keep or key.startswith("head.")
    }
    return {"context": model.context, "head": model.head_name, "loss": model.loss_name, "tensors": tensors}


def load_overlay(model: LSIModel, blob: dict) -> None:
    missing, unexpected = model.load_state_dict(blob["tensors"], strict=False)
    bad = [key for key in unexpected if key in blob["tensors"]]
    if bad:
        raise RuntimeError(f"unexpected overlay keys: {bad[:8]}")
    _ = missing


def tiny_saul(vocab: int = 128, d_model: int = 32, n_layers: int = 2, n_heads: int = 4):
    """A random dense Saul model for CPU tests. Not a checkpoint."""
    from saul.model.modeling_saul import SaulConfig, SaulForCausalLM

    cfg = SaulConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads,
        head_dim=d_model // n_heads, rope_dims=d_model // n_heads, ffn_dim=d_model * 2,
        max_position=256, attn_impl="sdpa",
    )
    return SaulForCausalLM(cfg)


def param_groups(model: LSIModel, technique: str, head_lr: float = 1e-3, adapter_lr: float = 5e-4,
                 base_lr: float = 2e-5, decay: float = 0.85, embed_lr: float = 1e-3) -> list[dict]:
    """Layer-decayed groups. Biases, norms and embeddings take no weight decay."""
    specs: list[tuple[torch.nn.Parameter, float, float]] = []
    fast = {"label_emb", "out_w", "out_bias"}

    def add(name: str, param: torch.nn.Parameter, lr: float) -> None:
        no_decay = param.ndim <= 1 or "emb" in name
        specs.append((param, lr, 0.0 if no_decay else 0.01))

    for name, param in model.head.named_parameters():
        if param.requires_grad:
            root = name.split(".")[0]
            add(name, param, embed_lr if root in fast else head_lr)
    peft = technique in ("lorma", "adapter", "adalora", "lora", "probe")
    if peft:
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                add(name, param, adapter_lr)
    else:
        n = len(model.backbone.blocks)
        for name, param in model.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("blocks."):
                idx = int(name.split(".")[1])
                lr = base_lr * (decay ** (n - 1 - idx))
            else:
                lr = base_lr * (decay ** n)
            add(name, param, lr)
    buckets: dict[tuple[float, float], list] = {}
    for param, lr, weight_decay in specs:
        buckets.setdefault((round(lr, 12), weight_decay), []).append(param)
    return [{"params": params, "lr": lr, "weight_decay": weight_decay} for (lr, weight_decay), params in buckets.items()]


def lr_mult(step: int, total: int, warmup: float = 0.05, floor: float = 0.0) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return floor + (1.0 - floor) * cosine
