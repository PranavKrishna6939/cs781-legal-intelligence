"""HF-compatible Saul language models: dense, hybrid, causal encoder-decoder, conv hybrid.

`labels` follow the HF convention (same length as `input_ids`, the token ids themselves, pad = -100).
Targets are built by `next_token_targets`: position t predicts labels[t+1] inside the same segment only.
With `loss_mask` (MNTP) only positions whose *next* input token was masked contribute (GPT-BERT).

Decoding uses `prefill` + `decode_step` with a `SaulCache`; every architecture decodes one token per step
without recomputing the prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from saul.model.attention import GatedAttention
from saul.model.cache import SaulCache
from saul.model.conv_block import ShortConvBlock
from saul.model.engram import Engram
from saul.model.gdn import GatedDeltaNet, set_varlen_context
from saul.model.lm_head import chunked_cross_entropy, next_token_targets, softcap
from saul.model.masks import build_attn_masks, resolve_impl
from saul.model.mlp import ReLU2, SwiGLU
from saul.model.norm import ZeroCenteredRMSNorm
from saul.model.residual import GatedResidual, MHC, PreNorm, StaticWidenedResidual
from saul.model.rope import rope_cache
from saul.utils.params import assert_param_budget, format_breakdown

try:
    import transformers
    from transformers import PretrainedConfig, PreTrainedModel
    # transformers 5 maps tied target -> source; 4.x takes a list of tied targets.
    _TIED = ({"lm_head.weight": "embed.weight"} if int(transformers.__version__.split(".")[0]) >= 5
             else ["lm_head.weight"])
except Exception:  # pragma: no cover
    PretrainedConfig = object
    PreTrainedModel = nn.Module
    _TIED = ["lm_head.weight"]


@dataclass
class SaulOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None

    def __getitem__(self, key: str):
        return getattr(self, key)


class SaulConfig(PretrainedConfig):
    model_type = "saul"

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1024,
        n_layers: int = 20,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        rope_dims: int = 32,
        rope_base: float = 10_000.0,
        ffn_dim: int = 2560,
        ffn_type: str = "swiglu",
        residual: str = "gr",
        gr_branches: int = 4,
        gr_rank: int = 64,
        arch: str = "dense",
        layer_types: list[str] | None = None,
        gdn_heads: int = 8,
        gdn_head_dim: int = 128,
        gdn_impl: str = "torch",
        swa_window: int = 0,
        n_encoder_layers: int = 0,
        use_engram: bool = False,
        engram_heads: int = 8,
        engram_head_dim: int = 32,
        engram_table: int = 100003,
        engram_orders: tuple[int, ...] = (2, 3),
        sinkhorn_iters: int = 10,
        attn_impl: str = "auto",
        attn_gate: bool = True,
        qk_norm: bool = True,
        logit_softcap: float = 0.0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        mask_token_id: int = 3,
        tie_word_embeddings: bool = True,
        max_position: int = 16384,
        clamp: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings, **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_dims = rope_dims
        self.rope_base = rope_base
        self.ffn_dim = ffn_dim
        self.ffn_type = ffn_type
        self.residual = residual
        self.gr_branches = gr_branches
        self.gr_rank = gr_rank
        self.arch = arch
        self.layer_types = layer_types or ["attn"] * n_layers
        self.gdn_heads = gdn_heads
        self.gdn_head_dim = gdn_head_dim
        self.gdn_impl = gdn_impl
        self.swa_window = swa_window
        self.n_encoder_layers = n_encoder_layers
        self.use_engram = use_engram
        self.engram_heads = engram_heads
        self.engram_head_dim = engram_head_dim
        self.engram_table = engram_table
        self.engram_orders = tuple(engram_orders)
        self.sinkhorn_iters = sinkhorn_iters
        self.attn_impl = attn_impl
        self.attn_gate = attn_gate
        self.qk_norm = qk_norm
        self.logit_softcap = logit_softcap
        self.mask_token_id = mask_token_id
        self.max_position = max_position
        self.clamp = clamp


def _ffn(cfg: SaulConfig) -> nn.Module:
    if cfg.ffn_type == "relu2":
        return ReLU2(cfg.d_model, cfg.ffn_dim, cfg.clamp)
    return SwiGLU(cfg.d_model, cfg.ffn_dim, cfg.clamp)


def _residual(cfg: SaulConfig, layer_idx: int) -> nn.Module:
    if cfg.residual == "prenorm":
        return PreNorm(cfg.d_model)
    if cfg.residual == "static":
        return StaticWidenedResidual(cfg.d_model, cfg.gr_branches, layer_idx)
    if cfg.residual == "mhc":
        return MHC(cfg.d_model, cfg.gr_branches, cfg.sinkhorn_iters)
    return GatedResidual(cfg.d_model, cfg.gr_branches, cfg.gr_rank)


class SaulBlock(nn.Module):
    """One mixer + FFN, each wrapped by its own residual module."""

    def __init__(self, cfg: SaulConfig, kind: str, layer_idx: int, window: int = 0):
        super().__init__()
        self.kind = kind
        self.window = window
        if kind == "attn":
            self.mixer = GatedAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dims, window,
                                        gate=cfg.attn_gate, qk_norm=cfg.qk_norm)
        elif kind == "gdn":
            self.mixer = GatedDeltaNet(cfg.d_model, cfg.gdn_heads, cfg.gdn_head_dim, impl=cfg.gdn_impl)
        elif kind == "conv":
            self.mixer = ShortConvBlock(cfg.d_model)
        else:
            raise ValueError(kind)
        self.ffn = _ffn(cfg)
        self.res1 = _residual(cfg, layer_idx)
        self.res2 = _residual(cfg, layer_idx)

    def forward(self, R, cos, sin, pos, doc_ids, cu_seqlens, mask, memory=None, cache=None,
                phase: str = "train", ctx=None):
        x, c = self.res1.read(R)
        if self.kind == "attn":
            y = self.mixer(x, cos, sin, pos, mask, memory, cache, phase, ctx)
        elif self.kind == "gdn":
            y = self.mixer(x, doc_ids, cu_seqlens, cache, phase)
        else:
            y = self.mixer(x, doc_ids, cache, phase)
        R = self.res1.write(R, y, c)
        x, c = self.res2.read(R)
        return self.res2.write(R, self.ffn(x), c)

    def serve_spec(self, ced_global: bool = False) -> dict:
        if self.kind == "attn":
            return self.mixer.serve_spec(ced_global)
        return self.mixer.serve_spec()

    def serve_export(self, R, cos, sin, pos, doc_ids, mask, memory):
        x, c = self.res1.read(R)
        if self.kind == "attn":
            y, exp = self.mixer.serve_export(x, cos, sin, pos, mask, memory)
        elif self.kind == "gdn":
            y, exp = self.mixer.serve_export(x, doc_ids)
        else:
            y, exp = self.mixer.serve_export(x, doc_ids)
        R = self.res1.write(R, y, c)
        x, c = self.res2.read(R)
        return self.res2.write(R, self.ffn(x), c), exp

    def serve_decode(self, R, cos, sin, pos, st, memory=None):
        x, c = self.res1.read(R)
        if self.kind == "attn":
            y = self.mixer.serve_decode(x, cos, sin, pos, st, memory)
        else:
            y = self.mixer.serve_decode(x, st)
        R = self.res1.write(R, y, c)
        x, c = self.res2.read(R)
        return self.res2.write(R, self.ffn(x), c)


class SaulForCausalLM(PreTrainedModel):
    config_class = SaulConfig
    _tied_weights_keys = _TIED

    def __init__(self, config: SaulConfig):
        super().__init__(config)
        self.config = config
        self.widened = config.residual != "prenorm"
        self.n_streams = config.gr_branches if self.widened else 1
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        kinds = list(config.layer_types)
        if len(kinds) != config.n_layers:
            raise ValueError("layer_types length must equal n_layers")
        self.is_ced = config.arch == "ced" and config.n_encoder_layers > 0
        blocks = []
        for i, kind in enumerate(kinds):
            window = 0
            if self.is_ced and kind == "attn" and (i < min(2, config.n_encoder_layers) or i >= config.n_encoder_layers):
                window = config.swa_window or 512
            blocks.append(SaulBlock(config, kind, i, window=window))
        self.blocks = nn.ModuleList(blocks)
        self.engram = None
        if config.use_engram:
            self.engram = Engram(config.engram_heads, config.engram_head_dim, config.engram_orders,
                                 config.engram_table, config.d_model, config.vocab_size)
        self.final_norm = ZeroCenteredRMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        self._rope_base = None
        if hasattr(self, "post_init"):
            self.post_init()

    def _init_weights(self, module):
        """No-op: every module initialises itself in its constructor, including the zero-initialised output maps
        that the transformers default (N(0, 0.02) for every Linear) would overwrite."""
        return

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    # ----------------------------------------------------------------------------------------------- helpers
    def _rope(self, device, need: int):
        base = self.config.rope_base
        n = max(int(self.config.max_position), int(need))
        c = self._rope_cos
        if (c.numel() == 0 or c.device != device or c.dtype != torch.float32 or self._rope_base != base
                or c.shape[0] < n):
            self._rope_cos, self._rope_sin = rope_cache(n, max(self.config.rope_dims, 2), base, device)
            self._rope_base = base
        return self._rope_cos, self._rope_sin

    def _streams(self, h: torch.Tensor) -> torch.Tensor:
        if not self.widened:
            return h
        return h.unsqueeze(2).expand(-1, -1, self.n_streams, -1).contiguous()

    def _collapse(self, R: torch.Tensor) -> torch.Tensor:
        return R.mean(dim=2) if self.widened else R

    def _mask_key(self, i: int, block: SaulBlock, mode: str) -> tuple[str, int]:
        if self.is_ced and i >= self.config.n_encoder_layers:
            return ("ced", block.window)
        if block.window:
            return ("window", block.window)
        return ("bidir", 0) if mode == "bidirectional" else ("causal", 0)

    def _engram_windows(self, input_ids, doc_ids, cache: SaulCache | None, phase: str) -> torch.Tensor:
        eng = self.engram
        if phase == "step":
            c = eng.compress[input_ids] + 1
            win = torch.cat([c, cache.engram["hist"]], dim=-1)[:, None, :]
        else:
            win = eng.windows(input_ids, doc_ids)
        if phase in ("prefill", "step"):
            cache.engram["hist"] = win[:, -1, :eng.max_order - 1]
        return win

    def _backbone(self, input_ids, pos, doc_ids, mode: str, cu_seqlens=None, cache: SaulCache | None = None,
                  phase: str = "train", impl: str | None = None, cu_seqlens_cpu=None) -> torch.Tensor:
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device
        cos, sin = self._rope(device, T if cache is None else cache.max_len)
        keys = {}
        masks: dict = {}
        if phase != "step":
            keys = {i: self._mask_key(i, b, mode) for i, b in enumerate(self.blocks) if b.kind == "attn"}
            if keys:
                masks = build_attn_masks(doc_ids, set(keys.values()), impl or resolve_impl(cfg.attn_impl, device),
                                         cu_seqlens)
        has_gdn = any(b.kind == "gdn" for b in self.blocks)
        if has_gdn:
            set_varlen_context(cu_seqlens if phase == "train" else None, cu_seqlens_cpu if phase == "train" else None)
        gdn_cu = None
        R = self._streams(self.embed(input_ids))
        memory = None
        for i, block in enumerate(self.blocks):
            lc = cache.layers[i] if cache is not None else None
            mem = memory if (self.is_ced and i >= cfg.n_encoder_layers) else None
            mask = masks.get(keys[i]) if i in keys else None
            R = block(R, cos, sin, pos, doc_ids, gdn_cu, mask, mem, lc, phase, cache)
            if self.engram is not None and i == 1:
                win = self._engram_windows(input_ids, doc_ids, cache, phase)
                delta = self.engram(self._collapse(R), win)
                R = R + (delta.unsqueeze(2) if self.widened else delta)
            if self.is_ced and i + 1 == cfg.n_encoder_layers:
                memory = self._collapse(R)
            # NER encoder-only runs set this. Unset, every Phase-1 forward is unchanged.
            if getattr(self, "_ner_stop_after", None) == i + 1:
                break
        if has_gdn:
            set_varlen_context(None)
        return self._collapse(R)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return softcap(self.lm_head(self.final_norm(hidden)), self.config.logit_softcap)

    # ----------------------------------------------------------------------------------------------- training
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        mode: str | int | torch.Tensor = "causal",
        cu_seqlens: torch.Tensor | None = None,
        return_hidden: bool = False,
        cu_seqlens_cpu: torch.Tensor | None = None,
        **kwargs,
    ) -> SaulOutput:
        if isinstance(mode, torch.Tensor):
            mode = int(mode.reshape(-1)[0])
        if mode in (1, "bidirectional"):
            mode = "bidirectional"
        else:
            mode = "causal"
        B, T = input_ids.shape
        if doc_ids is None:
            doc_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device)[None].expand(B, -1)
        hidden = self._backbone(input_ids, position_ids, doc_ids, mode, cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu)
        logits = self._logits(hidden)
        loss = None
        if labels is not None:
            loss = chunked_cross_entropy(logits, next_token_targets(labels, doc_ids, loss_mask))
        return SaulOutput(loss=loss, logits=logits, hidden=hidden if return_hidden else None)

    # ----------------------------------------------------------------------------------------------- decoding
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, valid: torch.Tensor, max_new: int) -> tuple[torch.Tensor, SaulCache]:
        """Left-padded prompts [B, T] with `valid` marking real tokens. Returns last-position logits and the cache."""
        B, T = input_ids.shape
        cache = SaulCache(len(self.blocks), B, T + max_new, input_ids.device)
        doc = torch.where(valid, torch.zeros_like(input_ids), torch.full_like(input_ids, -1))
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        cache.key_valid[:, :T] = valid
        hidden = self._backbone(input_ids, pos, doc, "causal", None, cache, "prefill", impl="sdpa")
        cache.length = T
        cache.next_pos = valid.long().sum(1)
        return self._logits(hidden[:, -1:])[:, 0], cache

    @torch.no_grad()
    def decode_step(self, next_ids: torch.Tensor, cache: SaulCache) -> torch.Tensor:
        """next_ids: [B]. Appends one token per row and returns logits [B, V] for the following position."""
        L = cache.length
        if L >= cache.max_len:
            raise RuntimeError(f"cache full ({cache.max_len} positions)")
        cache.key_valid[:, L] = True
        hidden = self._backbone(next_ids[:, None], cache.next_pos[:, None], None, "causal", None, cache, "step")
        cache.length = L + 1
        cache.next_pos = cache.next_pos + 1
        return self._logits(hidden)[:, 0]

    def serve_spec(self) -> tuple[list[dict], dict | None]:
        specs = []
        for i, block in enumerate(self.blocks):
            ced = self.is_ced and block.kind == "attn" and i >= self.config.n_encoder_layers
            specs.append(block.serve_spec(ced))
        eng = self.engram.serve_spec() if self.engram is not None else None
        return specs, eng

    @torch.no_grad()
    def serve_prefill_packed(self, input_ids, pos, doc_ids, cu=None, cu_cpu=None):
        """Packed prompts [1, T]. Returns last-token logits [T, V], per-layer exports, and Engram hist [T, H] or None.

        Exports use the training kernels (the same mask backend as a normal forward on this device).
        """
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device
        cos, sin = self._rope(device, max(T, int(cfg.max_position)))
        keys = {i: self._mask_key(i, b, "causal") for i, b in enumerate(self.blocks) if b.kind == "attn"}
        # SDPA, not flex: flex's unfused fallback materialises a [heads, T, T] score matrix.
        impl = "sdpa"
        masks = build_attn_masks(doc_ids, set(keys.values()), impl, cu) if keys else {}
        has_gdn = any(b.kind == "gdn" for b in self.blocks)
        if has_gdn:
            set_varlen_context(cu, cu_cpu)
        R = self._streams(self.embed(input_ids))
        memory = None
        exports = []
        engram_hist = None
        for i, block in enumerate(self.blocks):
            mem = memory if (self.is_ced and i >= cfg.n_encoder_layers) else None
            mask = masks.get(keys[i]) if i in keys else None
            R, exp = block.serve_export(R, cos, sin, pos, doc_ids, mask, mem)
            exports.append(exp)
            if self.engram is not None and i == 1:
                engram_hist = self.engram.serve_export_hist(input_ids, doc_ids)
                delta = self.engram(self._collapse(R), self.engram.windows(input_ids, doc_ids))
                R = R + (delta.unsqueeze(2) if self.widened else delta)
            if self.is_ced and i + 1 == cfg.n_encoder_layers:
                memory = self._collapse(R)
        if has_gdn:
            set_varlen_context(None)
        return self._logits(self._collapse(R))[0], exports, engram_hist

    @torch.no_grad()
    def serve_prefill_rows(self, input_ids, lengths):
        """Right-padded batch. Causal flash attention; padding sits to the right so it does not affect real tokens."""
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device)[None].expand(B, -1)
        doc = torch.zeros_like(input_ids)
        cos, sin = self._rope(device, max(T, int(cfg.max_position)))
        for block in self.blocks:
            if block.kind == "attn":
                block.mixer._flash_causal = True
        R = self._streams(self.embed(input_ids))
        memory = None
        exports = []
        try:
            for i, block in enumerate(self.blocks):
                mem = memory if (self.is_ced and i >= cfg.n_encoder_layers) else None
                R, exp = block.serve_export(R, cos, sin, pos, doc, None, mem)
                exports.append(exp)
                if self.is_ced and i + 1 == cfg.n_encoder_layers:
                    memory = self._collapse(R)
        finally:
            for block in self.blocks:
                if block.kind == "attn":
                    block.mixer._flash_causal = False
        hidden = self._collapse(R)
        last = (lengths - 1).clamp(min=0)
        logits = self._logits(hidden[torch.arange(B, device=device), last])
        return logits, exports

    @torch.no_grad()
    def serve_decode(self, ids: torch.Tensor, pos: torch.Tensor, st: dict) -> torch.Tensor:
        """ids, pos: [Sb]. `st` has `layers` (per-block pool views), `engram`, `block`. Logits [Sb, V]."""
        # RoPE tables are sized by the engine before the first step. Reading .max() here would sync.
        cos, sin = self._rope_cos, self._rope_sin
        R = self._streams(self.embed(ids)[:, None, :])
        memory = None
        pos2 = pos[:, None]
        for i, block in enumerate(self.blocks):
            mem = memory if (self.is_ced and i >= self.config.n_encoder_layers) else None
            R = block.serve_decode(R, cos, sin, pos2, st["layers"][i], mem)
            if self.engram is not None and i == 1:
                delta, new_hist = self.engram.serve_decode(self._collapse(R), ids, st["engram"])
                R = R + (delta.unsqueeze(2) if self.widened else delta)
                needs = st["layers"][0]["needs"].unsqueeze(-1)
                st["engram"].copy_(torch.where(needs, new_hist, st["engram"]))
            if self.is_ced and i + 1 == self.config.n_encoder_layers:
                memory = self._collapse(R)
        return self._logits(self._collapse(R))[:, 0]

    def set_rope_base(self, base: float) -> None:
        self.config.rope_base = base
        self._rope_base = None


def build_model(cfg: SaulConfig, enforce_budget: bool = True, enforce_min: bool = True) -> SaulForCausalLM:
    model = SaulForCausalLM(cfg)
    if enforce_budget:
        n = assert_param_budget(model, enforce_min=enforce_min)
        if os.environ.get("SAUL_LOG_PARAMS"):
            print(format_breakdown(model))
            print(f"total {n:,}")
    return model


def tiny_config(arch: str = "dense", **over) -> SaulConfig:
    base: dict[str, Any] = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, rope_dims=8,
        ffn_dim=128, residual="prenorm", arch=arch, layer_types=["attn", "attn"], max_position=256,
        gdn_heads=2, gdn_head_dim=16, engram_table=101, engram_heads=2, engram_head_dim=8,
    )
    if arch == "hybrid":
        base["layer_types"] = ["gdn", "gdn", "gdn", "attn"]
        base["n_layers"] = 4
        base["residual"] = "gr"
        base["gr_branches"] = 2
        base["gr_rank"] = 8
    elif arch == "ced":
        base["n_layers"] = 4
        base["n_encoder_layers"] = 2
        base["layer_types"] = ["attn"] * 4
        base["swa_window"] = 8
        base["use_engram"] = True
        base["residual"] = "mhc"
        base["gr_branches"] = 2
    elif arch == "conv":
        base["n_layers"] = 4
        base["layer_types"] = ["conv", "conv", "attn", "attn"]
    base.update(over)
    return SaulConfig(**base)
