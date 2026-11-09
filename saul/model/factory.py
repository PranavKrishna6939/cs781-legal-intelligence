"""Build a SaulConfig from a YAML file. `arch` fills in `layer_types` when they are omitted."""
from __future__ import annotations

from saul.model.modeling_saul import SaulConfig
from saul.utils.config import load_config, to_plain


def default_layer_types(arch: str, n_layers: int) -> list[str]:
    if arch == "hybrid":
        pattern = ["gdn", "gdn", "gdn", "attn"]
        return [pattern[i % 4] for i in range(n_layers)]
    if arch == "conv":
        attn_at = {3, 6, 8, 10, 12, 15}
        return ["attn" if i in attn_at else "conv" for i in range(n_layers)]
    return ["attn"] * n_layers


def config_from_yaml(path: str, overrides: list[str] | None = None) -> SaulConfig:
    raw = to_plain(load_config(path, overrides))
    model = raw.get("model", raw)
    arch = model.get("arch", "dense")
    n_layers = int(model.get("n_layers", 20))
    if not model.get("layer_types"):
        model["layer_types"] = default_layer_types(arch, n_layers)
    known = set(SaulConfig().to_dict()) if hasattr(SaulConfig(), "to_dict") else None
    if known:
        model = {k: v for k, v in model.items() if k in known or k in (
            "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "rope_dims",
            "rope_base", "ffn_dim", "ffn_type", "residual", "gr_branches", "gr_rank", "arch",
            "layer_types", "gdn_heads", "gdn_head_dim", "gdn_impl", "swa_window", "n_encoder_layers",
            "use_engram", "engram_heads", "engram_head_dim", "engram_table", "engram_orders",
            "sinkhorn_iters", "attn_impl", "logit_softcap", "bos_token_id", "eos_token_id",
            "pad_token_id", "mask_token_id", "tie_word_embeddings", "max_position", "clamp",
        )}
    return SaulConfig(**model)
