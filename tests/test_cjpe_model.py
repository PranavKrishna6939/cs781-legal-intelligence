"""Pooling head, logit-adjusted loss, and activation checkpointing."""
from __future__ import annotations

import torch

from saul.cjpe.const import PRIOR0, PRIOR1
from saul.cjpe.model import PoolHead, enable_activation_checkpointing, kd_loss, logit_adjusted_ce
from saul.model.modeling_saul import SaulConfig, SaulForCausalLM


def test_pool_head_masks_padding_and_uses_cls():
    torch.manual_seed(0)
    head = PoolHead(8, dropout=0.0, use_cls=True)
    hidden = torch.randn(2, 5, 8)
    doc = torch.tensor([[0, 0, 0, -1, -1], [0, 0, -1, -1, -1]])
    logits = head(hidden, doc)
    assert logits.shape == (2, 2)
    attn, mean, last = head.parts(hidden, doc)
    assert torch.allclose(last[0], hidden[0, 2])
    assert torch.allclose(last[1], hidden[1, 1])
    assert mean.shape == attn.shape == (2, 8)


def test_logit_adjustment_shifts_by_the_train_prior():
    logits = torch.zeros(4, 2)
    labels = torch.tensor([0, 0, 1, 1])
    adjusted = logit_adjusted_ce(logits, labels)
    manual = torch.nn.functional.cross_entropy(logits + torch.log(logits.new_tensor([PRIOR0, PRIOR1])), labels)
    assert torch.allclose(adjusted, manual)
    teacher = torch.full((4, 2), 0.5)
    value = kd_loss(logits, teacher, labels, 2.0)
    assert torch.isfinite(value)


def test_checkpoint_matches_the_plain_forward():
    cfg = SaulConfig(
        vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8, rope_dims=8,
        ffn_dim=64, residual="prenorm", max_position=32, attn_impl="sdpa", layer_types=["attn", "attn"],
    )
    torch.manual_seed(0)
    plain = SaulForCausalLM(cfg)
    torch.manual_seed(0)
    checked = SaulForCausalLM(cfg)
    checked.load_state_dict(plain.state_dict())
    enable_activation_checkpointing(checked, every=1)
    ids = torch.randint(4, 128, (2, 8))
    docs = torch.zeros_like(ids)
    plain.train()
    checked.train()
    out_plain = plain(ids, doc_ids=docs, mode="bidirectional", return_hidden=True).hidden
    out_checked = checked(ids, doc_ids=docs, mode="bidirectional", return_hidden=True).hidden
    assert torch.allclose(out_plain, out_checked, atol=1e-5)
    loss = out_checked.sum()
    loss.backward()
    assert checked.embed.weight.grad is not None
