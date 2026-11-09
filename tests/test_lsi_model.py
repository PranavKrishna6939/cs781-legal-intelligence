import torch

from saul.lsi.data import collate
from saul.lsi.decoder import LabelDecoderHead, db_ntr_loss
from saul.lsi.model import (
    LAATHead, LSIModel, PooledMaxHead, asl_loss, bce_loss, focal_loss, kd_loss, regroup_cases,
    tiny_saul,
)
from saul.lsi.train import run_steps


def _batch():
    cases = [
        {"id": "a", "labels": [1, 3], "chunks": [[4, 5, 6, 7], [8, 9, 10]]},
        {"id": "b", "labels": [2], "chunks": [[11, 12, 13, 14]]},
    ]
    return collate(cases, width=8)


def test_laat_shape_and_padding_gets_no_mass():
    head = LAATHead(16, n_labels=5, bottleneck=8)
    hidden = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    out = head(hidden, mask, return_extra=True)
    assert out["logits"].shape == (2, 5)
    assert torch.allclose(out["alpha"][0, :, 3:].sum(), torch.tensor(0.0), atol=1e-5)


def test_pooled_max_and_losses():
    head = PooledMaxHead(16, n_labels=5)
    hidden = torch.randn(3, 4, 16)
    docs = torch.ones(3, 4, dtype=torch.long)
    docs[:, -1] = -1
    case = torch.tensor([0, 0, 1])
    out = head(hidden, docs, case, n_cases=2)
    assert out["logits"].shape == (2, 5)
    logits = torch.randn(4, 5)
    labels = torch.randint(0, 2, (4, 5)).float()
    for loss in (bce_loss(logits, labels), asl_loss(logits, labels), focal_loss(logits, labels),
                 kd_loss(logits, labels, torch.sigmoid(torch.randn(4, 5)))):
        assert torch.isfinite(loss)


def test_regroup_concatenates_real_tokens():
    hidden = torch.arange(3 * 4 * 2, dtype=torch.float32).view(3, 4, 2)
    docs = torch.tensor([[0, 0, -1, -1], [0, 0, 0, -1], [0, -1, -1, -1]])
    case = torch.tensor([0, 0, 1])
    states, mask = regroup_cases(hidden, docs, case, 2)
    assert states.shape[0] == 2
    assert int(mask[0].sum()) == 5
    assert int(mask[1].sum()) == 1


def test_decoder_ignores_padding_and_db_matches_bce():
    head = LabelDecoderHead(16, n_labels=5, width=16, n_heads=4, n_slots=8, cooc=torch.rand(5, 5))
    head.eval()
    states = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    labels = torch.zeros(2, 8, 5)
    labels[:, :, 1] = 1
    batch = {"nbr_labels": labels, "nbr_sim": torch.rand(2, 8), "zlex": torch.randn(2, 5)}
    out = head(states, mask, batch, return_extra=True)
    assert out["logits"].shape == (2, 5)
    assert out["logits_deep"].shape == (2, 5)
    # Token padding is the first 6 keys. Neighbour slots follow them and may carry mass.
    assert torch.allclose(out["alpha"][0, :, :, 3:6].sum(), torch.tensor(0.0), atol=1e-4)
    logits = torch.randn(4, 5)
    target = torch.randint(0, 2, (4, 5)).float()
    target[:, 0] = 1
    got = db_ntr_loss(logits, target, torch.full((5,), 100.0), alpha=0.5, beta=0.0, lam=1.0, kappa=0.0)
    assert torch.allclose(got, bce_loss(logits, target), atol=1e-5)
    assert torch.isfinite(db_ntr_loss(logits, target, torch.tensor([10.0, 20, 50, 100, 400])))


def test_three_cpu_steps_on_a_tiny_backbone():
    torch.manual_seed(0)
    backbone = tiny_saul()
    model = LSIModel(backbone, LAATHead(backbone.config.d_model, n_labels=100, bottleneck=16),
                     "bidirectional", "laat", "bce")
    batch = _batch()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    last = run_steps(model, [batch, batch, batch], opt, "cpu")
    assert last == last  # finite
    decoder = LSIModel(
        tiny_saul(), LabelDecoderHead(32, n_labels=100, width=32, n_heads=4), "bidirectional", "decoder", "bce",
    )
    opt = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad], lr=1e-3)
    last = run_steps(decoder, [batch, batch, batch], opt, "cpu")
    assert last == last
