"""Document model shape, position counted from the end, and section masks."""
from __future__ import annotations

import torch

from saul.cjpe.hier import DocModel, section_mask


def test_doc_model_shape_and_position_from_the_end():
    torch.manual_seed(0)
    model = DocModel(in_dim=4, d_model=8, n_layers=1, n_heads=2, dropout=0.0)
    seen = {}
    original = model.pos.forward

    def wrapped(index):
        seen["index"] = index.detach().clone()
        return original(index)

    model.pos.forward = wrapped
    vectors = torch.randn(2, 4, 4)
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])
    logits, alpha = model(vectors, mask)
    assert logits.shape == (2, 2)
    assert alpha.shape == (2, 4)
    # Length 3 -> positions from the end are 2, 1, 0, and the pad is clamped.
    assert torch.equal(seen["index"][0, :3], torch.tensor([2, 1, 0]))
    assert torch.equal(seen["index"][1, :2], torch.tensor([1, 0]))


def test_section_masks():
    mask = torch.tensor([[True, True, True, True, False]])
    assert section_mask(mask, "first", 2).tolist() == [[True, True, False, False, False]]
    assert section_mask(mask, "last", 2).tolist() == [[False, False, True, True, False]]
    assert section_mask(mask, "middle", 2).tolist() == [[False, True, True, False, False]]
    assert section_mask(mask, "every_other", 1).tolist() == [[True, False, True, False, False]]
