"""Boundary masking and finite learning for continuation-only training."""
import importlib.util
from pathlib import Path

import torch
from saul.model.lm_head import next_token_targets
from saul.model.modeling_saul import build_model, tiny_config

spec = importlib.util.spec_from_file_location('sft', Path(__file__).resolve().parents[1]/'scripts/96_recovery_sft.py')
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


def test_continuation_boundary_padding_and_replay():
    rows = [dict(input_ids=[1, 4, 5, 6, 7], target_start=3),
            dict(input_ids=[1, 8, 9, 10], target_start=2)]
    batch = sft.batch_rows(rows, pad=0, length=8)
    target = next_token_targets(batch['labels'], batch['doc_ids'])
    assert target.tolist() == [[-100, -100, 6, 7, -100, -100, -100, -100],
                               [-100, 9, 10, -100, -100, -100, -100, -100]]
    assert batch['position_ids'][1].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert batch['cu_seqlens'].tolist() == [0, 5, 8, 12, 16]
    replay = sft.batch_rows(rows, pad=0, length=8, replay=True)
    assert next_token_targets(replay['labels'], replay['doc_ids'])[0, 0] == 4


def test_tiny_overfit_with_continuation_mask():
    torch.manual_seed(781)
    model = build_model(tiny_config('dense'), enforce_min=False).train()
    batch = sft.batch_rows([dict(input_ids=[1, 4, 5, 6, 7], target_start=3)], pad=0, length=8)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    values = []
    for _ in range(12):
        opt.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        values.append(float(loss.detach()))
    assert values[-1] < values[0]*.8, values
