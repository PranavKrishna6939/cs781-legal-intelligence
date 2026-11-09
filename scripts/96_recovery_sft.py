#!/usr/bin/env python
"""Bounded gold-continuation SFT, fresh AdamW and unchanged causal LM format.

All decoding remains in the inference server; this script only trains and evaluates CE.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from saul.data.loader import cu_seqlens_np
from saul.model.lm_head import next_token_targets
from saul.model.modeling_saul import SaulForCausalLM


def batch_rows(rows, pad=2, length=2048, replay=False):
    ids = np.full((len(rows), length), pad, dtype=np.int64)
    labels = np.full_like(ids, -100)
    docs = np.full_like(ids, -1, dtype=np.int32)
    positions = np.zeros_like(ids)
    for i, r in enumerate(rows):
        seq, start = r['input_ids'], r['target_start']
        assert 0 < start < len(seq) <= length
        n = len(seq)
        ids[i, :n] = seq
        docs[i, :n] = 0
        positions[i, :n] = np.arange(n)
        begin = 1 if replay else start
        labels[i, begin:n] = seq[begin:]
    return dict(input_ids=torch.from_numpy(ids), labels=torch.from_numpy(labels),
                doc_ids=torch.from_numpy(docs), position_ids=torch.from_numpy(positions),
                cu_seqlens=torch.from_numpy(cu_seqlens_np(docs)))


def device_batch(batch, device):
    return {**{k: v.to(device) for k, v in batch.items()}, 'cu_seqlens_cpu': batch['cu_seqlens'], 'mode': 'causal'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--effective_batch', type=int, default=64)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--max_steps', type=int, default=0, help='Smoke only; fresh restart required afterward')
    p.add_argument('--seed', type=int, default=781)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = [json.loads(line) for line in Path(a.data).read_text().splitlines() if line]
    assert len({r['cluster'] for r in rows}) == len(rows)
    random.shuffle(rows)
    total = math.ceil(len(rows)/a.effective_batch)
    steps = min(total, a.max_steps) if a.max_steps else total
    model = SaulForCausalLM.from_pretrained(a.checkpoint).float().cuda().train()
    # Explicit groups avoid the pretrainer builder's hard-coded norm weight decay.
    groups = {'embed': [], 'matrix': [], 'norm_bias': []}
    for name, param in model.named_parameters():
        groups['embed' if name == 'embed.weight' else 'matrix' if param.ndim >= 2 else 'norm_bias'].append(param)
    opt = torch.optim.AdamW([
        dict(params=groups['embed'], lr=a.lr*.5, base_lr=a.lr*.5, weight_decay=0.0, name='embed'),
        dict(params=groups['matrix'], lr=a.lr, base_lr=a.lr, weight_decay=.01, name='matrix'),
        dict(params=groups['norm_bias'], lr=a.lr, base_lr=a.lr, weight_decay=0.0, name='norm_bias')],
        betas=(.9, .95), eps=1e-8, fused=True)
    if a.compile:
        for block in model.blocks:
            block.compile(dynamic=False)
    config = dict(vars(a), n_examples=len(rows), total_steps=total, executed_steps=steps,
                  input_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                  optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups],
                  replay='every fifth microbatch: ordinary causal LM loss on the same official-train excerpt',
                  schedule='3% warmup, linear decay to zero', precision='FP32 master weights, BF16 autocast')
    (out/'manifest.json').write_text(json.dumps(config, indent=2))
    save_steps = {math.ceil(total*.25), math.ceil(total*.5), total}
    warm = max(1, round(total*.03))
    start_time = time.monotonic()
    micro_number = 0
    with (out/'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(steps):
            block_rows = rows[step*a.effective_batch:(step+1)*a.effective_batch]
            micros = [block_rows[i:i+a.batch] for i in range(0, len(block_rows), a.batch)]
            opt.zero_grad(set_to_none=True)
            losses = []
            for part in micros:
                replay = micro_number % 5 == 4
                cpu = batch_rows(part, model.config.pad_token_id, replay=replay)
                shifted = next_token_targets(cpu['labels'], cpu['doc_ids'])
                for i, r in enumerate(part):
                    if not replay:
                        assert shifted[i, r['target_start']-1] == r['input_ids'][r['target_start']]
                        assert (shifted[i, :r['target_start']-1] == -100).all()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**device_batch(cpu, 'cuda')).loss
                weight = len(part)/len(block_rows)
                (loss*weight).backward()
                losses.append(dict(loss=float(loss.detach()), replay=replay, n=len(part)))
                micro_number += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            mult = (step+1)/warm if step < warm else max(0., (total-step-1)/max(1,total-warm))
            for g in opt.param_groups:
                g['lr'] = g['base_lr']*mult
            opt.step()
            rec = dict(step=step+1, examples=min((step+1)*a.effective_batch, len(rows)),
                       loss=sum(v['loss']*v['n'] for v in losses)/len(block_rows), micro_losses=losses,
                       gnorm=float(norm), lr_mult=mult, seconds=time.monotonic()-start_time)
            assert math.isfinite(rec['loss'])
            log.write(json.dumps(rec)+'\n')
            if (step+1) % 10 == 0 or step == 0:
                print(json.dumps({k:v for k,v in rec.items() if k != 'micro_losses'}), flush=True)
            if step+1 in save_steps and not a.max_steps:
                dest = out/f'ckpt_{step+1}'
                model.save_pretrained(dest)
                print('saved', dest, flush=True)
    if a.max_steps:
        print('SMOKE PASSED; no checkpoint saved. Restart from original weights.', flush=True)
    else:
        (out/'COMPLETE').write_text(str(total))
        print('COMPLETE', total, flush=True)


if __name__ == '__main__':
    main()
