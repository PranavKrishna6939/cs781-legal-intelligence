#!/usr/bin/env python
"""Bounded gold CE or gold-anchored DPO on official-text identity corruptions.

No generator or metric model is used here. Generation evaluation is a separate
request to the inference server. Both modes share examples and update budgets.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from saul.model.modeling_saul import SaulForCausalLM
from saul.model.lm_head import next_token_targets

spec=importlib.util.spec_from_file_location('sft',Path(__file__).with_name('96_recovery_sft.py'))
sft=importlib.util.module_from_spec(spec);spec.loader.exec_module(sft)


def continuation_logps(logits,batch):
    targets=next_token_targets(batch['labels'],batch['doc_ids'])
    mask=targets!=-100
    losses=F.cross_entropy(logits.float().transpose(1,2),targets,reduction='none',ignore_index=-100)
    return -losses.sum(1),mask.sum(1)


def anchored_dpo(policy,reference,counts,beta=.1,ce_weight=.2):
    assert len(policy)%2==0
    n=len(policy)//2
    advantage=(policy[:n]-policy[n:])-(reference[:n]-reference[n:])
    preference=-F.logsigmoid(beta*advantage).mean()
    ce=-(policy[:n]/counts[:n]).mean()
    return preference+ce_weight*ce,preference,ce


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--mode',choices=['ce','dpo'],required=True)
    p.add_argument('--updates',type=int,default=64);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--effective_batch',type=int,default=32);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--beta',type=float,default=.1);p.add_argument('--ce_weight',type=float,default=.2)
    p.add_argument('--compile',action='store_true')
    a=p.parse_args();torch.manual_seed(781);rng=random.Random(781)
    assert a.updates>0 and a.batch>0 and a.effective_batch>0
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from transformers import PreTrainedTokenizerFast
    tok=PreTrainedTokenizerFast.from_pretrained('artifacts/tokenizers/legal32k')
    raw=[json.loads(l) for l in Path(a.data).read_text().splitlines()]
    assert len({r['cluster'] for r in raw})==len(raw)
    rng.shuffle(raw)
    def encode(prompt,target):
        enc=tok(prompt+' '+target,add_special_tokens=False,return_offsets_mapping=True)
        first=next(i for i,(_,end) in enumerate(enc['offset_mapping']) if end>len(prompt))
        return dict(input_ids=[tok.bos_token_id]+enc['input_ids'],target_start=first+1)
    rows=[]
    for r in raw:
        chosen=encode(r['prompt'],r['chosen']);rejected=encode(r['prompt'],r['rejected'])
        if max(len(chosen['input_ids']),len(rejected['input_ids']))<=2048:rows.append((chosen,rejected))
    steps=min(a.updates,math.ceil(len(rows)/a.effective_batch));assert steps>0
    model=SaulForCausalLM.from_pretrained(a.checkpoint).float().cuda().train()
    reference=None
    if a.mode=='dpo':
        reference=SaulForCausalLM.from_pretrained(a.checkpoint).float().cuda().eval()
        reference.requires_grad_(False)
    groups=[dict(params=[v for n,v in model.named_parameters() if v.ndim>=2 and n!='embed.weight'],weight_decay=.01,lr=a.lr,base_lr=a.lr),
            dict(params=[v for n,v in model.named_parameters() if v.ndim<2],weight_decay=0.,lr=a.lr,base_lr=a.lr),
            dict(params=[model.embed.weight],weight_decay=0.,lr=a.lr*.5,base_lr=a.lr*.5)]
    opt=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,fused=True)
    if a.compile:
        for block in model.blocks:block.compile(dynamic=False)
    (out/'manifest.json').write_text(json.dumps(dict(vars(a),steps=steps,examples=len(rows),
        data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        reference='frozen starting checkpoint; summed continuation log probabilities',
        gold_ce='mean over target tokens per example',seed=781),indent=2))
    began=time.monotonic();warm=max(1,round(steps*.05))
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(steps):
            block=rows[step*a.effective_batch:(step+1)*a.effective_batch]
            opt.zero_grad(set_to_none=True);total=0.
            for j in range(0,len(block),a.batch):
                part=block[j:j+a.batch]
                seq=[r[0] for r in part]+([r[1] for r in part] if reference is not None else [])
                cpu=sft.batch_rows(seq,model.config.pad_token_id)
                batch=sft.device_batch(cpu,'cuda')
                # The LM built-in loss is unnecessary; compute exact sequence
                # likelihoods from logits with our validated shifted targets.
                inputs={k:v for k,v in batch.items() if k!='labels'}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if reference is not None:
                        with torch.no_grad():ref,_=continuation_logps(reference(**inputs).logits,batch)
                    policy,counts=continuation_logps(model(**inputs).logits,batch)
                    if reference is not None:loss,_,_=anchored_dpo(policy,ref,counts,a.beta,a.ce_weight)
                    else:loss=-(policy/counts).mean()
                (loss*(len(part)/len(block))).backward();total+=float(loss.detach())*len(part)/len(block)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            mult=(step+1)/warm if step<warm else (steps-step-1)/max(1,steps-warm)
            for g in opt.param_groups:g['lr']=g['base_lr']*mult
            opt.step()
            row=dict(step=step+1,loss=total,gnorm=float(norm),lr_mult=mult,seconds=time.monotonic()-began)
            log.write(json.dumps(row)+'\n')
            if step%10==0:print(json.dumps(row),flush=True)
            if step+1 in {math.ceil(steps/2),steps}:model.save_pretrained(out/f'ckpt_{step+1}')
    (out/'COMPLETE').write_text(str(steps));print('COMPLETE',steps,flush=True)


if __name__=='__main__':main()
