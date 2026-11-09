"""Reference-free candidate features from the served model and its prompt context."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from saul.serve.finish import _f1


@torch.no_grad()
def contextual_features(model, tokenizer, prompt, candidates, batch_size=4, context_tokens=1024):
    if not candidates or len(candidates)>64 or not 1<=batch_size<=8:
        raise ValueError('Need 1..64 candidates and batch_size 1..8')
    if not 16<=context_tokens<=2048:
        raise ValueError('context_tokens must be 16..2048')
    device=next(model.parameters()).device
    rows=[]
    for text in candidates:
        joined=prompt+' '+text
        enc=tokenizer(joined,add_special_tokens=False,return_offsets_mapping=True)
        ids=list(enc['input_ids'])
        first=next((i for i,(_,end) in enumerate(enc['offset_mapping']) if end>len(prompt)),None)
        if first is None or first==0:
            raise ValueError('Empty prompt or continuation')
        bos=getattr(tokenizer,'bos_token_id',None)
        if bos is not None and ids[0]!=bos:
            ids=[bos]+ids
            first+=1
        left=max(0,first-context_tokens)
        ids=ids[left:]
        first-=left
        truncated=len(ids)-first>512
        ids=ids[:first+512]
        rows.append((ids,first,truncated))
    hidden=[]; logprob=[]; counts=[]
    prev_impl=model.config.attn_impl
    model.config.attn_impl='sdpa'
    try:
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            length=max(len(ids) for ids,_,_ in batch)
            x=torch.full((len(batch),length),tokenizer.pad_token_id,device=device,dtype=torch.long)
            docs=torch.full_like(x,-1)
            for i,(ids,_,_) in enumerate(batch):
                x[i,:len(ids)]=torch.tensor(ids,device=device)
                docs[i,:len(ids)]=0
            output=model(x,doc_ids=docs,mode='causal',return_hidden=True)
            for i,(ids,first,_) in enumerate(batch):
                n=len(ids)
                loss=F.cross_entropy(output.logits[i,first-1:n-1].float(),x[i,first:n],reduction='mean')
                logprob.append(-float(loss)); counts.append(n-first)
                hidden.append(F.normalize(output.hidden[i,first:n].float(),dim=-1).clone())
            del output
    finally:
        model.config.attn_impl=prev_impl
    return dict(contextual_matrix=_f1(hidden).tolist(),mean_logprob=logprob,target_tokens=counts,
                truncated_candidates=[i for i,r in enumerate(rows) if r[2]],context_tokens=context_tokens)
