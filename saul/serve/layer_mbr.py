"""Compare native MBR at fixed backbone depths without generating new pools."""
import numpy as np
import torch
import torch.nn.functional as F

from saul.decode.mbr import mbr_pick
from saul.serve.finish import _f1, filter_candidates, postprocess, rouge_groups, specials_of
from saul.serve.packing import pack_prompts, segment_spans


@torch.inference_mode()
def layer_mbr_features(model, tokenizer, texts, reasons, layers, token_budget=32768):
    if not isinstance(texts,list) or not 1<=len(texts)<=64 or len(texts)!=len(reasons):
        raise ValueError('Need 1..64 candidates with matching reasons')
    if any(not isinstance(t,str) or len(t)>100000 for t in texts):
        raise ValueError('Invalid candidate text')
    if not isinstance(layers,list) or not 1<=len(layers)<=4 or len(set(layers))!=len(layers):
        raise ValueError('Need 1..4 unique one-based layer numbers')
    if any(type(n) is not int or not 1<=n<=len(model.blocks) for n in layers):
        raise ValueError('Layer outside backbone')
    if not 128<=token_budget<=32768:
        raise ValueError('token_budget outside 128..32768')
    kept=filter_candidates(texts,reasons,128)
    device=next(model.parameters()).device
    rows=[tokenizer.encode(t,add_special_tokens=False)[:512] or [tokenizer.pad_token_id] for t in kept]
    captures={}; segments={depth:[] for depth in layers}
    handles=[]; previous=model.config.attn_impl
    def capture(depth):
        def hook(module,inputs,output):
            captures[depth]=model._collapse(output)[0].clone()
        return hook
    try:
        model.config.attn_impl='sdpa'
        for depth in layers:
            handles.append(model.blocks[depth-1].register_forward_hook(capture(depth)))
        mode='bidirectional' if model.config.arch=='dense' else 'causal'
        start=0
        while start<len(rows):
            end=start;size=0
            while end<len(rows) and (end==start or size+len(rows[end])<=token_budget):
                size+=len(rows[end]);end+=1
            packed=pack_prompts(rows[start:end],tokenizer.pad_token_id,multiple=1)
            ids=torch.tensor([packed['ids']],device=device,dtype=torch.long)
            docs=torch.tensor([packed['docs']],device=device,dtype=torch.long)
            positions=torch.tensor([packed['pos']],device=device,dtype=torch.long)
            final=model._backbone(ids,positions,docs,mode)[0]
            # Include any post-block operations in the native final representation.
            if len(model.blocks) in layers: captures[len(model.blocks)]=final
            spans=[s for s in segment_spans(packed['docs']) if s[2]>=0]
            for depth in layers:
                hidden=F.normalize(captures.pop(depth).float(),dim=-1)
                segments[depth].extend(hidden[left:right] for left,right,_ in spans)
            start=end
    finally:
        for handle in handles: handle.remove()
        model.config.attn_impl=previous
    rouge=rouge_groups([kept],1)[0]
    out={}
    for depth in layers:
        matrix=.5*_f1(segments.pop(depth))+.5*rouge
        scores=(matrix.sum(1)-np.diag(matrix))/max(1,len(kept)-1)
        out[str(depth)]=dict(scores=scores.tolist(),selected=postprocess(
            kept[mbr_pick(matrix)],128,specials_of(tokenizer)))
    return dict(candidates=kept,layers=out)
