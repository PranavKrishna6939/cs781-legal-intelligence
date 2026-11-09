"""Bounded-memory mixed MBR for saved pools, including N128.

The utility is unchanged token greedy-match F1 plus ROUGE. Encode at most 64
candidates per forward, matching the native N64 packing convention; tile pair
comparisons so N128 does not allocate an N*N*T*T similarity tensor at once.
"""
import torch
import torch.nn.functional as F

from saul.decode.mbr import mbr_pick
from saul.serve.finish import filter_candidates, postprocess, rouge_groups, specials_of
from saul.serve.packing import pack_prompts, segment_spans


def tiled_f1(segments, pair_block=8):
    n=len(segments)
    if n<=1:
        import numpy as np
        return np.zeros((n,n))
    length=max(len(h) for h in segments);dimension=segments[0].shape[-1]
    hidden=segments[0].new_zeros(n,length,dimension)
    valid=torch.zeros(n,length,device=hidden.device,dtype=torch.bool)
    for i,h in enumerate(segments):hidden[i,:len(h)]=h;valid[i,:len(h)]=True
    precision=hidden.new_empty(n,n)
    for left in range(0,n,pair_block):
        stop=min(n,left+pair_block)
        for right in range(0,n,pair_block):
            end=min(n,right+pair_block)
            similarity=torch.einsum('itd,jsd->ijts',hidden[left:stop],hidden[right:end])
            similarity.masked_fill_(~valid[None,right:end,None,:],-2.)
            best=similarity.max(-1).values
            precision[left:stop,right:end]=(best*valid[left:stop,None,:]).sum(-1)/valid[left:stop].sum(-1).clamp(min=1)[:,None]
    recall=precision.T
    return (2*precision*recall/(precision+recall).clamp(min=1e-8)).cpu().numpy()


@torch.inference_mode()
def large_mbr_features(model, tokenizer, texts, reasons, prompt=None):
    if not isinstance(texts,list) or not 1<=len(texts)<=128 or len(texts)!=len(reasons):
        raise ValueError('Need 1..128 candidates with matching reasons')
    if any(not isinstance(t,str) or len(t)>100000 for t in texts):raise ValueError('Invalid candidate text')
    if prompt is not None and (not isinstance(prompt,str) or len(prompt)>100000):
        raise ValueError('Invalid boundary prompt')
    kept=filter_candidates(texts,reasons,128)
    rows=[tokenizer.encode(t,add_special_tokens=False)[:512] or [tokenizer.pad_token_id] for t in kept]
    device=next(model.parameters()).device
    previous=model.config.attn_impl;segments=[]
    mode='bidirectional' if model.config.arch=='dense' else 'causal'
    try:
        model.config.attn_impl='sdpa'
        for start in range(0,len(rows),64):
            packed=pack_prompts(rows[start:start+64],tokenizer.pad_token_id,multiple=1)
            ids=torch.tensor([packed['ids']],device=device,dtype=torch.long)
            docs=torch.tensor([packed['docs']],device=device,dtype=torch.long)
            positions=torch.tensor([packed['pos']],device=device,dtype=torch.long)
            hidden=F.normalize(model._backbone(ids,positions,docs,mode)[0].float(),dim=-1)
            segments.extend(hidden[left:right] for left,right,doc in segment_spans(packed['docs']) if doc>=0)
    finally:
        model.config.attn_impl=previous
    embed=tiled_f1(segments)
    rouge=rouge_groups([kept],1)[0]
    matrix=.5*embed+.5*rouge
    scores=(matrix.sum(1)-matrix.diagonal())/max(1,len(kept)-1)
    embed_scores=(embed.sum(1)-embed.diagonal())/max(1,len(kept)-1)
    rouge_scores=(rouge.sum(1)-rouge.diagonal())/max(1,len(kept)-1)
    alpha_picks={str(alpha):postprocess(kept[mbr_pick(alpha*embed+(1-alpha)*rouge)],128,specials_of(tokenizer))
                 for alpha in [.25,.5,.75]}
    result=dict(candidate_count=len(kept),scores=scores.tolist(),
        selected=postprocess(kept[mbr_pick(matrix)],128,specials_of(tokenizer)),
        encoding_candidates_per_forward=64,pair_block=8,utility='mixed',alpha=.5,
        embedding_scores=embed_scores.tolist(),rouge_scores=rouge_scores.tolist(),alpha_picks=alpha_picks)
    if prompt is not None:
        from saul.serve.boundary import boundary_choice
        boundary=boundary_choice(kept,scores.tolist(),prompt,mbr_pick(matrix))
        boundary['selected']=postprocess(kept[boundary['selected_index']],128,specials_of(tokenizer))
        result['boundary']=boundary
    return result
