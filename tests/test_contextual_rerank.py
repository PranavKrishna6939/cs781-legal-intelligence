import torch
import torch.nn.functional as F
from saul.model.modeling_saul import build_model,tiny_config
from saul.serve.rerank import contextual_features


class Chars:
    pad_token_id=0
    bos_token_id=1
    def __call__(self,text,**kw):
        return {'input_ids':[ord(c)%100+4 for c in text],
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_contextual_logprob_boundary_and_batch_parity():
    torch.manual_seed(781)
    model=build_model(tiny_config('dense'),enforce_min=False).eval()
    tok=Chars(); prompt='Alice met Bob.'; candidates=[' Bob left.',' Alice stayed.']
    a=contextual_features(model,tok,prompt,candidates,batch_size=1)
    b=contextual_features(model,tok,prompt,candidates,batch_size=2)
    assert torch.allclose(torch.tensor(a['mean_logprob']),torch.tensor(b['mean_logprob']),atol=1e-5)
    text=prompt+' '+candidates[0]
    ids=torch.tensor([[tok.bos_token_id]+tok(text)['input_ids']]);start=len(prompt)+1
    logits=model(ids).logits
    expected=-F.cross_entropy(logits[0,start-1:-1],ids[0,start:])
    assert abs(a['mean_logprob'][0]-float(expected))<1e-5
    mat=torch.tensor(a['contextual_matrix'])
    assert torch.allclose(mat,mat.T,atol=1e-6)
    assert torch.isfinite(mat).all()
