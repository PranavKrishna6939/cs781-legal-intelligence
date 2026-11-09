"""Chunked layer features retain native controls and clean up hooks on failure."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
import torch
from saul.serve.large_layer_mbr import large_layer_mbr_features
from saul.serve.large_mbr import large_mbr_features
from saul.serve.layer_mbr import layer_mbr_features

helpers=runpy.run_path(str(Path(__file__).with_name('test_serve.py')))
class Blocks(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(781)
        self.embedding=torch.nn.Embedding(128,8)
        self.blocks=torch.nn.ModuleList([torch.nn.Linear(8,8),torch.nn.Linear(8,8)])
        self.norm=torch.nn.LayerNorm(8)
        self.config=SimpleNamespace(arch='hybrid',attn_impl='auto')
        self.sizes=[];self.fail=False
    def _collapse(self,x):return x
    def _backbone(self,ids,pos,docs,mode):
        self.sizes.append(int(docs.max())+1);x=self.embedding(ids)
        for block in self.blocks:
            x=block(x)
            if self.fail:raise RuntimeError('injected failure')
        return self.norm(x)

def test_chunked_layer_final_matches_existing_large_control():
    model=Blocks();tok=helpers['Tok']()
    texts=[' '.join(f'w{4+(i+j)%120}' for j in range(128)) for i in range(65)]
    result=large_layer_mbr_features(model,tok,texts,['words']*65,[1,2])
    assert model.sizes==[64,1]
    control=large_mbr_features(model,tok,texts,['words']*65)
    assert result['layers']['2']['selected']==control['selected']
    assert np.allclose(result['layers']['2']['scores'],control['scores'],atol=1e-6)
    assert result['layers']['2']['alpha_picks']['0.25']==control['alpha_picks']['0.25']
    small=texts[:3]
    actual=large_layer_mbr_features(model,tok,small,['words']*3,[1,2])
    native=layer_mbr_features(model,tok,small,['words']*3,[1,2])
    for depth in ['1','2']:
        assert actual['layers'][depth]['selected']==native['layers'][depth]['selected']
        assert np.allclose(actual['layers'][depth]['scores'],native['layers'][depth]['scores'],atol=1e-6)
    assert model.config.attn_impl=='auto' and not any(b._forward_hooks for b in model.blocks)

def test_failure_restores_attention_and_removes_hooks():
    model=Blocks();model.fail=True;tok=helpers['Tok']()
    try:large_layer_mbr_features(model,tok,[' '.join(['w4']*128)],['words'],[1,2])
    except RuntimeError as exc:assert str(exc)=='injected failure'
    else:raise AssertionError('Failure was swallowed')
    assert model.config.attn_impl=='auto' and not any(b._forward_hooks for b in model.blocks)

if __name__=='__main__':
    torch.set_num_threads(2)
    test_chunked_layer_final_matches_existing_large_control()
    test_failure_restores_attention_and_removes_hooks()
    assert not torch.cuda.is_initialized()
    print('large-layer CPU checks passed')
