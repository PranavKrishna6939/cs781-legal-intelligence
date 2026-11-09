"""Tiled greedy-match equivalence and bounded N128 encoding dispatch."""
import runpy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from saul.serve.finish import _f1,Finisher,specials_of
from saul.serve.large_mbr import tiled_f1,large_mbr_features


def test_tiled_f1_matches_full_with_padding_and_duplicates():
    torch.manual_seed(781)
    states=[F.normalize(torch.randn(n,16),dim=-1) for n in [1,3,8,2,11]]
    states.append(states[2].clone())
    for block in [1,2,4,8]:
        assert np.allclose(tiled_f1(states,block),_f1(states),atol=1e-6)
    assert np.array_equal(tiled_f1(states[:1]),np.zeros((1,1)))


def test_n128_dispatch_and_native_selection():
    helpers=runpy.run_path(str(Path(__file__).with_name('test_serve.py')))
    tok=helpers['Tok']()
    class IndependentStates(torch.nn.Module):
        def __init__(self):
            super().__init__();torch.manual_seed(781)
            self.embedding=torch.nn.Embedding(128,8)
            self.config=SimpleNamespace(arch='hybrid',attn_impl='auto')
            self.forward_sizes=[]
        def _backbone(self,ids,pos,docs,mode):
            self.forward_sizes.append(int(docs.max())+1)
            return self.embedding(ids)
        def forward(self,ids,**kwargs):
            return SimpleNamespace(hidden=self.embedding(ids))
    model=IndependentStates()
    texts=[' '.join(f'w{4+(i+j)%120}' for j in range(128)) for i in range(128)]
    result=large_mbr_features(model,tok,texts,['words']*128)
    assert model.forward_sizes==[64,64]
    assert result['candidate_count']==128 and len(result['selected'].split())==128
    assert result['alpha_picks']['0.5']==result['selected']
    assert np.allclose(result['scores'],.5*np.asarray(result['embedding_scores'])+.5*np.asarray(result['rouge_scores']),atol=1e-6)
    assert model.config.attn_impl=='auto'
    small=texts[:5]
    result=large_mbr_features(model,tok,small,['words']*5)
    expected=Finisher(model,tok,specials_of(tok)).choose([dict(
        texts=small,reasons=['words']*5,target=128,utility='mixed',alpha=.5,mbr=True)])[0]
    assert result['selected']==expected


if __name__=='__main__':
    torch.set_num_threads(2)
    test_tiled_f1_matches_full_with_padding_and_duplicates()
    test_n128_dispatch_and_native_selection()
    print('large MBR CPU checks passed')
