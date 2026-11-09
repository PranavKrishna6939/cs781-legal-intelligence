"""Feature geometry, packing isolation and HTTP baseline parity checks."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from saul.serve.latent import latent_features, make_bank, witnesses
from saul.serve.server import build_app
from saul.serve.finish import Finisher, specials_of

helpers = runpy.run_path(str(Path(__file__).with_name('test_serve.py')))


def test_hidden_parity_packing_and_special_exclusion():
    tok = helpers['Tok']()
    for arch in ('dense', 'hybrid', 'ced', 'conv'):
        model = helpers['_model'](arch)
        original = model.config.attn_impl
        texts = ['w4 w5 w6', 'w8 w9 w10 w11 w12']
        packed = latent_features(model, tok, texts, sample_tokens=32)
        for text, actual in zip(texts, packed):
            ids = torch.tensor([tok.encode(text)])
            model.config.attn_impl = 'sdpa'
            mode = 'bidirectional' if arch == 'dense' else 'causal'
            expected = model(ids, mode=mode, return_hidden=True).hidden[0]
            model.config.attn_impl = original
            assert torch.allclose(torch.tensor(actual['samples']), expected, atol=1e-5), arch
        p = latent_features(model, tok, ['w4 w5 w6'], kind='prefix', sample_tokens=32)[0]
        expected = model(torch.tensor([[1,4,5,6]]), return_hidden=True).hidden[0,1:]
        assert p['tokens'] == 3
        assert torch.allclose(torch.tensor(p['samples']), expected, atol=1e-5), arch
        assert model.config.attn_impl == original


def test_witness_math_and_bank_validation():
    bank = make_bank({'center':[0.,0.], 'anchors':[[1.,0.],[0.,1.]], 'tau':.1}, 2, 'cpu')
    support, mass = witnesses(torch.eye(2), bank)
    assert torch.allclose(support, torch.ones(2))
    assert torch.allclose(mass, torch.full((2,), .5))
    # Both features ignore identical repetition, avoiding raw length reward.
    again = witnesses(torch.eye(2).repeat(3,1), bank)
    assert all(torch.allclose(a,b) for a,b in zip((support,mass),again))
    try:
        make_bank({'center':[0.,0.], 'anchors':[[0.,0.],[0.,1.]]}, 2, 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted zero anchor')


def test_http_features_and_native_selection():
    model = helpers['_model']('hybrid')
    engine = helpers['_engine'](model)
    client = TestClient(build_app(SimpleNamespace(engine=engine)))
    d = model.config.d_model
    bank = {'center':[0.]*d, 'anchors':torch.eye(d)[:4].tolist(), 'tau':.1}
    bank_id = client.post('/v1/latent_bank',json=bank).json()['bank_id']
    response = client.post('/v1/latent_features',json={'texts':['w4 w5'],'bank_id':bank_id})
    assert response.status_code == 200
    assert abs(sum(response.json()['features'][0]['mass'])-1)<1e-6
    assert client.post('/v1/latent_features',json={'texts':['w4'],'bank_id':'bad'}).status_code == 400
    assert client.post('/v1/latent_features',json={'texts':['w1 w2']}).status_code == 400
    assert client.post('/v1/latent_features',json={'texts':['w4']*65}).status_code == 400
    candidates = [' '.join([f'w{i}']*128) for i in [4,5,6]]
    data = {'candidates':candidates,'finish_reasons':['words']*3}
    result = client.post('/v1/mbr_features',json=data)
    assert result.status_code == 200, result.text
    expected = Finisher(model,engine.tok,specials_of(engine.tok)).choose([
        dict(texts=candidates,reasons=['words']*3,target=128,utility='mixed',alpha=.5,mbr=True)])[0]
    assert result.json()['selected'] == expected
    large=client.post('/v1/large_mbr_features',json=data)
    assert large.status_code==200,large.text
    assert large.json()['selected']==expected
    assert torch.allclose(torch.tensor(large.json()['scores']),torch.tensor(result.json()['scores']),atol=1e-6)
    diagnostic=client.post('/v1/large_mbr_features',json=dict(**data,prompt='w4 w5'))
    assert diagnostic.status_code==200,diagnostic.text
    assert diagnostic.json()['selected']==large.json()['selected']
    assert diagnostic.json()['scores']==large.json()['scores']
    assert diagnostic.json()['boundary']['selected']==large.json()['selected']
    assert client.post('/v1/large_mbr_features',json=dict(**data,prompt=123)).status_code==400
    assert client.post('/v1/large_mbr_features',json={'candidates':candidates*43,'finish_reasons':['words']*129}).status_code==400
    depth=len(model.blocks)
    layered=client.post('/v1/layer_mbr_features',json=dict(**data,layers=[1,depth]))
    assert layered.status_code == 200, layered.text
    assert layered.json()['layers'][str(depth)]['selected']==expected
    assert torch.allclose(torch.tensor(layered.json()['layers'][str(depth)]['scores']),
                          torch.tensor(result.json()['scores']),atol=1e-6)
    assert not any(block._forward_hooks for block in model.blocks)
    from saul.serve.layer_mbr import layer_mbr_features
    chunked=layer_mbr_features(model,engine.tok,candidates,['words']*3,[1,depth],token_budget=128)
    for layer in ['1',str(depth)]:
        assert torch.allclose(torch.tensor(chunked['layers'][layer]['scores']),
                              torch.tensor(layered.json()['layers'][layer]['scores']),atol=1e-6)
    assert client.post('/v1/layer_mbr_features',json=dict(**data,layers=[0])).status_code==400
    large_layered=client.post('/v1/large_layer_mbr_features',json=dict(**data,layers=[1,depth]))
    assert large_layered.status_code==200,large_layered.text
    for layer in ['1',str(depth)]:
        assert large_layered.json()['layers'][layer]['selected']==layered.json()['layers'][layer]['selected']
        assert torch.allclose(torch.tensor(large_layered.json()['layers'][layer]['scores']),
                              torch.tensor(layered.json()['layers'][layer]['scores']),atol=1e-6)
    assert client.post('/v1/large_layer_mbr_features',json=dict(**data,layers=[0])).status_code==400
    assert not any(block._forward_hooks for block in model.blocks)
    assert client.get('/health').status_code == 200


if __name__ == '__main__':
    torch.set_num_threads(2)
    test_hidden_parity_packing_and_special_exclusion()
    test_witness_math_and_bank_validation()
    test_http_features_and_native_selection()
    print('latent feature tests passed')
