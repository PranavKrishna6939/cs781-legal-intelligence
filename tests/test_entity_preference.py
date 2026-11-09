import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('pref',Path(__file__).resolve().parents[1]/'scripts/106_entity_preference.py')
pref=importlib.util.module_from_spec(spec);spec.loader.exec_module(pref)


def test_dpo_gradient_and_reference_cancellation():
    policy=torch.tensor([-10.,-12.,-11.,-14.],requires_grad=True)
    reference=policy.detach().clone();counts=torch.tensor([5,6,5,6])
    loss,dpo,ce=pref.anchored_dpo(policy,reference,counts,ce_weight=0.)
    assert abs(float(dpo.detach())-0.69314718)<1e-6
    loss.backward()
    assert (policy.grad[:2]<0).all() and (policy.grad[2:]>0).all()


def test_logps_ignore_prompt_padding_and_include_first_target():
    rows=[dict(input_ids=[1,5,6,7],target_start=2),dict(input_ids=[1,8,9],target_start=1)]
    batch=pref.sft.batch_rows(rows,length=6)
    logits=torch.zeros(2,6,16)
    logps,counts=pref.continuation_logps(logits,batch)
    assert counts.tolist()==[2,2]
    assert torch.allclose(logps,torch.tensor([-2*2.772588722]*2))
    logits[0,1,6]=10
    improved,_=pref.continuation_logps(logits,batch)
    assert improved[0]>logps[0] and improved[1]==logps[1]


def test_tiny_model_preference_update_increases_chosen_margin():
    import copy
    from saul.model.modeling_saul import build_model,tiny_config
    torch.manual_seed(781)
    model=build_model(tiny_config('dense'),enforce_min=False).train()
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    rows=[dict(input_ids=[1,5,6,7,8],target_start=3),dict(input_ids=[1,5,6,9,10],target_start=3)]
    batch=pref.sft.batch_rows(rows,length=8)
    inputs={k:v for k,v in batch.items() if k!='labels'}
    with torch.no_grad():ref,counts=pref.continuation_logps(reference(**inputs).logits,batch)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        policy,_=pref.continuation_logps(model(**inputs).logits,batch)
        loss,_,_=pref.anchored_dpo(policy,ref,counts,ce_weight=.2)
        assert torch.isfinite(loss)
        loss.backward();opt.step()
    with torch.no_grad():after,_=pref.continuation_logps(model(**inputs).logits,batch)
    assert after[0]-after[1]>ref[0]-ref[1]
