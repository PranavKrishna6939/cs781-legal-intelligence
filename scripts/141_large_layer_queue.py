#!/usr/bin/env python
"""Test N128 x layer-12 interaction after both candidate archives release the GPU."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
ARCH=Path('/data/cs781-archive/recovery_20261006')
OUT=ARCH/'large_layer_followup';CK=Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
BASE=ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'
def read(p):return json.loads(Path(p).read_text())
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline-utc',required=True);p.add_argument('--check-only',action='store_true')
    args=p.parse_args();os.chdir(ROOT);deadline=datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    assert read(ARCH/'layer_followup_queue/results.json')[-1]['partition']=='confirmation'
    if args.check_only:print('LARGE_LAYER_QUEUE_PREFLIGHT_PASSED; full pool checks wait for test archive');return
    OUT.mkdir(parents=True,exist_ok=False)
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def state(stage,**extra):
        row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=OUT/'state.json.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(OUT/'state.json')
        print(json.dumps(row),flush=True)
    def run(*cmd):
        assert remaining()>900,'Final 15-minute buffer reached'
        subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT,timeout=max(1,remaining()-900))
    try:
        state('waiting_for_test_candidate',deadline_utc=args.deadline_utc)
        while True:
            if remaining()<100*60:state('skipped_deadline',remaining_seconds=remaining());return
            prior=read(ARCH/'alpha025_test_candidate/state.json')
            if prior['stage'] in ['complete','failed','skipped_deadline']:break
            time.sleep(55)
        layer=read(ARCH/'layer_followup_queue/state.json')
        if prior['stage']!='complete' or not layer.get('promoted'):
            state('skipped_prerequisites',test_candidate_stage=prior['stage'],layer_promoted=layer.get('promoted'));return
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
        assert hashlib.sha256((CK/'model.safetensors').read_bytes()).hexdigest()=='4221aab1b309a6e22100c75b421be46a87eac704e4e7ba6d8c29b8e6b75772e0'
        old=module('nested141','125_nested_mbr128.py');q=module('queue141','99_recovery_training_queue.py')
        f=module('features141','119_jepa_witness_gate.py');f.CHECKPOINT=CK
        originals={'screen':BASE/'screen','confirmation':BASE/'confirmation','test':ROOT/'submissions/B_anneal_ema_mbr64_20261005'}
        prior_roots={'screen':ARCH/'nested_mbr128','confirmation':ARCH/'n128_alpha025_followup','test':ARCH/'alpha025_test_candidate'}
        prompt_paths={part:ROOT/f'runs/recovery_20261005/dev/{part}_prompts.json' for part in ['screen','confirmation']}
        prompt_paths['test']=ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'
        pools={};extras={};previous={}
        for part in ['screen','confirmation','test']:
            extra=prior_roots[part]/'extra64'/part
            for source,seed in [(originals[part],781),(extra,1781)]:
                m=read(source/'manifest.json');assert m['checkpoint']==str(CK) and m['seed']==seed and m['sampling']['n']==64
                assert m['prompts_sha256']==hashlib.sha256(prompt_paths[part].read_bytes()).hexdigest()
            pools[part]=old.pool(originals[part]);extras[part]=old.pool(extra)
            rows=[json.loads(line) for line in (prior_roots[part]/'selected128'/part/'features.jsonl').open()]
            previous[part]={r['id']:r['nested'] for r in rows}
            assert len(rows)==len(previous[part]) and set(pools[part])==set(extras[part])==set(previous[part])==set(read(prompt_paths[part]))
        protocol=dict(checkpoint=str(CK),layer=12,alphas=[.5,.25],n=128,seeds=[781,1781],
            incumbent='confirmed layer12 N64; also compare alpha025 N128 mean',
            screen_min_delta=.0005,confirmation_min_delta=.0005,confirmation_ci_lower_min=0,
            adaptive_reused_dev=True,interaction_not_assumed_additive=True,minimum_start_minutes=100,
            deadline_utc=args.deadline_utc,no_new_sampling=True)
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        def body(part,key):
            a,b=pools[part][key],extras[part][key]
            return dict(candidates=a['candidates']+b['candidates'],finish_reasons=a['finish_reasons']+b['finish_reasons'],layers=[12,20])
        def verify(part,key,result):
            new=result['layers']['20'];saved=previous[part][key]
            assert len(new['scores'])==len(saved['scores'])
            error=max(abs(x-y) for x,y in zip(new['scores'],saved['scores']))
            assert error<=1e-5,(part,key,error)
            assert all(new['alpha_picks'][a]==saved['alpha_picks'][a] for a in ['0.25','0.5']),('Last-layer selection mismatch',part,key)
            return error
        # Real GPU controls at N64 and N128 precede the full screen.
        f.OUT=OUT/'gpu_smoke';f.OUT.mkdir();state('gpu_parity_check')
        keys=sorted(pools['screen'],key=lambda k:hashlib.sha256(('layer128smoke:'+k).encode()).hexdigest())[:5]
        keys=list(dict.fromkeys(keys+[max(pools['screen'],key=lambda k:sum(map(len,pools['screen'][k]['candidates'])))]))
        native12=read(ARCH/'layer_mbr_exact/screen/layer_12/prediction.json');checks=[]
        with f.server():
            for key in keys:
                record=pools['screen'][key]
                small=f.post('/v1/large_layer_mbr_features',dict(candidates=record['candidates'],finish_reasons=record['finish_reasons'],layers=[12,20]))
                assert small['layers']['12']['selected']==native12[key]['text']
                assert small['layers']['20']['selected']==record['text']
                result=f.post('/v1/large_layer_mbr_features',body('screen',key))
                checks.append(dict(id=key,max_final_layer_score_error=verify('screen',key,result)))
        (f.OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        def select(part,names):
            start=time.monotonic();dest=OUT/part;dest.mkdir()
            prompts=read(prompt_paths[part]);results={}
            f.OUT=dest/'server';f.OUT.mkdir();state('features',partition=part)
            with f.server(),(dest/'features.jsonl').open('w') as log:
                for i,key in enumerate(prompts):
                    assert remaining()>900,'Final upload buffer reached'
                    result=f.post('/v1/large_layer_mbr_features',body(part,key));error=verify(part,key,result)
                    results[key]=result
                    log.write(json.dumps(dict(id=key,max_final_layer_score_error=error,**result))+'\n');log.flush()
                    if (i+1)%50==0:state('features',partition=part,done=i+1,total=len(prompts))
            outputs=[]
            for name in names:
                alpha=.5 if name=='alpha_05' else .25
                path=dest/name;path.mkdir()
                pred={k:{'text':results[k]['layers']['12']['alpha_picks'][str(alpha)]} for k in prompts}
                (path/'prediction.json').write_text(json.dumps(pred,ensure_ascii=False))
                manifest=read(originals[part]/'manifest.json');manifest['sampling']=dict(manifest['sampling'],n=128,mbr=dict(enabled=True,utility='mixed',alpha=alpha))
                manifest.update(seed=None,pool_seeds=[781,1781],selection_pending=False,layer=12,selector='large_layer_mbr',
                    component_pools=[str(originals[part]),str(prior_roots[part]/'extra64'/part)],
                    component_candidates_sha256=[hashlib.sha256((s/'candidates.jsonl').read_bytes()).hexdigest() for s in [originals[part],prior_roots[part]/'extra64'/part]],
                    incomplete_outputs={k:len(v['text'].split()) for k,v in pred.items() if len(v['text'].split())!=128})
                (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');outputs.append(path)
            return outputs,time.monotonic()-start
        def evaluate(part,names):
            outputs,elapsed=select(part,names)
            sources=[ARCH/f'layer_mbr_exact/{part}/layer_12',ARCH/'cached_aux_screen/n128_alpha_025' if part=='screen' else ARCH/'n128_alpha025_followup/selected128/confirmation']
            anchors=[]
            for i,source in enumerate(sources):
                path=OUT/part/f'incumbent_{i}';path.mkdir()
                for name in ['prediction.json','manifest.json']:
                    if (source/name).exists():(path/name).write_bytes((source/name).read_bytes())
                anchors.append(path)
            state('score',partition=part)
            run(q.PY,'-u','scripts/92_recovery_score.py','--truth',ROOT/f'runs/recovery_20261005/dev/{part}_truth.json',
                '--meta',ROOT/f'runs/recovery_20261005/dev/{part}_meta.json','--runs',*anchors,*outputs)
            rows=[]
            for path in outputs:
                score=read(path/'scores.json');score.update(partition=part,name=path.name,
                    delta_vs_alpha025=score['bertscore']-read(anchors[1]/'scores.json')['bertscore'],feature_seconds=elapsed)
                rows.append(score)
            return rows
        names=['alpha_05','alpha_025'];start=time.monotonic();rows=evaluate('screen',names)
        (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');best=max(rows,key=lambda r:r['bertscore'])
        promoted=False;submission=None
        if min(best['delta_vs_first'],best['delta_vs_alpha025'])>=.0005:
            estimate=(time.monotonic()-start)*2000/385+15*60
            if remaining()<estimate+900:state('screen_passed_deadline_stop',estimated_seconds=estimate,remaining_seconds=remaining());return
            conf=evaluate('confirmation',[best['name']])[0];rows.append(conf)
            (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            promoted=min(conf['delta_vs_first'],conf['delta_vs_alpha025'])>=.0005 and conf['paired_cluster_95ci'][0]>0
            if promoted:
                paths,_=select('test',[best['name']]);source=paths[0];pred=read(source/'prediction.json')
                submission=ROOT/f'submissions/B_layer12_mbr128_{best["name"]}_20261006';submission.mkdir(exist_ok=False)
                path=submission/'prediction.json';path.write_text(json.dumps(pred,indent=2,ensure_ascii=True))
                assert list(pred)==[str(i) for i in range(1,1001)]
                check=module('packer141','51_make_submission.py').check_prediction(str(path),read(prompt_paths['test']),128)
                assert check['n']==check['exact_128']==1000 and check['special_token_leaks']==0
                archive=submission/'submission.zip'
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(path,arcname='prediction.json')
                with zipfile.ZipFile(archive) as z:assert z.namelist()==['prediction.json'] and z.testzip() is None and z.read('prediction.json')==path.read_bytes()
                manifest=read(source/'manifest.json');manifest.update(validation=check,dev_confirmation=conf,
                    zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (submission/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (submission/'protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
                (submission/'state.json').write_text(json.dumps(dict(stage='complete',zip=str(archive)),indent=2)+'\n')
        state('complete',promoted=promoted,submission=str(submission) if submission else None,result_rows=rows)
    except BaseException as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':main()
