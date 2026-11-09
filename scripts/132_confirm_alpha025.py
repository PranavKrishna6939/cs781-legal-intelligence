#!/usr/bin/env python
"""Confirm the predeclared cached-screen winner, N128 alpha .25, via Saul HTTP."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data/cs781-archive/recovery_20261006/n128_alpha025_followup')
CK=Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
BASE=ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'
BOUNDARY=Path('/data/cs781-archive/recovery_20261006/boundary_sparse_cap40')
AUX=Path('/data/cs781-archive/recovery_20261006/cached_aux_screen')
NESTED=Path('/data/cs781-archive/recovery_20261006/nested_mbr128')

def read(path): return json.loads(Path(path).read_text())
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline-utc',required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    os.chdir(ROOT);deadline=datetime.fromisoformat(a.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    old=module('nested132','125_nested_mbr128.py')
    q=module('queue132','99_recovery_training_queue.py')
    f=module('feature132','119_jepa_witness_gate.py');f.CHECKPOINT=CK
    variants=read(AUX/'prepared.json')['runs']
    winner=max(variants,key=lambda path:read(Path(path)/'scores.json')['bertscore'])
    assert Path(winner).name=='n128_alpha_025'
    screen=read(Path(winner)/'scores.json');assert screen['delta_vs_first']>=.0005
    checks=read(NESTED/'gpu_smoke/checks.json')
    assert len(checks)>=5 and all(c['max_score_difference']<=1e-5 and c['tiled_selection_matches_native'] for c in checks)
    for arm in ['fresh_hc_anneal','repaired_anneal','final_decode_queue']:
        assert read(ROOT/f'runs/recovery_20261006/{arm}/state.json')['stage']=='complete'
    sources={'confirmation':BASE/'confirmation','test':ROOT/'submissions/B_anneal_ema_mbr64_20261005'}
    prompts_paths={'confirmation':ROOT/'runs/recovery_20261005/dev/confirmation_prompts.json',
                  'test':ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'}
    for part,source in sources.items():
        manifest=read(source/'manifest.json')
        assert manifest['checkpoint']==str(CK) and manifest['seed']==781 and manifest['sampling']['n']==64
        assert manifest['prompts_sha256']==hashlib.sha256(prompts_paths[part].read_bytes()).hexdigest()
        assert set(old.pool(source))==set(read(prompts_paths[part]))
    assert hashlib.sha256((CK/'model.safetensors').read_bytes()).hexdigest()=='4221aab1b309a6e22100c75b421be46a87eac704e4e7ba6d8c29b8e6b75772e0'
    if a.check_only:print('ALPHA025_PREFLIGHT_PASSED');return
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    OUT.mkdir(parents=True,exist_ok=False)
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def state(stage,**extra):
        row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=OUT/'state.json.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(OUT/'state.json')
        print(json.dumps(row),flush=True)
    def run(*cmd):
        assert remaining()>900,'Final 15-minute buffer reached'
        subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT,timeout=max(1,remaining()-900))
    protocol=dict(selector='n128_alpha_025',alpha=.25,n=128,seeds=[781,1781],checkpoint=str(CK),
        screen=screen,selected_before_confirmation=True,screen_source=str(AUX),
        adaptive_reused_dev=True,confirmation_min_delta=.0005,confirmation_ci_lower_min=0,
        confirmation_control='boundary N64; also report original N64',deadline_utc=a.deadline_utc,
        minimum_start_hours=2,earlier_main_n128_failed_gate=True,
        prior_gpu_checks=checks,additional_generation='same recipe and seed as script 125',
        screen_features_sha256=hashlib.sha256((NESTED/'selected128/screen/features.jsonl').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def generate(part):
        start=time.monotonic();source=sources[part];prompts=read(prompts_paths[part]);base=old.pool(source)
        extra=OUT/'extra64'/part;dest=OUT/'selected128'/part;dest.mkdir(parents=True)
        state('generate_extra64',partition=part)
        with q.server(CK,'alpha025_'+part,block_size=64):
            run(q.PY,'-u','scripts/91_recovery_decode.py','--checkpoint',CK,'--prompts',prompts_paths[part],
                '--out',extra,'--n',64,'--seed',1781,'--defer-mbr','--allow_incomplete')
        extra_manifest=read(extra/'manifest.json');base_manifest=read(source/'manifest.json')
        assert extra_manifest['selection_pending'] and extra_manifest['seed']==1781
        assert extra_manifest['checkpoint']==str(CK)
        assert extra_manifest['prompts_sha256']==base_manifest['prompts_sha256']
        for key in ['temperature','truncation','min_p','no_repeat_ngram','frequency_penalty','frequency_window','target_words','max_new_tokens']:
            assert extra_manifest['sampling'][key]==base_manifest['sampling'][key]
        more=old.pool(extra);assert set(more)==set(prompts)
        f.OUT=OUT/'feature_server'/part;f.OUT.mkdir(parents=True)
        pred={};state('select_alpha025',partition=part)
        with f.server(),(dest/'features.jsonl').open('w') as log:
            for i,key in enumerate(prompts):
                assert remaining()>900,'Final buffer reached in feature extraction'
                left,right=base[key],more[key]
                result=f.post('/v1/large_mbr_features',dict(candidates=left['candidates']+right['candidates'],
                    finish_reasons=left['finish_reasons']+right['finish_reasons'],prompt=prompts[key]['text']))
                pred[key]={'text':result['alpha_picks']['0.25']}
                log.write(json.dumps(dict(id=key,nested=result))+'\n');log.flush()
                if (i+1)%50==0:state('select_alpha025',partition=part,done=i+1,total=len(prompts))
        (dest/'prediction.json').write_text(json.dumps(pred,ensure_ascii=False))
        manifest=dict(base_manifest);manifest['sampling']=dict(base_manifest['sampling'],n=128,
            mbr=dict(enabled=True,utility='mixed',alpha=.25))
        manifest.update(seed=None,pool_seeds=[781,1781],selection_pending=False,selector='large_mbr_features:alpha0.25',
            component_pools=[str(source),str(extra)],candidate_count=128*len(prompts),
            component_candidates_sha256=[hashlib.sha256((x/'candidates.jsonl').read_bytes()).hexdigest() for x in [source,extra]],
            seconds=time.monotonic()-start,incomplete_outputs={k:len(v['text'].split()) for k,v in pred.items() if len(v['text'].split())!=128})
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return dest
    try:
        if remaining()<7200:state('skipped_deadline',remaining_seconds=remaining());return
        dest=generate('confirmation')
        controls=[]
        for name,source in [('boundary',BOUNDARY/'confirmation'),('original',BASE/'confirmation')]:
            anchor=OUT/'controls'/name;anchor.mkdir(parents=True)
            for fn in ['prediction.json','manifest.json']:(anchor/fn).write_bytes((source/fn).read_bytes())
            controls.append(anchor)
        state('score_confirmation')
        run(q.PY,'-u','scripts/92_recovery_score.py','--truth','runs/recovery_20261005/dev/confirmation_truth.json',
            '--meta','runs/recovery_20261005/dev/confirmation_meta.json','--runs',*controls,dest)
        conf=read(dest/'scores.json');conf['partition']='confirmation'
        conf['delta_vs_original_mean']=conf['bertscore']-read(controls[1]/'scores.json')['bertscore']
        (OUT/'results.json').write_text(json.dumps([dict(partition='screen',**screen),conf],indent=2)+'\n')
        promoted=conf['delta_vs_first']>=.0005 and conf['paired_cluster_95ci'][0]>0
        submission=None
        if promoted:
            dest=generate('test');pred=read(dest/'prediction.json');test=read(prompts_paths['test'])
            assert list(pred)==list(test)==[str(i) for i in range(1,1001)]
            submission=ROOT/'submissions/B_mbr128_alpha025_20261006';submission.mkdir(exist_ok=False)
            path=submission/'prediction.json';path.write_text(json.dumps(pred,indent=2,ensure_ascii=True))
            packer=module('packer132','51_make_submission.py');check=packer.check_prediction(str(path),test,128)
            assert check['n']==check['exact_128']==1000 and check['special_token_leaks']==0
            archive=submission/'submission.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(path,arcname='prediction.json')
            with zipfile.ZipFile(archive) as z:assert z.namelist()==['prediction.json'] and z.testzip() is None and z.read('prediction.json')==path.read_bytes()
            manifest=read(dest/'manifest.json');manifest.update(validation=check,dev_confirmation=conf,
                zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (submission/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (submission/'protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
            (submission/'state.json').write_text(json.dumps(dict(stage='complete',zip=str(archive)),indent=2)+'\n')
        state('complete',promoted=promoted,submission=str(submission) if submission else None,confirmation=conf)
    except BaseException as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':main()
