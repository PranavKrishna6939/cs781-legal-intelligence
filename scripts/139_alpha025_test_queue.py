#!/usr/bin/env python
"""Package alpha .25 as an experimental candidate after the layer GPU job.

Positive paired confirmation versus the scored incumbent; the stricter boundary
CI gate failed. Preserve that distinction in the artifact and do not upload.
"""
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
OUT=Path('/data/cs781-archive/recovery_20261006/alpha025_test_candidate')
CONF=Path('/data/cs781-archive/recovery_20261006/n128_alpha025_followup')
LAYER=Path('/data/cs781-archive/recovery_20261006/layer_followup_queue')
CK=Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
def read(p):return json.loads(Path(p).read_text())
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline-utc',required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();os.chdir(ROOT);deadline=datetime.fromisoformat(a.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    conf_state=read(CONF/'state.json');assert conf_state['stage']=='complete' and not conf_state['promoted']
    conf=conf_state['confirmation'];comparison=read(CONF/'confirmation_vs_scored_incumbent.json')
    assert comparison['delta']>=.0005 and comparison['paired_cluster_95ci'][0]>0
    assert conf['delta_vs_first']>0 and conf['paired_cluster_95ci'][0]<=0
    sources={'test':ROOT/'submissions/B_anneal_ema_mbr64_20261005'}
    prompts_paths={'test':ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'}
    old=module('nested139','125_nested_mbr128.py');q=module('queue139','99_recovery_training_queue.py')
    f=module('features139','119_jepa_witness_gate.py');f.CHECKPOINT=CK
    manifest=read(sources['test']/'manifest.json')
    assert manifest['checkpoint']==str(CK) and manifest['seed']==781 and manifest['sampling']['n']==64
    assert manifest['prompts_sha256']==hashlib.sha256(prompts_paths['test'].read_bytes()).hexdigest()
    assert set(old.pool(sources['test']))==set(read(prompts_paths['test']))
    assert not (ROOT/'submissions/B_mbr128_alpha025_20261006').exists()
    if a.check_only:print('ALPHA025_EXPERIMENTAL_TEST_PREFLIGHT_PASSED');return
    OUT.mkdir(parents=True,exist_ok=False)
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def state(stage,**extra):
        row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=OUT/'state.json.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(OUT/'state.json')
        print(json.dumps(row),flush=True)
    def run(*cmd):
        assert remaining()>900,'Final 15-minute buffer reached'
        subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT,timeout=max(1,remaining()-900))
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
        protocol=dict(experimental_candidate=True,strict_boundary_gate_passed=False,scored_incumbent_gate_passed=True,
            scored_incumbent_comparison=comparison,boundary_comparison=conf,alpha=.25,n=128,seeds=[781,1781],
            adaptive_reused_dev=True,reason='Positive paired confirmation against actual scored submission 963010; boundary CI includes zero',
            minimum_start_minutes=75,deadline_utc=a.deadline_utc,automatic_upload=False)
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        state('waiting_for_layer',deadline_utc=a.deadline_utc)
        while True:
            if remaining()<75*60:state('skipped_deadline',remaining_seconds=remaining());return
            previous=read(LAYER/'state.json')
            if previous['stage'] in ['complete','skipped_deadline','failed']:break
            time.sleep(55)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
        assert hashlib.sha256((CK/'model.safetensors').read_bytes()).hexdigest()=='4221aab1b309a6e22100c75b421be46a87eac704e4e7ba6d8c29b8e6b75772e0'
        (OUT/'predecessor_state.json').write_text(json.dumps(previous,indent=2)+'\n')
        dest=generate('test');pred=read(dest/'prediction.json');test=read(prompts_paths['test'])
        assert list(pred)==list(test)==[str(i) for i in range(1,1001)]
        submission=ROOT/'submissions/B_mbr128_alpha025_20261006';submission.mkdir(exist_ok=False)
        path=submission/'prediction.json';path.write_text(json.dumps(pred,indent=2,ensure_ascii=True))
        check=module('packer139','51_make_submission.py').check_prediction(str(path),test,128)
        assert check['n']==check['exact_128']==1000 and check['special_token_leaks']==0
        archive=submission/'submission.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(path,arcname='prediction.json')
        with zipfile.ZipFile(archive) as z:assert z.namelist()==['prediction.json'] and z.testzip() is None and z.read('prediction.json')==path.read_bytes()
        manifest=read(dest/'manifest.json');manifest.update(validation=check,dev_confirmation=conf,
            experimental_candidate=True,strict_boundary_gate_passed=False,scored_incumbent_gate_passed=True,
            scored_incumbent_comparison=comparison,zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (submission/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (submission/'protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
        (submission/'state.json').write_text(json.dumps(dict(stage='complete',zip=str(archive)),indent=2)+'\n')
        state('complete',candidate_ready=True,experimental_candidate=True,submission=str(submission),
            result_rows=[conf],scored_incumbent_comparison=comparison,strict_boundary_gate_passed=False)
    except BaseException as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':main()
