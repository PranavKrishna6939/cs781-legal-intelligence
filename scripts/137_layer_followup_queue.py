#!/usr/bin/env python
"""Wait for alpha-.25 follow-up, then test fixed MBR depths with strict promotion."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data/cs781-archive/recovery_20261006/layer_followup_queue')
ALPHA=Path('/data/cs781-archive/recovery_20261006/n128_alpha025_followup')
FEATURES=Path('/data/cs781-archive/recovery_20261006/layer_mbr_exact')
BOUNDARY=Path('/data/cs781-archive/recovery_20261006/boundary_sparse_cap40')
CK=Path('/data/cs781-archive/recovery_20261005/hybrid_b_anneal/ckpt_final_ema')
BASE=ROOT/'runs/recovery_20261005/followthrough/runtime_fixed/pool_64'
PYTHON=ROOT/'.venv/bin/python'
def read(path):return json.loads(Path(path).read_text())
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    deadline=datetime.fromisoformat(args.deadline_utc);assert deadline.utcoffset().total_seconds()==0
    assert (CK/'model.safetensors').is_file() and (BASE/'screen/candidates.jsonl').is_file()
    assert not (FEATURES/'screen/results.json').exists(),'Layer screen already completed; inspect instead of duplicating'
    if args.check_only:print('LAYER_FOLLOWUP_PREFLIGHT_PASSED');return
    OUT.mkdir(parents=True,exist_ok=False)
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def state(stage,**extra):
        row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=OUT/'state.json.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(OUT/'state.json')
        print(json.dumps(row),flush=True)
    def run(*cmd):
        assert remaining()>900,'Final 15-minute buffer reached'
        process=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,start_new_session=True)
        try:
            code=process.wait(timeout=max(1,remaining()-900))
            if code:raise subprocess.CalledProcessError(code,cmd)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    def compare(candidate,anchor):
        import numpy as np
        c=read(candidate/'per_example_scores.json');b=read(anchor/'per_example_scores.json')
        cs=read(candidate/'scores.json');bs=read(anchor/'scores.json')
        assert set(c)==set(b) and cs['bertscore_hash']==bs['bertscore_hash']
        keys=list(b);assert all(c[k]['cluster']==b[k]['cluster'] for k in keys)
        delta={k:c[k]['bertscore']-b[k]['bertscore'] for k in keys}
        clusters=sorted({b[k]['cluster'] for k in keys})
        groups=[[k for k in keys if b[k]['cluster']==cl] for cl in clusters]
        sums=np.array([sum(delta[k] for k in g) for g in groups]);sizes=np.array([len(g) for g in groups])
        draws=np.random.default_rng(781).integers(len(groups),size=(5000,len(groups)))
        boot=sums[draws].sum(1)/sizes[draws].sum(1)
        return dict(n=len(keys),bertscore=cs['bertscore'],rouge_l=cs['rouge_l'],bleu4=cs['bleu4'],
            bertscore_hash=cs['bertscore_hash'],delta_vs_first=float(np.mean(list(delta.values()))),
            paired_cluster_95ci=np.quantile(boot,[.025,.975]).tolist(),comparator=str(anchor),candidate=str(candidate))
    try:
        state('waiting_for_alpha025',deadline_utc=args.deadline_utc)
        while True:
            if remaining()<100*60:state('skipped_deadline',remaining_seconds=remaining());return
            prior=read(ALPHA/'state.json')
            if prior['stage'] in ['complete','skipped_deadline']:break
            if prior['stage']=='failed':raise RuntimeError('Alpha follow-up failed; inspect before taking GPU')
            time.sleep(55)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
        sources={'screen':BOUNDARY/'screen','confirmation':BOUNDARY/'confirmation'}
        if prior.get('promoted'):
            sources={'screen':Path('/data/cs781-archive/recovery_20261006/cached_aux_screen/n128_alpha_025'),
                     'confirmation':ALPHA/'selected128/confirmation'}
        for part,source in sources.items():
            anchor=OUT/'incumbent'/part;anchor.mkdir(parents=True)
            for fn in ['prediction.json','manifest.json','scores.json','per_example_scores.json']:
                (anchor/fn).write_bytes((source/fn).read_bytes())
        (OUT/'protocol.json').write_text(json.dumps(dict(layers=[8,12,16,20],alpha=.5,n=64,
            checkpoint=str(CK),candidate_pool=str(BASE),incumbent_sources={k:str(v) for k,v in sources.items()},
            screen_min_delta=.0005,confirmation_min_delta=.0005,confirmation_ci_lower_min=0,
            minimum_start_minutes=100,deadline_utc=args.deadline_utc,adaptive_reused_dev=True),indent=2)+'\n')
        state('screen')
        run(PYTHON,'-u','scripts/121_layer_mbr_screen.py','--partition','screen')
        rows=[]
        for depth in [8,12,16,20]:
            rows.append(dict(depth=depth,partition='screen',**compare(FEATURES/f'screen/layer_{depth}',OUT/'incumbent/screen')))
        (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        best=max(rows,key=lambda r:r['bertscore']);promoted=False;submission=None
        if best['depth']!=20 and best['delta_vs_first']>=.0005:
            chosen=best['depth'];state('confirmation',depth=chosen)
            run(PYTHON,'-u','scripts/121_layer_mbr_screen.py','--partition','confirmation','--layers',chosen,20)
            conf=dict(depth=chosen,partition='confirmation',**compare(FEATURES/f'confirmation/layer_{chosen}',OUT/'incumbent/confirmation'))
            rows.append(conf);(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            promoted=conf['delta_vs_first']>=.0005 and conf['paired_cluster_95ci'][0]>0
            if promoted:
                f=module('features137','119_jepa_witness_gate.py');f.CHECKPOINT=CK
                f.OUT=OUT/'test_features';f.OUT.mkdir()
                pool=module('nested137','125_nested_mbr128.py').pool(ROOT/'submissions/B_anneal_ema_mbr64_20261005')
                test=read(ROOT/'data/raw/Phase_1_Pretraining/Test/test.json');assert set(pool)==set(test)
                prediction={};state('select_test',depth=chosen)
                with f.server(),(f.OUT/'features.jsonl').open('w') as log:
                    for i,key in enumerate(test):
                        assert remaining()>900,'Final buffer reached during test selection'
                        row=pool[key]
                        result=f.post('/v1/layer_mbr_features',dict(candidates=row['candidates'],finish_reasons=row['finish_reasons'],layers=[chosen,20]))
                        assert result['layers']['20']['selected']==row['text'],('Native parity mismatch',key)
                        prediction[key]={'text':result['layers'][str(chosen)]['selected']}
                        log.write(json.dumps(dict(id=key,**result))+'\n');log.flush()
                        if (i+1)%50==0:state('select_test',depth=chosen,done=i+1,total=len(test))
                submission=ROOT/f'submissions/B_layer{chosen}_mbr64_20261006';submission.mkdir(exist_ok=False)
                path=submission/'prediction.json';path.write_text(json.dumps(prediction,indent=2,ensure_ascii=True))
                assert list(prediction)==[str(i) for i in range(1,1001)]
                check=module('packer137','51_make_submission.py').check_prediction(str(path),test,128)
                assert check['n']==check['exact_128']==1000 and check['special_token_leaks']==0
                archive=submission/'submission.zip'
                with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(path,arcname='prediction.json')
                with zipfile.ZipFile(archive) as z:assert z.namelist()==['prediction.json'] and z.testzip() is None and z.read('prediction.json')==path.read_bytes()
                manifest=read(ROOT/'submissions/B_anneal_ema_mbr64_20261005/manifest.json')
                manifest.update(selector='intermediate_layer_mbr',layer=chosen,validation=check,dev_confirmation=conf,
                    zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (submission/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (submission/'protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
                (submission/'state.json').write_text(json.dumps(dict(stage='complete',zip=str(archive)),indent=2)+'\n')
        state('complete',promoted=promoted,submission=str(submission) if submission else None,result_rows=rows)
    except BaseException as exc:state('failed',error=repr(exc));raise

if __name__=='__main__':main()
