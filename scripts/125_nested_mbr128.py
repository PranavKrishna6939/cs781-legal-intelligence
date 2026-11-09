#!/usr/bin/env python
"""Extend incumbent N64 pools with an independent N64 draw, via Saul HTTP only.

Run after both data queues release the GPU. Preserve the old pool and score a
matched tiled N64 control. Only a confirmed gain can create a test submission.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def read(path):return json.loads(Path(path).read_text())


def pool(path):
    rows=[json.loads(line) for line in (Path(path)/'candidates.jsonl').open()]
    assert len({r['id'] for r in rows})==len(rows)
    assert all(len(r['candidates'])==len(r['finish_reasons'])==64 for r in rows)
    return {r['id']:r for r in rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--base',required=True,type=Path,help='Incumbent root containing screen/confirmation')
    p.add_argument('--test-base',required=True,type=Path,help='Incumbent test submission with N64 candidates')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--deadline-utc',required=True)
    p.add_argument('--submission-name',default='B_final_nested_mbr128_20261006')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    args.checkpoint=args.checkpoint.resolve();args.base=args.base.resolve();args.test_base=args.test_base.resolve()
    args.out=args.out.resolve();deadline=datetime.fromisoformat(args.deadline_utc)
    assert deadline.utcoffset().total_seconds()==0
    q=module('queue99_nested','99_recovery_training_queue.py')
    f=module('features119_nested','119_jepa_witness_gate.py')
    f.CHECKPOINT=args.checkpoint
    sources={'screen':args.base/'screen','confirmation':args.base/'confirmation','test':args.test_base}
    prompt_paths={part:ROOT/f'runs/recovery_20261005/dev/{part}_prompts.json' for part in ['screen','confirmation']}
    prompt_paths['test']=ROOT/'data/raw/Phase_1_Pretraining/Test/test.json'
    for part,source in sources.items():
        manifest=read(source/'manifest.json')
        assert manifest['checkpoint']==str(args.checkpoint)
        assert manifest['sampling']['n']==64 and manifest['seed']==781
        assert manifest['sampling']['mbr']=={'enabled':True,'utility':'mixed','alpha':.5}
        assert not manifest.get('selection_pending',False)
        assert manifest['prompts_sha256']==hashlib.sha256(prompt_paths[part].read_bytes()).hexdigest()
        assert set(pool(source))==set(read(prompt_paths[part]))
    if args.check_only:
        print('NESTED_PREFLIGHT_PASSED: three incumbent pools and checkpoint identity validated');return
    for arm in ['repaired_anneal','fresh_hc_anneal']:
        assert read(ROOT/f'runs/recovery_20261006/{arm}/state.json')['stage'] in ['complete','skipped_deadline']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'complete.json').exists(),'Completed experiment exists'

    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def state(stage,**extra):
        row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)
        (args.out/'state.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    def call(*argv):
        assert remaining()>600,'Final upload margin reached'
        subprocess.run(list(map(str,argv)),cwd=ROOT,check=True,timeout=max(60,remaining()-600))
    if remaining()<4*3600:
        state('skipped_deadline',remaining_hours=remaining()/3600);return
    protocol=dict(checkpoint=str(args.checkpoint),base=str(args.base),test_base=str(args.test_base),
        base_seed=781,additional_seed=1781,candidates_per_pool=64,total_candidates=128,
        utility='mixed',alpha=.5,encoding_candidates_per_forward=64,pair_block=8,
        screen_min_delta=.0005,confirmation_min_delta=.0005,confirmation_ci_lower_min=0,
        comparison='Must beat both actual incumbent and matched tiled N64 mean; CI is versus actual incumbent',
        auxiliary_alpha_diagnostics=[.25,.75],
        boundary_diagnostic=dict(policy='closure_then_new_case_header_v1',used_for_main_promotion=False,
            source_sha256=hashlib.sha256((ROOT/'saul/serve/boundary.py').read_bytes()).hexdigest()),
        allow_incomplete_provisional_extra_picks=True,
        deadline_utc=args.deadline_utc)
    if (args.out/'protocol.json').exists():assert read(args.out/'protocol.json')==protocol
    else:(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

    def generate_and_select(part,include_control):
        started=time.monotonic();source=sources[part];prompts=read(prompt_paths[part]);base_rows=pool(source)
        extra=args.out/'extra64'/part
        if not (extra/'prediction.json').exists():
            state('generate_extra64',partition=part,seed=1781)
            with q.server(args.checkpoint,'nested_extra_'+part,block_size=64):
                # The extra-only choice is provisional. The original pool may
                # supply valid candidates even if all extra draws finish short.
                # Only the final union selection is checked/packaged for test.
                flags=['--allow_incomplete']
                call(q.PY,'-u','scripts/91_recovery_decode.py','--checkpoint',args.checkpoint,
                     '--prompts',prompt_paths[part],'--out',extra,'--n',64,'--seed',1781,'--defer-mbr',*flags)
        extra_manifest=read(extra/'manifest.json')
        assert extra_manifest['selection_pending'] and extra_manifest['seed']==1781
        assert extra_manifest['checkpoint']==str(args.checkpoint)
        assert extra_manifest['prompts_sha256']==hashlib.sha256(prompt_paths[part].read_bytes()).hexdigest()
        extra_rows=pool(extra);assert set(extra_rows)==set(prompts)
        baseline_manifest=read(source/'manifest.json')
        for name in ['temperature','truncation','min_p','no_repeat_ngram','frequency_penalty',
                     'frequency_window','target_words','max_new_tokens']:
            assert extra_manifest['sampling'][name]==baseline_manifest['sampling'][name]
        dest=args.out/'selected128'/part;dest.mkdir(parents=True,exist_ok=True)
        control=args.out/'tiled64'/part;control.mkdir(parents=True,exist_ok=True)
        cache=dest/'features.jsonl'
        features={r['id']:r for r in map(json.loads,cache.open())} if cache.exists() else {}
        f.OUT=args.out/'feature_server'/part;f.OUT.mkdir(parents=True,exist_ok=True)
        state('select_nested128',partition=part)
        with f.server(),cache.open('a') as log:
            for i,key in enumerate(prompts):
                if key in features:continue
                if remaining()<600:raise TimeoutError('Final upload margin reached during selection')
                old,new=base_rows[key],extra_rows[key]
                merged=dict(candidates=old['candidates']+new['candidates'],finish_reasons=old['finish_reasons']+new['finish_reasons'],
                            prompt=prompts[key]['text'])
                result=f.post('/v1/large_mbr_features',merged)
                record=dict(id=key,nested=result)
                if include_control:
                    record['control']=f.post('/v1/large_mbr_features',dict(candidates=old['candidates'],finish_reasons=old['finish_reasons'],
                                                                 prompt=prompts[key]['text']))
                    record['control_matches_incumbent']=record['control']['selected']==old['text']
                features[key]=record;log.write(json.dumps(record)+'\n');log.flush()
                if (i+1)%50==0:state('select_nested128',partition=part,done=i+1,total=len(prompts))
        prediction={key:{'text':features[key]['nested']['selected']} for key in prompts}
        (dest/'prediction.json').write_text(json.dumps(prediction,ensure_ascii=False))
        manifest=dict(baseline_manifest)
        manifest['sampling']=dict(baseline_manifest['sampling'],n=128)
        manifest.update(seed=None,pool_seeds=[781,1781],selection_pending=False,
            component_pools=[str(source),str(extra)],candidate_count=128*len(prompts),
            selector='large_mbr_features:64-candidate encoding,8-candidate pair tiles',
            seconds=time.monotonic()-started,seconds_include_n64_control=include_control,
            incomplete_outputs={k:len(r['text'].split()) for k,r in prediction.items() if len(r['text'].split())!=128})
        manifest['component_candidates_sha256']=[hashlib.sha256((path/'candidates.jsonl').read_bytes()).hexdigest() for path in [source,extra]]
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if include_control:
            pred64={key:{'text':features[key]['control']['selected']} for key in prompts}
            (control/'prediction.json').write_text(json.dumps(pred64,ensure_ascii=False))
            (control/'manifest.json').write_text(json.dumps(dict(baseline_manifest,selection_pending=False,
                selector='large_mbr_features:N64 numerical control'),indent=2)+'\n')
        return dest,control,manifest['seconds']

    def evaluate(part):
        dest,control,elapsed=generate_and_select(part,True)
        anchor=args.out/'incumbent'/part;anchor.mkdir(parents=True,exist_ok=True)
        for filename in ['prediction.json','manifest.json']:
            (anchor/filename).write_bytes((sources[part]/filename).read_bytes())
        state('score',partition=part)
        call(q.PY,'scripts/92_recovery_score.py','--truth',ROOT/f'runs/recovery_20261005/dev/{part}_truth.json',
             '--meta',ROOT/f'runs/recovery_20261005/dev/{part}_meta.json','--runs',anchor,control,dest)
        result=read(dest/'scores.json');result['delta_vs_tiled64']=result['bertscore']-read(control/'scores.json')['bertscore']
        result['partition']=part;result['generation_selection_seconds']=elapsed
        result['control_selection_mismatches']=sum(not json.loads(line)['control_matches_incumbent'] for line in (dest/'features.jsonl').open())
        (dest/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result

    try:
        # Exercise the real checkpoint before spending time generating extra pools.
        f.OUT=args.out/'gpu_smoke';f.OUT.mkdir(parents=True,exist_ok=True)
        screen_pool=pool(sources['screen'])
        keys=sorted(screen_pool,key=lambda k:hashlib.sha256(('mbr-gpu-smoke:'+k).encode()).hexdigest())[:5]
        keys=list(dict.fromkeys(keys+[max(screen_pool,key=lambda k:sum(map(len,screen_pool[k]['candidates'])))]))
        checks=[];state('gpu_parity_check')
        with f.server():
            for key in keys:
                record=screen_pool[key]
                body=dict(candidates=record['candidates'],finish_reasons=record['finish_reasons'])
                native=f.post('/v1/mbr_features',body);tiled=f.post('/v1/large_mbr_features',body)
                assert native['selected']==record['text'],f'Native checkpoint parity failed: {key}'
                assert len(native['scores'])==len(tiled['scores'])
                difference=max(abs(x-y) for x,y in zip(native['scores'],tiled['scores']))
                assert difference<=1e-5,(key,difference)
                checks.append(dict(id=key,max_score_difference=difference,
                    tiled_selection_matches_native=tiled['selected']==native['selected']))
        (f.OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        screen_start=time.monotonic();rows=[evaluate('screen')]
        (args.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        promoted=False;submission=None
        if min(rows[0]['delta_vs_first'],rows[0]['delta_vs_tiled64'])>=.0005:
            estimate=(time.monotonic()-screen_start)*(2000/rows[0]['n'])+15*60
            if remaining()<estimate+3600:
                state('screen_passed_deadline_stop',estimated_remaining_seconds=estimate,remaining_seconds=remaining());return
            rows.append(evaluate('confirmation'))
            (args.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            conf=rows[-1]
            promoted=min(conf['delta_vs_first'],conf['delta_vs_tiled64'])>=.0005 and conf['paired_cluster_95ci'][0]>0
            if promoted:
                dest,_,_=generate_and_select('test',False)
                assert Path(args.submission_name).name==args.submission_name
                submission=ROOT/'submissions'/args.submission_name;submission.mkdir(parents=True,exist_ok=False)
                pred=read(dest/'prediction.json');test=read(prompt_paths['test'])
                assert list(pred)==list(test)==[str(i) for i in range(1,1001)]
                pred_path=submission/'prediction.json';pred_path.write_text(json.dumps(pred,indent=2,ensure_ascii=True))
                packer=module('packer51_nested','51_make_submission.py')
                validation=packer.check_prediction(str(pred_path),test,128)
                assert validation['n']==validation['exact_128']==1000 and validation['special_token_leaks']==0
                zip_path=submission/'submission.zip'
                with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:archive.write(pred_path,arcname='prediction.json')
                with zipfile.ZipFile(zip_path) as archive:
                    assert archive.namelist()==['prediction.json'] and archive.testzip() is None
                    assert archive.read('prediction.json')==pred_path.read_bytes()
                manifest=read(dest/'manifest.json');manifest.update(validation=validation,
                    prediction_sha256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                    dev_confirmation=conf)
                (submission/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (submission/'state.json').write_text(json.dumps(dict(stage='complete',zip=str(zip_path)),indent=2)+'\n')
        complete=dict(promoted=promoted,submission=str(submission) if submission else None,results=rows)
        (args.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
        state('complete',**complete)
    except BaseException as exc:
        state('failed',error=repr(exc));raise


if __name__=='__main__':main()
