#!/usr/bin/env python
"""After the B anneal queue, run bounded alignment controls and N32/N64 screens.

All generation uses managed instances of the existing HTTP inference server.
Stops on failure, records every tested recipe, never uploads a submission.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];PY=str(ROOT/'.venv/bin/python')
RUN=ROOT/'runs/recovery_20261005';OUT=RUN/'followthrough'
ARCHIVE=Path('/data/cs781-archive/recovery_20261005')
spec=importlib.util.spec_from_file_location('queue',ROOT/'scripts/99_recovery_training_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def read(p):return json.loads(Path(p).read_text())


def state(stage,**kw):
    OUT.mkdir(exist_ok=True)
    (OUT/'state.json').write_text(json.dumps(dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2))
    print(stage,kw,flush=True)


def call(*args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def evaluate(ck,name,anchor,partition='screen',n=16,ngram=4):
    dest=OUT/name/partition
    state('evaluate',checkpoint=str(ck),name=name,partition=partition,n=n)
    with queue.server(ck,'follow_'+name,block_size=max(16,n)):
        call(PY,'-u','scripts/91_recovery_decode.py','--checkpoint',ck,'--prompts',
             RUN/f'dev/{partition}_prompts.json','--out',dest,'--n',n,'--ngram',ngram,'--allow_incomplete')
    call(PY,'-u','scripts/92_recovery_score.py','--truth',RUN/f'dev/{partition}_truth.json',
         '--meta',RUN/f'dev/{partition}_meta.json','--runs',anchor,dest)
    return dict(checkpoint=str(ck),name=name,candidate_count=n,ngram=ngram,path=str(dest),**read(dest/'scores.json'))


def decoding_tail(ck,anchor,confirm_anchor,tested):
    for n in [32,64]:
        name=f'pool_{n}'
        if not any(r['name']==name and r['path'].endswith('/screen') for r in tested):
            tested.append(evaluate(ck,name,anchor,n=n))
            (OUT/'results.json').write_text(json.dumps(tested,indent=2))
    screens=[r for r in tested if r['path'].endswith('/screen')]
    best=max(screens,key=lambda r:r['bertscore'])
    if best['delta_vs_first']>=.0005:
        if not any(r['name']==best['name'] and r['path'].endswith('/confirmation') for r in tested):
            confirmed=evaluate(best['checkpoint'],best['name'],confirm_anchor,'confirmation',best['candidate_count'],best['ngram'])
            tested.append(confirmed)
            (OUT/'results.json').write_text(json.dumps(tested,indent=2))
    state('complete',results=str(OUT/'results.json'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume_decoding',action='store_true',help='Reuse saved base and preference results; run only unfinished pool screens and confirmation')
    args=parser.parse_args()
    if args.resume_decoding:
        os.chdir(ROOT)
        assert read(RUN/'queue_state.json')['stage']=='complete'
        base=read(OUT/'base.json');tested=read(OUT/'results.json')
        assert {'preference_ce','preference_dpo'} <= {r['name'] for r in tested}
        state('resume_decoding',checkpoint=base['checkpoint'])
        decoding_tail(base['checkpoint'],base['anchor'],base['confirmation_anchor'],tested)
        return
    os.chdir(ROOT);state('waiting_for_b_queue')
    deadline=time.monotonic()+8*3600
    while True:
        current=read(RUN/'queue_state.json')
        if current['stage']=='complete':break
        if current['stage']=='failed':raise RuntimeError('Original B queue failed: '+str(current))
        if time.monotonic()>deadline:raise TimeoutError('B queue did not finish within eight hours')
        time.sleep(30)
    dense=ROOT/'runs/dense_286m/anneal/ckpt_final_ema'
    anchor=RUN/'dense_d1/baseline';confirm_anchor=RUN/'dense_confirmation/baseline'
    ck=dense;score=read(confirm_anchor/'scores.json')['bertscore']
    original=read(RUN/'training_summary.json')
    b=original['anneal_confirmation']
    if b['bertscore']>score:
        winner=max(original['anneal'],key=lambda r:r['bertscore'])
        ck=Path(winner['checkpoint']);anchor=RUN/'models'/winner['name']/'screen'
        confirm_anchor=RUN/'models'/winner['name']/'confirmation';score=b['bertscore']
    tested=[]
    # Promote restored-data SFT only after a useful screen and confirmation.
    pilot=RUN/'entity_recovery/pilot_summary.json'
    if pilot.exists():
        win=max(read(pilot),key=lambda r:r['bertscore'])
        if win['delta_vs_first']>=.0005:
            row=evaluate(win['checkpoint'],'restored_confirmation',RUN/'dense_confirmation/baseline','confirmation')
            tested.append(row)
            if row['bertscore']>score:
                ck=Path(win['checkpoint']);anchor=RUN/'entity_models'/win['name']
                confirm_anchor=Path(row['path']);score=row['bertscore']
    state('selected_base',checkpoint=str(ck),confirmation_bertscore=score)
    (OUT/'base.json').write_text(json.dumps(dict(checkpoint=str(ck),anchor=str(anchor),confirmation_anchor=str(confirm_anchor),score=score),indent=2))
    for mode in ['ce','dpo']:
        dest=ARCHIVE/f'entity_preference_{mode}'
        state('preference_train',mode=mode,checkpoint=str(ck))
        call(PY,'-u','scripts/106_entity_preference.py','--checkpoint',ck,
             '--data',RUN/'entity_recovery/preferences_train.jsonl','--out',dest,
             '--mode',mode,'--updates','64','--compile')
        # Final checkpoints are the predeclared primary comparison; retain half
        # checkpoints for diagnosis without evaluating a broad checkpoint grid.
        final=max(dest.glob('ckpt_*'),key=lambda p:int(p.name.split('_')[-1]))
        tested.append(evaluate(final,'preference_'+mode,anchor))
        (OUT/'results.json').write_text(json.dumps(tested,indent=2))
    decoding_tail(ck,anchor,confirm_anchor,tested)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state('failed',error=repr(exc));raise
