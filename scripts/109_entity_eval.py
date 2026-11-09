#!/usr/bin/env python
"""Descriptive entity retention on dev; never used to select training examples."""
import argparse
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('entities',Path(__file__).with_name('108_refine_preferences.py'))
entities=importlib.util.module_from_spec(spec);spec.loader.exec_module(entities)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/recovery_20261005');a=p.parse_args()
    root=Path(a.root);prompt=json.loads((root/'dev/screen_prompts.json').read_text());truth=json.loads((root/'dev/screen_truth.json').read_text())
    supports={k:entities.candidates(dict(prompt=prompt[k]['text'],target=truth[k]['text'])) for k in truth}
    runs=['entity_models/baseline_slots128','dense_d1/baseline','entity_models/random_final','entity_models/entity_final','entity_models/entity_half']
    report={}
    for name in runs:
        path=root/name
        if not (path/'per_example_scores.json').exists():continue
        pred=json.loads((path/'prediction.json').read_text());scores=json.loads((path/'per_example_scores.json').read_text())
        rec={}
        for index,kind in enumerate(['personal_name_proxy','number_at_least_3_digits']):
            ids=[k for k in supports if supports[k][index]]
            total=sum(len(supports[k][index]) for k in ids)
            hits=sum(sum(entities.contains(pred[k]['text'],s) for s in supports[k][index]) for k in ids)
            rec[kind]=dict(examples=len(ids),gold_prompt_supported_entities=total,exact_retained=hits,
                micro_recall=hits/total if total else None,
                bertscore=sum(scores[k]['bertscore'] for k in ids)/len(ids) if ids else None)
        report[name]=rec
    out=root/'entity_recovery/entity_diagnostics.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(results=report,
        caveat='Small heuristic subsets; descriptive exact retention, not annotated NER or proof of factuality. Case-sensitive. Names require surname/honorific cues; numbers >=3 digits. Not used as training rewards.'),indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
