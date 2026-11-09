#!/usr/bin/env python
"""Post-failure, evaluation-only geometry diagnosis on the SAME 64 examples.

No fitting, generation, or additional feature extraction. These are adaptive
diagnostics, never independent confirmation. Squared distance retains the
candidate self-norm term omitted by a pure cross-feature dot product.
"""
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/cs781-archive/recovery_20261006/j09_witness')


def main():
    features=json.loads((BASE/'gate_features.json').read_text())
    output=BASE/'geometry_diagnostic';output.mkdir(exist_ok=True)
    names=['support_squared_distance','support_cosine','mass_hellinger']
    (output/'protocol.json').write_text(json.dumps(dict(
        status='adaptive_diagnostic_after_failed_witness_gate', n=64,
        representations=names, no_training=True, no_submission=True),indent=2))
    predictions={n:{} for n in names}; stats=[]
    for row in features:
        support=np.asarray(row['support']); mass=np.asarray(row['mass'])
        reference=row['reference']; rs=np.asarray(reference['support']); rb=np.asarray(reference['mass'])
        utilities={
            'support_squared_distance': -((support-rs)**2).mean(1),
            'support_cosine': support@rs / np.maximum(np.linalg.norm(support,axis=1)*np.linalg.norm(rs),1e-12),
            'mass_hellinger': np.sqrt(mass)@np.sqrt(rb),
        }
        selected={'native':int(np.argmax(row['native_scores'])), 'witness':int(np.argmax(row['oracle_scores']))}
        for name,score in utilities.items():
            selected[name]=int(score.argmax())
            predictions[name][row['id']]={'text':row['candidates'][selected[name]]}
        counts={name:sum(c.isdigit() for c in row['candidates'][i])/max(1,len(row['candidates'][i])) for name,i in selected.items()}
        stats.append(dict(id=row['id'],digit_fraction=counts,selected_indices=selected))
    (output/'selection_diagnostics.json').write_text(json.dumps(stats,indent=2))
    # Copy the incumbent so the common scorer cannot overwrite the original gate results.
    base=output/'native_mbr';base.mkdir(exist_ok=True)
    (base/'prediction.json').write_bytes((BASE/'native_mbr/prediction.json').read_bytes())
    paths=[base]
    for name,pred in predictions.items():
        path=output/name;path.mkdir(exist_ok=True)
        (path/'prediction.json').write_text(json.dumps(pred));paths.append(path)
    subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/92_recovery_score.py','--truth',str(BASE/'truth.json'),
        '--meta',str(BASE/'meta.json'),'--runs',*map(str,paths)],cwd=ROOT,check=True,timeout=300)
    result={p.name:json.loads((p/'scores.json').read_text()) for p in paths}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
