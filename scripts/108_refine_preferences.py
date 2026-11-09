#!/usr/bin/env python
"""Replace easy spelling negatives with plausible, prompt-conflicting entities."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT=Path('runs/recovery_20261005/entity_recovery')
NAMES=re.compile(r"\b[A-Z][a-z]{3,}(?:[ '-][A-Z][a-z]{2,}){0,3}\b")
NUMBERS=re.compile(r'\b[1-9]\d{2,}\b')
SURNAMES=set('Singh Kumar Sharma Das Rao Patel Reddy Mehta Banerjee Mukherjee Verma Desai Gupta Swamy Pillai Nair Prasad Shukla Shukul Pandey Sinha Ghosh Joshi Iyer Khan Ali Begum Bibi Gandhi Patil Bansal Prakash Mishra Tiwari Chatterjee Bose Yadav Jha Dutta Sen'.split())
HONORIFIC=re.compile(r'\b(?:Shri|Sri|Smt\.?|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][a-z]{2,}(?:[ -][A-Z][a-z]{2,}){0,2})')
INSTITUTIONS=set(('The This That These Those There Therefore However According Whether Where When What '
    'Court Courts Government State Central Union India Indian Section Sections Article Articles Act Acts '
    'Rule Rules Code Procedure Constitution Criminal Civil Appeal Appeals Appellant Respondent Petitioner '
    'Learned Counsel Bench Judge Judges Justice Chief Magistrate Commissioner Commission Committee '
    'Tribunal Board Authority Department Officer Office Division Sessions Privy Council Project '
    'Corporation Company Bank Limited Industries Services Service Tax Revenue Income Land Labour '
    'Industrial Special Assistant Additional Deputy General Secretary District Public Police '
    'President University Financial Schedule Accordingly Provided Further Since After Before '
    'Plaintiff Defendant Judgment Order Hence Thus Their Under Other Such First Second Third').split())


def h(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')
def contains(text,s):return re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text) is not None
def bucket(kind,s):
    if kind=='name':return ('name',len(s.split()))
    return ('year',4) if len(s)==4 and 1800<=int(s)<=2026 else ('number',len(s))


def candidates(row):
    prompt,target=row['prompt'],row['target']
    # Capitalization alone admitted Part, Notice, months and occupational titles.
    # Require a personal-name cue; retain numbers as a separate category.
    names={m.group() for m in NAMES.finditer(target)
           if m.group().split()[-1] in SURNAMES and len(m.group().split())>=2}
    names.update(m.group(1) for m in HONORIFIC.finditer(prompt) if contains(target,m.group(1)))
    names={re.sub(r'^(?:Shri|Sri|Smt|Mr|Mrs|Ms|Dr)\s+','',s) for s in names}
    names={s for s in names if not set(s.split())&INSTITUTIONS}
    names={s for s in names if contains(prompt,s) and (len(s.split())>1 or
           len(re.findall(r'(?<!\w)'+re.escape(s)+r'(?!\w)',prompt))>=2)}
    numbers={m.group() for m in NUMBERS.finditer(target) if contains(prompt,m.group())}
    return names,numbers


def main():
    rows=[json.loads(l) for l in (ROOT/'entity.jsonl').read_text().splitlines() if l]
    rows=[r for r in rows if r.get('restored')]
    df=collections.Counter();vocab=collections.defaultdict(set)
    for r in rows:
        if h('pref-split:'+r['cluster'])%10==0:continue
        names,nums=candidates(r);df.update(names)
        for s in names:vocab[bucket('name',s)].add(s)
        for s in nums:vocab[bucket('number',s)].add(s)
    vocab={k:sorted(v) for k,v in vocab.items()}
    result={'train':[],'holdout':[]};audit=collections.Counter()
    for r in rows:
        names,nums=candidates(r)
        options=[('name',s) for s in sorted(names,key=lambda s:(df[s],-len(s),s)) if df[s]<=20]
        options += [('number',s) for s in sorted(nums,key=lambda s:(-len(s),s))]
        for kind,s in options:
            key=bucket(kind,s);donors=vocab.get(key,[])
            if not donors:continue
            offset=h('donor:'+r['id']+s)%len(donors)
            replacement=next((donors[(offset+i)%len(donors)] for i in range(len(donors))
                              if not contains(r['prompt']+' '+r['target'],donors[(offset+i)%len(donors)])),None)
            if replacement is None:continue
            rejected=re.sub(r'(?<!\w)'+re.escape(s)+r'(?!\w)',lambda _:replacement,r['target'])
            assert rejected!=r['target'] and len(rejected.split())==128
            rec=dict(id=r['id'],cluster=r['cluster'],prompt=r['prompt'],chosen=r['target'],rejected=rejected,
                     entity=s,replacement=replacement,kind=kind,source_doc_sha256=r['source_doc_sha256'],
                     label_source='official gold versus plausible entity from a different official training excerpt')
            split='holdout' if h('pref-split:'+r['cluster'])%10==0 else 'train'
            result[split].append(rec);audit[kind]+=1;break
    assert len(result['train'])>=500
    assert not {r['cluster'] for r in result['train']}&{r['cluster'] for r in result['holdout']}
    for split,records in result.items():
        old=ROOT/f'preferences_{split}.jsonl';backup=ROOT/f'preferences_spelling_{split}.jsonl'
        if backup.exists():backup=ROOT/f'preferences_before_{hashlib.sha256(old.read_bytes()).hexdigest()[:8]}_{split}.jsonl'
        assert not backup.exists(),'Refinement already applied'
        old.rename(backup)
        old.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest=dict(counts={k:len(v) for k,v in result.items()},kinds=dict(audit),
                  note='Personal names require an honorific context or a common surname cue; capitalization alone is insufficient. Common institutions/title words excluded. Single names require >=2 prompt occurrences. Numeric mentions are separate. Donor vocabulary fits preference-training clusters only. Pairs remain programmatic, not human-verified factuality labels.')
    (ROOT/'preferences_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    for r in result['train'][:25]:print(r['entity'],'->',r['replacement'])


if __name__=='__main__':main()
