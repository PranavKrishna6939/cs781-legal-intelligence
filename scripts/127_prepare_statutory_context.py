#!/usr/bin/env python
"""Prepare a dev-only statutory-context pilot from verified official HC bodies.

No model generation, external encoder, references, or test queries are used.
The index is separate from the active training corpus. A later generation pilot
needs a matched original-prompt control on exactly the selected subset.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/data/cs781-archive/recovery_20261006/fresh_hc_bodies_final')
OUT=Path('/data/cs781-archive/recovery_20261006/statutory_context')
REF=re.compile(r'\b(section|article|rule|order)\s+([0-9]{1,3}[a-z]?)(?![a-z0-9])',re.I)
NORM=re.compile(r'\b(shall|provided|means|punishable|jurisdiction|interpretation|construed|principle|power|entitled)\b',re.I)
INTRO=re.compile(r'\b(reads|provides|stipulates|enacts)\b|\bas follows\b',re.I)
JUDICIAL=re.compile(r'\b(this court|learned counsel|the appellant|the respondent|the petitioner|the submission|the contention|we (?:are|have|take|find|hold)|in our opinion)\b',re.I)
TERM=re.compile(r'[a-z]{3,}')
HEADER=re.compile(r'IN THE SUPREME COURT|APPELLATE JURISDICTION|RECORD OF PROCEEDINGS|CORAM')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def passages(text):
    words=list(re.finditer(r'\S+',text));starts=[m.start() for m in words]
    candidates=[];seen=set()
    for match in REF.finditer(text):
        kind,number=(s.lower() for s in match.groups())
        at=max(0,bisect.bisect_right(starts,match.start())-1)
        start=max(0,at-24);end=min(len(words),start+160)
        if end-start<120:start=max(0,end-160)
        if end-start<120:continue
        key=(kind,number,start//40)
        if key in seen:continue
        seen.add(key)
        value=text[words[start].start():words[end-1].end()]
        norms=len(NORM.findall(value))
        letters=[c for c in value if c.isalpha()]
        if not norms or HEADER.search(value) or not letters:continue
        if sum(c.isupper() for c in letters)/len(letters)>.25:continue
        score=3*bool(INTRO.search(value))+min(norms,8)
        candidates.append((score,start,end,kind,number,value))
    counts=collections.Counter();kept=[]
    for row in sorted(candidates,key=lambda r:(-r[0],r[1],r[3],r[4])):
        key=row[3:5]
        if counts[key]>=2:continue
        kept.append(row);counts[key]+=1
        if len(kept)==16:break
    return kept


def build_index():
    import numpy as np
    from tokenizers import Tokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    source_manifest=read(SOURCE/'manifest.json')
    fingerprint=dict(selected_sha256=digest(SOURCE/'selected.jsonl'),
                     index_sha256=digest(SOURCE/'source/index.npy'),
                     source_manifest_sha256=digest(SOURCE/'manifest.json'),
                     builder_sha256=digest(__file__))
    database=OUT/'passages.sqlite'
    if database.exists():
        assert read(OUT/'index_manifest.json')['fingerprint']==fingerprint
        return database
    temporary=OUT/'passages.building.sqlite'
    assert not temporary.exists(),'Partial index exists; inspect before rebuilding'
    rows=[json.loads(line) for line in (SOURCE/'selected.jsonl').open()]
    index=np.load(SOURCE/'source/index.npy',mmap_mode='r')
    tokens=np.memmap(SOURCE/'source/tokens.bin',dtype=np.uint16,mode='r')
    assert len(rows)==len(index)==source_manifest['selected_docs']==49096
    assert int(index[:,1].sum())==len(tokens)==source_manifest['selected_tokens']
    assert len({r['id'] for r in rows})==len({str(r['cluster']) for r in rows})==len(rows)
    tok=Tokenizer.from_file(str(ROOT/'artifacts/tokenizers/legal32k/tokenizer.json'))
    bos,eos=tok.token_to_id('<|bos|>'),tok.token_to_id('<|eos|>')
    conn=sqlite3.connect(temporary)
    conn.execute('PRAGMA journal_mode=MEMORY')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('CREATE TABLE passages(id INTEGER PRIMARY KEY, doc_id TEXT, cluster TEXT, kind TEXT, number TEXT, word_start INTEGER, word_end INTEGER, body_sha256 TEXT, passage_sha256 TEXT UNIQUE, text TEXT)')
    conn.execute('CREATE TABLE anchors(passage_id INTEGER, kind TEXT, number TEXT, PRIMARY KEY(passage_id,kind,number))')
    conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text,content='passages',content_rowid='id',tokenize='unicode61')")
    added=0
    for i,meta in enumerate(rows):
        # selected.jsonl retains offsets into the earlier candidate store. Use
        # the FINAL source index, and verify every decoded body against its hash.
        offset,length,nbytes,_=map(int,index[i]);ids=tokens[offset:offset+length]
        assert len(ids)==meta['tokens'] and nbytes==meta['bytes']
        assert int(ids[0])==bos and int(ids[-1])==eos
        text=tok.decode(ids[1:-1].tolist(),skip_special_tokens=False)
        assert len(text.encode())==nbytes and hashlib.sha256(text.encode()).hexdigest()==meta['body_sha256']
        for _,start,end,kind,number,value in passages(text):
            sha=hashlib.sha256(value.encode()).hexdigest()
            cursor=conn.execute('INSERT OR IGNORE INTO passages(doc_id,cluster,kind,number,word_start,word_end,body_sha256,passage_sha256,text) VALUES(?,?,?,?,?,?,?,?,?)',
                (meta['id'],str(meta['cluster']),kind,number,start,end,meta['body_sha256'],sha,value))
            added+=cursor.rowcount
            passage_id=cursor.lastrowid if cursor.rowcount else conn.execute('SELECT id FROM passages WHERE passage_sha256=?',(sha,)).fetchone()[0]
            conn.executemany('INSERT OR IGNORE INTO anchors VALUES(?,?,?)',
                [(passage_id,k.lower(),v.lower()) for k,v in REF.findall(value)])
        if (i+1)%1000==0:
            conn.commit();print(json.dumps(dict(stage='index',verified_docs=i+1,passages=added)),flush=True)
    conn.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    conn.execute('CREATE INDEX anchor_idx ON anchors(kind,number,passage_id)')
    conn.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(fts,'row')")
    conn.commit()
    assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert conn.execute('SELECT count(*) FROM passages').fetchone()[0]==added
    conn.close();temporary.replace(database)
    write(OUT/'index_manifest.json',dict(status='ready_dev_only',fingerprint=fingerprint,
        source=str(SOURCE),verified_docs=len(rows),passages=added,uses_references=False,
        uses_external_encoder=False,source_provenance='Already reviewed, whole-document decontaminated official HC bodies',
        limitations=['Lexical/statutory filtering is a heuristic, not a relevance label.',
                    'No generation result or approval of inference retrieval is implied.']))
    return database


def prepare(database,partition):
    from tokenizers import Tokenizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    conn=sqlite3.connect(database);conn.row_factory=sqlite3.Row
    n=conn.execute('SELECT count(*) FROM passages').fetchone()[0]
    df={r['term']:r['doc'] for r in conn.execute('SELECT term,doc FROM vocab') if r['doc']>=5}
    stop=set(ENGLISH_STOP_WORDS)|set('court courts case cases petitioner petitioners respondent respondents appellant appellants learned counsel held state india judgment order section article rule act said whether therefore thus however present view taken supreme high'.split())
    tok=Tokenizer.from_file(str(ROOT/'artifacts/tokenizers/legal32k/tokenizer.json'))
    path=ROOT/f'runs/recovery_20261005/dev/{partition}_prompts.json'
    prompts=read(path);meta=read(ROOT/f'runs/recovery_20261005/dev/{partition}_meta.json')
    selected={};control={};audit={};reasons=collections.Counter()
    for key,row in prompts.items():
        prompt=row['text'];words=prompt.split();tail=' '.join(words[-128:])
        anchors=list(dict.fromkeys((m[0].lower(),m[1].lower()) for m in REF.findall(tail)))[-3:]
        if not anchors or not JUDICIAL.search(' '.join(words[-256:])):
            reasons['no_local_statutory_reasoning']+=1;continue
        if len(tok.encode(prompt,add_special_tokens=False).ids)>1800:
            reasons['original_context_too_long']+=1;continue
        terms={t for t in TERM.findall(tail.lower()) if t not in stop and t in df}
        ranked=sorted(terms,key=lambda t:(-math.log((n+1)/(df[t]+1)),t))[:12]
        if len(ranked)<5:reasons['insufficient_content_terms']+=1;continue
        weights={t:math.log((n+1)/(df[t]+1)) for t in ranked};total=sum(weights.values())
        query=' OR '.join('"'+t+'"' for t in ranked)
        candidates={}
        for kind,number in anchors:
            sql='SELECT p.*,a.kind AS matched_kind,a.number AS matched_number,bm25(fts) AS rank FROM fts JOIN passages p ON fts.rowid=p.id JOIN anchors a ON a.passage_id=p.id WHERE fts MATCH ? AND a.kind=? AND a.number=? ORDER BY rank LIMIT 20'
            for r in conn.execute(sql,(query,kind,number)):
                if r['doc_id']==meta[key]['doc_id']:continue
                matches=set(TERM.findall(r['text'].lower()))&set(ranked)
                coverage=sum(weights[t] for t in matches)/max(total,1e-9)
                if len(matches)<3 or coverage<.35:continue
                candidates[r['id']]=dict(r,coverage=coverage,matched_terms=sorted(matches))
        if not candidates:reasons['no_supported_passage']+=1;continue
        chosen=max(candidates.values(),key=lambda r:(r['coverage'],-r['rank'],-r['id']))
        augmented=chosen['text']+' '+prompt
        if len(tok.encode(augmented,add_special_tokens=False).ids)>2047:
            reasons['augmented_context_too_long']+=1;continue
        assert augmented.endswith(prompt)
        selected[key]={'text':augmented};control[key]=row
        audit[key]=dict(chosen,original_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        query_terms=ranked,query_anchors=anchors,source_label=meta[key]['source'])
    conn.close()
    dest=OUT/partition;dest.mkdir(parents=True,exist_ok=True)
    write(dest/'augmented_prompts.json',selected);write(dest/'control_prompts.json',control)
    write(dest/'context_records.json',audit)
    sample=sorted(audit,key=lambda k:hashlib.sha256(('retrieval-review781:'+k).encode()).hexdigest())[:12]
    write(dest/'manual_sample.json',[dict(audit[k],prompt_id=k,prompt_tail=' '.join(prompts[k]['text'].split()[-128:])) for k in sample])
    summary=dict(status='prepared_not_decoded',partition=partition,total=len(prompts),selected=len(selected),
        selection_by_source=dict(collections.Counter(meta[k]['source'] for k in selected)),skipped=dict(reasons),
        prompts_sha256=digest(path),augmented_prompts_sha256=digest(dest/'augmented_prompts.json'),
        control_prompts_sha256=digest(dest/'control_prompts.json'),index_manifest_sha256=digest(OUT/'index_manifest.json'),
        min_matched_terms=3,min_weighted_coverage=.35,max_query_terms=12,
        generation_requirement='HTTP N64 augmented and original controls on this exact subset; keep incumbent predictions outside it.',
        uses_reference_continuations=False,test_queries_processed=False)
    write(dest/'manifest.json',summary);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['screen','confirmation'],default='screen')
    args=p.parse_args();prepare(build_index(),args.partition)


if __name__=='__main__':main()
