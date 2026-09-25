#!/usr/bin/env python3
"""Offline validation of episode coverage, pairing, manifests and source syntax."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from reproduce import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);p.add_argument('--deny-pattern',action='append',default=[]);a=p.parse_args()
    catalog=json.loads((ROOT/'data/catalog.json').read_text())['runs'];assert len(catalog)==61
    runs={r['id']:r for r in catalog};assert len(runs)==len(catalog)
    expected=dict(camera=1599,robot=1550,language=1537,light=1142,background=1076,noise=1601,layout=1525)
    keys={};groups=Counter();episodes=0;family_counts={}
    for ident,run in runs.items():
        rows=load(run);groups[run['benchmark']]+=1;episodes+=len(rows)
        keys[ident]=set(rows)
        family_counts[ident]=Counter((r['family'],r['dimension']) for r in rows.values())
        assert len({r['family'] for r in rows.values()})==40
        assert all(type(r['success']) is bool for r in rows.values())
        if run['benchmark']=='libero_plus':assert Counter(r['dimension'] for r in rows.values())==expected
        elif run['benchmark']=='libero_plus_screen560':assert set(family_counts[ident].values())=={2}
        else:assert Counter(r['suite'] for r in rows.values())=={s:500 for s in ['libero_spatial','libero_object','libero_goal','libero_10']}
    for run in catalog:
        m=run['model'];benchmark=run['benchmark']
        base=m+{'libero_plus':'-plus-stock','libero_plus_screen560':'-screen-step0','libero_original':'-original-stock'}[benchmark]
        assert keys[run['id']]==keys[base]
    assert groups==dict(libero_plus=32,libero_plus_screen560=14,libero_original=15)
    assert episodes==358800
    ids={}
    for split in ['camera_only','noise_only']:
        obj=json.loads((ROOT/f'data/manifests/{split}_training_ids.json').read_text());ids[split]=set(obj['episode_ids'])
        assert hashlib.sha256('\n'.join(map(str,obj['episode_ids'])).encode()).hexdigest()==obj['episode_ids_sha256']
    assert not ids['camera_only'] & ids['noise_only']
    assert len(ids['camera_only'])==2871 and len(ids['noise_only'])==2852
    syntax=0
    for file in ROOT.rglob('*.py'):
        if any(x in file.parts for x in ['.venv','third_party','models','outputs']):continue
        ast.parse(file.read_text());syntax+=1
    # Inspect names and textual contents. External project license attribution is preserved.
    patterns=[r'/Users/[^/\s]+',r'/home/(?!user\b)[^/\s]+',r'/ssd\d+/',r'(?i)BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',r'(?i)(?:api_key|access_token)\s*[=:]\s*["\'](?!["\'])[^"\']{12,}']+a.deny_pattern
    failures=[]
    for file in ROOT.rglob('*'):
        if file.is_symlink():failures.append(str(file.relative_to(ROOT))+':symlink');continue
        if not file.is_file() or any(x in file.parts for x in ['.venv','third_party','models','outputs','__pycache__']):continue
        if file.suffix in ['.mp4','.pdf']:continue
        raw=gzip.decompress(file.read_bytes()) if file.suffix=='.gz' else file.read_bytes()
        text=raw.decode('utf-8',errors='replace')
        for i,pattern in enumerate(patterns):
            # The validator itself documents scanner rules, not private source values.
            if file.resolve()==Path(__file__).resolve() and i<len(patterns)-len(a.deny_pattern):continue
            if re.search(pattern,text):failures.append(str(file.relative_to(ROOT))+f':rule-{i}')
    assert not failures,failures
    report=dict(status='pass',quantitative_runs=len(catalog),episodes=episodes,groups=dict(groups),
                python_files_syntax_checked=syntax,pairing='exact episode-key equality',training_splits='disjoint and ordered-ID hash verified',
                privacy_scan='text and decompressed outcome fields; media inspected separately',gpu_end_to_end_rerun=False)
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
