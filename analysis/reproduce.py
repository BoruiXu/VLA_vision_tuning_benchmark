#!/usr/bin/env python3
"""Recompute paper statistics from the released episode-level outcomes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from bootstrap_core import DIMENSIONS, estimate

ROOT=Path(__file__).resolve().parents[1]
MODELS=['oft','pi05','internvla15','nora15','xvla']

def load(run):
    path=ROOT/run['episodes_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==run['episodes_sha256'], run['id']
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:rows=[json.loads(line) for line in f]
    keys=[(r['suite'],r['task_id'],r['episode_idx']) for r in rows]
    assert len(keys)==len(set(keys))==run['expected_episodes']
    assert sum(r['success'] for r in rows)==run['successes']
    return dict(zip(keys,rows))

def rates(rows):
    dims=sorted({r['dimension'] for r in rows.values()})
    result={}
    for d in dims:
        group=[r for r in rows.values() if r['dimension']==d]
        result[d]=100*sum(r['success'] for r in group)/len(group)
    result['overall']=100*sum(r['success'] for r in rows.values())/len(rows)
    return result

def paired(base,tuned,weights,target):
    assert base.keys()==tuned.keys(),'Mismatched paired episode keys'
    families=sorted({r['family'] for r in base.values()});assert len(families)==40
    b=np.zeros((40,7));t=b.copy();n=b.copy()
    for key,r in base.items():
        i=families.index(r['family']);j=DIMENSIONS.index(r['dimension'])
        assert r['dimension']==tuned[key]['dimension'] and r['family']==tuned[key]['family']
        b[i,j]+=r['success'];t[i,j]+=tuned[key]['success'];n[i,j]+=1
    assert np.all(n>0)
    return estimate(b,t,n,weights,target)

def main():
    p=argparse.ArgumentParser();p.add_argument('--draws',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--output',type=Path,default=ROOT/'results/recomputed')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs={r['id']:r for r in json.loads((ROOT/'data/catalog.json').read_text())['runs']}
    data={k:load(v) for k,v in runs.items()}
    stats={k:rates(v) for k,v in data.items()}
    suite_rates={k:{suite:rates({key:r for key,r in rows.items() if r['suite']==suite})
                    for suite in sorted({r['suite'] for r in rows.values()})} for k,rows in data.items()}
    sampled=np.random.default_rng(a.seed).integers(0,40,size=(a.draws,40))
    weights=np.stack([np.bincount(x,minlength=40) for x in sampled])
    pairs={};source_adjusted={}
    for ident,r in runs.items():
        if r['training_split']=='stock' or r['benchmark']=='libero_original':continue
        baseline=r['model']+('-screen-step0' if r['benchmark'].endswith('screen560') else '-plus-stock')
        target='noise' if r['training_split']=='noise_only' else 'camera'
        metrics=paired(data[baseline],data[ident],weights,target)
        if r['training_split']=='mixed':
            # Mixed training has no unique target domain.
            metrics={k:v for k,v in metrics.items() if k in DIMENSIONS or k=='overall_micro'}
        pairs[ident]={'baseline':baseline,'metrics':metrics}
    for model in MODELS:
        for split in ['camera_only','noise_only']:
            orig=f'{model}-original-{split.replace("_","-")}'
            original_change=stats[orig]['overall']-stats[f'{model}-original-stock']['overall']
            main_id=f'{model}-plus-{split}'
            source_adjusted[main_id]={d:pairs[main_id]['metrics'][d]['delta_pp']-original_change for d in DIMENSIONS}
    out=dict(bootstrap_draws=a.draws,bootstrap_seed=a.seed,confidence_level=.95,
             interval_method='paired task-family percentile bootstrap; pointwise, unadjusted',
             training_seed_uncertainty=False,absolute_rates=stats,suite_rates=suite_rates,paired=pairs,source_adjusted=source_adjusted)
    (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Recomputed results','','Success rates are percentages; changes are percentage points. Overall pools episodes. Off-target is the unweighted mean over six dimensions.','','| Run | Episodes | Successes | Overall (%) |','|---|---:|---:|---:|']
    for k,r in runs.items():lines.append(f'| {k} | {r["episodes"]} | {r["successes"]} | {stats[k]["overall"]:.3f} |')
    lines+=['','## Targeted tuning','','| Model | Training | Target base → tuned | Off-target change [95% CI] | Overall base → tuned |','|---|---|---:|---:|---:|']
    for model in MODELS:
        for split,target in [('camera_only','camera'),('noise_only','noise')]:
            m=pairs[f'{model}-plus-{split}']['metrics'];t=m[target];o=m['off_target_macro'];g=m['overall_micro']
            lo,hi=o['delta_ci95_pp']
            lines.append(f'| {model} | {split} | {t["stock"]:.2f} → {t["tuned"]:.2f} | {o["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] | {g["stock"]:.2f} → {g["tuned"]:.2f} |')
    lines+=['','## Training-seed sensitivity (fixed 560 episodes)','','| Model | Seed | Target gain | Off-target mean change | Overall gain |','|---|---:|---:|---:|---:|']
    for model in ['xvla','oft']:
        for seed in [42,43,44]:
            ident=f'{model}-screen-step12000' if seed==42 else f'{model}-screen-seed{seed}'
            m=pairs[ident]['metrics']
            lines.append(f'| {model} | {seed} | {m["camera"]["delta_pp"]:+.2f} | {m["off_target_macro"]["delta_pp"]:+.2f} | {m["overall_micro"]["delta_pp"]:+.2f} |')
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(runs)} runs / {sum(r["episodes"] for r in runs.values()):,} episodes; wrote {a.output}')

if __name__=='__main__':main()
