"""Descriptive interim report; primary six tests only after B1+B2 completion."""
import argparse,sys,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from common import *
SEED=20261005
def summarize(vals):
    keys=sorted(vals);v=np.array([vals[k] for k in keys]);rng=np.random.default_rng(SEED)
    bs=v[rng.integers(len(v),size=(20000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95_descriptive=np.quantile(bs,[.025,.975]).tolist(),units=dict(zip(keys,v.tolist())),sd=float(v.std(ddof=1)))
def contrasts(y):
    z=dict(total=y['11']-y['00'],EH=y['10']-y['00'],ET=y['01']-y['00'],I=y['11']-y['10']-y['01']+y['00'],ET_given_H=y['11']-y['10'],EH_given_T=y['11']-y['01'])
    assert abs(z['total']-z['EH']-z['ET']-z['I'])<1e-8
    return z
def main(root,out,scope):
    verify(root,out);sys.path.insert(0,str(root/'scripts'));import analyze_s1 as s1
    batches=['B1','B2'] if scope=='pooled' else [scope];bytrace=defaultdict(dict)
    for batch in batches:
        records=read(out/f'runs/{batch}/results.json');assert len(records)==192
        for r in records:
            assert not r['cell']['repeat'] and sha(out/f'runs/{batch}'/r['array_file'])==r['arrays_sha256']
            bytrace[r['trace_id']][r['cell']['phase']]=(r,batch)
    details=[];scales={}
    for tid,pair in bytrace.items():
        assert set(pair)=={'00','10','01','11'};r,batch=pair['00'];run=out/'runs'/batch
        x=np.load(run/r['array_file'])['logits'].astype(float);ids=[r['candidate_token_ids'][k] for k in 'ABCD']
        excluded=set(read(run/'runtime.json')['excluded_fit_token_ids'])|set(ids);splits=s1.token_split(x,excluded,tid)
        scales[tid]={};row={k:r[k] for k in ['trace_id','family_id','structure_class','base_id','arm','initial_answer','target_answer']};row.update(raw={},corrected={})
        for phase,(rr,bb) in pair.items():
            y=np.load(out/f'runs/{bb}'/rr['array_file'])['logits'].astype(float)
            a,b=(1.,0.) if phase=='00' else s1.fit(s1.sufficient(x,y,splits))['affine']
            scales[tid][phase]=dict(slope=a,intercept=b,positive=a>0,test_errors={n:s1.err(y[v['test']],a*x[v['test']]+b) for n,v in splits.items()})
            row['raw'][phase]=s1.margin(y[ids],r['initial_answer'],r['target_answer'])
            row['corrected'][phase]=s1.margin((y[ids]-b)/a,r['initial_answer'],r['target_answer']) if a>0 else None
        details.append(row)
    bases=defaultdict(dict)
    for r in details:bases[r['base_id']][r['arm']]=r
    endpoints=[]
    for b,arms in bases.items():
        assert set(arms)==set('NRVF')
        for scale in ['raw','corrected']:
            for metric in ['broad','specific']:
                if any(arms[a][scale][c] is None for a in 'NRVF' for c in ['00','10','01','11']):
                    endpoints.append(dict(base_id=b,scale=scale,metric=metric,status='invalid_nonpositive_scale'));continue
                y={c:float(np.mean([arms[a][scale][c][metric] for a in 'NRVF'])) if metric=='broad' else arms['V'][scale][c][metric]-arms['F'][scale][c][metric] for c in ['00','10','01','11']}
                endpoints.append(dict(base_id=b,family_id=arms['V']['family_id'],structure_class=arms['V']['structure_class'],scale=scale,metric=metric,values=y,contrasts=contrasts(y)))
    summaries={}
    for scale in ['raw','corrected']:
        for metric in ['broad','specific']:
            group=[r for r in endpoints if r['scale']==scale and r['metric']==metric and 'contrasts' in r]
            if len(group)!=len(bases):continue
            for c in ['total','EH','ET','I','ET_given_H','EH_given_T']:
                fams=defaultdict(list);structure={}
                for r in group:fams[r['family_id']].append(r['contrasts'][c]);structure[r['family_id']]=r['structure_class']
                fm={f:float(np.mean(v)) for f,v in fams.items()};gs=defaultdict(list)
                for f,v in fm.items():gs[structure[f]].append(v)
                summaries[f'{scale}_{metric}_{c}']=dict(structure_equal=summarize({g:float(np.mean(v)) for g,v in gs.items()}),family_equal=summarize(fm))
    if scope=='pooled':
        ps={}
        for m in ['broad','specific']:
            for c in ['EH','ET','I']:
                k=f'raw_{m}_{c}';v=np.array(list(summaries[k]['structure_equal']['units'].values()));assert len(v)==8
                p=float(np.mean(np.abs(np.array(list(itertools.product([-1,1],repeat=8)))@v/8)>=abs(v.mean())-1e-12));ps[k]=p
        running=0.
        for i,(k,p) in enumerate(sorted(ps.items(),key=lambda kv:kv[1])):
            running=max(running,min(1.,(6-i)*p));summaries[k]['primary_test']=dict(p_exact=p,p_holm=running,assumption='Independent symmetric structure effects; development evidence only')
    dest=out/'analysis'/scope
    save(dest/'details.json',details);save(dest/'base_endpoints.json',endpoints);save(dest/'frozen_scales.json',scales);save(dest/'SUMMARY.json',dict(scope=scope,bases=len(bases),summaries=summaries,status='descriptive_interim_no_primary_tests' if scope!='pooled' else 'complete_development_analysis',limitations=['Reused development structures','Intervals descriptive; group independence unproven','No equivalence boundary','Fixed replay is not natural behavior']))
    lines=['# B factorial results',f'Scope: {scope}. Bases: {len(bases)}. Fixed-trajectory development evidence.','|Raw outcome|Contrast|8-group mean|Descriptive 95% interval|','|---|---|---:|---|']
    for m in ['broad','specific']:
        for c in ['total','EH','ET','I','ET_given_H','EH_given_T']:
            s=summaries[f'raw_{m}_{c}']['structure_equal'];lines.append(f'|{m}|{c}|{s["mean"]:.6g}|{s["ci95_descriptive"]}|')
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--scope',choices=['B1','B2','pooled'],default='B1');a=p.parse_args();main(a.root,a.out,a.scope)
