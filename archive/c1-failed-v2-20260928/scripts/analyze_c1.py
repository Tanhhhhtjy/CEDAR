"""Analyze C1 only when every planned cell passes integrity checks."""
import argparse,itertools,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from prepare_c1 import sha,save
from run_c1 import matrix,key

def summarize(values):
    names=sorted(values);x=np.array([values[n] for n in names])
    assert len(x)==12 and np.isfinite(x).all()
    rng=np.random.default_rng(20260928)
    boot=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    null=np.array(list(itertools.product([-1.,1.],repeat=len(x))))@x/len(x)
    return dict(equal_family_mean=float(x.mean()),family_means=values,
        descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),
        exact_signflip_p=float((np.abs(null)>=abs(x.mean())-1e-12).mean()),
        inference_limit='Symmetric independent family sign-flip assumption; cross-fit training sets overlap; development evidence.')

def main(root,out,scope):
    sys.path.insert(0,str(root/'scripts'));import analyze_s1 as s1
    original=json.loads((root/'runs/formal/results.json').read_text())
    plan=matrix(original,scope);run=out/'runs'/scope
    assert json.loads((run/'status.json').read_text())['stage']=='complete'
    records=json.loads((run/'results.json').read_text());assert len(records)==len(plan)
    assert {key(r['cell']) for r in records}=={key(c) for c in plan}
    scales=json.loads((out/'prepared/scales.json').read_text())
    orig_idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in original))}
    splits=json.loads((root/'analysis-output/formal/s1/token_splits.json').read_text())
    main_records=records if scope=='main' else json.loads((out/'runs/main/results.json').read_text())
    noops={(r['trace_id'],r['cell']['target']):r for r in main_records if r['cell']['operation']=='noop'}
    endpoints=[];family_groups=defaultdict(lambda:defaultdict(list))
    for r in records:
        c=r['cell'];p=run/r['array_file'];assert sha(p)==r['arrays_sha256'];assert r['technical_pass']
        base=noops[r['trace_id'],c['target']]
        z=np.array([r['logit_diagnostics']['logits'][k] for k in 'ABCD'],float)
        z0=np.array([base['logit_diagnostics']['logits'][k] for k in 'ABCD'],float)
        item={k:r[k] for k in ['trace_id','source_id','base_id','family_id','arm']}
        item.update(cell=c,patch=r['patch'],argmax_changed=int(z.argmax()!=z0.argmax()),metrics={})
        for correction in ['raw','same','family']:
            a,b=1.,0.
            if c['target']=='P' and correction!='raw':
                fit=scales[r['trace_id']][correction];a,b=fit['slope'],fit['intercept'];assert a>0
            m=s1.margin((z-b)/a,r['initial_answer'],r['target_answer'])
            m0=s1.margin((z0-b)/a,r['initial_answer'],r['target_answer'])
            item['metrics'][correction]={k:m[k]-m0[k] for k in ['broad','specific']}
        # Predetermined diagnostic for NEW perturbation-scale effects, separate from locked primary coefficients.
        with np.load(p) as zz: patched=zz['logits'].astype(float)
        np0=(out/'runs/main'/base['array_file']) if scope=='dose' else run/base['array_file']
        with np.load(np0) as zz:unpatched=zz['logits'].astype(float)
        split=splits[r['trace_id']]
        a,b=s1.fit(s1.sufficient(unpatched,patched,split))['affine']
        item['new_patch_scale_diagnostic']=dict(slope=a,intercept=b,
            heldout_nonoption={k:s1.err(patched[v['test']],a*unpatched[v['test']]+b) for k,v in split.items()},
            note='Secondary scale fit; not used to replace locked primary endpoint.')
        endpoints.append(item)
    lookup={(e['base_id'],e['arm'],e['cell']['target'],e['cell']['operation'],
        e['cell']['dose'],e['cell']['random_index']):e for e in endpoints}
    bases=sorted({r['base_id'] for r in records})
    primary={};secondary={}
    if scope=='main':
        for corr in ['raw','same','family']:
            for target in 'UP':
                for metric in ['broad','specific']:
                    fams=defaultdict(list);rawf=defaultdict(list);rndf=defaultdict(list)
                    for base in bases:
                        def effect(j):
                            v=lookup[base,'V',target,'component',0.,j]
                            f=lookup[base,'F',target,'component',0.,j]
                            return v['metrics'][corr][metric]-f['metrics'][corr][metric],v['family_id']
                        true,fam=effect(-1);random=np.mean([effect(j)[0] for j in range(8)])
                        fams[fam].append(true-random);rawf[fam].append(true);rndf[fam].append(random)
                    name=f'{corr}_{target}_{metric}'
                    stat=summarize({f:float(np.mean(v)) for f,v in fams.items()})
                    stat['true_mean']=float(np.mean([np.mean(v) for v in rawf.values()]))
                    stat['random_mean']=float(np.mean([np.mean(v) for v in rndf.values()]))
                    (primary if corr=='same' else secondary)[name]=stat
        running=0.
        for i,(name,stat) in enumerate(sorted(primary.items(),key=lambda v:v[1]['exact_signflip_p'])):
            running=max(running,min(1.,(4-i)*stat['exact_signflip_p']))
            primary[name]['holm_four_primary']=running
    else:
        for corr in ['raw','same','family']:
            for dose in [.5,1.]:
                for metric in ['broad','specific']:
                    fams=defaultdict(list)
                    for base in bases:
                        def sensitivity(target,j):
                            vpos=lookup[base,'V',target,'dose',dose,j]
                            vneg=lookup[base,'V',target,'dose',-dose,j]
                            fpos=lookup[base,'F',target,'dose',dose,j]
                            fneg=lookup[base,'F',target,'dose',-dose,j]
                            return ((vpos['metrics'][corr][metric]-vneg['metrics'][corr][metric])-
                                (fpos['metrics'][corr][metric]-fneg['metrics'][corr][metric]))/(2*dose),vpos['family_id']
                        u,fam=sensitivity('U',-1);p,_=sensitivity('P',-1)
                        random=np.mean([sensitivity('P',j)[0]-sensitivity('U',j)[0] for j in range(8)])
                        fams[fam].append((p-u)-random)
                    secondary[f'{corr}_P_minus_U_alpha{dose}_{metric}']=summarize({f:float(np.mean(v)) for f,v in fams.items()})
    analysis=out/'analysis'/scope
    save(analysis/'details.json',endpoints)
    result=dict(stage=scope,status='completed_development_analysis',n_forwards=len(records),
        structural_families=12,primary=primary,secondary=secondary,
        argmax_changes=sum(e['argmax_changed'] for e in endpoints),
        noops=max((r['checks'].get('baseline_all_arrays_max_abs',0) for r in records),default=0),
        limits=['No natural-generation behavior measured.','H1 independent structural confirmation pending.',
            'Unit-RMS linear coordinate intervention is not an exact spherical component swap.',
            'Locked S1 calibration does not eliminate all new perturbation-specific scale effects.',
            'A negative result is limited to this site, learned direction and dose.',
            'Matched random controls and N/R effects must be reviewed before any causal specificity claim.'])
    save(analysis/'SUMMARY.json',result)
    print(json.dumps(dict(status=result['status'],scope=scope,n=len(records))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scope',choices=['main','dose'],required=True)
    a=p.parse_args();main(a.root,a.out,a.scope)

