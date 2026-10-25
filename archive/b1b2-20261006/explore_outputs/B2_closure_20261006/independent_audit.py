"""Post-completion audit: reads frozen results; never invokes model or original analysis."""
import json, hashlib, itertools, sys, time, os, fcntl
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path('/data4/WuMingrui/TianJinyu/CEDAR')
OUT=ROOT/'arr_next_steps_20261005'
DEST=OUT/'explore_outputs/B2_closure_20261006'
DEST.mkdir(exist_ok=True)
def read(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
errors=defaultdict(float)
def close(a,b,label):
    aa=np.asarray(a,dtype=float);bb=np.asarray(b,dtype=float)
    assert aa.shape==bb.shape,(label,aa.shape,bb.shape)
    delta=float(np.max(np.abs(aa-bb))) if aa.size else 0.
    errors[label]=max(errors[label],delta)
    assert np.allclose(aa,bb,rtol=1e-10,atol=1e-10),(label,delta)

def run():
    sys.path.insert(0,str(OUT/'operations/b2_20261006'))
    from b2_support import scope_audit,verify_prior_unchanged,verify_package,analysis_receipt
    sys.path.insert(0,str(OUT/'scripts'))
    from common import verify
    sys.path.insert(0,str(OUT/'operations'))
    from takeover import verify_operations
    verify(ROOT,OUT);verify_operations(OUT);verify_package(OUT)
    pre=read(OUT/'explore_outputs/B2_start_20261006T035159Z/preflight.json')
    verify_prior_unchanged(OUT,pre)
    scopes={s:scope_audit(OUT,s) for s in ['smoke','B1','B2']}
    for scope,n in [('smoke',24),('B1',192),('B2',192)]:
        assert scopes[scope]['completed']==n
        array_names={Path(p).name for p in scopes[scope]['hashes'] if p.endswith('.npz')}
        assert {p.name for p in (OUT/'runs'/scope).glob('*.npz')}==array_names
    for scope in ['B2','pooled']:
        assert analysis_receipt(OUT,scope)==read(OUT/f'explore_outputs/{scope}_analysis_receipt.json')
    prev=read(OUT/'explore_outputs/heartbeat_B2_20261006T092347Z.json')
    for scope in ['smoke','B1','B2']:
        for name,digest in prev['scopes'][scope]['verified_file_hashes'].items():
            assert scopes[scope]['hashes'][f'runs/{scope}/{name}']==digest
    print('Record hashes, first package preservation and manifests passed',flush=True)
    assets=read(OUT/'prepared/asset_verification.json')['files']
    verified_assets={}
    for idx,(name,meta) in enumerate(assets.items(),1):
        p=Path(name);st=p.stat()
        assert st.st_size==meta['size'] and st.st_mtime_ns==meta['mtime_ns']
        digest=sha(p);assert digest==meta['sha256']==pre['asset_hashes'][name]
        verified_assets[name]=digest
        print(f'Full asset hashes {idx}/{len(assets)}',flush=True)
    save(DEST/'asset_rehash.json',dict(status='passed',time=time.time(),files=verified_assets))
    rows=[];split_hashes={};scale_results={};choices={};phase_names=['00','10','01','11']
    for batch in ['B1','B2']:
        saved_details={r['trace_id']:r for r in read(OUT/f'analysis/{batch}/details.json')}
        saved_scales=read(OUT/f'analysis/{batch}/frozen_scales.json')
        records=read(OUT/f'runs/{batch}/results.json');pairs=defaultdict(dict)
        for r in records:pairs[r['trace_id']][r['cell']['phase']]=r
        assert len(pairs)==48
        excluded_runtime=set(read(OUT/f'runs/{batch}/runtime.json')['excluded_fit_token_ids'])
        for tid,pair in pairs.items():
            assert set(pair)==set(phase_names)
            ref=pair['00'];ids=[ref['candidate_token_ids'][k] for k in 'ABCD']
            with np.load(OUT/f'runs/{batch}'/ref['array_file']) as z:x=z['logits'].astype(float)
            excluded=excluded_runtime|set(ids)
            rng=np.random.default_rng(int(hashlib.sha256(('20260927'+tid).encode()).hexdigest()[:16],16))
            rank=np.argsort(-x,kind='stable');splits={}
            for name,subset in [('top256',rank[:256]),('rank257_4096',rank[256:4096]),('tail',rank[4096:])]:
                eligible=np.array([k for k in subset if int(k) not in excluded])
                shuffled=rng.permutation(eligible)[:1024];n=len(shuffled)//2
                splits[name]={'fit':shuffled[:n].tolist(),'test':shuffled[n:].tolist()}
                assert not (set(splits[name]['fit'])&set(splits[name]['test']))
            split_hashes[tid]=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
            fitids=np.concatenate([s['fit'] for s in splits.values()])
            weights=np.concatenate([np.full(len(s['fit']),1/(3*len(s['fit']))) for s in splits.values()])
            design=np.column_stack([x[fitids],np.ones(len(fitids))])*np.sqrt(weights[:,None])
            row={k:ref[k] for k in ['trace_id','base_id','family_id','structure_class','arm','initial_answer','target_answer']}
            row.update(batch=batch,raw={},corrected={});scale_results[tid]={};choices[tid]={}
            for phase in phase_names:
                r=pair[phase]
                for k in ['trace_id','base_id','family_id','structure_class','arm','initial_answer','target_answer']:
                    assert r[k]==ref[k]
                with np.load(OUT/f'runs/{batch}'/r['array_file']) as z:y=z['logits'].astype(float)
                a,b=(1.,0.) if phase=='00' else np.linalg.lstsq(design,y[fitids]*np.sqrt(weights),rcond=None)[0]
                stored=saved_scales[tid][phase]
                close([a,b],[stored['slope'],stored['intercept']],'affine_coefficients')
                assert bool(a>0)==stored['positive']
                scale_results[tid][phase]={'slope':float(a),'intercept':float(b),'positive':bool(a>0),'test_errors':{}}
                for name,split in splits.items():
                    residual=y[split['test']]-(a*x[split['test']]+b)
                    metrics=dict(mae=float(np.mean(np.abs(residual))),rmse=float(np.sqrt(np.mean(residual**2))),centered_rmse=float(np.std(residual)))
                    for k,v in metrics.items():close(v,stored['test_errors'][name][k],'heldout_errors')
                    scale_results[tid][phase]['test_errors'][name]=metrics
                z=y[ids];i='ABCD'.index(ref['initial_answer']);t='ABCD'.index(ref['target_answer'])
                background=[k for k in range(4) if k not in [i,t]]
                for scale,options in [('raw',z),('corrected',(z-b)/a if a>0 else None)]:
                    values=None if options is None else dict(broad=float(np.mean([options[j]-options[i] for j in range(4) if j!=i])),specific=float(np.mean([options[t]-options[j] for j in background])))
                    row[scale][phase]=values
                    saved=saved_details[tid][scale][phase]
                    if values is None:assert saved is None
                    else:
                        for k,v in values.items():close(v,saved[k],'trace_'+scale)
                choices[tid][phase]='ABCD'[int(np.argmax(z))]
            rows.append(row)
    print('Independent weighted least-squares fits and held-out errors passed for 96 traces',flush=True)
    all_summary={};primary={};diagnostics={}
    for scope,batches in [('B1',['B1']),('B2',['B2']),('pooled',['B1','B2'])]:
        selected=[r for r in rows if r['batch'] in batches];bases=defaultdict(dict)
        for r in selected:bases[r['base_id']][r['arm']]=r
        saved=read(OUT/f'analysis/{scope}/SUMMARY.json')
        endpoints={(e['base_id'],e['scale'],e['metric']):e for e in read(OUT/f'analysis/{scope}/base_endpoints.json')}
        saved_details={r['trace_id']:r for r in read(OUT/f'analysis/{scope}/details.json')}
        pooled_scales=read(OUT/f'analysis/{scope}/frozen_scales.json')
        for r in selected:
            tid=r['trace_id']
            for phase in phase_names:
                close([scale_results[tid][phase]['slope'],scale_results[tid][phase]['intercept']],[pooled_scales[tid][phase]['slope'],pooled_scales[tid][phase]['intercept']],'scope_coefficients')
                for scale in ['raw','corrected']:
                    if r[scale][phase] is None:assert saved_details[tid][scale][phase] is None
                    else:
                        for m,v in r[scale][phase].items():close(v,saved_details[tid][scale][phase][m],'scope_trace')
        summaries={}
        for scale in ['raw','corrected']:
            for metric in ['broad','specific']:
                effects=[]
                for base,arms in bases.items():
                    assert set(arms)==set('NRVF')
                    if any(arms[a][scale][p] is None for a in 'NRVF' for p in phase_names):
                        assert endpoints[base,scale,metric]['status']=='invalid_nonpositive_scale';continue
                    y={p:float(np.mean([arms[a][scale][p][metric] for a in 'NRVF'])) if metric=='broad' else arms['V'][scale][p][metric]-arms['F'][scale][p][metric] for p in phase_names}
                    d={'EH':y['10']-y['00'],'ET':y['01']-y['00'],'total':y['11']-y['00'],'ET_given_H':y['11']-y['10'],'EH_given_T':y['11']-y['01']}
                    d['I']=d['ET_given_H']-d['ET'];close(d['total'],d['EH']+d['ET']+d['I'],'factorial_identity')
                    e=endpoints[base,scale,metric]
                    for k,v in y.items():close(v,e['values'][k],'base_'+scale)
                    for k,v in d.items():close(v,e['contrasts'][k],'contrasts_'+scale)
                    effects.append((arms['V']['family_id'],arms['V']['structure_class'],d))
                if len(effects)!=len(bases):continue
                for contrast in ['total','EH','ET','I','ET_given_H','EH_given_T']:
                    families=defaultdict(list);groups=defaultdict(list);fgroup={}
                    for f,g,d in effects:families[f].append(d[contrast]);fgroup[f]=g
                    fm={f:float(np.mean(v)) for f,v in families.items()}
                    for f,v in fm.items():groups[fgroup[f]].append(v)
                    gm={g:float(np.mean(v)) for g,v in groups.items()};key=f'{scale}_{metric}_{contrast}'
                    summaries[key]={}
                    for name,units in [('structure_equal',gm),('family_equal',fm)]:
                        values=np.array([units[k] for k in sorted(units)])
                        boot=values[np.random.default_rng(20261005).integers(len(values),size=(20000,len(values)))].mean(1)
                        s=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),ci95_descriptive=np.quantile(boot,[.025,.975]).tolist(),units=units)
                        target=saved['summaries'][key][name]
                        assert set(units)==set(target['units'])
                        for k in ['mean','sd','ci95_descriptive']:close(s[k],target[k],'summary_'+scale)
                        for g,v in units.items():close(v,target['units'][g],'unit_'+scale)
                        summaries[key][name]=s
        tested={k for k,v in saved['summaries'].items() if 'primary_test' in v}
        expected={f'raw_{m}_{c}' for m in ['broad','specific'] for c in ['EH','ET','I']} if scope=='pooled' else set()
        assert tested==expected
        if scope=='pooled':
            for k in sorted(expected):
                vals=list(summaries[k]['structure_equal']['units'].values());assert len(vals)==8
                observed=abs(sum(vals));extreme=sum(abs(sum(v*s for v,s in zip(vals,signs)))>=observed-8e-12 for signs in itertools.product([-1.,1.],repeat=8))
                primary[k]=dict(p_exact=extreme/256)
            ordered=sorted(primary,key=lambda k:primary[k]['p_exact']);running=0.
            for rank,k in enumerate(ordered):
                running=max(running,min(1.,(6-rank)*primary[k]['p_exact']));primary[k]['p_holm']=running
                for field,v in primary[k].items():close(v,saved['summaries'][k]['primary_test'][field],'primary_tests')
        all_summary[scope]=summaries
        tids=[r['trace_id'] for r in selected]
        diagnostics[scope]=dict(traces=len(tids),any_option_argmax_change=sum(len(set(choices[tid].values()))>1 for tid in tids),nonpositive_slopes=sum(not scale_results[tid][p]['positive'] for tid in tids for p in ['10','01','11']))
    save(DEST/'independent_summaries.json',all_summary)
    save(DEST/'independent_scales.json',scale_results)
    save(DEST/'split_hashes.json',split_hashes)
    save(DEST/'option_choices.json',choices)
    save(DEST/'records_audit.json',scopes)
    # Read /proc only; never signal another process. Exact output argument avoids our audit session.
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode().split('\0')
            if '--out' in argv and argv[argv.index('--out')+1]==str(OUT):
                active.append(dict(pid=int(p.name),argv=argv,uid=p.stat().st_uid,cwd=os.readlink(p/'cwd')))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not active,active
    locks={}
    for name in ['campaign.lock','runner.lock']:
        with (OUT/name).open() as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]='free'
    budget=read(OUT/'explore_outputs/gpu_budget.json');assert not budget['active']
    assert read(OUT/'explore_outputs/campaign_status.json')['stage']=='B2_and_pooled_complete_C_D_H_not_started'
    verify_prior_unchanged(OUT,pre)
    result=dict(status='passed',time=time.time(),records={k:v['completed'] for k,v in scopes.items()},full_asset_hashes=len(verified_assets),prior_package_unchanged=True,prior_package_files=len(pre['prior_package_file_hashes']),max_abs_errors=dict(errors),primary_tests=primary,diagnostics=diagnostics,active_managed_processes=active,locks=locks,budget=budget,method='Independent array margins, weighted least-squares affine fits, frozen token splits, endpoint reconstruction, 20000-draw descriptive bootstrap and 256 sign flips with six-test Holm verification; original analyses unchanged',limitations=['24 bases within 12 families and 8 reused development structures, not 384 independent samples','Structure independence and symmetric effects are assumptions, not established facts','Intervals descriptive, not simultaneous Holm-adjusted intervals','Fixed replay is not natural generation; no equivalence claim'])
    save(DEST/'independent_audit.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:run()
    except Exception:
        import traceback
        save(DEST/f'audit_failure_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
