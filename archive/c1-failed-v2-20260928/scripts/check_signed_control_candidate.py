"""CPU-only prospective signed-control check. Does not modify C1 v2 gates or run a model."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from c1_core import make_patch

ROOT=Path('/data4/WuMingrui/TianJinyu/CEDAR')
OUT=ROOT/'c1_20260928_v2'
def read(p):return json.loads(p.read_text())
def main():
    records=read(ROOT/'runs/formal/results.json')
    rows=sorted([r for r in records if r['condition']=='uncapped' and r['trace_source']=='e2a_uncapped'],key=lambda r:r['source_id'])
    assert len(rows)==96 and set(r['arm'] for r in rows)==set('NRVF')
    specs=read(OUT/'prepared/directions.json')
    idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in records))}
    first_base=min(r['base_id'] for r in rows)
    allchecks=[];failures=[];smoke_failures=[];maxerr=0.
    for r in rows:
        spec=specs[r['family_id']+'_U']
        with np.load(OUT/spec['file']) as z:d=z['direction'];random=z['random_directions']
        caches={c:np.load(ROOT/f"runs/formal/{idx[r['trace_id']]:04d}_{n}.npz")['prompt_layer_53'] for c,n in [('U','uncapped'),('P','full_p25')]}
        for target in 'PU':
            donor='U' if target=='P' else 'P'
            for op,dose,j in [('noop',0.,-1),('replace',0.,-1)]+[('dose',a,j) for a in [-1.,1.] for j in [-1,*range(8)]]:
                cell=dict(trace_id=r['trace_id'],family=r['family_id'],base=r['base_id'],arm=r['arm'],target=target,operation=op,dose=dose,random_index=j)
                try:
                    # Matching scalar stays nonnegative; fixed null direction follows dose sign.
                    random_vector=None if j<0 else np.sign(dose)*random[j]
                    _,stats=make_patch(caches[target],caches[donor],d,spec['sigma'],op,dose,random_vector)
                    maxerr=max(maxerr,stats['random_norm_relative_error'] or 0.)
                    allchecks.append(dict(cell=cell,passed=True,realized_norm=stats['realized_norm'],relative_error=stats['random_norm_relative_error']))
                except RuntimeError as e:
                    fail=dict(cell=cell,error=str(e));failures.append(fail)
                    if r['base_id']==first_base and j<=0:smoke_failures.append(fail)
                    allchecks.append(dict(cell=cell,passed=False,error=str(e)))
    assert len(allchecks)==3840
    hours=sum(r['seconds'] for r in records if r['trace_source']=='e2a_uncapped')*20/3600
    smoke_hours=sum(r['seconds'] for r in records if r['trace_source']=='e2a_uncapped' and r['base_id']==first_base)*6/3600
    smoke_hours+=sum(r['seconds'] for r in records if r['trace_source']=='e2a_uncapped' and r['base_id']==first_base and r['arm']=='V')*2/3600
    result=dict(status='prospective_cpu_only_passed' if not failures else 'prospective_cpu_only_blocked',
        timestamp_utc=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),gpu_authorized=False,
        design='All 96 N/R/V/F prompts, both targets, noop/replace and +/-1 training SD along real and 8 fixed sign-aligned random directions.',
        norm_tolerance=.005,formal_candidates=3840,failures=failures,max_norm_relative_error=maxerr,
        proposed_smoke=dict(base=first_base,forwards=52,geometry_failures=smoke_failures,historical_hours=smoke_hours),
        historical_formal_hours=hours,checks=allchecks,
        warnings=['Numerical geometry feasibility only, no model outputs and no causal result.',
            'This prospective matrix changes the blocked v2 protocol and requires a separate decision.',
            'No v2 pass/fail record was modified.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'explore_outputs/signed_control_candidate.json'
    if path.exists():raise RuntimeError('Candidate diagnostic already exists; preserve it.')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','formal_candidates','max_norm_relative_error','proposed_smoke','historical_formal_hours']}),flush=True)
    print('failure_count',len(failures),flush=True)

if __name__=='__main__':main()
