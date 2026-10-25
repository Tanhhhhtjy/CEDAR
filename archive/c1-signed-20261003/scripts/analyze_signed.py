"""Analyze only a complete, fingerprinted signed-dose matrix."""
import argparse,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import read,save,sha,matrix,key,verify_prepared
from analysis_core import endpoints,summarize,holm

def main(root,out):
    prep=verify_prepared(root,out)
    sys.path.insert(0,str(root/'scripts'));import analyze_s1 as s1
    original=read(root/'runs/formal/results.json');plan=matrix(original,'formal')
    run=out/'runs/formal';assert read(run/'status.json')['stage']=='complete'
    assert read(out/'runs/smoke/gate.json')['technical_pass']
    rs=read(run/'results.json');assert len(rs)==len(plan)==3840
    assert {key(r['cell']) for r in rs}=={key(c) for c in plan}
    fingerprint=read(run/'manifest.json');assert fingerprint['prepared_sha256']==sha(out/'prepared/manifest.json')
    assert len({r['base_id'] for r in rs})==24 and len({r['family_id'] for r in rs})==12
    scales=read(out/'prepared/scales.json');splits=read(root/'analysis-output/formal/s1/token_splits.json')
    noops={(r['trace_id'],r['cell']['target']):r for r in rs if r['cell']['operation']=='noop'}
    details=[]
    for n,r in enumerate(rs):
        c=r['cell'];assert r['technical_pass'] and r['fingerprint']==fingerprint
        p=run/r['array_file'];assert sha(p)==r['arrays_sha256']
        with np.load(p) as f:
            assert all(np.isfinite(f[k]).all() for k in f.files)
            patched=f['logits'].astype(float)
        base=noops[r['trace_id'],c['target']]
        with np.load(run/base['array_file']) as f:unpatched=f['logits'].astype(float)
        z=np.array([r['logit_diagnostics']['logits'][k] for k in 'ABCD'],float)
        z0=np.array([base['logit_diagnostics']['logits'][k] for k in 'ABCD'],float)
        item={k:r[k] for k in ['trace_id','source_id','base_id','family_id','arm']}
        item.update(cell=c,patch=r['patch'],argmax_changed=int(z.argmax()!=z0.argmax()),
            option_order=np.argsort(-z,kind='stable').tolist(),baseline_option_order=np.argsort(-z0,kind='stable').tolist(),metrics={})
        for correction in ['raw','same','family']:
            a,b=1.,0.
            if c['target']=='P' and correction!='raw':
                fit=scales[r['trace_id']][correction];a,b=fit['slope'],fit['intercept'];assert a>0
            m=s1.margin((z-b)/a,r['initial_answer'],r['target_answer'])
            m0=s1.margin((z0-b)/a,r['initial_answer'],r['target_answer'])
            item['metrics'][correction]={k:float(m[k]-m0[k]) for k in ['broad','specific']}
        split=splits[r['trace_id']]
        a,b=s1.fit(s1.sufficient(unpatched,patched,split))['affine']
        item['new_patch_scale_diagnostic']=dict(slope=a,intercept=b,
            heldout_nonoption={k:s1.err(patched[v['test']],a*unpatched[v['test']]+b) for k,v in split.items()},
            note='Secondary fit; never replaces locked primary coefficients.')
        details.append(item)
        if n%100==0:print(f'Analysis integrity and diagnostics {n}/{len(rs)}',flush=True)
    lookup={(e['base_id'],e['arm'],e['cell']['target'],e['cell']['operation'],e['cell']['dose'],e['cell']['random_index']):e for e in details}
    bases=sorted({r['base_id'] for r in rs});primary={};secondary={};base_endpoints=[]
    for corr in ['same','raw','family']:
        for metric in ['broad','specific']:
            groups=defaultdict(lambda:defaultdict(list));arm_groups=defaultdict(lambda:defaultdict(list))
            for base in bases:
                responses={}
                for c in 'UP':
                    responses[c]={}
                    for arm in 'NRVF':
                        vals=[]
                        for j in [-1,*range(8)]:
                            pos=lookup[base,arm,c,'dose',1.,j];neg=lookup[base,arm,c,'dose',-1.,j]
                            fam=pos['family_id']
                            val=(pos['metrics'][corr][metric]-neg['metrics'][corr][metric])/2
                            vals.append(val);arm_groups[f'{c}_{arm}_direction{j}'][fam].append(val)
                        responses[c][arm]=vals
                        arm_groups[f'{c}_{arm}_true_minus_random'][fam].append(vals[0]-float(np.mean(vals[1:])))
                effects=endpoints(responses)
                for name,v in effects.items():groups[name][fam].append(v)
                base_endpoints.append(dict(base_id=base,family_id=fam,correction=corr,metric=metric,effects=effects,responses=responses))
            for name,fams in groups.items():
                stat=summarize({f:float(np.mean(v)) for f,v in fams.items()})
                (primary if corr=='same' else secondary)[f'{corr}_{metric}_{name}']=stat
            for name,fams in arm_groups.items():secondary[f'{corr}_{metric}_{name}']=summarize({f:float(np.mean(v)) for f,v in fams.items()})
    assert len(primary)==4;holm(primary)
    analysis=out/'analysis/formal';save(analysis/'details.json',details);save(analysis/'base_endpoints.json',base_endpoints)
    result=dict(status='completed_development_analysis',n_forwards=3840,structural_families=12,
        primary=primary,secondary=secondary,prepared_sha256=sha(out/'prepared/manifest.json'),
        argmax_changes=sum(e['argmax_changed'] for e in details),
        noops=max(r['checks'].get('baseline_all_arrays_max_abs',0) for r in rs),
        limits=['Fixed trajectory terminal readout; no new generation.','H1 independent structural confirmation pending.',
        'Cross-condition actual raw displacement is not exactly equal; review patch telemetry.',
        'Locked S1 correction does not remove all patch-induced scaling.',
        'Directional causality at one site and dose is not proof of abstract validity or natural mediation.',
        'Cross-fit training overlaps; family intervals and tests are exploratory.','Nonsignificance alone is not equivalence.'])
    save(analysis/'SUMMARY.json',result);print('Complete 3840-cell C1 analysis saved.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
