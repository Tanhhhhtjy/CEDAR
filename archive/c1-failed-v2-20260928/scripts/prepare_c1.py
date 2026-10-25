"""Prepare C1 v2 train-only directions and post-hoc S1/P1 diagnostics on server CPU."""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(p)

def main(root, out):
    sys.path.insert(0,str(root/'scripts'))
    import analyze_p1 as p1
    import analyze_s1 as s1
    run=root/'runs/formal'
    records=json.loads((run/'results.json').read_text())
    assert len(records)==384 and json.loads((run/'status.json').read_text())['stage']=='complete'
    original=json.loads((run/'manifest.json').read_text())
    for field,p in [('protocol_sha256',root/'PROTOCOL.md'),
            ('source_runner_sha256',root/'CEDAR_SUPPORTING_DATA/code/scripts/run_e2b_fixed_trace_cached.py'),
            ('trace_sha256',root/'inputs/frozen_traces192.jsonl'),
            ('supporting_archive_sha256',root/'CEDAR_SUPPORTING_DATA.zip')]:
        assert sha(p)==original[field],(field,p)
    ids=list(dict.fromkeys(r['trace_id'] for r in records))
    idx={t:i for i,t in enumerate(ids)}
    checked={}
    for r in records:
        p=run/f"{idx[r['trace_id']]:04d}_{r['condition']}.npz"
        got=sha(p);assert got==r['arrays_sha256'],str(p)
        checked[str(p.relative_to(root))]=got
    rows=[r for r in records if r['condition']=='uncapped' and r['trace_source']=='e2a_uncapped' and r['arm'] in 'VF']
    assert len(rows)==48
    families=np.array([r['family_id'] for r in rows])
    y=np.array([r['arm']=='V' for r in rows],int)
    nuisance=p1.design(rows)
    feats={c:np.array([np.load(run/f"{idx[r['trace_id']]:04d}_{cnd}.npz")['prompt_layer_53'] for r in rows],float)
           for c,cnd in [('U','uncapped'),('P','full_p25')]}
    q={c:p1.normalize(f,'unit_rms') for c,f in feats.items()}
    old=json.loads((root/'analysis-output/formal/p1/P1_SUMMARY.json').read_text())
    oldprimary=old['results'][old['primary']]
    scores={a+'->'+b:np.zeros(len(y)) for a in 'UP' for b in 'UP'}
    directions=out/'prepared/directions';directions.mkdir(parents=True,exist_ok=True)
    spec={}
    for fam in sorted(set(families)):
        train=families!=fam;test=~train
        for source in 'UP':
            beta=np.linalg.pinv(nuisance[train])@q[source][train]
            x=q[source]-nuisance@beta
            scaler=StandardScaler().fit(x[train])
            clf=LogisticRegression(C=1,solver='liblinear',dual=True,max_iter=10000,tol=1e-7,
                class_weight='balanced',random_state=20260927)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                clf.fit(scaler.transform(x[train]),y[train])
            d=clf.coef_[0]/scaler.scale_;d/=np.linalg.norm(d)
            sigma=float(np.std(x[train]@d,ddof=1));assert sigma>0
            rnd=np.stack([np.random.default_rng(int(hashlib.sha256(f"20260928:{fam}:{j}".encode()).hexdigest()[:16],16)).normal(size=d.size) for j in range(8)])
            rnd/=np.linalg.norm(rnd,axis=1,keepdims=True)
            file=directions/f'{fam}_{source}.npz'
            np.savez_compressed(file,direction=d,random_directions=rnd,beta=beta,
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
            spec[fam+'_'+source]=dict(file=str(file.relative_to(out)),sha256=sha(file),sigma=sigma,
                source=source,training_families=sorted(set(families[train])),
                training_trace_ids=[r['trace_id'] for r,t in zip(rows,train) if t])
            for target in 'UP':
                xt=q[target]-nuisance@beta
                scores[source+'->'+target][test]=clf.decision_function(scaler.transform(xt[test]))
    errors={k:float(np.max(np.abs(v-np.array(oldprimary[k]['scores'])))) for k,v in scores.items()}
    assert max(errors.values())<=1e-6,errors
    # Per-family bridge uses the already completed same-pipeline analyses.
    bridge={k:old['results'][k] for k in ['prompt_L53_unit_rms_nuisance1','prompt_L60_unit_rms_nuisance1',
             'answer_L53_unit_rms_nuisance1','answer_L60_unit_rms_nuisance1']}
    # Capped-source prompt should duplicate uncapped-source prompt exactly.
    prows=[r for r in records if r['condition']=='uncapped' and r['trace_source']!='e2a_uncapped' and r['arm'] in 'VF']
    byid={r['source_id']:r for r in prows}
    source_duplicate=0.
    capped={}
    for pos in ['prompt','answer']:
        for layer in [53,60]:
            f={c:np.array([np.load(run/f"{idx[byid[r['source_id']]['trace_id']]:04d}_{cnd}.npz")[f'{pos}_layer_{layer}'] for r in rows],float)
                for c,cnd in [('U','uncapped'),('P','full_p25')]}
            if pos=='prompt' and layer==53:
                source_duplicate=max(float(np.max(np.abs(f[c]-feats[c]))) for c in 'UP')
                assert source_duplicate<=1e-4
            if pos=='answer':
                capped[f'{pos}_L{layer}']=p1.evaluate(f,rows,y,'unit_rms',True)
    # Independent multiclass nuisance endpoints, with family-held-out predictions.
    identities={}
    for label in ['initial_answer','target_answer']:
        labels=np.array([r[label] for r in rows])
        for adjusted in [False,True]:
            preds={a+'->'+b:np.full(len(rows),'',dtype='<U1') for a in 'UP' for b in 'UP'}
            for fam in sorted(set(families)):
                train=families!=fam;test=~train
                for source in 'UP':
                    beta=np.linalg.pinv(nuisance[train])@q[source][train] if adjusted else np.zeros((9,q[source].shape[1]))
                    x=q[source]-nuisance@beta
                    scaler=StandardScaler().fit(x[train])
                    clf=LogisticRegression(C=1,max_iter=10000,tol=1e-7,class_weight='balanced',
                        solver='lbfgs',random_state=20260927)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        clf.fit(scaler.transform(x[train]),labels[train])
                    for target in 'UP':
                        preds[source+'->'+target][test]=clf.predict(scaler.transform((q[target]-nuisance@beta)[test]))
            identities[label+f'_adjusted{int(adjusted)}']={}
            for transfer, pred in preds.items():
                identities[label+f'_adjusted{int(adjusted)}'][transfer]=dict(
                    pooled_balanced_accuracy=float(balanced_accuracy_score(labels,pred)),
                    labels=labels.tolist(),predictions=pred.tolist(),
                    training_only_adjustment=adjusted,
                    note='Post-hoc nuisance diagnostic; no semantic validity or causal inference.')
    details=json.loads((root/'analysis-output/formal/s1/S1_DETAILS.json').read_text())
    scales={r['trace_id']:dict(same=r['same_prompt_fits']['affine'],family=r['family_out_fits']['affine']) for r in details}
    splits=json.loads((root/'analysis-output/formal/s1/token_splits.json').read_text())
    nonoption=[]
    for row in details:
        i=idx[row['trace_id']]
        x=np.load(run/f'{i:04d}_uncapped.npz')['logits'].astype(float)
        z=np.load(run/f'{i:04d}_full_p25.npz')['logits'].astype(float)
        fit=row['family_out_fits']['affine'];a,b=fit['slope'],fit['intercept']
        nonoption.append(dict(trace_id=row['trace_id'],source=row['trace_source'],family=row['family_id'],
            errors={name:s1.err(z[s['test']],a*x[s['test']]+b) for name,s in splits[row['trace_id']].items()}))
    save(out/'prepared/scales.json',scales)
    save(out/'explore_outputs/prerequisite_diagnostics.json',dict(
        status='completed_posthoc_existing_state_diagnostics',p1_score_reconstruction_max_errors=errors,
        prompt_duplicate_max_error=source_duplicate,p1_identity_endpoints=identities,
        capped_source_answer_boundary=capped,bridge=bridge,s1_family_out_nonoption=nonoption))
    save(out/'prepared/directions.json',spec)
    fingerprints={str(p.relative_to(out)):sha(p) for p in [out/'PROTOCOL.md',*sorted((out/'scripts').glob('*.py'))]}
    prep=dict(status='ready',original_manifest=original,p1_reconstruction=errors,
        original_array_sha256=checked,directions_sha256=sha(out/'prepared/directions.json'),
        scales_sha256=sha(out/'prepared/scales.json'),files=fingerprints,
        records_sha256=sha(run/'results.json'),unit='12 structural families; not forwards')
    save(out/'prepared/manifest.json',prep)
    print(json.dumps(dict(status='prepared',directions=len(spec),p1_score_errors=errors,
        cached_arrays_checked=len(checked),prompt_duplicate_max_error=source_duplicate)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)

