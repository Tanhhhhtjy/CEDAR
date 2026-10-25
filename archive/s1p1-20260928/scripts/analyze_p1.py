"""Bidirectional family-held-out validity probes, with training-only nuisance fits."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
SEED=20260927


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def design(rows):
    # Initial and induced target identity are shared by each V/F pair.
    return np.array([[1.]+[float(r['initial_answer']==k) for k in 'ABCD']+
                     [float(r['target_answer']==k) for k in 'ABCD'] for r in rows])


def normalize(x, representation):
    if representation=='unit_rms':
        return x/np.maximum(np.sqrt(np.mean(x*x,axis=1,keepdims=True)),1e-12)
    return x.copy()


def evaluate(features,rows,y,representation,nuisance,export=None):
    families=np.array([r['family_id'] for r in rows])
    nuisance_design=design(rows)
    n=len(rows)
    scores={a+'->'+b:np.full(n,np.nan) for a in ['U','P'] for b in ['U','P']}
    for family in sorted(set(families)):
        train=families!=family; test=~train
        for source in ['U','P']:
            x=normalize(features[source],representation)
            beta=np.linalg.pinv(nuisance_design[train])@x[train] if nuisance else np.zeros((9,x.shape[1]))
            adjusted=x-nuisance_design@beta
            scaler=StandardScaler().fit(adjusted[train])
            model=LogisticRegression(C=1.,solver='liblinear',dual=True,max_iter=10000,
                tol=1e-7,class_weight='balanced',random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(scaler.transform(adjusted[train]),y[train])
            for target in ['U','P']:
                xt=normalize(features[target],representation)-nuisance_design@beta
                scores[source+'->'+target][test]=model.decision_function(scaler.transform(xt[test]))
            if export is not None and representation=='raw' and nuisance:
                direction=model.coef_[0]/scaler.scale_
                direction/=np.linalg.norm(direction)
                name=hashlib.sha256((family+source).encode()).hexdigest()[:16]
                np.save(export/(name+'.npy'),direction)
                save(export/(name+'.json'),dict(heldout_family=family,trained_condition=source,
                    training_families=sorted(set(families[train])),label='V=1,F=0',
                    fitted_initial_and_target_identity_nuisance=True,feature_space='raw residual activation',
                    note='Frozen candidate direction. Predictive decoding alone does not demonstrate causal use.'))
    result={}
    for key,z in scores.items():
        assert np.isfinite(z).all()
        auc={f:float(roc_auc_score(y[families==f],z[families==f])) for f in sorted(set(families))}
        balanced={f:float(balanced_accuracy_score(y[families==f],z[families==f]>0)) for f in sorted(set(families))}
        vals=np.array(list(auc.values()))
        rng=np.random.default_rng(SEED)
        bs=vals[rng.integers(len(vals),size=(20000,len(vals)))].mean(1)
        result[key]=dict(equal_family_auc=float(vals.mean()),family_auc=auc,
            descriptive_family_bootstrap_ci95=np.quantile(bs,[.025,.975]).tolist(),
            equal_family_balanced_accuracy=float(np.mean(list(balanced.values()))),family_balanced_accuracy=balanced,
            pooled_oof_auc_diagnostic=float(roc_auc_score(y,z)),scores=z.tolist())
    return result


def main():
    run=ROOT/'runs/formal'
    assert json.loads((run/'status.json').read_text())['stage']=='complete'
    out=ROOT/'analysis-output/formal/p1';out.mkdir(parents=True,exist_ok=True)
    directions=out/'directions';directions.mkdir(exist_ok=True)
    records=json.loads((run/'results.json').read_text())
    traces=list(dict.fromkeys(r['trace_id'] for r in records))
    index={t:i for i,t in enumerate(traces)}
    rows=[r for r in records if r['trace_source']=='e2a_uncapped' and r['arm'] in ['V','F'] and r['condition']=='uncapped']
    assert len(rows)==48 and len(set(r['family_id'] for r in rows))==12
    basegroups=defaultdict(list)
    for i,r in enumerate(rows):basegroups[r['base_id']].append(i)
    assert all(len(ids)==2 and {rows[i]['arm'] for i in ids}=={'V','F'} for ids in basegroups.values())
    y=np.array([r['arm']=='V' for r in rows],dtype=int)
    results={};primary_features=None
    for position in ['prompt','answer']:
        for layer in [53,60,63]:
            features={c:np.array([np.load(run/f"{index[r['trace_id']]:04d}_{condition}.npz")[f'{position}_layer_{layer}'] for r in rows],dtype=float)
                for c,condition in [('U','uncapped'),('P','full_p25')]}
            if position=='prompt' and layer==53:primary_features=features
            for representation in ['raw','unit_rms']:
                for nuisance in [False,True]:
                    key=f'{position}_L{layer}_{representation}_nuisance{int(nuisance)}'
                    results[key]=evaluate(features,rows,y,representation,nuisance,
                        directions if position=='prompt' and layer==53 else None)
    primary='prompt_L53_unit_rms_nuisance1'
    # Permute labels by swapping V/F within complete bases, for both conditions.
    # Retrain every outer-fold probe for each permutation.
    rng=np.random.default_rng(SEED);null={k:[] for k in results[primary]}
    for iteration in range(100):
        yp=y.copy()
        for ids in basegroups.values():
            if rng.integers(2):yp[ids]=yp[ids[::-1]]
        perm=evaluate(primary_features,rows,yp,'unit_rms',True)
        for key,r in perm.items():null[key].append(r['equal_family_auc'])
        if iteration%10==0:print('paired_label_permutations',iteration,flush=True)
    pvalues={}
    for key,scores in null.items():
        observed=results[primary][key]['equal_family_auc']
        pvalues[key]=(1+sum(z>=observed for z in scores))/(1+len(scores))
        results[primary][key]['paired_label_permutation_p']=pvalues[key]
    last=0.
    for i,(key,p) in enumerate(sorted(pvalues.items(),key=lambda v:v[1])):
        last=max(last,min(1.,(4-i)*p));results[primary][key]['holm_p_four_transfer_directions']=last
    save(out/'P1_SUMMARY.json',dict(status='completed_exploratory_probes',n=48,families=12,bases=24,
        primary=primary,results=results,paired_label_permutation_null=null,
        limitations=['Four items per held-out family make family AUC coarse.',
        'Outer-fold training sets overlap; family-bootstrap intervals over fold AUC are descriptive, not an exact sampling distribution.',
        'Permutation p-values have resolution 1/101 and assume paired-label exchangeability.',
        'Decodable V/F labels may reflect prompt features; transfer does not prove the same semantic validity representation.',
        'Answer-boundary features may encode the forthcoming answer. Prompt-boundary results are primary.',
        'All layers, representations and nuisance variants except the prespecified primary are exploratory sensitivity analyses.']))
    save(out/'examples.json',[{k:r[k] for k in ['source_id','family_id','base_id','arm','initial_answer','target_answer']} for r in rows])
    lines=['# P1 family-held-out transfer probes','','Primary: prompt boundary, layer 53, unit-RMS activations, initial/target identity nuisance fitted on training families only.','',
        '| Transfer | Equal-family AUC | Descriptive 95% CI | Balanced accuracy | Paired-label permutation p |',
        '|---|---:|---|---:|---:|']
    for key,r in results[primary].items():lines.append(f"| {key} | {r['equal_family_auc']:.4f} | {r['descriptive_family_bootstrap_ci95']} | {r['equal_family_balanced_accuracy']:.4f} | {r['paired_label_permutation_p']:.4f} |")
    lines+=['','Predictive decoding is not causal evidence. Specific causal patches and independent new structures remain separate experiments.']
    (out/'analysis-report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
