"""Held-out vocabulary scale prediction with paired family-level uncertainty."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260927


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def token_split(x, excluded, key):
    rng = np.random.default_rng(int(hashlib.sha256((str(SEED)+key).encode()).hexdigest()[:16], 16))
    order = np.argsort(-x, kind='stable')
    strata = {}
    for name, ids in [('top256', order[:256]), ('rank257_4096', order[256:4096]), ('tail', order[4096:])]:
        ids = np.array([t for t in ids if int(t) not in excluded])
        ids = rng.permutation(ids)[:1024]
        n = len(ids)//2
        assert n > 0
        strata[name] = dict(fit=ids[:n].tolist(), test=ids[n:].tolist())
    return strata


def sufficient(x, y, strata):
    values = []
    for s in strata.values():
        a, b = x[s['fit']], y[s['fit']]
        values.append([np.mean(a), np.mean(b), np.mean(a*a), np.mean(a*b)])
    return np.mean(values, axis=0)


def fit(m):
    mx, my, xx, xy = m
    a = (xy-mx*my)/(xx-mx*mx)
    scale = xy/xx
    # Do not hide an invalid temperature interpretation by clipping the slope.
    return dict(identity=(1., 0.), scale=(float(scale), 0.), affine=(float(a), float(my-a*mx)))


def err(observed, predicted):
    diff = observed-predicted
    return dict(mae=float(np.abs(diff).mean()), rmse=float(np.sqrt(np.mean(diff**2))),
                centered_rmse=float(np.sqrt(np.mean((diff-diff.mean())**2))))


def margin(z, initial, target):
    i, t = 'ABCD'.index(initial), 'ABCD'.index(target)
    other = [j for j in range(4) if j not in (i,t)]
    return dict(broad=float(np.delete(z,i).mean()-z[i]), specific=float(z[t]-z[other].mean()))


def summary(rows, metric):
    fam = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            fam[r['family_id']].append(r[metric])
    names = sorted(fam)
    means = np.array([np.mean(fam[k]) for k in names])
    if len(means) == 0:
        return dict(n=0, families=0)
    rng = np.random.default_rng(SEED)
    samples = means[rng.integers(len(means), size=(20000,len(means)))].mean(axis=1)
    return dict(n=sum(len(v) for v in fam.values()), families=len(means), mean=float(means.mean()),
        family_sd=float(means.std(ddof=1)) if len(means)>1 else None,
        ci95=np.quantile(samples,[.025,.975]).tolist(), positive_families=int((means>0).sum()),
        family_means=dict(zip(names,means.tolist())))


def signflip(summary_row):
    vals = np.array(list(summary_row['family_means'].values()))
    null = np.array(list(itertools.product([-1.,1.],repeat=len(vals)))) @ vals / len(vals)
    return float((np.abs(null)>=abs(vals.mean())-1e-12).mean())


def main(scope):
    run = ROOT / 'runs' / scope
    state = json.loads((run/'status.json').read_text())
    assert state['stage']=='complete', 'Incomplete acquisition cannot produce a final S1 report'
    records = json.loads((run/'results.json').read_text())
    runtime = json.loads((run/'runtime.json').read_text())
    out = ROOT / 'analysis-output' / scope / 's1'
    out.mkdir(parents=True,exist_ok=True)
    pairs = defaultdict(dict)
    for r in records:
        pairs[r['trace_id']][r['condition']] = r
    infos = []
    for i, (key, pair) in enumerate(pairs.items()):
        u, p = pair['uncapped'], pair['full_p25']
        x = np.load(run/f'{i:04d}_uncapped.npz')['logits'].astype(float)
        y = np.load(run/f'{i:04d}_full_p25.npz')['logits'].astype(float)
        excluded = set(runtime['excluded_fit_token_ids']) | set(u['candidate_token_ids'].values())
        split = token_split(x,excluded,key)
        assert not (set(itertools.chain.from_iterable(s['fit'] for s in split.values())) & excluded)
        info = dict(index=i, record=u, x=x, y=y, split=split, sufficient=sufficient(x,y,split))
        infos.append(info)
    write(out/'token_splits.json', {v['record']['trace_id']:v['split'] for v in infos})
    details = []
    for v in infos:
        r, x, y = v['record'],v['x'],v['y']
        row = {k:r[k] for k in ['trace_id','source_id','trace_source','source_condition','family_id','base_id','arm','initial_answer','target_answer']}
        option_ids = [r['candidate_token_ids'][k] for k in 'ABCD']
        xu, yp = x[option_ids],y[option_ids]
        row['option_argmax_flip'] = int(xu.argmax()!=yp.argmax())
        row['same_prompt_fits'] = {}
        fits = fit(v['sufficient'])
        for kind,(a,b) in fits.items():
            prediction = dict(slope=a,intercept=b,positive_slope=a>0,
                options=err(yp,a*xu+b),nonoption_test={})
            for name,split in v['split'].items():
                ids = split['test']
                prediction['nonoption_test'][name] = err(y[ids],a*x[ids]+b)
            row['same_prompt_fits'][kind] = prediction
            row[kind+'_option_rmse'] = prediction['options']['rmse']
        others = [t['sufficient'] for t in infos if t['record']['family_id']!=r['family_id'] and t['record']['trace_source']==r['trace_source']]
        row['family_out_fits'] = {}
        if others:
            for kind,(a,b) in fit(np.mean(others,axis=0)).items():
                row['family_out_fits'][kind] = dict(slope=a,intercept=b,positive_slope=a>0,options=err(yp,a*xu+b))
        mu,mp = margin(xu,r['initial_answer'],r['target_answer']), margin(yp,r['initial_answer'],r['target_answer'])
        a,b = fits['affine']
        corrected = margin((yp-b)/a,r['initial_answer'],r['target_answer']) if a>0 else None
        for kind in ['broad','specific']:
            row['raw_delta_'+kind]=mp[kind]-mu[kind]
            row['restored_delta_'+kind]=corrected[kind]-mu[kind] if corrected else None
        row['affine_slope']=a
        norm = json.loads((run/f"{v['index']:04d}_norm_counterfactual.json").read_text())
        row['rms_ratio_p_over_u']=norm['rms_p']/norm['rms_u']
        cf = np.load(run/f"{v['index']:04d}_norm_counterfactual.npz")
        for direction in ['p_with_u_denominator','u_with_p_denominator']:
            row[direction+'_option_rmse_from_u'] = err(xu,cf[direction][option_ids])['rmse']
        hu=np.load(run/f"{v['index']:04d}_uncapped.npz")['answer_prenorm'].astype(float)
        hp=np.load(run/f"{v['index']:04d}_full_p25.npz")['answer_prenorm'].astype(float)
        row['prenorm_cosine']=float(hu@hp/(np.linalg.norm(hu)*np.linalg.norm(hp)))
        details.append(row)
    metrics=['raw_delta_broad','raw_delta_specific','restored_delta_broad','restored_delta_specific',
        'identity_option_rmse','scale_option_rmse','affine_option_rmse','affine_slope','option_argmax_flip',
        'rms_ratio_p_over_u','prenorm_cosine']
    aggregate={}
    selectivity=[]
    for source in sorted(set(r['trace_source'] for r in details)):
        selected=[r for r in details if r['trace_source']==source]
        aggregate[source]={arm:{m:summary([r for r in selected if r['arm']==arm],m) for m in metrics} for arm in 'NRVF'}
        bases=defaultdict(dict)
        for row in selected:
            bases[row['base_id']][row['arm']]=row
        for base,arms in bases.items():
            assert set(arms)==set('NRVF'), 'Unpaired base'
            d=dict(base_id=base,family_id=arms['V']['family_id'],trace_source=source)
            for m in metrics[:4]:
                d[m]=arms['V'][m]-arms['F'][m] if arms['V'][m] is not None and arms['F'][m] is not None else None
            selectivity.append(d)
        aggregate[source]['V_minus_F']={m:summary([r for r in selectivity if r['trace_source']==source],m) for m in metrics[:4]}
    primary=aggregate.get('e2a_uncapped',{}).get('V_minus_F',{})
    if scope=='formal':
        rawp={m:signflip(primary[m]) for m in metrics[:4] if primary[m].get('families')==12}
        running=0.
        for rank,(m,p) in enumerate(sorted(rawp.items(),key=lambda it:it[1])):
            running=max(running,min(1.,(len(rawp)-rank)*p))
            primary[m]['p_exact_signflip']=p
            primary[m]['p_holm_four_selectivity_contrasts']=running
    write(out/'S1_SUMMARY.json',dict(scope=scope,trace_pairs=len(details),aggregate=aggregate,
        status='engineering_smoke_only' if scope=='smoke' else 'completed_exploratory_analysis',
        uncertainty='20,000 paired family bootstrap draws, equal family weight; 12 structural clusters in formal',
        limitations=['Within-prompt fits use paired capped non-option logits; only family-out fits predict without them.',
        'RMS denominator swap is a readout decomposition, not proof of reason-related causal use.',
        'Fixed-trace argmax is not natural generated behavioral choice.',
        'Exact sign-flip tests assume symmetric independent family effects; secondary summaries are descriptive.']))
    write(out/'S1_DETAILS.json',details)
    write(out/'S1_SELECTIVITY.json',selectivity)
    lines=['# S1 held-out scale analysis', '',f'Scope: {scope}; paired traces: {len(details)}.','',
        '| Trace source | V−F contrast | Mean | Family bootstrap 95% CI | Families |',
        '|---|---|---:|---|---:|']
    for source,a in aggregate.items():
        for metric,s in a['V_minus_F'].items():
            if s['n']:
                lines.append(f"| {source} | {metric} | {s['mean']:.6g} | {s['ci95']} | {s['families']} |")
    lines+=['','All outcomes are exploratory. A good scale fit does not establish a mechanism, and a restored margin does not establish a causal validity pathway.',
        'Per-stratum held-out prediction errors and family-out predictions are in S1_DETAILS.json. Historical replay mismatches remain in acquisition records.']
    (out/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    (out/'stats-appendix.md').write_text('Primary unit: structural family. V/F remain paired within base. Four selectivity contrasts use exact family sign flips with Holm correction in the formal set; symmetry is an assumption, not inferred from a normality pretest. Other intervals are descriptive. Seed 20260927; 20,000 bootstrap draws. Smoke has one family and cannot support inference.\n')
    if scope=='formal':
        plot(out,details,selectivity)


def plot(out,rows,selectivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures'
    figdir.mkdir(exist_ok=True)
    source='e2a_uncapped'
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    for n,m in enumerate(['raw_delta_broad','restored_delta_broad','raw_delta_specific','restored_delta_specific']):
        s=summary([r for r in selectivity if r['trace_source']==source],m)
        vals=list(s['family_means'].values())
        ax.scatter(np.full(len(vals),n),vals,s=20,alpha=.65)
        ax.plot(n,s['mean'],'ks',ms=6)
    ax.axhline(0,color='gray',lw=.8)
    ax.set_xticks(range(4),['Raw broad','Restored broad','Raw specific','Restored specific'])
    ax.set_ylabel('V−F difference in paired logit change')
    fig.tight_layout();fig.savefig(figdir/'selectivity.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.6))
    selected=[r for r in rows if r['trace_source']==source]
    for arm in 'NRVF':
        vals=[r for r in selected if r['arm']==arm]
        ax.scatter([r['identity_option_rmse'] for r in vals],[r['affine_option_rmse'] for r in vals],label=arm,s=20,alpha=.7)
    ax.set_xlabel('Held-out option RMSE: identity')
    ax.set_ylabel('Held-out option RMSE: non-option affine fit')
    ax.legend();fig.tight_layout();fig.savefig(figdir/'heldout_prediction.pdf');plt.close(fig)
    (out/'figure-catalog.md').write_text('selectivity.pdf: each point is one family; squares are equal-family means. Inspect whether scale-restored selectivity remains heterogeneous. No token-level uncertainty is implied.\n\nheldout_prediction.pdf: paired option prediction errors by arm. Options are excluded from fitting. Improved prediction supports a descriptive scale account; within-prompt prediction is not unseen-family prediction.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['smoke','formal'],required=True)
    main(p.parse_args().scope)
