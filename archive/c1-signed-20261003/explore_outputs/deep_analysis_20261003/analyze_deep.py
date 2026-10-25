"""Post-completion descriptive C1 diagnostics; frozen primary tests stay unchanged."""
from pathlib import Path
import json,hashlib,datetime,itertools,csv
from collections import defaultdict
import numpy as np
R=Path('D:/CEDAR/arr_c1_signed_20260928')
B=R/'explore_outputs/closure_20261003/evidence'
O=R/'explore_outputs/deep_analysis_20261003'
O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=['analysis/formal/SUMMARY.json','analysis/formal/details.json','analysis/formal/base_endpoints.json','runs/formal/results.json','prepared/scales.json','prepared/directions.json','PROTOCOL.md']
manifest={x['path']:x for x in read(B/'bundle_manifest.json')}
assert all(sha(B/p)==manifest[p]['sha256'] for p in inputs)
s=read(B/inputs[0]);ds=read(B/inputs[1]);eps=read(B/inputs[2]);rs=read(B/inputs[3]);scales=read(B/inputs[4])
assert len(rs)==len(ds)==3840
F=sorted({r['family_id'] for r in rs})
idx=np.random.default_rng(20260928).integers(12,size=(20000,12))
def describe(x):
 x=np.array(x,float)
 return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,min=float(x.min()),q05=float(np.quantile(x,.05)),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),q95=float(np.quantile(x,.95)),max=float(x.max()),zero_count=int((x==0).sum()))
def famstat(pairs):
 g=defaultdict(list)
 for f,v in pairs:g[f].append(float(v))
 assert sorted(g)==F
 x=np.array([np.mean(g[f]) for f in F]);boot=x[idx].mean(1)
 return dict(mean=float(x.mean()),family_sd=float(x.std(ddof=1)),ci95=np.quantile(boot,[.025,.975]).tolist(),family_means=dict(zip(F,x.tolist())),positive=int((x>1e-12).sum()),negative=int((x< -1e-12).sum()),zero=int((np.abs(x)<=1e-12).sum()),leave_one_family_out=[float(np.delete(x,i).mean()) for i in range(12)])
def holm(ps):
 out={};last=0.
 for i,(k,p) in enumerate(sorted(ps.items(),key=lambda x:x[1])):
  last=max(last,min(1.,(len(ps)-i)*p));out[k]=last
 return out
def kind(r):
 c=r['cell']
 return 'true' if c['operation']=='dose' and c['random_index']==-1 else 'random' if c['operation']=='dose' else c['operation']
def key(r):
 c=r['cell'];return r['trace_id'],c['target'],c['operation'],c['dose'],c['random_index']
dl={key(d):d for d in ds};rl={key(r):r for r in rs}
noops={(r['trace_id'],r['cell']['target']):r for r in rs if kind(r)=='noop'}
telemetry={}
for k in ['true','random','replace']:
 vals=[r for r in rs if kind(r)==k]
 telemetry[k]={name:describe([r['patch'][name] for r in vals]) for name in ['realized_norm','realized_over_target_norm','actual_unit_rms_projection_change','cosine_to_validity']}
 if k=='true':
  telemetry[k]['signed_sigma_ratio']=describe([r['patch']['actual_unit_rms_projection_change']/(r['cell']['dose']*r['patch']['sigma_train']) for r in vals])
  telemetry[k]['sign_correct']=sum(r['patch']['actual_unit_rms_projection_change']*r['cell']['dose']>0 for r in vals)
 if k=='random':telemetry[k]['norm_match_error']=describe([r['patch']['random_norm_relative_error'] for r in vals])
# All entries finite before summary
changes={};family_outputs={};scale_stats={}
for k in ['noop','true','random','replace']:
 vals=[r for r in rs if kind(r)==k]
 dz=[np.array([r['logit_diagnostics']['logits'][a]-noops[r['trace_id'],r['cell']['target']]['logit_diagnostics']['logits'][a] for a in 'ABCD']) for r in vals]
 details=[dl[key(r)] for r in vals]
 changes[k]=dict(n=len(vals),option_any_change=sum(bool(np.any(z!=0)) for z in dz),argmax_changes=sum(d['argmax_changed'] for d in details),rank_changes=sum(d['option_order']!=d['baseline_option_order'] for d in details),max_abs_option_shift=describe([float(np.max(np.abs(z))) for z in dz]))
 for corr in ['raw','same']:
  for metric in ['broad','specific']:
   values=[d['metrics'][corr][metric] for d in details]
   changes[k][corr+'_'+metric+'_abs']=describe(np.abs(values))
   family_outputs[k+'_'+corr+'_'+metric]=famstat([(d['family_id'],abs(d['metrics'][corr][metric])) for d in details])
 scale_stats[k]={'slope':describe([d['new_patch_scale_diagnostic']['slope'] for d in details]),
                 'intercept':describe([d['new_patch_scale_diagnostic']['intercept'] for d in details])}
 for stratum in ['top256','rank257_4096','tail']:
  scale_stats[k][stratum+'_heldout_rmse']=describe([d['new_patch_scale_diagnostic']['heldout_nonoption'][stratum]['rmse'] for d in details])
# Primary sign heterogeneity and sensitivity
primary={}
for name,v in s['primary'].items():
 x=np.array([v['family_means'][f] for f in F])
 primary[name]=dict(v,family_sd=float(x.std(ddof=1)),positive=int((x>0).sum()),negative=int((x<0).sum()),standardized_family_effect=float(x.mean()/x.std(ddof=1)),leave_one_family_out=[float(np.delete(x,i).mean()) for i in range(12)])
sens={name:v for name,v in s['secondary'].items() if name.endswith(('common','interaction'))}
secondary16={k:v for k,v in s['secondary'].items() if k.startswith('same_') and k.endswith('true_minus_random')}
holm16=holm({k:v['exact_signflip_p'] for k,v in secondary16.items()})
holm488=holm({k:v['exact_signflip_p'] for k,v in s['secondary'].items()})
multiplicity=dict(secondary_count=len(s['secondary']),nominal_p_lt_05=sum(v['exact_signflip_p']<.05 for v in s['secondary'].values()),exploratory_holm488_discoveries=sum(v<.05 for v in holm488.values()),same_true_minus_random_16={k:dict(v,posthoc_holm16=holm16[k]) for k,v in secondary16.items()},note='Both extended Holm families are post hoc diagnostics only. Frozen primary family remains the original four tests.')
# Paired signed (odd) and sign-averaged (even) response; controls average within prompt, never independent trials.
odd_even={};arm_response={}
pairs_by=defaultdict(list)
for r in rs:
 c=r['cell']
 if c['operation']!='dose' or c['dose']!=1:continue
 m=dl[r['trace_id'],c['target'],'dose',-1.,c['random_index']];p=dl[key(r)]
 vals={}
 for metric in ['broad','specific']:
  vals[metric]={'odd':(p['metrics']['same'][metric]-m['metrics']['same'][metric])/2,
                'even':(p['metrics']['same'][metric]+m['metrics']['same'][metric])/2}
 pairs_by[r['trace_id'],c['target'],kind(r)].append((r['family_id'],r['arm'],vals))
aggregated=[]
for (trace,target,k),li in pairs_by.items():
 item=dict(trace=trace,target=target,kind=k,family=li[0][0],arm=li[0][1])
 for metric in ['broad','specific']:
  for order in ['odd','even']:
   item[metric+'_'+order]=float(np.mean([t[2][metric][order] for t in li]))
   item[metric+'_'+order+'_abs']=float(np.mean([abs(t[2][metric][order]) for t in li]))
 aggregated.append(item)
for k in ['true','random']:
 for target in 'UP':
  for metric in ['broad','specific']:
   for order in ['odd','even']:
    vals=[x for x in aggregated if x['kind']==k and x['target']==target]
    odd_even[k+'_'+target+'_'+metric+'_'+order]=famstat([(x['family'],x[metric+'_'+order]) for x in vals])
    odd_even[k+'_'+target+'_'+metric+'_'+order+'_abs']=famstat([(x['family'],x[metric+'_'+order+'_abs']) for x in vals])
 for target in 'UP':
  for arm in 'NRVF':
   vals=[x for x in aggregated if x['kind']==k and x['target']==target and x['arm']==arm]
   for metric in ['broad','specific']:
    arm_response[k+'_'+target+'_'+arm+'_'+metric]=famstat([(x['family'],x[metric+'_odd']) for x in vals])
# Replacement is a local last-prompt-token full-vector swap, not whole state/cache or mediated proportion.
replacement={}
for target in 'UP':
 vals=[d for d in ds if kind(d)=='replace' and d['cell']['target']==target]
 for metric in ['broad','specific']:
  replacement[target+'_'+metric]=famstat([(d['family_id'],d['metrics']['same'][metric]) for d in vals])
  for arm in 'NRVF':
   replacement[target+'_'+arm+'_'+metric]=famstat([(d['family_id'],d['metrics']['same'][metric]) for d in vals if d['arm']==arm])
# Donor-directed distance change in raw option-vector and margins; descriptive, no rescue ratio claims.
donor_motion={}
for metric in ['broad','specific']:
 for target in 'UP':
  points=[]
  for r in rs:
   if kind(r)!='replace' or r['cell']['target']!=target:continue
   own=noops[r['trace_id'],target];other=noops[r['trace_id'],'P' if target=='U' else 'U']
   def margin(v):
    z=v['logit_diagnostics']['logits'];a=v['initial_answer'];t=v['target_answer']
    return sum(z[x] for x in 'ABCD' if x!=a)/3-z[a] if metric=='broad' else z[t]-sum(z[x] for x in 'ABCD' if x not in (a,t))/2
   before=margin(own);after=margin(r);donor=margin(other)
   points.append((r['family_id'],after-before,donor-before,abs(after-donor)<abs(before-donor)))
  donor_motion[target+'_'+metric]=dict(n=len(points),same_direction_as_donor=sum(a*b>0 for f,a,b,c in points),closer_to_donor=sum(c for f,a,b,c in points),unchanged=sum(a==0 for f,a,b,c in points),mean_shift=famstat([(f,a) for f,a,b,c in points]),mean_full_condition_difference=famstat([(f,b) for f,a,b,c in points]))
# Raw capping gap in this exact metric and sample, as a different estimand, not causal mediation denominator.
baseline={}
for metric in ['broad','specific']:
 for arm in ['ALL','N','R','V','F']:
  vals=[]
  for (trace,target),u in noops.items():
   if target!='U' or (arm!='ALL' and u['arm']!=arm):continue
   p=noops[trace,'P'];a=u['initial_answer'];t=u['target_answer']
   def marg(r):
    z=r['logit_diagnostics']['logits']
    return sum(z[k] for k in 'ABCD' if k!=a)/3-z[a] if metric=='broad' else z[t]-sum(z[k] for k in 'ABCD' if k not in (a,t))/2
   vals.append((u['family_id'],marg(p)-marg(u)))
  baseline[metric+'_'+arm]=famstat(vals)
# Pairwise true-versus-random absolute response controls (descriptive effect, no new significance testing).
control_abs={}
for metric in ['broad','specific']:
 for order in ['odd','even']:
  g={(x['trace'],x['target'],x['kind']):x for x in aggregated}
  diffs=[]
  for x in aggregated:
   if x['kind']!='true':continue
   y=g[x['trace'],x['target'],'random']
   diffs.append((x['family'],x[metric+'_'+order+'_abs']-y[metric+'_'+order+'_abs']))
  control_abs[metric+'_'+order]=famstat(diffs)
result=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='posthoc_descriptive_diagnostics',
 primary=primary,sensitivity=sens,telemetry=telemetry,output_changes=changes,absolute_response_family=family_outputs,
 odd_even=odd_even,arm_response=arm_response,replacement=replacement,donor_motion=donor_motion,
 full_capping_baseline=baseline,patch_scale=scale_stats,paired_true_minus_random_absolute=control_abs,multiplicity=multiplicity)
save('diagnostics.json',result)
save('provenance.json',dict(inputs={str(B/p):sha(B/p) for p in inputs},script_sha256=sha(Path(__file__)),no_gpu_execution=True,
 policy='Primary results unchanged. Added analyses are explicitly post hoc/descriptive, family-unit, not confirmatory.',figures='Generated from saved logs/metadata and logits; raw high-dimensional NPZ remain on server.'))
with (O/'primary_endpoints.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['endpoint','mean','family_sd','ci_low','ci_high','raw_p','holm_p','positive_families','negative_families'])
 for k,v in primary.items():w.writerow([k,v['equal_family_mean'],v['family_sd'],*v['descriptive_ci95'],v['exact_signflip_p'],v['holm_four_primary'],v['positive'],v['negative']])
with (O/'family_endpoints.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['family',*primary])
 for family in F:w.writerow([family,*[v['family_means'][family] for v in primary.values()]])
print('TELEMETRY',json.dumps({k:{x:v for x,v in t.items() if x in ['realized_over_target_norm','signed_sigma_ratio','norm_match_error','sign_correct']} for k,t in telemetry.items()},ensure_ascii=False))
print('OUTPUTS',json.dumps({k:{x:v for x,v in t.items() if x in ['n','option_any_change','argmax_changes','rank_changes','raw_broad_abs','raw_specific_abs']} for k,t in changes.items()},ensure_ascii=False))
print('REPLACEMENT',json.dumps({k:{'mean':v['mean'],'ci':v['ci95']} for k,v in replacement.items() if len(k.split('_'))==2}))
print('FAMILY_SIGNS',[(k,v['positive'],v['negative'],[min(v['leave_one_family_out']),max(v['leave_one_family_out'])]) for k,v in primary.items()])
print('ODD_EVEN',json.dumps({k:{'mean':v['mean'],'ci':v['ci95']} for k,v in odd_even.items() if not k.endswith('_abs')}))
print('TRUE_MINUS_RANDOM_ABS',json.dumps({k:{'mean':v['mean'],'ci':v['ci95']} for k,v in control_abs.items()}))
print('MULTIPLICITY',{k:v for k,v in multiplicity.items() if k!='same_true_minus_random_16'})

