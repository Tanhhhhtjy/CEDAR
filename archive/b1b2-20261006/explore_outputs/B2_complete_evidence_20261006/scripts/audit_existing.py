"""CPU-only reuse analysis. No target-model inference or trajectory selection."""
import argparse,sys,re,csv,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from common import *
def main(root,out):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    verify(root,out);assets=read(root/'explore_outputs/assets_status.json');model=Path(assets['model'])
    index=read(model/'model.safetensors.index.json')
    with safe_open(model/index['weight_map']['lm_head.weight'],framework='pt',device='cpu') as f:head=f.get_slice('lm_head.weight')[32:36].float().numpy()
    weight=np.load(root/'runs/formal/norm_weight.npy').astype(np.float32);eps=read(root/'runs/formal/runtime.json')['norm_epsilon']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    traces=readl(root/'inputs/frozen_traces192.jsonl');ts={t['trace_id']:t for t in traces}
    records=read(root/'runs/formal/results.json');order=list(dict.fromkeys(r['trace_id'] for r in records));old=out/'analysis/A';old.mkdir(parents=True,exist_ok=True)
    decomposed=[];geo=[];residual=[]
    axis=torch.load(assets['assets']['assistant_axis.pt']['path'],map_location='cpu',weights_only=True)[53].float().numpy().astype(float);axis/=np.linalg.norm(axis)
    dirs=read(root/'c1_signed_20260928/prepared/directions.json')
    sys.path.insert(0,str(root/'scripts'));import run_arr_acquisition as acq
    caps,_=acq.historical.e2run.load_p25_assets(root/'cache/hub');cap=caps[53].direction.float().cpu().numpy();cap=cap/np.linalg.norm(cap)
    direction_sign=float(axis@cap);assert direction_sign<-.999
    for i,tid in enumerate(order):
        t=ts[tid];assert [t['candidate_token_ids'][k] for k in 'ABCD']==list(range(32,36))
        with np.load(root/f'runs/formal/{i:04d}_uncapped.npz') as z:u=dict(z)
        with np.load(root/f'runs/formal/{i:04d}_full_p25.npz') as z:p=dict(z)
        hu=u['answer_prenorm'].astype(np.float32);hp=p['answer_prenorm'].astype(np.float32)
        ru=float(np.sqrt(np.mean(hu*hu,dtype=np.float32)+eps));rp=float(np.sqrt(np.mean(hp*hp,dtype=np.float32)+eps))
        nu=(head@(weight*hu)).astype(float);np_=(head@(weight*hp)).astype(float)
        cu=(np_-nu)*(1/rp+1/ru)/2;cd=(np_+nu)*(1/rp-1/ru)/2
        delta=np_/rp-nu/ru;assert np.max(np.abs(cu+cd-delta))<1e-10
        raw=p['logits'][32:36].astype(float)-u['logits'][32:36].astype(float)
        reconstruction=max(float(np.max(np.abs(nu/ru-u['logits'][32:36]))),float(np.max(np.abs(np_/rp-p['logits'][32:36]))));residual.append(reconstruction)
        cf=read(root/f'runs/formal/{i:04d}_norm_counterfactual.json')
        with np.load(root/f'runs/formal/{i:04d}_norm_counterfactual.npz') as cfz:
            cf_options={k:cfz[k][32:36].tolist() for k in cfz.files}
        row={k:t[k] for k in ['trace_id','source_condition','family_id','structure_class','base_id','arm','initial_answer','target_answer']}
        row.update(rms_u=ru,rms_p=rp,old_rms_u=cf['rms_u'],old_rms_p=cf['rms_p'],option_raw_delta=raw.tolist(),option_numerator=cu.tolist(),option_denominator=cd.tolist(),option_rounding_residual=(raw-delta).tolist(),option_reconstruction_max_abs=reconstruction,existing_denominator_swaps=cf_options)
        decomposed.append(row)
        if t['source_condition']=='uncapped':
            dmeta=dirs[t['family_id']+'_U'];path=root/'c1_signed_20260928'/dmeta['file'];assert sha(path)==dmeta['sha256']
            with np.load(path) as f:d=f['direction'].astype(float)
            shift=p['prompt_layer_53'].astype(float)-u['prompt_layer_53'].astype(float)
            geo.append(dict(trace_id=tid,family_id=t['family_id'],structure_class=t['structure_class'],axis_vf_cosine=float(axis@d/np.linalg.norm(d)),raw_shift_axis_projection=float(shift@axis),raw_shift_vf_projection=float(shift@d),training_families=dmeta['training_families'],coordinate='raw residual after L53 cap; direction exported coef/scaler then L2 normalized',caveat='Original family-heldout fits may share a structure with test; geometry is descriptive, not structure-heldout validation'))
    save(old/'readout_decomposition.json',decomposed);save(old/'axis_geometry.json',geo)
    save(old/'reconstruction_audit.json',dict(status='completed_arithmetic_audit',traces=len(decomposed),rms_epsilon=eps,head_sha256=sha(model/index['weight_map']['lm_head.weight']),official_cap_cosine_to_human_axis=direction_sign,fp32_option_reconstruction_max_abs=max(residual),limitations=['FP32 offline arithmetic does not recover information lost in BF16','Symmetric terms are not causal mediation shares','Historical denominator swaps reused without inference']))
    annotations=[]
    pattern=re.compile(r'(?i)(?:the\s+(?:final\s+)?answer\s+(?:is|should be|would be)|(?:choose|select)\s+(?:option\s+)?|(?:correct|final)\s+(?:answer|option)\s*(?:is|:))\s*["\x27*`]*\s*([ABCD])\b')
    for t in traces:
        ids=t['trace_prefix_token_ids'];text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        matches=[]
        for m in pattern.finditer(text):
            lo,hi=0,len(ids)
            while lo<hi:
                mid=(lo+hi)//2
                if len(tokenizer.decode(ids[:mid],skip_special_tokens=False,clean_up_tokenization_spaces=False))<=m.start():lo=mid+1
                else:hi=mid
            excerpt=text[max(0,m.start()-100):min(len(text),m.end()+120)]
            matches.append(dict(prefix_token=max(0,lo-1),character=m.start(),candidate_label=m.group(1).upper(),excerpt=excerpt,excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),semantic_status='candidate_requires_review_not_proven_commitment'))
        annotations.append(dict(trace_id=t['trace_id'],source_condition=t['source_condition'],prompt_end_token=t['prompt_tokens']-1,reasoning_start_token=t['prompt_tokens'],terminal_input_token=t['prompt_tokens']+len(ids)-1,final_label_prediction_index=t['prompt_tokens']+len(ids),prefix_tokens=len(ids),prefix_hash=t['trace_prefix_token_ids_sha256'],decoded_sha256=hashlib.sha256(text.encode()).hexdigest(),reasoning_close_character=text.find('</think>'),candidate_commitments=matches,first_confirmed_commitment=None,review_status='manual_semantic_review_pending',used_for_sample_selection=False))
    (old/'trace_boundaries.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in annotations),encoding='utf-8')
    save(old/'annotation_summary.json',dict(traces=192,uncapped_source=96,with_lexical_candidates=sum(bool(a['candidate_commitments']) for a in annotations),confirmed_commitments=None,manual_review_pending=192,rule='Candidates only; quoted, hypothetical, negated, enumerated and changing conclusions require contextual review. No B sample exclusions.'))
    # H preparation: truthful dependency registry and precision sensitivity, no invented certified structures.
    cards=readl(root/'CEDAR_SUPPORTING_DATA/data/evidence_certified_pool/family_cards24.jsonl')
    registry=[]
    for c in cards:
        registry.append(dict(status='historical_not_new',prototype_id=c.get('structure_class',c['isomorphism_group']),dependency_group=c['isomorphism_group'],family_id=c['family_id'],bases=sorted({t['base_id'] for t in traces if t['family_id']==c['family_id']}),source_sha256=sha(root/'CEDAR_SUPPORTING_DATA/data/evidence_certified_pool/family_cards24.jsonl')))
    (out/'h1/structure_registry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in registry),encoding='utf-8')
    s1=read(root/'analysis-output/formal/s1/S1_DETAILS.json');bybase=defaultdict(dict)
    for r in s1:
        if r['source_condition']=='uncapped':bybase[r['base_id']][r['arm']]=r
    mapping={c['family_id']:c['isomorphism_group'] for c in cards};var={}
    for name in ['raw_delta_broad','restored_delta_specific']:
        fam=defaultdict(list)
        for b,arms in bybase.items():
            value=np.mean([arms[a][name] for a in 'NRVF']) if name=='raw_delta_broad' else arms['V'][name]-arms['F'][name]
            fam[arms['V']['family_id']].append(float(value))
        groups=defaultdict(list)
        for f,vals in fam.items():groups[mapping[f]].append(float(np.mean(vals)))
        fm={f:float(np.mean(v)) for f,v in fam.items()};gm={g:float(np.mean(v)) for g,v in groups.items()}
        var[name]=dict(family_means=fm,structure_means=gm,family_sd=float(np.std(list(fm.values()),ddof=1)),structure_sd=float(np.std(list(gm.values()),ddof=1)))
    from scipy.stats import t as student
    curves=[]
    for metric,v in var.items():
        for n in [8,12,24,48,64,96]:
            for mult in [1.,1.5,2.]:curves.append(dict(metric=metric,independent_groups=n,sd_multiplier=mult,bonferroni_three_halfwidth=float(student.ppf(1-.05/6,n-1)*v['structure_sd']*mult/math.sqrt(n))))
    rng=np.random.default_rng(20261005);behavior=[]
    for n in [8,12,24,48,96]:
        for rho in [0.,.25,.5]:
            # Illustrative null model: two bases/group, four paired treatment/arm outcomes with shared group propensity.
            draws=4000
            propensity=np.full((draws,n,1,1),.5) if rho==0 else rng.beta(.5*(1/rho-1),.5*(1/rho-1),size=(draws,n,1,1))
            z=rng.binomial(1,propensity,size=(draws,n,2,4));g=(z[:,:,:,3]-z[:,:,:,2]-z[:,:,:,1]+z[:,:,:,0]).mean(2)
            half=student.ppf(1-.05/6,n-1)*g.std(1,ddof=1)/math.sqrt(n)
            behavior.append(dict(groups=n,bases_per_group=2,illustrative_shared_propensity_correlation=rho,median_bonferroni_halfwidth=float(np.median(half)),model='illustrative binary joint model, no measured behavior variance or power guarantee'))
    save(out/'h1/precision_plan.json',dict(observed_development=var,continuous_scenarios=curves,behavior_scenarios=behavior,status='planning_not_sample_size_approval',new_certified_groups=0))
    (out/'h1/precision_plan.md').write_text('# H preparation\n\nHistorical registry records all24 families /8 declared graphs; only12 families /24 bases were used in S1/C1. No new certified structure yet. Prototype expansion is not independence.\n\nPrecision uses8-group and12-family variance separately; multiplicity provision is Bonferroni3 planning, with1/1.5/2 SD scenarios. Behavior receives separate binary joint simulations, whose assumptions are illustrative. Final sample size requires a scientifically meaningful interval/effect target and certified dependency groups.\n\nNatural behavior must retain every scheduled prompt, including invalid/truncated outputs; target-choice=1 only with an unambiguous parsed target, otherwise0 with invalid/truncated rate separately reported. Fixed replay eligibility and missing-boundary rates must be reported by arm and cap, without replacing the full behavioral denominator. H GPU protocol remains pending.\n',encoding='utf-8')
    save(old/'status.json',dict(stage='offline_analysis_complete_annotations_need_review',time=time.time(),readout_pairs=192,geometry_rows=96,annotated_traces=192,new_model_forwards=0,H_new_certified_groups=0))
    print('A offline arithmetic/geometry/annotation candidates and H dependency/precision preparation completed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
