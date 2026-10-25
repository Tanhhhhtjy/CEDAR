"""CPU-only H1 planning from old data. Produces no new-model observations."""
import argparse,json,hashlib,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

def main(root,out):
    src=root/'analysis-output/formal/s1/S1_SELECTIVITY.json'
    rows=[r for r in json.loads(src.read_text()) if r['trace_source']=='e2a_uncapped']
    assert len(rows)==24
    cardfile=root/'CEDAR_SUPPORTING_DATA/data/evidence_certified_pool/family_cards24.jsonl'
    cards={r['family_id']:r for r in map(json.loads,cardfile.read_text().splitlines())}
    families=sorted({r['family_id'] for r in rows});assert len(families)==12
    groups=defaultdict(list)
    for fam in families:groups[cards[fam]['isomorphism_group']].append(fam)
    inventory={g:dict(family_ids=f,signature=cards[f[0]]['structure_signature']) for g,f in groups.items()}
    curves=[];sizes=[];observed={}
    for metric in ['broad','specific']:
        values=defaultdict(list)
        for r in rows:values[r['family_id']].append(r['restored_delta_'+metric])
        fm={f:float(np.mean(v)) for f,v in values.items()};sd=float(np.std(list(fm.values()),ddof=1))
        gm={g:float(np.mean([fm[f] for f in fams])) for g,fams in groups.items()}
        observed[metric]=dict(family_means=fm,family_sd=sd,archetype_means=gm,
            archetype_sd=float(np.std(list(gm.values()),ddof=1)))
        for inflate in [1.,1.5,2.]:
            for n in [12,24,36,48,72,96,144,192]:
                curves.append(dict(metric=metric,sd_multiplier=inflate,independent_groups=n,
                    nominal95_halfwidth=float(t.ppf(.975,n-1)*sd*inflate/math.sqrt(n)),
                    four_endpoint_bonferroni95_halfwidth=float(t.ppf(1-.05/8,n-1)*sd*inflate/math.sqrt(n))))
            for halfwidth in [.1,.2,.3,.5]:
                for comparisons in [1,4]:
                    n=next((n for n in range(6,10001) if t.ppf(1-.05/(2*comparisons),n-1)*sd*inflate/math.sqrt(n)<=halfwidth),None)
                    sizes.append(dict(metric=metric,sd_multiplier=inflate,target_halfwidth=halfwidth,
                        comparisons=comparisons,approx_groups_required=n))
    result=dict(status='planning_only_no_H1_model_runs',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        cards_sha256=hashlib.sha256(cardfile.read_bytes()).hexdigest(),old_families=12,
        old_archetypes=len(groups),inventory=inventory,observed=observed,precision_curves=curves,required_groups=sizes,
        caveats=['Old family IDs can share archetypes; do not count renamed instances as independent new structures.',
        'Student-t planning approximation assumes independent exchangeable group means; it is not achieved coverage or power.',
        'Pilot variance comes from reused development data and may underestimate new-structure heterogeneity.',
        'S1 variance cannot determine P1 AUC precision or C1 interaction power.',
        'Target halfwidths and variance multipliers are sensitivity scenarios, not post-hoc success thresholds.',
        'No H1 sample count or scientific minimum effect selected from model results.'])
    out.mkdir(parents=True,exist_ok=True);(out/'precision_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# H1 精度规划（仅 CPU，未运行 H1 模型）','',
        f'旧主分析包含 12 个 family ID，对应 {len(groups)} 个已声明的结构类别。不能将其写成 12 个互不相关的推理原型。',
        '', '以下是独立新结构组均值的近似 95% 区间半宽，未作多重校正；只用于预算敏感性。', '',
        '| 指标 | 新结构组数 | 旧方差估计 | 标准差扩大1.5倍 | 标准差扩大2倍 |',
        '|---|---:|---:|---:|---:|']
    for metric in ['broad','specific']:
        for n in [24,48,96]:
            vals=[next(r['nominal95_halfwidth'] for r in curves if r['metric']==metric and r['independent_groups']==n and r['sd_multiplier']==x) for x in [1.,1.5,2.]]
            lines.append(f'| {metric} | {n} | '+ ' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','这些数字不是保证，也不决定样本数；新结构类别数、最小有意义效果、指标数量及资源需在 H1 输出前固定。P1/C1 应另做相应精度设计。']
    (out/'precision_plan.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],old_archetypes=len(groups),family_sds={m:v['family_sd'] for m,v in observed.items()})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
