"""Independent, CPU-only B1 closure audit and descriptive reporting."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
p=args.root;dest=Path(__file__).resolve().parent;fig=dest/'figures';fig.mkdir(exist_ok=True)
def read(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def write(n,s):(dest/n).write_text(s,encoding='utf-8')
summary=read(p/'analysis/B1/SUMMARY.json');summaries=summary['summaries']
scales=read(p/'analysis/B1/frozen_scales.json');details=read(p/'analysis/B1/details.json')
endpoints=read(p/'analysis/B1/base_endpoints.json');records=read(p/'runs/B1/results.json')
integrity=read(p/'explore_outputs/closure_20261006/closure_integrity.json')
assert summary['status']=='descriptive_interim_no_primary_tests'
assert len(records)==192 and len(details)==48 and len(endpoints)==48 and summary['bases']==12
assert all('primary_test' not in x for x in summaries.values())
bydetail={r['trace_id']:r for r in details};phases=['00','10','01','11']
raw={};nonpositive=[];slope_by_phase=defaultdict(list)
for r in records:
    tid=r['trace_id'];phase=r['cell']['phase'];a=p/'runs/B1'/r['array_file']
    assert sha(a)==r['arrays_sha256']
    with np.load(a) as z:logits=z['logits'].astype(float)
    ids=[r['candidate_token_ids'][k] for k in 'ABCD'];v=logits[ids]
    i='ABCD'.index(r['initial_answer']);t='ABCD'.index(r['target_answer']);assert i!=t
    def margin(v):return {'broad':float(np.mean(np.delete(v,i))-v[i]),'specific':float(v[t]-np.mean([v[k] for k in range(4) if k not in (i,t)]))}
    m=margin(v);raw[(tid,phase)]=m
    for k in m:assert abs(m[k]-bydetail[tid]['raw'][phase][k])<1e-10
    s=scales[tid][phase];slope_by_phase[phase].append(s['slope'])
    if s['slope']<=0:nonpositive.append((tid,phase));assert bydetail[tid]['corrected'][phase] is None
    else:
        cm=margin((v-s['intercept'])/s['slope'])
        for k in cm:assert abs(cm[k]-bydetail[tid]['corrected'][phase][k])<1e-10
assert not nonpositive
bases=defaultdict(dict)
for r in details:bases[r['base_id']][r['arm']]=r
contrasts=['total','EH','ET','I','ET_given_H','EH_given_T']
for r in endpoints:
    arms=bases[r['base_id']];scale=r['scale'];metric=r['metric'];assert set(arms)==set('NRVF')
    y={c:np.mean([arms[a][scale][c][metric] for a in 'NRVF']) if metric=='broad' else arms['V'][scale][c][metric]-arms['F'][scale][c][metric] for c in phases}
    v=[y[c] for c in phases];z=dict(zip(contrasts,[v[3]-v[0],v[1]-v[0],v[2]-v[0],v[3]-v[1]-v[2]+v[0],v[3]-v[1],v[3]-v[2]]))
    for c in phases:assert abs(y[c]-r['values'][c])<1e-10
    for c in contrasts:assert abs(z[c]-r['contrasts'][c])<1e-10
    assert abs(z['total']-z['EH']-z['ET']-z['I'])<1e-10
max_summary_error=0.
for scale in ['raw','corrected']:
  for metric in ['broad','specific']:
    group=[x for x in endpoints if x['scale']==scale and x['metric']==metric]
    for c in contrasts:
      family=defaultdict(list);sg={}
      for r in group:family[r['family_id']].append(r['contrasts'][c]);sg[r['family_id']]=r['structure_class']
      fv={k:float(np.mean(v)) for k,v in family.items()};groups=defaultdict(list)
      for k,v in fv.items():groups[sg[k]].append(v)
      gv={k:float(np.mean(v)) for k,v in groups.items()}
      assert len(fv)==12 and len(gv)==8
      for weight,units in [('family_equal',fv),('structure_equal',gv)]:
        s=summaries[f'{scale}_{metric}_{c}'][weight];assert set(units)==set(s['units'])
        v=np.array([units[k] for k in sorted(units)])
        bs=v[np.random.default_rng(20261005).integers(len(v),size=(20000,len(v)))].mean(axis=1)
        actual=np.r_[v.mean(),v.std(ddof=1),np.quantile(bs,[.025,.975])]
        wanted=np.r_[s['mean'],s['sd'],s['ci95_descriptive']]
        err=float(np.max(np.abs(actual-wanted)));max_summary_error=max(max_summary_error,err);assert err<1e-10
        for k in units:assert abs(units[k]-s['units'][k])<1e-10
audit={'status':'passed','records_recomputed_from_logits':192,'trace_phase_margins_checked':192,'base_endpoints_checked':48,'contrasts_per_endpoint':6,'summary_weighting_combinations_checked':48,'bootstrap_draws':20000,'bootstrap_seed':20261005,'max_summary_error':max_summary_error,'nonpositive_slopes':nonpositive,'slopes':{c:dict(min=min(v),median=float(np.median(v)),max=max(v),n=len(v)) for c,v in slope_by_phase.items()},'primary_tests_run':False,'analysis_seed_is_not_model_replication':True,'source_hashes':{str(f.relative_to(p)):sha(f) for f in (p/'analysis/B1').glob('*') if f.is_file()}}
write('analysis_verification.json',json.dumps(audit,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
labels=['Total','History (EH)','Terminal (ET)','Interaction (I)','Terminal | history','History | terminal']
f,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
for ax,metric,title in zip(axes,['broad','specific'],['Broad: mean non-initial minus initial','Specific: V-minus-F target margin']):
    ss=[summaries[f'raw_{metric}_{c}']['structure_equal'] for c in contrasts]
    means=np.array([s['mean'] for s in ss]);lo=np.array([s['ci95_descriptive'][0] for s in ss]);hi=np.array([s['ci95_descriptive'][1] for s in ss])
    ax.errorbar(means,np.arange(6),xerr=[means-lo,hi-means],fmt='o',capsize=4,color='#205b80')
    ax.axvline(0,color='gray',lw=.8);ax.set_yticks(np.arange(6),labels);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlabel('Raw logit-margin difference');ax.grid(axis='x',alpha=.18)
f.suptitle('B1 descriptive interim | 12 bases, 8 declared structure groups\n20,000 group bootstrap draws; intervals are not confirmatory tests',fontsize=12)
for ext in ['png','pdf']:f.savefig(fig/f'figure-01-main-comparison.{ext}',dpi=180)
plt.close(f)
f,axes=plt.subplots(1,2,figsize=(10.5,5.0),layout='constrained')
for ax,metric in zip(axes,['broad','specific']):
    cols=['EH','ET','I','total'];units=sorted(summaries[f'raw_{metric}_EH']['structure_equal']['units'])
    v=np.array([[summaries[f'raw_{metric}_{c}']['structure_equal']['units'][u] for c in cols] for u in units]);lim=max(abs(v.min()),abs(v.max()))
    im=ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),['History','Terminal','Interaction','Total']);ax.set_yticks(range(8),units);ax.set_title(metric.capitalize()+' / raw margin')
    for i in range(8):
      for j in range(4):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',color='white' if abs(v[i,j])>.6*lim else 'black',fontsize=9)
    f.colorbar(im,ax=ax,shrink=.7,label='Logit-margin difference')
f.suptitle('B1 structure heterogeneity | declared groups are not certified independent samples',fontsize=12)
for ext in ['png','pdf']:f.savefig(fig/f'figure-02-structure-breakdown.{ext}',dpi=180)
plt.close(f)
cn={'total':'总变化','EH':'仅历史阶段','ET':'仅末步','I':'交互','ET_given_H':'已有历史干预时再加末步','EH_given_T':'已有末步干预时再加历史'}
def table(scale,include_sd=False):
    lines=['|指标|对比|8 结构等权均值|结构间 SD|描述性 95% 区间|12 family 等权均值|','|---|---|---:|---:|---|---:|']
    for m in ['broad','specific']:
      for c in contrasts:
        s=summaries[f'{scale}_{m}_{c}']['structure_equal'];fm=summaries[f'{scale}_{m}_{c}']['family_equal']['mean'];lo,hi=s['ci95_descriptive']
        lines.append(f'|{m}|{cn[c]}|{s["mean"]:.6f}|{s["sd"]:.6f}|[{lo:.6f}, {hi:.6f}]|{fm:.6f}|')
    return '\n'.join(lines)
main='''# B1 描述性分析核验

问题：相同冻结轨迹上，原 cap 的影响主要出现在历史计算、答案输出前最后一步，还是两者的交互？同时区分普遍离开原答案的倾向（broad）与 V 相对 F 的目标答案优势（specific）。

192 条 B1 记录 = 12 个 base × 4 个 NRVF 条件 × 4 个阶段开关，覆盖 12 个 family / 8 个已声明结构组。24 次 smoke 是工程验证，不能增加科学样本量。这里没有多模型种子重复实验；重放沿用冻结 seed，20261005 是统计 bootstrap seed。

原始 broad 的历史项 +1.504，末步项 +0.225，总变化 +1.646；历史项的数值更大。原始 specific 总变化 −1.398，历史项 −1.355。换句话说，cap 可以提高其他选项相对于原答案的平均分数，却同时降低 V 相对 F 的目标优势。“更愿意改答案”与“更有选择地听理由”不是同一个指标。

仿射校正后，broad 总变化约 +0.606、历史项 +0.562；specific 总变化约 −0.480，描述性区间 [−1.165, +0.138] 跨零。效应大小对输出尺度校正敏感。校正仅作敏感性分析，不能替换原始主指标，也不能因校正区间跨零宣称没有效应。

这加强了检查历史计算路径的动机，削弱了把效果主要归于末步局部读出的直观解释。但两因素干预不能定位具体 KV、层或角色方向；不能把比值当作因果中介比例，也没有等效界限来宣称末步可以忽略。

'''+table('raw')+'''

![原始主比较](figures/figure-01-main-comparison.png)

图 1 的历史项幅度较大；specific 的方向与 broad 不同。图中的区间是描述性组 bootstrap，不是通过多重检验的显著性结论。

![结构差异](figures/figure-02-structure-breakdown.png)

图 2 显示结构之间的异质性；不能将总体均值外推成每种推理题都会变化。需要保留所有结构而非挑选有利组。

## Claim Candidates

- Claim：本批冻结重放中历史阶段项大于末步项。
  - Source evidence：原始 SUMMARY.json、图 1、独立核验 analysis_verification.json。
  - Allowed wording：B1 描述性结果提示历史阶段值得继续定位。
  - Forbidden stronger wording：已经证明所有效应由历史 KV 中介；末步无效。
  - Uncertainty：仅 12 bases / 8 个旧结构组，依赖未获独立性认证，尚缺 B2。
  - Next check：按已冻结 B2 样本完成合并分析，再预注册定位实验。
  - Decision：keep（限定为观察）。
- Claim：原始 broad 上升与 specific 下降同时发生。
  - Source evidence：两主指标及其全部分项、结构表。
  - Allowed wording：普遍离开初始答案的相对得分变化不等于理由选择性改善。
  - Forbidden stronger wording：已经证实自然行为更理性、角色机制或完整论文假说。
  - Uncertainty：固定轨迹测量的是最后一步 logits；不包含自由生成行为；不报告中期 p 值。
  - Next check：保留原始主分析，报告仿射校正敏感性，另立 H 行为协议。
  - Decision：keep（限定为观察）。
'''
write('analysis-report.md',main)
stats='''# 统计与完整性附录

主比较、样本单位与矩阵见 analysis-report.md。以 base 内配对为起点，先 base→family，再 family→声明结构等权平均；8 结构等权为主，12 family 等权为敏感性，它们对应不同加权对象。192 个运行格不是 192 个独立样本。旧结构组之间的独立、对称、可交换假设尚未确认。

区间采用 20,000 次有放回组 bootstrap，seed=20261005，2.5/97.5 百分位。SD 表示声明结构均值间离散度，不是模型运行 seed 方差。不作等效性判断。不把区间是否跨零当作显著性通关。

本包不执行主检验。B1+B2 全部完成后，仅六个原始 EH/ET/I × broad/specific 对比组成一个 Holm 家族，按冻结规则执行一次 8 结构精确双侧符号翻转；这仍属于开发证据。最小双侧原始 p=2/256，第一阶 Holm 最小值=0.046875。B1、B2、合并均需报告，不挑选有利视图。

独立核验从全部 192 个完整词表 logits 重算选项 margin，检查 48 条轨迹的四阶段、48 组 base/指标/尺度端点、24 个对比的两种加权及所有 bootstrap 区间，误差均小于 1e-10。每个 JSON/NPZ、冻结依赖及 26 个模型/cap 资产全量 SHA-256 已检查。不会重新运行模型。

## 原始主结果

'''+table('raw')+'''

## 仿射校正敏感性（不能替代原始主结果）

校正按冻结 S1 词表划分和同 prompt 仿射参数。独立核验使用保存的 slope/intercept 重新计算校正 margin；其拟合参数的重算验证另见远程 scale_refit_verification.json。所有 48×4 个尺度均为正，没有静默排除。任意更强解释都须考虑留出词表拟合误差。

'''+table('corrected')+'\n\n尺度统计：\n\n```json\n'+json.dumps(audit['slopes'],indent=2)+'\n```\n'
write('stats-appendix.md',stats)
write('figure-catalog.md','''# 图表索引

两个图均来自已核验的 analysis/B1/SUMMARY.json；提供 PNG 和 PDF。均为 B1 开发阶段描述，不作显著性星标。

- figure-01-main-comparison：六个阶段对比 × 两个原始指标，点为 8 声明结构等权均值，横线为 20,000 次组 bootstrap 的描述性 95% 区间。目的是区分历史、末步、交互，并观察 broad 与 specific 不同方向。注意：区间不保证结构独立，不支持等效或自然行为结论。下一步是 B2，而非选有利分项。
- figure-02-structure-breakdown：每个声明结构的 EH、ET、I、total，格内给出精确到两位小数的 logit-margin 差。每个面板独立色阶，不能靠两图颜色深浅比较效应量。用于识别异质性，禁止据此删除结构或修改 B2。没有误差条，因为每格是对应旧结构下的确定性样本均值。

验收：图题标明 B1 和描述性；坐标标明原始 logit-margin 单位；全部结构保留；对比顺序固定；没有 p 值/显著性暗示。详尽数值、SD 和两种加权区间见 stats-appendix.md 与原始 SUMMARY.json。
''')
for n in ['analysis-report.md','stats-appendix.md','figure-catalog.md']:assert (dest/n).stat().st_size>0
wall=integrity['gpu_budget']['used_seconds'];parts=integrity['gpu_budget']['scope_seconds']
report='''---
type: results-report
date: 2026-10-06
experiment_line: cedar-b
round: 1
purpose: initial-package-closure
status: complete-interim
source_artifacts:
  - analysis-report.md
  - stats-appendix.md
  - figure-catalog.md
  - analysis_verification.json
linked_experiments:
  - ../../EXPERIMENT_PLAN.md
  - ../../B_PROTOCOL.md
linked_results:
  - ../B1_complete_evidence_20261006/analysis/B1/SUMMARY.json
---

# CEDAR B / Round 1（B1 首包） / initial-package-closure / 2026-10-06

## 1. 执行结论

CPU A/H 准备、smoke 24/24、B1 192/192 及自动分析完成。全部 216 条 JSON 与 216 个数组逐项核验，计划格集合和 results.json 一致；冻结依赖、操作修订和 26 个模型/cap 资产全量哈希通过。无剩余本批 campaign、runner 或分析进程，两个锁已释放。监控在归档同步完成后暂停，最终暂停状态见 closure_status.json。

## 2. 实验身份与要解决的问题

B1 是已授权的历史×末步四格干预首包，用来分辨原 cap 效果的计算时间位置。Round 1 对应 B1 首包，不是新一次独立确认。旧 C1/S1/P1 和旧失败 v2 均未重启。

## 3. 设置与协议

12 bases × NRVF × 00/10/01/11 = 192 个完整重放。保持 BF16、原权重、cap 算法、轨迹、terminal 边界与数值门槛。24 次 smoke 包含重放重复，只验证工程一致性。00/11 锚点、重复和重建的最大误差均为 0；保留原冻结协议，资源修订仅取消累计 12h 和 smoke 3h 时限。

## 4. 主要发现

历史项 EH：broad +1.504，specific −1.355；末步项 ET：broad +0.225，specific −0.184；总变化：broad +1.646，specific −1.398。平均而言历史阶段项更大，但末步并非被证明为零。已有历史干预时再加末步：broad +0.142，specific −0.043。各结构不完全同向。

“让其他选项相对原答案分数上升”与“V 相比 F 更偏向目标答案”不能互换。前者上升而后者下降，因此本批不能被包装成理由利用普遍改善。仿射校正后 broad 总变化约 +0.606（历史 +0.562），specific 总变化约 −0.480，描述性区间 [−1.165, +0.138] 跨零。效应幅度对输出尺度校正敏感；原始与校正结果必须同时报告，不选择有利尺度，也不由跨零断言无效应。

## 5. 统计核验

全部原始 margin、配对端点、两种聚合和描述性 bootstrap 区间完成独立 CPU 核验；未运行中期主 p 值。等权对象是 8 个声明结构组，独立性尚未认证；12 family 是敏感性加权。B2 后的六项 Holm 家族只执行一次。无等效界限、无多模型 seed 复现、无自然生成行为证据。

## 6. 图表逐项解释

图 1 展示历史项大于末步项的数值模式及两个指标的不同方向，使后续定位优先考虑历史路径；不支持把历史项除总变化当作因果解释比例。图 2 显示各旧结构的异质性，要求继续保留所有结构和冻结 B2。

![阶段分解](figures/figure-01-main-comparison.png)

![结构异质性](figures/figure-02-structure-breakdown.png)

## 7. 失败、负结果与限制

初始部署遗漏空 h1 目录的失败记录已保留，工程修复后完成；原科学代码未因此更改。早期 SSH 曾短暂超时，重连恢复，没有重启推理。完成时原监控选中了更新的空分析日志，产生跨文件负增长；实际推理日志继续增长，closure_integrity.json 已说明并重算同一日志差值。

特异性原始指标下降是需保留的结果，不据此取消 B2，也不据有利 broad 选择继续。B1 只覆盖旧开发结构，H 新独立结构认证数为 0。既不能确认完整论文机制，也不能由此证明机制不存在。

## 8. 对假说的影响

“需要检查历史计算路径”得到描述性支持；“末步局部读出足以解释主要现象”更难与当前数值相符。“cap 会改善理由选择性/自然推理行为”本批没有提供支持。具体层、KV、助手方向的因果定位仍未解决。

## 9. 下一步

建议按原冻结矩阵执行剩余 B2 192 格（需要下一包明确授权，本批未启动），不根据 B1 效应符号改样本。随后同时报告 B1/B2/合并，执行预定六项检验。C 必须在 B 完整后确定一个定位范围，完整 KV 交换先作实现校验；D 暂缓。

H 可先完成原型→真实依赖组→family→base 注册和结构认证。48 组只是规划目标，不能同构改名凑数；需先定义有科学意义的精度/效应目标。另冻结自然行为分母、无效与截断处理及固定重放可用率。现有 H 精度文件是情景规划，不是功效保证；H 模型矩阵未获本包启动授权。

## 10. 资源与可复查产物

'''+f'累计 GPU 加载/推理墙钟 {wall:.3f} 秒（{wall/3600:.3f} 小时）；smoke {parts["smoke"]:.3f} 秒，B1 {parts["B1"]:.3f} 秒。这是 4 卡并行任务的经过时间，不是各卡 GPU 小时之和。无累计时限；保留 90 分钟无推进与磁盘/显存门槛。\n\n'+'''原始证据完整包 B1_complete_evidence_20261006.tar.gz，内部 evidence_manifest.json 逐文件列出哈希；报告及图另存 reporting_bundle_20261006.tar.gz。见 closure_integrity.json、analysis_verification.json、scale_refit_verification.json、transfer_receipt.json 和 closure_status.json。全部本地产物保存 D:/CEDAR；未向 Obsidian 或论文源文件写回。

当前决策：结束本包监管并保留全部证据；B2/H 为下一步候选，等待下一包授权，不自动扩展。
'''
write('2026-10-06--cedar-b--r1--initial-package-closure.md',report)
print(json.dumps(audit,ensure_ascii=True))
