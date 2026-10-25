"""Build an additive post-run evidence review from verified saved results."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'explore_outputs/batch_review_20260928'
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
audit=json.loads((ROOT/'explore_outputs/batch_integrity_20260928.json').read_text())
s1=json.loads((ROOT/'analysis-output/formal/s1/S1_SUMMARY.json').read_text())
p1=json.loads((ROOT/'analysis-output/formal/p1/P1_SUMMARY.json').read_text())
assert audit['status']=='passed'
S=audit['s1_primary']; P=audit['p1_primary']
metrics=['raw_delta_broad','restored_delta_broad','raw_delta_specific','restored_delta_specific']
directions=['U->U','U->P','P->P','P->U']


def write(name,text):
    (OUT/name).write_text(text,encoding='utf-8')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def ci(v): return f'[{v[0]:.4f}, {v[1]:.4f}]'


s1table=table(['主源 V−F 对比','均值','family SD','描述性 95% CI','Holm p','负向 family'],
    [[m,f"{S[m]['mean']:.4f}",f"{S[m]['family_sd']:.4f}",ci(S[m]['ci95']),f"{S[m]['p_holm_four_selectivity_contrasts']:.6f}",f"{12-S[m]['positive_families']}/12"] for m in metrics])
p1table=table(['训练→测试','family AUC','描述性 95% CI','balanced accuracy','原始置换 p','Holm p'],
    [[k,f"{P[k]['equal_family_auc']:.4f}",ci(P[k]['descriptive_family_bootstrap_ci95']),f"{P[k]['equal_family_balanced_accuracy']:.4f}",f"{P[k]['paired_label_permutation_p']:.6f}",f"{P[k]['holm_p_four_transfer_directions']:.6f}"] for k in directions])
scale=audit['descriptive_scale_prediction']['e2a_uncapped']
scaletable=table(['拟合方式','identity RMSE','scale RMSE','affine RMSE'],[[mode]+[f"{scale[mode][k]['mean']:.4f}" for k in ('identity','scale','affine')] for mode in ('same_prompt_fits','family_out_fits')])
sens=s1['aggregate']['e2a_full_p25']['V_minus_F']
senstable=table(['capped-source 敏感性对比','均值','描述性 95% CI'],[[m,f"{sens[m]['mean']:.4f}",ci(sens[m]['ci95'])] for m in metrics])

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.2,4.2))
for i,m in enumerate(metrics):
    r=S[m]; vals=np.array(list(r['family_means'].values()))
    ax.scatter(i+np.linspace(-.12,.12,12),vals,s=23,color='#4c78a8',alpha=.75,label='Family means (n=12)' if i==0 else None)
    ax.errorbar(i,r['mean'],yerr=np.array([[r['mean']-r['ci95'][0]],[r['ci95'][1]-r['mean']]]),fmt='s',color='#c54b30',capsize=5,label='Equal-family mean; descriptive 95% CI' if i==0 else None)
ax.axhline(0,color='.45',lw=1)
ax.set_xticks(range(4),['Raw broad','Restored broad','Raw specific','Restored specific'])
ax.set_ylabel('V - F in capped-minus-uncapped margin change (logits)')
ax.set_title('S1: residual selectivity after within-prompt affine correction')
ax.legend(fontsize=8,loc='lower left')
fig.tight_layout()
for ext in ('pdf','png'):fig.savefig(OUT/'figures'/('s1_selectivity_verified.'+ext),dpi=160)
plt.close(fig)

fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4),sharey=True)
for i,k in enumerate(directions):
    r=P[k]; vals=np.array(list(r['family_auc'].values()))
    ax.scatter(i+np.linspace(-.12,.12,12),vals,s=20,color='#4c78a8',alpha=.65)
    ax.errorbar(i,r['equal_family_auc'],yerr=np.array([[r['equal_family_auc']-r['descriptive_family_bootstrap_ci95'][0]],[r['descriptive_family_bootstrap_ci95'][1]-r['equal_family_auc']]]),fmt='s',color='#c54b30',capsize=4)
    bx.bar(i,r['equal_family_balanced_accuracy'],color='#679f76',width=.55)
    bx.text(i,r['equal_family_balanced_accuracy']+.035,f"{r['equal_family_balanced_accuracy']:.3f}",ha='center',fontsize=9)
for a in (ax,bx):
    a.axhline(.5,color='.4',ls='--',lw=1)
    a.set_xticks(range(4),directions);a.set_ylim(0,1.08)
ax.set_ylabel('Score');ax.set_title('Equal-family AUC; descriptive 95% CI')
bx.set_title('Balanced accuracy at frozen threshold')
fig.suptitle('P1: prompt layer 53, unit RMS, train-only nuisance adjustment',fontsize=11)
fig.tight_layout()
for ext in ('pdf','png'):fig.savefig(OUT/'figures'/('p1_transfer_verified.'+ext),dpi=160)
plt.close(fig)

analysis=f'''# 第一批 S1/P1 证据核验

主问题：capping 的固定轨迹读出变化能否由非选项词表拟合的尺度/仿射预测解释；去除该预测后，V−F 差异是否仍存在；主探针是否能在 U/P 间双向转移。

数据为 Qwen3-32B BF16、原始冻结轨迹的开发分析。192 条轨迹包含 96 条 uncapped-source 主源和 96 条 capped-source 敏感性源，各跑 U/P，合计 384 次正式前向；另有 12 次工程 smoke。独立结构单位为 12 个 family，24 个 base；轨迹和 U/P 前向不是独立重复。P1 主分析为主源 V/F 的 48 条记录，每个 family 4 条。单一冻结采集配置，无独立模型种子复现。

## 核验与主要证据

396 个原始前向 NPZ 的记录哈希全部匹配；输入轨迹哈希、6 项来源指纹、全部端点与有限值检查通过。正式 384 条记录的历史 A/B/C/D logits 最大偏差为 0，末层 logits 重建最大偏差为 0；smoke 全状态重复最大偏差为 0。此处精确历史一致只覆盖所保存的历史选项 logits。

{scaletable}

表中为 12 个 family 等权的选项预测 RMSE。same-prompt 使用同一条轨迹的配对 capped 非选项 logits；family-out 仅用其他 family 拟合参数。两者不能互换表述。尺度/仿射预测有明显描述性改善，但不能把 RMSE 改善解释为因果解释率。

{s1table}

restored 指正斜率 affine 校正。broad = 其他三个候选 logits 的均值减初始答案 logit；specific = 目标答案 logit 减两个非初始/非目标候选的均值。先算 P−U，再作 V−F。负值表示 capping 对 V 的该 margin 改变量比 F 更负。校正后 broad 12/12 family 为负，specific 10/12 为负。残差并未降到本次重复误差底噪。

{p1table}

四个方向的 Holm 校正 p 均为 0.039604，受 100 次配对标签置换的 1/101 分辨率限制。AUC 与 balanced accuracy 的差别必须同时报告：较好的 family 内排序不代表稳定的固定阈值分类，也不证明模型使用这些信息进行推理。

## 敏感性及边界

{senstable}

capped-source 的 restored specific 区间跨 0；不能把主源结果写成所有轨迹来源都一致。两个来源各 96 条的 A/B/C/D argmax flip 均为 0/96；这些是固定轨迹读出，不能宣称自然生成行为改变。

冻结程序已完成，但更广的科学协议仍有附加诊断缺口：P1 没有 capped-source 探针敏感性或单独预测 initial/target 身份的 nuisance endpoint；S1 family-out 保存了选项预测，但没有单独导出 held-out 非选项误差。它们未被标成已完成，也不改变已核验的主分析统计量。补充时应独立标注，保留本次原始产物。

## Claim Candidates

- Claim: 主源上，非选项拟合的 affine 校正降低选项预测误差，但未消除 V−F margin 差异。
  - Source evidence: S1_SUMMARY.json、S1_DETAILS.json、batch_integrity_20260928.json。
  - Allowed wording: 在这组固定开发轨迹与 12 个 family 中，仍观察到尺度校正后的差异。
  - Forbidden stronger wording: 已证明 capping 破坏语义有效性通路；尺度完全解释/完全不能解释干预。
  - Uncertainty: family 数少；符号翻转的对称性与独立性是假设；敏感性源不完全一致。
  - Next check: C1 双向干预和匹配随机对照。
  - Decision: keep
- Claim: 预先指定的 prompt L53 探针在四个 U/P 方向上具有高于置换零分布的 family AUC。
  - Source evidence: P1_SUMMARY.json、examples.json 与 100 次置换结果。
  - Allowed wording: 在该开发集和该探针流程下，V/F 标签具有跨条件可解码性。
  - Forbidden stronger wording: 发现同一语义表征、证明其被因果使用、证明两种条件等价。
  - Uncertainty: 48 条、24 配对 base、12 family；fold 训练集重叠；可能依赖提示表面特征；balanced accuracy 仅约 0.58–0.60。
  - Next check: nuisance endpoint、匹配 C1 对照、新结构 H1。
  - Decision: weaken
'''
write('analysis-report.md',analysis)
stats=f'''# 统计与来源附录

原始结果保留在 analysis-output/formal；本目录为独立事后核验与汇总，不改写原始输出。SEED=20260927。

S1：每个 base 的 V/F 配对，先求 family 内均值再对 12 个 family 等权。单位为 logit 差，family SD 不是 seed SD。20,000 次 family bootstrap 的 95% 区间均标为描述性。对四个预先指定主对比枚举 2^12=4096 个符号翻转，双侧检验，四项 Holm 校正；假设 family 效应独立且对称，未通过正态性预检替换该既定方法。

{s1table}

P1：48 个 V/F 例子、24 个成对 base、12 个外层 family fold；每个 family 4 条。initial/target nuisance 回归和标准化只拟合训练 family；固定 Logistic C=1。主效应量为 family AUC（高于 0.5 的排序性能）及 frozen-threshold balanced accuracy。100 次 base 内 V/F 标签交换，所有外层折重新拟合，四方向同一标签置换；p=(1+超过或等于观测值次数)/101，四项 Holm。单侧超零分布检验依赖配对标签可交换性；不是 AUC 等价性检验。

{p1table}

P1 AUC 描述性 CI 来自 20,000 次 fold-family 重采样；训练集重叠，不当作精确采样分布。24 种 layer/position/representation/nuisance 组合都已保存；只有 prompt_L53_unit_rms_nuisance1 为主分析。未从敏感性组合挑选最佳结果替代主分析。

核验重算了全部 24×4 组探针的 AUC、balanced accuracy、描述性 CI，以及主置换 p/Holm；重算了 S1 主均值、family SD、CI、符号翻转 p/Holm。没有重跑模型前向或 100 次探针拟合。规范 token split 不含选项 token，且 fit/test 不重叠。

1266 项原始产物/来源哈希见 ../batch_integrity_20260928.json。模型 65GB shard 哈希沿用采集前校验记录，未重复全盘读取；counterfactual 数组当时没有独立期望哈希，本次检查形状/有限值/配对身份并建立事后哈希，不能称为采集前哈希验证。
'''
write('stats-appendix.md',stats)
write('figure-catalog.md','''# 图表说明

## figures/s1_selectivity_verified.pdf / .png

目的：比较同一主源上 raw 与 affine-restored 的 V−F margin 差异。点为 12 个 family 均值，方块为等权总体均值，误差线为 20,000 次 paired-family bootstrap 的描述性 95% 区间。数据来自核验过的 S1_SUMMARY。

读图：校正后差异缩小但仍存在；broad 全为负，specific 有 2 个正向 family。决策：可以继续设计匹配 C1，不能直接推出原因使用通路。检查：只含 uncapped-source；没有 token 独立性推断；误差线不是跨模型种子置信区间。

## figures/p1_transfer_verified.pdf / .png

目的：同时展示四方向的 family AUC 与固定阈值 balanced accuracy。左图散点为 12 个 fold-family AUC，方块及误差线为等权 AUC 和描述性 bootstrap CI；右图是 balanced accuracy。虚线为 0.5。数据来自核验过的 P1_SUMMARY。

读图：AUC 在 0.81–0.85，但 balanced accuracy 仅 0.58–0.60；不能只报高 AUC。决策：保留可解码性描述，因果用法留待 C1，跨新结构泛化留待 H1。检查：4 个方向齐全、prompt L53 主端点、每 family 4 例、fold 训练集重叠，CI 只能作描述。

原始 S1 heldout_prediction.pdf 仍在 analysis-output/formal/s1/figures：点为单条轨迹的 identity 与 affine 选项 RMSE，支持描述性的误差改善；不能当作 unseen-family 预测图，也不是因果解释率。
''')
provenance=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_audit_sha256=hashlib.sha256((ROOT/'explore_outputs/batch_integrity_20260928.json').read_bytes()).hexdigest(),scientific_protocol_changed=False,outputs={})
for f in sorted(OUT.rglob('*')):
    if f.is_file():provenance['outputs'][str(f.relative_to(OUT))]=hashlib.sha256(f.read_bytes()).hexdigest()
write('provenance.json',json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'review_directory':str(OUT),'files':len(provenance['outputs']),'status':'review_bundle_written'}),flush=True)
