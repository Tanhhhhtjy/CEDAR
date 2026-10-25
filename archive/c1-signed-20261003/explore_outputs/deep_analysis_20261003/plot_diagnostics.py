from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path('D:/CEDAR/arr_c1_signed_20260928/explore_outputs/deep_analysis_20261003')
X=json.loads((O/'diagnostics.json').read_text());D=O/'figures';D.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
names=['same_broad_common','same_broad_interaction','same_specific_common','same_specific_interaction']
labels=['整体改答倾向：共同响应变化','整体改答倾向：V/F 差异响应变化','目标偏好：共同响应变化','目标偏好：V/F 差异响应变化']
colors=['#0072B2','#0072B2','#D55E00','#D55E00']
def save(fig,name):
 fig.savefig(D/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
 fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.8))
for i,(k,label,c) in enumerate(zip(names,labels,colors)):
 v=X['primary'][k];m=v['equal_family_mean'];lo,hi=v['descriptive_ci95'];y=3-i
 ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color=c,capsize=5,elinewidth=2,markersize=8)
 ax.text(.074,y,f"效应 {m:+.4f}\nHolm p = {v['holm_four_primary']:.4f}",va='center',fontsize=9)
ax.axvline(0,color='#777',ls='--',lw=1);ax.set_xlim(-.06,.125);ax.set_ylim(-.6,3.6)
ax.set_yticks(range(4),labels[::-1]);ax.set_xlabel('尺度校正后，capping 对定向响应的改变（logit margin 单位）')
ax.set_title('四个冻结主端点：区间均跨过零',loc='left',fontweight='bold',pad=20)
ax.grid(axis='x',alpha=.18)
fig.text(.03,-.015,'点：12 个题族等权均值；线：描述性 95% family bootstrap 区间（20,000 次）。\n先扣除 8 个匹配随机方向；p 值来自 family 符号翻转，并对四项统一 Holm 校正。固定开发轨迹，非自然生成。',fontsize=9)
fig.tight_layout();save(fig,'01_primary_endpoints')
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
groups=['true','random','replace'];glabels=['P1 定向修改','等范数随机修改','局部整向量替换']
for ax,metric,title in zip(axes,['broad','specific'],['整体改答 margin 的绝对变化','目标偏好 margin 的绝对变化']):
 for i,k in enumerate(groups):
  q=X['absolute_response_family'][k+'_raw_'+metric];m=q['mean'];lo,hi=q['ci95'];vals=list(q['family_means'].values())
  ax.scatter(np.arange(12)/70+i-.08,vals,color='#777',s=16,alpha=.45)
  ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='D',color=['#0072B2','#999999','#D55E00'][i],capsize=5,elinewidth=2,markersize=7)
 ax.set_xticks(range(3),glabels,rotation=12);ax.set_ylim(0,.15);ax.set_ylabel('绝对变化（raw logits）');ax.set_title(title,loc='left',fontsize=12)
 ax.grid(axis='y',alpha=.18)
fig.suptitle('改动确实影响分数，但定向修改没有显示明显的幅度优势',x=.01,ha='left',fontweight='bold')
fig.text(.01,-.03,'菱形/线：题族等权均值与描述性 95% 区间；灰点：题族均值。事后描述性诊断，无等效性结论。\n随机方向先在相同题目内汇总；完整向量替换幅度更大，作为位置响应诊断，不是匹配方向对照。',fontsize=9)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'02_output_response_controls')
fig,ax=plt.subplots(figsize=(10.5,5.9))
fam=sorted(next(iter(X['primary'].values()))['family_means'])
mat=np.array([[X['primary'][k]['family_means'][f] for k in names] for f in fam])
im=ax.imshow(mat,cmap='RdBu_r',vmin=-.18,vmax=.18,aspect='auto')
for i in range(12):
 for j in range(4):ax.text(j,i,f'{mat[i,j]:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(mat[i,j])>.12 else '#222')
ax.set_xticks(range(4),['整体·共同','整体·V/F交互','目标·共同','目标·V/F交互'])
ax.set_yticks(range(12),[f.split('-')[-1] for f in fam]);ax.set_ylabel('开发集题族')
ax.set_title('多数端点的题族方向互相抵消；目标共同响应更偏负',loc='left',fontweight='bold',pad=15)
fig.colorbar(im,ax=ax,label='尺度校正后的主端点值')
fig.text(.01,-.025,'每格为一个题族（两个 base）的平均效应；不是单次前向。\n正/负题族数依次为 6/6、6/6、3/9、6/6；不能事后只挑同方向题族宣称发现。',fontsize=9)
fig.tight_layout();save(fig,'03_family_heterogeneity')
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
full=X['full_capping_baseline']['broad_ALL']
for ax,target,title in zip(axes,'UP',['U 中换入 P 的局部向量','P 中换入 U 的局部向量']):
 swap=X['donor_motion'][target+'_broad']['mean_shift']
 vals=[full,swap] if target=='U' else [dict(mean=-full['mean'],ci95=[-full['ci95'][1],-full['ci95'][0]]),swap]
 for i,v in enumerate(vals):
  m=v['mean'];lo,hi=v['ci95'];ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',color=['#777','#D55E00'][i],capsize=6,markersize=8,elinewidth=2);ax.text(i,m+(.13 if m>=0 else -.27),f'{m:+.4f}',ha='center')
 ax.axhline(0,lw=1,color='#999');ax.set_xticks([0,1],['原有 U/P 整体差异','本次局部替换']);ax.set_ylim(-2.85,2.85)
 ax.set_title(title,fontsize=12);ax.set_ylabel('整体改答 margin 变化（raw logits）')
 ax.grid(axis='y',alpha=.15)
fig.suptitle('替换一个 prompt 位置的完整向量，没有重现或逆转整体 capping 差异',x=.01,ha='left',fontweight='bold',fontsize=12)
fig.text(.01,-.035,'仅换 block 54 输入处最后一个 prompt token 的向量；此前状态、其他 token、KV cache 和后续条件均未整体替换。\n点/线：12 题族均值与描述性 95% 区间。两种操作作用范围不同；此图不是因果中介比例估计。',fontsize=9)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'04_local_swap_context')
print('Created 4 PNG + 4 PDF figures',str(D))

