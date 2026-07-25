# CEDAR 论文架构专家审阅委托说明

> 用途：与 `2026-07-25-cedar-paper-architecture-draft.md` 一并交给外部专家。
>
> 审阅对象：论文故事、贡献定位、章节架构、证据接口、实验充分性和图表规划。
>
> 本轮不是英文润色，也不是最终论文打分。

---

## 可直接复制给专家的委托说明

你是一位熟悉 AAAI/NeurIPS/ICML 论文评审、LLM safety、representation engineering、mechanistic interpretability、belief revision 与 benchmark design 的资深研究者。请对附件中的 CEDAR 论文架构草稿进行严格、独立、可执行的评审。

附件：

- `2026-07-25-cedar-paper-architecture-draft.md`

项目仓库：

- https://github.com/Tanhhhhtjy/CEDAR

你的任务不是帮助作者证明预设结论，而是判断：这个研究问题是否成立、故事是否有足够分量、两个评测模块能否支撑所定义的主张、章节安排是否是最有说服力的组织方式，以及还缺少哪些科学上必要的证据。

请忽略投稿截止时间、剩余天数和短期执行压力。不要因为时间紧而建议缩小故事，也不要以“做不完”为理由降低证据标准。实验优先级只按科学必要性、主张依赖关系和信息增益排序。

所有 `TBD` 均表示尚无结果。不得补造数字、趋势、显著性或定性结论，也不得默认这些实验一定支持作者假设。请分别评价：

1. **当前证据状态下，论文实际上能支持什么；**
2. **如果标题级 `TBD` 最终由真实、审计过且统计充分的结果支持，这个故事的潜在分量是否达到 AAAI 主会水平。**

## 一、必须理解的研究边界

论文拟研究：用于抑制不期望行为或社会压力影响的 activation capping，是否也会抑制由有效新证据触发的合理修正。

暂定标题级主张是：

> Activation capping may suppress harmful drift while impairing warranted evidence-driven revision—an epistemic cost missed by standard capability evaluations.

这目前是待验证主张，不是已有结论。

作者选择的叙事优先级是：

> 现象发现是论文主角；评测资产是测量仪器；机制负责解释；控制器是条件性解决方案。

请不要因为作者已经选择该方案就默认它最佳。你需要独立比较 phenomenon-first、benchmark-first、mechanism-first 或你提出的其他组织方式。

### 两个评测模块

本项目不是一套统一的 `V/F/S/AO/N/R` 六臂实验，而是两个互补、统计上独立的模块：

1. **CEDAR-Evidence：N/R/V/F**
   - N：中性延续，无新理由，应保持；
   - R：要求复核但无新信息，应保持；
   - V：有效证据，规范上应更新；
   - F：表面匹配但逻辑无效的论证，应保持。

2. **CEDAR-Social：AO/S**
   - AO：只暴露错误目标答案；
   - S：在同一目标暴露上增加社会压力，但不增加任务事实。

Evidence 模块的认识论主终点为：

\[
WEG(c)=P(Update\mid V,c)-P(Update\mid R,c),
\]

\[
Cost_V(c)=WEG(uncapped)-WEG(c).
\]

Social 模块使用 target-minus-initial option-logit margin：

\[
SME(c)=M_S(c)-M_{AO}(c),
\]

\[
Benefit_S(c)=SME(uncapped)-SME(c).
\]

这些量不可与 BU-Acc、target-follow rate、V−F、S−AO 的其他实现或离散 argmax 指标互相替换。请专门审查这种双模块设计是否概念清晰、是否足以支持“epistemic selectivity”和“safety–epistemic frontier”。

### 当前实际实证范围

- 主模型：`Qwen/Qwen3-32B`；
- 固定 revision：`9216db5781bf21249d130ec9da846c4624c16137`；
- 官方 Assistant Axis 资产 revision：`3b3b788432ad33e3a28d9ff08e88a530c0740814`；
- 官方 p25 配置：`layers_46:54-p0.25`，实际层 46–53；
- upper-cap 算子：

\[
h'=h-v\max(\langle h,v\rangle-\tau,0).
\]

当前直接证据只覆盖一个模型、一个官方 axis 和一类 threshold capping。概念讨论可以涉及 activation-axis capping，但实证结论不能自动泛化到所有模型、axis 或干预算子。

### 当前真实证据状态

1. **CEDAR-Social / C2-V8：已完成并冻结。**

   - 600 bases，AO/S × uncapped/p25 严格配对；
   - direct 主结果：`cap × (S−AO) = −0.78625 logit`；
   - 10,000 次 base bootstrap 95% CI `[-0.86000, -0.71104]`；
   - AO cap effect `+1.26792`，S cap effect `+0.48167`；
   - 准确解释：p25 在 AO 与 S 中都提高 target margin，但在 S 中提高得更少，因此社会措辞相对纯目标暴露的增量下降；
   - 不能写成 S 条件绝对 target margin 下降，也不能称模型内部真实谄媚下降；
   - free-generation 离散 SocialExcess 为 0；
   - 原协议的正向判定文字与最终负向交互解释之间存在符号口径疑点，需要定义审计，不能静默重解释。

2. **CEDAR-Evidence：标题级确认结果尚未取得。**

   当前内部 development validation 只有 24 paired bases、12 family clusters：

   - V target-follow：p25−uncapped `+20.83pp`；
   - F target-follow：`+16.67pp`；
   - V−F discrimination change：`+4.17pp`，95% CI `[-20.83pp,+29.17pp]`。

   该结果不支持“p25 抑制有据更新”，也不是独立 held-out confirmation。正式认识论代价、剂量曲线和 Pareto 前沿均为 `TBD`。

3. **机制：尚未形成足以支持因果解释的证据。**

   - E0 只是 coarse observational diagnostic；
   - E1 的预注册 phase/layer 交互未同时过闸；
   - 尚无双向 patching/rescue；
   - J-Lens、MUSE、probe 尚未使用；
   - 因此 layer/phase、几何重叠和 causal mediation 均为 `TBD`。

4. **控制器：尚未验证。**

   门控正交 capping、L0/L1/L2 和 Pareto 改善均为计划，不能当作已有方法贡献。

## 二、去锚定审阅程序

请分两阶段工作，并在最终报告中保留两阶段的判断变化。

### 阶段一：独立诊断

在接受附件给出的章节安排之前，先只根据研究问题、证据状态和 estimands 独立回答：

1. 你认为这篇论文最值得发表的一句话贡献是什么？
2. 你会选择 phenomenon-first、benchmark-first、mechanism-first，还是其他叙事？为什么？
3. 如果所有标题级 `TBD` 都由严格、真实的结果支持，这个故事是否有 AAAI 主会分量？
4. 当前已完成证据实际能支撑到什么层级？
5. 哪三个问题最可能导致拒稿？
6. 不参考作者目录时，你会如何安排一级章节？

请先记录这些答案，再审查作者提供的完整架构。不要为了与作者保持一致而修改独立判断；若后续改变判断，请明确说明是什么信息导致了变化。

### 阶段二：逐项审查作者架构

阅读完整附件后，审查：

- 中心故事是否单一、清晰且非循环论证；
- CEDAR-Evidence 与 CEDAR-Social 是否应保持独立，如何在结果中合理连接；
- `WEG/Cost_V` 与 `SME/Benefit_S` 的定义、方向和解释是否正确；
- Section 3 是否足以建立规范 gold、对照条件和统计单位；
- Section 4 的结果顺序是否能支撑 trade-off，而不是把两个异质效应硬拼成 Pareto；
- Section 5 的机制证据层级是否合理，哪些实验真正必要；
- Section 6 是否值得独立成节，以及控制器失败时如何降级；
- 三条贡献是否准确，数据集/benchmark 贡献是否被低估或夸大；
- 7 页正文的篇幅分配是否现实，哪些内容必须进附录；
- Figure 1–3 的信息角色是否正确，是否缺少一张关键图或存在重复；
- 单模型单轴的证据能否支持当前标题和摘要范围；
- 与最近邻工作的差异是否足够明确。

## 三、需要重点裁决的问题

请对以下问题逐一给出明确结论、理由、置信度和所需证据，而不是只说“需要更多实验”。

### A. 故事与分量

1. 如果社会侧收益、认识论代价、剂量前沿、机制和控制器均按架构获得严格支持，这是否构成一篇有足够新颖性和技术分量的 AAAI 论文？
2. 如果只有“社会侧收益 + 正式认识论代价 + 严格 benchmark”成立，而机制/控制器不成立，论文是否仍足够？
3. benchmark/data contribution 在故事中应该是第二贡献、主贡献，还是仅作为实验工具？
4. “epistemic cost”“epistemic selectivity”“warranted revision”三个术语是否准确、必要且不会引发过强心智归因？

### B. 新颖性与最近邻

请主动检索或列出附件既有文献之外的最近邻，尤其关注 2025–2026 年：

- activation capping/activation steering 的 collateral effects；
- safety intervention 对 truthfulness、corrigibility 或 belief revision 的影响；
- dual-stance、bidirectional sycophancy 或 evidence-sensitive revision；
- reason-quality-aware evaluation；
- 同时优化行为安全与动态认知能力的 Pareto 工作。

对每篇最近邻说明：它解决什么、与 CEDAR 重叠在哪里、CEDAR 真正多做了什么、是否会削弱 novelty。请给出可核验链接；无法核验全文时标注 `conditional`，不要猜测论文内容。

### C. 构念与 estimand

1. V/R 差是否足以操作化 warranted-evidence gain？是否还需要 N、F 或 calibrated confidence 才能避免混淆？
2. AO/S 的 margin interaction 能否合理称为社会压力增量？当前负号解释是否一致？原协议的正向判断口径是否影响其确认性地位？
3. `Benefit_S` 与 `WEG` 来自不同任务和不同统计单位，将它们绘制为 Pareto/前沿是否科学合理？需要怎样的剂量匹配、归一化和不确定性表达？
4. “static capability preserved but dynamic revision impaired”需要哪些能力对照和等效性标准才能成立？
5. 当前 10pp 实质阈值应如何绑定到 `Cost_V`？三力实验中的 TOST 和双地板是否应完全排除在主终点之外？

### D. 章节与论证顺序

逐节判断：

- 这一节是否必要；
- 它应该回答的唯一问题是什么；
- 当前内容是否放错位置；
- 是否有重复、跳步或缺失的论证；
- 建议页数；
- 若删减，移到何处。

请特别判断以下顺序是否最佳：

```text
问题与缺口
-> 两个测量模块
-> 社会侧目标效应
-> 有据更新代价
-> 联合前沿
-> 机制
-> 控制器
```

如果你认为顺序不佳，请给出完整替代目录，不要只指出问题。

### E. 最小充分证据集

请把后续证据分成三类：

- **Must-have：** 缺失就不能支持标题或主要贡献；
- **Strongly recommended：** 不一定致命，但显著影响可信度/分量；
- **Optional/appendix：** 有帮助但不应阻塞主故事。

每项实验必须注明：

1. 它检验哪个具体 claim；
2. 主 estimand 和独立统计单位是什么；
3. 什么结果支持、削弱或证伪该 claim；
4. 它排除哪个替代解释；
5. 失败后论文应如何重写；
6. 是否需要第二模型、第二独立 axis、跨域或开放式生成。

不要根据希望得到的方向设计停止规则，也不要建议挑题、挑 seed、挑 layer 或根据结果重新定义 gold。

### F. 机制证据标准

请给出从弱到强的证据阶梯，并判断主文最低需要达到哪一级：

```text
behavioral association
< observational trajectory
< decodable/probe or geometric alignment
< phase/layer intervention
< bidirectional patching/rescue
< mediation with preserved controls
```

具体评估 E2 validity-vs-choice、prefill/decode 2×2、projection regression、J-Lens 和双向 patching/rescue 各自回答什么。若某项不增加独立信息，请建议删除。

### G. 控制器定位

1. L0/L1/L2 是否构成自然、可理解且公平的比较？
2. “只用 V 构造保护子空间”的逻辑是否充分？
3. 应按社会收益、实际 correction dose、trigger rate 还是其他量匹配？
4. 怎样的结果才算真正移动 Pareto 前沿？
5. 控制器失败时，是否仍值得在主文保留一个负结果小节？

### H. 图表规划

请逐张评价 Figure 1–3：

- 最重要的信息是什么；
- 是否能在一眼内读懂；
- 哪些面板重复或信息过载；
- 什么必须是概念图，什么必须由真实数据生成；
- 是否需要用表替代图；
- 7 页内的优先级。

如果你认为现有图表设计不理想，请给出新的三图方案，并说明每张图如何推进论证。

## 四、强制输出格式

请用中文提交评审，保留必要英文术语和公式。按照以下结构输出：

### 1. 独立复述

用不超过 200 字复述你理解的研究问题、方法和潜在贡献。若无法清晰复述，说明是哪一处定义或叙事造成困难。

### 2. 双重总评

分别给出：

| 评价对象 | 等级 | 核心理由 | 置信度 |
|---|---|---|---|
| 当前已有证据能够支撑的论文 | Strong / Promising / Borderline / Not viable |  | 高/中/低 |
| 标题级 TBD 全部被严格验证后的潜在论文 | Strong / Promising / Borderline / Not viable |  | 高/中/低 |

等级含义：

- **Strong：** 核心问题、创新和证据链足以形成明确的主会论文；
- **Promising：** 有主会潜力，但仍有一个可明确修复的关键缺口；
- **Borderline：** 问题有意义，但创新、证据或故事尚不足以形成稳定主张；
- **Not viable：** 中心构念、因果链或最近邻重合使当前方向难以成立。

不要输出虚构的接收概率。

### 3. 最强之处与最危险之处

- 最强的 3 点；
- 至少 3 个可能导致拒稿的高风险问题；
- 对每点标注“可通过写作修复 / 需要新分析 / 需要新实验 / 可能不可修复”。

### 4. 新颖性与最近邻表

| 工作 | 与 CEDAR 的重叠 | CEDAR 的差异 | 风险判断 | 来源/置信度 |
|---|---|---|---|---|

必须包含附件清单之外的最近邻；若认为不存在，也要说明检索范围和局限。

### 5. Claim--evidence 审计

| Claim | 当前证据 | 尚缺证据 | 允许措辞 | 禁止措辞 |
|---|---|---|---|---|

至少覆盖：social-side effect、epistemic cost、dose frontier、mechanism、controller、跨模型/跨轴泛化。

### 6. Estimand 与统计审计

逐项审查 `WEG`、`Cost_V`、`SME`、`Benefit_S`、`Benefit_F`，特别说明是否可以联合形成 Pareto、如何处理符号和置信区间、统计单位是否正确。

### 7. 章节逐项审查

| 章节 | 应回答的问题 | 当前问题 | 建议修改 | 建议篇幅 |
|---|---|---|---|---:|

随后给出你推荐的完整一级/二级目录。

### 8. 最小充分证据集

| 优先级 | 实验/分析 | 解锁的 claim | 主 estimand/统计单位 | 证伪条件 | 失败后的叙事 |
|---|---|---|---|---|---|

### 9. 图表审查与替代方案

逐张评价现有 Figure 1–3，并给出你推荐的最终三图结构。

### 10. 作者需要决定的问题

列出不超过 7 个真正需要作者判断、且不能仅靠查资料解决的决策。每项给出你的推荐选项与取舍。

### 11. 优先修改清单

给出按顺序排列的 Top 10 修改项，区分：

- 立即修改论文架构；
- 等正式结果后决定；
- 可留到附录或后续工作。

## 五、评审纪律

- 不要因作者已有偏好而默认接受故事；
- 不要把实验计划、概念示意或期待方向写成结果；
- 不要使用“p>0.05 所以等价”“probe 可读所以因果介导”等错误推理；
- 不要把 social target-following 直接称为内部真实谄媚；
- 不要把输出层 revision 直接称为人类式内部信念变化；
- 不要为了得到预期结论而建议结果依赖的筛选、补样、阈值修改或后验换主终点；
- 无法访问仓库或原始材料时，将相应判断标为 `conditional`；
- 对每项重要判断标注置信度；
- 若认为作者的主故事错误，请直接提出替代故事；
- 请直接提交完整评审报告，不必先向作者询问截止时间或执行资源。

---

## 给作者的使用说明

将本委托说明与论文架构草稿一起发送。专家返回后，优先处理：

1. 是否接受双模块及其 estimand；
2. 当前主故事在条件性成功下是否有足够分量；
3. must-have 证据是否与现有实验计划一致；
4. Section 4 的 C2 符号问题与 Pareto 合法性；
5. 专家提出的替代目录和前三个拒稿风险。

不要直接按专家建议改科学结论。每条建议先回到冻结协议、结果文件、统计单位和原论文核验，再决定接受、修改或拒绝。
