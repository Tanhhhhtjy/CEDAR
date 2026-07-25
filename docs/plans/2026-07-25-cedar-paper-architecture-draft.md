# CEDAR AAAI-27 论文架构草稿 v0.2

> 状态：供作者讨论的本地架构蓝图，不是论文正文，不同步 Overleaf，不代表任何 `TBD` 已取得结果。
>
> 日期：2026-07-25（Asia/Shanghai）  
> 修订：2026-07-26 将 Llama-3.3-70B 官方 Assistant-Axis capping 纳入预先规划的跨模型验证；第二条语义 axis 不作为主论文必需项。  
>
> 目标：先固定论文的问题、论证顺序、章节职责、证据接口和图表角色，再逐节写英文正文。
>
> 科学纪律：已有结果照实使用；探索性结果不升级；未知结果不预填方向、趋势或数字。

## 0. 这份草稿解决什么、不解决什么

这份文档只回答五个问题：

1. 论文究竟讲哪一个故事；
2. 每一节在整条论证链中承担什么职责；
3. 两个评测模块、机制实验和控制器如何衔接；
4. 每个标题级主张需要哪类证据；
5. 主文应该安排哪些图表，以及结果变化时如何诚实降级。

这份文档暂时不做以下工作：

- 不写完整英文段落；
- 不填任何尚未获得的结果；
- 不把内部 validation 写成 held-out confirmation；
- 不把概念图、期望结果或实验计划写成实证发现；
- 不决定最终标题、最终 CEDAR 全称和最终图形风格；
- 不修改或重跑 C1/C2/C3 冻结证据。

## 1. 论文的中心故事

### 1.1 一句话版本

内部工作版英文：

> Official Assistant-Axis capping reduces the incremental effect of social phrasing beyond target exposure at a direct answer boundary, yet may do so by bluntly reducing a model's responsiveness to reasons; CEDAR separates warranted revision from fallacy- and pressure-driven change, measures the resulting dose-indexed safety--epistemic trade-off, and tests whether selective capping can improve it.

中文含义：

> 官方 Assistant-Axis 钳制降低了直接答案边界上社会性措辞相对纯目标暴露的增量效应，但它可能采用了一种过于粗糙的方式——整体降低模型对“改变答案之理由”的响应；CEDAR 将有据更新、谬误诱导和社会压力诱导分开测量，刻画以剂量为索引的安全收益—认识论保留权衡，并检验选择性钳制能否改善这一权衡。

当前证据状态：第一分句已有 C2/V8 的外部行为证据，但必须使用精确的相对效应表述；“降低有据更新”仍是标题级待验证主张；机制解释和控制器效果均为 `TBD`。

若正式 CEDAR-Evidence 结果确认认识论代价，最终摘要可把 `may do so` 改成与实证强度相匹配的陈述。结果确认前不得提前改写。

### 1.2 真正的新意

论文不是再证明一次“模型会谄媚”，也不是只提出一个新的 activation intervention。核心新意是改变安全干预的评价单位：

> 不能只问模型是否改变了答案；必须同时问，促使它改变答案的理由在规范上是否足以支持改变。

由此形成两个方向相反、但必须同时衡量的错误：

- false positive revision：面对谬误、纯目标暴露或社会压力却改变；
- false negative revision：面对足以改变结论的有效证据却不改变。

传统静态能力 benchmark 很难发现第二种错误。CEDAR 要说明，一个看似保留通用能力、也确实降低某类有害漂移的干预，仍可能产生隐藏的认识论副作用。

### 1.3 论证主链

```text
Activation capping has an intended behavioral effect
    -> standard capability checks do not test responsiveness to valid new evidence
    -> CEDAR separates why the model is being asked to revise
       - CEDAR-Evidence: valid evidence vs recheck/fallacy controls
       - CEDAR-Social: social pressure vs target-exposure control
    -> compare social-side benefit with warranted-revision preservation
    -> if warranted revision falls, identify when and where the cost arises
    -> test whether a more selective intervention moves the frontier
```

故事主角始终是“隐藏认识论代价”。评测资产是测量仪器；机制回答“为什么”；控制器回答“能否降低”。机制或控制器失败时，不应倒过来摧毁已被正式行为证据支持的主发现。

## 2. 证据状态与主张防火墙

全文起草时统一使用下列内部标签；最终论文中删除这些标签，但保留相应措辞强度。

| 标签 | 含义 | 写作权限 |
|---|---|---|
| `[V]` | 已冻结并验收的正式结果 | 可写成限定范围内的事实 |
| `[E]` | 探索性、开发集或诊断结果 | 只能写为 exploratory/diagnostic |
| `[TBD-R]` | 标题级主结果尚待完成 | 只能保留占位，不得写趋势 |
| `[TBD-M]` | 机制证据尚待完成 | 不得使用 mediate/cause 等因果措辞 |
| `[TBD-C]` | 控制器/缓解结果尚待完成 | 不得声称改善 Pareto 前沿 |
| `[SRC]` | 来自前人工作的事实 | 必须有可核验引用，不能当作本项目结果 |

### 2.1 当前 claim--evidence map

| 拟议主张 | 当前状态 | 主文处理 |
|---|---|---|
| p25 改变了社会压力相对纯目标暴露的答案边界效应 | `[V]` C2/V8 | 精确报告 `cap x (S-AO)`；只描述外部行为 |
| p25 抑制模型内部“真实谄媚” | 不支持 | 禁止出现 |
| activation capping 损害有据更新 | `[TBD-R]` | Section 4 主结果槽位；当前 validation 不能代替 |
| 当前内部 Evidence validation 已支持认识论代价 | 不支持 | 不作为正面主证据；如需说明，只能作为开发阶段诊断 |
| 代价随剂量系统变化，并形成收益--保留前沿 | `[TBD-R]` | 主图槽位，等待完整 dose sweep |
| 代价由特定 phase/layer 上的方向重叠造成 | `[TBD-M]` | E0/E1 仅能放诊断边界或附录 |
| 某通路因果介导该代价 | `[TBD-M]` | 必须有双向 patching/rescue 或同等级因果证据 |
| 门控正交控制器改善前沿 | `[TBD-C]` | 未实现/未验证；只保留条件性章节 |
| 结论适用于所有模型和所有 activation axis | 不支持 | 实证范围必须限定到模型、轴、层和阈值 |

### 2.2 当前正式实证范围

- 主模型：`Qwen/Qwen3-32B`，固定 revision `9216db5781bf21249d130ec9da846c4624c16137`；
- 轴资产：官方 Assistant Axis，固定 revision `3b3b788432ad33e3a28d9ff08e88a530c0740814`；
- 主要配置：`layers_46:54-p0.25`，实际 decoder layers 46--53；
- 预先规划的跨模型验证：`meta-llama/Llama-3.3-70B-Instruct`，模型 revision 待冻结，使用同一官方资产仓库中的 Llama Assistant Axis 和 `layers_56:72-p0.25`（实际 decoder layers 56--71）；Llama 的 Social、Evidence、静态能力与剂量结果目前全部为 `[TBD-R]`；
- 算子作用于 decoder block 输出，并按运行协议覆盖相应 token 位置；
- 官方 cap vector 与 human-readable Assistant Axis 近似反向，正文必须统一符号约定；
- 当前冻结证据直接支持的是 Qwen 上的具体实例；Llama 仅是已选定的验证轨道，尚不是结果，更不是 activation capping 的普遍定律。

概念层面可以讨论 activation-axis capping 这一类方法，但每个实证句都必须回落到对应模型、axis、层范围、阈值和任务。Llama 结果只有完成并通过审计后才能写入贡献；第二条独立语义 axis 不作为主论文必需项，也不能用独立重提取同一 Assistant Axis 冒充跨语义泛化。

## 3. 拟议贡献结构

Introduction 中建议先保持三条贡献，避免把每个实验拆成一条：

1. **问题与度量。** 提出安全干预的 epistemic selectivity 问题，区分应该发生与不应该发生的答案修正，并定义 warranted-evidence gain、epistemic cost 和独立的 social-pressure benefit。
2. **评测资产。** 提出 CEDAR 评测套件：新的 `CEDAR-Evidence` 形式证据模块，以及基于外部自然题目的 `CEDAR-Social` 配对评测协议。前者是数据/benchmark 贡献；后者是经过严格 AO/S 去混杂的外部模块。二者不是一套伪装成统一六臂的实验。
3. **实证发现与分析。** 在明确的模型与干预范围内，检验并在成立时展示安全侧行为收益与有据更新保留之间的权衡；随后用层/phase、表示几何和因果干预解释该权衡，并测试选择性控制器能否改善它。主结果、机制和控制器分别保留 `[TBD-R/M/C]`。

如果控制器取得很强且稳健的改善，可在终稿把第三条拆成“现象/机制”和“缓解”两条。若控制器失败，则保持三条结构，把失败结果压缩为探索性诊断，不让 Introduction 许诺一个未实现的方法贡献。

## 4. 全文架构与篇幅预算

目标是把正文控制在约 6.2 页，为 AAAI 模板的浮动、图注和排版保留约 0.8 页缓冲。下表的页数包含本节内图表所占空间。

| 部分 | 核心问题 | 目标篇幅 | 主要视觉元素 |
|---|---|---:|---|
| Abstract | 问题、方法、主发现和边界是什么？ | 0.22 | 无 |
| 1. Introduction | 现有安全评测漏掉了什么？ | 0.78 | Figure 1 可跨栏进入首页下部 |
| 2. Background and Related Work | 哪三条文献线尚未回答此问题？ | 0.55 | 无 |
| 3. CEDAR: Measuring Epistemic Selectivity | 如何把好更新和坏更新分开测量？ | 1.10 | Figure 1 + 紧凑条件表 |
| 4. A Hidden Cost of Activation Capping | 是否存在收益--认识论代价权衡？ | 1.90 | Figure 2 主结果 |
| 5. Why Does the Cost Arise? | 行为代价在哪里形成、为何形成？ | 0.90 | Figure 3 机制图 |
| 6. Reducing the Cost | 选择性控制能否移动前沿？ | 0.40 | Figure 3/小面板或紧凑表 |
| 7. Discussion and Conclusion | 结论边界、影响和限制是什么？ | 0.35 | 无 |
| **合计** |  | **6.20** | 预留约 0.80 页 |

若 Figure 3 同时容纳机制和缓解后过密，应优先保留机制主证据；控制器结果缩成一个小面板或表。不要为了保住四张图而压字号、负 `vspace` 或牺牲可读性。

## 5. Abstract 的五句功能骨架

这不是摘要正文，只规定每句话完成什么任务。

1. **背景与缺口：** activation-space safety interventions 可改变不期望行为，但常用副作用评测主要是静态能力。
2. **问题：** 这些干预是否在抑制无依据漂移时，也抑制有效新证据触发的合理修正？
3. **方法：** CEDAR 用两个互补模块分开测量有据更新、谬误诱导、纯目标暴露和社会压力诱导，并定义互不混用的 estimands。
4. **结果：** 写入 `[V]` 的 C2 社会侧结果，以及 `[TBD-R]` 的正式认识论结果；只有结果支持时才写 trade-off。
5. **解释与意义：** 用 `[TBD-M]` 说明机制证据，用 `[TBD-C]` 说明缓解结果；结尾强调安全干预需要 reason-sensitive evaluation，而不是泛称“所有 capping 都有害”。

摘要中不出现内部代号 `C1/V7`、`C2/V8`、`E1/E2`、`Gate F` 或 `p25` 的未解释简称。

## 6. Section 1 — Introduction

### 本节职责

在不提前塞入技术细节的情况下，让审稿人接受三个前提：问题重要、现有评测看不见它、CEDAR 的对照设计能看见它。

### 建议段落顺序

**P1 — 安全干预的承诺。** 介绍 representation-level intervention/activation capping 的吸引力：部署时修改内部状态、降低某类不期望行为、无需重新训练。不要一上来介绍 CEDAR 缩写或所有实验臂。

**P2 — 被遗漏的副作用。** 指出静态能力保持不等于动态认识论能力保持。模型在收到新信息后是否应改变，取决于信息的规范效力；MMLU/IFEval 一类静态分数不能测这一点。

**P3 — 核心反例。** 并列两种外部输入：纯社会压力不应导致改答，有效反证应导致改答。一个只会“减少改答”的干预可能在第一种情形有益、在第二种情形有害。这里提出 epistemic selectivity，而不是声称模型具有人类式内部信念。

**P4 — 方法概览。** 引入两个互补模块：

- `CEDAR-Evidence (N/R/V/F)` 给出可由形式证明或 countermodel 判定的规范 gold；
- `CEDAR-Social (AO/S)` 在自然 MCQ 上用纯目标暴露控制社会压力增量。

说明两个模块分别估计认识论侧与社会侧效应，最后在收益--保留平面上比较；不要说“六臂统一 benchmark”。

**P5 — 发现概览。** 第一小句写已有 C2 结果的精确限定表述；第二小句留 `[TBD-R: Cost_V and dose-response]`；第三小句留 `[TBD-M: mechanism]`；第四小句按控制器结果决定是否写 `[TBD-C]`。不得用期待趋势填充占位。

**P6 — 贡献列表。** 使用第 3 节的三条贡献，并在每条中标明实证范围是具体模型/轴/配置。

### 本节禁止事项

- 不把 target-following 称为“真实谄媚”；
- 不把有效证据下的输出改变等同于内部信念状态改变；
- 不声称“通用能力无损”除非本项目自己的匹配评测和 CI 支持；
- 不把前人 Assistant Axis 论文中的结果写成 CEDAR 自己的复现结果；
- 不在 Introduction 许诺第二模型、第二 axis 或控制器成功，除非已经完成。

## 7. Section 2 — Background and Related Work

### 本节职责

这节不是八篇论文的逐篇摘要，而是建立一个三线交叉的空白：每条线解决了一部分问题，但没有联合回答“内部安全干预是否过度矫正合理更新”。

### 2.1 Activation-space control and capping

组织逻辑：CAA/ITI 等工作说明内部表示可被部署时干预；Refusal direction 工作提供双向因果验证标准；Assistant Axis 提供本文使用的 capping 实例。随后指出既有副作用检查多依赖远端静态能力或总体质量，不直接检验有效新证据下的修正能力。

必须区分 addition、ablation、steering 与 threshold capping，不能把它们当同一个算子。

### 2.2 Sycophancy, social influence, and overridden truth

组织逻辑：Sycophancy 与 *When Truth Is Overridden* 说明模型会受错误意见或社会输入影响，并展示“行为现象 -> 层级定位 -> 表示差异 -> 因果干预”的机制范式。CEDAR 的问题不同：不是再次解释模型为什么被错误意见带跑，而是检验阻止这一现象的干预是否同时阻止合理修正。

### 2.3 Belief revision and safety side-effect evaluation

组织逻辑：Belief-R 提供 update/maintain 的双向规范视角，但不研究 representation-level safety intervention；WMDP/HarmBench 展示安全收益必须和邻近能力、副作用及标准化评测共同报告，但其任务不直接测动态有据更新。

### 2.4 收束句

用一句明确差异结束：

> Prior work asks whether models can be steered, whether they yield to users, or whether they revise beliefs; CEDAR asks whether a safety-motivated internal intervention preserves the *selectivity* of revision with respect to the quality of the reason supplied.

正式写作前必须更新 2026 最近邻检索，并要求列出既有八篇清单之外的工作，尤其核验 Buchan 等与双向社会影响/修正相关的工作。无法核验全文的文献只标 conditional，不用于强 novelty claim。

## 8. Section 3 — CEDAR: Measuring Epistemic Selectivity

### 本节职责

建立可复现的测量对象，而不是提前讨论结果。审稿人读完后应明确知道：干预是什么、两个模块分别测什么、每个对照排除什么、主 estimand 如何定义、统计单位是什么。

### 3.1 Problem setup and intervention

先定义输出层面的 revision，不对模型内部心理状态作本体论承诺。给出 upper-cap 算子：

\[
h' = h-v\max(\langle h,v\rangle-\tau,0).
\]

正文只需给出模型、轴、层、阈值档和 token/phase 作用范围；哈希、revision、hook 路径和完整 telemetry 放附录/复现材料。必须解释官方 cap vector 的符号方向，避免把 human-readable Assistant Axis 与实际 cap vector 混写。

经验范围写成：本文以 Qwen3-32B 上官方 Assistant Axis capping 为主研究对象，并把它视为 activation-axis capping 的一个具体、可复现实例。除非补完独立轴和跨模型实验，不能写成全类别普遍结论。

### 3.2 CEDAR-Evidence: reasons of different validity

这是新的 benchmark/data contribution。主文介绍四类共享基础题目的配对条件：

| 条件 | 新增内容 | 规范行为 | 主要控制对象 |
|---|---|---|---|
| N — Neutral | 中性延续，无新理由 | 保持 | 自发漂移 |
| R — Recheck | 要求复核，无新信息 | 保持 | “被追问就改”的效应 |
| V — Valid evidence | 形式上足以支持目标修正的证据 | 更新 | 有据更新能力 |
| F — Fallacious evidence | 与 V 尽量匹配但逻辑上不足的论证 | 保持 | 内容具体性/说服性与有效性混淆 |

关键设计不是“V 写得更像证据”，而是由 proof graph、atomic edit certification、完整证明或 countermodel 固定规范 gold。V/F 在内容长度、具体性和表面论证结构上匹配；N/R 提供自然漂移和复核基线。

旧 Belief-R `BU-Acc/BM-Acc` 测量因 uncapped BU-Acc 过低且与答案形式混淆，已由 Amendment A2 归档，不进入 capping 主实验。论文可在附录简述测量演化，主文不能继续把 BU-Acc 当主终点。

当前 24-base/12-family Evidence validation 是内部 development validation，且结果不支持标题级认识论代价；正式主结果必须来自 `[TBD-R: prespecified confirmation set, N, family count, generation protocol]`。

### 3.3 CEDAR-Social: pressure beyond target exposure

这是独立的自然社会影响模块，不与 Evidence 四臂合并。每个 SycoBench base 构造：

- AO — Answer/target only：只暴露错误目标，不加入施动者、确信、权威或理由；
- S — Social pressure：在同一目标暴露上增加第一人称确信和重新作答要求，但不增加任务事实。

主目的不是测“模型是否跟目标”，而是估计社会措辞相对纯目标暴露的增量。Direct forced-choice margin 是当前正式主终点；free generation 是次终点，不能覆盖 direct 结论。

### 3.4 Estimands: do not collapse the modules

令 \(U_X(c)\) 表示在 Evidence 条件 \(X\) 和 capping 条件 \(c\) 下发生目标修正的概率。认识论主终点为：

\[
WEG(c)=U_V(c)-U_R(c),
\]

\[
Cost_V(c)=WEG(uncapped)-WEG(c).
\]

F 条件可独立定义 fallacy harm 及其改善：

\[
FH(c)=U_F(c)-U_R(c), \qquad
Benefit_F(c)=FH(uncapped)-FH(c).
\]

Social 模块使用 option-logit margin：

\[
SME(c)=M_S(c)-M_{AO}(c),
\]

\[
Benefit_S(c)=SME(uncapped)-SME(c).
\]

论文同时报告原始有符号交互 \(SME(c)-SME(uncapped)\) 和正向定义的 \(Benefit_S\)，不得只通过改名静默翻转符号。当前 C2 预设判定文字与实际负向交互之间存在符号口径问题；在写正式结果段前必须完成一次只涉及定义与文档的审计，不改变数据或结果。

`WEG/Cost_V`、`FH/Benefit_F`、`SME/Benefit_S` 是不同 estimands。BU-Acc、target-follow rate、continuous margin、`V-F` 和 `S-AO` 不能互相替代。

### 3.5 Statistical discipline

- Evidence 以 base/family cluster 为独立统计单位，配对重采样；重复生成不增加独立 N；
- Social 以 SycoBench base 为单位，在 domain/difficulty 内分层 bootstrap；
- 两个模块分别给 CI，不把异质样本池化成一个显著性检验；
- 主文报告 effect size、CI、有效 N、失败/截断和输出空间完整性；
- 认识论代价使用明确绑定到 `Cost_V` 的实质阈值；现有 10pp 标准在最终协议中须再次确认作用域；
- `TOST δ=0.10`、`Δ_min=0.15` 和 split/random 双地板属于三力方向可分离性判据，不能移植成 `Cost_V` 的主判据；
- 只有 uncapped 的 `WEG` 有足够动态范围时，才报告比例化 retention；否则优先报告绝对 `WEG` 与 `Cost_V`，避免不稳定分母。

## 9. Section 4 — A Hidden Cost of Activation Capping

### 本节职责

这是全文的实证中心。论证顺序必须先证明干预在目标行为侧确实产生了相关效应，再问它是否损害有据更新，最后展示二者的联合前沿。不能先放漂亮的 Pareto 图，再让读者猜横纵轴来自什么任务。

### 4.1 Intended behavioral effect: social pressure beyond exposure

使用 `[V]` C2/V8：600 bases，AO/S 与 uncapped/p25 严格配对。报告原始交互及 CI：

```text
cap x (S-AO) = -0.78625 logit
95% base-bootstrap CI = [-0.86000, -0.71104]
```

解释必须紧跟分解：p25 在 AO 和 S 中都提高 target margin，但在 S 中提高得更少，因此社会措辞相对纯目标暴露的增量下降。不能简写成“S 条件 target following 下降”，也不能称内部谄媚减少。

free-generation 的离散 SocialExcess 为 0，应作为模态差异/边界共同报告，而不是隐藏。C1/V7 可作为较弱的辅助观察，必须披露单一数据负责人审核、无双人一致率。C3 只能作为探索性连续/离散分离诊断，不能充当有效证据模块或 held-out replication。

### 4.2 Main finding: warranted revision under capping

本节填入正式 CEDAR-Evidence confirmation：

- uncapped 与各 cap 剂量下的 \(U_V, U_R, WEG\)；
- 主效应 \(Cost_V(c)\) 与配对/cluster CI；
- `[TBD-R: prespecified materiality decision]`；
- N/R 稳定性、invalid/truncation 与 output-integrity；
- 结果按题型/证据类型分层时只作预注册次要或稳健性分析。

当前内部 validation 的 V target-follow `+20.83pp`、F `+16.67pp`、V-F discrimination `+4.17pp` 且 CI 跨 0，不支持“p25 压制正确更新”。它不能被挑选性地改写成主结果趋势；如正文需要解释研究演化，应放附录并标为 internal development validation。

### 4.3 The safety--epistemic frontier

只有 4.1 和 4.2 都建立后才进入联合图：

- 横轴：`Benefit_S(c)`，即社会压力增量的减少；
- 纵轴：`WEG(c)` 或在基线稳定时使用 warranted-revision retention；
- 点：uncapped、p1、p25、p50、p75 或最终冻结剂量；
- 误差：两个模块各自的 cluster bootstrap，不伪造联合独立性；
- 标注：官方 p25 与可能的 knee point，但 knee 的选取规则必须预先固定或明确标探索性。

`Benefit_F` 可作为第三维度的小面板或颜色编码，但不要把图做成难以解释的三维散点。首要比较仍是独立的社会行为收益与有据更新保留。

### 4.4 Controls, robustness, and scope

按排除替代解释的顺序组织，不按“我们还跑了什么”堆实验：

1. N/R：排除自然漂移和仅复核效应；
2. F：区分有效性选择与对具体论证的一般响应；
3. AO：排除纯目标标签暴露；
4. output integrity：排除答案空间质量塌陷、格式失败和截断；
5. static capability `[TBD-R]`：检验主效应是否只是一般能力退化；
6. dose/layer/phase `[TBD-R]`：检验效应是否与实际干预剂量一致；
7. Llama-3.3-70B replication `[TBD-R planned]`：用官方 Llama Assistant Axis 和官方 p25 层范围重复 Social、Evidence 与静态能力边界；逐模型报告效应与 CI，不把两个模型简单合并成总体模型结论；
8. second semantic axis `[TBD-R optional]`：仅在需要支持跨 axis 构念外推时加入，不是当前主故事的必要条件。

### 本节的结果路由

| Social-side effect | `Cost_V` | 允许的主故事 |
|---|---|---|
| 支持 | 正且达到预设实质标准 | 目标故事：安全--认识论权衡 |
| 支持 | 近零/负 | capping 具有认识论选择性或低成本；原标题级“cost”主张不成立 |
| 不支持 | 正 | 发现副作用，但不能称“以安全收益换代价”；需重写定位 |
| 不支持 | 近零/负 | 当前配置下无主要 trade-off 证据；论文需根本重构 |

这张路由表是诚实解释规则，不是对结果方向的期待。

跨模型结果同样采用路由规则：Qwen 与 Llama 分别报告完整估计量和不确定性，并报告方向或效应量异质性。只有两模型均支持且协议可比时，才能写“across two model families”；不得用两模型的简单 pooled CI 推断模型总体。

## 10. Section 5 — Why Does the Cost Arise?

### 本节职责

从行为相关性推进到可定位、可干预的解释。机制证据必须分层陈述：描述性轨迹 < 关联几何 < phase/layer intervention < 双向 causal rescue/patching。

### 5.1 When does the divergence emerge?

使用独立的 validity assay 与 answer/choice assay，区分：

- 有效性判断本身是否受损；
- 模型能判断证据有效，但最终选择阈值被改变；
- thinking path/termination 是否改变；
- prefill 与 decode 哪个阶段承载主要效应。

计划采用 prefill/decode 的 2x2 干预和 layer-resolved telemetry，结果为 `[TBD-M]`。现有 E0 是 coarse aggregate observational diagnostic；E1 虽完成 token-resolved 观测，但预注册 phase/layer 交互未同时过闸，不能写成已定位后层 pre-answer 状态。

### 5.2 Is warranted revision aligned with the capped direction?

检验条目级 uncapped 更新位移与实际 cap direction/触发量的关系：

- 轴投影与后续 `Cost_V` 的相关/回归；
- 控制题目长度、基线难度、register-only readout 和实际 correction dose；
- family/item permutation null；
- 与 F、N/R 的对照几何。

这类结果即使显著，也只能写 `is associated with` 或 `is geometrically aligned with`。PCA、probe、cosine 或 KL 不能单独支持“共同因果电路”。

### 5.3 Does the representation causally mediate the cost?

因果主证据应至少包含两个方向：

1. 在会损失有据更新的状态/层进行 patching 或 ablation，能否复现损失；
2. 在 capped 条件进行相反方向的 rescue，能否恢复 `WEG`，同时不完全消除社会侧收益。

两者需在预先固定项目上相互印证，并报告对 F/N/R/AO/S 的副作用。只有达到这一层级才考虑 `causally mediates`；否则 Section 5 标题或措辞降为 `Diagnosing the Cost`。

### 5.4 J-Lens/MUSE/probes 的位置

J-Lens、MUSE 或线性 probe 是解释工具，不是当前已完成贡献。它们只有在能回答一个明确、且前述实验未回答的问题时使用，例如“有效性已可读出但选择阶段未利用”。建议作为支持性小面板或附录，而不是为了显得机制丰富而单独占主节。J-Lens 的语言化结果必须通过稳定性和因果干预验证，不能把可读标签当作真实内部信念。

## 11. Section 6 — Reducing the Cost

### 本节职责

证明机制理解是否能转化为更好的干预，而不是再提出一个未经验证的新名字。该节对主故事是条件性加分项。

### 6.1 Controller ladder

建议保持清晰的三层对照：

- **L0 — original capping：** 官方 Assistant Axis threshold cap；
- **L1 — orthogonalized actuation：** 在保护有效更新子空间的前提下修改实际发力方向，但保留原始触发逻辑；
- **L2 — gated orthogonal capping：** 触发与发力都只使用保护子空间之外的分量。

保护子空间只能从 V 的有效更新运动构造，不能把所有内容驱动更新混在一起，否则可能主动保护 F 条件下的谬误易感性。L1/L2 的精确定义、训练/校准数据隔离和代码状态在实现前均为 `[TBD-C]`。

### 6.2 Evaluation criterion

控制器必须在匹配的社会侧收益或实际 correction dose 下比较：

- `WEG`/warranted-revision retention；
- `FH`/fallacy susceptibility；
- `Benefit_S`/social-side effect；
- static capability、output integrity 和计算成本。

成功标准是 Pareto 前沿外移，而不是只找到一个 WEG 更高、但社会侧收益更低的点。剂量匹配规则必须在看结果前冻结。

### 6.3 结果路由

- L2 稳健外移前沿：保留主文 Section 6，并将缓解写入贡献；
- 只在单一剂量改善或 CI 宽：写探索性 proof of concept；
- 不改善但机制证据强：报告 readable-but-not-writable 或保护/控制冲突，压缩正文；
- 未实现或验证：删除任何方法贡献措辞，仅在 Discussion 作为未来工作。

## 12. Section 7 — Discussion and Conclusion

### 本节职责

收束概念意义和证据边界，不重复结果表。

建议按四个短段落组织：

1. **核心含义：** 安全干预评测需要检验模型对理由质量的选择性，而不只是行为频率和静态能力；
2. **适用范围：** 当前结论限定于已经完成验证的具体模型、官方 axis、threshold capping、任务和输出读数；Llama 未完成前仍按 Qwen 单模型表述；
3. **认识论边界：** 本文测的是可观察的 revision 行为，不证明模型具有与人类相同的内部信念、谄媚或认识论状态；
4. **限制与下一步：** 跨模型异质性、第二条语义 axis、更自然的开放式证据、longitudinal revision、人工标注强度、机制因果性和控制器泛化。

Conclusion 最后一句回到评价原则，而不是夸大方法：一个有效的安全干预不仅应减少错误行为，还应保留模型在获得真正理由时改变的能力。

## 13. 主图与表格设计

### Figure 1 — Why reason quality matters / CEDAR overview

**类型：** 概念图 + 实验设计图；Inkscape 主导，可编辑 SVG/PDF。

**作用：** 一张图说明问题和两个模块，不展示任何伪结果。

**建议布局：**

- 左：同一初始正确答案与同一 activation cap；
- 中上：valid evidence -> normative update；
- 中下：fallacy/social pressure -> normative maintain；
- 右：分别进入 CEDAR-Evidence 与 CEDAR-Social 的独立 estimands；
- 底部：在 `Benefit_S`--`WEG` 平面联合评价，而非把六类输入混成一个分数。

应避免大脑、锁链、盾牌等装饰性图标堆叠。重点是“同样的答案改变，不同的规范含义”。

### Table 1 — Conditions, controls, and estimands

用一个紧凑表列出 N/R/V/F 和 AO/S 的新增信息、规范响应、所排除混淆和对应指标。该表必须视觉上分成两个模块，不能给读者造成统一六臂 factorial design 的错觉。

### Figure 2 — Headline behavior and trade-off

**类型：** MATLAB 数据图 + Inkscape 组合。

**建议面板：**

- (a) AO/S 下 uncapped 与 p25 的 option-margin 分解，清楚显示交互来自哪里；
- (b) 各剂量 `WEG` 与 `Cost_V`，带 paired/cluster CI；
- (c) `Benefit_S`--`WEG` frontier，标出官方 p25；
- (d) 最关键的一个 control/robustness，而不是把所有 subgroup 塞进主图。

如果完整 dose sweep 尚不可用，不能用点间连线暗示单调曲线。没有数据的面板在草稿中直接标 `[TBD]`，不画示意趋势线。

### Figure 3 — Mechanism and mitigation

**类型：** MATLAB heatmap/scatter + Inkscape 组合。

**优先面板：**

- (a) phase x layer 的 validity/choice effect；
- (b) axis projection 或 actual correction dose 对 item-level update loss 的关系；
- (c) patching/rescue 的双向因果结果；
- (d) 仅在控制器成功时加入 L0/L1/L2 frontier shift。

若因果证据未完成，图名改为 diagnostic evidence；若控制器不强，(d) 移附录。不要把 J-Lens 文本标签与定量因果结果放成同等证据层级。

### Appendix figures/tables

- 完整 dose/layer/model/axis 敏感性；
- Evidence 数据构造、proof/countermodel oracle 与质量审核；
- C2 domain/template/difficulty 分层和完整输出完整性；
- C1/C3、Evidence internal validation、E0/E1 等探索性或边界结果；
- 所有迁移矩阵、invalid/truncation、free-vs-direct 差异；
- capping telemetry、符号方向审计和复现哈希；
- L0/L1/L2 细节与负结果。

## 14. 图形制作流程

当前建议工作流保持为 `MATLAB + Inkscape`：

1. MATLAB 从冻结分析表直接生成数据层，固定最终双栏/单栏尺寸、字体、线宽和颜色；
2. 导出可编辑 SVG 与嵌入字体 PDF，不把数据图栅格化；
3. Inkscape 负责多面板组合、标注、箭头、对齐、间距和 Figure 1 的概念图形；
4. 数据曲线和 CI 仍由 MATLAB 生成，不在 Inkscape 中手工移动数据点；
5. 保存原始 `.m`、中间 `.svg`、最终 `.svg` 和插入论文的 `.pdf`；
6. 在最终 AAAI 尺寸检查统一字体、最小 9 pt、无重叠、留白、灰度可读和字体嵌入；
7. LaTeX 中仅用普通 `includegraphics`，不依赖 `trim/clip` 修图。

Inkscape 与 Draw.io 的关系：Draw.io 更适合快速搭结构和拖拽布局；Inkscape 是更精细的矢量编辑与排版工具。对正式论文，Draw.io 可以作为早期骨架，但 MATLAB 数据层与最终视觉整合应以 Inkscape 为主。现有 Draw.io demo 只是可编辑性验证，不是正式 Figure 1 的审美基线。

## 15. 写作顺序

为避免正文被尚未完成的结果牵着走，建议按以下顺序落稿：

1. 固定本架构及术语表；
2. 写 Section 3 的任务、算子、estimands 与统计单位；
3. 写 Section 2 的差异化 related work；
4. 为 Section 4--6 建立只含小标题、结果槽位和允许措辞的英文骨架；
5. 设计 Figure 1 与 Table 1；
6. 正式结果完成后填 Section 4，再据结果决定 Section 5/6 强度；
7. 最后写 Introduction 和 Abstract，避免提前把预期当发现；
8. 做 claim--evidence 审阅、最近邻审阅、统计审阅和 AAAI 格式审阅。

## 16. 多轮审阅计划

### Round 1 — Story and novelty

- 审稿人能否用一句话复述问题？
- 最近邻是否已经回答同一问题？
- 现象、benchmark、机制、控制器的主次是否清楚？
- 是否错误泛化到所有 activation capping？

### Round 2 — Claim--evidence audit

- 每个结果句能否指向确定文件、配置、N、统计单位和 CI？
- `[V]/[E]/[TBD]` 是否被混用？
- social 与 evidence estimand 是否被偷换？
- 因果措辞是否超过证据层级？

### Round 3 — Methods and statistics

- V/F 的规范 oracle 与匹配是否足够；
- AO/S 是否只相差社会压力；
- 配对、cluster、bootstrap 和多重比较是否正确；
- 阈值与停止规则是否在正确作用域内；
- C2 符号约定是否完成审计。

### Round 4 — Adversarial AAAI review

- 至少列出 3 个可能导致 reject 的问题；
- 搜索预设文献清单之外的最近邻；
- 对 novelty、correctness、significance、clarity 分别给置信度；
- 无法访问原始材料的判断必须标 conditional；
- 先独立判断，再查看作者的叙事理由，减少 target-following 式锚定。

### Round 5 — Presentation and compliance

- 7 页正文内是否留有真实缓冲；
- 每张图在最终尺寸是否可读；
- 匿名、引用、图题、字体、PDF 和 checklist 是否符合 AAAI-27 Author Kit；
- 所有图的 editable source、脚本和 provenance 是否完整。

## 17. 需要作者共同决策、但不阻碍架构起草的事项

1. **最终标题。** 在认识论主结果确认前，内部标题继续用 *The Epistemic Cost of Activation Capping*；终稿建议在“强结论标题”和更限定的 *When Activation Capping Suppresses Warranted Revision* 之间选择。
2. **CEDAR 全称。** `Cost of Epistemology ...` 英文不自然；推荐候选为 `Characterizing Epistemic Dynamics under Activation Restraint`，但需要作者确认。
3. **C2 符号口径。** 结果文件对负交互的解释与原协议的正向判定文字不一致。必须做定义审计并在论文中展示分解，不能静默改阈值或重跑数据。
4. **最终实证外推范围。** 已决定补充 Llama-3.3-70B 官方 Assistant-Axis capping 作为跨模型验证；完成前标题/摘要仍按 Qwen 单模型实例表述。第二条独立语义 axis 不作为必需项，除非后续明确把实证主张扩展到 Assistant Axis 之外。
5. **Section 6 权重。** 控制器强则保留独立章节；弱或未完成则压缩到 Discussion/Appendix，不改变现象发现优先的结构。
6. **数据集命名。** 建议把新形式证据资产固定称为 `CEDAR-Evidence`，把 AO/S 自然模块称为 `CEDAR-Social`；需要确认是否与最终公开仓库目录和许可证命名一致。

## 18. 当前推荐的最终骨架

```text
Abstract
1. Introduction
2. Background and Related Work
   2.1 Activation-Space Control and Capping
   2.2 Sycophancy, Social Influence, and Overridden Truth
   2.3 Belief Revision and Safety Side-Effect Evaluation
3. CEDAR: Measuring Epistemic Selectivity
   3.1 Problem Setup and Activation Capping
   3.2 CEDAR-Evidence: Normatively Grounded Revision
   3.3 CEDAR-Social: Pressure Beyond Target Exposure
   3.4 Estimands and Statistical Protocol
4. A Hidden Cost of Activation Capping
   4.1 The Intended Social-Side Effect
   4.2 Warranted Revision under Capping
   4.3 The Safety--Epistemic Frontier
   4.4 Controls, Robustness, and Scope
5. Why Does the Cost Arise?
   5.1 Phase and Layer Localization
   5.2 Alignment with the Capped Direction
   5.3 Causal Patching and Rescue
6. Reducing the Cost
   6.1 Selective Capping Controllers
   6.2 Matched-Benefit Pareto Evaluation
7. Discussion and Conclusion
```

这套结构的关键不是章节名字，而是证据顺序：先证明目标侧行为效应，再建立有据更新代价，再谈联合前沿；机制必须晚于现象，控制器必须晚于机制。这样即使后两部分没有达到最强预期，前面的现象发现和测量贡献仍有独立、可审查的科学含义。
