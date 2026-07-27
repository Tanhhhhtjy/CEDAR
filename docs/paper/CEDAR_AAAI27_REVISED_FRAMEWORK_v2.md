# CEDAR AAAI-27 新版论文框架 v2.1（审计修订版）

> **首选标题：An Axis Is Not a Selective Controller: Auditing the Epistemic Effects of Assistant-Axis Capping**
>
> 备选标题：**Capping the Assistant, Shifting the Answer: Epistemic Non-Selectivity in Activation-Space Control**
>
> 论文定位：以 CEDAR 为核心测量框架、以 Assistant-Axis capping 为审计对象的实证评测与机制诊断论文。本文不是以提出新 controller 为成立前提的方法论文。
>
> 当前主模型：Qwen3-32B。跨模型验证采用**解码制度匹配**的 Qwen3-32B forced-choice anchor 与 Llama-3.3-70B-Instruct；两个模型分别使用原作者公开的模型专属 Assistant Axis 与官方 p25 配置。

---

## 0. 这份框架如何使用

本文件替代 `CEDAR_AAAI27_REVISED_FRAMEWORK_v1(1).md` 中所有已经被新结果更新的条件分支，尤其包括：

- E2B 不再是待定项：fixed-trace cache-faithful replay 已成功；
- boundary account 已完成，结果为 Gate B，不能作为统一解释的 headline；
- RG1-P Stage A 已形成冻结实验方案，结果记为 `[TBD-RG1]`；
- Llama-3.3-70B 最小跨模型验证已明确，且必须同时运行 regime-matched Qwen forced-choice anchor，结果记为 `[TBD-QWEN-FC]` 与 `[TBD-LLAMA]`；
- E2C 分臂离散结果已从结果提交 `ca729de0…` 的冻结分析文件核验：V `−12.5pp`、F `+16.7pp`、V−F `−29.17pp`；
- activation patching 仅在完整论文 PDF 通过写作闸门后才允许启动；
- 不再启动 MUSE、J-Lens、SAE、shown/hidden 或新 controller 开发。

本文不再讲：

> 一种已经被我们独立验证有效的安全 capping，以固定的正确更新损失换取安全收益。

原因是原论文没有公开或完整保留关键 persona jailbreak prompts，严格等价的安全收益复现不可执行。我们基于公开资源构建的替代攻击未通过预注册攻击有效性门槛。因此：

- 不能说原文安全收益被推翻；
- 不能说我们完成了等价条件下的失败复现；
- 不能以“已验证安全收益”为本文前提；
- 可以审计该官方算子在原文未测试的认识论维度上的行为。

新版主线是：

> **Assistant-Axis capping 会系统性改变模型放弃当前答案的连续几何，但这种变化不可靠地区分“为什么应该改答”。同一个算子可以推动正确更新、错误更新或只产生潜伏的 margin 位移。固定可见推理轨迹后，该位移仍然存在，说明改变写出的 reasoning 并不是该效应的必要条件。**

写作纪律：

1. 论文正文始终采用研究已经完整执行的成稿口吻；缺失的配置、数值、分析或依赖结果的结论只在最终语义位置写窄范围 `[TBD]`，不写“计划进行”“尚未完成”或“等待冻结”。内部框架和 claim–evidence 台账仍如实记录真实进度。
2. 正文先陈述发现，不把每一条限制重复插入结果句。
3. development、new-salt、external、controlled replay 的证据身份集中在方法和限制中披露。
4. “认识论代价”在本文中具体指**选择性损失或理由不敏感的决策扰动**，不是笼统的“模型变笨”。
5. `[TBD-RG1]`、`[TBD-LLAMA]` 与 `[TBD-GEMMA]` 只作为固定插槽；无论结果如何，不重写全文主线。
6. 主张强度由 claim–evidence 台账控制，不用形容词弥补证据缺口。

---

## 1. 论文类型与逻辑骨架

### 1.1 Paper type

- 类型：**New Problem/Setting + Empirical Evaluation**
- 不是：纯 benchmark、纯 mechanism、纯方法或 safety-benefit trade-off 论文。
- 核心新问题：激活控制器是否保留模型对“改答理由”的敏感性。

### 1.2 Thinking template

| 阶段 | 内容 |
|---|---|
| Research background | 激活 steering 与 capping 可在不重新训练模型的情况下改变人格、安全行为和输出倾向，但通常只按目标行为是否下降评价。 |
| Limitation 1 | 行为抑制指标不区分有效证据、错误论证、社会压力和纯 target exposure。 |
| Limitation 2 | 表示方向常被直接当作控制策略；方向可读出某种状态，不代表沿该方向干预会得到选择性行为。 |
| Limitation 3 | 离散准确率会隐藏连续 logit 几何变化，也可能把边界附近的偶然兑现误写为普遍能力变化。 |
| Our goal | 建立 reason-sensitive 的控制器审计，并检验 official Assistant-Axis capping 是否保留认识论选择性。 |
| Challenge 1 | 必须在各自的配对模块内固定 base、initial answer 与 target，同时保持 Evidence 的客观 gold 和 Social 的 target-exposure 对照干净。 |
| Challenge 2 | 必须联合测量离散迁移、连续 margin、输出完整性和干预暴露。 |
| Challenge 3 | 必须区分 reasoning 轨迹变化、一般答案几何位移和更深层的表示传播。 |
| Module A | CEDAR-Evidence：solver-certified 的 N/R/V/F 配对评测数据集。 |
| Module B | CEDAR-Social：在外部自然任务上以 AO/S 配对隔离 target exposure 与 social phrasing 的评测协议。 |
| Module C | 四态行为与连续 margin 读出、family-clustered inference、phase ablation、E2B fixed trace、RG1-P propagation 与条件性 patching。 |
| Contribution 1 | 将 epistemic selectivity 形式化为激活控制器的独立评测目标，并以 CEDAR 框架、CEDAR-Evidence 数据集和统计上独立的 CEDAR-Social 协议将其操作化（§3）。 |
| Contribution 2 | 审计官方 Assistant-Axis p25 capping，揭示 initial-answer advantage 的广泛压缩及其不可靠的理由选择性；完成后的跨模型结果只作为本项的 transportability 证据（§4）。 |
| Contribution 3 | 用 fixed-trace replay 排除可见 reasoning 改写的必要性，并以分阶段干预和条件性 RG1-P 分析对位移进行受控诊断（§5）。 |

### 1.3 自洽性检查

- Limitations → Goal：通过。
- Goal → Challenges：通过。
- Challenges → Modules：通过。
- Modules → Contributions：通过。

---

## 2. 一句话主线与研究问题

### 面向评审的一句话

> **Capping changes the geometry of answer revision without reliably tracking whether the offered reason warrants revision.**

### 面向普通读者的一句话

> 它不是在教模型分辨真证据和歪理，而是在改变模型放弃当前答案的内部条件。

### 最有记忆点的概念

> **A representation axis is not automatically a selective control policy.**

### RQ1：算子改变的是 target，还是一般答案几何？

检验：

- `m_general`；
- `m_target`；
- `m_background`；
- `m_specific`；
- initial / target / other / invalid 四态迁移。

### RQ2：变化是否尊重理由有效性？

检验：

- V warranted update；
- F unwarranted update；
- V−F selectivity；
- AO/S SocialExcess；
- N/R stability。

### RQ3：连续变化为何能产生不同离散后果？

检验：

- E2A 自然生成；
- E2C new-salt；
- E2B fixed-trace replay；
- boundary Gate B；
- margin–choice dissociation。

### RQ4：效应是否跨官方模型配置迁移？

用 Llama-3.3-70B-Instruct 的官方 axis 与 `layers_56:72-p0.25`，复用同一冻结 CEDAR bank，检验：

- `Δm_general` 是否同向；
- V/F selectivity 是否仍未改善；
- N/R 是否存在潜伏位移；
- 效应是否只是 Qwen 特有。

---

## 3. 核心贡献

### C1：Epistemic selectivity、CEDAR 框架与评测资产

控制器不应只减少目标行为，还应同时：

- 接受有效证据；
- 抵抗无效论证；
- 隔离社会压力和纯 target exposure；
- 避免拒答、输出损坏和答案冻结制造的表面收益。

CEDAR 是 reason-sensitive activation-control audit framework，而不是数据集名称。它通过两个互补但统计上独立、并非统一六臂析因设计的模块落实这些要求：

- `CEDAR-Evidence`：新的 solver-certified paired evaluation dataset，在各自共享的 base、initial answer 和 target 上比较 N/R/V/F；
- `CEDAR-Social`：基于 SycoBench-600 的独立 paired evaluation protocol，在自然任务中用 AO/S 隔离 social phrasing 与纯 target exposure。

论文身份是 **New Problem/Setting + Empirical Evaluation**；数据集是关键测量资产，但不是整篇论文的身份。

### C2：限定范围的核心实证发现

E2A 中 full p25：

\[
\Delta m_{\mathrm{general}}=3.130,
\qquad
95\%\ \mathrm{CI}=[1.834,4.486].
\]

该位移出现在 N/R/V/F；N/R 的离散答案保持稳定，说明连续变化可以潜伏而不立即改答。与此同时，V/F 的离散迁移没有表现出跨任务稳定的理由选择性：

E2A：

- V target-follow：`+20.8pp`；
- F target-follow：`+16.7pp`；
- CommonUpdate：`+18.75pp`，CI `[+4.17,+33.33]`；
- V−F selectivity 变化：`+4.17pp`，CI `[-20.83,+29.17]`。

E2C new-salt：

- complete-case `Δm_general=+1.538`，CI `[+0.455,+2.741]`；
- V target-follow `−12.5pp`；
- F target-follow `+16.7pp`；
- V−F selectivity `−29.17pp`，CI `[-58.33,+4.17]`；
- 1 条主 logit 缺失使正式 Gate 技术性未决，但 tipping-point 分析表明单个缺失值很难实质推翻总体方向。

因此稳定对象是总体几何位移，具体选项分配和离散后果具有任务依赖性。Llama/Gemma 等完成并通过审计的跨模型结果并入本项，用于限定或增强 transportability，不单列为概念贡献。

### C3：固定推理轨迹下的受控诊断

E2B：

- cache-faithful replay：192/192；
- formal records：384/384；
- uncapped-source trace：
  - `Δm_general=+2.078`，CI `[+1.759,+2.408]`；
- p25-source trace：
  - `Δm_general=+0.700`，CI `[+0.487,+0.907]`；
- 逐题 option argmax 翻转：`0/192`。

这说明：

> 改变可见 reasoning 内容或长度不是连续 margin 位移的必要条件。

但它不证明效应只发生在最终答案层，也不构成完整中介分析。

---

## 4. AAAI 七页正文结构

| 部分 | 页数目标 | 功能 | 图表 |
|---|---:|---|---|
| 1. Introduction | 0.70 | 漏洞、直观例子、发现、贡献 | Figure 1（单栏半高） |
| 2. Background and Related Work | 0.45 | 定位最近工作与缺口 | 无 |
| 3. CEDAR | 0.90 | framework、Evidence dataset、Social protocol、指标与统计 | 无（完整方法契约表进附录） |
| 4. Epistemic Non-Selectivity | 1.25 | E2A/E2C、社会、跨模型泛化 | Figure 2 |
| 5. Diagnosing the Shift | 1.00 | phase、E1、E2B；RG1-P视结果路由 | Figure 3 |
| 6. Implications, Scope, and Future Control | 0.40 | 设计原则、限制、未来方向 | 无 |
| 7. Conclusion | 0.15 | 一段结论 | 无 |
| 主图与caption预留 | 1.55 | Figure 1–3 | — |
| 浮动与排版缓冲 | 0.50 | 防止最终溢出七页 | — |

总目标 `7.00` 页，但文字和图表初排必须控制在约 `6.5` 页，保留约 `0.5` 页排版缓冲。正文上限：

- 3 张主图；
- 正文暂不设置与 Figure 1 重复的主表；完整方法契约表进入附录；
- Llama 完成后优先合并进 Figure 2，不新增第四张大图；
- 完整 assay matrix、协议偏离、随机方向 battery 和扩展表进入附录。

---

# 1 Introduction

## 段落1：从“让行为符合人类期望”切入

应补入：

- post-training、alignment、activation steering/capping 的目标都是使模型行为更稳定、更符合人类期望；
- 与重新训练相比，表示层干预成本低、可即时部署；
- 但“改变了目标行为”不等于“保留了模型进行可靠判断的方式”。

建议逻辑：

> Modern alignment methods increasingly intervene not only on model weights but also on runtime representations. Activation steering and threshold capping can suppress refusal failures, persona drift, or unsafe behavior without retraining. Yet their success is usually judged by whether an unwanted behavior decreases, leaving open whether the intervention preserves the model’s sensitivity to why its answer should change.

## 段落2：同一target、不同理由

用一个 MCQ base：

- 模型最初选 A；
- V 用有效证据支持 B；
- F 用错误论证支持 B；
- AO 只提到 B；
- S 要求模型同意 B。

只看“是否从 A 改成 B”无法区分：

- 正确更新；
- 易受骗；
- 谄媚；
- target priming。

## 段落3：提出CEDAR

CEDAR 固定：

- 原问题；
- initial answer；
- target；
- 答案位置；
- 基础结构。

只改变支持 target 的信息性质，并联合使用 solver gold、连续 margin、离散四态和 telemetry。

## 段落4：主结果

正文数字锚点：

- E2A `Δm_general=+3.13`；
- V `+20.8pp`、F `+16.7pp`；
- selectivity 无可靠改善；
- N/R 连续移动但答案不变。

## 段落5：机制与贡献

加入 E2B：

> The margin shift persists when the visible reasoning trajectory is held token-for-token fixed, indicating that changes to written reasoning are not necessary for the effect.

若 `[TBD-RG1]` Route A/C 成立，再增加一句局部单轴修正向下游答案空间传播的描述。

贡献列表固定为三条：框架与评测资产、限定范围的实证发现、受控诊断。Llama/Gemma 只有在完成并通过审计后才能并入第二条，作为跨模型 transportability 证据，不单列贡献。

---

# 2 Background and Related Work

正文直接采用 `CEDAR_RELATED_WORK_AAAI27_v2.md` 的三段递进结构，并压缩至约半页：

## 2.1 Activation steering and capping are different operators

先连接 representation engineering、CAA、persona vectors、Assistant Axis、threshold capping 与条件性 steering。重点不是罗列工具，而是指出：持续加法、阈值整流和条件门控即使使用同一表示方向，也定义了不同算子，不能根据轴的语义直接推断行为后果。

> Existing work primarily asks whether an intervention moves or suppresses a target behavior. CEDAR asks whether the intervention preserves reason-sensitive revision.

## 2.2 Existing selectivity is usually defined over inputs, not reasons

再连接 SycophancyEval、SycoBench、PhilPapers、Playing Devil’s Advocate、When Truth Is Overridden、LoFa 与 LOGICOM。明确区分用户意见、社会压力、纯 target exposure、有效证据与无效论证。Wang et al. 比较的是 plain/opinion prompts；CEDAR 在同一 base 和 target 下比较 uncapped/capped interventions，因此研究的是 controller 是否保留理由选择性。

## 2.3 Solver certification makes reason quality experimentally identifiable

最后连接 Belief-R、ReviseQA、ProverGen、BASIL/贝叶斯更新与 BeliefTrack。现有 belief-revision 数据很少审计 activation controller，而 controller 研究又很少用 solver-certified 的 V/F 对照。CEDAR 用唯一 gold、单一致命错误与 countermodel 让“理由是否足够”成为可复算变量。

正文不展开 MUSE/J-Lens/SAE：它们属于可能的后续解释工具，不是本稿已经使用的证据。

---

# 3 CEDAR: Auditing Epistemic Selectivity

## 3.1 Epistemic selectivity and controlled contrasts

### 方法契约表（当前移至附录；仅在 Figure 1 无法承载模块边界时再考虑回正文）

标题：

> **CEDAR separates revision propensity from reason validity.**

| Module | Arm | 输入变化 | 规范行为 | 主要构念 |
|---|---|---|---|---|
| CEDAR-Evidence dataset | N | 中性继续 | initial | 自发漂移/输出损坏 |
| CEDAR-Evidence dataset | R | 无新信息复核 | initial | 复核诱导 |
| CEDAR-Evidence dataset | V | solver认证有效修改 | target | warranted revision |
| CEDAR-Evidence dataset | F | 单一致命错误 | initial | fallacious persuasion |
| CEDAR-Social protocol | AO | 无施动者target提及 | initial | target exposure |
| CEDAR-Social protocol | S | AO上增加社会压力 | initial | social excess |

Figure 1 和附录方法表必须用视觉分隔强调两个模块不是统一六臂数据集：N/R/V/F 在 CEDAR-Evidence 内配对，AO/S 在 CEDAR-Social 内配对；两个模块不共享样本、统计单位或联合显著性检验。完整 schema 和字段进入附录。正文暂不保留与 Figure 1 重复的 Table 1。

## 3.2 CEDAR-Evidence: A solver-certified paired dataset

必须明确工作量：

1. ProverGen-style 初始证明图；
2. ReviseQA-style 原子邻域修改；
3. V 后 target 是唯一 gold；
4. F 后 initial 仍是唯一 gold；
5. F 只有一个可定位错误；
6. 每条 F 带 countermodel；
7. easy/medium 由证明深度、规则数和干扰事实控制；
8. deterministic rebuild、leakage audit、人工自然化检查。

## 3.3 CEDAR-Social: Controlling for target exposure

必须明确：

1. `CEDAR-Social` 是在固定版本 SycoBench-600 上构造 AO/S 配对的 evaluation protocol，而不是新的原生数据集或 CEDAR-Evidence 的 split；
2. AO 与 S 对每个自然 MCQ 共享 base、initial answer 和 target；唯一关键差别是社会性措辞；
3. 正文主动披露来源、版本、许可、派生规则、模板平衡、人工审核和 base-level bootstrap；
4. 它与 CEDAR-Evidence 互补，但 estimand、量纲与独立统计单位保持分离。

## 3.4 Intervention and models

### Qwen

- `Qwen/Qwen3-32B`；
- pinned revision；
- official Assistant Axis；
- `layers_46:54-p0.25`，实际46–53；
- phase：prefill / decode / full；
- thinking和decoding按assay冻结。

### Regime-matched Qwen anchor

跨模型比较不能直接拿 Qwen thinking/free generation 与 Llama no-reasoning/forced-choice 相比。为此必须增加：

- 与 Llama 完全相同的 96 个 N/R/V/F prompts；
- `reasoning off`；
- deterministic forced-choice；
- uncapped / official Qwen p25；
- 共 `96×2=192` 次 forward；
- 不修改题目、prompt、answer order 或 Qwen p25 参数。

该 anchor 使 `Qwen(FC) vs Llama(FC)` 的主要变化只剩模型和模型专属官方配置。`Qwen(FC) vs Qwen(thinking)` 只能作为 regime sensitivity 的描述性对照；除非两套 prompt、模板和读出完全一致，否则不写成严格因果交互。

### Llama最小泛化

- `meta-llama/Llama-3.3-70B-Instruct`；
- pinned revision `[TBD-LLAMA-REVISION]`；
- 原作者公开的Llama专属axis；
- official `layers_56:72-p0.25`，实际56–71；
- 不迁移Qwen axis、层位或threshold；
- 不重新调参；
- reasoning off / deterministic forced-choice；
- 同一冻结24-base、12-family N/R/V/F bank。

官方仓库明确列出 Llama-3.3-70B-Instruct 的预计算 axis 与 capping 配置，因此这属于同一官方算子的跨模型审计，不是临时重建轴。

## 3.5 Behavioral and continuous estimands

离散四态：

1. initial；
2. induced target；
3. other wrong；
4. invalid/abstain。

连续指标：

\[
m_{\mathrm{general}}
=
\frac{1}{3}\sum_{a\neq a_0}\ell(a)-\ell(a_0)
\]

\[
m_{\mathrm{target}}=\ell(a_t)-\ell(a_0)
\]

\[
m_{\mathrm{background}}
=
\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\ell(a)-\ell(a_0)
\]

\[
m_{\mathrm{specific}}
=
\ell(a_t)
-
\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\ell(a).
\]

正的 `Δm_general` 表示 initial-answer advantage 被压缩。

## 3.6 Statistical discipline

正文一句：

> All uncertainty is clustered at the independently generated family level; full estimands, bootstrap procedures, missing-data sensitivity, and protocol deviations are reported in Appendix C.

具体：

- family-cluster bootstrap；
- hierarchical models；
- ITT保留截断/无效；
- 连续和离散结果联合报告；
- development / new-salt / external / controlled replay 分层；
- 不按模型响应删题。

样本量与统计单位必须在方法正文主动说明：

> The controlled bank contains 24 bases from 12 independently generated structural families. All contrasts are prompt-paired, and uncertainty is clustered at the family level. We trade raw item count for solver-certified gold labels, exact V/F counterfactual pairing, and replayable proof traces; a second independently salted instantiation tests whether the aggregate effect survives a new realization of the same task grammar.

`m_specific` 呈现规则预先固定为：

- 正文和 Figure 2C 只报告冻结的 pooled primary：`−0.084 [−1.110,+0.953]`；
- N/R/V/F 分臂分解属于验收后探索性重算，进入附录并明确标 `post-hoc`；
- 可在正文一句说明 N/R 的分臂值为负、并未显示 target-specific amplification，但不把该分解升级为预设主结果。

Claude 审计重算给出的候选分臂值为 N `−0.831 [−1.336,−0.307]`、R `−0.786 [−1.211,−0.331]`、V `−0.198 [−2.581,+2.240]`、F `+1.479 [−1.018,+4.049]`。这些值在进入论文前必须由仓库内确定性分析脚本复算、落盘并绑定 commit；在此之前只视为待复核的 post-hoc 候选，不视为冻结结果。

---

# 4 Epistemic Non-Selectivity under Assistant-Axis Capping

## 4.1 The axis is reproducible; the operator determines the consequence

压缩呈现 E−1a/E−1b：

- `+200` continuous steering 在 PhilPapers 上复现 `−2.4312` margin；
- 正负大剂量均改变 top-token/entropy；
- official p25 几乎处处触发；
- p25 连续 margin `−0.09375`，离散 sycophancy `+1.444pp`；
- 翻转集中在基线 margin 附近，但跨数据 boundary account 未统一成立。

结论：

> **The same axis does not imply the same intervention.**

详细曲线进入附录 D，正文只保留一段和必要数字。

## 4.2 Broad compression of initial-answer advantage

### Figure 2（正文主结果图，§4.2–§4.5共用）

采用 dot-and-whisker / paired-slope / forest plot，不使用缺乏不确定性的普通柱状图。

#### Panel A：Qwen `Δm_general` by arm

- N/R/V/F；
- 点估计和 family-bootstrap 95% CI；
- overall 用黑色菱形；
- 零线清楚；
- arm颜色与全文固定。

#### Panel B：Warranted vs unwarranted update

- V与F在uncapped/p25下的target-follow；
- 用 paired slope 或 dumbbell；
- 同时标 `ΔV`、`ΔF`、`Δ(V−F)`。

#### Panel C：General vs target-specific geometry

- `Δm_background`；
- `Δm_target`；
- pooled primary `Δm_specific`；
- 说明位移不是单纯target amplification。
- 分臂 `Δm_specific` 不放主图，进入附录并标明 post-hoc。

#### Panel D：Cross-model transportability

若 `[TBD-LLAMA]` 完成：

- matched forced-choice Qwen 与 Llama 的 within-model overall `Δm_general`；
- 两模型各自的 V−F selectivity change；
- 两个模型各自使用官方capping配置；
- forest plot，不比较未经标准化的跨模型原始 logit 水平；
- cross-model interaction 仅作二级分析，不以“一个显著、一个不显著”代替模型差异检验。

若 Llama 未完成，Panel D 改为 E2C new-salt forest，不留空白。

主图标题：

> **Assistant-Axis capping shifts revision geometry without reliable evidence selectivity.**

配色：

- V：teal `#0E8C7A`；
- F：orange-red `#DC542A`；
- N：gray `#9AA3AD`；
- R：blue-gray `#5E7C99`；
- uncapped：hollow；
- p25：filled/navy；
- Llama：同色但不同marker。

颜色可以使用，但信息不得仅由颜色承载：同时使用 marker、实心/空心或线型编码；灰度打印后仍可区分，并按 AAAI-27 要求检查 WCAG 2.0 对比度。

## 4.3 Latent in controls, consequential under revision

不再单独画流程装饰图。Figure 2A/B已表达：

- N/R：margin变化、argmax不变；
- V/F：同类连续变化可能兑现成更新。

正文用一段解释：

> N/R expose a latent shift: capping changes continuous answer geometry without changing the selected answer. V/F reveal its epistemic consequence, because the same proposed target is paired with reasons that differ in validity.

不得把 boundary Gate B 写成统一因果解释。允许：

> This pattern is consistent with thresholded realization near some decision boundaries, although the post-hoc boundary analysis was not uniform across datasets.

## 4.4 New-salt and natural-task evidence

### E2C

- 冻结结果提交：`ca729de0e6b55952266f1b32169b8fc2cd53005c`；
- overall complete-case `+1.538 [0.455,2.741]`；
- V `−12.5pp`、F `+16.7pp`、V−F `−29.17pp [-58.33,+4.17]`；
- V/F细分后果与E2A不同，但总体 margin 位移同向；
- missing-logit与tipping-point放脚注/附录；
- 明确称为“new-salt instantiation of the same task grammar”，不称 independent-family held-out confirmation。

### C2/C3

- C2：relative SocialExcess可改善，但AO/S可能同时向target移动；
- C3：连续双向selectivity结果强，但离散镜像Gate未通过；
- 共同说明连续margin不能自动解释成行为质量。

本节控制在约半栏，完整结果进附录 E。

## 4.5 Minimal Llama-3.3-70B generalization

### 目的

只回答：

> Qwen上发现的一般答案几何位移和V/F非选择性，是否也出现在另一套原作者正式支持的模型与capping配置上？

### 冻结矩阵

```text
24 bases
× N/R/V/F
= 96 prompts

Qwen matched anchor:
96 prompts × uncapped / official Qwen p25
= 192 deterministic evaluations

Llama generalization:
96 prompts × uncapped / official Llama p25
= 192 deterministic evaluations

total = 384 forwards
```

### 运行要求

- 使用原作者Llama专属axis；
- official `layers_56:72-p0.25`；
- pinned model revision；
- CPU Gate 同时验证两个模型的 A/B/C/D 均为预期单 token、answer order、chat template、axis/threshold 资产哈希与实际 cap exposure；
- 不使用Qwen参数；
- 不根据Llama输出改题；
- 不筛family；
- forced-choice logits为主；
- deterministic option argmax为离散读出；
- 记录trigger、correction、option mass、entropy、top-token；
- family=12为统计单位；
- 同一bootstrap方案和指标定义。
- 跨模型主比较使用各模型自身的 `p25−uncapped` paired effect；不直接比较两模型的 raw-logit level。

### Primary estimands

1. overall `Δm_general`；
2. N/R/V/F arm breakdown；
3. `Δm_specific`；
4. V−F continuous selectivity；
5. V/F target-follow change；
6. N/R argmax stability；
7. output integrity。

### 路由

- **G-A transport**：在 matched forced-choice regime 下，Qwen 与 Llama 的 overall `Δm_general` 同向，且 Llama CI 排除0，同时V/F选择性无可靠改善；
- **G-B partial transport**：Llama overall同向但CI跨0，或arm分配明显不同；
- **G-C model-specific**：Qwen matched anchor 保持主方向而 Llama 为零或反向；
- **G-D invalid**：tokenization、cap exposure或output space未通过。

无论结果为何都如实报告。跨模型不要求数值相等，也不把“未显著改善”写成严格等效。

若 regime-matched Qwen anchor 未完成，Llama 结果不得进入 Figure 2D、摘要或第二条实证贡献，只能作为附录中的非匹配探索性结果。

### 不做

- 不在Llama上跑完整E2B；
- 不跑RG1-P；
- 不跑五剂量；
- 不调threshold；
- 不补新数据；
- 不把Llama结果用于挑选Qwen主张。

---

# 5 Diagnosing the Shift

## 5.1 Phase-localized interventions are subadditive

结果：

- prefill-only CommonUpdate `+18.75pp`；
- decode-only `+20.83pp`；
- full `+18.75pp`；
- full没有简单叠加。

正文不单独做大图，作为 Figure 3A 的小型phase inset或一张紧凑表。

允许：

> Either phase reproduces much of the shift, while their combination is subadditive.

禁止：

> We have identified a shared downstream bottleneck.

## 5.2 Full capping does not show a simple global validity-erasure pattern

- prefill validity tendency下降；
- decode tendency上升；
- full净变化接近零；
- 区间较宽。

结论：

> A net null under full capping can conceal opposing phase-specific tendencies.

不写“推理能力完全不受损”。

## 5.3 The selective-resistance-state hypothesis is rejected

E0 late-decode相关被E1解释为轨迹长度/组成：

- resist/yield方向不符合预测；
- yield轨迹更长；
- 长度匹配后效应接近零；
- 修正量控制长度后不再预测yield。

这是一项明确的假说淘汰，正文保留一段，完整图进附录 F。

## 5.4 Fixed-trace replay: written reasoning changes are not necessary

E2B 是正文机制支柱。

### 主结果

- uncapped-source：`+2.078 [1.759,2.408]`；
- p25-source：`+0.700 [0.487,0.907]`；
- trace-source interaction：`−1.378 [−1.720,−1.084]`；
- argmax翻转：`0/192`。

uncapped-source fixed-trace 位移与 E2A 自然生成估计量级相近：`2.078/3.130≈0.66`。这只能作为描述性比较，不能解释为“可见 reasoning 只中介了三分之一”，因为两个 assay 的生成和读出条件不同。

`0/192` 是逐题配对核验后的观察结果，不是实验设计保证的结果。E2B 的主要读出是单个固定答案位置的连续 logits；它证明潜伏位移可以不越过 argmax 边界，但不能单独证明自然行为已经改变。离散行为证据由 E2A/E2C 承担。

### 分臂

| Trace | N | R | F | V |
|---|---:|---:|---:|---:|
| uncapped-source | +2.679 | +2.639 | +2.309 | +0.686 |
| p25-source | +1.144 | +1.464 | +0.415 | −0.224 |

### 选择性

- uncapped-source V−F：`−2.422 [−3.411,−1.448]`；
- p25-source V−F：`−0.797 [−1.219,−0.391]`。

必须分开解释：

- uncapped-source：主要由F target margin上升形成；
- p25-source：主要由V target margin下降形成；
- p25-source是处理后轨迹条件化，只作辅助；
- 不称完整中介。

允许：

> The continuous margin shift persists when the visible trace is held fixed, so changes to written reasoning are not necessary for the effect.

禁止：

- effect occurs only at the answer layer；
- final-layer circuit；
- natural direct effect；
- complete mediation。

## 5.5 RG1-P Stage A: from local rectification to downstream geometry

状态：`[TBD-RG1]`

### Figure 3（机制图）

页面安全版本默认只有 Panel A–B。只有 RG1-P 结果清楚且 2026-07-28 12:00 的完整 PDF 闸门已经通过，才把传播结果加入正文；否则 RG1-P 全部进入附录。

#### Panel A：E2B fixed-trace cross

- uncapped-source和p25-source；
- uncapped/full-p25 forward；
- `Δm_general` CI；
- 强调相同trace。

#### Panel B：E2B arm breakdown

- N/R/F/V；
- 两个trace source分开；
- 不平均。

#### Panel C：Axis propagation（条件加入）

若RG1-P通过：

- layers45–63；
- layers46–53用逐层官方axis；
- layers54–63使用固定layer53 exit direction；
- p25 off-exit-axis曲线；
- schedule- and norm-matched随机方向battery作为灰色地板；
- layer53、60预注册标记；
- family-level CI。

#### Panel D：条件性patching（原则上附录）

只有Stage B完成、至少一个双向patch通过且正文仍不溢页时才加入；否则 patching 进入附录，Figure 3 保持两联或三联。

### RG1-P结果路由

- Route A：p25传播超出随机地板 → 正文机制发现；
- Route B：与随机地板重叠 → 一般扰动重混合；
- Route C：旋转普通但margin符号不同 → 方向决定行为后果；
- Route D：不稳定 → 附录。

Stage A无论结果为何都不能单独声称因果中介。

---

# 6 Implications, Scope, and Future Control

## 6.1 Representation is not policy

核心设计结论：

> A direction can encode or predict a model state without implementing a policy that selectively preserves desirable behavior.

Assistant Axis原工作说明：

- persona projection可测；
-沿轴steering可改变角色倾向；
- threshold capping可限制persona drift。

CEDAR进一步表明：

- 相同算子也会改变答案修订几何；
- 它不自动区分有效证据与错误理由；
- 行为减少、relative contrast改善、连续margin变化三者不能互相替代。

## 6.2 What a better controller must preserve

未来controller至少需要联合保护：

1. persona safety；
2. validity representation；
3. warranted revision；
4. resistance to invalid arguments；
5. social-pressure isolation；
6. normal deliberation/output completion。

本文提出评测和设计规范，不声称已经完成多维controller。

## 6.3 Evidence scope

集中披露：

- Qwen是主要完整实验模型；
- 跨模型结论只来自 matched forced-choice Qwen/Llama；Qwen thinking 结果不直接充当 Llama 对照；
- persona safety没有完成等价条件下独立确认；
- E2A是development/mechanism assay；
- E2C不是全新family且有1个missing logit；
- E2B是controlled fixed-trace diagnostic，不是自然直接效应；
- boundary analysis为Gate B/post-hoc；
- C2/C3为构念互补而非同分布复制；
- RG1-P若成功仍是传播证据，patching后才能提升为块级因果定位。

不要把这些限制重复散布在每段结果后。

## 6.4 Future work

一段即可：

- held-out new families；
- 更多模型；
- persona safety × epistemic selectivity双轴验证；
- commitment shown/hidden；
- selective controller；
- activation patching / circuit localization。

---

# 7 Conclusion

建议结尾：

> Activation-space control is often evaluated by whether it suppresses an unwanted behavior. CEDAR asks a stricter question: does the intervention preserve sensitivity to why the model should change? Across solver-certified revision tasks and complementary social assays, Assistant-Axis capping altered the geometry of answer revision without reliably separating valid from invalid reasons. The shift persisted when the visible reasoning trajectory was held fixed, showing that changes to written reasoning are not necessary for the effect. These findings establish epistemic selectivity as a necessary evaluation target and show why a representation axis should not be treated as a selective controller by default. Future controllers should stabilize persona without collapsing validity, revision, and social influence into a single intervention dimension.

若Llama为G-A，可将第二句改为：

> Across two officially supported model-specific capping configurations...

若仅G-B/C，不进入结论首句，只在§4.5和限制报告。

---

## 8. 最终图表清单

## Figure 1：Why reason matters

**位置：Introduction，第1页下半部；单栏、半高。**

形式：motivation diagram。

内容：两个并列但不合并的模块。每个模块内部固定 base、initial answer 与 proposed target；不得画成 Evidence 与 Social 共享同一道 base 的统一六臂设计。

```text
CEDAR-Evidence dataset                    CEDAR-Social protocol
solver-certified base                    natural-task base
same initial A + same proposed B          same initial A + same proposed B
├─ N neutral       → retain               ├─ AO target exposure  → retain
├─ R recheck       → retain               └─ S social phrasing   → retain
├─ V valid reason  → update
└─ F single-fault  → retain
```

右侧显示：

```text
Behavior-only metric: “Did B increase?”
CEDAR-Evidence: “Did B increase when the reason warrants revision?”
CEDAR-Social: “What changes beyond exposure to B itself?”
```

Caption 必须自包含，不展示真实危险内容；末尾以语义引用占位符提示更多配对样例、solver certificates 与构造细节进入 `[TBD-APPENDIX]`，不预先锁定附录编号。

## Appendix method-contract table：CEDAR modules and arms

**位置：附录；若正式 Figure 1 无法完整承载模块边界，再评估是否回到 §3.1。**

按模块分区列出各 arm、规范行为、唯一操纵属性和读出；不得称为统一六臂数据集。

## Figure 2：Epistemic non-selectivity

**位置：§4.2–§4.5，正文最重要结果图。**

- A：Qwen `Δm_general` by arm；
- B：V/F warranted vs unwarranted update；
- C：background/target/pooled-specific geometry；
- D：matched forced-choice Qwen–Llama forest；若Llama未完成则改E2C。

## Figure 3：Fixed-trace and propagation diagnosis

**位置：§5.4–§5.5。**

- A：E2B fixed-trace cross；
- B：E2B分臂；
- C：仅当 RG1-P Route A/C 清楚且页面允许时加入；
- patching 默认进附录，不作为正文完成条件。

## Table 2：Evidence scope（可选）

**位置：§6或附录首页。**

列：

- Assay；
- model；
- construct；
- intervention；
- primary readout；
- evidence identity；
- main conclusion。

若正文拥挤，Table 2放附录，不影响理解。

## 附录图表

- Fig. A1：E−1a signed steering curve及top-token/entropy；
- Fig. A2：E−1b cap exposure与margin-choice迁移；
- Fig. A3：E2C missing-value sensitivity；
- Fig. A4：C2/C3自然社会结果；
- Fig. A5：E0/E1长度混杂与假说淘汰；
- Fig. A6：boundary Gate B完整分数据集结果；
- Fig. A7：RG1-P完整45–63层曲线和四个随机方向；
- Fig. A8：Llama trigger/correction与输出完整性；
- Table A1：完整assay matrix；
- Table A2：全部claim–evidence台账；
- Table A3：protocol deviations；
- Table A4：模型、revision、axis、layers、环境、commit。

---

## 9. 摘要草稿

> Activation-space interventions are increasingly used to suppress undesirable language-model behaviors, yet they are typically evaluated by whether a target behavior decreases—not by whether the intervention preserves sensitivity to the reasons for changing an answer. We introduce CEDAR, a reason-sensitive audit framework with two statistically separate modules: the new solver-certified CEDAR-Evidence paired dataset and a CEDAR-Social protocol over natural tasks. Within each module, CEDAR holds the base, initial answer, and proposed target fixed while varying reason validity or social phrasing. On Qwen3-32B, official Assistant-Axis p25 capping shifted non-initial option margins relative to the model’s initial answer (`+3.13` logits; 95% family-cluster CI `[+1.83,+4.49]`). In the development bank, warranted and unwarranted updating moved together (`+20.8` and `+16.7` points); in a new-salt instantiation, the aggregate margin shift replicated (`+1.54` logits) while valid and invalid updates redistributed in opposite directions (`−12.5` and `+16.7` points). Thus, the stable object was the continuous geometry rather than a fixed behavioral direction. The shift was latent in neutral controls and persisted under token-for-token fixed reasoning replay (`+2.08` logits on treatment-naive traces), where no paired argmax migrations were observed. `[TBD-CROSS-MODEL: one sentence only if matched forced-choice Qwen/Llama/Gemma results provide interpretable transport evidence.]` These results identify epistemic non-selectivity as a failure mode of activation-space control: a representation axis can alter model state without constituting a reason-sensitive policy.

若RG1-P Route A/C强，可用一句替换Llama句，而不是把所有新结果都塞入摘要。

---

## 10. Claim–Evidence 台账

| ID | 正文主张 | 证据 | 状态 | 禁止升级 |
|---|---|---|---|---|
| C1 | p25压缩initial-answer advantage | E2A `+3.13` | 强支持于Qwen assay | 所有任务/所有模型 |
| C2 | pooled结果不支持纯target amplification | E2A pooled specific `−0.084 [−1.110,+0.953]`；分臂post-hoc进附录 | 支持 | 所有arm严格等效 |
| C3 | V/F选择性无可靠改善 | E2A V/F同升 | 支持 | 证明变化为0 |
| C4 | 连续位移可无离散改答 | E2A N/R；E2B 0/192 flips | 强支持 | 行为完全不受影响 |
| C5 | new-salt总体方向同向、细分异质 | E2C `+1.538` | 支持，技术未决 | held-out family confirmation |
| C6 | margin变化不等同于行为改善 | E−1b、C2、E2A/E2C | 强支持 | 统一boundary机制 |
| C7 | prefill/decode非简单相加 | E2A phase | 支持于当前assay | 已定位共享瓶颈 |
| C8 | 简单resistance-state假说不成立 | E1长度匹配 | 强反证 | 所有状态门控假说 |
| C9 | boundary account跨数据统一成立 | boundary Gate B | 不支持强主张 | reconciles all datasets |
| C10 | fixed visible trace下位移仍存在 | E2B U/P | 强支持于controlled replay | natural direct effect |
| C11 | E2B选择性损失在两轨迹存在 | U `−2.422`；P `−0.797` | 支持，机制分解不同 | 两轨迹同一机制 |
| C12 | 单轴修正被下游重混合 | RG1-P | `[TBD-RG1]` | 无随机地板的强机制结论 |
| C13 | 效应跨Llama迁移 | matched Qwen-FC/Llama-FC minimal | `[TBD-QWEN-FC/TBD-LLAMA]` | 跨所有模型/所有regime |
| C14 | 某block因果承载位移 | bidirectional patching | `[TBD-CONDITIONAL]` | 未做patch时声称因果 |

---

## 11. 附录结构

### A. Preregistration, amendments, and unavailable resources

- M2.1资源不可得；
- 替代攻击与过闸失败；
- development/new-salt/external身份；
- 单人审核；
- missing/truncation/ITT；
- E2B工程修正。

### B. Dataset construction

- generator feasibility；
- solver traces；
- countermodels；
- naturalization；
- leakage；
- hashes；
- blind review。

### C. Statistics and sensitivity

- family bootstrap；
- hierarchical models；
- LOFO；
- missing tipping point；
- multiplicity；
- exact paired migrations。
- E2A `m_specific` 分臂 post-hoc 分解；

### D. Instrument and operator calibration

- E−1a；
- E−1b；
- top-token/entropy；
- trigger/exposure。

### E. External and social assays

- V7/V7.1；
- C2；
- C3；
- AO/S pairing。

### F. Mechanism diagnostics

- E0；
- E1；
- E2A phase；
- E2B；
- boundary Gate B；
- RG1-P；
- patching（如有）。

### G. Cross-model generalization

- matched Qwen forced-choice anchor；
- Llama CPU/smoke gates；
- model revision；
- tokenization；
- official axis/config；
- exposure；
- complete estimates；
- failure routes。

### H. Reproducibility

- hardware；
- software；
- entry points；
- commits；
- manifests；
- artifact hashes。

---

## 12. TODO列表

## P0：立即并行，决定能否交稿

- [ ] 将本文框架同步到 Overleaf 的章节骨架。
- [ ] 完成 Abstract、Introduction、Related Work、Conclusion 初稿。
- [ ] 用 `CEDAR_RELATED_WORK_AAAI27_v2.md` 替换旧 Related Work 骨架并压缩到约半页。
- [ ] 将 `references.bib` 从172字节占位文件扩展为可编译的真实文献库。
- [ ] 恢复并验证 `\bibliography{references}`。
- [ ] 完成 Figure 1。
- [ ] 用现有E2A/E2B结果完成 Figure 2A–C 和 Figure 3A–B，不等待新实验。
- [ ] 把 E2B `0/192` paired argmax flips 写入 claim register。
- [ ] 把 boundary Gate B 的措辞改为“consistent in some assays, not uniform”。
- [ ] 把 E2B 的 cache-faithful replay amendment 写入方法/附录。
- [x] 从冻结 E2C JSON 核验 V `−12.5pp`、F `+16.7pp`、V−F `−29.17pp`。
- [ ] 将 E2C 结果提交 `ca729de0…` 与论文引用的所有 artifacts 收敛到同一可追溯分支。
- [ ] 在方法中固定 `m_specific`：正文 pooled primary，分臂 post-hoc 进附录。
- [ ] 用仓库内确定性脚本复算 Claude 提供的 E2A 分臂 `m_specific`，保存 JSON、bootstrap 配置、哈希与 commit；未复算前不落笔。
- [ ] 在 §3 末尾加入 24 bases / 12 families、paired design 与 family-cluster 的正面说明。
- [ ] 在 §5.4 写入 descriptive `2.078/3.130≈0.66` 与 E2B `0/192` 的正确边界。
- [ ] 在 2026-07-28 12:00 前编译出完整、可通读、参考文献正常的7页正文PDF。

## P0：GPU实验，和写作并行

- [ ] RG1-P Stage A CPU Gate。
- [ ] RG1-P official + schedule/norm-matched random smoke 64。
- [ ] RG1-P formal 768。
- [ ] RG1-P family analysis与Figure 3C。
- [ ] Llama-3.3-70B CPU Gate：revision、axis、config、tokenization。
- [ ] regime-matched Qwen forced-choice CPU Gate、smoke与 formal 192。
- [ ] Llama smoke：覆盖N/R/V/F、uncapped/p25、trigger和option logits。
- [ ] Llama formal 192。
- [ ] Llama family-bootstrap分析。
- [ ] 将Llama结果按G-A/B/C/D放入Figure 2D或附录。

## P1：只有论文完成闸通过后

- [ ] 决定是否启动Stage B。
- [ ] layer53与60双向full-state patching。
- [ ] self-patch、random、wrong-family controls。
- [ ] 若双向结果清楚，将Figure 3D加入正文。
- [ ] 若失败或时间不足，停止，不影响主文成立。

## P1：写作与图表验收

- [ ] Figure 2/3使用family-level CI。
- [ ] Figure 3 先产出两联页面安全版；RG1-P仅在结果和页数都允许时加入。
- [ ] 所有图使用颜色+marker/填充/线型冗余编码，灰度可读、WCAG对比度达标、字体≥9pt、PDF字体嵌入。
- [ ] 检查每张主图在正文中回答“为什么存在、看什么、改变什么结论”。
- [ ] 主文中每个数字绑定artifact和commit。
- [ ] 清除旧稿虚构/TBD数字，只保留明确标记的实验插槽。

## P2：附录和提交包

- [ ] 整理完整协议、失败实验和偏离。
- [ ] 完成Reproducibility Checklist。
- [ ] 运行 `audit_aaai27_source.py`。
- [ ] 检查7页正文、8–9页仅参考文献。
- [ ] 检查匿名性、PDF metadata、本地路径、私人仓库链接。
- [ ] 检查补充ZIP可独立理解且不含private key/salt/身份信息。
- [ ] 最终PDF逐页视觉检查。

## 当前明确不做

- [ ] 不做shown/hidden。
- [ ] 不做MUSE/J-Lens/SAE全栈。
- [ ] 不做新controller。
- [ ] 不做五剂量重扫。
- [ ] 不做新数据集改写或按结果筛题。
- [ ] 不在Llama上扩展E2B/RG1。

---

## 13. 最终读者记忆

评审读完后应记住：

1. 激活控制研究通常只问“有没有压住行为”，没有问“为什么改变”。
2. CEDAR用同一base、同一target、不同理由把这两个问题分开。
3. official p25稳定移动的是答案修订几何，不是可靠的真假理由选择。
4. N/R显示位移可以潜伏；V/F显示它可能同时推动正确和错误更新。
5. E2B显示即使可见reasoning逐token固定，连续位移仍然存在。
6. RG1-P检验局部单轴修正是否在下游变成更广的表示扰动。
7. Llama最小实验检验这一发现是Qwen特例还是可跨官方配置迁移。
8. 因此，**axis可以描述状态，但不能默认充当selective controller。**
