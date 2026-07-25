# CEDAR AAAI-27 Writing Cards / CEDAR AAAI-27 写作卡片 v0.3

> Status: local writing plan, not paper prose and not synchronized to Overleaf.
> **中文：** 状态：本地写作计划，不是论文正文，且尚未同步到 Overleaf。
> Bilingual reading convention: English comes first and Chinese follows; equations, identifiers, evidence labels, and result slots refer to the same underlying object in both languages.
> **中文：** 双语阅读约定：英文在前、中文紧随其后；公式、标识符、证据标签和结果槽在两种语言中指向同一对象。
> Date: 2026-07-25 (Asia/Shanghai).  
> **中文：** 日期：2026-07-25（Asia/Shanghai）。  
> Revision: 2026-07-26; add a planned Llama-3.3-70B replication using the official Assistant Axis, without making a second semantic axis a requirement.  
> **中文：** 修订：2026-07-26；加入使用官方 Assistant Axis 的 Llama-3.3-70B 跨模型验证计划，不把第二条语义 axis 设为必需项。  
> Narrative decision: phenomenon-first.
> **中文：** 叙事决策：phenomenon-first（现象优先）。
> Working headline hypothesis: `Cost_V > 0`.
> **中文：** 当前标题级工作假设：`Cost_V > 0`。
> Scientific rule: `[TBD]` denotes a result slot, not evidence that the expected result has occurred.
> **中文：** 科学纪律：`[TBD]` 表示结果槽位，不表示预期结果已经发生。

## 1. Fixed Story and Claim Boundary / 已确定的故事与主张边界

### One-Sentence Story / 一句话故事

> Official Assistant-Axis capping reduces the incremental effect of social phrasing beyond target exposure at a direct answer boundary and, under the working headline outcome, suppresses warranted evidence-driven revision, revealing a hidden epistemic cost missed by standard capability evaluations.

> **中文：** 官方 Assistant-Axis 钳制降低了直接答案边界上社会性措辞相对纯目标暴露的增量效应，并在当前标题级预期结果下抑制由充分证据驱动的合理修正，从而揭示标准能力评测未能发现的隐性认识论代价。

This is the title-level working claim. The first clause has a frozen but readout-limited result; the second clause, the dose relation, the mechanism, and the controller remain result slots until their corresponding experiments pass audit.

**中文：** 这是当前标题级工作主张。第一分句已有冻结但受读数边界限制的结果；第二分句、剂量关系、机制与控制器仍是结果槽位，只有在相应实验通过审计后才能转为实证结论。

### Empirical Scope / 实证范围

- Primary model: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`.<br>
  **中文：** 主要模型：`Qwen/Qwen3-32B`，revision `9216db5781bf21249d130ec9da846c4624c16137`。
- Primary intervention: the official Assistant-Axis threshold-capping assets, with `layers_46:54-p0.25` as the source configuration (decoder layers 46--53).<br>
  **中文：** 主要干预：官方 Assistant-Axis 阈值钳制资产，以 `layers_46:54-p0.25` 为源配置（解码器第 46--53 层）。
- Planned cross-model replication: `meta-llama/Llama-3.3-70B-Instruct`, model revision to be frozen, using the official Llama Assistant Axis and `layers_56:72-p0.25` (decoder layers 56--71). All Llama outcomes remain `[TBD-R]` until audited.<br>
  **中文：** 计划中的跨模型验证：`meta-llama/Llama-3.3-70B-Instruct`，模型 revision 待冻结，使用官方 Llama Assistant Axis 与 `layers_56:72-p0.25`（解码器第 56--71 层）。所有 Llama 结果在通过审计前均保持 `[TBD-R]`。
- The paper studies this configuration as a reproducible instance of activation-axis capping. It does not claim that every axis, model, threshold, or capping operator has the same effect.<br>
  **中文：** 本文将该配置作为 activation-axis capping 的一个可复现实例进行研究，不声称所有 axis、模型、阈值或钳制算子都具有相同效应。
- A second semantic axis is not required for the main paper. Re-extracting the same Assistant construct would test estimator robustness, not generalization across axis semantics.<br>
  **中文：** 主论文不要求第二条语义 axis。重新提取同一 Assistant 构念只能检验估计稳定性，不能证明跨 axis 语义泛化。
- “Revision,” “warrant,” and “epistemic cost” refer to output behavior under normatively labeled input conditions. They do not assert a human-like internal belief state.<br>
  **中文：** “修正”“充分理由”和“认识论代价”均指规范性标注输入条件下的输出行为，并不主张模型具有类人的内部信念状态。

### Draft Evidence Labels / 草稿证据标签

| Label / 标签 | Meaning / 含义 | Draft permission / 草稿权限 |
|---|---|---|
| `[V]` | Frozen, audited result / 已冻结且完成审计的结果 | May be stated with its exact scope and limitations / 可在准确限定范围和局限的前提下陈述 |
| `[V-AUDIT]` | Frozen result with an unresolved definition/documentation audit / 已冻结但定义或文档审计尚未解决的结果 | May be drafted, but not promoted to final confirmatory wording / 可以起草，但不能升级为最终确认性措辞 |
| `[E]` | Exploratory or diagnostic result / 探索性或诊断性结果 | May motivate a test; may not establish the headline claim / 可用于提出检验动机，不得确立标题级主张 |
| `[TBD-R]` | Main behavioral result pending / 主要行为结果待完成 | Expected conclusion may be written only inside an explicit result slot / 预期结论只能写在明确标注的结果槽中 |
| `[TBD-M]` | Mechanism result pending / 机制结果待完成 | No causal verb before bidirectional intervention evidence / 双向干预证据完成前不得使用因果动词 |
| `[TBD-C]` | Controller result pending / 控制器结果待完成 | No mitigation contribution before matched-benefit evaluation / 匹配收益评估完成前不得声称缓解贡献 |
| `[SRC]` | Prior-work fact / 既有工作事实 | Requires a verified citation / 需要经核验的引用 |

## 2. Writing Rules Taken from the Feishu Guide / 来自飞书指南的写作规则

1. Write in the order `contribution -> method -> results -> conclusion -> introduction -> abstract`.<br>
   **中文：** 按 `贡献 -> 方法 -> 结果 -> 结论 -> 引言 -> 摘要` 的顺序写作。
2. A reader who sees only the section titles and each paragraph's first sentence should recover the argument.<br>
   **中文：** 读者只看章节标题和每段首句，也应能够还原全文论证。
3. Introduction logic is `important problem -> missing evaluation -> our solution and findings`.<br>
   **中文：** 引言逻辑为 `重要问题 -> 缺失的评估 -> 我们的方案与发现`。
4. Related Work is organized around the gap, not as paper-by-paper summaries.<br>
   **中文：** 相关工作围绕研究缺口组织，而不是逐篇论文罗列摘要。
5. Prior work, source assets, and CEDAR contributions must be explicitly separated.<br>
   **中文：** 必须明确区分既有工作、来源资产与 CEDAR 自身贡献。
6. Figure 1 must function as a graphical abstract: problem, paired design, estimands, and claimed finding.<br>
   **中文：** Figure 1 必须承担图形摘要的功能：展示问题、配对设计、估计量和所主张的发现。
7. Claims should be concrete and compact; adjectives such as “novel,” “strong,” and “effective” need evidence.<br>
   **中文：** 主张应具体、紧凑；“novel”“strong”“effective”等形容词必须有证据支撑。
8. The Feishu suggestion to compress layout with negative `vspace` is not used because it conflicts with the AAAI-27 author instructions.<br>
   **中文：** 不采用飞书指南中使用负 `vspace` 压缩版面的建议，因为这与 AAAI-27 作者指南冲突。

## 3. Structural Lessons Taken from the Reference Papers / 参考论文带来的结构启示

### *When Truth Is Overridden* / 《当真相被覆盖》

Use the progression `behavioral phenomenon -> stage/layer localization -> representation comparison -> bidirectional intervention -> constrained takeaway`. Each mechanism subsection must open with the question it resolves. Do not copy its stronger interpretation of logit-lens or KL divergence; in CEDAR these remain diagnostic until intervention confirms them.

**中文：** 借鉴其 `行为现象 -> 阶段/层定位 -> 表征比较 -> 双向干预 -> 有边界的结论` 推进方式。每个机制小节必须以其要回答的问题开头。不要照搬其对 logit-lens 或 KL divergence 的较强解释；在 CEDAR 中，这些结果在干预证据确认前仍只是诊断性的。

### *Refusal in Language Models Is Mediated by a Single Direction* / 《语言模型中的拒答由单一方向介导》

Use complementary induction and rescue/ablation to support necessity and sufficiency. Follow the main behavioral effect with a dedicated collateral-effect check. Explicitly distinguish a functional direction name from its unknown full semantic content.

**中文：** 使用互补的 induction 与 rescue/ablation 支撑必要性和充分性。在主要行为效应之后，单独设置副作用检查。明确区分方向的功能性命名与其尚未知晓的完整语义内容。

### WMDP / 大规模机器学习模型危险知识测量基准

Present the intended safety effect and the nearest capability loss together. Report failures of selectivity directly rather than hiding them behind a best operating point. Keep the benchmark/measurement contribution primary if the controller is only a demonstration.

**中文：** 将预期安全效果与最邻近的能力损失放在一起呈现。直接报告选择性失败，不要把它藏在最佳运行点之后。如果控制器只是演示性质，则仍以 benchmark/测量贡献为主。

## 4. Section-by-Section Writing Cards / 逐节写作卡片

## Abstract / 摘要

**Only question / 唯一问题：** What is missed, what does CEDAR measure, what happens, and why does it matter? / 现有评估遗漏了什么、CEDAR 测量什么、发生了什么，以及为什么重要？

1. Static capability tests do not determine whether an activation-space safety intervention preserves responsiveness to valid new evidence.<br>
   **中文：** 静态能力测试无法判断激活空间安全干预是否保留对有效新证据的响应能力。
2. We ask whether suppressing socially induced answer changes also suppresses warranted revision.<br>
   **中文：** 我们追问：抑制社会因素诱发的答案变化，是否也会抑制有充分理由的修正。
3. CEDAR combines two independent paired modules: formal valid/fallacious evidence and natural social pressure/target exposure.<br>
   **中文：** CEDAR 结合两个独立的配对模块：形式化的有效/谬误证据，以及自然社会压力/目标暴露。
4. `[V]` State the direct AO/S interaction and its free-generation boundary; `[TBD-R]` state `Cost_V` and the dose-indexed trade-off; `[TBD-M/C]` state mechanism/controller only if passed.<br>
   **中文：** `[V]` 陈述直接 AO/S 交互及其自由生成边界；`[TBD-R]` 陈述 `Cost_V` 与以剂量为索引的权衡；只有相应检验通过时，才陈述 `[TBD-M/C]` 机制/控制器结果。
5. Conclude that safety interventions require reason-sensitive side-effect evaluation, not merely static capability preservation.<br>
   **中文：** 最后指出，安全干预需要对理由质量敏感的副作用评估，而不能只检查静态能力是否保持。

**Do not include / 不得写入：** internal experiment codes, unexplained `p25`, pilot trends, or a universal claim about capping. / 内部实验代号、未经解释的 `p25`、pilot 趋势，或关于钳制的普遍性主张。

## 1. Introduction / 引言

### P1 — Why Activation-Space Control Is Attractive / 为什么激活空间控制具有吸引力

**Topic sentence / 主题句：** Inference-time interventions can suppress undesirable behavior without retraining a model, but their behavioral selectivity is rarely evaluated directly.<br>
**中文：** 推理时干预无需重新训练模型即可抑制不良行为，但其行为选择性很少受到直接评估。

**Evidence/citations / 证据与引用：** activation steering, inference-time intervention, refusal-direction work, Assistant Axis / activation steering、推理时干预、refusal-direction 工作与 Assistant Axis。

### P2 — The Missing Side Effect / 被遗漏的副作用

**Topic sentence / 主题句：** Preserving performance on static benchmarks does not show that an intervention preserves the ability to revise an answer when genuinely decisive information arrives.<br>
**中文：** 在静态 benchmark 上保持性能，并不能说明干预保留了模型在真正具有决定性的信息到来时修正答案的能力。

**Interface / 接口：** matched static-capability result `[TBD-R]` / 匹配的静态能力结果 `[TBD-R]`。

### P3 — The Motivating Counterexample / 作为动机的反例

**Topic sentence / 主题句：** A useful intervention should resist an unsupported social cue while remaining responsive to a valid proof or counterexample that changes the correct answer.<br>
**中文：** 有用的干预应当抵抗无依据的社会线索，同时仍能响应会改变正确答案的有效证明或反例。

**Concept introduced / 引入概念：** epistemic selectivity; false-positive versus false-negative revision / 认识论选择性；假阳性修正与假阴性修正。

### P4 — What CEDAR Measures / CEDAR 测量什么

**Topic sentence / 主题句：** CEDAR separates reason quality with two independent paired modules rather than treating every answer change as the same behavior.<br>
**中文：** CEDAR 使用两个独立的配对模块区分理由质量，而不是把所有答案变化视为同一种行为。

**Content / 内容：** N/R/V/F in CEDAR-Evidence; AO/S in CEDAR-Social; do not call this a six-arm factorial experiment. / CEDAR-Evidence 中的 N/R/V/F；CEDAR-Social 中的 AO/S；不得称为六臂析因实验。

### P5 — What We Find / 我们发现什么

**Topic sentence / 主题句：** Official Assistant-Axis capping changes the social-pressure increment and, in our working headline result, incurs a measurable cost to warranted revision.<br>
**中文：** 官方 Assistant-Axis 钳制改变了社会压力增量，并在当前标题级预期结果中，对有充分理由的修正造成可测量的代价。

**Slots / 槽位：** exact `[V]` social result; `[TBD-R] Cost_V`; `[TBD-M]`; `[TBD-C]` only if strong. / 精确的 `[V]` 社会侧结果；`[TBD-R] Cost_V`；`[TBD-M]`；只有结果足够强时才纳入 `[TBD-C]`。

### P6 — Contributions / 贡献

1. A reason-sensitive formulation and estimands for epistemic selectivity under activation intervention.<br>
   **中文：** 面向激活干预、对理由质量敏感的认识论选择性形式化与估计量。
2. CEDAR-Evidence plus a separately deconfounded CEDAR-Social protocol.<br>
   **中文：** CEDAR-Evidence，以及单独进行去混杂设计的 CEDAR-Social 协议。
3. A scoped empirical finding on official Assistant-Axis capping, followed conditionally by mechanism and mitigation evidence.<br>
   **中文：** 关于官方 Assistant-Axis 钳制、范围严格限定的实证发现；机制与缓解证据则依据结果有条件地加入。

## 2. Background and Related Work / 背景与相关工作

### 2.1 Activation-Space Control and Capping / 激活空间控制与钳制

**Question / 问题：** What can activation intervention change, and what collateral effects have prior evaluations measured? / 激活干预能够改变什么，既有评估测量了哪些连带效应？

**Organization / 组织：** CAA/ITI -> refusal direction and causal standard -> Assistant Axis and threshold capping -> static side-effect gap. Explicitly distinguish addition, ablation, steering, and threshold capping.<br>
**中文：** CAA/ITI -> refusal direction 与因果证据标准 -> Assistant Axis 与阈值钳制 -> 静态副作用评估缺口。明确区分 addition、ablation、steering 和 threshold capping。

### 2.2 Sycophancy, Social Influence, and Overridden Truth / 谄媚、社会影响与真相被覆盖

**Question / 问题：** What is known about unsupported user influence, and what is CEDAR not trying to rediscover? / 关于无依据的用户影响已经知道什么，CEDAR 不打算重复发现什么？

**Organization / 组织：** behavioral sycophancy -> *When Truth Is Overridden* mechanism pattern -> CEDAR asks whether an intervention that reduces this behavior also blocks justified correction. Use “social-pressure-induced target following” for our readout, not unqualified internal sycophancy.<br>
**中文：** 行为性 sycophancy -> *When Truth Is Overridden* 的机制范式 -> CEDAR 追问降低该行为的干预是否也会阻碍合理修正。对我们的读数使用“由社会压力诱发的目标跟随”，不能不加限定地称为内部谄媚。

### 2.3 Belief Revision and Intervention Side Effects / 信念修正与干预副作用

**Question / 问题：** Which works already distinguish desirable from undesirable change, and what remains missing? / 哪些工作已经区分期望与不期望的变化，目前还缺少什么？

**Organization / 组织：** Belief-R/update-maintain -> dual-stance/Buchan and other 2026 nearest neighbors -> WMDP/ITI/KL-then-steer style benefit-retention evaluations -> missing combination of threshold capping, formal normative gold, target-exposure control, and dose resolution.<br>
**中文：** Belief-R/update-maintain -> dual-stance/Buchan 与其他 2026 年最近邻 -> WMDP/ITI/KL-then-steer 式收益—保留评估 -> 尚缺少“阈值钳制 + 形式化规范金标准 + 目标暴露对照 + 剂量分辨”的组合。

### Closing Gap Sentence / 研究缺口收束句

> Prior work asks whether models can be steered, whether they yield to users, or whether they revise outputs; CEDAR asks whether a safety-motivated internal intervention preserves the selectivity of revision with respect to the quality of the reason supplied.

> **中文：** 既有工作追问模型能否被 steering、是否会屈从于用户，或是否会修正输出；CEDAR 追问的是，以安全为动机的内部干预能否依据所给理由的质量，保留输出修正的选择性。

## 3. CEDAR: Measuring Epistemic Selectivity / CEDAR：测量认识论选择性

### 3.1 Problem Setup and Intervention / 问题设定与干预

**Question / 问题：** What is revised, what is capped, and what empirical scope is claimed? / 什么被修正、什么被钳制，以及我们主张怎样的实证范围？

- Define output revision without a mental-state claim.<br>
  **中文：** 定义输出修正，但不作心理状态主张。
- Define the official per-layer threshold operator and its sign convention.<br>
  **中文：** 定义官方逐层阈值算子及其符号约定。
- Give model, axis, layers, token/phase scope, and tested doses.<br>
  **中文：** 给出模型、axis、层、token/阶段作用范围以及所测试剂量。
- Present Qwen as the primary model and Llama as a prespecified cross-model replication; freeze the Llama model revision before execution.<br>
  **中文：** 将 Qwen 作为主模型、Llama 作为预先规划的跨模型验证，并在执行前冻结 Llama 模型 revision。
- Keep hashes, hook implementation, and full telemetry in the appendix/reproducibility material.<br>
  **中文：** 将哈希、hook 实现和完整 telemetry 放入附录或可复现性材料。
- `[TBD]` exact dose set and final confirmation decoding configuration.<br>
  **中文：** `[TBD]` 精确剂量集合与最终确认实验解码配置。

### 3.2 CEDAR-Evidence: Normatively Grounded Revision / 具有规范依据的修正

**Question / 问题：** How can valid evidence be separated from mere rechecking and persuasive but invalid argument? / 如何把有效证据与单纯复核、具有说服力但无效的论证区分开来？

- N: unchanged world, neutral continuation, preserve initial answer.<br>
  **中文：** N：世界不变，中性续写，保留初始答案。
- R: unchanged world, independent recheck, preserve initial answer.<br>
  **中文：** R：世界不变，独立复核，保留初始答案。
- V: a valid justification warrants one certified atomic edit; update to the new unique answer.<br>
  **中文：** V：有效论证足以支持一次已认证原子编辑；更新到新的唯一答案。
- F: the same proposed edit and target are supported by an argument with exactly one fatal inference; reject the edit and preserve the initial answer.<br>
  **中文：** F：针对同一所提编辑与目标，提供恰含一个致命推理错误的论证；拒绝编辑并保留初始答案。
- Formal gold: Python fixed-point solver plus Prover9 agreement; F additionally has a countermodel or opposite proof.<br>
  **中文：** 形式化金标准：Python 不动点求解器与 Prover9 结论一致；F 还必须具备反模型或相反结论证明。
- Matching: shared base, target, edit, option mapping, response contract, and wrapper; difficulty and answer positions are balanced.<br>
  **中文：** 匹配：共享 base、target、edit、选项映射、响应契约和 wrapper；平衡难度与答案位置。
- `[TBD-R]` confirmation families, variants, error classes, sample size, and audit counts.<br>
  **中文：** `[TBD-R]` 确认实验的 families、变体、错误类别、样本量与审计数量。

### 3.3 CEDAR-Social: Pressure Beyond Target Exposure / 超越目标暴露的社会压力

**Question / 问题：** Does social wording add an effect beyond merely showing the model a target answer? / 除了单纯向模型展示目标答案外，社会性措辞是否会增加额外效应？

- AO exposes only the target option.<br>
  **中文：** AO 只暴露目标选项。
- S adds a first-person confidence statement and a request to reconsider, without adding facts, evidence, authority, reward, or majority information.<br>
  **中文：** S 增加第一人称确信表达与重新考虑请求，但不增加事实、证据、权威、奖励或多数意见信息。
- 600 SycoBench bases are strictly paired across AO/S and uncapped/p25.<br>
  **中文：** 600 个 SycoBench bases 在 AO/S 与 uncapped/p25 之间严格配对。
- Direct target-minus-initial option margin is primary; free generation is secondary.<br>
  **中文：** 直接的目标减初始选项 margin 是主要读数，自由生成是次要读数。
- Human review must be described as one data lead plus automated full validation, without agreement statistics.<br>
  **中文：** 人工审核必须准确描述为“一名数据负责人 + 全量自动验证”，不得报告不存在的一致率统计。

### 3.4 Estimands / 估计量

**Question / 问题：** Which contrasts correspond to warranted evidence, fallacy susceptibility, and social pressure? / 哪些对比分别对应充分证据、谬误易感性和社会压力？

Use `WEG`, `Cost_V`, `FH`, `Benefit_F`, `SME`, and `Benefit_S` exactly as defined in the draft. Keep each in its native module and scale. Report the signed social interaction alongside the positive-oriented benefit definition.

**中文：** 严格按照工作稿中的定义使用 `WEG`、`Cost_V`、`FH`、`Benefit_F`、`SME` 和 `Benefit_S`。每个估计量保留在其原生模块与尺度中。在按正向定义的 benefit 之外，同时报告带符号的社会交互项。

### 3.5 Statistical Discipline / 统计纪律

**Question / 问题：** What is the independent unit and how is uncertainty propagated? / 独立统计单位是什么，不确定性如何传播？

- Evidence: pair within base, cluster/resample by formal family; repeated generations do not increase independent N.<br>
  **中文：** Evidence：在 base 内配对，按 formal family 聚类/重采样；重复生成不增加独立样本量 N。
- Social: pair within SycoBench base and stratify bootstrap by domain/difficulty.<br>
  **中文：** Social：在 SycoBench base 内配对，并按领域/难度分层 bootstrap。
- Report module-specific effect sizes and confidence intervals; no pooled cross-module significance test.<br>
  **中文：** 报告模块专属效应量与置信区间；不得进行跨模块合并显著性检验。
- Report invalidity, truncation, option mass, entropy, and full-vocabulary integrity.<br>
  **中文：** 报告无效输出、截断、选项概率质量、熵与全词表完整性。
- Use proportional retention only when baseline WEG has sufficient dynamic range.<br>
  **中文：** 只有当基线 WEG 具有足够动态范围时，才使用比例保留率。

## 4. A Hidden Epistemic Cost of Activation Capping / 激活钳制的隐性认识论代价

### 4.1 Capping Reduces the Incremental Effect of Social Phrasing at the Direct Answer Boundary / 钳制降低社会性措辞对直接答案边界的增量效应

**Question / 问题：** Does the intervention have its intended behavioral-side effect after target exposure is controlled? / 控制目标暴露后，干预是否产生预期的行为侧效应？

**Evidence / 证据：** `[V-AUDIT]` 600 bases; `cap x (S-AO) = -0.78625` logit, 95% base-bootstrap CI `[-0.86000, -0.71104]`; AO `+1.26792`, S `+0.48167`; free-generation strict SocialExcess `0`.<br>
**中文：** `[V-AUDIT]` 600 个 bases；`cap x (S-AO) = -0.78625` logit，95% base-bootstrap CI `[-0.86000, -0.71104]`；AO `+1.26792`，S `+0.48167`；自由生成 strict SocialExcess 为 `0`。

**Mandatory interpretation / 强制解释：** Both AO and S target margins rise under p25, but the rise is smaller under S. The result is a relative reduction in the social-phrasing increment in a direct forced-choice readout, not an absolute fall in S target following and not proof of reduced internal sycophancy.<br>
**中文：** p25 下 AO 与 S 的目标 margin 均上升，但 S 下升幅较小。该结果表示直接强制选择读数中社会性措辞增量相对下降，不表示 S 条件目标跟随绝对下降，也不能证明内部谄媚减少。

**Before final wording / 最终措辞前：** close the documented sign-convention audit without changing data or analysis. / 在不改变数据或分析的前提下，闭合已有记录的符号约定审计。

### 4.2 Capping Suppresses Warranted Evidence-Driven Revision / 钳制抑制由充分证据驱动的修正

**Question / 问题：** Does WEG decline under the same intervention family? / WEG 是否在同一干预族下下降？

**Result slots / 结果槽位：** `[TBD-R] U_V`, `U_R`, `WEG`, `Cost_V`, family-cluster CI, materiality decision, and decomposition into `Delta U_V` and `Delta U_R`.<br>
**中文：** `[TBD-R] U_V`、`U_R`、`WEG`、`Cost_V`、family-cluster CI、实质性判定，以及对 `Delta U_V` 与 `Delta U_R` 的分解。

**Secondary checks / 次要检查：** N stability, F/FH, V-F discrimination, invalid/truncation, and output integrity. / N 稳定性、F/FH、V-F 判别、无效/截断输出及输出完整性。

### 4.3 The Effect Is Dose-Dependent / 效应具有剂量依赖性

**Question / 问题：** Is the result a systematic property of intervention strength rather than a single threshold accident? / 该结果是否是干预强度的系统性属性，而不是单个阈值上的偶然现象？

**Result slots / 结果槽位：** `[TBD-R]` uncapped and at least three capped doses; dose trend; module-specific CIs; official p25 marked in the plot. / `[TBD-R]` uncapped 与至少三个 capped 剂量；剂量趋势；模块专属 CI；图中标出官方 p25。

**Terminology / 术语：** call this a `dose-indexed trade-off`, not a Pareto frontier. Reserve “frontier” for comparison among L0/L1/L2 controllers. / 称为 `dose-indexed trade-off`（以剂量为索引的权衡），而不是 Pareto frontier；“frontier”只用于 L0/L1/L2 控制器之间的比较。

### 4.4 Static Capability Tests Miss the Dynamic Cost / 静态能力测试遗漏动态代价

**Question / 问题：** Is the WEG loss separable from general degradation? / WEG 损失能否与一般性退化区分开来？

**Result slots / 结果槽位：** `[TBD-R]` matched static tasks, same readout/harness, independent equivalence margin, and output-integrity checks at the relevant doses. / `[TBD-R]` 匹配静态任务、相同 readout/harness、独立等效性界，以及相关剂量下的输出完整性检查。

### 4.5 Cross-Model Replication / 跨模型验证

**Question / 问题：** Does the Assistant-Axis capping effect replicate beyond Qwen3-32B when the axis construct and threshold-capping operator are held fixed? / 在保持 Assistant Axis 构念和阈值钳制算子不变时，该效应能否在 Qwen3-32B 之外复现？

**Design / 设计：** Repeat CEDAR-Social and CEDAR-Evidence on `meta-llama/Llama-3.3-70B-Instruct` using the official Llama axis and `layers_56:72-p0.25`; run uncapped/p25 as the primary comparison and the frozen percentile dose set as the dose extension. / 使用官方 Llama axis 与 `layers_56:72-p0.25`，在 `meta-llama/Llama-3.3-70B-Instruct` 上重复 CEDAR-Social 和 CEDAR-Evidence；以 uncapped/p25 为主要比较，以冻结后的 percentile 剂量集作为剂量扩展。

**Reporting / 报告：** `[TBD-R]` report `Benefit_S`, `WEG`, `Cost_V`, component outcomes, free-generation boundaries, static controls, and CIs separately for each model. Do not pool two models into a population-level estimate; report heterogeneity explicitly. / `[TBD-R]` 对每个模型分别报告 `Benefit_S`、`WEG`、`Cost_V`、分解结果、free-generation 边界、静态对照与 CI。不得把两个模型合并为模型总体估计，必须显式报告异质性。

**Claim gate / 主张门槛：** Use “across two model families” only if both complete, audited model-specific results support it. / 只有两个模型各自完成、通过审计且结果支持时，才能使用“跨两个模型家族”的措辞。

## 5. Why Does the Cost Arise? / 这种代价为何产生？

### 5.1 Validity versus Choice, Prefill versus Decode / 有效性与选择、Prefill 与 Decode

**Question / 问题：** Does capping change assessment of argument validity, final answer selection, or both? / 钳制改变的是论证有效性评估、最终答案选择，还是二者？

**Design / 设计：** independent Validity and Answer calls; uncapped, prefill-only, decode-only, full-p25; layer-resolved dose telemetry. `[TBD-M]`. / 独立的 Validity 与 Answer 调用；uncapped、prefill-only、decode-only、full-p25；逐层剂量 telemetry。`[TBD-M]`。

### 5.2 Geometric Alignment with the Capped Direction / 与钳制方向的几何对齐

**Question / 问题：** Are items with greater uncapped exposure or valid-update displacement more strongly suppressed? / uncapped 暴露更高或有效更新位移更大的样本是否受到更强抑制？

**Design / 设计：** prespecified item/family regression, actual correction dose, length/difficulty controls, and permutation nulls. Allowed verb before intervention: `is associated with` or `is aligned with`.<br>
**中文：** 预先指定的 item/family 回归、实际 correction dose、长度/难度控制与 permutation 零分布。干预证据完成前只允许使用 `is associated with` 或 `is aligned with`。

### 5.3 Bidirectional Intervention / 双向干预

**Question / 问题：** Is the localized representation necessary and sufficient for the cost? / 被定位的表征对该代价是否必要且充分？

**Design / 设计：** induce the capped effect in an uncapped trajectory and rescue WEG in a capped trajectory using matched layer/token windows and random/orthogonal controls. `[TBD-M]`. / 在 uncapped 轨迹中诱导 capped 效应，并在 capped 轨迹中恢复 WEG；使用匹配 layer/token 窗口以及随机/正交对照。`[TBD-M]`。

**Causal threshold / 因果门槛：** use `causally mediates` only if both directions succeed and the intervention does not simply destroy output integrity or all social-side benefit. / 只有两个方向均成功，且干预不是简单破坏输出完整性或全部社会侧收益时，才能使用 `causally mediates`。

## 6. Reducing the Cost / 降低这种代价

### 6.1 Controller Ladder / 控制器阶梯

- L0: official Assistant-Axis capping.<br>
  **中文：** L0：官方 Assistant-Axis 钳制。
- L1: orthogonalized actuation with the original trigger.<br>
  **中文：** L1：保留原始 trigger 的正交化 actuation。
- L2: gated orthogonal capping, with both trigger and actuation defined outside the protected valid-update subspace.<br>
  **中文：** L2：门控正交钳制；trigger 和 actuation 均定义在受保护的有效更新子空间之外。

The protected subspace is learned from V motion only, with strict development/held-out separation. Exact algorithms remain `[TBD-C]` until implementation is frozen.

**中文：** 受保护子空间只从 V 的运动中学习，并严格区分 development 与 held-out。具体算法在实现冻结前仍标记为 `[TBD-C]`。

### 6.2 Matched-Benefit Evaluation / 匹配收益评估

Compare WEG, FH, social benefit, static capability, output integrity, and cost at matched `Benefit_S` or matched actual intervention dose. A controller succeeds only if it improves warranted-revision preservation without buying that improvement by giving up the intended behavioral effect.

**中文：** 在匹配 `Benefit_S` 或匹配实际干预剂量时，比较 WEG、FH、社会侧收益、静态能力、输出完整性和成本。只有当控制器在不牺牲预期行为效应的情况下改善有充分理由修正的保留时，才算成功。

### 6.3 Section Routing / 章节路由

- Strong, held-out matched-benefit improvement: retain Section 6 and a mitigation contribution.<br>
  **中文：** 强且在 held-out 上成立的匹配收益改善：保留第 6 节和缓解贡献。
- Single-point or imprecise improvement: short proof-of-concept paragraph.<br>
  **中文：** 单点或不精确改善：缩为简短的概念验证段。
- Null/negative result: two honest discussion sentences or an appendix analysis.<br>
  **中文：** 零结果/负结果：在 Discussion 中如实写两句，或移入附录分析。
- Not implemented: remove the controller from claimed contributions.<br>
  **中文：** 未实现：从贡献主张中删除控制器。

## 7. Discussion and Conclusion / 讨论与结论

1. Safety-intervention evaluation should test sensitivity to reason quality, not only static accuracy or raw answer-change frequency.<br>
   **中文：** 安全干预评估应检验对理由质量的敏感性，而不能只测静态准确率或原始答案变化频率。
2. The empirical claim is bounded by model, official axis, layer range, capping configuration, task, and readout.<br>
   **中文：** 实证主张受到模型、官方 axis、层范围、钳制配置、任务和读数的共同限定。
3. Output revision is not direct evidence of human-like belief, epistemic agency, or internal sycophancy.<br>
   **中文：** 输出修正并非类人信念、认识论主体性或内部谄媚的直接证据。
4. Discuss cross-model heterogeneity, a possible second semantic axis, natural open-ended evidence, longitudinal revision, stronger human audit, and mechanism/controller generalization.<br>
   **中文：** 讨论跨模型异质性、可能的第二条语义 axis、自然开放式证据、纵向修正、更强人工审核，以及机制/控制器泛化。
5. Final sentence: a useful intervention must not only suppress undesirable changes; it must preserve the capacity to change when the model is given a genuinely sufficient reason.<br>
   **中文：** 结尾句：有用的干预不仅应抑制不良变化，还必须在模型获得真正充分的理由时保留其改变的能力。

## 5. Figure and Table Interfaces / 图表接口

### Figure 1 — Graphical Abstract and Design / 图形摘要与设计

- Left: unsupported social pressure versus valid evidence as the motivating contrast.<br>
  **中文：** 左侧：以无依据的社会压力与有效证据构成动机性对比。
- Center: official capping and the possibility of blunt response suppression.<br>
  **中文：** 中间：官方钳制，以及粗粒度抑制响应的可能性。
- Right upper: two visibly separate modules, CEDAR-Evidence and CEDAR-Social.<br>
  **中文：** 右上：在视觉上明确分开的两个模块——CEDAR-Evidence 与 CEDAR-Social。
- Right lower: social benefit and WEG shown as separate estimands linked only by dose.<br>
  **中文：** 右下：将社会侧收益与 WEG 作为独立估计量展示，二者只通过剂量关联。
- No numerical result until final data are available; no visual arrow that pre-commits an unknown direction in an ostensibly conceptual panel.<br>
  **中文：** 最终数据到位前不显示数值结果；在概念性面板中，不使用预先承诺未知结果方向的视觉箭头。

### Table 1 — Conditions and Confounds / 条件与混杂因素

Columns: module, condition, new information, normative response, excluded confound, estimand. Use a heavy visual separation between Evidence and Social.

**中文：** 列：模块、条件、新信息、规范反应、所排除的混杂、估计量。Evidence 与 Social 之间使用明显的视觉分隔。

### Figure 2 — Main Behavioral Result / 主要行为结果

- (a) AO/S target-margin decomposition with uncapped and p25.<br>
  **中文：** (a) uncapped 与 p25 下的 AO/S target-margin 分解。
- (b) N/R/V/F update rates across doses with family-cluster CIs.<br>
  **中文：** (b) 各剂量下 N/R/V/F 更新率及 family-cluster CI。
- (c) `Benefit_S` versus `WEG`, labeled `dose-indexed trade-off`; axes retain native units and independent uncertainty.<br>
  **中文：** (c) `Benefit_S` 对 `WEG`，标注为 `dose-indexed trade-off`；坐标轴保留原生单位和独立不确定性。
- (d) static dynamic contrast or F-side selectivity; move to appendix if dense.<br>
  **中文：** (d) 静态—动态对比或 F 侧选择性；若过于拥挤则移入附录。

### Figure 3 — Mechanism and Mitigation / 机制与缓解

- (a) Validity versus Answer effects.<br>
  **中文：** (a) Validity 与 Answer 效应对比。
- (b) prefill/decode or layer localization.<br>
  **中文：** (b) prefill/decode 或层定位。
- (c) bidirectional induction/rescue, only after the causal test is complete.<br>
  **中文：** (c) 双向 induction/rescue；仅在因果检验完成后保留。
- (d) L0/L1/L2 matched-benefit frontier, only if the controller meets the retention criterion.<br>
  **中文：** (d) L0/L1/L2 匹配收益 frontier；仅在控制器满足保留判据时保留。

## 6. Immediate Drafting Order / 立即执行的起草顺序

1. Section 3, because its definitions must remain valid under any numerical outcome.<br>
   **中文：** 先写第 3 节，因为其中定义必须在任何数值结果下都成立。
2. Section 4 with exact `[V]` social text and explicit `[TBD-R]` result paragraphs.<br>
   **中文：** 再写第 4 节，使用精确的 `[V]` 社会侧文本和明确标注的 `[TBD-R]` 结果段。
3. Sections 5--6 as claim-conditional prose with mechanism/controller gates.<br>
   **中文：** 将第 5--6 节写成依赖结果才能成立的条件性文本，并设置机制/控制器 gate。
4. Related Work after citation verification and explicit Buchan/KL-then-steer differentiation.<br>
   **中文：** 完成引用核验并明确区分 Buchan/KL-then-steer 后，再写 Related Work。
5. Introduction after the body terminology and figure interfaces stabilize.<br>
   **中文：** 正文术语与图表接口稳定后撰写 Introduction。
6. Abstract last, after every result slot has either a value or has been removed.<br>
   **中文：** 最后写 Abstract；此时每个结果槽都必须已有数值或已被删除。
