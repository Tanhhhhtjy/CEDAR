# CEDAR AAAI-27 Working Draft / CEDAR AAAI-27 论文工作稿 v0.4

> This bilingual working draft mirrors the manuscript structure in Overleaf. The English submission source is `main.tex`; the Chinese review companion is `main_zh.tex`.
>
> **中文：** 本双语工作稿与 Overleaf 论文结构保持一致。英文投稿源为 `main.tex`，中文审阅稿为 `main_zh.tex`。
>
> Fixed drafting rule: write in completed-study prose. Missing configurations, values, analyses, or result-dependent conclusions occupy narrow `TBD` slots; the paper does not tell the reader that an experiment is unfinished.
>
> **中文：** 固定写作规则：正文始终采用研究已经完整执行的成稿口吻。尚缺的配置、数值、分析或依赖结果的结论只放入窄范围 `TBD` 槽位；正文不向读者说明某项实验尚未完成。

## Provisional Title / 暂定标题

**CEDAR: A Solver-Backed Audit of Epistemic Selectivity under Activation Capping**

**CEDAR：对激活钳制之认识论选择性的求解器支撑审计**

## Abstract / 摘要

`[TBD: final five-sentence abstract with audited Qwen/Llama/Gemma results]`

`[TBD：填入包含经审计 Qwen/Llama/Gemma 结果的最终五句式摘要]`

## 1. Introduction / 引言

Paragraph contract / 段落接口：

1. Runtime activation interventions can suppress undesirable behavior without retraining.<br>
   **中文：** 推理时激活干预无需重新训练即可抑制不良行为。
2. Behavior-only and static-capability evaluations do not establish whether revision remains sensitive to the reason for changing.<br>
   **中文：** 只看目标行为和静态能力的评测，无法证明模型改答时仍然对改答理由保持敏感。
3. The same proposed target can be supported by valid evidence, an invalid argument, mere target exposure, or social phrasing; these cases require different normative responses.<br>
   **中文：** 同一个 proposed target 可以由有效证据、无效论证、纯 target exposure 或社会性措辞引出，而这些情况在规范上要求不同响应。
4. CEDAR makes this selectivity auditable through two complementary but statistically separate modules.<br>
   **中文：** CEDAR 通过两个互补但统计上独立的模块，使这种选择性可以被审计。
5. `[TBD: concise audited headline result across continuous geometry, discrete updating, and cross-model heterogeneity.]`<br>
   **中文：** `[TBD：填入涵盖连续几何、离散更新与跨模型异质性的经审计标题级结果。]`
6. Three contributions: framework and evaluation assets, a scoped empirical audit, and controlled diagnostics.<br>
   **中文：** 三项贡献：框架与评测资产、限定范围的实证审计，以及受控诊断。

Approved contribution architecture / 已确认的贡献架构：

1. **Framework and evaluation assets.** We formulate epistemic selectivity as a distinct evaluation target and introduce CEDAR. It comprises CEDAR-Evidence, a new solver-certified paired evaluation dataset, and CEDAR-Social, a statistically separate paired protocol over SycoBench-600.<br>
   **中文：框架与评测资产。** 我们将认识论选择性形式化为独立评测目标并提出 CEDAR。CEDAR 包含新的 solver-certified 配对评测数据集 CEDAR-Evidence，以及统计上独立的 SycoBench-600 配对协议 CEDAR-Social。
2. **Scoped empirical audit.** We audit Assistant-Axis threshold capping on Qwen, Llama, and Gemma, retaining model-specific configurations, uncertainty, and heterogeneity.<br>
   **中文：限定范围的实证审计。** 我们在 Qwen、Llama 与 Gemma 上审计 Assistant-Axis 阈值钳制，并保留模型专属配置、不确定性与异质性。
3. **Controlled diagnostics.** We use phase-localized intervention, validity-versus-choice analysis, fixed-trace replay, and controlled representation analysis to diagnose the observed shift without exceeding the available causal evidence.<br>
   **中文：受控诊断。** 我们使用分阶段干预、有效性与选择分解、fixed-trace replay 和受控表示分析诊断观测到的位移，且不使用超出因果证据层级的措辞。

## 2. Background and Related Work / 背景与相关工作

### 2.1 Inference-Time Activation Control: Steering, Ablation, and Capping / 推理时激活控制：Steering、消融与钳制

Post-training pipelines such as supervised fine-tuning and preference optimization remain a common route to aligning model behavior (Ouyang et al. 2022; Rafailov et al. 2023). Activation-level methods provide a complementary control surface at inference time, modifying internal states without updating base-model weights. Representation engineering treats population-level activation patterns as objects that can be read and manipulated (Zou et al. 2023). Existing methods span additive steering along contrastive directions (Turner et al. 2023; Rimsky et al. 2024), localized interventions on attention heads (Li et al. 2023), directional ablation (Arditi et al. 2024), and context-conditioned control (Lee et al. 2025).<br>
**中文：** 监督微调和偏好优化等训练后方法，是使模型行为符合预期的常用途径（Ouyang et al. 2022；Rafailov et al. 2023）。与此同时，激活层方法提供了互补的推理期控制界面，可在不更新基础模型权重的情况下直接调整内部状态。表征工程将群体层面的激活模式作为可读取和操控的对象（Zou et al. 2023）。现有方法包括沿对比方向进行加性 steering（Turner et al. 2023；Rimsky et al. 2024）、对注意力头实施局部干预（Li et al. 2023）、方向消融（Arditi et al. 2024），以及依赖上下文的条件控制（Lee et al. 2025）。

The Assistant Axis identifies a population-level direction associated with assistant-like behavior and constrains its projection using a one-sided, percentile-calibrated threshold (Lu et al. 2026). The cap activates only when a state crosses this boundary, with its magnitude determined by the distance beyond the threshold. CEDAR uses Assistant-Axis capping as its primary case study to audit responses to social pressure and warranted correction. This motivates operator-specific evaluation, since a cap that achieves its behavioral target may also change how the model responds to different grounds for revision.<br>
**中文：** Assistant Axis 提取与助手式行为相关的群体层面方向，并以按分位数校准的单侧阈值约束该方向上的投影（Lu et al. 2026）。只有当状态越过这一边界时，capping 才介入，干预幅度由越界距离决定。CEDAR 以 Assistant-Axis capping 为主要实例，审计其对社会压力响应和合理修正的影响。这也要求对不同算子分别进行评估：capping 即使达成了目标行为，也可能改变模型对不同改答依据的响应。

### 2.2 Sycophancy, Social Influence, and Answer Revision / 谄媚、社会影响与答案修正

`[TBD: verified synthesis of social influence, target exposure, valid evidence, and fallacious persuasion.]`

`[TBD：填入经核验的社会影响、target exposure、有效证据与谬误说服相关工作综合。]`

### 2.3 Reason-Sensitive Evaluation and Intervention Side Effects / 理由敏感评测与干预副作用

`[TBD: nearest-neighbor comparison with dual-stance evaluation, Belief-R, and benefit-retention work.]`

`[TBD：填入与 dual-stance evaluation、Belief-R 及收益—保留评测的最近邻比较。]`

## 3. CEDAR: Auditing Epistemic Selectivity / CEDAR：审计认识论选择性

CEDAR examines how activation capping changes a model's response to a proposed answer revision, with particular attention to sensitivity to the basis for that revision. It combines two complementary paired evaluations. CEDAR-Evidence makes formal reason validity solver-verifiable; CEDAR-Social estimates the incremental contribution of social framing beyond exposure to the proposed answer. The modules are analyzed on their own tasks, readouts, and sampling units. The remainder of this section defines their shared revision setting, describes the datasets and model-specific interventions, and specifies behavioral measures and statistical inference.

**中文：** CEDAR 考察激活钳制如何改变模型对拟议答案修正的响应，重点关注这种响应能否区分不同的修正依据。它包含两套互补的配对评测。CEDAR-Evidence 使理由的形式有效性可以由求解器核验；CEDAR-Social 在控制拟议答案暴露后，估计社会性措辞的增量影响。两个模块分别采用各自的任务、读数和采样单位。本节随后定义二者共享的答案修正设定，并介绍数据集、模型专属干预、行为度量与统计推断。

### 3.1 Shared Revision Setting and Controlled Contrasts / 答案修正的共同设定与受控对照

Consider a model that initially selects $a_0$ for a base question. A follow-up proposes a different option $a_t$ and elicits another response. Any valid final choice different from $a_0$ counts as an *answer revision*; a revision to $a_t$ counts as *target following*. Epistemic selectivity captures the behavioral separation between follow-ups that warrant adopting $a_t$ and matched follow-ups that do not.

Within each paired comparison, the base question, $a_0$, $a_t$, and response contract remain fixed. CEDAR-Evidence varies the formal validity of the supplied justification; CEDAR-Social varies social framing while holding target exposure constant. Figure 1 summarizes these controlled contrasts.

**中文：** 考虑模型在一道基础题上最初选择 $a_0$。后续输入提出另一选项 $a_t$，并要求模型再次作答。只要最终有效选择离开 $a_0$，就记为一次*答案修正*；其中，转向 $a_t$ 的修正记为*目标跟随*。认识论选择性刻画的是：在其他要素匹配时，足以支持采用 $a_t$ 的后续输入与不支持这一修正的后续输入能够产生多大的行为差异。

在每组配对比较中，基础题、$a_0$、$a_t$ 和响应约定均保持一致。CEDAR-Evidence 改变所提供论证的形式有效性；CEDAR-Social 在保持目标暴露不变的情况下改变社会性措辞。Figure 1 概括了这些受控对照。

**Figure 1 placeholder / 图 1 占位：** CEDAR’s two-module audit design / CEDAR 双模块审计设计。

**Caption / 图注：** **Controlled contrasts for answer revision.** CEDAR-Evidence constructs four follow-ups from the same formal base, initial answer, and proposed target: neutral continuation ($N$), reason-free rechecking ($R$), valid evidence ($V$), and a matched argument containing one certified inferential error ($F$). Revision is formally warranted only under $V$. CEDAR-Social independently pairs target-only exposure ($AO$) with socially framed reconsideration ($S$) on the same natural question and proposed target. The two modules use distinct readouts and sampling units. Additional paired examples, solver certificates, and construction details appear in *[TBD-APPENDIX]*.

**中文图注：** **答案修正的受控对照。** CEDAR-Evidence 基于相同的形式逻辑基础题、初始答案和目标选项构造四种后续条件：中性延续（$N$）、无理由复核（$R$）、有效证据（$V$），以及仅含一个经认证推理错误的匹配论证（$F$）。其中，只有 $V$ 在形式上支持模型改答。CEDAR-Social 则在同一道自然任务及同一目标选项上，将纯目标暴露（$AO$）与带有社会性措辞的复核请求（$S$）配对。两个模块分别使用各自的读数和采样单位。更多配对示例、求解器验证记录和构造细节见 *[TBD-APPENDIX]*。

### 3.2 CEDAR-Evidence: A Solver-Certified Paired Dataset / CEDAR-Evidence：经求解器认证的配对数据集

Each CEDAR-Evidence item specifies a finite, function-free Horn theory with explicit negation and a four-way query: supported, undetermined, contradicted, or both. Each base has a unique initial answer $a_0$; a deterministic atomic edit changes it to $a_t$. Four follow-ups share this base and target. $N$ continues the dialogue without requesting reconsideration; $R$ requests another check without adding information; $V$ supplies a formally valid justification for the edit, making $a_t$ the uniquely warranted answer; and $F$ advances the same edit through an argument containing one certified invalid inference, leaving $a_0$ correct.

**中文：** CEDAR-Evidence 的每个基础样本均包含一个带显式否定、且不含函数符号的有限 Horn 逻辑理论，并配有一道四分类查询：得到支持、无法确定、遭到反驳，或同时得到支持与反驳。该理论唯一确定初始答案 $a_0$；按照确定性规则施加一项原子编辑后，唯一正确答案变为 $a_t$。四种后续条件沿用同一基础题和目标选项：$N$ 自然延续对话，不要求重新作答；$R$ 要求再次检查，但不提供新信息；$V$ 给出形式有效的论证，使 $a_t$ 成为具有充分依据的唯一答案；$F$ 则通过包含一个经认证无效推理的论证提出同一编辑，此时 $a_0$ 仍为正确答案。

Within each $V/F$ pair, we hold the speaker identity, speech act, proposed edit, prompt scaffold, and response format constant. A Python fixed-point engine and Prover9 independently verify the object-level answer and whether the edit is warranted; the dataset contains instances on which they agree. Each $F$ argument has a Mace4 countermodel and exactly one certified fatal inference, with all remaining steps supported. Before model evaluation, we rebuild every instance deterministically, check for duplicates and leakage, and conduct a blinded review of linguistic naturalness.

Each evaluation bank contains 24 bases: easy and medium variants of 12 independently generated structural families spanning six error classes. A second bank, generated with an independent salt, is analyzed separately. Structural families define the independent sampling units; follow-up conditions and model runs contribute paired observations within each family.

**中文：** 每组 $V/F$ 配对均保持说话者身份、言语行为、拟议编辑、提示框架和输出约定一致。Python 不动点推理器与 Prover9 分别验证对象层答案以及拟议编辑是否有充分依据，数据集收录二者结论一致的样本。每个 $F$ 论证都配有 Mace4 反模型，并恰好包含一个经认证的致命推理错误，其余步骤均有支持。模型评测前，我们对全部样本进行确定性重建、重复与泄漏检查，以及盲态语言自然度审核。

每个评测批次包含 24 个基础样本，对应 12 个独立生成结构族的简单与中等难度变体，共覆盖六类错误。第二批数据使用独立随机盐生成并单独分析。结构族构成统计独立单位，同一结构族下的后续条件和模型运行均按配对观测处理。

### 3.3 CEDAR-Social: Controlling for Target Exposure / CEDAR-Social：控制目标暴露效应

CEDAR-Social constructs paired follow-ups for the 600 natural multiple-choice questions in SycoBench-600. For each question, the assistant's initial response is the gold option, and $a_t$ is a different option. Both follow-ups present the same $a_t$ and elicit a new answer. The answer-only ($AO$) condition presents the target without attributing it to a speaker or invoking correctness, evidence, authority, reward, or consensus. The matched social ($S$) condition frames the same target exposure with a first-person expression of confidence and a request for reconsideration, while adding no task-relevant fact or reason. The paired $S-AO$ contrast estimates the incremental effect of social phrasing beyond target exposure.

Four minimally edited template pairs are balanced across domain and difficulty strata. Automated checks verify source-field identity, permitted template differences, and token-length bounds under the frozen tokenizer across all 600 pairs. Two researchers independently audit every pair for template fidelity and linguistic naturalness without access to model outputs; agreement and adjudication outcomes are *[TBD-SOCIAL-AUDIT]*. Direct next-option margins provide the primary readout. A frozen 240-question subset assesses whether the findings extend to free generation.

**中文：** CEDAR-Social 基于 SycoBench-600 的 600 道自然多选题构造配对后续输入。每道题中，助手的初始回答固定为标准答案，$a_t$ 为另一个选项。两种后续输入都呈现相同的 $a_t$，并要求模型再次作答。纯目标暴露条件（$AO$）直接呈现目标选项，不引入说话者，也不诉诸正确性、证据、权威、奖励或群体共识。匹配的社会压力条件（$S$）以第一人称确信和重新考虑请求来表述相同的目标暴露，同时不提供与题目有关的新事实或理由。配对的 $S-AO$ 对比用于估计社会性措辞在目标暴露之外带来的增量效应。

四组最小改写模板在各领域和难度层内均衡分配。自动检查覆盖全部 600 对样本，验证源字段一致性、允许的模板差异以及冻结分词器下的 token 长度范围。两名研究人员在不接触模型输出的情况下，分别检查每一对样本的模板一致性和语言自然度；审核一致率、分歧及裁决结果为 *[TBD-SOCIAL-AUDIT]*。主要读数是直接 next-option margin；冻结的 240 题子集用于检验结果能否延伸到自由生成。

### 3.4 Intervention and Model Scope / 干预与模型范围

We study the released, model-specific Assistant-Axis threshold cap. At layer $\ell$, let $\mathbf{v}_{\ell}$ denote the unit intervention direction stored in the released capping configuration and $\tau_{\ell,c}$ its threshold under configuration $c$. The stored direction is oppositely oriented to the corresponding Assistant Axis vector, with the threshold calibrated in that coordinate. Consequently, the upper-cap form below is equivalent to the minimum cap defined along the Assistant-Axis orientation in the source paper. Given residual activation $\mathbf{h}_{\ell,t}$, the operator maps it to

$$
\mathbf{h}'_{\ell,t}
=
\mathbf{h}_{\ell,t}
-
\mathbf{v}_{\ell}
\left[
\langle\mathbf{h}_{\ell,t},\mathbf{v}_{\ell}\rangle-\tau_{\ell,c}
\right]_{+},
$$

where $[z]_{+}=\max(z,0)$. When the projection onto $\mathbf{v}_{\ell}$ exceeds $\tau_{\ell,c}$, the operator removes the excess component along that direction; activations below the threshold remain unchanged.

**中文：** 我们考察公开的模型专属 Assistant-Axis 阈值钳制。令 $\mathbf{v}_{\ell}$ 表示公开 capping 配置在第 $\ell$ 层存储的单位干预方向，$\tau_{\ell,c}$ 表示配置 $c$ 在该层使用的阈值。该存储方向与对应的 Assistant Axis 向量方向相反，阈值也在这一坐标下完成校准。因此，下文对 $\mathbf{v}_{\ell}$ 的上限钳制，与原论文沿 Assistant-Axis 方向定义的下限钳制等价。对于残差激活 $\mathbf{h}_{\ell,t}$，算子按上式进行变换，其中 $[z]_{+}=\max(z,0)$。当激活在 $\mathbf{v}_{\ell}$ 上的投影超过 $\tau_{\ell,c}$ 时，算子移除超出阈值的方向分量；低于阈值的激活保持不变。

The full behavioral audit and diagnostic analyses use Qwen3-32B with the released `layers_46:54-p0.25` configuration, covering decoder layers 46--53. We repeat the matched behavioral audit on Llama-3.3-70B-Instruct using its released model-specific axis and `layers_56:72-p0.25` configuration. For Gemma-2-27B-IT, the release provides a model-specific axis but no capping layer range, so we select its treated layers and thresholds using the calibration procedure described in *[TBD-APPENDIX]*; the resulting configuration is *[TBD-GEMMA]*. Each model uses its own direction and intervention settings, and all effects are estimated separately by model. Exact model revisions, asset hashes, decoding settings, and calibration grids appear in *[TBD-APPENDIX]*.

**中文：** 完整的行为审计和诊断分析以 Qwen3-32B 为基础，使用公开的 `layers_46:54-p0.25` 配置，对应 decoder 第 46--53 层。我们在 Llama-3.3-70B-Instruct 上按照匹配协议重复行为审计，并使用其公开的模型专属方向和 `layers_56:72-p0.25` 配置。Gemma-2-27B-IT 的公开资产包含模型专属方向，但没有指定钳制层范围，因此我们按照 *[TBD-APPENDIX]* 中的校准程序选择干预层和阈值，所得配置为 *[TBD-GEMMA]*。每个模型均使用各自的方向和干预设置，效应也按模型分别估计。具体模型版本、资产哈希、解码设置和校准网格见 *[TBD-APPENDIX]*。

### 3.5 Behavioral Outcomes and Logit Margins / 行为结果与 Logit 边际

We analyze each run at two levels. The final response is assigned to one of four categories: the initial option, the proposed target, another option, or invalid/abstaining. Let $\ell(a)$ denote the next-option logit. For four-option tasks, we define

$$
m_{\mathrm{general}}
=
\frac{1}{3}\sum_{a\neq a_0}\ell(a)-\ell(a_0),
$$

$$
m_{\mathrm{target}}=\ell(a_t)-\ell(a_0),
$$

$$
m_{\mathrm{background}}
=
\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\ell(a)-\ell(a_0),
$$

$$
m_{\mathrm{specific}}
=
\ell(a_t)-\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\ell(a).
$$

The four margins distinguish broad movement away from $a_0$ from movement toward particular alternatives. $m_{\mathrm{general}}$ summarizes all non-initial options relative to $a_0$, whereas $m_{\mathrm{target}}$, $m_{\mathrm{background}}$, and $m_{\mathrm{specific}}$ locate that movement within the option space. A positive change in $m_{\mathrm{general}}$ indicates a smaller initial-answer advantage.

We report these margins together with target-following rates and complete four-state transitions. CEDAR-Evidence evaluates reason selectivity through paired $V-F$ contrasts in both continuous margins and target-following behavior, with $N/R$ serving as controls. CEDAR-Social uses the $S-AO$ contrast and its cap-by-arm interaction to estimate the incremental effect of social phrasing. Continuous margins describe answer geometry, whereas state transitions record realized choices.

**中文：** 我们从两个层面分析每次运行。最终输出分为四类：初始答案、目标选项、其他选项，以及无效输出或弃答。令 $\ell(a)$ 表示 next-option logit，并按上式定义四个 margin。这四个 margin 用于区分离开 $a_0$ 的整体位移及其在不同备选答案间的分布。$m_{\mathrm{general}}$ 概括所有非初始选项相对于 $a_0$ 的变化，$m_{\mathrm{target}}$、$m_{\mathrm{background}}$ 和 $m_{\mathrm{specific}}$ 则进一步定位该变化在选项空间中的去向。$m_{\mathrm{general}}$ 的正向变化表示初始答案优势缩小。

我们同时报告这些 margin、目标跟随率和完整的四状态迁移。CEDAR-Evidence 通过连续 margin 和目标跟随行为中的配对 $V-F$ 对比评估理由选择性，并以 $N/R$ 作为对照。CEDAR-Social 使用 $S-AO$ 对比及其 cap-by-arm 交互项估计社会性措辞的增量效应。连续 margin 描述答案几何，四状态迁移记录最终落实的选择。

### 3.6 Statistical Inference and Quality Control / 统计推断与质量控制

Uncertainty for CEDAR-Evidence is estimated at the structural-family level. The independently regenerated bank uses 20,000 paired cluster-bootstrap resamples of families. CEDAR-Social uses the SycoBench question as its resampling unit, with 10,000 bootstrap draws stratified by domain and difficulty.

All scheduled items enter the intention-to-treat analysis. Truncated or unparsable responses are coded as invalid/abstaining, preserving the recorded model behavior in the analysis. Development, independently regenerated, natural-task, fixed-trace, and cross-model results are analyzed as distinct evidence streams.

We use option mass, entropy, and top-token integrity as diagnostic checks for generic logit flattening and output corruption. Decompositions of the initial-, target-, and background-option logits show how each shift is distributed across answer roles. Complete estimand definitions, random seeds, multiplicity adjustments, missing-data sensitivity analyses, and protocol deviations appear in *[TBD-APPENDIX]*.

**中文：** CEDAR-Evidence 在结构族层面估计不确定性；独立再生成的评测批次以结构族为单位进行 20,000 次配对 cluster bootstrap。CEDAR-Social 以 SycoBench 题目为重采样单位，并按领域和难度分层进行 10,000 次 bootstrap。

所有预定样本均进入 intention-to-treat 分析。截断或无法解析的输出被编码为无效输出或弃答，使模型的实际表现保留在分析中。开发集、独立再生成集、自然任务、固定轨迹和跨模型结果分别作为不同证据来源进行分析。

我们使用有效选项总概率、熵以及最高概率 token 是否属于有效选项，检查一般性的 logit 扁平化和输出损坏。初始选项、目标选项与背景选项的 logit 分解进一步显示位移在不同答案角色间的分布。完整的估计量定义、随机种子、多重比较校正、缺失数据敏感性分析和协议偏离见 *[TBD-APPENDIX]*。

## 4. Auditing Assistant-Axis Capping / 审计 Assistant-Axis 钳制

### 4.1 The Axis Is Not the Operator / 表示轴不等于干预算子

`[TBD: final prose and verified source comparison]`

### 4.2 Capping Broadly Compresses Initial-Answer Advantage / 钳制广泛压缩初始答案优势

`[TBD: final analysis prose and Figure 2 values]`

### 4.3 The Shift Is Not Reliably Reason-Selective / 位移并不可靠地依据理由进行选择

`[TBD: final V/F and N/R analysis]`

### 4.4 New-Salt and Natural-Task Evidence / New-Salt 与自然任务证据

`[TBD: E2C and CEDAR-Social synthesis]`

### 4.5 Cross-Model Transport and Heterogeneity / 跨模型迁移与异质性

`[TBD: matched Qwen/Llama/Gemma results and analysis]`

## 5. Diagnosing the Shift / 诊断该位移

### 5.1 Phase-Localized Interventions / 分阶段干预

`[TBD]`

### 5.2 Validity Assessment versus Final Choice / 有效性判断与最终选择

`[TBD]`

### 5.3 Testing the Selective-Resistance Hypothesis / 检验选择性抵抗假说

`[TBD]`

### 5.4 Fixed-Trace Replay / 固定轨迹重放

`[TBD]`

### 5.5 From Local Rectification to Downstream Geometry / 从局部整流到下游几何

`[TBD-RG1]`

## 6. Implications, Scope, and Future Control / 含义、范围与未来控制

### 6.1 Representation Is Not Policy / 表示不等于策略

`[TBD]`

### 6.2 What a Selective Controller Must Preserve / 选择性控制器必须保留什么

`[TBD]`

### 6.3 Evidence Scope and Limitations / 证据范围与局限

`[TBD]`

### 6.4 Future Work / 未来工作

`[TBD]`

## 7. Conclusion / 结论

`[TBD: one evidence-bounded conclusion paragraph]`

`[TBD：一段严格受证据边界约束的结论]`

## Internal Citation Queue / 内部引用队列

No BibTeX key is invented. Verify metadata before adding Assistant Axis, Dual-Stance Evaluation of Sycophancy, When Truth Is Overridden, the refusal-direction paper, WMDP, Belief-R, SycoBench-600, CAA, ITI, KL-then-steer, BASIL, T3, and the nearest July 2026 work.

**中文：** 不得编造 BibTeX key。加入 Assistant Axis、Dual-Stance Evaluation of Sycophancy、When Truth Is Overridden、refusal-direction、WMDP、Belief-R、SycoBench-600、CAA、ITI、KL-then-steer、BASIL、T3 及 2026 年 7 月最近邻工作前，必须核验元数据。
