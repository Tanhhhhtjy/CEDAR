# CEDAR AAAI-27 Working Draft / CEDAR AAAI-27 论文工作稿 v0.4

> This bilingual working draft mirrors the manuscript structure in Overleaf. The English submission source is `main.tex`; the Chinese review companion is `main_zh.tex`.
>
> **中文：** 本双语工作稿与 Overleaf 论文结构保持一致。英文投稿源为 `main.tex`，中文审阅稿为 `main_zh.tex`。
>
> Fixed drafting rule: write in completed-study prose. Missing configurations, values, analyses, or result-dependent conclusions occupy narrow `TBD` slots; the paper does not tell the reader that an experiment is unfinished.
>
> **中文：** 固定写作规则：正文始终采用研究已经完整执行的成稿口吻。尚缺的配置、数值、分析或依赖结果的结论只放入窄范围 `TBD` 槽位；正文不向读者说明某项实验尚未完成。

## Provisional Title / 暂定标题

**An Axis Is Not a Selective Controller: Auditing Epistemic Selectivity under Assistant-Axis Capping**

**表征轴不等同于选择性控制器：Assistant-Axis 钳制的认识论选择性审计**

## Abstract / 摘要

`[TBD: final five-sentence abstract with audited Qwen/Llama results]`

`[TBD：填入包含经审计 Qwen/Llama 结果的最终五句式摘要]`

## 1. Introduction / 引言

Paragraph contract / 段落接口：

1. Runtime activation interventions can alter model behavior without retraining.<br>
   **中文：** 推理时激活干预无需重新训练即可改变模型行为。
2. Existing evaluations often pair the intended behavioral change with broad capability tests or checks on benign inputs, providing limited evidence about sensitivity to the grounds for revision.<br>
   **中文：** 现有评测常把预期行为变化与一般能力测试或良性输入检查一并报告，但这些结果对模型能否依据改答理由作出区分提供的信息有限。
3. Intervention evaluation poses two separate comparison problems: isolating social framing beyond target exposure and comparing reasons that warrant revision with matched reasons that do not.<br>
   **中文：** 干预评测包含两类分别分析的比较：隔离超出 target exposure 的社会性措辞影响，以及比较足以支持修正的理由与匹配的无效理由。
4. CEDAR makes these comparisons auditable through two complementary modules that are analyzed separately.<br>
   **中文：** CEDAR 通过两个互补且分别分析的模块，使这些比较可以被审计。
5. `[TBD: concise audited headline result across continuous geometry, discrete updating, and cross-model heterogeneity.]`<br>
   **中文：** `[TBD：填入涵盖连续几何、离散更新与跨模型异质性的经审计标题级结果。]`
6. Three contributions: framework and evaluation assets, a scoped empirical audit, and controlled diagnostics.<br>
   **中文：** 三项贡献：框架与评测资产、限定范围的实证审计，以及受控诊断。

Approved contribution architecture / 已确认的贡献架构：

1. **Framework and evaluation assets.** We formulate epistemic selectivity as a distinct evaluation target and introduce CEDAR. It comprises CEDAR-Evidence, a new solver-certified paired evaluation dataset, and CEDAR-Social, a paired protocol over SycoBench-600 that is analyzed separately.<br>
   **中文：框架与评测资产。** 我们将认识论选择性形式化为独立评测目标并提出 CEDAR。CEDAR 包含新的 solver-certified 配对评测数据集 CEDAR-Evidence，以及在 SycoBench-600 上构造并单独分析的 CEDAR-Social 配对协议。
2. **Scoped empirical audit.** We audit Assistant-Axis threshold capping on Qwen and Llama, retaining model-specific configurations, uncertainty, and heterogeneity.<br>
   **中文：限定范围的实证审计。** 我们在 Qwen 与 Llama 上审计 Assistant-Axis 阈值钳制，并保留模型专属配置、不确定性与异质性。
3. **Controlled diagnostics.** We use phase-localized intervention, validity-versus-choice analysis, fixed-trace replay, and controlled representation analysis to diagnose the observed shift without exceeding the available causal evidence.<br>
   **中文：受控诊断。** 我们使用分阶段干预、有效性与选择分解、fixed-trace replay 和受控表示分析诊断观测到的位移，且不使用超出因果证据层级的措辞。

## 2. Related Work / 相关工作

**Inference-Time Activation Control. / 推理时激活控制。**

Post-training pipelines such as supervised fine-tuning and preference optimization remain a common route to aligning model behavior (Ouyang et al. 2022; Rafailov et al. 2023). Activation-level methods provide a complementary control surface at inference time, modifying internal states without updating base-model weights. Representation engineering treats population-level activation patterns as objects that can be read and manipulated (Zou et al. 2023). Existing methods span additive steering along contrastive directions (Turner et al. 2023; Rimsky et al. 2024), localized interventions on attention heads (Li et al. 2023), directional ablation (Arditi et al. 2024), and context-conditioned control (Lee et al. 2025).<br>
**中文：** 监督微调和偏好优化等训练后方法，是使模型行为符合预期的常用途径（Ouyang et al. 2022；Rafailov et al. 2023）。与此同时，激活层方法提供了互补的推理期控制途径，可在不更新基础模型权重的情况下直接调整内部状态。表征工程将样本总体上的激活模式作为可读取和操控的对象（Zou et al. 2023）。现有方法包括沿对比方向进行加性 steering（Turner et al. 2023；Rimsky et al. 2024）、对注意力头实施局部干预（Li et al. 2023）、方向消融（Arditi et al. 2024），以及依赖上下文的条件控制（Lee et al. 2025）。

Threshold capping defines a state-dependent operator: the controlled projection remains unchanged below a calibrated boundary, and the cap removes only the excess above it. The Assistant Axis work extracts model-specific directions associated with the default assistant persona and pairs them with one-sided, percentile-calibrated caps (Lu et al. 2026). CEDAR audits the released model-specific caps as instances of this intervention, including their effects on responses to social pressure and corrections supported by sufficient evidence.<br>
**中文：** 阈值钳制定义了一种状态依赖算子：受控投影低于校准边界时保持不变，超过边界时仅移除超出的部分。Assistant Axis 工作为不同模型提取与默认助手角色相关的方向，并为其配置按分位数校准的单侧 cap（Lu et al. 2026）。CEDAR 将公开的模型专属 cap 作为这一干预的实例，审计其对社会压力响应和有充分依据的修正所产生的影响。

**Sycophancy and Answer Revision. / 谄媚与答案修正。**

Social influence makes the grounds for answer revision difficult to infer from behavior alone. Language models often shift answers toward a user's stated view, including when that view conflicts with the correct answer (Sharma et al. 2024). SycoBench-600 measures susceptibility to doubt, authority, and explicit wrong suggestions alongside correction selectivity, showing that willingness to update does not by itself establish selective updating (Sinha 2026). Mechanistic work traces how incorrect user opinions alter answer representations across layers before affecting the final prediction (Wang et al. 2026). Intervention evaluation therefore requires explicit controls for the grounds of revision. This produces two comparison problems: isolating social framing beyond target exposure and comparing reasons that warrant revision with matched reasons that do not.<br>
**中文：** 社会影响使得仅凭外显行为很难判断模型改答的依据。语言模型常会把答案移向用户表达的观点，包括该观点与正确答案冲突时（Sharma et al. 2024）。SycoBench-600 同时测量模型对怀疑、权威和明确错误建议的易感性及其修正选择性，并表明愿意改答本身不足以证明更新具有选择性（Sinha 2026）。机制研究进一步追踪错误用户意见如何在最终预测改变前逐层改变答案表征（Wang et al. 2026）。因此，干预评测需要明确控制促使模型改答的依据，包括两类分别分析的比较：隔离超出 target exposure 的社会性措辞影响，以及比较足以支持修正的理由与匹配的无效理由。

**Reason-Sensitive Evaluation and Intervention Side Effects. / 理由敏感评测与干预副作用。**

Belief-revision benchmarks provide criteria for judging such answer changes. Belief-R treats answer revision as a two-sided problem: models should incorporate premises that change the conclusion and preserve conclusions that remain supported (Wilie et al. 2024). ProverGen combines language-model generation with symbolic proving to construct problems with checkable first-order structure (Qi et al. 2025), while LoFa evaluates robustness to fallacious persuasion after accounting for limitations in base knowledge (Shen et al. 2026). These resources support controlled evaluation of reason quality. A parallel line of work evaluates intervention side effects by pairing target behavior with retention criteria. KL-then-steer limits distributional change on benign inputs while maintaining additive-steering efficacy (Stickland et al. 2024); dual-stance evaluation shows that centroid-difference anti-sycophancy steering can also reduce agreement with factually correct statements (Buchan 2026). An audit of state-dependent capping therefore needs estimates of the intervention's effect on warranted-versus-unwarranted revision and on the additional target following attributable to social framing beyond target exposure.<br>
**中文：** 信念修正 benchmark 为判断这类答案变化提供了判据。Belief-R 将答案修正视为一个双向问题：模型应当吸收足以改变结论的新前提，同时维持仍有充分支持的原结论（Wilie et al. 2024）。ProverGen 将语言模型生成与符号证明结合，构造具有可核验一阶逻辑结构的题目（Qi et al. 2025）；LoFa 则在控制基础知识不足后评估模型抵抗谬误说服的能力（Shen et al. 2026）。这些资源支持对理由质量进行受控评测。与之并行，另一类工作借助保留指标评估干预副作用。KL-then-steer 在维持加性 steering 效果的同时限制良性输入上的分布变化（Stickland et al. 2024）；dual-stance evaluation 表明，基于 centroid difference 的反谄媚 steering 也可能降低模型对事实正确陈述的赞同（Buchan 2026）。因此，对状态依赖 capping 的审计需要估计干预对有据与无据修正差异的影响，以及它对社会性措辞在 target exposure 之上所引起的额外 target following 的影响。

## 3. CEDAR: Auditing Epistemic Selectivity / CEDAR：审计认识论选择性

CEDAR combines CEDAR-Evidence, a new solver-certified paired dataset, with CEDAR-Social, a separate protocol over natural tasks. The modules estimate responses to reason validity and social framing with distinct estimands and sampling units. We next define their shared answer-change setting, interventions, outcomes, and inference.

**中文：** CEDAR 包含两个模块：CEDAR-Evidence 是一套经求解器认证的新配对数据集，CEDAR-Social 是基于自然任务的独立评测协议。二者分别估计模型对理由有效性和社会性措辞的响应，并采用不同的估计量与统计单位。下文依次定义共同的答案改变设定、干预、行为指标和统计推断。

### 3.1 Shared Answer-Change Setting and Controlled Contrasts / 答案改变的共同设定与受控对照

Each evaluation begins with a dialogue containing a base question and a supplied initial answer $a_0$. We associate the base with a different, paired target $a_t$ used to classify the subsequent response. A parseable final option different from $a_0$ is an *answer change*; a final answer equal to $a_t$ is *target following*. When the follow-up supplies sufficient support for $a_t$, target following constitutes a *warranted revision*.

CEDAR-Evidence fixes the paired target across four conditions: $N/R$ retain it for analysis without revealing it, whereas $V/F$ present the same update with valid or fallacious justification. CEDAR-Social reveals $a_t$ in both $AO/S$ and varies the accompanying social framing. Each comparison fixes the base, $a_0$, target identity, and response format (Figure 1).

**中文：** 每项评测都从一段包含基础问题和预置初始答案 $a_0$ 的对话开始。每个基础样本还关联一个不同于 $a_0$ 的配对目标 $a_t$，用于划分后续回答。可解析的最终选项与 $a_0$ 不同时，记为*答案改变*；最终答案等于 $a_t$ 时，记为*目标跟随*。当后续信息为 $a_t$ 提供充分支持时，目标跟随构成*有充分依据的改答*。

CEDAR-Evidence 的四种条件使用同一个配对目标：$N/R$ 在分析中保留该目标，但不向模型展示；$V/F$ 呈现同一候选修改，论证分别有效或包含谬误。CEDAR-Social 的 $AO/S$ 都展示 $a_t$，仅改变伴随的社会性措辞。各项对比均固定基础题、$a_0$、目标身份和回答格式（Figure 1）。

**Figure 1 / 图 1：** CEDAR paired, reason-sensitive audit design / CEDAR 成对理由敏感审计设计。

**Caption / 图注：** **CEDAR's paired, reason-sensitive audit design.** CEDAR-Evidence holds the formal-task instance, initial answer $a_0$, and target answer $a_t$ fixed while varying the follow-up prompt. Neutral ($N$) and Recheck ($R$) serve as controls; Valid ($V$) provides sufficient support for updating to $a_t$; and Fallacy ($F$) proposes the same update through an invalid inference. CEDAR-Social holds the natural-task instance and answer pair fixed: Target re-mention ($AO$) exposes $a_t$ without task evidence, whereas Confident endorsement ($S$) adds an explicit social endorsement. Comparing uncapped and activation-capped responses separates whether $a_t$ becomes more likely from whether that movement is warranted, while the $S-AO$ contrast isolates the increment attributable to social pressure. Additional prompt templates and construction details appear in *[TBD-APPENDIX]*.

**中文图注：** **CEDAR 的成对理由敏感审计设计。** CEDAR-Evidence 固定形式化任务实例、初始答案 $a_0$ 和目标答案 $a_t$，仅改变后续提示。Neutral（$N$）与 Recheck（$R$）构成控制条件；Valid（$V$）为更新至 $a_t$ 提供充分支持；Fallacy（$F$）则通过无效推理提出同一更新。CEDAR-Social 固定自然任务实例及答案对：Target re-mention（$AO$）仅再次呈现 $a_t$，不提供任务证据；Confident endorsement（$S$）在此基础上加入明确的社会性背书。比较未钳制与激活钳制后的响应，可以区分 $a_t$ 是否变得更可能与这一变化是否具有充分理由；$S-AO$ 对比则隔离由社会压力带来的增量。更多提示模板与构造细节见 *[TBD-APPENDIX]*。

### 3.2 CEDAR-Evidence: A Solver-Certified Paired Dataset / CEDAR-Evidence：经求解器认证的配对数据集

CEDAR-Evidence pairs formally valid and invalid justifications whose status is determined independently of model responses. Each item specifies a finite, function-free Horn theory with explicit negation and a four-way query: supported, undetermined, contradicted, or both. The base theory has the unique answer $a_0$; a deterministic atomic edit yields the paired target $a_t$. $N$ continues neutrally, and $R$ requests rechecking without new information. $V$ and $F$ present the same candidate edit: $V$ provides a solver-certified justification that warrants $a_t$, whereas $F$ contains one certified invalid inference and does not warrant changing $a_0$.

**中文：** CEDAR-Evidence 将形式有效与无效的论证配对，并在模型作答前确定其规范标签。每个样本包含一个带显式否定且不含函数符号的有限 Horn 逻辑理论，以及一道四分类查询：得到支持、无法确定、遭到反驳，或同时得到支持与反驳。基础理论的唯一答案是 $a_0$；一项确定性的原子修改产生配对目标 $a_t$。$N$ 中性延续对话，$R$ 在不增加信息的情况下请求复核。$V$ 与 $F$ 呈现相同的候选修改：$V$ 提供经求解器认证、足以支持 $a_t$ 的论证；$F$ 则包含一处经认证的无效推理，不构成改变 $a_0$ 的充分依据。

A Python fixed-point engine and Prover9 independently verify the labels and whether each justification warrants the edit; retained instances require agreement. Every $F$ argument contains one certified invalid step and a Mace4 countermodel, with all remaining steps supported. The prompt scaffold and response format are matched within $V/F$ pairs. Certificates and review records appear in *[TBD-APPENDIX]*.

Each evaluation bank contains 24 bases: easy and medium variants of 12 structural families spanning six error classes. A second bank uses a fresh generation salt to instantiate the same task grammar and family structures, and is analyzed separately. Structural families are the independent sampling units; conditions and model runs are paired within family.

**中文：** Python 不动点推理器与 Prover9 均独立核验对象层标签及论证能否支持候选修改；仅保留二者结论一致的样本。每个 $F$ 论证都包含一处经认证的无效推理和一个 Mace4 反模型，其余步骤均有支持。$V/F$ 配对使用相同的提示框架和回答格式。完整证书和审核记录见 *[TBD-APPENDIX]*。

每个评测 bank 包含 24 个基础样本，由 12 个结构族的简单与中等难度变体组成，共覆盖六类错误。第二个 bank 使用新的随机盐，对同一任务语法和结构族重新实例化，并单独分析。结构族是统计推断的独立单位；条件和模型运行在结构族内配对。

### 3.3 CEDAR-Social: Controlling for Target Exposure / CEDAR-Social：控制目标暴露效应

CEDAR-Social applies the answer-change setting to the 600 natural multiple-choice questions in SycoBench-600. Each dialogue supplies the gold option as $a_0$ and a different option as $a_t$. Both follow-ups present $a_t$. The answer-only condition ($AO$) contains no speaker, correctness, evidence, authority, reward, or consensus cues. The social condition ($S$) adds a first-person expression of confidence and a request for reconsideration, without adding task-relevant facts or reasons. The paired $S-AO$ contrast estimates the incremental contribution of social framing.

Four minimally edited template pairs are balanced across domain and difficulty strata. Automated checks verify matching and token-length bounds under the frozen tokenizer. Two researchers independently review all pairs for template fidelity and linguistic naturalness without access to model outputs; agreement and adjudication are *[TBD-SOCIAL-AUDIT]*. Templates and review materials appear in *[TBD-APPENDIX]*.

**中文：** CEDAR-Social 将答案改变设定用于 SycoBench-600 的 600 道自然多选题。每段对话将标准答案作为 $a_0$，另一个选项作为 $a_t$；两种后续输入都展示 $a_t$。纯目标暴露条件（$AO$）不加入说话者、正确性、证据、权威、奖励或共识线索。社会条件（$S$）增加第一人称确信和请求重新考虑的表达，但不提供与题目有关的事实或理由。配对的 $S-AO$ 对比估计社会性措辞的增量作用。

四组最小改写模板在领域和难度层内均衡分配。自动检查核验配对一致性及冻结分词器下的 token 长度范围。两名研究人员在不接触模型输出的情况下，分别检查全部配对的模板一致性和语言自然度；一致率和分歧裁决为 *[TBD-SOCIAL-AUDIT]*。模板与审核材料见 *[TBD-APPENDIX]*。

### 3.4 Intervention and Model Scope / 干预与模型范围

We evaluate the released, model-specific Assistant-Axis threshold cap. At layer $\ell$, let $\mathbf{v}_{\ell}$ be the unit direction stored in the released configuration and $\tau_{\ell,c}$ its threshold under configuration $c$. Because $\mathbf{v}_{\ell}$ is oriented opposite to the human-readable Assistant Axis, the released minimum cap is an upper cap in this coordinate:

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

where $[z]_{+}=\max(z,0)$. The operator removes only the projection above $\tau_{\ell,c}$.

We use the released directions, layer ranges, thresholds, and stored sign convention verbatim.

**中文：** 我们评测公开的模型专属 Assistant-Axis 阈值钳制。令 $\mathbf{v}_{\ell}$ 表示公开配置在第 $\ell$ 层存储的单位方向，$\tau_{\ell,c}$ 表示配置 $c$ 在该层使用的阈值。由于 $\mathbf{v}_{\ell}$ 与可读的 Assistant Axis 方向相反，原始 minimum cap 在这一坐标中对应 upper cap。对于残差激活 $\mathbf{h}_{\ell,t}$，算子按上式进行变换，其中 $[z]_{+}=\max(z,0)$。该算子仅移除超过 $\tau_{\ell,c}$ 的投影分量。我们原样沿用公开资产中的方向、层范围、阈值和存储符号约定。

Qwen3-32B is the primary model, using the released `layers_46:54-p0.25` configuration (layers 46--53). Its primary Evidence assays use reasoning-enabled greedy decoding with seed 0, frozen before any capped run. Because decoding regime materially changes baseline revision behavior, all estimates are conditional on the assay-specific frozen regime; cross-regime comparisons are reported separately as sensitivity analyses.

For model--configuration transport, Qwen3-32B and Llama-3.3-70B-Instruct use the same frozen 24-base, 12-family Evidence bank, reasoning disabled, deterministic forced choice, and uncapped/p25 conditions. Llama uses its released axis and `layers_56:72-p0.25` configuration (layers 56--71); a regime-matched Qwen run provides the reference. Each model retains its released direction, layer range, and threshold, and effects are estimated within model. The estimates therefore describe the two tested model--configuration pairs; the design does not separate model identity from configuration. CEDAR-Social and diagnostic analyses remain Qwen-only. Exact revisions and decoding settings appear in *[TBD-APPENDIX]*.

**中文：** Qwen3-32B 是主要模型，使用公开的 `layers_46:54-p0.25` 配置（第 46--53 层）。其主要 Evidence assay 开启 reasoning，采用 greedy 解码和 seed 0，并在任何 capped 运行前冻结。鉴于解码制度会实质改变基线改答行为，所有估计均以各 assay 的冻结制度为条件；跨制度比较单独作为敏感性分析报告。

跨模型迁移评测在 Qwen3-32B 与 Llama-3.3-70B-Instruct 上使用同一个冻结的 24 题、12 结构族 Evidence bank，并统一关闭推理模式、采用确定性 forced choice，比较未钳制与 p25。Llama 使用其公开方向和 `layers_56:72-p0.25` 配置（第 56--71 层），以运行制度匹配的 Qwen 结果为参照。各模型沿用自身发布的方向、层范围和阈值，效应在模型内估计。因此，结果描述两个模型—配置组合，设计本身无法分离模型与配置的作用。CEDAR-Social 和诊断分析仅在 Qwen 上进行。具体模型版本和解码设置见 *[TBD-APPENDIX]*。

### 3.5 Behavioral Outcomes and Logit Margins / 行为结果与 Logit 边际

Changes in option support can precede a change in the selected answer. We therefore record both four-way final outcomes and continuous logit margins. A final response is assigned to the initial option, the paired target, another option, or invalid/abstaining. Let $z(a)$ denote the next-token logit of option label $a$ at the fixed answer position. For four-option tasks, define

$$
m_{\mathrm{general}}
=
\frac{1}{3}\sum_{a\neq a_0}z(a)-z(a_0),
$$

$$
m_{\mathrm{target}}=z(a_t)-z(a_0),
$$

$$
m_{\mathrm{background}}
=
\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}z(a)-z(a_0),
$$

$$
m_{\mathrm{specific}}
=
z(a_t)-\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}z(a).
$$

Here $m_{\mathrm{general}}$ measures broad movement away from $a_0$; a positive capped--uncapped change indicates a smaller initial-answer advantage. The remaining margins locate the shift across the target and background options, with $m_{\mathrm{specific}}$ comparing the target directly with the two background alternatives. We also report changes in the initial, target, and mean background logits to identify which answer roles contribute to each margin shift.

For outcome $Y$, follow-up condition $x$, and cap configuration $c$, let

$$
\delta_x^Y(c)=\mathbb{E}[Y\mid x,c]-\mathbb{E}[Y\mid x,0],
$$

where $0$ denotes uncapped inference. The Evidence estimands are

$$
\Delta_{\mathrm{broad}}(c)
=\frac{1}{4}\sum_{x\in\{N,R,V,F\}}\delta_x^{m_{\mathrm{general}}}(c),
$$

$$
\Delta_{\mathrm{sel}}^{(m)}(c)
=\delta_V^{m_{\mathrm{target}}}(c)-\delta_F^{m_{\mathrm{target}}}(c),
$$

$$
\Delta_{\mathrm{sel}}^{(p)}(c)
=[p_V(c)-p_V(0)]-[p_F(c)-p_F(0)],
$$

where $p_x(c)$ is the target-following probability. The selectivity contrasts measure how capping changes the model's differential response to valid and fallacious reasons. The pooled change in $m_{\mathrm{specific}}$ is a prespecified target-specificity check; condition-wise decompositions are post hoc.

CEDAR-Social uses

$$
\Delta_{\mathrm{social}}(c)
=\delta_S^{m_{\mathrm{target}}}(c)-\delta_{AO}^{m_{\mathrm{target}}}(c)
=[S-AO]_c-[S-AO]_0.
$$

A negative value indicates that capping reduces the additional target support associated with social framing beyond target exposure. We report the $S$ and $AO$ component effects with the interaction. Direct next-option margins are the primary Social readout; a frozen 240-question subset provides secondary free-generation outcomes.

**中文：** 选项支持度的变化可能先于最终答案改变，因此我们同时记录四类最终结果和连续 logit margin。最终输出分为初始答案、配对目标、其他选项，以及无效输出或弃答。令 $z(a)$ 表示固定答案位置上选项标签 $a$ 的 next-token logit，并按上式定义四个 margin。其中，$m_{\mathrm{general}}$ 衡量整体离开 $a_0$ 的位移；其 capped--uncapped 差值为正时，初始答案优势缩小。其余 margin 定位该位移在目标选项和背景选项之间的分布，$m_{\mathrm{specific}}$ 直接比较目标与两个背景选项。我们还分别报告初始选项、目标选项和背景选项平均 logit 的变化，以判断各类选项对 margin 位移的贡献。

对于结果 $Y$、后续条件 $x$ 和钳制配置 $c$，$\delta_x^Y(c)$ 表示该条件相对于未钳制推理的效应。$\Delta_{\mathrm{broad}}$ 汇总四个 Evidence 条件中的 $m_{\mathrm{general}}$ 变化；$\Delta_{\mathrm{sel}}^{(m)}$ 和 $\Delta_{\mathrm{sel}}^{(p)}$ 分别从连续 margin 与目标跟随概率衡量模型对有效理由和谬误理由的差异化响应。汇总的 $m_{\mathrm{specific}}$ 变化是预设的目标特异性检查；按条件分解属于 post hoc 分析。

CEDAR-Social 使用 $\Delta_{\mathrm{social}}$ 衡量钳制对 $S-AO$ 差异的影响。负值表示钳制降低了社会性措辞相对于纯目标暴露带来的额外目标支持。我们将该交互与 $S$ 和 $AO$ 的分量效应一并报告。直接 next-option margin 是 CEDAR-Social 的主要读数；冻结的 240 题子集提供自由生成的次要结果。

### 3.6 Statistical Inference and Diagnostic Checks / 统计推断与诊断检查

CEDAR-Evidence treats its 12 structural families as the independent units. Paired capped--uncapped differences are averaged within family, and families receive equal weight. Primary intervals use the frozen family-cluster bootstrap; family effects and equal-weight leave-one-family-out analyses assess influence. Small-cluster sensitivities appear in *[TBD-APPENDIX]*. CEDAR-Social resamples questions with domain/difficulty stratification.

All scheduled evaluations enter the discrete analysis; truncated or unparsable responses are coded as invalid/abstaining and are not selectively rerun based on content. Continuous contrasts require the relevant option logits in both paired conditions, with incomplete coverage disclosed beside the result. Effects are estimated separately by data bank, evaluation protocol, and model. Cross-model results report both within-model effects and intervals on their native scales. A family-paired interaction standardized by each model's uncapped scale appears in *[TBD-APPENDIX]*; a nonsignificant interaction does not establish equivalence.

The development bank is exploratory. Primary contrasts are $\Delta_{\mathrm{broad}}$ for the new-salt and matched Llama studies and $\Delta_{\mathrm{social}}$ for CEDAR-Social; the matched Qwen estimate serves as the reference for Llama. Other analyses are labeled prespecified secondary, sensitivity, or post hoc. The four-margin decomposition, the equivalence region for pooled $m_{\mathrm{specific}}$, and the decision rules for follow-up analyses were frozen in a CPU-only protocol gate before any capped-model execution. Diagnostics are the total full-vocabulary probability mass assigned to the four option labels, full-vocabulary Shannon entropy, and whether the vocabulary-wide argmax is an option label. Complete sensitivities appear in *[TBD-APPENDIX]*; material amendments are also disclosed with the affected result.

**中文：** CEDAR-Evidence 将 12 个结构族作为独立统计单位。每项对比先在结构族内汇总 capped--uncapped 配对差异，再对各结构族等权平均。主要区间使用冻结的结构族聚类 bootstrap；同时报告结构族效应和等权的 leave-one-family-out 分析。小聚类敏感性分析见 *[TBD-APPENDIX]*。CEDAR-Social 以题目为重采样单位，并按领域和难度分层。

全部预定评测均进入离散分析；截断或无法解析的输出归为无效输出或弃答，且不根据输出内容选择性重跑。连续对比要求配对条件中相关选项 logit 均完整，并随结果披露不完整覆盖。不同数据批次、评测协议和模型的效应分别估计。跨模型结果在原始尺度上报告两组模型内效应及区间。以各模型未钳制尺度标准化、并保持结构族配对的交互见 *[TBD-APPENDIX]*；交互项不显著不构成等效证据。

development bank 用于探索。new-salt 与匹配 Llama 研究的主要对比是 $\Delta_{\mathrm{broad}}$，CEDAR-Social 的主要对比是 $\Delta_{\mathrm{social}}$；匹配的 Qwen 估计作为 Llama 的参照。其余分析标记为预先指定的次要分析、敏感性分析或 post hoc 分析。四项 margin 分解、汇总 $m_{\mathrm{specific}}$ 的等效区间，以及决定后续分析路径的规则，均在任何 capped 模型运行前通过 CPU-only 协议闸冻结。诊断指标包括完整词表分布中分配给四个选项标签的总概率质量、完整词表 Shannon 熵，以及词表范围内的 argmax 是否为选项标签。完整敏感性分析见 *[TBD-APPENDIX]*；实质性方案调整也随受影响的结果披露。

## 4. Auditing Assistant-Axis Capping / 审计 Assistant-Axis 钳制

Across the Qwen assays, Assistant-Axis p25 capping shifted continuous answer margins, while its effects on selected answers varied across reason conditions and readouts. We first calibrate the released procedures, then audit the separation between continuous margins and final choices in CEDAR-Evidence and CEDAR-Social before testing transport in a matched forced-choice comparison of the released Qwen and Llama configurations.

**中文：** 在各项 Qwen 评测中，Assistant-Axis p25 钳制改变了连续答案 margin，但它对最终选择的影响随理由条件和读出方式而异。我们先校准公开流程，再通过 CEDAR-Evidence 与 CEDAR-Social 审计连续 margin 与最终选择之间的分离，最后在公开 Qwen 与 Llama 配置的匹配 forced-choice 比较中检验其迁移性。

### 4.1 Calibrating Steering Direction and Cap Exposure / 校准 Steering 方向与钳制暴露

Signed steering maps the controlled scalar projection $s$ at an intervention site to $s+\alpha$. A one-sided cap maps $s$ to $\min(s,\tau)$ and activates only above $\tau$. Because the released procedures use different intervention locations and schedules, we calibrate each in its deployed form.

**中文：** Signed steering 在干预位置将受控标量投影 $s$ 映射为 $s+\alpha$。单侧 capping 将 $s$ 映射为 $\min(s,\tau)$，且只在投影超过 $\tau$ 时触发。由于两套公开流程采用不同的干预层位与调度，我们分别校准其实际部署形式。

We first ran a complete seven-dose signed-steering calibration at layer 32 using the source-style PhilPapers setup. At $\alpha=+200$, the sycophancy margin changed by $-2.4312$ logits relative to $\alpha=0$ (95% CI $[-2.6248,-2.2407]$), closely matching the approximately $-2.41$ shift reported by Kelkar et al. (2026). The exploratory $\alpha=-200$ endpoint also produced a negative change of $-1.6757$ logits (95% CI $[-1.8506,-1.4971]$). At both large-magnitude endpoints, the margin shifts were accompanied by changes in top-token identity and full-vocabulary entropy. We therefore treat this margin as a benchmark readout and evaluate behavioral quality and selectivity separately in the target assays below. The full seven-dose curve and distributional diagnostics appear in *[TBD-APPENDIX]*.

**中文：** 我们首先采用源工作风格的 PhilPapers 设定，完成了第 32 层的七剂量 signed-steering 校准。相较于 $\alpha=0$，当 $\alpha=+200$ 时，sycophancy margin 变化为 $-2.4312$ logits（95% CI $[-2.6248,-2.2407]$），与 Kelkar 等（2026）报告的约 $-2.41$ 位移接近。探索性的 $\alpha=-200$ 端点同样产生负向变化，为 $-1.6757$ logits（95% CI $[-1.8506,-1.4971]$）。在正负两个大剂量端点，margin 位移均伴随 top-token identity 和全词表熵的变化。因此，我们将这一 margin 作为 benchmark readout，并在下文目标评测中分别检验行为质量与选择性。完整七剂量曲线和分布诊断见 *[TBD-APPENDIX]*。

We next calibrated the released layerwise p25 procedure on the same source-style PhilPapers benchmark. Across layers 46--53, every capped evaluation received at least one nonzero correction, and $56.66\%$ of eligible token--layer states triggered the cap. The intervention therefore reached every evaluation while modifying only its above-threshold states. Relative to uncapped inference, p25 changed the sycophancy margin by $-0.094$ logits (95% CI $[-0.133,-0.054]$). The discrete sycophancy rate increased by $1.44$ percentage points (post-hoc cluster-bootstrap 95% CI $[0.11,2.78]$), comprising 25 non-sycophantic-to-sycophantic migrations and 12 in the reverse direction. On this benchmark, the continuous margin and discrete sycophancy rate moved in opposite directions. Sections 4.2--4.4 therefore report these readouts separately when analyzing answer revision and reason selectivity.

**中文：** 随后，我们在同一源工作风格的 PhilPapers benchmark 上校准公开的分层 p25 流程。在第 46--53 层中，每次 capped 评测都至少发生一次非零校正，所有可干预 token--layer 状态的实际触发率为 $56.66\%$。因此，该干预覆盖了全部评测样本，同时只修改其中超过阈值的状态。相较于未钳制推理，p25 下的 sycophancy margin 变化为 $-0.094$ logits（95% CI $[-0.133,-0.054]$）。离散 sycophancy rate 则增加 $1.44$ 个百分点（事后 cluster-bootstrap 95% CI $[0.11,2.78]$），其中包括 25 次由 non-sycophantic 到 sycophantic 的迁移和 12 次反向迁移。在这一 benchmark 上，连续 margin 与离散 sycophancy rate 的变化方向相反。因此，§4.2--4.4 在分析答案修正与理由选择性时分别报告这两类读数。

**Figure 2 placeholder:** Broad revision geometry, warranted and unwarranted updating, margin decomposition, and matched Qwen--Llama transport.

**Caption / 图注：** **Answer geometry and revision effects under model-specific p25 capping.** **(A)** In the Qwen3-32B E2A development assay, points show full-p25-minus-uncapped changes in $m_{\mathrm{general}}$ for neutral continuation ($N$), evidence-free recheck ($R$), valid argument ($V$), and single-fault fallacy ($F$); positive values indicate weaker initial-answer advantage relative to the three alternatives. **(B)** Target-following rates under uncapped and p25 inference for $V$ and $F$; the difference between the two arm changes is the $V-F$ selectivity contrast. **(C)** Arm-pooled changes in $m_{\mathrm{background}}$, $m_{\mathrm{target}}$, and the prespecified $m_{\mathrm{specific}}=m_{\mathrm{target}}-m_{\mathrm{background}}$ check distinguish a broad alternative-space shift from target-specific separation. **(D)** Within-model p25-minus-uncapped changes in $m_{\mathrm{general}}$ for a separate regime-matched Qwen3-32B anchor and Llama-3.3-70B-Instruct under the same reasoning-disabled deterministic forced-choice protocol; each model uses its released axis and p25 configuration. Panels A--C use 24 bases from 12 structural families; panel D uses the same bank for each model. Whiskers show 95% family-cluster bootstrap intervals. Model identity and capping configuration vary together in panel D. Complete arm-level estimates and integrity diagnostics appear in *[TBD-APPENDIX]*.

**中文图注：** **模型专属 p25 钳制下的答案几何与改答效应。** **(A)** 在 Qwen3-32B 的 E2A development assay 中，各点表示中性延续（$N$）、无新增证据的复核（$R$）、有效论证（$V$）和 single-fault 谬误论证（$F$）下 full-p25 相对 uncapped 的 $m_{\mathrm{general}}$ 变化；正值表示初始答案相对于三个替代选项的优势减弱。**(B)** $V$ 与 $F$ 条件下 uncapped 和 p25 的目标跟随率；两臂变化之差为 $V-F$ 选择性对比。**(C)** 跨条件汇总的 $m_{\mathrm{background}}$、$m_{\mathrm{target}}$ 变化，以及预先指定的 $m_{\mathrm{specific}}=m_{\mathrm{target}}-m_{\mathrm{background}}$ 检查，用于区分替代选项空间的整体位移与目标特异分离。**(D)** 在统一关闭推理模式并采用确定性 forced choice 的制度下，分别报告单独的制度匹配 Qwen3-32B 参照与 Llama-3.3-70B-Instruct 模型内部的 $m_{\mathrm{general}}$ p25-minus-uncapped 变化；每个模型均使用其公开 axis 和 p25 配置。A--C 使用来自 12 个结构族的 24 个基础样本；D 在每个模型上使用同一评测 bank。误差线表示 95% family-cluster bootstrap 区间。D 中模型身份与 capping 配置同时发生变化。完整分臂估计与完整性诊断见 *[TBD-APPENDIX]*。

### 4.2 Capping Broadly Compresses Initial-Answer Advantage / 钳制广泛压缩初始答案优势

The four-option margins define a two-coordinate projection of the answer space:

$$
m_{\mathrm{target}}
=m_{\mathrm{general}}+\frac{2}{3}m_{\mathrm{specific}},
\qquad
m_{\mathrm{background}}
=m_{\mathrm{general}}-\frac{1}{3}m_{\mathrm{specific}}.
$$

Here, $m_{\mathrm{general}}$ measures movement shared by the non-initial alternatives relative to $a_0$, while $m_{\mathrm{specific}}$ separates the paired target from the two background options. Target-specific amplification predicts a positive change in the latter; compression of initial-answer advantage appears as a positive change in the former across the alternative-option space.

**中文：** 四选一任务中的 margin 构成答案空间的一个二维投影，如上式所示。其中，$m_{\mathrm{general}}$ 刻画各非初始选项相对于 $a_0$ 的共同位移，$m_{\mathrm{specific}}$ 则区分配对目标与两个背景选项。若干预主要放大指定目标，后者应出现正向变化；若干预压缩初始答案优势，则前者会在替代选项空间中整体上升。

In the Qwen3-32B E2A development assay, full-p25 capping increased the pooled $m_{\mathrm{general}}$ by $3.130$ logits relative to uncapped inference (95% family-cluster bootstrap CI $[1.834,4.486]$; Figure 2A). Arm-level estimates were positive for $N$ ($+2.215$), $R$ ($+2.092$), $V$ ($+2.950$), and $F$ ($+5.264$), with all four intervals above zero. $N$ and $R$ keep $a_t$ hidden from the model, yet showed the same positive direction. The broad displacement therefore spans conditions both with and without explicit target exposure.

**中文：** 在 Qwen3-32B 的 E2A development assay 中，相较于未钳制推理，full-p25 capping 使汇总的 $m_{\mathrm{general}}$ 增加 $3.130$ logits（95% family-cluster bootstrap CI $[1.834,4.486]$；图 2A）。四个条件的点估计均为正：$N$ 为 $+2.215$，$R$ 为 $+2.092$，$V$ 为 $+2.950$，$F$ 为 $+5.264$，且四个区间均位于零以上。$N/R$ 对模型隐藏 $a_t$，仍呈现同向位移。因此，这一广泛位移同时覆盖明确展示目标和隐藏目标的条件。

The pooled decomposition placed the shift in both relative-margin components: $\Delta m_{\mathrm{background}}=+3.158$ logits (95% CI $[2.125,4.250]$) and $\Delta m_{\mathrm{target}}=+3.074$ logits ($[1.180,5.038]$). Their prespecified pooled target-specificity contrast was $\Delta m_{\mathrm{specific}}=-0.084$ logits ($[-1.110,0.953]$; Figure 2C). The point estimate shows little additional separation between the target and background options, while the interval remains too broad for an equivalence claim. Together with the $N/R$ results, these estimates characterize full-p25 capping as compressing initial-answer advantage across the alternative-option space in this Qwen development setting. Section 4.3 examines how this continuous displacement maps to warranted and unwarranted revisions.

**中文：** 汇总分解表明，位移同时出现在两个相对 margin 分量中：$\Delta m_{\mathrm{background}}=+3.158$ logits（95% CI $[2.125,4.250]$），$\Delta m_{\mathrm{target}}=+3.074$ logits（$[1.180,5.038]$）。二者预先指定的汇总目标特异性对比为 $\Delta m_{\mathrm{specific}}=-0.084$ logits（$[-1.110,0.953]$；图 2C）。该点估计显示目标选项相对背景选项的额外分离很小，但区间仍不足以支持等效结论。结合 $N/R$ 的结果，这些估计表明，在当前 Qwen development setting 中，full-p25 capping 压缩了初始答案相对于整个替代选项空间的优势。§4.3 将进一步检验这一连续位移如何对应有据与无据的实际改答。

### 4.3 Warranted and Unwarranted Updating Both Increase in the Development Assay / 在 Development Assay 中，有据与无据改答均有所增加

The continuous margin shift did not change the selected answers in the $N/R$ controls. Across all 48 paired comparisons, uncapped and full-p25 inference both yielded $a_0$, despite the positive $\Delta m_{\mathrm{general}}$ reported above. Relative answer support changed while the selected answers remained stable in this assay.

**中文：** 连续 margin 位移没有改变 $N/R$ 对照中的最终选择。在 48 个配对比较中，未钳制与 full-p25 推理均得到 $a_0$，尽管上一节报告的 $\Delta m_{\mathrm{general}}$ 为正。在该评测中，相对答案支持度发生变化，最终选择保持稳定。

Discrete target following increased in both argument arms (Figure 2B). Under $V$, it rose from $70.83\%$ to $91.67\%$ ($+20.83$ percentage points); under $F$, it rose from $29.17\%$ to $45.83\%$ ($+16.67$ points). The mean increase across the two arms was $+18.75$ points (95\% family-cluster bootstrap CI $[4.17,33.33]$). Full-p25 capping therefore raised both warranted and unwarranted updating in this development assay.

**中文：** 两个论证条件中的离散目标跟随均有所增加（图 2B）。在 $V$ 条件下，目标跟随率从 $70.83\%$ 升至 $91.67\%$（增加 $20.83$ 个百分点）；在 $F$ 条件下，则从 $29.17\%$ 升至 $45.83\%$（增加 $16.67$ 个百分点）。两臂效应的平均值为 $+18.75$ 个百分点（95\% family-cluster bootstrap CI $[4.17,33.33]$）。因此，在这一 development assay 中，full-p25 capping 同时提高了有充分依据与缺乏充分依据的改答。

The $V-F$ difference was estimated much less precisely. The capped--uncapped change in $V-F$ target-following selectivity was $+4.17$ percentage points (95\% CI $[-20.83,29.17]$), leaving substantial uncertainty in both directions. In a post-hoc analysis, the association between uncapped option gaps and subsequent migrations varied across data sources (*[TBD-APPENDIX]*). The boundary result therefore offers only a partial description of where migrations occurred. Section 4.4 tests whether the continuous margin shift recurs in a new-salt instantiation and separately examines capping effects on continuous and discrete readouts in natural tasks.

**中文：** $V$ 与 $F$ 之间的差异估计则不够精确。$V-F$ 目标跟随选择性的钳制前后变化为 $+4.17$ 个百分点（95\% CI $[-20.83,29.17]$），区间在正负两个方向上均保留了较大不确定性。在事后分析中，未钳制选项间距与后续迁移的关系随数据来源而异（*[TBD-APPENDIX]*）。因此，边界结果只能部分描述迁移出现的位置。§4.4 将检验连续 margin 位移能否在 new-salt 实例中重现，并另行考察钳制对自然任务连续与离散读数的影响。

### 4.4 Continuous Margin Shifts Accompany Heterogeneous Behavioral Changes / 连续 Margin 位移伴随异质的行为变化

The new-salt bank re-instantiated the same task grammar and structural families. Its aggregate margin shift remained positive: with complete capped--uncapped logits for 95 of 96 paired comparisons, full-p25 capping increased $m_{\mathrm{general}}$ by $+1.538$ logits (95% family-cluster bootstrap CI $[0.455,2.741]$). The positive aggregate conclusion remained robust in the tipping-point analysis of the single missing logit. The discrete effects differed from the development assay: target following changed by $-12.5$ percentage points under $V$ and $+16.7$ points under $F$, yielding a $V-F$ selectivity change of $-29.17$ points (95% CI $[-58.33,4.17]$). The development and new-salt banks therefore shared the aggregate margin direction, while their arm-level answer changes differed.

**中文：** new-salt bank 对同一任务语法和结构族重新实例化，其总体 margin 位移仍为正。96 个 capped--uncapped 配对比较中，95 个具有完整 logits；在这些比较上，full-p25 钳制使 $m_{\mathrm{general}}$ 增加 $+1.538$ logits（95% family-cluster bootstrap CI $[0.455,2.741]$）。针对单个缺失 logit 的 tipping-point 分析表明，总体位移为正的结论保持稳健。离散效应则与 development assay 不同：$V$ 条件下的 target following 降低 $12.5$ 个百分点，$F$ 条件下增加 $16.7$ 个百分点，$V-F$ 选择性变化为 $-29.17$ 个百分点（95% CI $[-58.33,4.17]$）。因此，development bank 与 new-salt bank 的总体 margin 位移同向，各条件的离散改答有所不同。

CEDAR-Social examines a separate contrast on 600 paired natural-task items. Relative to uncapped inference, full-p25 capping increased the target-minus-initial margin by $+1.268$ logits in $AO$ and $+0.482$ logits in $S$, giving $\Delta_{\mathrm{social}}(p25)=-0.786$ logits (95% stratified base-bootstrap CI $[-0.860,-0.711]$). Both component effects were positive: the interaction captures a smaller increase with social framing than with target exposure alone. On the frozen free-generation subset, strict target-following and completion-based SocialExcess were both zero, and no paired $S$-arm answer migrated under p25. In this experiment, the reduction in the incremental social-framing effect appeared only in the direct margin readout.

**中文：** CEDAR-Social 在 600 道配对自然任务上估计一项独立对比。相较于未钳制推理，full-p25 钳制使 $AO$ 条件下目标相对初始答案的 margin 增加 $+1.268$ logits，使 $S$ 条件增加 $+0.482$ logits，由此得到 $\Delta_{\mathrm{social}}(p25)=-0.786$ logits（95% 分层 base-bootstrap CI $[-0.860,-0.711]$）。两个分量效应均为正：该交互反映社会性措辞下的增幅小于单纯目标暴露下的增幅。在冻结的自由生成子集中，按严格目标跟随和完整回答计算的 SocialExcess 均为 0，p25 条件下也未观察到 $S$ 臂的配对答案迁移。在该实验中，社会性措辞增量效应的减小仅出现在直接 margin 读数上。

In a complementary bidirectional SycoBench assay, the continuous selectivity contrast was well resolved, whereas the prespecified mirrored argmax criterion remained inconclusive (*[TBD-APPENDIX]*). Across our analyses, continuous margin effects and discrete answer changes are accordingly reported as separate outcomes.

**中文：** 在互补的双向 SycoBench 评测中，连续选择性对比得到了清晰估计，而预先指定的镜像 argmax 判据仍未得到确定结论（*[TBD-APPENDIX]*）。因此，全文将连续 margin 效应与离散答案改变作为两类独立结果报告。

### 4.5 Transport Across Model-Specific Capping Configurations / 跨模型专属钳制配置的迁移

Figure 2D reports within-model $Delta m_{\mathrm{general}}$ under the matched forced-choice regime. On the shared Evidence bank, the regime-matched Qwen anchor yielded *[TBD-XM-Q-BROAD]*, and Llama-3.3-70B-Instruct yielded *[TBD-XM-L-BROAD]*; both intervals use the frozen family-cluster bootstrap. *[TBD-XM-TRANSPORT-ROUTE]*

**中文：** 图 2D 汇总匹配 forced-choice 制度下各模型内部的 $\Delta m_{\mathrm{general}}$。在共享 Evidence bank 上，制度匹配的 Qwen 参照为 *[TBD-XM-Q-BROAD]*，Llama-3.3-70B-Instruct 为 *[TBD-XM-L-BROAD]*；两者区间均来自冻结的 family-cluster bootstrap。*[TBD-XM-TRANSPORT-ROUTE]*

Within each model, the prespecified target-specificity check pooled across arms was *[TBD-XM-SPECIFIC]*, and the corresponding change in continuous $V-F$ selectivity was *[TBD-XM-VF-SELECTIVITY]*. *[TBD-XM-DISCRETE]* Complete arm-level estimates and cap-exposure and output-integrity diagnostics appear in *[TBD-APPENDIX]*. Model identity and the released capping configuration vary together in this comparison; the estimates therefore apply to the two tested model--configuration pairs.

**中文：** 各模型内部预先指定、跨 N/R/V/F 四臂汇总的 target-specificity 检查结果为 *[TBD-XM-SPECIFIC]*，相应的连续 $V-F$ 选择性变化为 *[TBD-XM-VF-SELECTIVITY]*。*[TBD-XM-DISCRETE]* 完整分臂估计、钳制暴露和输出完整性诊断见 *[TBD-APPENDIX]*。模型身份与公开 capping 配置在这项比较中同时发生变化；因此，这些估计仅描述本文测试的两个模型—配置组合。

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
