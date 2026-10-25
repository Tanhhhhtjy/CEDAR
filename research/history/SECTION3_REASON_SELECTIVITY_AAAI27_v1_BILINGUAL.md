# 3 Identifying Reason Selectivity under Activation Capping

Section 2 leaves four explanations observationally entangled: capping may alter evidence discrimination, lower the general criterion for abandoning an answer, attract the model toward a proposed target, or amplify social accommodation. An audit based only on accuracy or answer flips cannot separate them. We therefore vary three factors independently—warrant validity, target exposure, and social framing—while holding the underlying task and candidate answer fixed. Two paired contrasts provide the identification: a solver-certified valid–fallacious contrast tests responsiveness to reasons, and a target-exposure-controlled social contrast isolates the contribution of social language. Discrete answer transitions are then paired with continuous answer geometry, allowing a latent shift to be detected even when it does not cross the final decision boundary (Figure 1).

> **中文：** 第二节留下了四种在普通观察中彼此纠缠的解释：capping 可能改变证据分辨能力、降低放弃当前答案的一般门槛、将模型吸引到某个候选目标，或放大社会性迎合。只依靠准确率或答案翻转无法区分这些解释。因此，我们独立操控三个因素——理由有效性、目标暴露和社会框架——同时固定基础任务和候选答案。两组严格配对的对照承担识别任务：经求解器认证的“有效–谬误”对照检验模型是否依据理由作出响应；控制目标暴露的社会对照则隔离社会性措辞的增量作用。最后，我们将离散答案迁移与连续答案几何配对，使尚未跨过最终决策边界的潜在位移也能够被观测（图 1）。

## 3.1 Paired Revision Setting and Identification Strategy

Every instance begins with a base problem and a fixed initial answer \(a_0\). A different option \(a_t\) is designated as the paired target for analysis. A parseable final answer is classified as the initial answer, the paired target, or another option; malformed or incomplete outputs form a fourth category. Choosing \(a_t\) is therefore not treated as intrinsically good or bad. It is warranted only when the follow-up supplies sufficient support for the corresponding revision.

> **中文：** 每个样本都从一道基础问题和一个固定的初始答案 \(a_0\) 开始，并指定另一个选项 \(a_t\) 作为分析中的配对目标。可解析的最终答案被分为初始答案、配对目标或其他选项；格式错误或不完整输出构成第四类。因而，选择 \(a_t\) 本身并不被预设为好或坏。只有当后续信息为相应修订提供充分支持时，目标跟随才构成有根据的改答。

The six conditions are not six unrelated tasks. They form two identification contrasts anchored to the same revision setting. In the evidence contrast, \(N\) continues the exchange without a revision request, \(R\) asks for an independent recheck without adding evidence, and \(V/F\) present the same candidate edit and target under a matched wrapper. Only the warrant changes: it is sufficient in \(V\) and contains one fatal error in \(F\). In the social contrast, both \(AO\) and \(S\) expose the same target without task evidence; only \(S\) adds first-person confidence and a request to reconsider. Thus \(V-F\) isolates warrant validity, while \(S-AO\) subtracts target mention and priming from the effect of social framing.

> **中文：** 这六个条件并不是六种互不相关的任务，而是建立在同一个改答设定上的两组识别对照。在证据对照中，\(N\) 仅延续对话而不要求修订，\(R\) 在不增加证据的情况下要求独立复核，\(V/F\) 则用相同外壳呈现同一个候选编辑和目标。二者之间唯一改变的是理由：\(V\) 中的理由充分，\(F\) 中则含有一个致命错误。在社会对照中，\(AO\) 与 \(S\) 都展示同一目标且不提供任务证据；只有 \(S\) 增加第一人称确信和重新考虑的请求。因此，\(V-F\) 隔离理由有效性，而 \(S-AO\) 从社会框架效应中扣除了目标出现和启动效应。

**Revised Figure 1 caption.** *Paired identification design. The evidence contrast hides the target in \(N/R\) and presents the same candidate revision in \(V/F\), where only \(V\) supplies a sufficient warrant. The social contrast pairs target-only exposure (\(AO\)) with the same exposure plus social framing (\(S\)). Together, the contrasts distinguish reason validity from target exposure and social influence.*

> **中文图注：** **配对识别设计。** 证据对照在 \(N/R\) 中不展示目标，并在 \(V/F\) 中呈现相同的候选修订，其中只有 \(V\) 提供充分理由。社会对照将纯目标暴露（\(AO\)）与相同目标暴露加社会框架（\(S\)）配对。两组对照共同区分理由有效性、目标暴露和社会影响。

## 3.2 Solver-Certified Valid–Fallacious Contrast

The evidence assay is built from finite Horn theories with explicit negation and no function symbols. Each base theory has a unique four-way answer \(a_0\): supported, unknown, refuted, or both supported and refuted. A deterministic atomic edit defines a paired alternative \(a_t\). The \(V\) condition supplies an argument that entails this edit and makes \(a_t\) the uniquely warranted answer. The matched \(F\) condition proposes the same edit through an argument with exactly one invalid inference, so the evidence is insufficient to displace \(a_0\). \(N\) and \(R\) preserve the same latent target for analysis but do not reveal it to the model.

> **中文：** 证据评测建立在带显式否定、且不含函数符号的有限 Horn 理论上。每个基础理论在四类答案中具有唯一的 \(a_0\)：得到支持、无法确定、遭到反驳，或同时得到支持与反驳。一项确定性的原子编辑定义配对备选 \(a_t\)。在 \(V\) 条件中，论证能够推出该编辑，并使 \(a_t\) 成为唯一有充分依据的答案。与之配对的 \(F\) 条件通过仅含一个无效推理的论证提出相同编辑，因此现有证据不足以取代 \(a_0\)。\(N\) 和 \(R\) 在分析中保留同一个潜在目标，但不向模型展示。

Certification is performed at two levels before any audited model is run. A Python fixed-point reasoner and Prover9 independently verify the object-level answer and whether the argument entails the proposed edit; only cases on which they agree are retained. Every \(F\) argument additionally has a Mace4 countermodel and a certificate locating its single fatal step, while all remaining steps are supported. Valid and fallacious versions use the same natural-language renderer, reviewer role, proposed edit, wrapper skeleton, and output contract. The solver is therefore not merely a quality-control tool: it makes warrant validity—the independent variable of the audit—checkable without asking either the evaluated model or the argument author to determine the gold label.

> **中文：** 在任何被审计模型运行之前，认证在两个层面完成。Python 不动点推理器和 Prover9 分别独立核验对象层答案，以及论证能否推出候选编辑；只有两者结论一致的样本才被保留。每个 \(F\) 论证还必须具有一个 Mace4 反模型和一份定位唯一致命步骤的证书，其余步骤均需得到支持。有效和谬误版本使用相同的自然语言渲染器、复核者角色、候选编辑、外壳结构和输出约束。因此，求解器不只是质量控制工具：它使本审计的自变量——理由有效性——能够在不依赖被评估模型或论证作者判断 gold label 的情况下被机器检验。

Each bank contains 24 bases from 12 structural families, with an easy and a medium variant per family and six certified error classes. Difficulty is varied through proof structure and distractors rather than deliberately obscure wording. The development bank is used for exploratory calibration. A second bank re-instantiates the same task grammar and structural families with a new random salt and is analyzed independently; it is not formed by selecting favorable model responses. Families, rather than individual prompts, are the independent units for inference.

> **中文：** 每个题库包含 24 个基础样本，来自 12 个结构族；每个结构族各有一个简单和一个中等难度变体，并覆盖六类经认证的错误。难度通过证明结构和干扰信息调节，而不是依靠刻意晦涩的措辞。development bank 用于探索性校准。第二个题库使用新的随机盐，对相同任务语法和结构族重新实例化并独立分析；它不是根据模型输出筛选得到的。统计推断的独立单位是结构族，而不是单条 prompt。

## 3.3 Target-Exposure-Controlled Social Contrast

The social assay uses 600 natural multiple-choice items from SycoBench-600. The standard answer is \(a_0\), and a different option is selected as \(a_t\). Both follow-ups explicitly mention \(a_t\). The target-only condition \(AO\) introduces no speaker, correctness claim, evidence, authority, reward, threat, or consensus cue. The matched social condition \(S\) adds first-person confidence and asks the model to reconsider, but contributes no task-relevant fact or reason. The paired \(S-AO\) comparison therefore estimates the incremental effect of social framing beyond exposure to the target itself; it should not be interpreted as a generic “sycophancy rate.”

> **中文：** 社会评测使用 SycoBench-600 中的 600 道自然多选题。标准答案记为 \(a_0\)，另一个选项被指定为 \(a_t\)。两种后续输入都会明确提及 \(a_t\)。纯目标条件 \(AO\) 不包含说话者、正确性声明、证据、权威、奖励、威胁或共识线索。与之配对的社会条件 \(S\) 增加第一人称确信并要求模型重新考虑，但不提供任何与任务有关的事实或理由。因此，配对的 \(S-AO\) 比较估计的是超出目标暴露本身的社会框架增量效应，而不应被直接解释成一个泛化的“谄媚率”。

Four minimal wrapper variants are balanced within domain and difficulty strata. Automated checks verify pair integrity and tokenizer-level length constraints before model execution. A pre-output human review assessed template fidelity and linguistic naturalness; the recorded deviation from the originally intended independent-review procedure is disclosed with the audit materials rather than being represented as completed dual annotation. Fixed short-answer margins are the primary readout, while free-generation responses on the frozen subset are secondary diagnostics so that truncation or deliberation length cannot determine the headline estimate.

> **中文：** 四种最小外壳变体在领域和难度分层内均衡分配。模型运行前，自动检查会核验配对完整性和冻结 tokenizer 下的长度约束。预输出人工审核检查了模板忠实度和语言自然度；项目中相对于原定独立复核流程的偏离会随审计材料如实披露，而不会被表述为已经完成双人独立标注。固定短答案 margin 是主要读数，冻结子集上的自由生成回答仅作为次要诊断，从而避免截断或思考长度决定核心估计。

## 3.4 Released Capping Operator and Model Scope

We audit the released, model-specific Assistant-Axis threshold cap without fitting its direction or threshold to either assay. Let \(v_\ell\) be the stored unit direction at layer \(\ell\), and let \(\tau_{\ell,c}\) be the threshold for configuration \(c\). Because the released vector uses the opposite sign from the human-readable Assistant direction, the published minimum cap is implemented as

\[
h'_{\ell,t}
=h_{\ell,t}
-v_\ell\big[\langle h_{\ell,t},v_\ell\rangle-\tau_{\ell,c}\big]_+,
\qquad [z]_+=\max(z,0).
\tag{1}
\]

The operator removes only the projection beyond the calibrated boundary. We preserve the released directions, layer ranges, thresholds, and sign convention unchanged. This is important for scope: the experiment audits a deployed operator configuration rather than training a controller on the proposed benchmark.

> **中文：** 我们审计官方发布、模型特定的 Assistant-Axis 阈值 capping，不根据任何一个评测任务重新拟合方向或阈值。令 \(v_\ell\) 表示第 \(\ell\) 层存储的单位方向，\(\tau_{\ell,c}\) 表示配置 \(c\) 在该层使用的阈值。由于公开资产中的向量与人类可读的 Assistant 方向符号相反，原始 minimum cap 按式（1）实现。该算子只移除超过校准边界的投影分量。我们原样保留公开方向、层范围、阈值和符号约定。这一区别决定了本文的作用域：我们是在审计一个已经发布的算子配置，而不是利用所提出的评测集训练新的控制器。

Qwen3-32B is the primary model, using the released `layers_46:54-p0.25` configuration over layers 46–53. The main evidence assay uses its frozen reasoning and greedy-decoding regime; comparisons across decoding regimes are treated as sensitivity analyses rather than pooled estimates. Cross-model evaluation uses the same frozen 24-item evidence bank under a matched deterministic forced-choice regime for Qwen3-32B and Llama-3.3-70B-Instruct. Llama uses its released direction and `layers_56:72-p0.25` configuration over layers 56–71. Effects are estimated within each model and configuration. The design therefore tests transfer across two released model–cap pairs, but does not identify whether heterogeneity arises from the base model or its separately calibrated cap.

> **中文：** Qwen3-32B 是主要模型，使用公开的 `layers_46:54-p0.25` 配置，作用于第 46–53 层。主要证据评测采用其冻结的 reasoning 和 greedy 解码制度；跨解码制度比较作为敏感性分析单独处理，而不与主估计混合。跨模型评测在 Qwen3-32B 与 Llama-3.3-70B-Instruct 上使用相同的冻结 24 题证据库和匹配的确定性 forced-choice 制度。Llama 使用其公开方向和 `layers_56:72-p0.25` 配置，作用于第 56–71 层。所有效应均在各自模型和配置内部估计。因此，该设计检验的是两个官方“模型–cap”组合之间的迁移，而不能进一步识别异质性究竟来自基础模型还是分别校准的 cap。

## 3.5 Behavioral Outcomes and Answer Geometry

Discrete outputs are assigned to four mutually exclusive outcomes: the initial answer, the paired target, another option, or invalid/abstaining output. Because a distributional shift may precede an answer flip, we also read the next-token logits at a fixed answer position. Let \(z(a)\) denote the logit of option \(a\). For a four-option item, we define

\[
\begin{aligned}
m_{\mathrm{general}}
&=\frac{1}{3}\sum_{a\ne a_0}\big[z(a)-z(a_0)\big],\\
m_{\mathrm{target}}
&=z(a_t)-z(a_0),\\
m_{\mathrm{background}}
&=\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\big[z(a)-z(a_0)\big],\\
m_{\mathrm{specific}}
&=z(a_t)-\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}z(a).
\end{aligned}
\tag{2}
\]

> **中文：** 离散输出被划分为四个互斥类别：初始答案、配对目标、其他选项，以及无效输出或弃答。由于分布位移可能先于答案翻转，我们还读取固定答案位置上的 next-token logits。令 \(z(a)\) 表示选项 \(a\) 的 logit，四项 margin 按式（2）定义。

The decomposition is designed around the competing explanations rather than around metric coverage. \(m_{\mathrm{general}}\) measures a common movement away from \(a_0\): a positive capped–uncapped change means that the initial-answer advantage has narrowed across the alternative space. \(m_{\mathrm{specific}}\) compares the proposed target with the two background alternatives and therefore isolates target-specific attraction. \(m_{\mathrm{target}}\) and \(m_{\mathrm{background}}\) show which option groups contribute to the broad shift. Thus, a controller that merely lowers the initial-answer advantage should move \(m_{\mathrm{general}}\) without a corresponding positive \(m_{\mathrm{specific}}\); genuine attraction toward the proposal should appear in \(m_{\mathrm{specific}}\). Reporting both continuous geometry and discrete transitions directly operationalizes the margin–choice dissociation introduced in Section 2.

> **中文：** 这一分解不是为了增加指标数量，而是围绕竞争解释构造的。\(m_{\mathrm{general}}\) 衡量整体离开 \(a_0\) 的共同位移：其 capped–uncapped 变化为正，表示初始答案相对于整个替代选项空间的优势缩小。\(m_{\mathrm{specific}}\) 将候选目标与两个背景选项直接比较，因此隔离目标特异吸引。\(m_{\mathrm{target}}\) 和 \(m_{\mathrm{background}}\) 则说明广泛位移分别由哪些选项组贡献。因而，如果控制器只是降低初始答案优势，它应改变 \(m_{\mathrm{general}}\)，却不必带来正向的 \(m_{\mathrm{specific}}\)；真正对候选目标的吸引才应体现在 \(m_{\mathrm{specific}}\) 中。同时报告连续几何和离散迁移，正是对第二节所定义 margin–choice 分离的直接操作化。

## 3.6 Estimands, Statistical Units, and Audit Discipline

For outcome \(Y\), follow-up condition \(x\), and cap configuration \(c\), let

\[
\delta_x^Y(c)=\mathbb{E}[Y\mid x,c]-\mathbb{E}[Y\mid x,0],
\tag{3}
\]

where \(0\) denotes uncapped inference. The evidence assay uses three complementary estimands:

\[
\begin{aligned}
\Delta_{\mathrm{broad}}(c)
&=\frac14\sum_{x\in\{N,R,V,F\}}
\delta_x^{m_{\mathrm{general}}}(c),\\
\Delta_{\mathrm{sel}}^{(m)}(c)
&=\delta_V^{m_{\mathrm{target}}}(c)
-\delta_F^{m_{\mathrm{target}}}(c),\\
\Delta_{\mathrm{sel}}^{(p)}(c)
&=[p_V(c)-p_V(0)]-[p_F(c)-p_F(0)],
\end{aligned}
\tag{4}
\]

where \(p_x(c)\) is the probability of selecting \(a_t\). \(\Delta_{\mathrm{broad}}\) tests whether the cap generally changes the initial-answer advantage; the two selectivity contrasts test whether it separates warranted from fallacious revision in margin and behavior. The pooled change in \(m_{\mathrm{specific}}\) is the prespecified target-specificity check; condition-wise decompositions are labeled post hoc.

> **中文：** 对结果 \(Y\)、后续条件 \(x\) 和 capping 配置 \(c\)，式（3）定义相对于未钳制推理的条件效应。证据评测使用式（4）中的三类互补估计量：\(\Delta_{\mathrm{broad}}\) 检验 capping 是否广泛改变初始答案优势；两个选择性对比则分别在连续 margin 和离散行为上检验它是否区分有根据与谬误修订。汇总的 \(m_{\mathrm{specific}}\) 变化是预先指定的目标特异性检查；按条件分解明确标记为 post hoc。

For the social contrast, we define

\[
\Delta_{\mathrm{social}}(c)
=\delta_S^{m_{\mathrm{target}}}(c)
-\delta_{AO}^{m_{\mathrm{target}}}(c)
=[S-AO]_c-[S-AO]_0.
\tag{5}
\]

This interaction measures how capping changes the additional target support caused by social framing after target exposure has been subtracted. Component effects for \(S\) and \(AO\) are always reported alongside the interaction so that a change in both conditions is not mislabeled as a social-specific effect.

> **中文：** 社会对照按式（5）定义。该交互衡量的是：扣除目标暴露之后，capping 如何改变由社会框架额外带来的目标支持。\(S\) 与 \(AO\) 的分量效应始终与交互项共同报告，从而避免把两个条件共同发生的位移误写成社会压力特异效应。

Evidence-assay uncertainty is computed at the level of 12 independently generated structural families. Capped–uncapped differences are first aggregated within family and then averaged with equal family weight; primary intervals use a family-cluster bootstrap, accompanied by family effects and equal-weight leave-one-family-out analyses. The social assay resamples items with domain and difficulty stratification. Development, new-salt, social, and cross-model effects are estimated separately rather than pooled across incompatible protocols.

> **中文：** 证据评测的不确定性以 12 个独立生成的结构族为单位计算。capped–uncapped 差异先在结构族内部汇总，再对结构族等权平均；主要区间使用结构族聚类 bootstrap，并同时报告结构族效应和等权 leave-one-family-out 分析。社会评测以题目为重采样单位，并按领域和难度分层。development、new-salt、社会评测和跨模型效应分别估计，不跨越不兼容的协议进行汇总。

All prespecified rows enter the discrete intention-to-treat analysis. Truncated or unparseable outputs are retained as invalid/abstaining outcomes and are not selectively rerun. Continuous contrasts require the relevant logits in both paired conditions; incomplete coverage is reported and subjected to sensitivity analysis rather than imputed. Model revisions, decoding regimes, cap assets, data hashes, exclusions, and analysis status are frozen and audited separately. This discipline keeps the central claim tied to the intended estimand: whether the released cap is reason-selective, not whether one condition happens to produce a favorable accuracy change.

> **中文：** 所有预先指定的记录都进入离散的意向分析。截断或无法解析的输出保留为无效输出或弃答，不根据内容选择性重跑。连续对比要求配对条件中的相关 logits 均完整；覆盖不完整时如实报告并进行敏感性分析，而不进行插补。模型版本、解码制度、capping 资产、数据哈希、排除规则和分析状态分别冻结并审计。这一纪律使核心主张始终对应真正的估计目标：官方 capping 是否具有理由选择性，而不是某个条件是否偶然产生了更好看的准确率变化。

---

## Revision Notes (Not Part of the Manuscript)

1. **Removed the promotional section title.** “CEDAR: Auditing Epistemic Selectivity” is replaced by the scientific question, “Identifying Reason Selectivity under Activation Capping.” All subsection titles likewise describe the identification role rather than a branded module.
2. **Replaced the component inventory with an identification argument.** The six arms are introduced as two paired contrasts, not as dataset features.
3. **Made solver certification a scientific contribution.** The solver establishes both the object-level answer and the meta-level warrant before model exposure; it is not described as ordinary data cleaning.
4. **Made the margin decomposition hypothesis-driven.** The section now states exactly which geometric pattern corresponds to broad initial-answer compression and which corresponds to target-specific attraction.
5. **Separated social framing from target exposure.** \(S-AO\) is defined as an incremental social effect, not a generic sycophancy score.
6. **Corrected the human-review claim.** The PDF's statement that two researchers independently reviewed all social pairs is stronger than the recorded procedure. The revision discloses the pre-output human review and points to the protocol deviation.
7. **Tightened model-generalization scope.** Cross-model results are explicitly interpreted as two model–configuration pairs, not as an isolated model effect.
8. **Preserved the audit discipline.** Family-level inference, new-salt separation, intention-to-treat handling, missing-logit rules, and post-hoc labels remain visible because they materially support the paper's credibility.
9. **Conditional inclusion of the Llama paragraph.** Keep the cross-model paragraph in the submitted Methods only if the matched Qwen/Llama experiment and its audit are complete. If it remains planned at submission time, remove it rather than describing a future run as an implemented method.
