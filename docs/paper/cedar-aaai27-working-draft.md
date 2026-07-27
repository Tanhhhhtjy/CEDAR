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

### 2.1 Activation Steering and Threshold Capping / 激活 Steering 与阈值钳制

`[TBD: verified synthesis distinguishing addition, ablation, steering, and state-dependent upper capping.]`

`[TBD：填入经核验的相关工作综合，区分 addition、ablation、steering 与状态依赖的 upper capping。]`

### 2.2 Sycophancy, Social Influence, and Answer Revision / 谄媚、社会影响与答案修正

`[TBD: verified synthesis of social influence, target exposure, valid evidence, and fallacious persuasion.]`

`[TBD：填入经核验的社会影响、target exposure、有效证据与谬误说服相关工作综合。]`

### 2.3 Reason-Sensitive Evaluation and Intervention Side Effects / 理由敏感评测与干预副作用

`[TBD: nearest-neighbor comparison with dual-stance evaluation, Belief-R, and benefit-retention work.]`

`[TBD：填入与 dual-stance evaluation、Belief-R 及收益—保留评测的最近邻比较。]`

## 3. CEDAR: Auditing Epistemic Selectivity / CEDAR：审计认识论选择性

### 3.1 Epistemic Selectivity and Controlled Contrasts / 认识论选择性与受控对照

Let a model answer a base question with option $a_0$ and then receive a follow-up that proposes a different option $a_t$. We call a final choice other than $a_0$ an *answer revision*. CEDAR asks whether an activation-space intervention preserves the dependence of revision on the warrant supplied for that change, rather than merely raising or lowering the overall revision rate. We operationalize this question at the output level through two complementary but statistically separate modules. CEDAR-Evidence contrasts reasons with different formal validity; CEDAR-Social isolates social phrasing from exposure to the same target. Comparisons are paired within each module, never pooled as a six-arm experiment.

**中文：** 设模型先对一个基础问题选择选项 $a_0$，随后收到一条提出不同选项 $a_t$ 的后续输入。我们把最终选择不再是 $a_0$ 称为一次*答案修正*。CEDAR 关心的不是干预是否简单提高或降低总体改答率，而是干预后“是否改答”是否仍然取决于支持这种改变的理由是否充分。我们在输出层面通过两个互补但统计上独立的模块操作化这一问题：CEDAR-Evidence 对比形式有效性不同的理由；CEDAR-Social 将社会性措辞与相同 target 的暴露效应分离。配对只在各模块内部进行，两个模块不合并成统一六臂实验。

**Figure 1 placeholder / 图 1 占位：** CEDAR’s two-module audit design / CEDAR 双模块审计设计。

**Caption / 图注：** **Same target, different reasons.** Within CEDAR-Evidence, neutral continuation ($N$), reason-free rechecking ($R$), valid evidence ($V$), and a matched single-fault argument ($F$) share a formal base, initial answer, and proposed target; only $V$ warrants revision. Separately, CEDAR-Social pairs target exposure alone ($AO$) with social phrasing ($S$) on the same natural-task base. The modules retain their own readouts and statistical units. Additional paired examples, solver certificates, and construction details are provided in *[TBD-APPENDIX]*.

**中文图注：** **同一 target，不同理由。** 在 CEDAR-Evidence 中，中性延续（$N$）、无理由复核（$R$）、有效证据（$V$）和匹配的单错误论证（$F$）共享同一形式基础问题、初始答案与 proposed target；只有 $V$ 在规范上支持改答。与之统计独立的 CEDAR-Social 在同一自然任务基础问题上，将纯 target exposure（$AO$）与社会性措辞（$S$）配对。两个模块分别保留自己的读数与统计单位。更多配对示例、solver certificates 与构造细节见 *[TBD-APPENDIX]*。

### 3.2 CEDAR-Evidence: A Solver-Certified Paired Dataset / CEDAR-Evidence：经求解器认证的配对数据集

Each CEDAR-Evidence base defines a finite function-free Horn world with explicit negation and a four-option query: supported, undetermined, contradicted, or both supported and contradicted. The base world has a unique initial answer $a_0$. A deterministically selected atomic edit produces a different unique answer $a_t$. Four follow-ups reuse this base and target: $N$ continues without a revision request; $R$ requests reconsideration without new grounds; $V$ presents a justification that uniquely proves the edit is warranted; and $F$ proposes the same edit through one structurally certified invalid inference, leaving $a_0$ correct.

**中文：** 每个 CEDAR-Evidence 基础样本都定义一个带显式否定的有限无函数 Horn 世界，并提出一道四选一问题：支持、无法确定、矛盾，或同时支持且矛盾。基础世界具有唯一初始答案 $a_0$；一次确定性选择的原子编辑会产生不同的唯一答案 $a_t$。四种后续输入复用该基础问题与 target：$N$ 只做中性延续，不要求修正；$R$ 要求重新检查，但不给出新依据；$V$ 给出能够唯一证明该编辑有充分依据的论证；$F$ 则通过一个经结构认证的无效推理提出相同编辑，因此 $a_0$ 仍然正确。

The $V/F$ pair fixes reviewer identity, speech act, proposed edit, wrapper skeleton, and output contract. Python fixed-point inference and Prover9 agree on the object-level answer and the meta-level warrant. Every $F$ instance additionally has a Mace4 countermodel, exactly one certified fatal step, and no second unsupported step. Deterministic rebuilds, duplicate and leakage audits, and a blinded human naturalization review precede model execution. Each controlled bank contains 24 bases from 12 independently generated structural families, with easy and medium variants spanning six error classes. A second, independently salted instantiation is retained as a separate evidence identity rather than pooled with the development bank. Repeated arms and model conditions therefore do not increase the independent sample size.

**中文：** $V/F$ 对固定 reviewer identity、speech act、proposed edit、wrapper skeleton 与输出格式。Python fixed-point inference 与 Prover9 在对象层答案和 meta-level warrant 上一致。每个 $F$ 样本还具有 Mace4 countermodel、恰好一个经认证的致命错误，且不存在第二个无支撑步骤。模型运行前完成确定性重建、重复与泄漏审计，以及盲化的人工自然化审核。每个受控 data bank 包含 24 个 bases，来自 12 个独立生成的结构 family；easy/medium 变体共覆盖六类错误。第二套独立加盐实例保留为单独证据身份，不与开发 bank 合并。同一 family 内增加 arm 或模型条件因此不会增加独立样本量。

### 3.3 CEDAR-Social: Controlling for Target Exposure / CEDAR-Social：控制 Target Exposure

CEDAR-Social is a paired protocol over the 600 natural multiple-choice bases in SycoBench-600, not a split of CEDAR-Evidence. For each base, the initial assistant answer is the gold option and $a_t$ is a different option. The $AO$ follow-up mentions $a_t$ without an agent, testimony, confidence, or correctness claim. The matched $S$ follow-up adds first-person confidence and a request to answer again, but no fact, reason, authority, reward, or group consensus. Four minimal-edit template pairs are balanced within domain and difficulty, and their token-length difference is bounded before execution. Automated validation and a full single-data-lead audit cover all 600 pairs. Direct next-option margins are the primary readout; a frozen 240-base subset supplies a free-generation modality check.

**中文：** CEDAR-Social 是构建在 SycoBench-600 的 600 个自然多选题之上的配对协议，而不是 CEDAR-Evidence 的 split。对于每个 base，初始 assistant 答案是 gold 选项，$a_t$ 是另一个选项。$AO$ 后续输入只提及 $a_t$，不包含施动者、证言、确信或正确性断言；匹配的 $S$ 输入增加第一人称确信和重新作答请求，但不增加事实、理由、权威、奖惩或群体共识。四组最小编辑模板在 domain 与 difficulty 内平衡，并在运行前约束两臂 token 长度差。自动验证与单一数据负责人的全量审核覆盖全部 600 对。Direct next-option margin 是主要读数；冻结的 240-base 子集用于检查 free-generation 模态边界。

### 3.4 Intervention and Model Scope / 干预与模型范围

Our primary intervention is the released, model-specific Assistant-Axis threshold cap. Let $\mathbf{v}_{\ell}$ denote the stored unit cap direction at layer $\ell$ and $\tau_{\ell,c}$ its threshold under configuration $c$. For residual activation $\mathbf{h}_{\ell,t}$, the operator applies

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

where $[z]_{+}=\max(z,0)$. Unlike additive steering, the intervention is inactive until a state crosses the configured boundary.

**中文：** 主要干预是公开的、模型专属的 Assistant-Axis 阈值钳制。令 $\mathbf{v}_{\ell}$ 表示第 $\ell$ 层存储的单位钳制方向，$\tau_{\ell,c}$ 表示配置 $c$ 下的阈值。对于残差激活 $\mathbf{h}_{\ell,t}$，算子按上式执行，其中 $[z]_{+}=\max(z,0)$。与 additive steering 不同，只有当前状态越过配置边界时，该干预才生效。

Qwen3-32B is the primary discovery and diagnostic model, using the official `layers_46:54-p0.25` configuration (decoder layers 46--53). We extend the matched behavioral audit to Llama-3.3-70B-Instruct with its released model-specific axis and official `layers_56:72-p0.25` configuration. We further include Gemma-2-27B-IT using its released model-specific axis and a *source-faithful calibrated extension*, because the source release does not specify a Gemma capping layer range. The Gemma configuration uses `[TBD-GEMMA: calibrated layer range and thresholds]`. No vector, layer range, or numeric threshold is transferred between models, and model-specific effects are never pooled into an average-model estimate. Exact revisions, hashes, decoding regimes, and calibration grids are reported in *[TBD-APPENDIX]*.

**中文：** Qwen3-32B 是主要发现与诊断模型，使用官方 `layers_46:54-p0.25` 配置，即 decoder layers 46--53。我们将匹配的行为审计扩展到 Llama-3.3-70B-Instruct，并使用其公开的模型专属 axis 与官方 `layers_56:72-p0.25` 配置。我们还纳入 Gemma-2-27B-IT，使用其公开的模型专属 axis；由于源发布没有给出 Gemma 的钳制层范围，我们采用*遵循源方法的校准扩展*，其配置为 `[TBD-GEMMA：校准后的层范围与阈值]`。不同模型之间不迁移 vector、层范围或数值阈值，也不把模型专属效应合并成“平均模型效应”。精确 revision、hash、decoding regime 与 calibration grid 见 *[TBD-APPENDIX]*。

### 3.5 Behavioral and Continuous Readouts / 行为与连续读数

We encode each final output as the initial option, the proposed target, another option, or invalid/abstaining. Let $\ell(a)$ be the next-option logit. For four-option tasks, we define

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

A positive change in $m_{\mathrm{general}}$ means that the initial-answer advantage is compressed; it does not by itself identify which alternative will be selected. We therefore report margins jointly with target-following rates and complete four-state transitions. Evidence selectivity is summarized by the paired $V-F$ target-following contrast while retaining $N/R$ controls. Social influence is summarized by the $S-AO$ contrast and its cap-by-arm interaction. Continuous and discrete estimands are not treated as interchangeable.

**中文：** 每个最终输出被编码为 initial、proposed target、other 或 invalid/abstaining。令 $\ell(a)$ 表示 next-option logit，并按上式定义四个连续 margin。正的 $m_{\mathrm{general}}$ 变化表示 initial-answer advantage 被压缩，但它本身不能确定最终会选择哪个替代项。因此，margin 必须与 target-following rate 和完整四态迁移联合报告。Evidence selectivity 由配对的 $V-F$ target-following contrast 概括，同时保留 $N/R$ 对照；social influence 由 $S-AO$ 对比及 cap-by-arm 交互概括。连续与离散估计量不能互换解释。

### 3.6 Statistical and Audit Discipline / 统计与审计纪律

Evidence uncertainty is clustered at the independently generated family level; the frozen new-salt analysis uses 20,000 paired family-bootstrap draws. Social uncertainty uses the SycoBench base as the unit and 10,000 bootstrap draws stratified by domain and difficulty. Truncated or unparsable outputs remain invalid/abstaining under intention-to-treat analysis, and no item is removed or rerun because of model behavior. Development, new-salt, natural-task, fixed-trace, and cross-model evidence retain separate identities. We report option mass, entropy, top-token integrity, and initial/target/background logit decompositions to distinguish reason-structured effects from generic flattening or output corruption. Full estimands, seeds, multiplicity rules, missing-data sensitivity, and protocol deviations appear in *[TBD-APPENDIX]*.

**中文：** Evidence 模块的不确定性在独立生成的 family 层级聚类；冻结的 new-salt 分析使用 20,000 次配对 family bootstrap。Social 模块以 SycoBench base 为单位，进行按 domain 与 difficulty 分层的 10,000 次 bootstrap。截断或无法解析的输出在 intention-to-treat 分析中保留为 invalid/abstaining，任何样本都不会因模型行为而被删除或重跑。Development、new-salt、natural-task、fixed-trace 与 cross-model 证据保留各自身份。我们报告 option mass、entropy、top-token integrity，以及 initial/target/background logit 分解，用于区分理由结构化效应、一般性 flattening 与输出损坏。完整估计量、随机种子、多重比较规则、缺失数据敏感性和协议偏离见 *[TBD-APPENDIX]*。

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
