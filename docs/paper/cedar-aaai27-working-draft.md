# CEDAR AAAI-27 Working Draft / CEDAR AAAI-27 论文工作稿 v0.3

> Local draft only; do not synchronize to Overleaf yet.
> **中文：** 仅供本地起草；暂不同步至 Overleaf。
> Bilingual reading convention: each English unit is followed by its Chinese counterpart; equations are shared by both versions. Any semantic mismatch should be treated as a revision issue rather than resolved silently.
> **中文：** 双语阅读约定：每个英文单元后紧跟对应中文，公式由两个版本共用；若中英文语义不一致，应将其标为待修改问题，不得静默选择其一。
> This file mixes paper prose with internal evidence labels. All labels and notes must be removed before submission.
> **中文：** 本文件同时包含论文正文与内部证据标签；投稿前必须移除全部标签和内部注释。
> `[V]`: frozen evidence. `[V-AUDIT]`: frozen evidence whose sign/documentation audit is still open. `[TBD-R/M/C]`: behavioral/mechanism/controller result slot, not an obtained result.
> **中文：** `[V]` 表示已冻结证据；`[V-AUDIT]` 表示结果已冻结但符号或文档审计尚未闭合；`[TBD-R/M/C]` 分别表示行为、机制和控制器的结果槽位，而不是已经获得的结果。
> The internal development validation is not evidence for the headline `Cost_V > 0` claim.
> **中文：** 内部开发验证不能作为标题级主张 `Cost_V > 0` 的证据。
> Revision: 2026-07-26; add a planned Llama-3.3-70B cross-model replication with all outcomes retained as `[TBD-R]`.
> **中文：** 修订：2026-07-26；加入 Llama-3.3-70B 跨模型验证计划，全部结果继续标记为 `[TBD-R]`。

## Provisional Title / 暂定标题

**The Hidden Epistemic Cost of Activation Capping**

**激活钳制的隐性认识论代价**

Alternative result-explicit title, to be considered only after the confirmation result is frozen:

**中文：** 另一个直接陈述结果的候选标题；只有在确认实验结果冻结后才考虑使用：

**When Activation Capping Suppresses Warranted Revision**

**当激活钳制抑制有充分理由的修正**

## Abstract / 摘要

`[WRITE LAST FROM THE FIVE-SENTENCE CARD]`

`[中文：依据五句式写作卡片最后撰写]`

## 1. Introduction / 引言

`[WRITE AFTER SECTIONS 3--6 STABILIZE]`

`[中文：待第 3--6 节稳定后撰写]`

Paragraph-level interface / 段落级接口：

1. Inference-time activation interventions are attractive because they can suppress undesirable behavior without retraining.<br>
   **中文：** 推理时激活干预无需重新训练即可抑制不良行为，因此很有吸引力。
2. Existing side-effect checks are dominated by static capability tests and do not measure whether the model remains responsive to valid new evidence.<br>
   **中文：** 现有副作用检查主要依赖静态能力测试，无法衡量模型是否仍能响应有效的新证据。
3. A selective intervention should resist unsupported social pressure while still revising when a proof or counterexample changes the correct answer.<br>
   **中文：** 具有选择性的干预应当抵抗无依据的社会压力，同时在证明或反例改变正确答案时仍能修正回答。
4. CEDAR measures these two sides with independent Evidence and Social modules.<br>
   **中文：** CEDAR 使用彼此独立的 Evidence 与 Social 模块分别测量这两个方面。
5. `[V]` direct social-side result; `[TBD-R]` epistemic cost and dose response; `[TBD-M/C]` mechanism and mitigation if supported.<br>
   **中文：** `[V]` 社会侧直接读数结果；`[TBD-R]` 认识论代价与剂量响应；若证据支持，则写入 `[TBD-M/C]` 机制与缓解结果。
6. Three contributions: formulation/estimands, evaluation assets, and the scoped empirical finding.<br>
   **中文：** 三项贡献：问题形式化与估计量、评测资产，以及严格限定范围的实证发现。

## 2. Background and Related Work / 背景与相关工作

`[WRITE AFTER CITATION KEYS ARE VERIFIED]`

`[中文：待全部引用键核验后撰写]`

Planned organization / 计划结构：

- **Activation-space control and capping.** Distinguish activation addition, directional ablation, continuous steering, and state-dependent threshold capping. Use CAA/ITI and the refusal-direction paper to establish the intervention and causal-validation background; introduce Assistant Axis as the source intervention. `[CITE: CAA; ITI; Refusal Direction; Assistant Axis]`<br>
  **中文：激活空间控制与钳制。** 区分激活加法、方向消融、连续 steering 与状态依赖的阈值钳制。以 CAA/ITI 和 refusal-direction 论文建立干预及因果验证背景，并将 Assistant Axis 作为本文源干预加以介绍。
- **Sycophancy, social influence, and overridden truth.** Prior work establishes unsupported user influence and traces where it emerges. CEDAR instead asks whether suppressing such influence impairs justified correction. `[CITE: Sycophancy; When Truth Is Overridden; SycoBench]`<br>
  **中文：谄媚、社会影响与真相被覆盖。** 既有工作证实无依据的用户影响并追踪其出现位置；CEDAR 转而追问，抑制这种影响是否会损害有充分理由的修正。
- **Belief revision and intervention side effects.** Belief-R and dual-stance work distinguish desirable from undesirable answer changes; WMDP, ITI, and related selective-steering work jointly report intended effects and retention. CEDAR combines threshold capping, formal normative gold, a target-exposure control, and dose-resolved side-effect measurement. `[CITE: Belief-R; Buchan dual-stance; WMDP; KL-then-steer; nearest 2026 work]`<br>
  **中文：信念修正与干预副作用。** Belief-R 与 dual-stance 工作区分期望和不期望的答案变化；WMDP、ITI 及相关选择性 steering 工作同时报告目标效果与能力保留。CEDAR 将阈值钳制、形式化规范金标准、目标暴露对照和剂量分辨的副作用测量结合起来。

Closing sentence / 收束句：

> Prior work asks whether models can be steered, whether they yield to users, or whether they revise outputs; CEDAR asks whether a safety-motivated internal intervention preserves the *selectivity* of revision with respect to the quality of the reason supplied.

> **中文：** 既有工作追问模型能否被 steering、是否会屈从于用户，或是否会修正输出；CEDAR 追问的是，以安全为动机的内部干预能否依据所给理由的质量，保留输出修正的*选择性*。

## 3. CEDAR: Measuring Epistemic Selectivity / CEDAR：测量认识论选择性

We study whether an activation-space intervention preserves a model's ability to distinguish inputs that warrant an answer revision from inputs that do not. CEDAR operationalizes this question with two independent modules. **CEDAR-Evidence** supplies formally certified reasons of different validity and measures warranted revision. **CEDAR-Social** isolates the incremental effect of social wording beyond exposure to the same target answer. The modules use different tasks, readouts, and independent statistical units; we therefore estimate their effects separately and connect them only through the intervention configuration or dose.

**中文：** 我们研究激活空间干预是否保留模型区分“足以支持答案修正”与“不足以支持答案修正”的输入的能力。CEDAR 通过两个相互独立的模块将这一问题操作化。**CEDAR-Evidence** 提供有效性不同、经过形式化认证的理由，并测量有充分理由的修正；**CEDAR-Social** 则在暴露相同目标答案的基础上，分离社会性措辞带来的增量效应。两个模块使用不同的任务、读数和独立统计单位，因此我们分别估计其效应，只通过干预配置或剂量将它们联系起来。

### 3.1 Problem Setup and Intervention / 问题设定与干预

Let a model first answer a question with an initial answer $y_0$, then receive a follow-up input $r$, and finally produce $y_1$. We call $y_1 \neq y_0$ an *output revision*. Whether a revision is desirable depends on the normative force of $r$: a valid argument that changes the correct answer should induce revision, whereas a recheck request, a fallacious argument, or unsupported social pressure should not. This definition concerns observable outputs; it does not require the model to possess a human-like internal belief state.

**中文：** 设模型先对一个问题给出初始答案 $y_0$，随后接收后续输入 $r$，并最终产生 $y_1$。我们将 $y_1 \neq y_0$ 称为一次*输出修正*。修正是否可取取决于 $r$ 的规范性效力：能够改变正确答案的有效论证应当触发修正，而复核请求、谬误论证或无依据的社会压力则不应如此。该定义只涉及可观测输出，并不要求模型具有类人的内部信念状态。

Our primary intervention is the official Assistant-Axis threshold cap for Qwen3-32B `[CITE: Assistant Axis]`. To make the sign convention explicit, let $\mathbf{v}^{-}_{\ell}$ be the stored unit cap vector at layer $\ell$, which points approximately opposite to the human-readable Assistant Axis, and let $\tau_{\ell,c}$ be the threshold for cap configuration $c$. For a residual activation $\mathbf{h}_{\ell,t}$, the intervention is

**中文：** 我们的主要干预是面向 Qwen3-32B 的官方 Assistant-Axis 阈值钳制 `[CITE: Assistant Axis]`。为明确符号约定，令 $\mathbf{v}^{-}_{\ell}$ 表示第 $\ell$ 层存储的单位钳制向量，其方向与便于人类理解的 Assistant Axis 近似相反；令 $\tau_{\ell,c}$ 表示钳制配置 $c$ 在该层的阈值。对于残差激活 $\mathbf{h}_{\ell,t}$，干预定义为

\[
\mathbf{h}'_{\ell,t}
=
\mathbf{h}_{\ell,t}
-
\mathbf{v}^{-}_{\ell}
\big[\langle \mathbf{h}_{\ell,t},\mathbf{v}^{-}_{\ell}\rangle-\tau_{\ell,c}\big]_{+},
\]

where $[z]_{+}=\max(z,0)$. Thus, the operator is state dependent: it changes an activation only when its projection crosses the configured boundary. In the human-readable orientation, the same operation enforces a lower bound on Assistant-Axis projection. It is neither continuous activation addition nor an unconditional shift applied with a coefficient such as $+25$.

**中文：** 其中 $[z]_{+}=\max(z,0)$。因此，该算子依赖于当前状态：只有当激活在相应方向上的投影越过所设边界时，算子才会改变该激活。若换到便于人类理解的方向，同一操作等价于对 Assistant-Axis 投影施加下界。它既不是连续的激活加法，也不是以 $+25$ 之类系数无条件施加的位移。

Unless stated otherwise, we use `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137` and the official per-layer axis and capping assets. The source p25 configuration, `layers_46:54-p0.25`, intervenes on decoder layers 46--53. `[TBD-R: freeze and insert the complete dose set, token/phase scope, and final confirmation decoding protocol.]` Our empirical conclusions are restricted to the tested model, axis, layer range, thresholds, and tasks; we use this setting as a reproducible instance of activation-axis capping rather than evidence for a universal law of capping.

**中文：** 除非另有说明，我们使用 `Qwen/Qwen3-32B` 的 revision `9216db5781bf21249d130ec9da846c4624c16137`，以及官方逐层 axis 和 capping 资产。源配置 p25，即 `layers_46:54-p0.25`，干预解码器第 46--53 层。`[TBD-R：冻结并填入完整剂量集合、token/阶段作用范围以及最终确认实验的解码协议。]` 我们的实证结论严格限定于所测试的模型、轴、层范围、阈值和任务；该设定被用作 activation-axis capping 的一个可复现实例，而非钳制普遍规律的证据。

We additionally prespecify a cross-model replication on `meta-llama/Llama-3.3-70B-Instruct`, using the official model-specific Assistant Axis and the source configuration `layers_56:72-p0.25` (decoder layers 56--71). `[TBD-R: freeze the exact Llama model revision, asset hash, decoding protocol, complete dose set, and all Social/Evidence/static-control results.]` The axis is model specific; no activation vector is transferred between Qwen and Llama. We report estimates and confidence intervals separately by model and do not treat two models as a random sample supporting a universal model-level claim.

**中文：** 我们还预先规划在 `meta-llama/Llama-3.3-70B-Instruct` 上进行跨模型验证，使用该模型专属的官方 Assistant Axis 和源配置 `layers_56:72-p0.25`（解码器第 56--71 层）。`[TBD-R：冻结精确的 Llama 模型 revision、资产哈希、解码协议、完整剂量集合，以及全部 Social/Evidence/静态对照结果。]` Axis 是模型专属的，不会在 Qwen 与 Llama 之间迁移激活向量。我们按模型分别报告估计量与置信区间，不把两个模型视作足以支持模型总体普遍结论的随机样本。

### 3.2 CEDAR-Evidence: Normatively Grounded Revision / CEDAR-Evidence：具有规范依据的修正

CEDAR-Evidence tests whether the intervention preserves sensitivity to the *validity* of a supplied reason. Each base item defines a finite symbolic world using explicit facts and rules, asks a four-option question, and maps the semantic labels `true`, `false`, `unknown`, and `invalid` to A/B/C/D. The initial world has a unique non-invalid answer $y_0$. A certified atomic edit produces an updated world with a different unique answer $y_T$, which serves as the target revision.

**中文：** CEDAR-Evidence 检验干预是否保留模型对所给理由之*有效性*的敏感性。每个基础样本使用显式事实和规则定义一个有限符号世界，提出一道四选一问题，并将 `true`、`false`、`unknown` 和 `invalid` 四个语义标签映射到 A/B/C/D。初始世界具有唯一的非 invalid 答案 $y_0$。一次经过认证的原子编辑会生成一个更新后的世界，其中存在不同的唯一答案 $y_T$，该答案即目标修正。

From each base we construct four paired follow-up conditions / 我们从每个基础样本构造四个配对的后续条件：

| Condition / 条件 | Added input / 新增输入 | Normative response / 规范反应 | Confound controlled / 所控制的混杂 |
|---|---|---|---|
| N — Neutral / 中性 | Neutral continuation; no new reason / 中性续写，不提供新理由 | Keep $y_0$ / 保持 $y_0$ | Spontaneous drift / 自发漂移 |
| R — Recheck / 复核 | Request an independent recheck; no new information / 请求独立复核，不提供新信息 | Keep $y_0$ / 保持 $y_0$ | Revision caused by being asked again / 仅因再次被询问而修正 |
| V — Valid evidence / 有效证据 | A valid justification that warrants the certified atomic edit / 足以支持已认证原子编辑的有效论证 | Revise to $y_T$ / 修正为 $y_T$ | Ability to update for a sufficient reason / 因充分理由而更新的能力 |
| F — Fallacious evidence / 谬误证据 | A matched argument for the same edit and target with one fatal inference / 针对同一编辑和目标、但含一个致命推理错误的匹配论证 | Keep $y_0$ / 保持 $y_0$ | Surface persuasiveness without validity / 只有表面说服力而无有效性 |

The critical contrast is not that V merely *sounds* more evidential than F. V and F share the base world, proposed edit, induced target, option mapping, response contract, and argument wrapper. Their difference is certified formally. A Python fixed-point reasoner and Prover9 must agree on the initial and edited labels. The V edit must be the unique certified edit yielding the target answer. The F argument contains exactly one structurally identified fatal inference; its premises are entailed but its critical claim is not, and a Mace4 countermodel or an opposite proof demonstrates why the proposed target is unwarranted. N and R retain the unchanged world and never expose the target.

**中文：** 关键对比并不是 V 仅仅比 F *听起来*更像证据。V 与 F 共享基础世界、所提编辑、诱导目标、选项映射、响应契约和论证外壳；二者差异经过形式化认证。Python 不动点推理器与 Prover9 必须对初始标签和编辑后标签达成一致。V 中的编辑必须是能够产生目标答案的唯一已认证编辑。F 论证则恰好包含一个在结构上可定位的致命推理错误：其前提可由世界推出，但关键主张不可推出；Mace4 反模型或相反结论的证明进一步说明为何所提目标没有充分依据。N 与 R 保持世界不变，且从不暴露目标答案。

The construction balances transition direction, target answer position, formal structure, and difficulty variants before model execution. Difficulty is controlled through proof depth, necessary rules, distractors, edit location, and the location of the fatal F step rather than through opaque language alone. All conditions for a base retain the same answer mapping and response contract. `[TBD-R: insert the final confirmation-set family count, base count, structure classes, error-class balance, human/automatic audit counts, and source/license statement.]`

**中文：** 在执行模型之前，构造过程会平衡转换方向、目标答案位置、形式结构和难度变体。难度通过证明深度、必要规则、干扰项、编辑位置以及 F 中致命步骤的位置进行控制，而不是只依赖不透明的语言差异。同一基础样本的所有条件保留相同的答案映射和响应契约。`[TBD-R：填入最终确认集的 family 数、base 数、结构类别、错误类别平衡、人工/自动审计数量以及来源/许可证说明。]`

For each condition, the model is first shown the base problem and its initial correct answer, then receives the paired follow-up and answers the original question again. The primary response is scored by intention to treat as `initial`, `induced_target`, `other_wrong`, or `invalid_or_abstain`; truncated or unparsable generations are never repaired or selectively rerun. `[TBD-R: insert the final thinking/non-thinking configuration, decoding parameters, maximum length, and number of generations.]`

**中文：** 在每个条件下，模型先看到基础问题及其初始正确答案，随后接收配对的后续输入，并再次回答原问题。主要响应按意向性分析原则被计为 `initial`、`induced_target`、`other_wrong` 或 `invalid_or_abstain`；被截断或无法解析的生成不会被修复，也不会被选择性重跑。`[TBD-R：填入最终 thinking/non-thinking 配置、解码参数、最大长度和生成次数。]`

### 3.3 CEDAR-Social: Pressure Beyond Target Exposure / CEDAR-Social：超越目标暴露的社会压力

CEDAR-Social asks a different question: does social wording change the answer boundary beyond the effect of merely exposing the model to a target option? We derive 600 bases from the official SycoBench-600 release `[CITE: SycoBench]`. For each natural multiple-choice question, the initial assistant response is the gold option and the target is a different option. Two follow-ups are paired:

**中文：** CEDAR-Social 研究另一个问题：在仅仅向模型暴露目标选项的效应之外，社会性措辞是否会进一步改变答案边界？我们从官方 SycoBench-600 版本中构造 600 个基础样本 `[CITE: SycoBench]`。对于每道自然多选题，assistant 的初始响应为金标准选项，目标则是另一个选项。我们配对构造两种后续输入：

- **AO (answer/target only)** exposes the target label and asks the model to answer again, without a speaker, correctness claim, evidence, authority, reward, or majority cue.<br>
  **中文：AO（仅答案/目标）** 只暴露目标标签并要求模型再次作答，不包含说话者、正确性主张、证据、权威、奖励或多数意见线索。
- **S (social pressure)** exposes the same target and adds only a first-person confidence statement and a request to reconsider, without adding task-relevant facts or reasons.<br>
  **中文：S（社会压力）** 暴露同一个目标，并且只增加第一人称的确信表达和重新考虑的请求，不添加与任务相关的事实或理由。

AO and S share the source question, initial answer, target, option order, and response contract. Four minimal-edit templates are balanced within domain and difficulty, and their token-length difference is bounded under the frozen tokenizer. Construction is validated automatically over all records and was reviewed in full by one data lead. We therefore do not claim two independent annotators, inter-annotator agreement, or Cohen's kappa.

**中文：** AO 与 S 共享源问题、初始答案、目标答案、选项顺序和响应契约。四种最小编辑模板在领域与难度内保持平衡，并在冻结 tokenizer 下约束其 token 长度差。全部记录均经过自动构造验证，并由一名数据负责人完整审阅。因此，我们不声称存在两名独立标注者，也不报告标注者一致率或 Cohen's kappa。

The primary direct readout does not generate free text. With thinking disabled, we read the next-token logits of A/B/C/D after the fixed assistant prefix `{"answer":"` and define the target-relative margin for base $i$, arm $a\in\{AO,S\}$, and cap condition $c$ as

**中文：** 主要的直接读数不生成自由文本。在关闭 thinking 的情况下，我们读取固定 assistant 前缀 `{"answer":"` 之后 A/B/C/D 的 next-token logits，并将基础样本 $i$、实验臂 $a\in\{AO,S\}$ 和钳制条件 $c$ 下相对于目标的 margin 定义为

\[
m_{i,a}(c)=
\operatorname{logit}_{i,a,c}(y_T)-
\operatorname{logit}_{i,a,c}(y_0).
\]

This readout covers all 600 bases under both arms and uncapped/p25 conditions. A 240-base subset, selected by a frozen hash rule before model execution, additionally measures free-generation target following, completion, response length, and reasoning/final-answer consistency. Direct margin is the prespecified primary readout; the free-generation track defines its behavioral boundary rather than overriding it.

**中文：** 该读数覆盖全部 600 个基础样本、AO/S 两个实验臂以及 uncapped/p25 两种条件。在模型执行前，另有 240 个基础样本依据冻结的哈希规则选出，用于测量自由生成中的目标跟随、完成情况、响应长度以及推理与最终答案的一致性。直接 margin 是预先指定的主要读数；自由生成轨道用于界定其行为边界，而不是取代它。

### 3.4 Estimands / 估计量

Let $U_X(c)$ denote the probability of revising to the paired target in CEDAR-Evidence condition $X\in\{N,R,V,F\}$ under cap condition $c$. We use the recheck condition to remove answer changes caused merely by a second request. The primary epistemic estimand is the *warranted-evidence gain*

**中文：** 令 $U_X(c)$ 表示在钳制条件 $c$ 下，CEDAR-Evidence 条件 $X\in\{N,R,V,F\}$ 中修正到配对目标的概率。我们使用 recheck 条件扣除仅由再次请求所引起的答案变化。主要认识论估计量是*充分证据增益*（warranted-evidence gain）：

\[
\operatorname{WEG}(c)=U_V(c)-U_R(c).
\]

The epistemic cost at dose $c$ is the loss in this gain relative to the uncapped model:

**中文：** 剂量 $c$ 下的认识论代价，定义为该增益相对于未钳制模型的损失：

\[
\operatorname{Cost}_V(c)
=
\operatorname{WEG}(\mathrm{uncapped})-
\operatorname{WEG}(c).
\]

Thus, $\operatorname{Cost}_V(c)>0$ means that the intervention reduces the additional target revision elicited by valid evidence over a content-free recheck. We always decompose this contrast into changes in $U_V$ and $U_R$: a fall in valid-evidence response and an increase in generic recheck susceptibility have different interpretations even if they yield the same cost.

**中文：** 因此，$\operatorname{Cost}_V(c)>0$ 表示：与不含实质内容的复核相比，干预削弱了有效证据额外诱发的目标修正。我们始终将这一对比分解为 $U_V$ 和 $U_R$ 的变化；即使两者产生相同的 cost，有效证据响应下降与一般性复核易感性上升也具有不同解释。

Fallacious arguments provide a separate selectivity diagnostic / 谬误论证提供一个独立的选择性诊断量：

\[
\operatorname{FH}(c)=U_F(c)-U_R(c),
\qquad
\operatorname{Benefit}_F(c)=
\operatorname{FH}(\mathrm{uncapped})-
\operatorname{FH}(c).
\]

We report FH and the change in V--F discrimination as secondary outcomes. They do not replace $\operatorname{Cost}_V$ as the headline endpoint.

**中文：** 我们将 FH 以及 V--F 判别的变化作为次要结果报告；它们不取代 $\operatorname{Cost}_V$ 这一标题级主要终点。

For CEDAR-Social, let $M_a(c)=\mathbb{E}_i[m_{i,a}(c)]$. The incremental social-phrasing effect and its positive-oriented reduction are

**中文：** 对于 CEDAR-Social，令 $M_a(c)=\mathbb{E}_i[m_{i,a}(c)]$。社会性措辞的增量效应及其按“收益为正”方向定义的下降量分别为

\[
\operatorname{SME}(c)=M_S(c)-M_{AO}(c),
\]

\[
\operatorname{Benefit}_S(c)=
\operatorname{SME}(\mathrm{uncapped})-
\operatorname{SME}(c).
\]

We report both the signed interaction $\operatorname{SME}(c)-\operatorname{SME}(\mathrm{uncapped})$ and $\operatorname{Benefit}_S(c)$. The latter is a change of orientation for readability, not a different analysis.

**中文：** 我们同时报告带符号的交互项 $\operatorname{SME}(c)-\operatorname{SME}(\mathrm{uncapped})$ 和 $\operatorname{Benefit}_S(c)$。后者只是为便于阅读而改变方向，并不是另一项分析。

WEG/Cost, FH/Benefit, and SME/Benefit are distinct estimands. CEDAR-Evidence and CEDAR-Social do not share items or statistical units, and target-following probability is not interchangeable with continuous option margin. We therefore do not pool the modules into a single significance test or scalar score. A dose-indexed plot may place $\operatorname{Benefit}_S(c)$ and $\operatorname{WEG}(c)$ on separate axes, but each point retains module-specific uncertainty and native units.

**中文：** WEG/Cost、FH/Benefit 与 SME/Benefit 是不同的估计量。CEDAR-Evidence 和 CEDAR-Social 不共享样本或统计单位，目标跟随概率也不能与连续的选项 margin 互换。因此，我们不会把两个模块合并成一个显著性检验或标量分数。以剂量为索引的图可以分别以 $\operatorname{Benefit}_S(c)$ 和 $\operatorname{WEG}(c)$ 为轴，但每个点都必须保留所属模块的不确定性和原生单位。

### 3.5 Statistical Analysis / 统计分析

CEDAR-Evidence pairs intervention conditions within each base and treats the formal family as the independent cluster. Confidence intervals are obtained by paired family-cluster resampling; repeated conditions or generations within a family do not increase the independent sample size. `[TBD-R: insert the frozen number of bootstrap draws, random seed, multiplicity correction, materiality criterion, and precision target for the confirmation set.]` We report $U_N,U_R,U_V,U_F$, WEG, Cost, FH, V--F discrimination, full transition matrices, invalidity, truncation, option mass, entropy, and target-minus-initial margin. Proportional warranted-revision retention is reported only if uncapped WEG supplies a stable denominator.

**中文：** CEDAR-Evidence 在每个基础样本内部配对各干预条件，并将形式 family 视为独立 cluster。置信区间通过配对的 family-cluster 重采样获得；同一 family 内的重复条件或重复生成不会增加独立样本量。`[TBD-R：填入冻结的 bootstrap 次数、随机种子、多重比较校正、实质性判据和确认集精度目标。]` 我们报告 $U_N,U_R,U_V,U_F$、WEG、Cost、FH、V--F 判别、完整转移矩阵、无效输出率、截断率、选项概率质量、熵，以及目标减初始答案的 margin。只有当 uncapped WEG 提供稳定分母时，才报告有充分理由修正的比例保留率。

CEDAR-Social uses the SycoBench base as the resampling unit and a 10,000-draw bootstrap stratified by domain and difficulty. A hierarchical margin model with cap-by-arm interaction and random intercepts for base and domain serves as a sensitivity analysis. We report the AO and S cap effects separately, leave-one-domain/template analyses, option-space mass and top-token integrity, and all free-generation discrete outcomes.

**中文：** CEDAR-Social 以 SycoBench 的基础样本作为重采样单位，进行按领域和难度分层的 10,000 次 bootstrap。含 cap-by-arm 交互项以及 base 和 domain 随机截距的分层 margin 模型作为敏感性分析。我们分别报告 AO 与 S 的钳制效应、leave-one-domain/template 分析、选项空间概率质量与 top-token 完整性，以及所有自由生成离散结果。

The two modules are analyzed separately. Any static-capability comparison uses the same relevant readout and harness at matched doses and a separately prespecified equivalence margin. `[TBD-R: insert the final static tasks and equivalence criterion.]` No result is excluded because its direction conflicts with the headline hypothesis.

**中文：** 两个模块分别进行分析。任何静态能力比较都应在匹配剂量下使用相同的相关读数和评测 harness，并采用单独预先指定的等效性界值。`[TBD-R：填入最终静态任务和等效性判据。]` 不会因某项结果的方向与标题级假设冲突而将其排除。

## 4. A Hidden Epistemic Cost of Activation Capping / 激活钳制的隐性认识论代价

### 4.1 Capping Reduces the Incremental Effect of Social Phrasing at the Direct Answer Boundary / 钳制降低社会性措辞对直接答案边界的增量效应

`[V-AUDIT: frozen numbers; close the documented sign-convention audit before final confirmatory wording.]`

`[中文：数字已冻结；在使用最终确认性措辞前，须闭合已有记录的符号约定审计。]`

We first verify that the official p25 cap changes the behavioral quantity it is intended to control after target exposure is separated from social wording. Across 600 strictly paired SycoBench bases, p25 increased the target-minus-initial margin in both conditions: by $+1.26792$ logits under AO and $+0.48167$ under S. The resulting cap-by-arm interaction was

**中文：** 我们首先检验：在将目标暴露与社会性措辞分离后，官方 p25 钳制是否改变其预期控制的行为量。在 600 个严格配对的 SycoBench 基础样本上，p25 在两种条件下都提高了“目标减初始答案”的 margin：AO 条件提高 $+1.26792$ logits，S 条件提高 $+0.48167$。由此得到的 cap-by-arm 交互项为

\[
\Delta_{\mathrm{cap}}(S-AO)=-0.78625\ \text{logit},
\]

with a 95% stratified base-bootstrap interval of $[-0.86000,-0.71104]$. Leave-one-domain-out and leave-one-template-out estimates were all negative, while the option-mass and top-token-in-option interactions were effectively zero.

**中文：** 其 95% 分层 base-bootstrap 区间为 $[-0.86000,-0.71104]$。leave-one-domain-out 和 leave-one-template-out 的估计均为负，而选项概率质量和 top-token-in-option 的交互项实际上为零。

The decomposition is essential to the interpretation. Capping did **not** lower the S target margin in absolute terms; instead, it increased the margin less under S than under target exposure alone. We therefore describe the result as a reduction in the *incremental effect of social phrasing* on a direct forced-choice answer boundary. It is not evidence that p25 reduces an internal state of “true sycophancy.”

**中文：** 这一分解对解释至关重要。钳制并**没有**在绝对意义上降低 S 条件的目标 margin；相反，它在 S 条件下带来的 margin 增幅小于仅有目标暴露时的增幅。因此，我们将该结果表述为：社会性措辞对直接强制选择答案边界的*增量效应下降*。这不能证明 p25 降低了某种内部的“真实谄媚”状态。

The free-generation track further limits this claim. Strict target-following SocialExcess and completion SocialExcess were both $0$, with no discrete S answer migrations under p25. The observed benefit is therefore specific to the direct margin readout in this experiment, rather than a demonstrated change in free-generation choice.

**中文：** 自由生成轨道进一步限定了这一主张。严格目标跟随 SocialExcess 与完成型 SocialExcess 均为 $0$，且 p25 下没有发生离散的 S 答案迁移。因此，本实验所观察到的收益只适用于直接 margin 读数，并未证明自由生成选择发生改变。

### 4.2 Capping Suppresses Warranted Evidence-Driven Revision / 钳制抑制由充分证据驱动的修正

`[TBD-R: intended headline result paragraph; every value and inferential verb must be replaced or removed after the prespecified confirmation analysis.]`

`[中文：预期的标题级结果段；预先指定的确认分析完成后，必须替换或删除其中每一个数值和推断性动词。]`

On the independent CEDAR-Evidence confirmation set, warranted-evidence gain fell from `[TBD: WEG_uncapped]` without capping to `[TBD: WEG_p25]` under the official p25 configuration. This corresponds to

**中文：** 在独立的 CEDAR-Evidence 确认集上，充分证据增益从未钳制时的 `[TBD: WEG_uncapped]` 下降到官方 p25 配置下的 `[TBD: WEG_p25]`。这对应于

\[
\operatorname{Cost}_V(p25)=\text{[TBD]}\ \text{percentage points},
\]

with a family-clustered 95% confidence interval of `[TBD]` and a prespecified materiality decision of `[TBD]`. Under the working headline outcome, the interval and materiality test support $\operatorname{Cost}_V(p25)>0$: capping suppresses the extra revision elicited by a formally sufficient reason over a reason-free recheck.

**中文：** 其 family-clustered 95% 置信区间为 `[TBD]`，预先指定的实质性判定为 `[TBD]`。在当前标题级预期结果下，该区间与实质性检验支持 $\operatorname{Cost}_V(p25)>0$：相对于不含理由的复核，钳制抑制了形式上充分的理由额外诱发的修正。

`[TBD-R: decomposition paragraph.]` The decline in WEG is driven by `[TBD: change in U_V versus change in U_R]`, rather than by `[TBD: alternative explanation]`. Neutral and recheck controls remain `[TBD]`, while invalid/truncated outputs and option-space integrity are `[TBD]`. Fallacious-evidence following changes by `[TBD]`, yielding $\operatorname{FH}(p25)=\text{[TBD]}$ and a V--F discrimination change of `[TBD]`. These secondary outcomes determine whether the cost is best understood as missed warranted revision alone or as a broader loss of reason selectivity.

**中文：** `[TBD-R：分解结果段。]` WEG 的下降主要由 `[TBD：U_V 与 U_R 的相对变化]` 驱动，而不是由 `[TBD：替代解释]` 导致。Neutral 与 recheck 对照保持 `[TBD]`，无效/截断输出及选项空间完整性为 `[TBD]`。谬误证据跟随变化 `[TBD]`，由此得到 $\operatorname{FH}(p25)=\text{[TBD]}$，V--F 判别变化为 `[TBD]`。这些次要结果将决定该代价应被理解为单纯错失有充分理由的修正，还是更广泛的理由选择性损失。

The internal 24-base development validation is not used to support this conclusion. It shared formal structures with calibration data, was not a new-family/new-salt confirmation, and did not support a positive cost; its role is documented only as part of measurement development.

**中文：** 内部 24-base 开发验证不用于支持这一结论。它与校准数据共享形式结构，不是 new-family/new-salt 确认实验，而且不支持正的 cost；其作用只作为测量开发过程的一部分予以记录。

### 4.3 Stronger Capping Produces a Dose-Indexed Safety--Epistemic Trade-off / 更强钳制形成以剂量为索引的安全—认识论权衡

`[TBD-R: intended dose result.]`

`[中文：预期的剂量结果。]`

Across uncapped and `[TBD: frozen cap configurations]`, the reduction in the social-phrasing increment grows from `[TBD]` to `[TBD]`, while warranted-evidence gain declines from `[TBD]` to `[TBD]`. A prespecified dose trend of `[TBD]` shows that the result is not unique to one threshold. Figure 2c displays $\operatorname{Benefit}_S(c)$ against $\operatorname{WEG}(c)$, with separate module-specific confidence intervals and the official p25 point marked explicitly.

**中文：** 在 uncapped 与 `[TBD：已冻结的钳制配置]` 之间，社会性措辞增量的下降从 `[TBD]` 增至 `[TBD]`，而充分证据增益从 `[TBD]` 降至 `[TBD]`。预先指定的剂量趋势 `[TBD]` 表明该结果并非某一个阈值的偶然现象。Figure 2c 绘制 $\operatorname{Benefit}_S(c)$ 与 $\operatorname{WEG}(c)$ 的关系，分别保留各模块专属的置信区间，并明确标出官方 p25 点。

We call this a *dose-indexed trade-off*, not a Pareto frontier: the axes come from independent tasks with different units and independent sampling structures, and the points trace one intervention family rather than a set of mutually nondominated methods.

**中文：** 我们称之为*以剂量为索引的权衡*，而不是 Pareto frontier：两个坐标轴来自相互独立、单位不同且采样结构独立的任务，这些点追踪的是单一干预族，而不是一组互不支配的方法。

### 4.4 Static Capability Tests Do Not Reveal the Dynamic Cost / 静态能力测试无法揭示动态代价

`[TBD-R: required to retain “missed by standard capability evaluations” in the title-level claim.]`

`[中文：若要在标题级主张中保留“被标准能力评测遗漏”，本结果不可缺少。]`

At the doses that produce a positive epistemic cost, performance on the matched static controls changes by `[TBD]` and remains `[TBD: equivalent/not equivalent]` within the independently prespecified margin `[TBD]`. In contrast, WEG changes by `[TBD]` under the same readout family and cap configuration. This dissociation shows `[TBD: only if supported]` that static answer accuracy can remain apparently preserved while responsiveness to valid new evidence is degraded.

**中文：** 在产生正认识论代价的剂量下，匹配静态对照上的表现变化 `[TBD]`，并在独立预先指定的界值 `[TBD]` 内保持 `[TBD：等效/不等效]`。相比之下，在同一读数族和钳制配置下，WEG 变化 `[TBD]`。这种分离仅在数据支持时才能表明 `[TBD：仅在得到支持时保留]`：静态答案准确率表面上可以保持不变，而对有效新证据的响应能力已经受损。

### 4.5 Cross-Model Replication on Llama-3.3-70B / 在 Llama-3.3-70B 上的跨模型验证

`[TBD-R: all results in this subsection.]`

`[中文：本小节全部结果均为 TBD-R。]`

We repeat the paired CEDAR-Social and CEDAR-Evidence protocols on Llama-3.3-70B with the official model-specific axis and cap configuration. The primary comparison is uncapped versus `layers_56:72-p0.25`; the dose extension uses `[TBD: frozen model-specific percentile configurations]`. For Llama, the social-phrasing interaction is `[TBD]`, warranted-evidence gain changes from `[TBD]` to `[TBD]`, and $\operatorname{Cost}_V(p25)=\text{[TBD]}$ with a family-clustered 95\% confidence interval of `[TBD]`. Free-generation and matched static-control boundaries are `[TBD]`.

**中文：** 我们使用 Llama-3.3-70B 专属的官方 axis 与钳制配置，重复配对的 CEDAR-Social 和 CEDAR-Evidence 协议。主要比较为 uncapped 与 `layers_56:72-p0.25`；剂量扩展使用 `[TBD：已冻结的模型专属 percentile 配置]`。在 Llama 上，社会性措辞交互为 `[TBD]`，充分证据增益从 `[TBD]` 变为 `[TBD]`，且 $\operatorname{Cost}_V(p25)=\text{[TBD]}$，其 family-clustered 95\% 置信区间为 `[TBD]`。Free-generation 与匹配静态对照边界均为 `[TBD]`。

We interpret replication at the level of model-specific effects. If both audited models support the same bounded pattern, we may write “across two model families.” If they differ in sign, magnitude, modality, or dose response, we report that heterogeneity as a result and retain the Qwen-scoped claim rather than averaging it away.

**中文：** 跨模型复现按模型专属效应解释。若两个通过审计的模型都支持同一受限模式，我们可以写“跨两个模型家族”；若它们在方向、幅度、输出模态或剂量响应上不同，则把这种异质性作为结果报告，并保留限于 Qwen 的主张，而不是用平均值将差异抹去。

## 5. Why Does the Cost Arise? / 这种代价为何产生？

The behavioral result establishes *what* changes but does not identify whether capping alters argument evaluation, final choice, or the computation connecting them. We therefore use progressively stronger evidence: independent readouts, phase/layer interventions, geometric diagnostics, and bidirectional activation patching. Descriptive geometry alone is never treated as causal mediation.

**中文：** 行为结果说明了*什么*发生变化，却不能确定钳制改变的是论证评估、最终选择，还是连接二者的计算过程。因此，我们依次使用证据强度逐步提高的方法：独立读数、阶段/层干预、几何诊断以及双向 activation patching。仅凭描述性几何关系绝不会被视为因果中介证据。

### 5.1 Separating Validity Assessment from Final Choice / 区分有效性评估与最终选择

The Answer assay repeats the N/R/V/F revision task, whereas an independent Validity assay asks whether the V or F justification is sufficient without asking for the original answer. Each assay is run under uncapped, prefill-only, decode-only, and full-p25 conditions. This $2\times2$ intervention separates processing of the supplied reason from autoregressive answer formation while retaining the historical full-p25 path as an anchor.

**中文：** Answer assay 重复 N/R/V/F 修正任务，而独立的 Validity assay 只询问 V 或 F 的论证是否充分，不要求回答原问题。两种 assay 均在 uncapped、prefill-only、decode-only 和 full-p25 条件下运行。该 $2\times2$ 干预在保留历史 full-p25 路径作为锚点的同时，将所给理由的处理与自回归答案形成区分开来。

`[TBD-M: result paragraph.]` Full p25 changes `[TBD: Validity discrimination, Answer selectivity, or both]`. The effect is concentrated in `[TBD: prefill/decode/interaction]` and layers `[TBD]`, with `[TBD]` family-cluster uncertainty. This pattern supports the bounded interpretation that capping primarily affects `[TBD: evaluation/choice/cross-stage computation]`; if the prespecified phase/layer gate fails, this subsection will be retitled as a diagnostic analysis rather than a localization result.

**中文：** `[TBD-M：结果段。]` Full p25 改变了 `[TBD：Validity 判别、Answer 选择性，或二者]`。效应集中在 `[TBD：prefill/decode/交互]` 和 `[TBD]` 层，family-cluster 不确定性为 `[TBD]`。该模式只支持一个有边界的解释：钳制主要影响 `[TBD：评估/选择/跨阶段计算]`；若预先指定的阶段/层 gate 未通过，本小节将改名为诊断分析，而不会表述为定位结果。

### 5.2 Alignment with the Capped Direction / 与钳制方向的对齐

We next test whether the cost is larger on items whose uncapped trajectories enter the capping region more strongly or whose valid-update displacement aligns with the actual cap direction. The analysis relates item-level change in warranted-revision margin to counterfactual exposure and actual correction dose, controlling for baseline difficulty, sequence length, and family. Family-level permutation tests provide a null that preserves the paired structure.

**中文：** 接下来，我们检验以下样本是否具有更大的代价：其 uncapped 轨迹更强地进入钳制区域，或其有效更新位移与实际钳制方向更为对齐。分析将样本级有充分理由修正 margin 的变化与反事实暴露、实际 correction dose 联系起来，同时控制基线难度、序列长度和 family。family 级 permutation test 提供保留配对结构的零分布。

`[TBD-M: geometry result.]` Greater `[TBD: exposure/alignment]` is associated with `[TBD: larger/smaller]` loss of warranted-revision margin, whereas the corresponding relation for F and N/R is `[TBD]`. This result may be described as geometric alignment or association. It cannot by itself establish a shared circuit or causal mediator.

**中文：** `[TBD-M：几何结果。]` 更高的 `[TBD：暴露/对齐]` 与 `[TBD：更大/更小]` 的有充分理由修正 margin 损失相关，而 F 与 N/R 上的对应关系为 `[TBD]`。该结果可以描述为几何对齐或关联，但不能单独证明共享回路或因果中介。

### 5.3 Bidirectional Intervention Tests Causal Mediation / 双向干预检验因果中介

Finally, we intervene at the layer/token windows fixed by the preceding analysis. In the induction direction, post-cap activations are patched into an uncapped trajectory to test whether they reproduce the loss. In the rescue direction, corresponding uncapped activations are patched into a capped trajectory to test whether they restore WEG. Identity patches, wrong-window patches, norm-matched random deltas, and matched orthogonal directions control for generic activation replacement.

**中文：** 最后，我们在前述分析所固定的 layer/token 窗口实施干预。在 induction 方向，将钳制后的激活 patch 到 uncapped 轨迹中，以检验其能否复现损失；在 rescue 方向，将相应的 uncapped 激活 patch 到 capped 轨迹中，以检验其能否恢复 WEG。identity patch、错误窗口 patch、范数匹配的随机 delta 和匹配的正交方向用于控制一般性的激活替换效应。

`[TBD-M: causal result.]` Induction changes WEG by `[TBD]`, while rescue restores `[TBD]` of the capped loss and retains `[TBD]` of the social-side benefit. The real patches exceed the matched controls by `[TBD]`. We will use “causally mediates” only if both directions succeed without causing broad option-space collapse; otherwise the conclusion will remain “localized by intervention” or “diagnostically associated.”

**中文：** `[TBD-M：因果结果。]` Induction 使 WEG 改变 `[TBD]`；rescue 恢复了钳制损失的 `[TBD]`，并保留社会侧收益的 `[TBD]`。真实 patch 相比匹配对照高出 `[TBD]`。只有当两个方向均成功且没有造成广泛的选项空间塌缩时，我们才使用“因果中介”这一表述；否则，结论仍限定为“通过干预定位”或“具有诊断性关联”。

## 6. Reducing the Cost / 降低这种代价

Mechanistic localization suggests that a fixed-axis cap may be too coarse: it acts along a direction that may contain both the intended control signal and valid-update motion. We therefore evaluate a controller ladder, treating mitigation as conditional on held-out matched-benefit evidence rather than as an automatic consequence of the diagnosis.

**中文：** 机制定位提示，固定轴钳制可能过于粗糙：其作用方向可能同时包含预期的控制信号和有效更新运动。因此，我们评估一个控制器阶梯，并将缓解结论严格建立在 held-out、匹配收益的证据之上，而不是把它视为诊断结果自动推出的结论。

### 6.1 Controller Ladder / 控制器阶梯

- **L0: original capping.** The official Assistant-Axis threshold cap provides the reference intervention.<br>
  **中文：L0：原始钳制。** 官方 Assistant-Axis 阈值钳制作为参考干预。
- **L1: orthogonalized actuation.** The intervention preserves the original trigger but projects its actuation direction outside a protected subspace estimated from valid V updates.<br>
  **中文：L1：正交化 actuation。** 干预保留原始 trigger，但将 actuation 方向投影到由有效 V 更新估计得到的受保护子空间之外。
- **L2: gated orthogonal capping.** Both the trigger and actuation are computed outside the protected subspace, so protected valid-update motion neither activates nor receives the full correction.<br>
  **中文：L2：门控正交钳制。** trigger 与 actuation 都在受保护子空间之外计算，使受保护的有效更新运动既不会触发干预，也不会承受完整 correction。

`[TBD-C: insert the frozen operator, protected-subspace estimator, development/held-out split, regularization, and calibration protocol.]` The protected subspace is estimated from V motion only. Including all content-driven changes would risk protecting the same fallacious responses that CEDAR is designed to reject.

**中文：** `[TBD-C：填入冻结后的算子、受保护子空间估计器、development/held-out 划分、正则化和校准协议。]` 受保护子空间只从 V 的运动中估计。若纳入所有由内容驱动的变化，就可能把 CEDAR 原本要拒绝的谬误响应也保护起来。

### 6.2 Matched-Benefit Evaluation / 匹配收益评估

Controllers are compared at matched social benefit or, secondarily, matched actual correction dose. The evaluation jointly reports WEG, FH, direct and free-generation social effects, static capability, output integrity, and computational cost. A higher WEG at an operating point with substantially weaker social-side control is not considered an improvement.

**中文：** 控制器首先在匹配社会侧收益的条件下比较，其次可在匹配实际 correction dose 的条件下比较。评估同时报告 WEG、FH、直接读数和自由生成的社会侧效应、静态能力、输出完整性以及计算成本。如果某个运行点虽然 WEG 更高，却显著削弱了社会侧控制，则不视为改进。

`[TBD-C: result paragraph.]` At matched $\operatorname{Benefit}_S=\text{[TBD]}$, L2 retains `[TBD]` more warranted-evidence gain than L0, with a 95% confidence interval of `[TBD]`, while FH and output integrity change by `[TBD]`. L1 yields `[TBD]`. If the held-out WEG improvement does not exclude zero or appears at only one unmatched dose, this result will be reported as a proof of concept or negative result and removed from the contribution list.

**中文：** `[TBD-C：结果段。]` 在匹配 $\operatorname{Benefit}_S=\text{[TBD]}$ 时，L2 相比 L0 多保留 `[TBD]` 的充分证据增益，95% 置信区间为 `[TBD]`；FH 和输出完整性变化 `[TBD]`。L1 的结果为 `[TBD]`。如果 held-out WEG 的改善未能排除零，或只出现在一个未匹配剂量上，该结果将被报告为概念验证或负结果，并从贡献列表中移除。

## 7. Discussion and Conclusion / 讨论与结论

`[WRITE AFTER RESULTS ARE FROZEN]`

`[中文：待结果冻结后撰写]`

Required closing sequence / 必须采用的收束顺序：

1. The evaluation unit for a safety intervention should include the quality of the reason for an answer change.<br>
   **中文：** 安全干预的评估单位应当纳入促使答案变化的理由质量。
2. Scope all empirical claims to the tested model, official axis, threshold family, layer range, tasks, and readouts.<br>
   **中文：** 将全部实证主张限定于所测试的模型、官方 axis、阈值族、层范围、任务和读数。
3. Reiterate that output revision does not prove human-like belief, epistemic agency, or internal sycophancy.<br>
   **中文：** 再次说明，输出修正不能证明类人信念、认识论主体性或内部谄媚。
4. State limitations: cross-model heterogeneity, a second semantic axis, open-ended natural evidence, longitudinal interaction, human-review strength, and mechanism/controller generalization.<br>
   **中文：** 陈述局限：跨模型异质性、第二条语义 axis、开放式自然证据、纵向交互、人工审核强度，以及机制/控制器的泛化。
5. Close with: an effective safety intervention should not only suppress undesirable changes; it should preserve the capacity to change when given a genuinely sufficient reason.<br>
   **中文：** 最后指出：有效的安全干预不仅应抑制不良变化，还应在模型获得真正充分的理由时保留其改变的能力。

## Internal Citation Queue / 内部引用队列

No BibTeX key in this draft should be invented. Verify metadata and create keys for / 本稿不得编造任何 BibTeX key；请核验元数据并为以下文献创建引用键：

- Assistant Axis / Assistant Axis。
- Dual-Stance Evaluation of Sycophancy (Buchan et al., 2026) / 双立场谄媚评估。
- When Truth Is Overridden (AAAI-26) / 当真相被覆盖。
- Refusal in Language Models Is Mediated by a Single Direction (NeurIPS 2024) / 语言模型中的拒答由单一方向介导。
- WMDP (ICML 2024).
- Belief-R (EMNLP 2024).
- SycoBench-600 (Findings of ACL 2026).
- CAA, ITI, KL-then-steer/Steering Without Side Effects, BASIL, T3, and the nearest July 2026 work after full-text verification / CAA、ITI、KL-then-steer/Steering Without Side Effects、BASIL、T3，以及完成全文核验后确定的 2026 年 7 月最近邻工作。
