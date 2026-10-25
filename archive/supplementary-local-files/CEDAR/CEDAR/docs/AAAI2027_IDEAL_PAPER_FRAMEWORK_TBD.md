# AAAI 2027 理想论文框架（TBD 目标状态版）

状态：面向后续实验的理想目标框架，不是当前证据结论，也不是投稿定稿  
日期：2026-07-22  
用途：提前固定论文的理论弧线、章节接口、图表位置和结果槽位；后续实验只填充 `TBD` 或切换预设结果模块，不因单轮结果反复重写整篇论文  
配套审计底稿：`docs/AAAI2027_PAPER_FRAMEWORK_V0.md`

## 0. 如何使用这份框架

这份文件有意描述项目在理想实验结果下可以形成的最完整论文。它不是对尚未完成结果的宣称。

全文使用四种状态标记：

- **[KNOWN]**：已经由仓库实验或原始论文支持，可直接保留。
- **[TBD]**：必须由后续实验填写的数字、机制、模型或方法结果。
- **[TARGET]**：希望成立的理论命题或理想结果，用于指导实验，不得在投稿时伪装成事实。
- **[BRANCH]**：若主假设失败，直接替换的预设叙事模块。

写作纪律：

1. 论文骨架固定，单轮实验只更新“结果登记表”和对应模块。
2. 摘要中的所有 `TBD` 必须在最终投稿前替换或删除。
3. 结果不符合预期时，使用已经写好的分支，不临时发明新故事。
4. 开发集可以用于发现和校准，确认性结论必须来自 held-out、外部数据或第二模型。
5. 新方法未通过 persona safety、证据更新、社会压力和输出完整性四项验收前，不写成贡献。

## 1. 理想论文的一句话故事

> **现有人格稳定方法把一个有语义的激活轴当成控制目标，却只用静态能力和人格安全评测验证“无损”。我们证明，轴的语义并不能决定干预的行为效果：threshold capping 在抑制 persona drift 的同时，会在特定交互条件下改变证据更新与社会压力下的答案选择。通过分离轴、算子、阶段和上下文，我们定位这一副作用，并据此提出一个同时保护人格安全、证据分辨、抗社会压力和动态推理的多约束控制方法。**

这条故事由六个环节组成：

1. **Blind spot**：静态能力保持不等于交互式决策保持。
2. **Measurement**：CEDAR 把正确更新、错误论证、社会压力、目标提及和自然稳定分开。
3. **Finding**：官方 capping 存在条件性、方向明确的交互副作用，而非简单的全面退化。
4. **Mechanism**：副作用取决于轴、干预算子、作用阶段和上下文，而不是轴语义的直接外推。
5. **Mitigation**：根据机制构造 context-/phase-aware、evidence-preserving 的多约束控制器。
6. **Validation**：新控制器保留 persona safety，减少错误顺从，同时维持正确更新与输出完整性。

## 2. 标题与定位

### 2.1 理想最终标题

**An Axis Is Not a Controller: Context-Aware Activation Control for Persona-Safe and Epistemically Reliable Language Models**

使用条件：机制比较和新控制器均成功，并至少完成一个外部数据集或第二模型确认。

### 2.2 更稳健的完整标题

**Beyond Persona Stability: Auditing and Mitigating the Interactive Side Effects of Activation Capping**

使用条件：副作用得到确认，新方法有正结果，但机制可能尚未完全定位。

### 2.3 无新方法时的保底标题

**Beyond Persona Stability: A Contrastive Audit of Activation Capping in Interactive Decision Making**

使用条件：评测与副作用成立，但改进控制器未达到验收标准。

### 2.4 项目与方法命名

- **CEDAR**：*Contextual Evaluation of Drift and Answer Revision*，作为评测框架。
- **新控制器名称**：`[TBD-Controller]`。机制明确后再命名，避免名称预设错误原理。
- 不再将 CEDAR 展开为预设“认识论代价必然存在”的名称。

## 3. 理想摘要模板

下面是最终目标摘要。投稿前只需替换 `TBD`，而不是重新设计摘要结构。

> Activation capping has been proposed to stabilize the default persona of language models while preserving performance on static capability benchmarks. However, static preservation does not establish whether a model retains the ability to revise answers selectively during interaction. We introduce **CEDAR**, a contrastive evaluation framework that separates warranted revision, fallacious persuasion, target exposure, social pressure, and no-information controls, while jointly measuring answer transitions, decision margins, intervention exposure, reasoning integrity, and persona safety. Across **[TBD: number]** models and **[TBD: datasets/tasks]**, we find that the official Assistant-Axis cap reduces persona-jailbreak success by **[TBD]**, yet changes interactive answer policies: social-pressure-induced target following changes by **[TBD]**, valid-evidence updating changes by **[TBD]**, and output-distribution integrity changes by **[TBD]**. Matched operator and phase interventions show that these effects arise from **[TBD: evidence encoding / reasoning dynamics / answer-policy arbitration / mixed mechanism]**, rather than **[TBD: general degradation / simple answer inertia / target priming]**. These results support an **Axis–Operator–Context** account: a semantically meaningful representation direction is a measurement coordinate, not automatically a safe control objective. Guided by this diagnosis, we develop **[TBD-Controller]**, which conditions activation control on **[TBD signals/phases]** and protects **[TBD evidence or decision subspace]**. Relative to official capping, it retains **[TBD]%** of persona-safety gains while improving warranted updating by **[TBD]pp**, reducing socially induced answer shifts by **[TBD]pp**, and preserving output integrity across **[TBD]** held-out and external evaluations. Our findings establish interactive epistemic behavior as a necessary dimension for evaluating and designing representation-level safety interventions.

摘要的固定逻辑是：

1. 现有方法及遗漏；
2. 我们提出什么评测；
3. 发现了什么副作用；
4. 定位了什么机制；
5. 提出了什么改进；
6. 改进带来了什么多目标收益。

## 4. 理想贡献列表

最终引言建议写成四项贡献：

1. **Interactive evaluation.** We introduce CEDAR, a contrast-cluster framework that distinguishes warranted evidence-based revision from fallacious persuasion, social pressure, target exposure, and spontaneous drift. It evaluates not only task accuracy but also transition direction, decision margin, reasoning–answer consistency, intervention exposure, and output integrity.
2. **Previously unmeasured side effects.** We provide the first systematic audit of Assistant-Axis capping on interactive answer revision. Across `[TBD]`, we show that static capability preservation can coexist with `[TBD: social backfire / evidence-discrimination loss / operator-dependent instability]`.
3. **Mechanistic account.** Through matched steering–capping comparisons, phase-specific interventions, probes and causal rescue, we show that `[TBD mechanism]`. This motivates the Axis–Operator–Context principle: behavioral effects are jointly determined by representation semantics, intervention operators and interaction context.
4. **Safer control.** We introduce `[TBD-Controller]`, a multi-constraint controller that retains persona-stability benefits while protecting valid updating, isolating social pressure and preserving dynamic reasoning. It improves the safety–epistemic Pareto frontier by `[TBD]` across `[TBD]`.

如果第四项失败，删除 contribution 4，论文仍保留完整的评测、发现和机制贡献。

## 5. 理论框架

### 5.1 静态能力保持不等于交互决策保持

模型在 MMLU、GSM8K 或 IFEval 上保持平均分，只说明它仍能完成静态任务。它不能说明模型在已经给出答案后，是否能区分：

- 真正推翻答案的有效证据；
- 具体但无效的错误论证；
- 不含证据的社会压力；
- 单纯出现的目标答案；
- 没有任何新信息的复核请求。

因此，表示干预的完整安全性至少包含：

\[
\text{Safety} = \text{Persona Safety} + \text{Epistemic Selectivity} + \text{Social Robustness} + \text{Output Integrity}.
\]

### 5.2 Axis–Operator–Context 原则

核心理论命题保持为：

\[
\Delta \text{Behavior}
= f(\text{Axis},\text{Operator},\text{Phase},\text{Context},\text{Model}).
\]

- **Axis**：方向编码或相关的语义。
- **Operator**：持续加法、减法、阈值整流、部分整流、多层 capping。
- **Phase**：prefill、thinking、pre-answer、final answer、完整生成。
- **Context**：有效证据、谬误、社会压力、persona drift、解码方式和提示格式。
- **Model**：不同模型上的方向极性、层、尺度和基线能力。

关键主张：

> 一条轴可以可靠地测量人格变化，但仍然不是一个可被任意操纵而不产生副作用的控制旋钮。

### 5.3 多维安全区域

理想控制目标不是把模型永远推到 Assistant Axis 的某个端点，而是在动态交互中保持一个可行区域：

\[
\mathcal{R}_{safe}=\{h:\
s_{persona}(h)\ge \tau_p,\
d'_{evidence}(h)\ge \tau_e,\
s_{social}(h)\le \tau_s,\
q_{output}(h)\ge \tau_o\}.
\]

其中：

- `s_persona`：默认助手人格或 persona-drift 风险；
- `d'_evidence`：有效证据与无效论证的分辨能力；
- `s_social`：超出纯目标提及的社会压力效应；
- `q_output`：格式、答案质量、概率质量和推理完整性。

新方法的目标是用最小必要干预维持该区域，而不是把单一轴钳得越强越好。

## 6. 可证伪的竞争假设

以下假设保持并行，避免项目只押注一个故事。

| 假设 | 理想可观察模式 | 关键处决实验 | 若成立，论文如何解释 |
|---|---|---|---|
| H1 普遍退化 | V/N/F/S 均恶化，other-wrong、无效输出和 entropy 同时上升 | 四分类、输出质量、随机/范数匹配扰动 | capping 损伤一般计算或输出分布 |
| H2 变顽固/答案冻结 | V 与 F/S 的 update 同时下降；`d'` 稳定而 criterion 上升；N 表面稳定 | SDT、V/F 配对、AO、直接 logit | capping 抬高总体更新门槛 |
| H3 证据分辨受损 | V 降、F 升或 `d'` 下降；validity verdict 本身恶化 | validity assay、hierarchical SDT、证据探针 | capping 损害证据编码或评估 |
| H4 社会性反作用 | S−AO 上升；N 稳定；validity verdict 保持但 final choice 向 target 移动 | held-out S/AO、reasoning–answer 分离、final-phase ablation | capping 改变社会条件下的答案策略 |
| H5 提示暗示伪影 | 删除 “no new information” 后效应消失 | V7.1 cue ablation、自然外部提示 | 原现象主要由提示设计驱动 |
| H6 一般扰动 | 随机或范数匹配方向复制同方向效应 | 多随机方向、正负成对、剂量匹配 | 不是 Assistant Axis 语义，而是模型对干预敏感 |
| H7 算子特异 | 同层同剂量 addition、capping 和部分整流产生不同符号或选择性 | matched-operator grid | 轴语义不能替代算子分析 |
| H8 阶段特异 | 效应集中在 thinking、pre-answer 或 final-answer 阶段 | phase ablation、token-resolved margin | 可构造 phase-aware 控制器 |

理想主结果是 H4 + H7/H8：社会性副作用存在，并且能够由算子与阶段解释。若结果落在 H1、H2 或 H3，论文骨架仍成立，只需替换机制模块。

## 7. 研究问题

- **RQ1 — Evaluation gap**：静态能力保持能否保证交互式证据更新与社会抗性保持？
- **RQ2 — Behavioral decomposition**：capping 改变的是证据分辨力、更新门槛、社会压力效应，还是一般输出质量？
- **RQ3 — Operator dependence**：同一轴上的 addition、threshold capping 和 partial rectification 是否产生不同效应？
- **RQ4 — Mechanistic locus**：效应位于证据编码、thinking 轨迹、答案映射还是最终输出阶段？
- **RQ5 — Safe control**：能否用多信号、上下文和阶段条件控制，在保留 persona safety 的同时减少交互副作用？
- **RQ6 — Generalization**：发现与方法能否跨 prompt、family、数据集、seed、解码方式和模型成立？

## 8. 实验总图：以后只填结果，不重写结构

### Study A — 原方法复现与仪器校准

目的：证明模型版本、Assistant Axis、层、符号、hook、官方 threshold 和 telemetry 实现正确。

固定内容：

- PhilPapers close replication；
- 正负 addition 剂量曲线；
- 官方 p25/p1 触发与修正量；
- persona/default axis positive control；
- 输出 token、A/B mass、entropy 和截断审计。

最终槽位：

- steering replication effect：`TBD / 已有初步值待最终表格统一`；
- official cap exposure：`TBD`；
- output integrity change：`TBD`。

### Study B — CEDAR 交互式行为审计

目的：在统一 base task 上分离五种条件。

- **N**：没有新信息，正确行为是维持；
- **AO**：只提及 target，不含施动者、证据或压力；
- **V**：有效证据，正确行为是更新；
- **F**：无效论证，正确行为是维持；
- **S**：社会压力，正确行为是维持。

主要指标：

\[
\text{Cost}_{V}=P(update\mid V,uncapped)-P(update\mid V,capped)
\]

\[
\text{Benefit}_{F}=P(update\mid F,uncapped)-P(update\mid F,capped)
\]

\[
\text{SocialExcess}=P(update\mid S)-P(update\mid AO)
\]

\[
d'=\Phi^{-1}(P(update\mid V))-\Phi^{-1}(P(update\mid F))
\]

最终槽位：

- `ΔV = TBD`；
- `ΔF = TBD`；
- `ΔS = TBD`；
- `ΔAO = TBD`；
- `ΔN = TBD`；
- `Δd' = TBD`；
- `Δcriterion = TBD`；
- family-clustered interval：`TBD`。

### Study C — 外部与跨域确认

目的：排除自建形式域、审计式 prompt 和开发选择造成的特异性。

理想构成：

- 外部社会压力/flip-flop 数据：`TBD dataset`；
- 外部谬误或说服数据：`TBD dataset`；
- 新 held-out family：`TBD number`；
- 至少两个模型或两个官方配置：`TBD models`；
- thinking/nonthinking 与 greedy/sampling 分层；
- 至少三个 seed 或确定性 logit 主分析。

最终槽位：跨数据集效应 `TBD`，跨模型异质性 `TBD`。

### Study D — 机制定位

目的：把“行为发生了变化”推进到“变化发生在哪一步、由什么算子造成”。

按由简到强的顺序：

1. **行为与 SDT**：区分分辨力和门槛。
2. **连续 logit/margin**：避免 0/1 翻转只反映临界边界。
3. **validity verdict vs final choice**：判断模型是否识别论证有效性却在最终输出顺从。
4. **phase ablation**：prefill-only、thinking-only、pre-answer-only、final-only、full。
5. **matched operator**：同层、同 token、同实际修正范数下比较 addition、cap、partial rectification。
6. **简单内部基线**：DiffMean、L2 probe、attention-head probe，按 family grouped CV。
7. **工具辅助解释**：MUSE、JLens、SAE 或 Activation Oracle 只解释已由行为和探针锁定的现象。
8. **因果 rescue**：activation patching、direction ablation、双向 intervention、随机/邻层/错误 family 对照。

最终机制句模板：

> Capping changes `[TBD internal signal or phase]` while preserving/disrupting `[TBD]`; bidirectional intervention and rescue alter `[TBD behavior]` by `[TBD]`, supporting `[TBD mechanism]` over `[TBD alternatives]`.

### Study E — Persona safety leg

目的：证明我们讨论的是有真实安全收益但存在交互代价的干预，而不是一个没有收益的算子。

必须先通过攻击有效性闸：

- uncapped persona attack ASR ≥ `TBD preregistered threshold`；
- persona 相对 behavior-only lift ≥ `TBD`；
- 公开强 persona、经冻结的多轮 persona，或原作者可比数据；
- judge 与人工抽查一致性达到 `TBD`。

最终槽位：

- official cap safety benefit：`TBD`；
- new controller safety benefit：`TBD`；
- benign refusal/utility：`TBD`。

### Study F — 新控制器

目的：根据 Study D 的机制，仅实现一个最有依据的控制器，而不是同时堆叠多种复杂方法。

候选路径：

1. **Context-gated control**：只有 persona drift 风险高且不是有效证据更新时才触发。
2. **Phase-aware control**：只在机制确定的安全阶段干预，保护证据读取或最终答案映射。
3. **Protected-subspace control**：消除 Assistant Axis 修正中与证据/社会决策方向重叠的分量。
4. **Partial rectification**：根据缺口比例柔性修正，避免 hard threshold 破坏临界轨迹。

选择规则：

- 若 final-answer policy 受损 → 优先 phase-aware 或 final-stage guard；
- 若 evidence verdict 受损 → 优先 protected-subspace；
- 若任意非零 rectification 都有害 → 优先 context gate，而非继续调剂量；
- 若随机扰动同样有害 → 优先降低干预频率和范数，论文改为 intervention fragility；
- 若算子而非方向决定符号 → 将 matched operator 结果作为方法设计核心。

### Study G — 新方法确认

比较：

- uncapped；
- official p25；
- official p1 或安全强度对照；
- continuous steering baseline；
- `[TBD-Controller]`；
- random/norm-matched control。

四项共同验收：

1. **Persona safety**：达到 official p25 安全收益的 `TBD%` 以上；
2. **Epistemic selectivity**：V/F 的 `d'` 不低于 uncapped `TBD equivalence margin`；
3. **Social robustness**：S−AO 不高于 uncapped，并显著优于 p25；
4. **Output integrity**：无效、截断、entropy、A/B mass 和通用能力不恶化超过 `TBD`。

## 9. 理想正文结构

### 1 Introduction

段落 1 主题句：Representation-level interventions can stabilize model personas, but their safety cannot be certified by static capability scores alone.

段落 2 主题句：Interactive answer revision creates a missing test because changing an answer can be either warranted, fallaciously induced, or socially induced.

段落 3 主题句：We show that the official Assistant-Axis cap has operator- and context-dependent effects that are invisible to aggregate static evaluation.

段落 4 主题句：We introduce CEDAR to isolate these effects and use causal diagnostics to locate their source.

段落 5 主题句：The diagnosis leads to `[TBD-Controller]`, a multi-constraint alternative that retains persona safety without sacrificing interactive reliability.

结尾放四项贡献，使用第 4 节模板。

### 2 Background and Related Work

#### 2.1 Persona representations and Assistant-Axis capping

讲清轴如何构造、官方 capping 如何作用、原文验证了什么以及没有验证什么。避免将“没测”写成“做错”。

#### 2.2 Sycophancy, persuasion and answer revision

区分价值附和、客观答案翻转、谬误说服、社会压力和目标曝光。

#### 2.3 Representation engineering and intervention side effects

覆盖 steering、activation addition、ablation、conditional control、Buchan 类能力代价和 operator dependence。

#### 2.4 Mechanistic tools

把 probe、causal patching、MUSE、JLens、SAE 放在“证据等级”中，而不是列工具目录。

### 3 CEDAR Evaluation Framework

#### 3.1 Contrast-cluster construction

同一 base 的 N/AO/V/F/S 只改变新信息性质；family-grouped split 防止结构泄漏。

#### 3.2 Behavioral and continuous estimands

报告 transition matrix、V/F SDT、SocialExcess、option margin、reasoning–answer consistency 和四类错误。

#### 3.3 Intervention and output telemetry

区分 counterfactual/actual exposure、prefill/decode、layer、thinking token、finish reason、entropy 和输出质量。

#### 3.4 Preregistration and audit discipline

说明失败基线、开发集选择、held-out 冻结、未重试截断、模型输出完整保留和人工审查限制。

### 4 Reproducing and Stress-Testing Assistant-Axis Control

#### 4.1 Reproducing public signed steering

证明轴符号、hook 和 margin 读出正确，同时展示大剂量下完整输出分布的变化。

#### 4.2 Auditing official threshold capping

比较 margin、离散选择、exposure 和输出质量，说明“margin 变好”不能自动等于“决策更健康”。

#### 4.3 Establishing persona safety

在有效强攻击基线上比较 uncapped、official cap 与后续新方法。

### 5 Interactive Side Effects of Activation Capping

#### 5.1 Development diagnostics

把 Belief-R、v3–v6 作为范式设计历史，说明为何需要五臂、AO、thinking 分层和 held-out。

#### 5.2 Held-out main effect

填入 V7.1/最终 held-out：`TBD`。

#### 5.3 External and cross-model confirmation

填入自然提示、外部集和第二模型：`TBD`。

#### 5.4 What kind of cost is it?

用 SDT 与四分类在 H1–H4 中做选择，结论为：`TBD`。

### 6 Why Does the Effect Occur?

#### 6.1 Ruling out simple low-axis resistance

保留 E0→E1 否证链，说明粗暴露关联如何被轨迹长度解释。

#### 6.2 Axis versus operator

同层同剂量比较 addition、cap、partial rectification 和随机方向：`TBD`。

#### 6.3 Locating the affected phase

validity/final-choice 与 phase ablation：`TBD`。

#### 6.4 Causal representation analysis

probe、工具解释与双向 rescue：`TBD`。

本节最后给出一句冻结机制结论：`TBD`。

### 7 From One-Dimensional Capping to Multi-Constraint Control

#### 7.1 Design objective

从多维安全区域推导最小必要干预目标。

#### 7.2 `[TBD-Controller]`

只介绍由第 6 节结果直接支持的 gate、phase 或 protected-subspace 组件。

#### 7.3 Main comparison

报告 persona safety、V/F `d'`、S−AO、N、输出质量和通用能力：`TBD`。

#### 7.4 Ablations and Pareto frontier

证明每个组件的必要性，并展示安全—认识论—社会鲁棒性的 Pareto 前沿：`TBD`。

### 8 Discussion

- 为什么同一轴的不同算子可产生不同效果；
- 为什么 persona drift 中的谄媚改善与短形式任务中的社会顺从增加可以同时成立；
- 轴作为 measurement coordinate 与 controller 的区别；
- 模型、层、剂量、prompt 和解码方式的边界；
- 何时应使用 capping，何时应使用条件控制。

### 9 Limitations

- Qwen/模型数量限制：`TBD`；
- 自建形式任务的生态有效性；
- judge 与人工标注限制；
- reasoning 文本不等同真实内部信念；
- probe 可解码不等于因果机制；
- persona safety 数据与有害内容治理限制；
- 新控制器可能增加推理成本：`TBD`。

### 10 Conclusion

固定结论逻辑：

> Static capability preservation is insufficient to certify representation-level safety interventions. CEDAR reveals `[TBD main effect]`; causal analysis attributes it to `[TBD mechanism]`; and `[TBD-Controller]` demonstrates that persona stability and interactive epistemic reliability can be jointly optimized rather than traded blindly along a single axis.

## 10. 图表计划

### Figure 1 — Graphic abstract

一张图完成完整故事：

1. 左：原方法用 Assistant Axis capping 防止 persona drift，静态 benchmark 显示“无损”；
2. 中：CEDAR 用 N/AO/V/F/S 揭示交互式盲区；
3. 中下：同一轴、不同 operator/phase 产生不同结果；
4. 右：多维安全区域与 `[TBD-Controller]`；
5. 右下：安全、V/F、S−AO 和 output integrity 四项最终结果 `TBD`。

### Figure 2 — CEDAR contrast-cluster design

展示共享 base、五臂唯一变化、规范行为和主要 estimands。

### Figure 3 — Reproduction and operator audit

左：signed steering 七剂量曲线；中：top-token/entropy 变化；右：official cap 的 margin 与离散选择分离。

### Figure 4 — Main held-out behavior

V/F/S/AO/N 的 uncapped→cap paired effects、family-bootstrap CI、SDT `d'` 和 criterion。

### Figure 5 — Mechanism localization

validity verdict → thinking → pre-answer margin → final choice 的流程，标出不同 phase 干预与因果 rescue 结果。

### Figure 6 — Multi-objective controller

官方 cap 与 `[TBD-Controller]` 的 Persona Safety × Epistemic Selectivity × Social Robustness × Output Integrity Pareto 图。

### Table 1 — Evaluation coverage in prior work

列：persona drift、jailbreak、static capability、valid evidence、invalid argument、social pressure、target exposure、output integrity、operator telemetry。

### Table 2 — Models, datasets and interventions

集中列出模型 revision、thinking、decode、层、阈值、operator、seed、family 和 split。

### Table 3 — Main behavioral results

所有最终数字 `TBD`，不放开发集调参结果。

### Table 4 — Mechanism tests

每个竞争假设的预测、观测、是否支持和证据等级。

### Table 5 — New controller comparison

persona ASR、V/F `d'`、S−AO、N、other-wrong、invalid、entropy、latency 和静态能力。

## 11. 模块化结果登记表

以后每轮实验只更新此表及对应正文模块。

| 结果槽位 | 当前状态 | 最终值 | 更新正文位置 |
|---|---|---|---|
| Cue-free held-out S effect | V7.1 待运行/待回填 | `TBD` | Abstract, 5.2, Fig.4 |
| Valid update cost | 待确认 | `TBD` | Abstract, 5.2, Table 3 |
| Evidence discrimination `d'` | 待正式计算 | `TBD` | 5.4, Fig.4 |
| SocialExcess | 待 cue-free/外部确认 | `TBD` | 5.2–5.3 |
| External dataset effect | 未完成 | `TBD` | 5.3 |
| Second-model effect | 未完成 | `TBD` | 5.3 |
| Official persona safety benefit | 本地有效基线未完成 | `TBD` | 4.3, 7.3 |
| Matched operator result | 未完成 | `TBD` | 6.2 |
| Phase locus | 未完成 | `TBD` | 6.3 |
| Validity vs final-choice separation | 未完成 | `TBD` | 6.3 |
| Probe generalization | 未完成 | `TBD` | 6.4 |
| Causal rescue | 未完成 | `TBD` | 6.4 |
| New controller design | 等机制选择 | `TBD` | Section 7 |
| New controller safety retention | 未完成 | `TBD` | Abstract, 7.3 |
| New controller epistemic/social gain | 未完成 | `TBD` | Abstract, 7.3 |
| Cross-domain/model generalization | 未完成 | `TBD` | 5.3, 7.3 |

## 12. 不重写全文的分支切换规则

### Branch A — 理想主线：社会性反作用 + 算子/阶段机制 + 新控制器

保留理想标题、完整摘要和 Sections 6–7。

### Branch B — 正确更新受损

把主结果改为“evidence discrimination cost”；Section 6 重点从 final policy 转为 evidence encoding；新方法采用 protected-subspace。其余结构不变。

### Branch C — 普遍退化

把主结果改为“intervention-induced distributional degradation”；强调大剂量 steering、随机方向和 output integrity；新方法采用 minimum-norm/context gate。其余结构不变。

### Branch D — 只有 operator difference，没有稳定行为副作用

论文转为 measurement/methodology：同一轴的 operator-dependent semantics；删除强 social-cost 标题，保留 CEDAR 作为诊断框架。

### Branch E — V7 效应被 cue ablation 推翻

V7 降为 prompt-contingent case；自然外部数据成为主结果。若外部也不成立，转 Branch D，不继续围绕社会顺从硬凑故事。

### Branch F — 新控制器失败

删除摘要最后两句、contribution 4、Section 7 的结果，只保留设计启示；使用保底标题。Sections 1–6 不需要重写。

## 13. 实验优先级与论文写作并行安排

### P0：锁住行为主结果

1. V7.1 cue ablation；
2. 固定 response audit 与四分类；
3. 外部自然 S/AO 或 flip-flop confirmation；
4. 正式 hierarchical SDT。

### P1：补齐安全腿和普适性

5. 有效 persona attack baseline；
6. official cap safety replication；
7. 第二模型或至少跨 seed、跨数据集确认。

### P2：定位机制

8. validity verdict vs final choice；
9. phase ablation；
10. matched operator + partial rectification + random controls；
11. 简单 probe；
12. 只有前述通过后才进入 MUSE/JLens/SAE 和 causal rescue。

### P3：构建方法

13. 根据机制选择一个控制器；
14. 小规模安全—认识论—社会—输出四闸；
15. held-out、外部、第二模型正式比较。

### 可以立刻写且不受结果影响的部分

- Introduction 的问题定义与 evaluation gap；
- Related Work；
- CEDAR 方法、五臂与指标；
- 工程与预注册纪律；
- E−1a/E−1b reproduction；
- E0/E1 的否证链；
- Figure 1 线框和 Table 1；
- Limitations 的固定部分。

### 必须等待的部分

- 摘要最终数字；
- 主标题中的 `increase`、`cost` 或 `mitigating`；
- 冻结机制句；
- 新方法名称与组成；
- 最终 conclusion。

## 14. 理想成功判据

理想论文不是要求所有预期方向都成立，而是要求证据链闭合：

1. **评测有效**：五臂能分离有效证据、无效论证、社会压力与目标曝光。
2. **现象稳健**：至少一个关键交互效应跨 held-out、外部数据或模型成立。
3. **备择被处理**：提示暗示、一般退化、答案冻结、随机扰动和输出损坏都有对应对照。
4. **机制可干预**：不仅可解码，还能通过双向 intervention/rescue 改变行为。
5. **方法有真实多目标收益**：不是靠拒答、冻结答案或破坏输出换取安全。
6. **所有结论与证据等级一致**：开发发现、事后诊断、确认性结果和因果结论不混写。

## 15. 当前最需要团队拍板的三件事

1. 是否接受完整理想标题把最终目标定位为“审计 + 机制 + 新控制器”。
2. 是否接受 CEDAR 的中性展开：*Contextual Evaluation of Drift and Answer Revision*。
3. 是否同意在机制明确前只用 `[TBD-Controller]`，不提前绑定 JLens、MUSE、SAE、CAST 或某一种具体实现。

一旦这三点拍板，全文大结构保持不动；以后只更新第 11 节结果登记表、摘要数字、Section 5/6 的结论句和 Section 7 的实际方法。

