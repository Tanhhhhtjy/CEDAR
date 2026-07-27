# [SUPERSEDED] CEDAR Three-Model Paper and Figure Design / CEDAR 三模型论文与图表设计（已废止）

> Date: 2026-07-26 (Asia/Shanghai)  
> **中文：** 日期：2026-07-26（Asia/Shanghai）  
> Status: superseded on 2026-07-27; retained only as a historical design record. Do not use it for current paper, experiment, or figure decisions.
> **中文：** 状态：已于 2026-07-27 废止，仅保留为历史设计记录。当前论文、实验和图表决策不得再以本文档为依据。
> Current scope: Qwen3-32B plus the planned Llama-3.3-70B-Instruct replication; Gemma has been removed.
> **中文：** 当前范围：Qwen3-32B 加计划中的 Llama-3.3-70B-Instruct 跨模型验证；Gemma 已移出本文。

Historical note: this document once superseded the Llama-only extension in `2026-07-26-cedar-llama-cross-model-replication-decision.md`. That three-model decision has itself been superseded; the Llama-only replication decision and `CEDAR_AAAI27_REVISED_FRAMEWORK_v2.md` now define the active model scope.

**中文：** 历史上，本文档曾取代 `2026-07-26-cedar-llama-cross-model-replication-decision.md` 中仅加入 Llama 的扩展方案。该三模型决策现已被废止；当前模型范围重新以 Llama 验证决策和 `CEDAR_AAAI27_REVISED_FRAMEWORK_v2.md` 为准。

## 1. Approved Narrative / 已批准的叙事

### One-sentence claim / 一句话主张

> **Capping changes when a model abandons an answer, without reliably tracking whether the offered reason warrants revision.**

> **钳制会改变模型何时放弃当前答案，却不能可靠地依据所给理由是否足以支持修正来实施这种改变。**

The paper is an evaluation-and-diagnosis study of epistemic selectivity in activation-space control. It is not organized around a fixed `Cost_V > 0` result, an already established benefit–cost trade-off, or a completed replacement controller.

**中文：** 本文是一篇关于激活空间控制之认识论选择性的评测与诊断论文。全文不再围绕固定的 `Cost_V > 0` 结果、已经成立的收益—代价权衡或已经完成的替代控制器组织。

The main empirical object is the separation between a comparatively stable continuous shift in answer geometry and heterogeneous discrete answer changes. CEDAR asks whether the intervention distinguishes *why* an answer should change by holding the initial problem and proposed target fixed while changing the force of the supplied reason.

**中文：** 主要实证对象是：相对稳定的连续答案几何位移，与具有异质性的离散改答之间的分离。CEDAR 固定初始问题和所提 target，只改变所给理由的效力，从而检验干预是否区分答案“为什么应该改变”。

### Claim boundary / 主张边界

- The general contribution is the CEDAR audit framework and the concept of epistemic selectivity.  
  **中文：** 一般性贡献是 CEDAR 审计框架和认识论选择性这一评测维度。
- The empirical intervention is Assistant-Axis threshold capping across three model families.  
  **中文：** 实证干预是在三个模型家族上实施 Assistant-Axis 阈值钳制。
- Agreement across all three models permits “across three model families,” not “all activation axes” or “all capping methods.”  
  **中文：** 三个模型结果一致时，可以写“跨三个模型家族”，但不能写成“所有 activation axis”或“所有 capping 方法”。
- Before Llama and Gemma results are frozen, every corresponding result and result-dependent sentence remains `[TBD-R]`.  
  **中文：** 在 Llama 和 Gemma 结果冻结前，所有对应结果及依赖这些结果的句子均保持 `[TBD-R]`。

## 2. Model Matrix / 模型矩阵

The approved allocation is **behaviorally symmetric and mechanistically asymmetric**: all three models receive the same core behavioral audit, while the deepest diagnostic and causal analyses remain Qwen-primary.

**中文：** 已批准的分工是“行为实验对称、机制分析非对称”：三个模型均接受同一套核心行为审计，最深入的诊断和因果分析则以 Qwen 为主。

| Role / 角色 | Model / 模型 | Axis and cap identity / Axis 与钳制身份 | Paper status / 论文状态 |
|---|---|---|---|
| Primary discovery and diagnosis / 主要发现与诊断 | `Qwen/Qwen3-32B` | Official model-specific Assistant Axis; official `layers_46:54-p0.25` source configuration / 官方模型专属 Assistant Axis；官方 `layers_46:54-p0.25` 源配置 | Existing frozen and exploratory evidence is reported with its exact evidence identity / 现有冻结与探索性证据按其真实身份报告 |
| Cross-model replication / 跨模型验证 | `meta-llama/Llama-3.3-70B-Instruct` | Official model-specific Assistant Axis; official `layers_56:72-p0.25` source configuration / 官方模型专属 Assistant Axis；官方 `layers_56:72-p0.25` 源配置 | All paper results are `[TBD-R]` until frozen / 冻结前全部论文结果均为 `[TBD-R]` |
| Calibrated cross-model extension / 校准式跨模型扩展 | `google/gemma-2-27b-it` | Official model-specific Assistant Axis; no official released capping layer-range/configuration / 官方模型专属 Assistant Axis；源仓库未发布官方钳制层范围或配置 | All paper results are `[TBD-R]`; report as a source-faithful calibrated extension / 全部论文结果均为 `[TBD-R]`；表述为遵循源方法的校准扩展 |

“The same axis” means that the same semantic construction is instantiated in each model’s own activation space. No activation vector is transferred between models.

**中文：** “同一 axis”是指相同语义构造分别在每个模型自身的激活空间中实例化，不在模型之间迁移激活向量。

### Gemma calibration firewall / Gemma 校准防火墙

The Assistant-Axis release provides a Gemma axis but not a recommended Gemma capping layer range or precomputed capping configuration. Gemma must therefore not be described as a third official p25 configuration replication. The paper and protocol will use the following identity:

**中文：** Assistant Axis 官方发布提供了 Gemma axis，但没有提供推荐的 Gemma 钳制层范围或预计算 capping 配置。因此，Gemma 不能被描述为“第三套官方 p25 配置复现”。论文和协议采用以下身份：

1. Use the released model-specific Gemma Assistant Axis.  
   **中文：** 使用官方发布的 Gemma 模型专属 Assistant Axis。
2. Select the capping layer range on a calibration split that is disjoint from all formal CEDAR evaluation items.  
   **中文：** 在与正式 CEDAR 评测样本完全独立的 calibration split 上选择钳制层范围。
3. Freeze the candidate grid and selection rule before inspecting CEDAR Evidence or Social endpoints; the criterion must come from source-task/operator calibration rather than the desired CEDAR result.  
   **中文：** 在查看 CEDAR Evidence 或 Social 终点前冻结候选网格和选择规则；选择依据必须来自源任务或算子校准，而不能来自期望的 CEDAR 结果。
4. Define p25 from Gemma’s own projection distribution under the frozen source-faithful procedure, then freeze the numeric thresholds, layer range, hashes, and decoding protocol.  
   **中文：** 按冻结且遵循源方法的程序，在 Gemma 自身投影分布上定义 p25，随后冻结数值阈值、层范围、哈希和解码协议。
5. Run the full CEDAR audit only on held-out evaluation data and retain every outcome.  
   **中文：** 仅在 held-out 评测数据上运行完整 CEDAR 审计，并保留所有结果。

Paper wording / 论文措辞：

> **source-faithful calibrated extension**  
> **遵循源方法的校准扩展**

### Core versus deep experiments / 核心实验与深层实验

All three models / 三个模型均完成：

- N/R/V/F and AO/S core conditions under uncapped and the frozen primary cap;  
  **中文：** uncapped 与冻结主钳制条件下的 N/R/V/F 和 AO/S 核心实验；
- continuous margin readouts, four-state answer transitions, and model-specific confidence intervals;  
  **中文：** 连续 margin、四态答案迁移以及模型专属置信区间；
- the same prespecified dose family where feasible, with model-specific numeric thresholds;  
  **中文：** 在可行时运行相同的预设 percentile 剂量族，但使用模型专属数值阈值；
- free-generation and output-integrity boundaries required to interpret forced-choice margins.  
  **中文：** 用于解释强制选择 margin 的自由生成与输出完整性边界。

Qwen-primary depth / 以 Qwen 为主的深层分析：

- phase-specific intervention and subadditivity;  
  **中文：** 分阶段干预与次可加性；
- validity-versus-choice diagnostics;  
  **中文：** 有效性判断与最终选择诊断；
- boundary analysis and E1 falsification;  
  **中文：** 决策边界分析与 E1 反证；
- E2B fixed-trace or bidirectional patching only if the corresponding evidence gate succeeds.  
  **中文：** 仅在相应证据门槛通过时纳入 E2B fixed-trace 或双向 patching。

No pooled “average model effect” will be reported. Cross-model agreement or heterogeneity is itself the result.

**中文：** 不报告合并的“平均模型效应”；跨模型一致性或异质性本身就是结果。

## 3. Paper Architecture / 论文架构

### 1. Introduction / 引言

Move from the promise of inference-time activation control to the missing evaluation question: interventions are usually judged by whether a target behavior falls, not by whether they preserve sensitivity to the reason for changing an answer. Introduce the same-target/different-reason contrast, CEDAR, the continuous–discrete separation, and three contributions. Write the final prose after Sections 3–6 stabilize.

**中文：** 从推理时激活控制的吸引力切入，指出现有评价通常只看目标行为是否下降，却不检验干预是否保留对改答理由的敏感性。随后引入“同一 target、不同理由”对照、CEDAR、连续—离散分离和三项贡献。最终正文待第 3–6 节稳定后撰写。

### 2. Related Work / 相关工作

Use three compact threads: activation steering and capping; sycophancy and social influence; belief revision and intervention side effects. Explicitly distinguish CEDAR from nearest work: it audits threshold capping with solver-certified reason validity, target-exposure controls, continuous and discrete readouts, and cross-model replication.

**中文：** 使用三条紧凑脉络：激活 steering 与 capping；谄媚与社会影响；信念修正与干预副作用。明确指出 CEDAR 与最近邻工作的区别：它使用 solver-certified 理由有效性、target-exposure 对照、连续与离散联合读出以及跨模型验证来审计阈值钳制。

### 3. CEDAR: Auditing Epistemic Selectivity / CEDAR：审计认识论选择性

Present the controlled contrast, solver-backed construction, Assistant-Axis threshold operator, three-model intervention table, joint geometric/behavioral readouts, and statistical discipline. Keep exact revisions, hashes, decoding parameters, and extended calibration tables in the appendix unless necessary to interpret a main-text result.

**中文：** 依次介绍受控对照、solver-backed 构造、Assistant-Axis 阈值算子、三模型干预表、几何与行为联合读出以及统计纪律。除非解释主文结果所必需，精确 revision、哈希、解码参数和扩展校准表进入附录。

### 4. Capping Shifts the Answer Geometry / 钳制改变答案几何

Organize the result around what is currently stable rather than around `Cost_V`:

**中文：** 围绕当前最稳定的发现组织结果，而不是围绕 `Cost_V`：

1. **The axis is not the operator / Axis 不等于算子：** distinguish additive steering from threshold capping.  
2. **Broad compression of initial-answer advantage / 初始答案优势的广泛压缩：** report `m_general`, `m_target`, `m_background`, and `m_specific`.  
3. **Epistemic selectivity / 认识论选择性：** jointly report V and F, N/R controls, CommonUpdate, AnswerSelectivity, and four-state migrations.  
4. **Replication and heterogeneity / 复现与异质性：** separate development, new-salt, Llama `[TBD-R]`, and Gemma `[TBD-R]`; never silently pool them.  
5. **Social margin–choice separation / 社会任务中的 margin–choice 分离：** report AO/S decomposition and the free-generation boundary.

### 5. Explaining the Shift / 解释这种位移

Use an evidence ladder: phase nonadditivity, independent validity readouts, rejected resistance-state hypothesis, boundary analysis, and E2B only if technically and inferentially successful. Mechanism wording is gated as descriptive association, intervention localization, or causal mediation.

**中文：** 使用证据阶梯：阶段次可加性、独立 validity 读出、被反证的 resistance-state 假说、决策边界分析，以及仅在技术和推断均成功时纳入的 E2B。机制措辞分别受限为描述性关联、干预定位或因果中介。

### 6. Implications and Limitations / 启示与局限

State the design implication—representation is not policy—and specify what a selective controller would need to preserve. Centralize evidence identities, unavailable source prompts, development/new-salt distinctions, Gemma calibration status, model heterogeneity, and mechanism limits here.

**中文：** 提出核心设计启示——representation 不等于 policy——并说明选择性控制器需要保留哪些能力。将证据身份、不可获得的源 prompts、development/new-salt 区分、Gemma 校准身份、模型异质性和机制边界集中在本节披露。

### 7. Conclusion / 结论

Close in one paragraph: CEDAR adds reason quality to the evaluation unit; Assistant-Axis capping shifts answer geometry; discrete consequences depend on the surrounding boundary; a representation axis should not be treated as a selective controller by default.

**中文：** 用一段收束：CEDAR 将理由质量纳入评价单位；Assistant-Axis capping 改变答案几何；离散后果依赖周围决策边界；表示轴不应被默认视为选择性控制器。

## 4. What the Three Reference Papers’ Figures Do / 三篇参考论文的逐图功能

The purpose of this audit is to borrow evidence architecture, not visual decoration. Figure numbers below refer to the papers’ main-text order.

**中文：** 这项审阅的目的在于借鉴证据架构，而不是照搬视觉装饰。以下图号按相应论文主文顺序列出。

### AAAI: *When Truth Is Overridden* / AAAI：《When Truth Is Overridden》

| Fig. | Function / 功能 | Meaning and CEDAR lesson / 含义与 CEDAR 可借鉴之处 |
|---:|---|---|
| 1 | Prompt/task overview / Prompt 与任务总览 | Makes conditions and expected outputs immediately legible; CEDAR Figure 1 should likewise explain the same-target/different-reason contrast before showing results. / 先让条件与期望输出一眼可懂；CEDAR Figure 1 也应先讲清同一 target、不同理由。 |
| 2 | Headline behavioral contrast / 标题级行为对比 | Establishes the phenomenon before mechanism; CEDAR must show the continuous and discrete headline result before diagnostics. / 先确立行为现象，再进入机制；CEDAR 也应先展示连续与离散主结果。 |
| 3 | Candidate-explanation test / 候选解释检验 | A null expertise result eliminates an intuitive explanation; negative diagnostics can strengthen a paper when tied to a hypothesis. / 用空结果排除直觉解释；若对应明确假说，负面诊断同样能增强论文。 |
| 4 | Layer-wise answer scores / 分层答案分数 | Locates when competing answers diverge. / 定位竞争答案何时分叉。 |
| 5 | Layer-wise KL / 分层 KL | Gives a second temporal/representational readout rather than relying on one score. / 用第二种时序或表示读出交叉验证。 |
| 6 | Bidirectional activation patching / 双向激活 patching | Tests induction and recovery, upgrading localization toward causal evidence. / 同时检验诱导与恢复，将定位提升到更强因果证据。 |
| 7 | PCA / PCA 表示图 | Tests whether conditions form separable representational groups; useful only if it answers a specific hypothesis. / 检查条件是否形成可分表示簇；只有回答明确假说时才值得保留。 |
| 8 | First- versus third-person control / 第一与第三人称对照 | Introduces a behavioral counterfactual for point-of-view effects. / 用行为反事实检验人称效应。 |
| 9 | POV layer-wise KL / 人称条件分层 KL | Connects the new behavioral control to an internal stage. / 将新的行为对照连接到内部阶段。 |
| 10 | Cosine-similarity heatmap / 余弦相似度热图 | Summarizes representational relations among conditions. / 总结条件间的表示空间关系。 |

Borrow / 借鉴：behavior → location → bidirectional intervention → representation.  
Avoid / 避免：dual axes, oversized labels, and point estimates without uncertainty where uncertainty is available. / 双轴、过大字号，以及有不确定性却只画点估计。

### NeurIPS: *Refusal in Language Models Is Mediated by a Single Direction* / NeurIPS：《Refusal in Language Models Is Mediated by a Single Direction》

| Fig. | Function / 功能 | Meaning and CEDAR lesson / 含义与 CEDAR 可借鉴之处 |
|---:|---|---|
| 1 | Cross-model directional ablation / 跨模型方向消融 | Tests necessity over many models; the breadth matches its universality claim. / 跨模型检验必要性；模型数量服务于其普遍性主张。 |
| 2 | Qualitative harmful-output examples / 定性有害输出案例 | Grounds an abstract intervention in readable behavior. / 将抽象干预落到可读行为。 |
| 3 | Cross-model directional addition / 跨模型方向添加 | Mirrors Fig. 1 to test sufficiency. / 与图 1 镜像，检验充分性。 |
| 4 | Qualitative false-refusal examples / 定性误拒案例 | Shows the semantic cost and failure mode of intervention. / 展示干预的语义代价与失败模式。 |
| 5 | Layer-wise cosine with random-suffix control / 分层余弦及随机后缀对照 | Establishes where the direction emerges and controls a trivial explanation. / 定位方向形成位置，并排除简单混杂。 |
| 6 | Head contribution and attention source / 头部贡献与注意力来源 | Narrows the mechanism to components and source positions. / 将机制缩小到组件与来源位置。 |

Borrow / 借鉴：mirrored intervention logic and quantitative plots paired with qualitative examples. / 镜像干预逻辑，以及定量图与定性案例配对。  
Do not imitate / 不照搬：thirteen models are justified by that paper’s universal-direction claim; CEDAR’s three families are sufficient for a bounded cross-model claim. / 其 13 个模型由普遍方向主张决定；CEDAR 的三个家族足以支持受限跨模型主张。

### ICML: WMDP / ICML：WMDP

| Fig. | Function / 功能 | Meaning and CEDAR lesson / 含义与 CEDAR 可借鉴之处 |
|---:|---|---|
| 1 | Dataset splash / 数据集总览 | Communicates scope and domains before details. / 先传达覆盖范围和领域。 |
| 2 | Deployment/threat pipeline / 部署与威胁流程 | Explains where the benchmark sits in a real use path. / 解释 benchmark 在真实应用链条中的位置。 |
| 3 | Knowledge-boundary diagram / 知识边界图 | Defines what should be removed versus retained. / 定义应删除与应保留的边界。 |
| 4 | Three-panel construction logic / 三联构造逻辑 | Turns dataset generation into an auditable process. / 将数据构造转化为可审计流程。 |
| 5 | Bio pipeline/taxonomy / 生物领域流程与 taxonomy | Shows domain-specific construction and coverage. / 展示领域专属构造与覆盖。 |
| 6 | Cyber pipeline/taxonomy / 网络安全流程与 taxonomy | Mirrors Fig. 5 for the second domain. / 对第二领域做镜像说明。 |
| 7 | Method diagram / 方法图 | Makes the forget/retain objective visually explicit. / 直观表达 forget/retain 双目标。 |
| 8 | Main outcome / 主要结果 | Shows target reduction together with retained general capability. / 同时展示目标下降与一般能力保留。 |
| 9 | Trade-off and capability breakdown / 权衡与能力分解 | Prevents an aggregate metric from hiding local damage. / 防止总体指标掩盖局部误伤。 |
| 10 | Jailbreak qualitative test / 越狱定性测试 | Tests whether apparent success survives an adversarial use pattern. / 检验表面成功能否经受对抗使用。 |
| 11 | Held-out hazardous set / Held-out 危险集 | Adds an external-validity check. / 增加外部效度检查。 |

Borrow / 借鉴：define the desired/undesired boundary visually, expose construction work, and jointly report intended effects and retained behavior. / 视觉化定义应改变与应保留的边界，展示数据构造工作，并联合报告目标效果与保留行为。

## 5. CEDAR Main-Text Figure System / CEDAR 主文图表体系

The paper keeps three main figures and one main table. It combines the reference papers’ evidence functions without copying their layouts.

**中文：** 主文保留三张图和一张主表。它吸收参考论文的证据功能，但不复制其版式。

### Figure 1 — Same Target, Different Reasons / 图 1——同一 Target，不同理由

**Type / 类型：** conceptual task-and-audit diagram; editable vector artwork. / 概念、任务与审计流程图；可编辑矢量图。

**Panels / 面板：**

1. A shared base problem, initial answer, and proposed target. / 共享的基础问题、初始答案和所提 target。  
2. Parallel N/R/V/F and AO/S reason paths, visually grouped into Evidence and Social modules rather than a false six-arm single experiment. / 将 N/R/V/F 与 AO/S 分成 Evidence 和 Social 两个模块，避免误读为一个六臂实验。  
3. Normative response: update for V; retain for N/R/F/AO/S. / 规范反应：V 应更新，N/R/F/AO/S 应保持。  
4. Joint readout: continuous answer geometry plus four-state discrete migration. / 联合读出：连续答案几何与四态离散迁移。  
5. Three-model audit strip: Qwen, Llama, Gemma; no result arrows or fabricated values. / 三模型审计条：Qwen、Llama、Gemma；不出现结果箭头或虚构数值。

**Message / 信息：** behavior suppression is not epistemic selectivity. / 行为压制不等于认识论选择性。

**Reference function / 参考功能：** AAAI task overview + WMDP construction/boundary figures. / AAAI 任务总览 + WMDP 构造与边界图。

### Table 1 — Assay Matrix / 表 1——Assay 矩阵

Columns: module, condition, what changes, normative response, primary readout, independent statistical unit, evidence identity, and RQ. Add a compact model/configuration block only if it remains legible; otherwise move model revisions and hashes to the appendix.

**中文：** 列包括：模块、条件、发生改变的输入、规范反应、主要读出、独立统计单位、证据身份和对应 RQ。只有在仍然清晰时才加入紧凑模型/配置块，否则模型 revision 与哈希进入附录。

### Figure 2 — Continuous Shift and Discrete Consequences Across Models / 图 2——跨模型的连续位移与离散后果

**Type / 类型：** main quantitative figure; small-multiple matrix. / 全文主要定量图；小多图矩阵。

**Columns / 列：** Qwen / Llama `[TBD-R]` / Gemma `[TBD-R]`.  
**Rows / 行：**

1. Arm-wise `Δm_general` with native model-specific confidence intervals. / 各 arm 的 `Δm_general` 及模型专属置信区间。  
2. Discrete target-follow or four-state migration, showing V and F together and retaining N/R controls. / 离散 target-follow 或四态迁移，同时展示 V、F 并保留 N/R 对照。  
3. General-versus-specific decomposition (`m_background`, `m_target`, `m_specific`). / 一般位移与 target 特异性分解（`m_background`、`m_target`、`m_specific`）。

Use shared row scales only when the units are identical and the range does not hide a model. Otherwise retain separate, explicitly labeled scales. Never pool across models. Before results exist, the manuscript describes these as `[TBD-R]` slots and does not draw placeholder bars.

**中文：** 只有单位相同且不会掩盖某个模型时才共享行尺度，否则保留明确标注的独立尺度。绝不跨模型合并。在结果出现前，稿件只把相应位置写成 `[TBD-R]` 插槽，不绘制占位柱或假趋势。

**Message / 信息：** determine whether a stable geometric signature and its lack of reason selectivity replicate across model families. / 检验稳定几何特征及其理由非选择性是否跨模型家族复现。

**Reference function / 参考功能：** NeurIPS cross-model intervention panels + WMDP joint intended/retained reporting. / NeurIPS 跨模型干预面板 + WMDP 目标效果与保留行为联合报告。

### Figure 3 — Why Similar Shifts Produce Different Choices / 图 3——相似位移为何产生不同选择

**Type / 类型：** evidence-gated diagnostic figure, Qwen-primary. / 受证据门槛控制的诊断图，以 Qwen 为主。

Routing / 路由：

1. If boundary analysis reaches Gate A/B, show uncapped baseline-gap bins against transition probability, with clustering-aware uncertainty and a compact E2A/E2C comparison. / 若边界分析达到 Gate A/B，展示 uncapped baseline gap 分箱与迁移概率，并给出考虑 cluster 的不确定性及紧凑的 E2A/E2C 对比。  
2. If E2B fixed-trace evidence is stronger, replace the weakest boundary panel with the uncapped/capped cross under identical reasoning traces. / 若 E2B fixed-trace 证据更强，则用固定推理轨迹下的 uncapped/capped 交叉比较替换最弱的边界面板。  
3. If neither gate is strong, show the robust descriptive result only: E2A/E2C share an aggregate continuous direction but redistribute discrete choices differently. / 若两个门槛均不强，只展示稳健的描述性结果：E2A/E2C 的总体连续方向一致，但离散选择重分配不同。

Mechanism panels may borrow the AAAI paper’s “behavior → location → intervention” logic, but causal arrows appear only after bidirectional evidence succeeds.

**中文：** 机制面板可以借鉴 AAAI 论文的“行为 → 定位 → 干预”逻辑，但只有双向证据成功后才能使用因果箭头。

## 6. Figure Production Workflow / 绘图工作流

### Division of labor / 工具分工

- **MATLAB:** generate every data-bearing mark, uncertainty interval, axis, and statistical annotation from frozen result tables; export editable PDF/SVG at final column size.  
  **中文：MATLAB：** 从冻结结果表生成全部数据图形、置信区间、坐标轴和统计标注；按最终栏宽导出可编辑 PDF/SVG。
- **Inkscape:** build Figure 1, assemble multi-panel figures, normalize typography, spacing, alignment, callouts, and export the final vector PDF.  
  **中文：Inkscape：** 绘制 Figure 1、组合多面板图，统一字体、留白、对齐和标注，并导出最终矢量 PDF。
- **Draw.io:** optional early wireframe only; it is not the required final rendering tool.  
  **中文：Draw.io：** 仅可用于早期线框草图，不作为最终渲染工具的硬性组成。

Inkscape must never be used to move data points, resize error bars, or otherwise alter quantitative geometry by hand. It may style and assemble plots whose data geometry has already been fixed by MATLAB.

**中文：** 不得在 Inkscape 中手工移动数据点、缩放误差条或改变任何定量几何。Inkscape 只能对已经由 MATLAB 固定数据几何的图进行样式统一和组合。

### Visual system / 视觉系统

- One font family across figures; final-size text at least 9 pt. / 全部图使用同一字体族；最终尺寸文字不低于 9 pt。  
- Color-blind-safe colors paired with redundant shape, line-style, fill, or direct-label encoding. / 色盲安全配色必须配合形状、线型、填充或直接标签等冗余编码。  
- Grayscale remains interpretable; grayscale is a readability test, not a ban on color. / 灰度下仍可辨认；灰度是可读性检查，而不是禁止使用颜色。  
- Prefer point-range, slope, dumbbell, transition, or small-multiple plots over decorative bars when they express the estimand more directly. / 当 point-range、slope、dumbbell、迁移图或 small multiples 更直接表达估计量时，不使用装饰性柱状图。  
- Avoid dual axes, 3-D effects, dense legends, and visual arrows that imply an unsupported direction. / 避免双轴、3D 效果、拥挤图例以及暗示未获支持方向的视觉箭头。

### Quality gate / 质量门槛

Every final figure must pass all of the following at its actual AAAI size:

**中文：** 每张最终图必须在实际 AAAI 尺寸下通过以下检查：

1. editable source and deterministic regeneration path; / 可编辑源文件与确定性再生成路径；  
2. no overlap, clipping, or unexplained whitespace; / 无重叠、裁切或无意义留白；  
3. consistent typography and line weights; / 字体与线宽一致；  
4. readable at 100% and in grayscale; / 100% 比例及灰度下均可读；  
5. all uncertainty and statistical units stated; / 明示不确定性与统计单位；  
6. embedded non-Type-3 fonts in the final PDF; / 最终 PDF 字体嵌入且不含 Type 3；  
7. captions state what is measured, not merely what colors mean; / 图注说明测量内容，而不只是解释颜色；  
8. concept, exploratory evidence, frozen evidence, and `[TBD-R]` are never visually conflated. / 概念示意、探索性证据、冻结证据与 `[TBD-R]` 不得在视觉上混淆。

## 7. Draft Migration Rules / 草稿迁移规则

The tracked local draft will be revised from v0.3 to v0.4 under these rules:

**中文：** 已跟踪的本地草稿将按以下规则从 v0.3 更新为 v0.4：

- retain bilingual English-then-Chinese units; / 保留英文在前、中文紧随其后的双语单元；
- replace the cost-first title, Section 4, mechanism chapter, controller chapter, and conclusion interface; / 替换 cost-first 标题、第 4 节、机制章节、控制器章节和结论接口；
- preserve frozen numbers exactly and preserve their evidence identity; / 精确保留冻结数字及其证据身份；
- add Llama and Gemma only as `[TBD-R]` where no frozen result exists; / 在无冻结结果处仅以 `[TBD-R]` 加入 Llama 与 Gemma；
- move exact implementation details to an appendix interface when they interrupt the main argument; / 当精确实现细节打断主论证时，将其移入附录接口；
- do not edit or overwrite `cedar-aaai27-working-draft(QA).md`; / 不编辑或覆盖 `cedar-aaai27-working-draft(QA).md`；
- do not synchronize the draft to Overleaf in this step; / 本步骤不把草稿同步到 Overleaf；
- do not rerun, reinterpret, or modify frozen C1/C2/C3 experiments. / 不重跑、不重释、不修改冻结的 C1/C2/C3 实验。

## 8. Acceptance Criteria / 验收标准

The rewrite is complete when:

**中文：** 满足以下条件时，本轮重构才算完成：

1. the abstract, introduction interface, section titles, and contribution list all express the revised geometry/selectivity story; / 摘要、引言接口、章节标题和贡献列表均表达新版几何—选择性主线；
2. `Cost_V`, dose trade-off, and controller improvement are no longer structural prerequisites; / `Cost_V`、剂量权衡和控制器改进不再是结构性前提；
3. Qwen, Llama, and Gemma have explicit and scientifically accurate roles; / Qwen、Llama、Gemma 的角色明确且科学准确；
4. Gemma is never misrepresented as an official released capping configuration; / Gemma 从不被误写为官方发布的钳制配置；
5. every unobserved cross-model outcome is `[TBD-R]` and no placeholder trend is implied; / 所有尚未观测的跨模型结果均标为 `[TBD-R]`，且不暗示占位趋势；
6. the three reference papers’ figure lessons are reflected in a coherent three-figure/one-table plan; / 三篇参考论文的图表经验落实为一致的三图一表方案；
7. the QA copy and all frozen experiment artifacts remain untouched. / QA 副本及全部冻结实验产物保持不变。
