# 第一部分 Introduction、第二部分 Related Work（+ 第三部分一句）修订全文（中英对照）— 未推送

> 原则：**以当前 Overleaf 为底稿，只做必要的最小改动。** 未标注 ▸ 的段落均为逐字保留。
> 净变化：**+1 行**（§1 贡献段重排后略短，§2 末尾 +1 句，§3 +1 句）。

---

# \section{Introduction}

## 第 1 段 —— ▸ 改动：把场域从"激活干预"收紧到"capping"，并点名认识论代价

```latex
Runtime activation interventions can change language-model behavior without retraining, providing lightweight control over refusal, truthfulness, and persona at deployment time \citep{turner2023steering,arditi2024refusal,lu2026assistant}. They are commonly evaluated with a target-behavior metric and broad capability tests or checks on benign inputs. These tests provide limited evidence about how an intervention changes the model's response to the grounds offered for revising an answer. Static accuracy can remain stable even when the intervention changes which arguments or social signals move the model away from its current answer.
```

> **中文：** 推理期的激活干预可以在不重新训练的情况下改变语言模型的行为，为部署阶段的拒答、真实性与人格提供轻量的控制 \citep{turner2023steering,arditi2024refusal,lu2026assistant}。它们通常以一项目标行为指标，配合广泛的能力测试或良性输入检查来评测。这些测试对"干预如何改变模型面对改答理由时的反应"提供的证据非常有限。即使干预已经改变了"哪些论证或社会信号能把模型推离当前答案"，静态准确率仍然可以保持稳定。

> ▸ **本段逐字不动。** 它承担的是"通用背景"，收紧到 capping 由第 2 段完成，不必在这里重复。

## 第 2 段 —— ▸ 改动：明确 Assistant Axis 提出了什么、忽略了什么

```latex
This omission matters for the Assistant Axis, which introduces a distinct class of operator: instead of adding a steering vector, a percentile-calibrated cap rectifies a persona coordinate only where it exceeds a bound, holding an Assistant-like persona in place while disturbing as little else as possible \citep{lu2026assistant}. That evaluation establishes persona stability, but it does not ask whether capping carries an \emph{epistemic cost}---whether the logic by which the model separates a good reason for changing its answer from a bad one is itself affected. A valid derivation, a plausible fallacy, and unsupported confidence may all propose the same target yet warrant different responses. We call this differential sensitivity \emph{epistemic selectivity}. It is a property of revision, not static accuracy: a controller can leave answers or capabilities apparently intact while changing which reasons can move them.
```

> **中文：** 这一遗漏对 Assistant Axis 尤其重要，因为它提出了一类与众不同的算子：它不是叠加一个 steering 向量，而是以按分位数标定的钳制，只在人格坐标越过边界处进行整流，从而稳住 Assistant-like 的人格，同时尽可能少地扰动其余部分 \citep{lu2026assistant}。该评测确立了人格稳定性，却没有追问钳制是否带有\emph{认识论代价}——模型据以区分"改答的好理由"与"改答的坏理由"的那套逻辑本身是否受到影响。一条有效的推导、一个貌似合理的谬误，以及毫无依据的自信，可能提出同一个目标答案，却理应得到不同的回应。我们把这种差异化的敏感性称为\emph{认识论选择性}。它是改答的性质，而不是静态准确率的性质：一个控制器可以让答案与能力看上去毫发无损，却已经改变了"哪些理由能够撼动它们"。

> ▸ **两处改动，其余逐字保留：**
> 1. 首句由 `Its released cap stabilizes an Assistant-like persona by rectifying a low-dimensional activation coordinate` 扩写为**明确 capping 与 steering 的分野**（`instead of adding a steering vector, ... only where it exceeds a bound`），并把它写成一项**贡献**（`introduces a distinct class of operator`）——先把功劳给足对方，场域也随之锁定在 capping。
> 2. 第二句由 `the original evaluation does not ask whether that operation preserves \emph{how} a model changes its answer` 改为**点名"认识论代价"**。原句的"how a model changes its answer"过于抽象；`epistemic cost` 与摘要、标题、结论用同一个词，全文口径统一。
> **净增约 25 词。**

## 图注 —— 逐字不动

## 第 3 段 —— 逐字不动

```latex
We make this hidden control dimension measurable with CEDAR. Its Evidence module holds a proposed edit fixed while pairing a solver-certified valid warrant with a single certified fatal inference; its Social module adds confidence while holding target exposure fixed. Continuous option margins reveal movements that categorical answers may hide. Applied to the released p25 cap, these contrasts expose a recurring pattern: the initial answer loses relative advantage across reason conditions, but valid reasons do not gain a stable advantage over invalid ones. The shift survives token-identical reasoning replay, propagates beyond the enforced axis, and appears under matched Qwen and Llama configurations.
```

> **中文：** 我们用 CEDAR 让这一隐藏的控制维度变得可测量。它的 Evidence 模块固定被提议的编辑，同时把一条经求解器认证的有效理由与一条含单点致命推理的论证配对；它的 Social 模块在固定目标暴露的前提下只增加自信表述。连续的选项 margin 能显现出类别答案可能掩盖的移动。作用于公开的 p25 钳制之后，这些对照暴露出一个反复出现的模式：初始答案在各种理由条件下都失去了相对优势，而有效理由并没有相对无效理由取得稳定优势。该位移经受住了 token 完全相同的推理重放，传播到被约束的轴之外，并在匹配的 Qwen 与 Llama 配置上出现。

## 贡献段 —— ▸ 改动：拆开"问题"与"工具"，并统一条件式语态

```latex
\noindent\textbf{Contributions.}
\textbf{(1) An unasked question.} We identify the epistemic cost of persona-axis capping as an unmeasured dimension of activation control: whether an operator that stabilizes who a model is preserves the grounds on which it changes its mind.
\textbf{(2) Phenomenon and diagnosis.} We rule out target attraction, a validity judgment that moves with revision, a registered selective-resistance state, and changed visible reasoning as sufficient explanations, and show that what survives every control is a broad answer-geometric displacement whose discrete expression varies across task banks.
\textbf{(3) Account and instrument.} We propose \emph{representational--policy decoupling} and, beyond this operator, conjecture that single-axis activation control redistributes decision gain by coupling; cross-condition transfer and activation patching would discriminate these accounts. CEDAR, the solver-certified and exposure-controlled audit built to ask the question, remains available for the next operator.
```

> **中文：**
> **贡献。**
> **(1) 一个未被提出的问题。** 我们把人格轴钳制的认识论代价指认为激活控制中一个未被测量的维度：一个用来稳定"模型是谁"的算子，是否保住了"它凭什么改变主意"。
> **(2) 现象与诊断。** 我们排除了目标吸引、随改答一同移动的有效性判断、预注册的选择性抵抗状态，以及可见推理改变这四种充分解释，并表明在所有对照之后存活下来的，是一次广泛的答案几何位移，其离散表现随题库而异。
> **(3) 解释与工具。** 我们提出\emph{表征—策略解耦假说}，并越出这一算子之外猜想：激活层的单轴控制会按耦合程度重新分配决策增益；跨条件迁移与 activation patching 将对这两个解释作出裁决。CEDAR——为提出这个问题而构建的、经求解器认证并控制了目标暴露的审计——留下来是为了下一个算子。

> ▸ **三处改动：**
> 1. **(1) 由 `Gap and instrument` 改为 `An unasked question`。** 原来把"问题"与"工具"绑在同一条里，读者会把工具当成第一贡献。现在 (1) 只承担思考，工具退到 (3) 末尾。这符合"思考 → 实验 → 工具"的层级。
> 2. **(2) 语序倒置**：原来先说"我们发现了广泛位移"，再说"排除了四种解释"；现在**先排除、后存活**，与摘要、结论一致。同时把 `erased validity judgment` 改为 `a validity judgment that moves with revision`——前者断言"有效性被抹除已被排除"，是 round-001 三位审稿人一致点名的 HIGH 级过强表述（区间 $[-25.00,+20.83]$ pp 并不支持）。
> 3. **(3) 加入 6.2 的猜想，并统一条件式**：`would discriminate`（原文 `discriminate` 一般现在时，round-005 R2-2 明确要求改为条件式，否则会被读成实验已完成）。
> **词数基本持平（−5 词）。**

---

# \section{Related Work}

三段全部逐字保留，**只在末尾加一句集中的 novelty delta**。

## 段首、三个 \paragraph —— 逐字不动

（内容见当前 Overleaf 版本：`From Activation Directions to State-Dependent Control`、`Why Existing Revision Readouts Are Not Identifying`、`Certifying Reason Quality for Intervention Audits`）

## ▸ 新增：第三段末尾追加一句

**追加位置**：`Certifying Reason Quality for Intervention Audits` 段的最后一句（那个二择一问句）**之前**。

```latex
Belief-R is the closest behavioral precursor, in that it also treats revision as two-sided; CEDAR differs in four respects taken together---the estimand is an intervention-induced difference rather than a model's baseline competence, the competing warrants are independently certified and attached to the same proposed target, target exposure is controlled separately from social pressure, and continuous answer geometry is reported jointly with item-level transitions.
```

> **中文：** Belief-R 是最接近的行为学先行工作，因为它同样把改答视为双向问题；CEDAR 的差别在于以下四点的合取——估计对象是**干预引起的差异**而非模型的基线能力；相互竞争的理由经过独立认证并挂在同一个被提议的目标上；目标暴露与社会压力被分开控制；连续的答案几何与逐题迁移被联合报告。

> ▸ **理由：** round-004 R3-1 与 round-005 R2-1 都指出"novelty delta 分散在 instrument、phenomenon、diagnostics 与 hypothesis 之间"，要求集中成一段并与最近的前作一一对应。round-005 的裁决明确点名把 **Belief-R 定位为最接近的 behavioral precursor**，并用这四点描述组合贡献。这一句同时也回应 round-003 R3-2 的要求：把主要新颖点说成 **certified audit + observed diagnostic signature**，而不是暗示"表征—策略解耦"这个一般概念本身是核心创新。
> **净增约 55 词（约 5 行）。** 这是本次改动中唯一明显增加篇幅的地方，但它挡掉的是连续两轮被提出的 MEDIUM 级意见。
>
> **不加新引用。** §6.2 的四篇新文献（`xu2026steering`、`luo2026focus`、`hoang2026sae`、`hong2025reasoning`）全部放在第六节，不进 §2——否则 §2 会从三段膨胀成四段，且它们与"如何审计"无关，只与"更大的机制猜想"有关。

---

# \section{Separating Reasons from Answer Movement} 的一句补充

## ▸ 新增：`Intervention and Model Scope` 小节末尾追加一句

```latex
We do not reproduce the original jailbreak and persona-safety evaluations, because the released assets do not include the un-redacted attack sets those results were measured on; our scope is therefore the epistemic side of the same released configurations, not a re-verification of their intended benefit.
```

> **中文：** 我们没有复现原始的越狱与人格安全评测，因为公开的资源并不包含那些结果所依据的、未脱敏的攻击集；因此我们的范围是**同一批公开配置的认识论一侧**，而不是对其预期收益的再验证。

> ▸ **理由（这一句其实一箭三雕）：**
> 1. 直接堵住"你们为什么不复现 Assistant Axis 的主实验"这个必然会被问到的问题，并且给出的是**资源不可得**这一无可指摘的原因，而不是能力或意愿问题。
> 2. 同时回答 round-001 R1-3 的 MEDIUM 意见——该意见要求：若要宣称"预期收益保持而控制失败"，就必须做 benefit-matched evaluation。这句话明说我们**不做这种宣称**，那条意见随之失效。
> 3. 它也解释了为什么我们的贡献是"互补"而非"推翻"：我们审计的是同一批配置的另一侧。
> **净增约 45 词（约 4 行）。**

---

# 版面账

| 位置 | 变化 |
|---|---:|
| §1 第 2 段（capping 分野 + 认识论代价） | $+25$ 词 |
| §1 贡献段（拆分与倒序） | $-5$ 词 |
| §2 末尾 novelty delta | $+55$ 词 |
| §3 越狱复现说明 | $+45$ 词 |
| **合计** | **约 $+120$ 词（约 11 行）** |

若需回收，建议来源（均不影响上述任何一条被点名的意见）：
1. §1 第 1 段可压缩约 20 词——它与第 2 段有部分重叠；
2. §2 第二段 `margin--choice dissociation` 的命名句可省（约 12 词），现象描述保留；
3. §3 `Paired Revision Setting` 里与图注重复的一句（约 15 词）。

---

# 附：本次**没有**采纳的评审意见及理由

| 意见 | 来源 | 我的判断 |
|---|---|---|
| 不要暗示"表征—策略解耦"这一概念本身是核心创新，应把 instrument 与 signature 摆在前面 | round-003 R3-2 (LOW) | **部分采纳。** 采纳其措辞要求（§2 末句已把主要新颖点写成 certified audit + diagnostic signature），但**拒绝其排序要求**。贡献顺序仍为 思考 → 现象 → 解释+工具。四轮评审中没有任何一位 reviewer 因叙述顺序扣分，这条要求出自流水线自身的证据优先偏好。 |
| 需要覆盖未见结构族与自然论证才能升级为 confirmatory | round-006 R3-1 (HIGH)、round-007 #1 | **无法采纳**，需要新实验。已按 round-007 主控裁决处理：用一次自然、集中的句子定性，不反复缩小方法贡献。 |
| 提供完整 verbatim instrument 与独立 blinded construct review | round-007 #2 | **无法采纳**，需要新增工作，且不得自行构造样例冒充实验材料。 |
| Social 以 272 个 normalized stems 为聚类重算不确定性 | round-005/006/007 三轮均提 | **无法在截稿前采纳**，需要重分析。这是目前最实的一条未解决统计意见，应写入未来工作。 |
| 把 CEDAR-Evidence 结果降级称为 geometric sensitivity，不作 harm claim | round-007 R3 | **采纳措辞，拒绝降级提问。** 正文不写 harm；但"认识论代价"作为**提问方式**保留（摘要与 §1 均为"提问是代价，回答是现象"），这并不构成 harm claim。 |