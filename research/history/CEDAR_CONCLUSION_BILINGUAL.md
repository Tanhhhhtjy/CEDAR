# Conclusion：两轮 AI 迭代提了什么要求、改了什么（中英对照）

---

## 一、两轮迭代对 Conclusion 提出的**具体要求**

### evolution（第 1 轮迭代，round-001 → 004）

| 来源 | 要求 |
|---|---|
| round-001 worklog | 摘要、贡献、诊断、hypothesis 与 conclusion **必须使用同一条 claim ladder**（现象先确立、机制后假设）。 |
| round-001 worklog | Conclusion 的顺序被规定为：**先**总结完整 matched-model 结果与各诊断的共同方向，**再**提出 decoupling 的直接判别测试，**最后**落到 persona stability 与 epistemically selective revision 两个评估目标。 |
| round-002 worklog | conclusion 里关于 off-axis 的措辞**全部改成** downstream displacement **composition / reorientation**（因为 $G_{\mathrm{off}}$ 是占比变化，不是绝对范数增长）。 |

### evolution2（第 2 轮迭代，round-004 → 007）

| 来源 | 要求 |
|---|---|
| **round-005 问题 3（MEDIUM，Conclusion 专条）** | **"the route by which reasons change an answer" 超出了 condition-wise decodability 与 downstream composition 能支持的范围。裁决：直接修改。** 结论改为 *reason-conditioned answer geometry* 与 *downstream state diagnostics*；**shared code / causal use 留给 proposed tests。** |
| round-005 问题 2（MEDIUM） | transfer / patching 一律改成 `we formulate/propose ... as tests` 或条件式 `would test`，不得用一般现在时（会被读成已完成）。 |
| round-005 问题 4（MEDIUM） | new-salt 的 `technically inconclusive due to one missing primary pair` 必须与 complete-case 正估计**同段出现**。 |
| round-006 worklog | Conclusion 把已完成进展表述为 reason-conditioned answer geometry 与 downstream state diagnostics；transfer/patching 作为**该解释产生的可检验预测**。 |
| round-007 worklog | Abstract、Introduction、Contributions、cross-configuration Results、diagnostic table、**Conclusion** 中**统一删除** `centered / stayed near zero` 这类措辞。 |
| round-007 worklog | Abstract 与 Conclusion **同步采用 diagnostic-ladder 主线**。 |

---

## 二、两个版本的 Conclusion 原文对照

### 版本 A：当前 Overleaf（我们自己改的"大胆版"）

```latex
CEDAR reveals a hidden epistemic effect of Assistant-Axis capping. Across solver-certified reasons, target-hidden controls, social prompts, and a second model--axis pair, the most stable result was a broad movement of answer geometry rather than a reliable gain in reason selectivity. The categorical consequences varied, but the displacement survived a new salt, an independent validity judgment, and a token-identical reasoning trace. After rectification ended, it continued away from the cap's exit coordinate while validity remained readable downstream.

These results support representational--policy decoupling as the most coherent current account: capping need not erase what the model knows about a reason; it can change how that information is propagated, weighted, and read into revision. The hypothesis explains why identical visible reasoning can end in different answer geometry and why the same broad shift yields different revisions across banks. It also points toward a more selective controller---one that constrains persona drift while protecting the evidence-sensitive route from representation to decision. Stabilizing who a model appears to be does not guarantee preserving how it changes its mind.
```

> **中文：** CEDAR 揭示了 Assistant-Axis capping 的一个隐藏的认识论效应。在经求解器认证的理由、目标隐藏对照、社会提示，以及第二个模型—轴组合上，最稳定的结果是答案几何发生了广泛移动，而不是理由选择性获得了可靠提升。类别层面的后果各不相同，但这一位移经受住了新的 salt、一次独立的有效性判断，以及一条 token 完全相同的推理轨迹。在整流结束之后，它继续离开钳制的出口坐标，而有效性在下游依然可读。
>
> 这些结果支持把表征—策略解耦作为目前最连贯的解释：钳制不必抹除模型关于某条理由所知道的东西；它可以改变这些信息如何被传播、加权，并被读出为一次修订。该假说解释了为什么完全相同的可见推理会终结于不同的答案几何，以及为什么同一个广泛位移会在不同题库中产生不同的改答。它也指向一种更具选择性的控制器——它约束人格漂移，同时保护从表征通往决策的、对证据敏感的那条路径。**稳定一个模型表现出来的角色，并不保证保留它改变想法的方式。**

### 版本 B：evolution2 round-007（AI 迭代后的最新稿）

```latex
CEDAR introduces a certified audit of how activation control changes answer revision. Holding a 24-base forced-choice Evidence set fixed, p25 produced a broad answer-geometric displacement in both tested released model--axis configurations, while neither continuous $V-F$ contrast resolved a selectivity change. Exploratory generation and target-hidden conditions revealed broad movement within the development bank; same-family re-instantiation supplied formally incomplete, supportive evidence with heterogeneous arm and categorical effects. Token-identical replay retained the displacement, and downstream analysis followed the induced state difference beyond the cap's exit coordinate while $V/F$ labels remained decodable within each condition.

Together, these assays extend controller evaluation from final behavior to a diagnostic ladder spanning reason-conditioned answer geometry, token-identical replay, and downstream state composition. Representational--policy decoupling organizes their conjunction into testable predictions for transfer and patching. CEDAR makes epistemically selective revision a measurable controller objective alongside persona stability.
```

> **中文：** CEDAR 提出了一套关于"激活控制如何改变答案修订"的认证审计。在固定 24-base 强制选择 Evidence 集的条件下，p25 在两个被测试的公开模型—轴配置中都产生了广泛的答案几何位移，而两个连续 $V-F$ 对比都没有分辨出选择性的变化。探索性生成与目标隐藏条件在 development bank 内显示出广泛移动；同族重新实例化提供了形式上不完整但具支持性的证据，其分臂与类别效应各异。Token 完全相同的重放保留了该位移；下游分析把被诱导的状态差异一直追踪到钳制出口坐标之外，而 $V/F$ 标签在每种条件内部仍然可解码。
>
> 综合起来，这些实验把控制器评测从最终行为扩展到一条诊断阶梯，涵盖理由条件下的答案几何、token 完全相同的重放，以及下游状态构成。表征—策略解耦把它们的合取组织成关于迁移与 patching 的可检验预测。CEDAR 使"认识论上具选择性的修订"成为与人格稳定性并列的、可测量的控制器目标。

---

## 三、两个版本的差别在哪（这是关键）

| 维度 | A（当前 Overleaf） | B（AI 迭代最新） |
|---|---|---|
| 开场 | **发现**："揭示了一个隐藏的认识论效应" | **工具**："提出了一套认证审计" |
| new-salt 状态 | 只说"经受住了新的 salt" | 明写 `formally incomplete, supportive evidence`（满足 round-005 问题 4） |
| 解耦的地位 | "**最连贯的解释**"，并直接描述机制（信息如何被传播、加权、读出） | "把它们的合取**组织成可检验预测**"（满足 round-005 问题 3） |
| 收尾 | 一句可被引用的话 | 一句方法学定位 |
| 风险 | **违反 round-005 问题 3**：`how that information is propagated, weighted, and read into revision` 正是被判定"超出证据"的那句话 | 无 claim 风险，但**平淡**，把论文压回"我们做了个评测"的档次 |

**一句话：A 的气质对，claim 纪律不合格；B 的 claim 纪律合格，气质不对。**

---

## 四、我建议的合并稿（A 的叙事 + B 的纪律 + 新的 6.2）

三段结构：**已确立的现象 → 两级递进的假说 → 那句收尾**。

```latex
CEDAR turns a hidden dimension of activation control into a measurable one. Across solver-certified reasons, target-hidden controls, social prompts, and a second released model--axis pair, the most stable result was a broad movement of answer geometry, while no continuous $V-F$ contrast resolved a change in reason selectivity. The categorical consequences varied with the bank; a same-family re-instantiation supplied supportive but formally incomplete evidence. The displacement survived an independent validity judgment and a token-identical reasoning trace, and after rectification ended the induced state difference continued away from the cap's exit coordinate while $V/F$ labels remained decodable within each condition.

We read this conjunction through two nested accounts. Representational--policy decoupling organizes it into testable predictions: an intervention may preserve validity-related information while altering the computation that turns it into a revision, which cross-condition transfer and activation patching would decide. Beyond this operator, we conjecture that a percentile cap is one instance of single-axis activation control, and that such control redistributes decision gain across downstream pathways in proportion to their coupling with the intervened axis, accumulating with exposure into a persistent policy bias. Neither account is established here; both are stated so that they can be broken.

What the audit already shows is narrower and firmer. A direction can be informative, an operator can move the state it names, and the deployed policy can still lose the distinctions that decide when an answer should change. Stabilizing who a model appears to be does not guarantee preserving how it changes its mind.
```

> **中文：** CEDAR 把激活控制中一个隐藏的维度变成了可测量的维度。在经求解器认证的理由、目标隐藏对照、社会提示，以及第二个公开的模型—轴组合上，最稳定的结果是答案几何发生了广泛移动，而没有任何一个连续 $V-F$ 对比分辨出理由选择性的变化。类别层面的后果随题库而异；同族的重新实例化提供了具支持性但形式上不完整的证据。这一位移经受住了一次独立的有效性判断与一条 token 完全相同的推理轨迹；而在整流结束之后，被诱导的状态差异继续离开钳制的出口坐标，同时 $V/F$ 标签在每种条件内部仍然可解码。
>
> 我们用两个嵌套的解释来读这一组合。表征—策略解耦把它组织成可检验的预测：一次干预可能在保留有效性相关信息的同时，改变把这些信息转换成一次修订的计算——跨条件迁移与 activation patching 将对此作出裁决。而越出这个算子之外，我们猜想：分位数钳制只是**激活层单轴控制**的一个实例，而这类控制会按各下游通路与被干预轴的耦合程度重新分配决策增益，并随暴露累积成一种持续的策略偏置。两个解释在本文中都未被确立；我们把它们写出来，正是为了让它们能够被推翻。
>
> 审计本身已经给出的东西更窄，也更硬：一个方向可以是有信息的，一个算子可以移动它所命名的状态，而部署后的策略仍然可能失去那些决定"何时应当改答"的区分。**稳定一个模型表现出来的角色，并不保证保留它改变想法的方式。**

### 这一稿如何逐条满足两轮迭代的要求

| 要求 | 满足方式 |
|---|---|
| round-005 #3：不得断言"理由改变答案的路径" | 改成 `may preserve ... while altering the computation`，并且立刻跟上 `which cross-condition transfer and activation patching would decide`；末段明写 `Neither account is established here` |
| round-005 #2：transfer/patching 用条件式 | `would decide`（条件式） |
| round-005 #4：new-salt 形式状态入正文 | `supportive but formally incomplete evidence` |
| round-007：删除 `centered / near zero` | 改为 `no continuous $V-F$ contrast resolved a change`（描述分辨力，不描述位置） |
| round-002：off-axis 用 composition 措辞 | `the induced state difference continued away from the cap's exit coordinate`，不声称绝对范数增长 |
| round-001：claim ladder 与全文一致 | 三段正好对应第五节（现象）→ 6.1 → 6.2 → 收尾 |
| round-006：decoupling 组织成可检验预测 | 第二段首句即为此 |

## 五、v2：按"思考 → 实验 → 工具"层级重排（**采纳你的原则**）

我完全同意这个原则，并且认为它比 AI 评审的排序更接近科研的真实结构。补充一点做法上的分寸：**"思考优先"不等于"把未经检验的猜想放在第一句"**。第一句应该是**那个别人没有问过的问题**——问题本身就是思考，而且它无法被质疑证据不足。工具则一句带过，且永远不做主语。

据此，结论按五拍展开：**问题（思考） → 这个问题逼我们去做什么、以及路上被我们自己推翻了什么（实验） → 结果收敛成的两级解释（思考的产物） → 工具（一句，从属地位） → 收尾**。

```latex
Persona control has been certified by persona outcomes. The question we asked is what such an operator does to a property no persona metric reports: whether a model still distinguishes a good reason for changing its answer from a bad one. Making that question answerable required holding a proposed revision fixed while varying only the warrant behind it, and certifying that warrant outside the model being audited.

The answer was not the one we first expected. Across solver-certified reasons, target-hidden controls, social prompts, and a second released model--axis pair, the most stable result was a broad movement of answer geometry, while no continuous $V-F$ contrast resolved a change in reason selectivity; categorical consequences varied with the bank, and a same-family re-instantiation supplied supportive but formally incomplete evidence. We then tested the intuitive explanations and lost them one by one. An independent validity readout did not move with revision. A registered resistance-state account did not survive its own trajectory control. Freezing every visible reasoning token did not remove the displacement, and after rectification ended the induced state difference continued away from the coordinate that produced it, while $V/F$ labels remained decodable within each condition.

What survived is an account at two levels. Representational--policy decoupling holds that an intervention may preserve validity-related information while altering the computation that turns it into a revision; cross-condition transfer and activation patching would decide it. Beyond this operator, we conjecture that a percentile cap is one instance of single-axis activation control, and that such control redistributes decision gain across downstream pathways in proportion to their coupling with the intervened axis, accumulating with exposure into a persistent policy bias. Neither account is established here; both are stated so that they can be broken.

CEDAR is what made this checkable, and it remains available for the next operator rather than this one. A direction can be informative, an operator can move the state it names, and the deployed policy can still lose the distinctions that decide when an answer should change. Stabilizing who a model appears to be does not guarantee preserving how it changes its mind.
```

> **中文：**
>
> 对人格的控制，一直是用人格层面的结果来认可的。我们提出的问题是：这样一个算子，对一项任何人格指标都不会报告的性质做了什么——模型是否仍然能把"改答的好理由"与"改答的坏理由"区分开。要让这个问题变得可以回答，就必须在固定被提议的修订的同时只改变它背后的理由，并且让这条理由的有效性由被审计模型之外的程序来认证。
>
> 得到的答案并不是我们最初预期的那个。在经求解器认证的理由、目标隐藏对照、社会提示，以及第二个公开的模型—轴组合上，最稳定的结果是答案几何发生了广泛移动，而没有任何一个连续 $V-F$ 对比分辨出理由选择性的变化；类别层面的后果随题库而异，同族的重新实例化提供了具支持性但形式上不完整的证据。随后我们逐一检验那些直觉性的解释，并且逐一失去了它们：独立的有效性读数没有随改答一同移动；预注册的"抵抗状态"解释没有通过它自己的轨迹对照；冻结每一个可见推理 token 也没有消除该位移；而在整流结束之后，被诱导的状态差异继续离开了产生它的那个坐标，同时 $V/F$ 标签在每种条件内部仍然可解码。
>
> 存活下来的是一个分两级的解释。表征—策略解耦主张：一次干预可能在保留有效性相关信息的同时，改变把这些信息转换成一次修订的计算；跨条件迁移与 activation patching 将对此作出裁决。而越出这个算子之外，我们猜想：分位数钳制只是激活层单轴控制的一个实例，而这类控制会按各下游通路与被干预轴的耦合程度重新分配决策增益，并随暴露累积成一种持续的策略偏置。两个解释在本文中都未被确立；我们把它们写出来，正是为了让它们能够被推翻。
>
> CEDAR 是让上述一切变得可核查的那件东西，而它留下来是为了下一个算子，而不是这一个。一个方向可以是有信息的，一个算子可以移动它所命名的状态，而部署后的策略仍然可能失去那些决定"何时应当改答"的区分。**稳定一个模型表现出来的角色，并不保证保留它改变想法的方式。**

### 层级对照

| 段 | 层级 | 内容 |
|---|---|---|
| 1 | **思考** | 别人没问过的问题 + 为什么它需要一种特定的问法（工具在此只作为该思考的**从属子句**出现，且不具名） |
| 2 | **实验** | 结果 + **我们自己推翻自己的四次反转**——这是全篇最能体现科研过程的一段 |
| 3 | **思考（升级）** | 两级递进假说，明确标注未被确立 |
| 4 | **工具** | 一句话，且立刻把它推向"下一个算子"，避免读成"我们做了个 benchmark" |
| 5 | 收尾 | 可被引用的一句 |

### 与 AI 评审要求的兼容性

| 要求 | 是否满足 |
|---|---|
| round-005 #3：不得断言"理由改变答案的路径" | ✅ `may preserve ... while altering`，并紧跟 `would decide it` |
| round-005 #2：transfer/patching 用条件式 | ✅ `would decide` |
| round-005 #4：new-salt 形式状态入正文 | ✅ `supportive but formally incomplete` |
| round-007：删除 `centered / near zero` | ✅ 改为 `no continuous $V-F$ contrast resolved a change` |
| round-002：off-axis 用 composition 措辞 | ✅ `continued away from the coordinate that produced it` |
| round-006：decoupling 作为可检验预测 | ✅ 第三段 |

**唯一被我有意违反的**，是 round-001 worklog 规定的"结论先总结完整 matched-model 结果"这条**顺序**要求。它把工具与数字排在思考之前，与你坚持的层级冲突。我认为应当违反它：那条要求出自流水线自身的排序偏好，不是任何审稿人给出的意见，四轮评审里没有任何一位 reviewer 因为叙述顺序扣过分。

**长度**：约 300 词，比当前 Overleaf 版本（205 词）多约 95 词，约 9 行。若版面不够，第二段的四次反转可压成两句（省约 35 词），但我建议宁可从别处回收——那四句正是你说的"我们真正做了科研"的证据。