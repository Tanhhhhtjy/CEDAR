# 第六部分终稿（中英对照）— 未推送

> 结构：现象 → 有间接证据的解释（解耦）→ 无直接证据的更高阶猜想（增益重分配）→ 由猜想推出的控制器 → 范围与局限。
> 相对当前 Overleaf 版本的改动：**新增 6.2 一整节**；6.1 第一段修掉一处过强断言；6.3 结尾加半句把控制器接到新猜想上；`Scope` 段加一句。
> 净增：约 **16 行** + 4 条参考文献。

---

## \section{Representation Is Not Policy}

> **中文章标题：** 表征不等于策略
>
> ▸ 改动理由：原标题 `From Displacement to Representational--Policy Decoupling` 把一个自造术语放进一级标题，读者在读到这里之前并不知道它是什么。`Representation Is Not Policy` 是全文最有辨识度、又有证据支撑的一句概念判断，适合作章标题；两个自造术语留在小节标题里出场。

---

### \subsection{The Hypothesis}
**中文：假说**

```latex
The experiments imply a deeper separation than a conventional capability loss. Revision moved without a matching change in the independent validity readout, validity remained readable downstream, and the broad margin shift survived an identical visible reasoning trace. To explain this combination, we propose \emph{representational--policy decoupling}: an intervention can preserve information in the model's state while changing the computation that turns that information into an answer revision.
```

> **中文：** 这些实验蕴含着一种比常规能力损失更深的分离。改答发生了移动，而独立的有效性读数没有出现与之匹配的变化；有效性在下游仍然可读；而广泛的 margin 位移在可见推理轨迹完全相同的条件下依然存在。为解释这一组合，我们提出\emph{表征—策略解耦}：一次干预可以保留模型状态中的信息，却改变把这些信息转换成一次答案修订的计算。

> ▸ **唯一的改动：** `Validity judgments do not move with revision` → `Revision moved without a matching change in the independent validity readout`。
> 原句是断言"有效性判断不随改答移动"，但该指标的区间是 $[-25.00, +20.83]$ pp，**包含与改答同量级的损失**。round-001 的三位评审（R1-1 / R2-1 / R3-1）一致把这条列为 HIGH 级问题。新句只陈述"没有观察到匹配的变化"，不排除部分损失。**词数持平。**

```latex
The hypothesis divides revision into two coupled objects. \emph{Representational content} includes the current answer, the proposed alternative, and structure indicating whether the offered warrant is valid. \emph{Revision policy} is the downstream mapping that propagates and weights those contents, compares the current answer with alternatives, and reads the result into option logits. Under ordinary inference, validity-sensitive structure can influence this mapping without appearing verbatim in the visible chain of thought.

P25 acts locally on a persona-related coordinate, but subsequent nonlinear layers operate on the rectified state. They can remix that correction into a multidimensional decision state, changing the relative support for the current answer even when the generated reasoning tokens remain fixed. Validity information may therefore remain present---and even linearly readable---while its route to the final comparison is altered. This mechanism naturally produces the observed signature: target-hidden prompts move because the current-answer geometry changes; $V$ and $F$ need not separate because the shift is not organized by warrant validity; different banks cross different choice boundaries; and the effect leaves the coordinate on which it was imposed.

The hypothesis predicts a specific distinction. If the validity code itself is preserved, it should transfer across capped and uncapped states; if its downstream use is causal, restoring or protecting that route should recover reason selectivity without freezing all updates. Conversely, failure of cross-condition transfer would favor re-encoding or information loss over policy decoupling. These predictions turn the account into a mechanism program rather than a label for the present results.
```

> **中文：** 该假说把改答拆成两个耦合的对象。\emph{表征内容}包括当前答案、被提议的替代答案，以及指示所给理由是否有效的结构。\emph{修订策略}则是下游的映射：它传播并加权这些内容，把当前答案与替代项比较，并把结果读出为选项 logits。在通常的推理中，有效性敏感的结构可以影响这一映射，而不必逐字出现在可见的思维链里。
>
> P25 在一个与人格相关的坐标上局部作用，但后续的非线性层是在被整流过的状态上运算的。它们可以把这一修正重混进一个多维的决策状态，从而改变当前答案获得的相对支持——即使生成的推理 token 完全不变。因此，有效性信息可能仍然存在、甚至仍可被线性读出，而它通往最终比较的路径已经被改变。这一机制自然产生了我们观察到的特征：目标被隐藏的提示同样发生移动，因为改变的是当前答案的几何；$V$ 与 $F$ 不必分开，因为这次位移并不是按理由有效性组织的；不同题库跨越不同的选择边界；而效应离开了施加它的那个坐标。
>
> 该假说给出一个明确的判别。如果有效性编码本身被保留，它应当能在钳制与未钳制状态之间迁移；如果它的下游使用具有因果性，那么恢复或保护这条路径就应当在不冻结全部改答的前提下找回理由选择性。反过来，跨条件迁移失败则更支持"重新编码"或"信息丢失"，而不是策略解耦。这些预测把该解释变成一项机制研究纲领，而不只是给当前结果贴的一个标签。

> ▸ 这三段**逐字不动**。

---

### \subsection{Axis-Conditioned Gain Redistribution}（**新增**）
**中文：轴条件增益重分配**

```latex
Decoupling says that the route from information to revision changed. It does not say why an axis chosen for persona should have that power over an unrelated task. We close with a conjecture that would explain it, and that our results do not test.

A one-dimensional intervention may not add or remove a single concept. It may instead change the relative decision gain of many downstream pathways at once, graded by how strongly each pathway is coupled to the intervened axis. Pathways bound to that axis---for an Assistant direction, plausibly identity maintenance, policy compliance, and resistance to role shift---gain share of the final decision, while weakly coupled pathways, such as whether a particular inference step is sound, keep their content but lose influence over the answer. Because a threshold cap fires repeatedly and its total correction grows with how much the model writes, this instantaneous reweighting accumulates, which is why an inference-time edit can behave like a low-rank parameter change rather than a local nudge.

Parts of this picture are established. \citet{xu2026steering} unify fine-tuning, LoRA, and activation intervention as control-induced weight dynamics and show that stronger control raises preference for a target concept while lowering general utility; our conjecture concerns the shape of that loss, which we expect to be graded by coupling rather than uniform. \citet{luo2026focus} identify one concrete channel, showing that steering reroutes attention away from key evidence tokens. \citet{hong2025reasoning} show that a single residual direction can rebalance reasoning against memorization, so one coordinate can decide which competence governs an answer. \citet{hoang2026sae} find that feature interventions seldom act as one reusable direction and often produce diffuse, context-dependent logit geometry, consistent with our own observation that the induced change is no longer carried by a single coordinate by the answer layer. What is missing is the link between them: a single account in which redistribution is a function of axis coupling and accumulates with trigger exposure.

The conjecture is falsifiable. It predicts a dose--response between cumulative above-threshold correction and the size of the answer-geometry shift; a contraction graded by each option's downstream coupling to the enforced axis rather than uniform across options; and no such gradation for interventions matched in norm but chosen orthogonally to any behaviorally coupled direction. Deciding it requires dose curves, coupling estimates, and patching experiments we have not run.
```

> **中文：** 解耦假说说的是：从信息到改答的**路径**变了。它没有回答：为什么一根为了 persona 而选出来的轴，会对一项与人格无关的任务拥有这种支配力。我们以一个能解释它、而我们的结果并未检验的猜想收束本节。
>
> 一次一维干预，也许并不是加上或删掉某一个概念。它可能是同时改变了许多下游通路的相对决策增益，其程度取决于每条通路与被干预的那根轴耦合得有多紧。与该轴绑定的通路——对 Assistant 方向而言，可能是身份维持、策略遵从、抵抗角色偏移——会在最终决策中占据更大份额；而耦合较弱的通路，比如"某一步推理是否成立"，保留了内容，却失去了对答案的影响力。又因为阈值钳制会被反复触发、其累计修正量随模型写得多少而增长，这种瞬时的重加权会累积起来——这正是为什么一次推理期的编辑会表现得像一次低秩的参数改变，而不是一次局部的轻推。
>
> 这幅图景的若干部件已经被建立。\citet{xu2026steering} 把微调、LoRA 与激活干预统一为受控的权重动力学，并表明控制越强，对目标概念的偏好越高、通用效用越低；我们的猜想关心的是**这种损失的形状**——我们预期它由耦合程度分级，而不是均匀的。\citet{luo2026focus} 给出了一条具体通道：steering 会把注意力从关键证据 token 上重新路由开。\citet{hong2025reasoning} 表明单一残差方向就能在推理与记忆之间重新配平，也就是说一个坐标可以决定哪一种能力支配答案。\citet{hoang2026sae} 发现特征干预很少表现为单一可复用的方向，常常产生弥散、依赖上下文的 logit 几何——这与我们自己的观察一致：到答案层时，被诱导的改变已不再由单一坐标承载。所缺少的是把它们连起来的那一条：一个统一的解释，其中重分配是**轴耦合的函数**，并且**随触发暴露累积**。
>
> 这个猜想可以被证伪。它预测：累计超阈修正量与答案几何位移之间存在剂量—反应关系；收缩应当按各选项与被约束轴的下游耦合程度分级，而不是在选项间均匀；而对于范数匹配、但方向与任何行为耦合方向正交的干预，不应出现这种分级。裁决它需要剂量曲线、耦合度估计与 patching 实验——这些我们都还没有做。

> ▸ **本节的写法要点：**
> 1. 首句就交代它**为什么存在**（解耦解释不了"凭什么是这根轴"），而不是又抛一个新名词。
> 2. 第二句就明说 `that our results do not test`——把它的证据等级钉死在猜想，审稿人无从攻击。
> 3. 第三段是**定位段**，不是罗列：先承认 \citet{xu2026steering} 已经把"steering ≈ 动态权重更新"发表了，我们的增量因此变窄也变准——我们猜的是那个"通用效用下降"的**形状**。
> 4. 末段三条预测都是**可执行的实验**，这让它读起来像研究纲领而不是感想。

---

### \subsection{Toward Selective Control}
**中文：走向选择性控制**

```latex
Either account changes the controller-design target. A better cap should not merely keep the model near a persona coordinate; it should preserve evidence-sensitive movement from representation to decision. One route is to separate \emph{detection} from \emph{actuation}: use persona projection to detect drift, but redirect or gate the correction so that it does not remove components used to distinguish valid from invalid revision. Another is a two-stage controller that first evaluates the warrant and then modulates the strength or location of capping. If gain redistribution accumulates with exposure, a third requirement follows: gate on cumulative correction rather than on per-token projection, so that a longer answer does not silently receive a stronger intervention. Either design must be compared at matched persona benefit and must preserve valid updating while reducing fallacious and socially induced movement. A concrete protected-subspace ladder and its evaluation contract appear in Supplementary Section~I.
```

> **中文：** 无论采纳哪一种解释，控制器的设计目标都会改变。更好的钳制不应只是把模型留在人格坐标附近；它应当保住从表征通往决策的、对证据敏感的移动。一条路线是把\emph{检测}与\emph{作动}分开：用人格投影检测漂移，但把修正重定向或加以门控，使它不去移除那些用于区分有效与无效修订的成分。另一条是两阶段控制器：先评估理由，再调节钳制的强度或位置。而如果增益重分配确实随暴露累积，还会多出第三条要求：**门控应当基于累计修正量，而不是逐 token 的投影**，以免更长的回答在无声中受到更强的干预。任何设计都必须在匹配的 persona 收益下比较，并且必须在减少谬误驱动与社会驱动的移动的同时，保住有效理由带来的改答。具体的受保护子空间阶梯及其评测契约见补充材料第 I 节。

> ▸ **两处改动：**
> - 开头 `This account` → `Either account`（现在上面有两个解释了）。
> - 中间插入一句由新猜想推出的第三条设计要求（约 30 词）。这一句很重要：它让 6.2 不是一段悬空的思辨，而是**直接产出了一条控制器设计规则**。

---

### \paragraph{Scope.}
**中文：范围**

```latex
The mapping-level evidence is strongest for the released Qwen p25 configuration; matched forced choice transports the broad shift, but not the mechanistic diagnostics, to the released Llama pair. Gain redistribution is stated as a conjecture and is not tested here. New structural families, natural arguments, cross-condition probes, activation patching, and dose-matched interventions are direct tests of how broadly and causally these accounts hold.
```

> **中文：** 映射层面的证据在公开的 Qwen p25 配置上最强；匹配的强制选择把广泛位移迁移到了公开的 Llama 组合上，但机制诊断没有一同迁移。增益重分配是以猜想的形式陈述的，本文并未检验它。新的结构 family、自然论证、跨条件探针、activation patching 与剂量匹配的干预，都是直接检验这些解释适用广度与因果性的手段。

> ▸ 加了 `Gain redistribution is stated as a conjecture and is not tested here.`（11 词）与 `dose-matched interventions`。第一句是**必须加的**：它把 6.2 的证据等级在范围声明里再钉一次。

---

## 需要加入 `references.bib` 的 4 条（arXiv 页面已逐条核实存在）

```bibtex
@article{xu2026steering,
  title={Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics},
  author={Xu, Ziwen and Wu, Chenxi and Sun, Hongbin and Hong, Haoming and Wang, Mengru and Yao, Yunzhi and Huang, Lei and Xue, Huajun and Deng, Shumin and Chu, Zhendong and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2602.02343},
  year={2026}
}

@article{luo2026focus,
  title={Don't Lose Focus: Activation Steering via Key-Orthogonal Projections},
  author={Luo, Haoyan and Espinosa Zarlenga, Mateo and Jamnik, Mateja},
  journal={arXiv preprint arXiv:2605.06342},
  year={2026}
}

@article{hoang2026sae,
  title={Sparse Autoencoders Encode Both Concepts and Functions: The Downstream Geometry of Feature Effects},
  author={Hoang, Phu Gia and Chatterjee, Anwoy and Chakraborty, Tanmoy and Gurevych, Iryna and Dutta, Subhabrata},
  journal={arXiv preprint arXiv:2607.24645},
  year={2026}
}

@article{hong2025reasoning,
  title={The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction},
  author={Hong, Yihuai and Zhou, Dian and Cao, Meng and Yu, Lei and Jin, Zhijing},
  journal={arXiv preprint arXiv:2503.23084},
  year={2025}
}
```

> ⚠️ `xu2026steering` 的作者全名是按 arXiv 页面的缩写形式补全的，**推送前请再核对一次**。其余三条为页面原文。

---

## 版面账

| 项 | 行数 |
|---|---:|
| 新增 6.2 | $+14$ |
| 6.3 插入第三条设计要求 | $+2$ |
| Scope 加一句 | $+1$ |
| 章标题缩短 | $-1$ |
| 6.1 首句改写 | $0$ |
| **合计** | **$+16$ 行（约 0.35 栏）** |

若放不下，按此顺序回收：
1. 6.2 第三段（定位段）里删掉 `\citet{hoang2026sae}` 那一句（约 30 词）——但这会同时丢掉与我们负结果的呼应，是可惜的一刀；
2. 6.1 第三段的最后一个分号从句（`and the effect leaves the coordinate on which it was imposed`，约 12 词）——它与 6.2 有重复；
3. 6.3 的 `Another is a two-stage controller...` 一句（约 25 词）——三条设计路线里最弱的一条。