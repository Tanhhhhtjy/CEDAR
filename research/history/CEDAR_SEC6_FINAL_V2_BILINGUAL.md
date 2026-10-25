# 第六部分终稿 v2（中英对照）— 未推送

> 结构：两个**递进**的假说 + 由它们推出的控制器构想。
> 6.1 在 capping 场域内解释现象；6.2 跳出 capping、上升到激活层单轴干预的一般机制；6.3 纯构想。

---

## \section{Representation Is Not Policy}
**中文章标题：表征不等于策略**

---

### \subsection{From Geometric Displacement to Representational--Policy Decoupling}
**中文：从几何位移到表征—策略解耦假说**

```latex
The displacement we traced is not a capability loss. Revision moved without a matching change in the independent validity readout; validity remained linearly readable at the layer where the answer is decided; and the shift survived a token-identical reasoning trace. We therefore propose \emph{representational--policy decoupling}: revision involves two separable objects---the \emph{representational content} that encodes the current answer, the proposed alternative, and whether the offered warrant is valid, and the \emph{revision policy} that propagates, weights, and compares those contents before reading them into option logits---and an intervention can leave the first intact while altering the second.

The two objects come apart because they live at different places in the forward pass. P25 rectifies a persona-related coordinate at a fixed window of layers, but every later layer operates on the rectified state. Those layers remix the correction into a multidimensional decision state, so the relative support for the current answer changes even when the generated tokens do not. Validity information is not removed by this---it is simply no longer read the same way. Every part of the observed signature follows. Target-hidden prompts move because what changed is the geometry of the current answer, not attraction to a proposed one. $V$ and $F$ need not separate because the change is not organized by warrant validity. Different banks yield different revisions because the same continuous displacement meets different local decision boundaries. And the effect leaves the coordinate on which it was imposed because the remixing is what carries it.

The account is discriminating. If the validity code is preserved, a probe trained on uncapped states should transfer to capped states and back; failure would favor re-encoding or loss over decoupling. If its downstream use is causal, patching that route from uncapped into capped inference should restore part of $V-F$ selectivity without freezing revision in general.
```

> **中文：** 我们追踪到的位移不是一次能力损失。改答发生了移动，而独立的有效性读数没有出现与之匹配的变化；在答案被决定的那一层，有效性仍然可被线性读出；而这一位移在推理 token 完全相同的条件下依然存在。我们据此提出\emph{表征—策略解耦假说}：改答涉及两个可分离的对象——\emph{表征内容}，它编码当前答案、被提议的替代答案，以及所给理由是否有效；与\emph{修订策略}，它传播、加权并比较这些内容，最后把结果读出为选项 logits——而一次干预可以在保持前者完好的同时改变后者。
>
> 二者之所以能分开，是因为它们位于前向计算的不同位置。P25 在固定的层窗口上整流一个与人格相关的坐标，但其后的每一层都是在被整流过的状态上运算的。这些层把该修正重混进一个多维的决策状态，于是即便生成的 token 没有变化，当前答案获得的相对支持也已改变。有效性信息并未因此被移除——它只是不再以同样的方式被读取。观察到的全部特征都由此推出：目标被隐藏的提示同样移动，因为改变的是当前答案的几何，而不是对某个被提议答案的吸引；$V$ 与 $F$ 不必分开，因为这次改变并不是按理由有效性组织的；不同题库给出不同的改答，因为同一个连续位移遇到的是不同的局部决策边界；而效应之所以离开了施加它的坐标，正是因为承载它的就是那次重混。
>
> 该假说具有判别力。如果有效性编码被保留，那么在未钳制状态上训练的探针应当能迁移到钳制状态并可反向迁移；迁移失败则更支持"重新编码"或"信息丢失"，而不是解耦。如果它的下游使用具有因果性，那么把这条路径从未钳制补丁到钳制推理中，应当在不整体冻结改答的前提下恢复一部分 $V-F$ 选择性。

> ▸ 三段各司其职：**第一段提出并完整陈述假说**；第二段解释机制并逐条对上观察；第三段给判别实验。相比 v1 删掉了约 40 词的重复铺垫。

---

### \subsection{From Capping to Axis-Conditioned Gain Redistribution}
**中文：从 capping 场域到轴条件增益重分配假说**

```latex
Everything above is stated inside one operator. But a percentile cap is one instance of a larger class: an intervention that acts on a single direction in activation space, whether by addition, ablation, projection, or thresholded rectification. If decoupling is not a quirk of this cap but a property of that class, then what we measured is a special case of a general mechanism. We conjecture that it is, and state the general form as \emph{axis-conditioned gain redistribution}: a one-dimensional intervention does not add or remove a concept, but reweights the relative decision gain of many downstream pathways at once, graded by how strongly each pathway is coupled to the intervened axis; pathways bound to that axis gain share of the final decision, weakly coupled pathways keep their content but lose influence over it; and because the operator fires repeatedly and its total correction grows with the length of what the model produces, this instantaneous reweighting accumulates into a persistent policy bias, which is why an inference-time edit can behave like a low-rank parameter change rather than a local nudge. For an Assistant direction, the pathways that gain are plausibly identity maintenance, policy compliance, and resistance to role shift; the pathway that loses is whether a particular inference step is sound.

Its components are separately established, and it is their conjunction that is new. \citet{xu2026steering} unify fine-tuning, LoRA, and activation intervention as control-induced weight dynamics, showing that stronger control raises preference for a target concept while lowering general utility; we conjecture the shape of that loss, which we expect to be graded by coupling rather than uniform. \citet{luo2026focus} give one concrete channel, in which steering reroutes attention away from key evidence tokens. \citet{hong2025reasoning} show that a single residual direction can rebalance reasoning against memorization, so one coordinate can decide which competence governs an answer. \citet{hoang2026sae} find that feature interventions seldom act as one reusable direction, producing diffuse and context-dependent logit geometry---as we also observe once the correction has left its exit coordinate. Read together, these results answer \emph{that} single-axis control has global consequences; a coupling-graded, exposure-accumulating account would answer \emph{which} consequences and \emph{how much}. What it ultimately points to is an analysis we do not yet have: a map of the pathways through which a language model's decisions are actually formed at the activation level, against which any intervention could be scored before it is deployed.
```

> **中文：** 以上一切都是在一个算子内部陈述的。但分位数钳制只是一个更大类别的一个实例：**在激活空间的单一方向上施加的干预**——无论它采取加法、消融、投影，还是阈值整流的形式。如果解耦不是这个钳制的怪癖，而是这一类别的性质，那么我们所测到的就是一个一般机制的特例。我们猜想事实如此，并把它的一般形式陈述为\emph{轴条件增益重分配假说}：一次一维干预并不是加上或删掉某个概念，而是**同时重新加权了许多下游通路的相对决策增益**，其程度取决于每条通路与被干预的那根轴耦合得有多紧；与该轴绑定的通路在最终决策中占据更大份额，耦合较弱的通路则保留了内容却失去了对答案的影响力；又因为该算子被反复触发、其累计修正量随模型产出的长度而增长，这种瞬时的重加权会累积成一种**持续的策略偏置**——这正是为什么一次推理期的编辑会表现得像一次低秩的参数改变，而不是一次局部的轻推。对 Assistant 方向而言，获益的通路可能是身份维持、策略遵从与抵抗角色偏移；失势的通路则是"某一步推理是否成立"。
>
> 它的各个部件已被分别建立，新的是它们的合取。\citet{xu2026steering} 把微调、LoRA 与激活干预统一为受控的权重动力学，表明控制越强，对目标概念的偏好越高、通用效用越低；我们猜想的是**这种损失的形状**——它应当按耦合分级，而不是均匀。\citet{luo2026focus} 给出一条具体通道：steering 把注意力从关键证据 token 上重新路由开。\citet{hong2025reasoning} 表明单一残差方向就能在推理与记忆之间重新配平，也就是说一个坐标可以决定哪一种能力支配答案。\citet{hoang2026sae} 发现特征干预很少表现为单一可复用的方向，而产生弥散、依赖上下文的 logit 几何——在修正离开其出口坐标之后，我们也观察到同样的情形。合起来看，这些结果回答的是：单轴控制**确实**具有全局后果；而一个按耦合分级、随暴露累积的解释，要回答的是**哪些**后果、**有多大**。它最终指向的是一项我们尚不具备的分析：**一张语言模型在激活层上真正形成决策的通路图**——任何干预都可以在部署之前对照它被评分。

> ▸ 结构：第一段先说"我们此前都在 capping 里"，再把 capping 归入"单轴激活干预"这一更大类别，然后**由小见大**地提出假说并一次性讲完整条链路；第二段做定位并落到那句更大的指向。

---

### \subsection{Toward a Selective Controller}
**中文：走向选择性控制器**

```latex
If both hypotheses hold, a better controller follows. It would detect drift along the persona axis as the released cap does, but actuate elsewhere: away from the components that carry evidence-sensitive movement, so that stabilizing who the model is does not cost it the ability to tell a valid reason from a fallacious one. And because redistribution would accumulate with exposure rather than with any single activation, such a controller would meter its total correction rather than each projection, so that a longer answer does not silently receive a stronger intervention. We sketch a protected-subspace realization in Supplementary Section~I.
```

> **中文：** 如果上述两个假说成立，一种更好的控制器随之可以被推想出来。它会像公开的钳制那样沿人格轴检测漂移，但**在别处作动**：避开那些承载证据敏感移动的成分，使"稳定模型是谁"不必以"分辨有效理由与谬误理由的能力"为代价。又因为重分配是随暴露累积、而非取决于任何单次激活，这样的控制器会计量它的**累计修正总量**，而不是每一次投影，以免更长的回答在无声中受到更强的干预。我们在补充材料第 I 节给出一种受保护子空间的实现草图。

> ▸ 按你的要求：只写构想，不写评测契约、不写限制、不写比较规则。原来那些内容留在补充材料。

---

### \paragraph{Scope.}
**中文：范围**

```latex
The mapping-level evidence is strongest for the released Qwen p25 configuration, and matched forced choice transports the broad shift, but not the mechanistic diagnostics, to the released Llama pair. Gain redistribution is stated as a conjecture and is not tested here. Cross-condition probes, activation patching, dose-matched interventions, and axes other than the Assistant direction are its direct tests.
```

> **中文：** 映射层面的证据在公开的 Qwen p25 配置上最强；匹配的强制选择把广泛位移迁移到了公开的 Llama 组合上，但机制诊断没有一同迁移。增益重分配是以猜想的形式陈述的，本文并未检验它。跨条件探针、activation patching、剂量匹配的干预，以及 Assistant 方向之外的其他轴，都是对它的直接检验。

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

> ⚠️ `xu2026steering` 的作者全名按 arXiv 页面缩写补全，推送前再核对一次。

---

## 版面账（相对当前 Overleaf 版本）

| 项 | 行数 |
|---|---:|
| 6.1 三段压缩重写 | $-5$ |
| 新增 6.2 | $+16$ |
| 6.3 精简为纯构想 | $-4$ |
| Scope 微调 | $0$ |
| 章标题缩短 | $-1$ |
| **合计** | **$+6$ 行** |

若仍需回收：删 6.2 第二段的 `\citet{hoang2026sae}` 一句（约 32 词，但会丢掉与我们负结果的呼应），或删 6.1 第二段末尾 `And the effect leaves the coordinate...` 一句（约 16 词，与 6.2 略有重复）。