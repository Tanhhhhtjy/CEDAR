# §6 新增小节草案：轴条件增益重分配假说（中英对照）

> 位置：紧跟在 `\subsection{The Hypothesis}`（表征—策略解耦）之后，`\subsection{Toward Selective Control}` 之前。
> 定位：**明确标注为猜想**，不引用我们自己的新数据作为证据，只用逻辑 + 已发表工作定位。
> 篇幅：正文约 14 行 + 4 条新参考文献。

---

## 英文正文

```latex
\subsection{Axis-Conditioned Gain Redistribution}

Decoupling says that the route from information to revision changed. It does not say why an axis chosen for persona should have that power over an unrelated task. We close this section with a conjecture that would explain it, and that our results do not test.

A one-dimensional intervention may not add or remove a single concept. It may instead change the relative decision gain of many downstream pathways at once, graded by how strongly each pathway is coupled to the intervened axis. Pathways bound to the axis---for an Assistant direction, plausibly identity maintenance, policy compliance, and resistance to role shift---gain share of the final decision, while weakly coupled pathways, such as whether a particular inference step is sound, retain their content but lose influence over the answer. Because a threshold cap fires repeatedly and its total correction grows with how much the model writes, this instantaneous reweighting accumulates, which is why an inference-time edit can behave like a low-rank parameter change rather than a local nudge.

Parts of this picture are established. \citet{xu2026steering} unify fine-tuning, LoRA, and activation intervention as control-induced weight dynamics and show that stronger control raises preference for a target concept while lowering general utility; our conjecture concerns the shape of that loss, which we expect to be graded by coupling rather than uniform. \citet{luo2026focus} identify one concrete channel, showing that steering reroutes attention away from key evidence tokens. \citet{hong2025reasoning} show that a single residual direction can rebalance reasoning against memorization, so one coordinate can decide which competence governs an answer. \citet{hoang2026sae} find that feature interventions seldom act as one reusable direction and often produce diffuse, context-dependent logit geometry. What is missing is the link between them: a single account in which the redistribution is a function of axis coupling and accumulates with trigger exposure.

The conjecture is falsifiable. It predicts a dose--response between cumulative above-threshold correction and the size of the answer-geometry shift; a contraction graded by each option's downstream coupling to the enforced axis rather than uniform across options; and no such gradation for interventions matched in norm but chosen orthogonally to any behaviorally coupled direction. Deciding it requires dose curves, coupling estimates, and patching experiments we have not run.
```

---

## 中文对照

**轴条件增益重分配**

解耦假说说的是：从信息到改答的**路径**变了。它没有回答：为什么一根为了 persona 而选出来的轴，会对一项与人格无关的任务拥有这种支配力。我们用一个能解释它的猜想收束本节——而我们的结果并没有检验这个猜想。

一次一维干预，也许并不是加上或删掉某一个概念。它可能是**同时改变了许多下游通路的相对决策增益**，其权重取决于每条通路与被干预的那根轴耦合得有多紧。与该轴绑定的通路——对 Assistant 方向而言，可能是身份维持、策略遵从、抵抗角色偏移——在最终决策中占据更大份额；而耦合较弱的通路，比如"某一步推理是否成立"，内容仍然保留，却失去了对答案的影响力。又因为阈值钳制会被反复触发、其累计修正量随模型写得多少而增长，这种瞬时的重加权会**累积**起来——这正是为什么一次推理期的编辑会表现得像一次低秩的参数改变，而不是一次局部的轻推。

这幅图景的若干部件已经被建立。\citet{xu2026steering} 把微调、LoRA 与激活干预统一为受控的权重动力学，并表明控制越强，对目标概念的偏好越高、通用效用越低；我们的猜想关心的是**这种损失的形状**——我们预期它由耦合程度分级，而不是均匀的。\citet{luo2026focus} 给出了一条具体通道：steering 会把注意力从关键证据 token 上重新路由开。\citet{hong2025reasoning} 表明单一残差方向就能在推理与记忆之间重新配平，也就是说一个坐标可以决定哪一种能力支配答案。\citet{hoang2026sae} 发现特征干预很少表现为单一可复用的方向，往往产生弥散、依赖上下文的 logit 几何。所缺少的是把它们连起来的那一条：一个统一的解释，其中重分配是**轴耦合的函数**，并且**随触发暴露累积**。

这个猜想是可以被证伪的。它预测：累计超阈修正量与答案几何位移之间存在剂量—反应关系；收缩应当按各选项与被约束轴的下游耦合程度分级，而不是在选项间均匀；而对于范数匹配、但方向与任何行为耦合方向正交的干预，不应出现这种分级。裁决它需要剂量曲线、耦合度估计与 patching 实验——这些我们都还没有做。

---

## 需要加入 `references.bib` 的 4 条（已逐条核实 arXiv 页面真实存在）

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

> ⚠️ `xu2026steering` 的作者名我只从 arXiv 页面拿到缩写形式（Xu, Z.; Wu, C.; …），上面的全名是按常见拼写补的，**推送前请核对一次 arXiv 页面**。其余三条作者为页面原文。

**暂不建议加入的两条**（真实存在，但对我们的论证是支撑而非近邻，且版面紧张）：
- `2606.08365` Pre-Intervention Prediction of SAE Steering Side Effects
- `2603.06495` COLD-Steer

若审稿人质疑覆盖面，可在附录一并列出。

---

## 为什么这样处理（三点理由）

1. **`xu2026steering` 是必须引的。** 它已经把"微调 / LoRA / 激活干预 = 受控权重动力学"统一发表了。如果我们不引而直接提"steering 等效于微调"，审稿人会判定我们在重述已有框架。引了之后，我们的贡献自动变窄也变准：**他们给出"偏好↑、通用效用↓"这个事实，我们猜想的是这个"↓"的形状——按耦合分级，而非均匀。**

2. **`hoang2026sae` 恰好与我们自己的一个负结果对上。** 我们检验"逐条目 logit 变化是否由一个固定方向承载"时，第一主成分只解释 37%–51%（随机参考 44%–54%，没有更差）。也就是说，在答案头上它**不是**一个可复用的固定方向——这正是他们所说的弥散、上下文依赖的那一类。把这条引上，我们的负结果就从"没测出来"变成"与独立工作一致"。

3. **放在解耦之后独立成节是对的。** 第六节因此形成一个清晰的阶梯：
   现象（第五节）→ 表征—策略解耦（有间接证据的解释）→ 轴条件增益重分配（无直接证据的更高阶猜想）→ 由猜想推出的控制器设计 → 范围与局限。
   读者一眼就能看出哪一级有证据、哪一级是我们提出的前沿假设，这既是研究品味的体现，也是最安全的写法。