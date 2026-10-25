# 从 CEDAR 到 FMTS：激活干预结论能否迁移到认识论世界模型

**日期：2026-09-02**  
**目标会议：FMTS @ NeurIPS 2026（Foundation Models for Temporal Systems）**  
**目标形态：4 页以内、双盲、非归档短论文**

## 0. 最终判断

可以迁移，而且迁移后有机会形成一篇比“再做一次 activation steering”更有新意的 FMTS 小论文。但迁移的对象必须分清：

- **可以直接迁移的是识别框架、测量方法和竞争假设；**
- **不能直接迁移的是 CEDAR 在 Assistant Axis 上得到的经验结论；**
- CEDAR 的两个理论表述中，`representational--policy decoupling` 是论文提出但尚未被因果 patching 完全建立的 account，`axis-conditioned gain redistribution` 更明确地仍是 conjecture；它们应在新领域中被检验，而不是被当前项目当成前提。

最值得做的新问题是：

> **Knowing Is Not Asking：语言模型可以在内部表示关于时间机制的正确 posterior，却未必用它选择最有信息量的下一次实验。**

这把 CEDAR 的核心洞见“representation is not policy”迁移成：

> **belief is not inquiry / posterior representation is not experiment policy。**

这与现有 epistemic-world-model 方向形成自然连接，也比“找一个 uncertainty direction 然后 steering”更能抵抗已有工作的覆盖。

---

## 1. CEDAR 的真实结论和假设

以下区分以当前 `CEDAR_overleaf_AAAI27/main.tex` 和 `CLAIM_EVIDENCE_REGISTER.md` 为准。

### 1.1 已有较强证据支持、可以作为方法经验迁移的结论

#### A. 表示可读与策略使用可以分离

CEDAR 中，valid/fallacious 信息在后层仍可线性读取；与此同时，answer geometry 已发生变化，且 validity 读数没有同步改变。固定可见推理后，连续答案 margin 仍然移动。

这支持一个重要方法论事实：

> 从激活中读到某种信息，不能证明下游决策策略以预期方式使用它。

但当前 CEDAR 并未完成双向 patching，因此更强的“已经证明 representational--policy decoupling 是因果机制”仍不能写。

#### B. 连续 margin 与离散选择发生分离

CEDAR 发现 activation capping 可以明显改变连续 option-margin，而 argmax 选择不动；在其他题上，相似的连续位移又可能产生不同类别迁移。

迁移后的意义：不能只报告模型是否选中 oracle-best experiment。必须同时报告：

- 每个实验选项的 logits / probability mass；
- oracle EIG 与模型 action score 的排序相关；
- EIG regret；
- action-margin；
- categorical experiment choice。

否则一个一般性 logit 缩放可能被误读成“模型更愿意探索”。

#### C. 单轴局部修正会在下游被重新混合

CEDAR 的局部 Assistant-Axis 修正到后层后不再由原退出方向概括，off-axis displacement share 增加。这说明干预一条可命名方向并不等于只改变该概念。

迁移后的意义：即使找到“posterior direction”或“uncertainty direction”，沿它 steering 后实验选择改变，也不能立即声称 posterior 因果控制 policy。必须排除下游一般性增益变化和分布外扰动。

#### D. action-head 的 affine contraction 是一个真实替代解释

CEDAR 中，p25 对四个答案 logit 的变化很大部分可由 affine contraction 描述，且部分随机方向也会产生 contraction。

迁移后的意义：对实验选项 logits 做 activation patching/steering 后，应拟合

\[
\ell^{\mathrm{intervened}}_e=a\ell^{\mathrm{base}}_e+b+\varepsilon_e,
\]

检查所谓“信息寻求增强”究竟来自特定高-EIG选项的 residual，还是仅仅来自所有 action logits 的统一缩放。

#### E. 固定可见推理是隔离隐式 policy 的有效工具

CEDAR 使用 token-identical replay，证明连续位移不要求可见推理文本改变。

迁移后的意义：可以让模型先生成完全相同的显式 hypothesis summary / reasoning trace，再仅在最终 experiment-choice boundary 交换内部 belief state。这样可以区分：

- patch 改变了模型说出来的推理，因此间接改变行动；
- 即使可见推理相同，隐藏 posterior state 仍直接改变行动策略。

### 1.2 只能作为待检验假设迁移的内容

#### F. Representational--policy decoupling

CEDAR 将它定义为：模型保留了表示内容，但 intervention 改变了传播、加权和比较这些内容的 revision policy。

这是新课题最值得直接检验的假设，但不能写成 CEDAR 已经证明的定律。

#### G. Axis-conditioned gain redistribution

CEDAR 猜测单轴干预会按各下游路径与该轴的耦合程度重新分配 decision gain，并随 exposure 累积。

在新课题中，它更适合作为**干预伪影假设**：如果 posterior steering 改变实验选择，究竟是改变了 epistemic belief，还是沿某方向重分配了 action-policy gain？

当前不能把它当成已验证机制，也不建议在四页 FMTS 中同时承担对这一大理论的全面证明。

---

## 2. 旧概念到新领域的一一映射

| CEDAR | 新的 temporal epistemic-world-model 研究 |
|---|---|
| 当前答案 \(a_0\) | 当前最受支持的时间机制 \(M_{\text{MAP}}\) |
| 候选替代答案 \(a_t\) | 竞争机制或下一候选实验 |
| valid / fallacious warrant | 对不同机制具有不同 likelihood ratio 的诊断证据 |
| reason validity | evidence diagnosticity / Bayesian likelihood |
| revision policy | 机制 posterior 更新及 experiment-selection policy |
| option logits | candidate-intervention logits |
| initial-answer margin | 当前默认/启发式实验相对其他实验的 action margin |
| solver certification | 精确 Bayesian posterior 与 EIG oracle |
| target-hidden control | 标签、选项位置、变量名随机化及无固定目标实验控制 |
| fixed-trace replay | 固定 hypothesis summary / CoT 的 experiment-boundary replay |
| validity probe | posterior log-odds / entropy probe |
| capping/steering artifact | action-logit contraction / gain redistribution control |

关键变化在于：CEDAR 问“模型为什么改变答案”，新论文问“模型是否用自己的不确定性决定应该观察什么”。二者共享 representation-versus-policy 的骨架，但研究对象、任务、oracle 和行动语义均不同，因此不是对 AAAI 工作的简单切片。

---

## 3. 推荐的 FMTS 论文命题

### 3.1 推荐标题

首选：

> **Knowing Is Not Asking: Do Internal Beliefs over Temporal Mechanisms Guide Experiment Selection in Language Models?**

更机制化：

> **Belief Is Not Inquiry: Testing Posterior--Policy Coupling in Foundation Models for Temporal Systems**

如果得到强正结果：

> **From Internal Model Beliefs to Informative Interventions in Temporal Foundation Models**

### 3.2 一句话核心问题

> 在多个时间生成机制仍与观测相容时，开放权重 LLM 是否形成与精确 Bayesian posterior 对齐的内部状态，并将该状态因果地用于选择最能区分这些机制的下一次干预？

### 3.3 四个竞争假设

| 假设 | 表征预测 | 行动预测 | patching 预测 |
|---|---|---|---|
| H1 Posterior--policy coupling | posterior 可跨模板解码并正确更新 | action 与 EIG 排序一致 | 交换 posterior state 会按 oracle 方向翻转实验 |
| H2 Representational--policy decoupling | posterior 可解码、预测也可正确 | 实验选择仍依赖固定启发式或默认策略 | patch 可改变 posterior readout/预测，但不改变实验策略 |
| H3 No stable epistemic state | posterior 只在同模板可读或不可读 | 行动不稳定 | patch 无方向性效果 |
| H4 Generic gain redistribution | 方向干预改变 action margin | 看似提高/降低探索 | 变化主要由 affine contraction 或一般扰动解释，而非高-EIG residual |

无论得到哪一种，都有可以写的结论。对 FMTS 而言，H2、H3、H4 的负结果也符合其明确鼓励的 preliminary、exploratory、negative 和 critical analysis 类型。

---

## 4. 为了与 FMTS 真正吻合，微世界必须是“时间系统”

纯静态 DAG 可能只勉强落在 causal structure，不足以让审稿人立即感到这是 Foundation Models for Temporal Systems。

建议使用小型离散时间机制：

- 3–4 个变量，时间索引 \(t=1,\ldots,T\)；
- 2–4 个候选 transition mechanisms；
- 候选机制在被动序列上相同或近似相同，但在受控 intervention 后产生不同未来轨迹；
- observation history 逐步到达，模型每轮更新 belief；
- 可选 action 是在某个变量、某个时间点施加 intervention，然后观察后续 1–3 步；
- 外部程序精确计算 \(p(M\mid D_t)\)、posterior entropy 和每个 intervention 的 EIG。

一个最小例子：

- \(M_1\)：\(A_t\to B_{t+1}\to C_{t+2}\)；
- \(M_2\)：\(C_t\to B_{t+1}\to A_{t+2}\)；
- \(M_3\)：\(B\) 是共同时间驱动源；
- 被动观测通过参数对称性制造近似等价；
- 对 \(A_t,B_t,C_t\) 的干预具有随 posterior 改变的不同 EIG。

重点不是模拟复杂物理，而是形成一个真正的 sequential belief state：历史增长、posterior 演化、未来预测和干预决策都带时间含义。

这对应 FMTS 的以下官方范围：

- adaptive or agentic forecasting；
- histories conditioned on actions/events；
- memory and persistent state；
- calibrated probabilistic forecasting；
- causal structure；
- reliable decision-making and evaluation。

---

## 5. 四页内可完成的最小实验

### Module A：Posterior-aligned temporal micro-world

生成 2–3 个机制家族，每个 episode 输出：

- 时间观测历史；
- 精确 posterior odds；
- posterior entropy；
- 每个候选 intervention 的 EIG；
- 使最优实验翻转的 paired histories。

无需追求一个大型 benchmark。主贡献是 matched identification design。

### Module B：Representation versus behavior

对模型同时测：

1. 显式机制概率或相对可信度；
2. 未来轨迹预测；
3. 下一实验选择及选项 logits；
4. EIG regret；
5. 内部 posterior log-odds / entropy probe。

probe 必须 cross-template、cross-label、最好 cross-mechanism-family 测试，不能只做同分布读出。

### Module C：Posterior-to-policy causal test

在 matched history pair 间交换候选层/token 的 hidden state：

- 检查 posterior readout 是否按预期转移；
- 检查 experiment choice/logits 是否转移；
- 固定显式 reasoning token 后重放 final decision；
- 加入随机方向、正交方向、错误层、错误 token 和 norm-matched controls；
- 对 action logits 做 affine fit，分离 contraction 与 EIG-specific residual。

### 模型配置

- 小模型用于定位：Qwen 7B/14B；
- 主结果：一个可部署的 Qwen 32B/40B 开放模型；
- ChatGPT、Claude、Qwen4 API：只做行为参照，不承担机制结论。

如果时间非常紧，可以只对一个开放模型完成完整机制链，API 结果放附录或直接不做。四页论文中，一个干净的因果结果比六个模型的浅表表格更有价值。

---

## 6. 什么结果足以构成一篇 FMTS 短论文

### 强正结果

- posterior 在内部跨模板可解码；
- evidence arrival 后方向和幅度与 oracle 更新一致；
- matched patch 双向改变 posterior readout；
- experiment policy 按 oracle EIG 方向同步翻转；
- random/orthogonal controls 和 affine contraction 解释失败。

可写：存在一个受控任务范围内的 posterior-to-policy causal bridge。

### 最有 CEDAR 风格的负结果

- posterior 可稳定解码；
- 模型的显式预测随 posterior 正确变化；
- 但实验选择不随 posterior patch 转移，或仍由根节点/覆盖度启发式决定。

可写：**temporal belief is represented but not used for inquiry**，即新的 representational--policy decoupling。

这可能比普通“模型 EIG 选得不好”更有技术含量。

### 干预伪影结果

- posterior steering 似乎改变实验选择；
- 但 action logits 的变化大部分由 affine contraction、length/exposure 或一般随机方向解释。

可写：对内部 epistemic direction 的 steering 会制造“认识控制”的假象；需要 action-geometry controls。

这会成为 CEDAR 方法论向 AI-for-Science/temporal world models 的直接新贡献。

### 不足以成篇的结果

- 只有同模板 probe AUC；
- 只有 prompt 后选实验准确率；
- 只有一次 steering 后 choice flip；
- 没有 posterior oracle；
- 没有随机/正交/错误位置对照；
- 没有把 action-margin 与 categorical choice 分开。

---

## 7. Tech-paper thinking template

### 7.1 Paper type

**New Problem/Setting Paper，带跨领域方法迁移。**

论文不是提出更强的 EIG 算法，也不是大型 benchmark；它提出一个已有研究尚未严格测量的对象：内部 temporal-model belief 是否进入 information-seeking policy。CEDAR 提供识别思想，Bayesian experiment design 提供规范 oracle。

### 7.2 Thinking template

| Stage | Content |
|---|---|
| Research background | Temporal foundation models increasingly forecast, simulate and support decisions, but successful predictions or informative questions do not reveal whether the model maintains and uses uncertainty over competing temporal mechanisms. |
| Limitation 1 | Behavioral scientific-discovery and BED work evaluates selected experiments but does not identify the LLM's internal posterior-to-policy computation. |
| Limitation 2 | Belief-representation work decodes or intervenes on hidden beliefs, but rarely tests whether those beliefs control information-seeking experiments. |
| Limitation 3 | Activation interventions can change action logits through general contraction/gain effects, so a choice flip alone is not evidence of epistemic control. |
| Our Goal | Test whether internal beliefs over competing temporal mechanisms causally guide information-gaining interventions, while separating belief transfer from generic policy-geometry perturbation. |
| Challenge 1 | The true epistemic state and optimal experiment must be known exactly. |
| Challenge 2 | Decodable posterior information must be distinguished from posterior information actually used by policy. |
| Challenge 3 | Causal interventions must be separated from nonspecific logit rescaling and visible-reasoning changes. |
| Methodology | A posterior-verified temporal micro-world combines matched histories, internal belief readout, and controlled posterior-to-policy intervention. |
| Module A | Construct temporal mechanisms with exact posterior/EIG and paired histories whose optimal experiment flips. |
| Module B | Jointly measure posterior calibration, temporal prediction, experiment logits and cross-template internal probes. |
| Module C | Use matched patching, fixed-trace replay and affine action-geometry controls to test causal policy use. |
| Contribution 1 | Formulate and operationalize posterior--policy coupling for temporal foundation models. |
| Contribution 2 | Provide an exactly verified temporal micro-world and matched identification design. |
| Contribution 3 | Establish coupling, decoupling, or intervention artifact with causal and geometry-aware evidence. |

### 7.3 Self-consistency checks

- Limitations → Goal: PASS.
- Goal → Challenges: PASS.
- Challenges → Modules: PASS, one-to-one mapping.
- Modules → Contributions: PASS, provided at least one causal intervention result is completed.

当前严重缺口只有一个：**尚无试验结果。** 这是研究执行缺口，不是逻辑链缺口。

---

## 8. 推荐的四页结构

### Page 1：问题与识别缺口

- temporal world models 与 active experiment selection；
- 现有研究分别覆盖行为 EIG 和内部 belief，却没有连接两者；
- CEDAR 启发的 representation-versus-policy 问题；
- Figure 1：同一 posterior state 是否进入 experiment policy 的四种机制。

### Page 2：任务与方法

- temporal micro-world；
- exact posterior/EIG；
- matched histories；
- representation、prediction、policy 三类读出；
- patching 和 controls。

### Page 3：结果

- 行为 calibration/EIG regret；
- probe 随层和证据时间的曲线；
- matched posterior patch 对 prediction 与 action 的效应；
- affine contraction/residual 分解。

### Page 4：机制判定与意义

- 四个假设中哪一个得到支持；
- 与 BED-LLM、belief geometry、HOT-3、BoxingGym、CEDAR 的边界；
- FMTS 相关性：temporal state、uncertainty、reliable decisions；
- 局限：合成世界、单模型/少模型、短 horizon。

---

## 9. 时间与执行建议

FMTS 官方截止时间是 **2026-09-16 11:59 UTC（Sep 15 AoE）**。短论文允许 work in progress、preliminary、exploratory 和 negative findings，参考文献与附录不计入 4 页，但审稿人没有义务阅读附录。

建议采用单一决定性实验，而不是构建完整“认识论世界模型平台”：

1. 先生成最小 temporal paired-world 和 oracle；
2. 用 Qwen 7B/14B 确认任务不是随机也不是满分；
3. 定位 posterior readout；
4. 做少量 matched patching；
5. 只有出现稳定结果后才上 32B/40B 定点确认；
6. 论文主线根据实际结果在 coupling、decoupling、artifact 三者中选择，不预设必须是正结果。

---

## 10. 不应怎样迁移

- 不把 Assistant Axis、persona capping 或 sycophancy 重新命名成 world-model uncertainty；
- 不直接声称 CEDAR 已证明 representational--policy decoupling；
- 不把 CEDAR 的 affine contraction 数值带入新模型或新方向；
- 不将 steering、capping、patching 当作同一种 operator；
- 不复用 AAAI 结果作为 FMTS 新实验；
- 不为了四页篇幅省略能区分 posterior use 与一般扰动的关键对照；
- 不把静态 causal graph 包装成 temporal system，必须让 history、transition、prediction 和 intervention 真正带时间结构。

## 11. 一句话立项建议

> **用 CEDAR 的“表示不等于策略”与 action-geometry 审计框架，研究 LLM 是否把对竞争时间机制的内部 posterior 真正转化为信息性实验；这是一项新问题迁移，而不是复用旧轴或重复旧论文。**
