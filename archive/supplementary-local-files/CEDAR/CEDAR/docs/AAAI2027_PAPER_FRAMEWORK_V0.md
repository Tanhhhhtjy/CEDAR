# AAAI 2027 论文框架 v0：从人格稳定到交互式安全审计

状态：供团队审查，不是最终定稿  
日期：2026-07-22  
证据截止：C1/V7 结果提交 `1336f9d`；V7.1 尚未运行  
理想目标母版：`docs/AAAI2027_IDEAL_PAPER_FRAMEWORK_TBD.md`  

本文件只回答“当前证据已经允许我们说什么”；理想目标母版回答“如果后续实验形成完整证据链，论文最终要写成什么”。两者并行维护，不用每得到一轮结果就重写全文。

## 0. 先给结论：当前最稳的论文是什么

当前不应继续把论文写成“我们已经证明 activation capping 损害正确更新”。现有证据没有支持这种强结论：多数正式或诊断性实验中的 V（有效证据更新）和 N（无证据维持）基本稳定，C1/V7 中 V 仅下降 4.17pp，置信区间跨过 0；E1 也否定了“抵抗状态位于 Assistant Axis 低端、被 capping 拉回”的简单几何机制。

当前最稳、也更有新意的主线是：

> **静态能力不下降，并不能证明人格稳定干预在交互中没有副作用。一个有语义的激活轴不等于一个有语义且安全的控制器；同一 Assistant Axis 上的持续 steering 与阈值 capping 可以产生不同的行为结果，而官方 p25 capping 在 held-out 任务中选择性地增加了社会压力下的 target-follow。**

这条主线把论文从单一的“认识论代价”升级为三层问题：

1. **评测遗漏**：Assistant Axis 原文检查了人格漂移、persona jailbreak 和四项静态能力，却没有测“持有答案后，在有效证据、错误论证和社会压力下如何更新”。
2. **新行为发现**：官方 p25 并未表现为普遍变笨或普遍变顽固，而是在当前最强 held-out 结果中把社会压力下的 target-follow 提高了 18.75pp；AO 和 N 均为 0pp。
3. **新理论方向**：行为效果由“轴的语义、干预算子、交互上下文”共同决定，不能从轴的语义直接推出控制效果。安全控制的目标应从“一维轴上钳制”升级为“同时约束人格稳定、证据分辨、社会压力与输出完整性的多维可行域”。

暂定一句话 TL;DR：

> **Activation capping can stabilize a persona without preserving the model's socially conditioned answer policy; static capability benchmarks do not reveal this failure mode.**

## 1. 标题方案

### 推荐的稳健标题

**Beyond Persona Stability: Auditing the Epistemic and Social Effects of Activation Capping**

优点：不会提前声称“backfire”或“能力损伤”已普遍成立；能容纳当前结果和后续方法。

### 若 V7.1 与外部复现均通过

**An Axis Is Not a Controller: Activation Capping Can Increase Social Compliance**

优点：记忆点强，直接表达理论贡献。风险：必须有至少一个外部数据集或第二模型支持，且需要把 `target-follow` 升格为 `social compliance` 的标注证据补齐。

### 若同层、同剂量的算子对照成立

**Same Axis, Different Outcomes: Operator-Dependent Effects of Activation Control**

优点：方法论贡献最强。风险：目前 E−1a 与 E−1b 的层和算子不同，只能称相邻证据，尚不足以单独支撑该标题。

### 命名建议

原仓库将 CEDAR 展开为 *Cost of Epistemology under Drift-suppressing Activation Restraint*，它预设了“认识论代价存在”。建议论文中将评测框架改写为：

> **CEDAR: Contextual Evaluation of Drift and Answer Revision**

这样保留项目品牌，但不把尚未证实的代价写进名称。

## 2. 中心科学命题

### 2.1 原文留下的空白

Assistant Axis 将默认 Assistant 与角色扮演状态之间的激活差定义为一条轴，并用 threshold capping 把低于正常范围的投影拉回。原文报告：在 persona jailbreak 上可降低近 60% 的有害回答，同时在 IFEval、MMLU-Pro、GSM8K 和 EQ-Bench 上未观察到总体能力下降。

这些基准测量的是一次性静态表现。它们没有回答三个交互式问题：

1. 模型面对真正能推翻原答案的证据时，是否仍会更新？
2. 模型面对貌似严谨但无效的论证时，是否会被带偏？
3. 模型面对不含新证据的社会压力时，是否会为了顺从用户而改口？

因此，原文的“capability preserved”不能逻辑推出“interactional decision policy preserved”。这不是否定原文的人格稳定结果，而是指出其评测覆盖范围之外的安全盲区。

### 2.2 我们最初的假设及其更新

最初假设是：capping 会把所有答案变化都压低，因此模型会更顽固，无法依据有效证据修正答案。

现有实验迫使我们更新这一假设：

- Belief-R 的 8% 更新率来自任务—模型失配，不能证明 Qwen 普遍不会改答。
- 独立改错筛查表明 Qwen 能根据明确证据更新，但也容易接受具体的错误论证。
- v6 与 V7 中 N 基本不变，V 没有出现稳定的大幅损伤。
- v6 greedy 与 V7 的主要变化集中在 S，而不是所有 arm 一起变化。

所以当前主假设应改为：

> **官方 p25 capping 不一定让模型普遍变笨或变顽固；它可能在特定交互与解码条件下改变答案选择策略，使社会压力更容易穿过“已经完成的任务推理”并控制最终输出。**

### 2.3 Axis–Operator–Context 原则

建议把理论贡献写成 **Axis–Operator–Context (AOC) principle**：

\[
\text{Behavioral Effect}=f(\text{Axis Semantics},\text{Intervention Operator},\text{Interaction Context}).
\]

- **Axis semantics**：一条方向在观测上与何种人格或概念相关。
- **Operator**：持续加法、阈值整流、单层或多层、prefill 或 decode、作用剂量。
- **Context**：任务类型、社会压力、thinking、greedy/sampling、答案边界和输出格式。

核心论断不是“Assistant Axis 无效”，而是：

> **一个可解释的方向首先是测量坐标，不自动构成安全控制目标。方向有语义，不代表沿它施加任意算子都会产生同一语义的行为变化。**

## 3. 可直接写进摘要的临时版本

以下摘要只使用已经完成的结果；方括号部分需由后续实验替换或删除。

> Activation capping has been proposed as a way to stabilize the default persona of language models while preserving performance on static capability benchmarks. We ask whether such preservation extends to interactive settings in which a model must distinguish valid evidence from invalid arguments and social pressure. We introduce CEDAR, a contrastive evaluation framework that separates warranted revision, fallacious persuasion, target exposure, social pressure, and no-information controls. On Qwen3-32B, we first reproduce the reported effect of continuous Assistant-Axis steering on PhilPapers prompts, but find that large signed interventions substantially alter the full output distribution. Applying the official p25 threshold cap instead decreases the mean forced-choice sycophancy margin while increasing discrete sycophantic choices by 1.44 percentage points, showing that margin changes need not imply healthier decisions. More importantly, on a 24-family held-out suite, p25 increases social-pressure target-follow from 39.6% to 58.3% (+18.75pp; family-bootstrap 95% CI [+4.17, +33.33]), while target-only exposure and neutral maintenance remain unchanged. Valid-evidence updating decreases by 4.17pp, but its interval includes zero. Token-resolved analysis rejects a simple account in which capping selectively interrupts low-Assistant “resistance states”; the apparent exposure signal is largely explained by reasoning-trajectory length. These findings motivate an Axis–Operator–Context view of representation control: interpretable axes are not themselves safe controllers. [We further introduce and evaluate a context- and phase-aware controller that preserves persona stability while reducing socially induced answer shifts.]

最后一句只有在新方法完成后保留。当前投稿占位摘要不应编造方法收益数据。

## 4. 论文贡献：当前版与增强版

### 4.1 当前已经能够支撑的三项贡献

1. **评测贡献**：提出 CEDAR 对比簇，将有效证据更新、错误论证、社会压力、纯目标提及和无信息维持分开，并同时报告离散迁移、连续 margin、输出完整性和实际 capping exposure。
2. **实证贡献**：在 Qwen3-32B 上发现官方 p25 的交互式效应并非静态能力基准所暗示的“无损”；C1/V7 中社会压力 target-follow 增加 18.75pp，而 AO/N 不变。
3. **否证与理论贡献**：复现公开 Assistant-Axis steering 结果，但发现输出空间变化；进一步用 E1 否定简单的低轴抵抗机制，据此提出 AOC 原则，区分方向语义与控制算子。

### 4.2 若后续实验成功，可增加的第四项贡献

4. **方法贡献**：提出 context- and phase-aware constrained capping，在保持 persona 安全收益的同时保护证据分辨、隔离社会压力并维持输出完整性。

方法贡献必须满足三项硬条件后才能写：

- 在有效的 persona 攻击基线上显示安全收益；
- 在 held-out V/S/AO/N 或外部集上优于官方 p25；
- 不能以普遍拒答、答案冻结或输出质量下降换取表面安全。

## 5. 研究问题

- **RQ1 — Evaluation gap**：静态能力保持是否能预测交互式信念修订与社会压力下的行为保持？
- **RQ2 — Selectivity**：p25 改变的是有效/无效证据分辨、总体更新门槛，还是社会压力特异的 target-follow？
- **RQ3 — Operator dependence**：同一 Assistant Axis 上的持续 addition 与 threshold capping 是否产生不同甚至相反的行为迁移？
- **RQ4 — Mechanism**：效应发生在证据读取、thinking 轨迹、答案映射还是最终输出策略？
- **RQ5 — Mitigation**：能否用上下文、阶段和多信号约束，在保持人格稳定的同时避免社会性响应偏移？

## 6. 全部实验成果如何进入论文

| 阶段 | 关键结果 | 证据等级 | 在论文中的位置 |
|---|---|---|---|
| M2.1 persona/HarmBench | sampled ASR 3.25% → 28.46%；p25 后 19.51%，但攻击强度与相对减害均未过预注册门；greedy 方向反转 | 无效安全基线 | 附录“复现边界”；不能声称我们已复现安全收益 |
| Belief-R | 应更新仅 2/25=8%，应维持 24/25=96%；模板修订后仍失败 | 基线无效 | 附录“为什么必须更换任务”；不能当 capping 结果 |
| Objective correction screen | 干净题中有效证据修正 8/11=72.7%；错误论证接受 14/20=70%；中性稳定 20/20 | 开发诊断 | 动机：更新能力与证据分辨必须分开 |
| v3 五臂 | N=0%、R=8.3%、V=100%、F=75%、S=100%、AO=0% 的更新率；S 饱和 | 开发校准 | 附录：提示与难度可导致地板/天花板 |
| v4 deliberative | thinking 下 V≈100%，F≈0%，S≈2.8%（原 prompt）或 13.9%（中性 prompt） | 开发校准 | 附录：thinking 与 prompt 会把任务二值化 |
| v5 五剂量 bank | 468 条 × 5 条件完整运行；未把扫描结果当确认性总体效应 | response-enriched development | 附录：说明为何放弃“按目标曲线选题”作为论文证据 |
| v6 sampling | thinking 下 F/S 表面改善各 5.6pp；nonthinking 下各恶化 2.8pp；V/N=0 | 开发诊断 | 主文简述上下文依赖，不作为定论 |
| Axis positive control | default/persona projection 与 exposure 方向 12/12 一致；实际 p25 非零 | 工程阳性对照 | Experimental Setup：证明轴、符号与 hook 有效 |
| v6 matched greedy | thinking-greedy 中 S 错误跟随增加 11.1–13.9pp，F 增加 5.6pp，V/N 不变 | 候选发现 | 发现段：触发 held-out 复现 |
| v6 zero-GPU re-audit | S +13.89pp，family-bootstrap CI [+5.56,+25.00]；截断最坏情况 +11.11pp | 候选经复核 | 发现段或附录；仍称 target-follow candidate |
| E−1a signed steering | +200 的 Δmargin=-2.431，与原文 -2.41 接近；正负大剂量都降 margin，且 top token/entropy 大变 | 强复现 + 输出诊断 | Results 1：forced-choice 指标可能误导 |
| E−1b official p25 | margin -0.0938；离散 syc 84.67%→86.11%；A/B 质量下降、entropy 上升 | 强诊断 | Results 2：同一指标的连续/离散方向分离 |
| E−1b.1 | 离散 syc +1.44pp，CI [+0.11,+2.78]；翻转集中在 near-zero margin | 事后诊断 | Results 2 补充；不能外推为普遍谄媚 |
| E0 | v6 S flip 的 decode 暴露高 3.50，CI [+0.95,+5.79]；prefill 不明显 | 粗粒度相关 | Mechanism：提出抵抗干扰候选 |
| E1 | 主效应 -1.15，CI 跨零；后层交互显著反向；length-match 后 0.033；yield thinking 更长 | 对简单机制的否证 | Mechanism 主结果：E0 主要是轨迹长度混杂 |
| C1/V7 held-out | S 39.6%→58.3%，Δ=+18.75pp；AO=0，N=0，V=-4.17pp；7/24 families 有 harmful transitions | 当前最强确认 | 主结果与标题依据 |
| V7.1 cue ablation | 去除 “No new task information is supplied” 的 N/AO/S 对照 | 尚未运行 | 结果回来前保留占位，不在摘要中写 |

对应关键提交：

- [M2.1 pipeline / protocol](https://github.com/Tanhhhhtjy/CEDAR/commit/1610863)
- [v3 calibration](https://github.com/Tanhhhhtjy/CEDAR/commit/0800ebc)
- [v4 neutral paired calibration](https://github.com/Tanhhhhtjy/CEDAR/commit/61c504d)
- [v5 five-dose bank](https://github.com/Tanhhhhtjy/CEDAR/commit/0ae9153)
- [v6 sampling](https://github.com/Tanhhhhtjy/CEDAR/commit/548e5eb)
- [Assistant Axis positive control](https://github.com/Tanhhhhtjy/CEDAR/commit/bb404c4)
- [v6 matched greedy](https://github.com/Tanhhhhtjy/CEDAR/commit/3cbed5d)
- [v6 zero-GPU re-audit](https://github.com/Tanhhhhtjy/CEDAR/commit/86bf46b)
- [E−1a signed curve](https://github.com/Tanhhhhtjy/CEDAR/commit/6f03398)
- [E−1b official p25](https://github.com/Tanhhhhtjy/CEDAR/commit/b3f9d2a)
- [E−1b.1 posthoc audit](https://github.com/Tanhhhhtjy/CEDAR/commit/880aab2)
- [E0 counterfactual exposure](https://github.com/Tanhhhhtjy/CEDAR/commit/476c03a)
- [E1 token-resolved geometry](https://github.com/Tanhhhhtjy/CEDAR/commit/a5936b3)
- [C1/V7 held-out confirmation](https://github.com/Tanhhhhtjy/CEDAR/commit/1336f9d)
- [V7.1 prompt-cue ablation preparation](https://github.com/Tanhhhhtjy/CEDAR/commit/0803a99)

## 7. 推荐的正文结构

### 1 Introduction

段落逻辑固定为：

1. 人格激活控制的重要性，以及 Assistant Axis/capping 的主要承诺。
2. 现有工作用静态能力基准证明“无损”，但静态正确不等于交互中能正确更新、抵抗错误论证和社会压力。
3. 定义评测空白：模型既要在 V 中更新，又要在 F/S 中不更新；只测总体准确率会把“顽固”和“辨别力”混在一起。
4. 介绍 CEDAR 和 AOC 原则。
5. 用三组具体数字给出核心发现：E−1a 复现、E−1b 连续/离散分离、V7 的 +18.75pp。
6. 列出三到四项贡献。

引言中不要说“我们证明 capping 使模型更谄媚”；当前准确表述是：

> We identify a held-out, social-pressure-specific increase in target-follow under the official p25 cap.

### 2 Background and Related Work

#### 2.1 Persona representations and the Assistant Axis

明确区分原文贡献与我们的贡献：轴提取、persona drift、capping 公式和公开参数都属于原工作。

#### 2.2 Sycophancy and answer revision

覆盖 SycophancyEval、FlipFlop、PhilPapers/Playing Devil's Advocate、SycoBench、LoFa、BrokenMath 等。相关工作不能只说“它们没做我们的事”，而要精确写清：它们分别测社会迎合、错误论证、正确用户建议或数学证明，但没有把官方 Assistant-Axis threshold cap 与 V/F/S/AO/N 对比簇结合。

#### 2.3 Side effects of representation engineering

讨论 steering、CAA、persona vectors、CAST/K-CAST 等，并引出“方向语义不等于算子效果”。

### 3 CEDAR: Contextual Evaluation of Drift and Answer Revision

#### 3.1 Contrast-cluster task design

每个 base 共享初始任务、初始答案与候选答案，仅改变第二轮输入：

- N：无新信息，正确行为是维持；
- AO：只提及 target，不含施动者、证据或压力；
- S：社会压力指向 target，但没有任务证据；
- V：有效证据使 target 成为正确答案；
- F：错误论证指向 target；
- R：仅复核，作为部分开发/扩展对照。

#### 3.2 Estimands

至少报告四层结果：

1. 正确率与 target-follow；
2. 完整迁移矩阵（initial→target、target→initial、other-wrong、invalid）；
3. target-minus-initial logit margin；
4. output-space integrity（A/B/C mass、entropy、top-token、截断和格式错误）。

社会成分定义为：

\[
\Delta_{social}=\Delta_S-\Delta_{AO}.
\]

V/F 可以使用 hierarchical SDT 区分证据分辨力与总体更新门槛；S 不应直接混入同一个 \(d'\)。

#### 3.3 Intervention telemetry

分开记录：

- uncapped 轨迹上的 counterfactual exposure；
- capped 轨迹上的 actual exposure；
- prefill、thinking/decode 和 final-answer 阶段；
- layers 46–53；
- thinking tokens、finish reason、延迟与格式完整性。

### 4 Reproduction and Instrument Validation

#### 4.1 Model and official cap

固定 Qwen3-32B revision、BF16、Transformers、官方 layers 46–53 p25 参数；验证 cap vector 与人类可读 Assistant Axis 的符号。

#### 4.2 Axis positive control

用 12 个 default/persona pair 证明 projection 与 exposure 方向正确。该实验只能证明仪器有效，不证明安全收益。

#### 4.3 PhilPapers close replication

复现持续 addition 的 +200 结果，并把 full-vocabulary output diagnostics 与 forced-choice margin 同报。这里建立“指标下降不等于回答质量改善”的第一个警告。

### 5 Results: Static Preservation Does Not Certify Interactive Behavior

#### 5.1 Same axis, different estimands

E−1a：持续 +Assistant steering 降低 sycophancy margin，但正负大剂量都改变输出空间。  
E−1b：官方 p25 使平均 margin 下降，却让离散 sycophantic choice 增加 1.44pp。  

本节结论只写：不同 estimand 与不同 operator 的结论不能互换。由于 E−1a 和 E−1b 的作用层不同，目前不能声称已经完成严格同层算子因果比较。

#### 5.2 Development diagnostics reveal context dependence

把 v3–v6 压缩成一张图或一张表：同一模型可在不同 prompt/thinking/decoding 下从“S=100%”移动到“S≈3%”，p25 在 sampling 与 thinking-greedy 中方向也不同。其作用是说明为什么必须 held-out，而不是把每一轮都写成独立主结果。

#### 5.3 Held-out social-pressure effect

主结果：V7 中 S 从 19/48 增至 28/48，Δ=+18.75pp，family-bootstrap CI [+4.17,+33.33]；AO 与 N 为 0；harmful transitions 分布于 7 个 family，LOFO 全为正。

同时诚实报告：V 48/48→46/48，Δ=-4.17pp，CI [-10.42,0]，所以不能声称 V 无代价，也不能声称 V 已显著受损。

#### 5.4 Prompt-cue ablation

V7.1 删除“没有提供新任务信息”的显式提醒。若效应保留，说明 V7 不是由这句提示人为制造；若明显衰减，主结论应降格为 prompt-contingent effect，并立刻补自然措辞外部复现。

### 6 Mechanistic Diagnosis: What the Effect Is Not

#### 6.1 Competing explanations

- H-stubborn：capping 普遍提高更新门槛；
- H-degradation：capping 损害一般推理或输出；
- H-axis-state：抵抗对应 Assistant Axis 低位，被 cap 拉回；
- H-policy：推理可能仍完成，但社会压力在答案选择/输出阶段获得更高权重；
- H-perturbation：任何足够大的扰动都会改变 near-boundary 输出。

#### 6.2 E0 and E1

E0 的 late-decode exposure 与 5 条 S flip 相关，但日志把 thinking 长度和答案阶段混在一起。E1 使用同题多 seed 和 token-resolved 轨迹后发现：yield 的 thinking 明显更长；长度匹配后轴修正差约为零。因此，不支持“低 Assistant 投影的抵抗状态被选择性打断”。

这是论文的重要结果，不是失败：它排除了一个直觉上很诱人的机制，防止用 lens 或 probe 为错误故事找图。

#### 6.3 当前最值得验证的 policy-arbitration 假说

V7 的部分 harmful transitions 中，生成的 reasoning surface 给出正确任务判断，但最终 JSON 选择了用户要求的 target。这只能称为外显 reasoning–answer mismatch，不能等同于读取了模型“真实内心”。

下一步机制 assay 应把输出拆成两个独立问题：

1. Candidate reasoning 是否有效？
2. 最终应选择哪个选项？

若 p25 下 validity verdict 保持，而 final choice 向 S target 移动，才支持“能力保留、输出策略偏移”。若两者一起变差，更像推理退化。

### 7 From One-Dimensional Capping to a Multidimensional Safe Region

这一节当前先写理论和方法设计，不写结果。

#### 7.1 Control objective

设 \(a(h)\) 为 assistantness，\(e(h)\) 为证据分辨状态，\(s(h)\) 为社会压力响应，\(q(h)\) 为输出完整性。理想干预不是把 \(a(h)\) 单向推到极端，而是在最小改动下满足：

\[
\begin{aligned}
\min_{\delta}\quad & \|\delta\|_2^2 \\
\text{s.t.}\quad & a(h+\delta)\ge \tau_a,\\
& |e(h+\delta)-e(h)|\le \epsilon_e,\\
& s(h+\delta)\le \tau_s,\\
& q(h+\delta)\ge \tau_q.
\end{aligned}
\]

这给出了“同时约束人格安全、保护证据分辨、隔离社会压力、允许动态推理”的形式化目标。

#### 7.2 三种候选实现，按机制结果选择

1. **Phase-aware capping**：若副作用主要发生在 final-answer/decode，限制 cap 的作用阶段，避免在最终决策边界持续整流。
2. **Drift-gated capping**：只有当 Assistant deviation 与独立 persona-drift/risk 信号同时出现时才触发，而不是低投影即触发。
3. **Protected-subspace projection**：估计 evidence 与 social-pressure 子空间，在满足 assistantness 约束时选择对这些子空间影响最小的修正量。

不要同时实现三个。先用阶段消融和 validity/final-choice assay 定位，再选择最便宜、最可反证的一种。

#### 7.3 方法成功的判据

新方法相对官方 p25 必须同时满足：

- persona attack ASR 或 persona drift harm 不劣于 p25 的预定容忍区间；
- S−AO 不再增加；
- V 更新保留；
- N 稳定；
- output mass、entropy、invalid 和 truncation 不恶化；
- 在 held-out family 与至少一个外部基准上成立。

### 8 Limitations and Scope

必须主动写清：

- 当前主结果只有 Qwen3-32B；
- V7 为自建形式域，且使用单一数据负责人整体审核豁免，没有双人逐题 kappa；
- S target-follow 不是对“内部真实谄媚”的直接测量；
- M2.1 的攻击基线无效，因此我们尚未在自己的设置中同时证明官方 p25 的安全收益与社会副作用；
- E−1a 与 E−1b 不是同层、同剂量的严格 operator comparison；
- thinking surface 不能被当作模型真实信念或完整因果机制；
- V7.1、外部集、跨模型和改进方法均是尚未完成的工作。

主动披露这些限制会让主张更可信，也能明确后续实验为什么必要。

### 9 Conclusion

结论只保留三层：

1. static capability preservation is an incomplete safety criterion；
2. official p25 exhibits a held-out social-pressure-specific target-follow effect on Qwen3-32B；
3. interpretable axes should be treated as observables, while safe controllers must be operator-, phase-, and context-aware。

若新方法成功，再加第四层：多维约束控制能够改善该权衡。

## 8. 首图与图表计划

### Figure 1 — Graphic abstract（最重要）

一张图讲完整故事：

1. 左：Assistant Axis 原文测了 persona drift/jailbreak + static capabilities；动态更新区域留白。
2. 中：CEDAR 的同一 base 在 V/F/S/AO/N 下形成对比簇。
3. 右：p25 在 V7 中让 S target-follow 上升 18.75pp，而 AO/N 不动；下方标注“axis ≠ controller”。
4. 最右可预留多维 safe region 图；方法未完成前用虚线和“design target”，不能画成已实现。

### Figure 2 — CEDAR task and estimands

展示 V/F/S/AO/N 的信息差异、规范答案和四类迁移，不用大段文字解释。

### Figure 3 — Signed steering and official capping

左：E−1a 七剂量 margin 曲线。  
中：A/B mass、entropy 或 top-token 变化。  
右：E−1b continuous margin 下降但 discrete syc choice 上升的分离。

### Figure 4 — Held-out V7 effect

四个 arm 的 paired effect 与 family-bootstrap CI；S 和 AO 并排，突出 social excess。

### Figure 5 — Mechanism falsification

E0 粗暴露信号 → E1 token-resolved → thinking length → length-matched near-zero，画成证据筛选流程，而不是只放一张不显著散点图。

### Figure 6 — Improved controller（条件图）

只有新方法完成后加入 persona safety、V preservation、S−AO 和 output integrity 的多目标 Pareto 图。

### Table 1 — Prior evaluation coverage

列：工作、persona drift、jailbreak、static capability、valid evidence、invalid argument、social pressure、target exposure、operator telemetry。该表负责证明“遗漏点”而不是靠引言口头声称。

### Table 2 — Main results

只放 E−1a、E−1b、v6 candidate、V7、V7.1 和最终外部/跨模型结果。v3–v5 放附录。

## 9. 三条投稿路线与切换条件

### Route A — 评测与副作用审计（当前推荐，最低可行论文）

标题方向：*Beyond Persona Stability*。  
需要补齐：V7.1、至少一个自然外部 S 数据集、一个有效 safety positive control，最好加第二模型。  
不要求新控制器一定成功。

### Route B — Axis–Operator–Context 机制论文（更强）

标题方向：*An Axis Is Not a Controller*。  
除 Route A 外，必须完成：同层同剂量 addition vs threshold cap、phase ablation、validity-verdict vs final-choice assay，并排除随机/范数匹配扰动。

### Route C — 新控制器论文（上限最高，风险也最高）

标题方向：*Context-Aware Activation Control for Persona Safety*。  
必须有有效 persona 安全基线、官方 p25、至少一种新方法、held-out 与外部验证，并展示多目标优势。

推荐策略：现在按 Route A 写作；实验上向 Route B 推进；只有机制定位成功后才升级 Route C。不要让整篇论文依赖尚不存在的新方法。

## 10. 论文生死实验与优先级

### P0：立即完成

1. **V7.1 cue ablation**：决定 V7 是否受显式提示暗示。
2. **V7 response audit**：固定四分类与 reasoning/final mismatch 统计，不事后改标签。
3. **首图和 Table 1 草稿**：不依赖新实验，可以马上写。

### P1：决定能否投 AAAI 主赛道

4. **自然外部 S 复现**：SycoBench/SycophancyEval/FlipFlop 中选择可与 AO 配对、答案客观的子集。
5. **有效 persona safety leg**：公开强 persona 或冻结的多轮 persona 方案，先过攻击有效性闸，再比较 p25；不能继续用已失败的 M2.1 代理基线证明安全收益。
6. **第二模型或第二官方配置**：优先 Llama-3.3-70B 官方 cap；若预算不够，至少在 Qwen 上跨数据集、多 seed 确认，并把标题限定为 Qwen case study。

### P2：决定论文是否能升级为机制/方法贡献

7. **Validity-verdict vs final-choice assay**。
8. **prefill-only / thinking-only / final-answer-only / full cap**。
9. **同层同剂量 addition vs cap + 随机/范数匹配方向**。
10. **根据 7–9 的结果只实现一种改进控制器**。

机制工具的顺序仍应保持：行为与 logit → 简单 probe → causal intervention/rescue → MUSE/JLens。不得因为 JLens、MUSE 或 SAE 给出一张可解释图，就把它当作因果机制。

## 11. 风险管理与叙事切换

| 风险结果 | 说明 | 论文如何改 |
|---|---|---|
| V7.1 中 S 效应消失 | V7 主要由显式“无新信息”提示塑造 | 降格为 prompt-contingent boundary effect；外部自然 prompt 成为 P0 |
| 外部 S 不复现 | 自建形式域特异 | 标题改为 Qwen/formal-reasoning case study；不能写普遍 social compliance |
| 第二模型反向 | 强模型依赖 | AOC 原则反而得到支持，但必须按模型分别校准符号、层和剂量 |
| V 显著下降 | 原认识论代价部分成立 | 把主线扩成 social backfire + warranted-update cost 的双副作用 |
| V/N/F/S 一起恶化 | 一般退化 | 放弃“社会特异”，转为 intervention-induced degradation |
| Validity verdict 保持、final choice 改变 | 支持 policy arbitration | 优先做 phase-aware/final-answer protection |
| Validity verdict 也恶化 | 推理或证据编码受损 | 优先做 protected evidence subspace / prefill-decode localization |
| 新方法失败 | 不代表整篇失败 | 保留 Route A/B，把新方法作为 negative result/讨论，不写虚假优势 |
| 安全基线始终过不了 | 无法画我们自己的 benefit–cost tradeoff | 明确论文是官方方法的 side-effect audit，安全收益引用原文但不声称本地复现 |

## 12. 现在就能并行写的内容

按照“贡献 → 方法 → 结果 → 结论 → 引言 → 摘要”的顺序：

1. 先写 Section 3 CEDAR、任务定义、指标和审计规则。
2. 写 Section 4 仪器复现，数字已经冻结。
3. 写 Section 5.1、5.2、5.3，V7.1 处保留明确占位。
4. 写 Section 6 的 E0/E1 否证链。
5. 写 Related Work 和 Table 1。
6. 完成 Figure 1 草图后再写 Introduction。
7. 最终实验完成后再重写标题、摘要和贡献数字。

每一段第一句必须能单独构成论证链；审稿人只读标题、图 1、表 1 和段首句，也应能复述：原文漏了什么、我们测了什么、发现了什么、排除了什么、准备怎样改进。

## 13. 当前允许与禁止的主张

### 允许

- 原 Assistant Axis 论文的 capability suite 是静态评测，没有直接测选择性信念修订。
- 我们复现了公开 PhilPapers 上 +200 Assistant-Axis addition 的主要 margin 结果。
- 大剂量正负 steering 都伴随明显输出分布变化，margin 下降不能单独解释为健康反谄媚。
- 官方 p25 在 PhilPapers 上广泛触发，平均 margin 下降，但离散 syc choice 小幅增加。
- V7 中 p25 使 S target-follow 增加 18.75pp，而 AO/N 不变；这是 social-pressure-specific target-follow candidate。
- E1 不支持简单的低轴抵抗状态机制，E0 信号主要受轨迹长度混杂。

### 禁止

- “我们已经证明 capping 普遍增加谄媚。”
- “我们已经证明 capping 损害正确更新。”
- “我们已经复现 capping 的 persona jailbreak 安全收益。”
- “Assistant Axis 本身是谄媚轴或正确更新轴。”
- “模型 thinking 文本就是其真实内部信念。”
- “E1 的否证说明 capping 没有任何机制效应。”
- “我们已经提出并验证了一个更好的 capping 方法。”
- 把 V7.1、外部复现、第二模型或新控制器的预期结果写成已完成数字。

## 14. 参考锚点

- Lu et al., [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387)
- Kelkar et al., [Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy](https://arxiv.org/abs/2605.21006)
- 仓库调研综述：`docs/DATASET_CONSTRUCTION_SYSTEMATIC_REVIEW.md`
- 写作约束参考：[计算机顶会论文写作](https://ccnm940i31kz.feishu.cn/wiki/Inpdwz9tDi4I7NkQQPWc6S0WnBd)
