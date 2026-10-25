# “认识论世界模型”选题的完整背景与空白审计

**审计日期：2026-09-02**  
**当前拟议问题：开放权重语言模型是否在内部维护多个候选因果机制，随证据更新其相对可信度，并因果地利用这一内部状态选择高信息增益实验？**

## 0. 结论先行

这个方向不是一块无人进入的空白地带。更准确的判断是：

1. **宽泛版本已经不新。** “LLM 会不会维护信念”“会不会根据证据更新假设”“会不会主动询问”“能否用 EIG 选问题”“能否在微世界里做科学发现”“内部信念会不会影响输出”，这些子问题均已有直接研究。
2. **各关键环节已被不同研究分别覆盖。** Bayesian belief-state 表征、竞争信念的因果干预、行为层的假设更新、EIG 驱动的主动提问、交互式科学发现微世界、外部 Bayesian 模型发现代理，都已有很强的近邻工作。
3. **目前仍未找到把整条链放在同一个预训练开放权重 LLM 中严格验证的工作：**
   
   > 精确的候选因果机制 posterior → 内部表征随证据更新 → 该表征因果地控制下一次 epistemic experiment → 行动与外部 EIG oracle 一致。

4. 因而，这不是“完全空白”，而是一个**窄而高风险的机制连接缺口**。它的创新性不来自任何单个零件，而来自对“内部 posterior 到实验策略”的因果闭环做第一次干净、可证伪的测量。
5. 这个缺口正在快速收窄。2026 年 2 月的 HOT-3 工作已经报告竞争信念形成并因果影响行动；2026 年 5 月的 *Belief or Circuitry?* 已报告模型同时编码竞争图结构并可通过 patching/steering 改变图偏好；ICLR 2026 的 BED-LLM 已把 LLM、belief update 和 EIG 行动接到一起，只不过 posterior/EIG 是外部算法实现。因此项目必须比旧方案更精确。

建议把论文中心命题收缩为：

> **Posterior-to-Policy Causal Bridge：预训练开放权重 LLM 中，对候选因果机制的内部相对可信度是否作为中介变量，因果控制用于辨识机制的下一次实验？**

不要使用“首次发现 LLM 有信念”“首次让 LLM 做 Bayesian 实验设计”“首次构建科学发现微世界”等表述。

---

## 1. 必须先区分五种常被混称为“belief”的对象

| 类型 | 例子 | 与本项目关系 |
|---|---|---|
| 隐藏状态 belief | HMM/POMDP 中对当前不可见状态的分布 | 是重要理论先例，但层级较低；我们关心生成机制本身 |
| 事实或命题 belief | 两个互相冲突的事实解释、真假命题 | HOT-3、ACL 2026 等已覆盖；不是我们的最终对象 |
| 候选模型 posterior | 对因果图、方程或机制身份的概率分布 | **本项目真正目标** |
| confidence / epistemic stance | 确信、怀疑、无知、混淆等状态 | 可作为辅助变量，不能替代候选模型 posterior |
| 外部 agent belief state | 由 Python、SMC、RAG、记忆模块显式保存的假设和概率 | 系统功能上很接近，但不能证明 LLM 内部形成同样计算 |

如果不做这一区分，审稿人很容易用已有的 belief-state、uncertainty 或 agent-memory 论文否定“新概念”的主张。

---

## 2. 研究谱系：这个问题从哪里来

### 2.1 世界模型传统

从 Craik 的内部模型思想，到 model-based RL 中的 *World Models*、PlaNet、MuZero 和 Dreamer，主线问题通常是：模型能否以压缩内部状态预测环境演化并支持行动。LLM 研究中，OthelloGPT 表明 transformer 可以形成可解码、可干预的棋盘状态；随后空间、时间、图结构等内部表示工作进一步支持“生成模型会形成结构化世界状态”。

但**能预测一个世界**不等于**在多个可能世界之间维护不确定性**。本项目关注的是更高一层的 belief over world models，而不是单一 world state。

需要特别重视 [Evaluating the World Model Implicit in a Generative Model（NeurIPS 2024）](https://papers.nips.cc/paper_files/paper/2024/hash/2f6a6317bada76b26a4f61bb70a7db59-Abstract-Conference.html)：它说明常见行为测试可能把局部预测能力误判为全局一致的世界模型。我们的任务必须测试跨表述、跨变量、跨历史和干预一致性，不能只看一次答对。

### 2.2 Bayesian 模型选择与实验设计传统

Bayesian experimental design 的规范目标是选择最大化预期信息增益的实验：

\[
\operatorname{EIG}(e)=I(M;Y\mid e,D_t),
\]

其中 \(M\) 是候选机制，\(D_t\) 是已有证据，\(Y\) 是实验结果。这个数学问题并不新；新问题只能是：**预训练 LLM 是否在没有收到外部 posterior/EIG 数值的情况下，内部近似实现这一计算链。**

### 2.3 机制可解释性传统

OthelloGPT、belief-state geometry、activation patching、steering、causal mediation 等工作已经把证据标准从“可解码”推进到“因果使用”。本项目必须遵守同样标准：

- probe 成功只能说明信息可读取；
- steering 改变输出可能是非特异扰动；
- patching 只有在匹配对照、方向特异性、跨模板复现和行为中介成立时，才支持“该状态被策略使用”。

[Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) 对度量、corruption 和方法选择的敏感性给出了直接警告。

---

## 3. 最接近的六条研究线

### 3.1 交互式科学发现与微世界基准

这一部分已非常拥挤。

- [BoxingGym](https://arxiv.org/abs/2501.01540) 构建了十个由概率生成模型定义的交互环境，让 LLM 设计实验、观察结果、发现与修订模型，并直接使用 EIG 衡量实验设计质量。它已经占据“概率微世界 + LLM 实验设计 + EIG 评估”的大部分空间。
- [DiscoveryWorld（NeurIPS 2024 Datasets and Benchmarks）](https://papers.nips.cc/paper_files/paper/2024/hash/13836f251823945316ae067350a5c366-Abstract-Datasets_and_Benchmarks_Track.html) 提供端到端科学任务，要求智能体形成假设、实验和解释。
- [NewtonBench（ICLR 2026）](https://openreview.net/pdf?id=Gk6umqW74m) 包含 12 个物理领域、324 个交互式规律发现任务，并通过改变经典规律抵抗训练记忆捷径。结果显示复杂性和噪声迅速破坏发现能力，工具使用还可能导致过早满足。
- [DiscoverPhysics](https://sampsonml.github.io/DiscoverPhysicsLeaderboard/) 让代理在 22 个隐藏物理世界中设计 N-body 初始条件并输出规律，强调预测拟合与概念理解可能分离。
- [CausalGame](https://causalgame.github.io/) 把选择偏差、测量误差、隐藏混杂等放入 14 个交互情景；即便较强代理完成任务，真正获得因果推理 rubric 分数的会话仍很少。
- [Are Large Language Models Reliable AI Scientists?](https://arxiv.org/abs/2505.17968) 以程序、形式语言和方程黑箱测试系统反向工程，证明干预可以帮助模型摆脱过度复杂化和遗漏。
- [LLM-AutoSciLab / ActiveSciBench](https://arxiv.org/abs/2605.24043) 已形成“假设生成—条件化实验选择—机制修订—记忆”的闭环系统，并包含酶动力学和基因调控网络因果发现任务。

**判定：**“建立一个合成微世界，让 LLM 主动做实验发现规律”绝对不是空白。新的微世界只有在为内部机制识别提供了现有基准没有的**可辨识性设计**时才有贡献。

### 3.2 行为层的多假设更新与证伪

- [Hypothesis Generation and Updating in LLMs](https://arxiv.org/abs/2605.05851) 用 Tenenbaum number game 从 posterior prediction、candidate evaluation、hypothesis generation 三种行为测量更新。模型呈现近似 Bayesian 但有系统偏差，并出现“评估—生成鸿沟”。
- [Failing to Falsify](https://arxiv.org/abs/2604.02485) 用交互式 Wason 规则发现测试 11 个模型家族，显示模型偏好确认性而非证伪性实验；提示干预能改善但不能消除问题。
- [In-context learning agents are asymmetric belief updaters](https://arxiv.org/abs/2402.03969) 发现 LLM 对好于预期和差于预期的反馈更新不对称，且这种偏差受 agency framing 影响。
- [Thinking Under Uncertainty](https://arxiv.org/abs/2607.26845) 用两臂 bandit 的认知模型分解 value choice 与信息寻求，发现更长思考改善价值选择但不一定改善探索。
- [MT-InfoSeek](https://arxiv.org/abs/2608.14808) 在 5,251 个多轮约束问题上测量“问什么、何时问、是否使用答案”，显示模型常低估缺失信息、无法找到最小充分问题集并过早停止。

**判定：**“LLM 会不会随证据更新多个假设”在行为层已经被直接研究。我们的增量必须来自精确 posterior 对齐、内部表示和因果使用，而不是再做一次准确率表。

### 3.3 主动提问、信息增益和 Bayesian experimental design

- [Entity-Deduction Arena（ACL 2024）](https://aclanthology.org/2024.acl-long.82/) 用 20 Questions 测试多轮状态追踪和提问规划。
- [Adaptive Elicitation of Latent Information Using Natural Language](https://arxiv.org/abs/2504.04204) 用前向模拟估计问题如何降低潜在实体的不确定性，并在 20 Questions、意见调查、学生评估上比较 greedy、information gain 和 MCTS。
- [Uncertainty of Thoughts（NeurIPS 2024）](https://papers.nips.cc/paper_files/paper/2024/file/2b0e14abd8128e6bf98b6b0bec1cfcbf-Paper-Conference.pdf) 已把不确定性模拟和信息增益奖励用于提问规划。
- [BED-LLM（ICLR 2026）](https://openreview.net/pdf?id=qyylZMLYT8) 是最重要近邻：它用 LLM 产生/评估 belief distribution，迭代选择最大 EIG 的问题，并做 sample-then-filter 更新。
- [BALAR](https://arxiv.org/abs/2605.05386) 显式维护潜变量 belief、按 mutual information 选澄清问题，还可动态扩充状态空间。
- [CA-BED](https://arxiv.org/abs/2606.01182) 将对话历史纳入 Bayesian experiment design 和 LLM likelihood estimation。

**判定：**“让 LLM 选高 EIG 问题”也不是空白。BED-LLM 几乎实现了外部功能闭环。我们的区别必须写成：BED-LLM 把 belief 和 EIG 显式算法化，而我们测试**模型内部是否自发形成并使用功能对应物**。

### 3.4 LLM 内部 belief/world-state 表征

- [Transformers Represent Belief State Geometry in Their Residual Stream（NeurIPS 2024）](https://papers.nips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html) 表明在 HMM 序列上训练的小 transformer 的 residual stream 线性表示精确 Bayesian belief state，甚至重现其分形几何。它是“内部 Bayesian state”最强理论先例之一，但对象是固定生成器的隐藏状态，不是候选因果机制，也没有主动实验选择。
- [Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in LLMs](https://arxiv.org/abs/2602.02467) 是最大的直接威胁。它在 Llama-3 70B、Gemma-3 27B 等模型上量化竞争信念，发现外部操纵会改变信念形成，对隐藏信念的干预会改变最终行动，模型还能一定程度报告自身信念。它已经否定“首次证明内部信念驱动行动”的广泛主张。
- [Belief or Circuitry? Causal Evidence for In-Context Graph Learning](https://arxiv.org/abs/2605.08405) 在两个竞争图结构的随机游走任务中发现模型同时在近正交子空间编码两种图拓扑，patching 和 graph-difference steering 能转移图偏好。这与“多个候选因果图的内部表征”极接近。
- [LLM Beliefs Are in Their Heads（ACL 2026）](https://aclanthology.org/2026.acl-long.1905/) 报告命题真假在 residual stream/attention heads 中的可读性、一定一致性，以及 steering 对行为的影响。
- [Tower of Hanoi world-model maintenance](https://arxiv.org/abs/2608.07077) 显示开放大模型可在提示结束时形成可解码、因果相关的世界状态，但在长推理中丢失；重新注入提示时状态可恢复表现。这提示我们的 posterior 也可能不是“没有形成”，而是“在推理/行动阶段丢失”。
- [Epistemic State Representations in Large Language Models](https://www.preprints.org/manuscript/202608.2226) 于 2026-08-31 刚发布，报告 15 种认识状态在五个开放模型中形成 confidence 主轴和 self/world uncertainty 次结构，并可通过 steering/patching 改变行为。它未经同行评审，且研究的是信心与混淆，不是候选机制 posterior；但术语空间已被占用。

**判定：**“内部存在可干预 belief/epistemic state”也已经不是空白。尚可能成立的是：把内部连续状态**定量对齐精确因果模型 posterior**，追踪其顺序更新，并证明它中介了 EIG 实验策略。

### 3.5 外部 Bayesian 模型发现代理

- [Model Discovery Agent](https://arxiv.org/abs/2608.09696) 把 LLM 提案器、结构/参数/潜变量上的 SMC、simulation-based inference 和 value-of-information 实验设计接在一起；预测检验失败时可进入 M-open 新假设生成。功能流程与我们的理想闭环高度同构，但关键 epistemic machinery 位于 LLM 外部。
- [A Probabilistic Framework for LLM-Based Model Discovery / ModelSMC](https://arxiv.org/abs/2602.18266) 同样由 LLM 提出/修订机制模型，再用 SMC 按 likelihood 加权。
- [Mechanistic World Models](https://arxiv.org/abs/2607.12474) 提出仅靠预测不足以构成科学发现，主张可复用、可组合的机制模型；它更接近研究纲领而非完整实证。
- [Evidence-Informed LLM Beliefs for Continual Scientific Discovery](https://arxiv.org/abs/2606.29182) 通过检索和历史发现更新外部 belief/prior，说明静态 surprise 会制造大量伪发现。

**判定：**系统层的“科学 agent 维护机制分布并用 VoI/EIG 选实验”已经存在。我们不应与其比赛系统效果，而应把它当作明确的功能参照：外部算法做得到，单个预训练 LLM 内部究竟有没有同构状态？

### 3.6 真实实验优化与 AI for Science

[GOLLuM：Large language models as uncertainty-calibrated optimizers for experimental discovery](https://doi.org/10.1038/s42256-026-01283-z) 于 2026-08-28 发表在 *Nature Machine Intelligence*。它把 GP 的概率目标与语言模型表征联合优化，在 23 个化学/材料任务上进行不确定性驱动实验选择。它说明“LLM + uncertainty-guided experimental discovery”已经进入高水平期刊，但它研究外部优化性能和表征适配，不是未经训练的 LLM 是否内部维护候选机制 posterior。

**判定：**不要把项目包装成一个更小、更合成的 AI-for-Science 优化器。我们的优势只能是机制可辨识性和因果证据。

---

## 4. 重合矩阵：已有工作究竟覆盖到哪一步

符号：✓ = 明确覆盖；△ = 部分或近似覆盖；— = 未覆盖。

| 工作 | 多候选机制 | 精确 posterior oracle | 顺序证据更新 | 主动实验/EIG | 内部读出 | 内部因果干预 | 预训练开放 LLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Belief-state geometry | — | ✓ | ✓ | — | ✓ | △ | —（任务训练小模型） |
| HOT-3 belief-guided agency | △（竞争语义 belief） | — | △ | —（答案行动） | ✓ | ✓ | ✓ |
| Belief or Circuitry? | ✓（竞争图） | △（混合比例） | △ | — | ✓ | ✓ | △ |
| Hypothesis Generation & Updating | ✓ | ✓/△ | ✓ | — | — | — | ✓/黑箱行为 |
| BoxingGym | ✓ | ✓ | ✓ | ✓ | — | — | ✓/黑箱行为 |
| BED-LLM | ✓/△ | ✓（外部） | ✓ | ✓ | — | — | ✓/系统组件 |
| LLM-AutoSciLab | ✓ | △（外部状态） | ✓ | ✓ | — | — | ✓/系统组件 |
| Model Discovery Agent | ✓ | ✓（SMC 外部） | ✓ | ✓（VoI） | — | — | ✓/系统组件 |
| **本项目严格版** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

这个矩阵给出最重要的事实：**我们的行不是凭空出现的新范式，而是把多篇工作的不同列连接成一条受控因果链。** 这既是潜在贡献，也是审稿风险。若实验只完成其中四五列，论文很容易被视为已有工作的简单拼接。

---

## 5. “是否真空白”的分层判定

| 可声明内容 | 新颖性判断 | 建议 |
|---|---|---|
| 提出“认识论世界模型”这个术语 | 低；术语不稳定且已有相近用法 | 仅作为工作定义，不作为贡献 |
| 测试 LLM 是否维护多个假设 | 低 | 已有 number game、竞争 belief、竞争图结构 |
| 测试 LLM 是否根据证据更新 | 低 | 行为研究很多 |
| 让 LLM 主动问问题/做实验 | 低 | EDA、BoxingGym、NewtonBench 等已覆盖 |
| 用 EIG 评估问题质量 | 低 | BoxingGym、BED-LLM、Adaptive Elicitation 已覆盖 |
| 从激活中解码 belief/confidence | 中低 | belief geometry、HOT-3、ACL 2026 等已覆盖 |
| 在预训练 LLM 中解码**精确候选因果模型 posterior** | 中高 | 找到近邻，但尚未找到严格等价工作 |
| 证明内部 posterior 随诊断证据做定量、方向正确的更新 | 中高 | 比单次 belief 读出更窄、更强 |
| 证明 posterior 表征因果改变**辨识实验**而非最终答案 | 高 | 当前最可辩护缺口 |
| 证明内部 posterior 对实验选择的中介作用并对齐 EIG oracle | 很高但高风险 | 最适合作为主贡献 |

最终答案：

> **不是“真空白”；是一个多条成熟研究线交叉处、目前尚未找到完全闭合实证的窄缺口。**

学术上这并不比“空白”差。真正好的机制论文经常不是发明一个无人听过的主题，而是指出已有证据链缺少哪一个决定性环节，并用比前人更严格的识别设计补上它。

---

## 6. 为了保住创新性，实验必须如何改写

### 6.1 研究对象

不要只用“两个故事哪一个更可能”或固定根节点干预。需要一个可计算的机制集合：

- 3–6 个候选 DAG、结构方程或离散生成程序；
- 多个候选在初始观察数据下相同或近似 observationally equivalent；
- 每一步证据都有精确 likelihood，因此外部 oracle 可计算 posterior odds；
- 同一个世界、同一组实验中，最优实验会随 posterior、噪声和成本改变；
- 候选文本、标签、变量名、选项位置全部随机化。

### 6.2 决定性识别：matched posterior-policy pairs

构造两段历史 \(D_A,D_B\)：

- 表面长度、变量、可选实验和最终可见事实尽量匹配；
- 但它们对候选机制的 posterior odds 不同；
- 因此 EIG 最优实验相反。

然后验证：

1. 模型选实验是否随 oracle posterior 改变；
2. 内部激活能否跨模板、跨变量名解码 log posterior odds / entropy；
3. 把 \(D_A\) 的 posterior state patch 到 \(D_B\) 后，实验选择是否按预测翻转；
4. 这种翻转是否在保持事实内容、答案倾向和通用置信度方向不变的对照下成立；
5. posterior state 是否统计中介 evidence → experiment choice 的效应。

这比“patch 后答案变了”强，因为它测试的是**认识状态到认识行动**的特殊连接。

### 6.3 必须排除的替代机制

- **固定启发式：**总选根节点、覆盖最多节点或文本最长的选项；
- **单一 MAP 猜测：**只选当前最可能模型，不维护剩余不确定性；
- **答案 token 路径：**patch 只改变候选标签或下一 token 偏好；
- **一般 confidence：**读到的是“确信/不确信”，不是候选机制相对概率；
- **显式算术复述：**模型在文字里复述概率，但策略并未读取该状态；
- **外部 oracle 泄漏：**posterior/EIG 只能用于评估，不得写进主提示；
- **probe illusion：**信息可解码但并未被下游计算使用；
- **patching artifact：**替换造成分布外激活或总体困惑。

### 6.4 最小主张与强主张

最小主张：

> 在一类受控微世界中，开放权重 LLM 的中间激活包含与候选机制 posterior 对齐且随证据更新的信息，该信息被下一次实验选择部分使用。

强主张只有在跨世界、跨模板、跨模型、多个证据长度和多种干预下成立时再写：

> 预训练 LLM 实现了一个功能性的 epistemic world-model circuit，将模型不确定性转化为主动辨识行动。

---

## 7. 40B 与多家 API 对项目意味着什么

### 7.1 它们提高的是证据强度，不自动提高新颖性

40B 开放模型让我们能验证这种机制是否只存在于小模型，或是否随规模出现更稳定的 posterior-policy coupling。ChatGPT、Claude、Qwen4 API 可以建立行为能力上界/参照，帮助回答“内部失败是开放模型能力不足，还是普遍的认识缺陷”。

但闭源 API 无法提供 residual stream、attention head 或 activation patching，所以不能承担论文的机制主张。

### 7.2 推荐的资源分层

| 层次 | 用途 |
|---|---|
| 7B/14B 开放模型 | 大规模任务筛选、逐层扫描、probe、粗 patching、消融 |
| 32B/40B 开放模型 | 只在已定位层段和关键 matched pairs 上做确认性干预 |
| ChatGPT/Claude/Qwen4 API | 行为 benchmark、科学发现能力参照、提示与 CoT 条件比较 |

40B 全层全 token 缓存和大规模 patching 仍然昂贵。建议先在 7B/14B 找到稳定现象，再对 40B 做小规模、预注册式确认，而不是一开始就对 40B 穷举层×token×head。

### 7.3 可形成的实验结构

1. **Behavioral landscape：**3 家闭源 API + 3–4 个开放规模，测 posterior calibration、EIG regret、证伪率、停止过早等。
2. **Mechanism study：**对 7B/14B 做完整定位和 causal intervention。
3. **Scale confirmation：**在 32B/40B 上只复验最关键的 2–3 个因果结果。
4. **External scaffold control：**与显式 BED/MDA 式 posterior + EIG oracle 对照，区分“模型本身不会”与“外部结构可以补齐”。

---

## 8. 技术博客、作者帖子与论坛给出的额外信号

这些材料不能替代论文证据，但能暴露社区正在关心什么、哪些失败模式会被审稿人追问。

- Ai2 的 [Evaluating Scientific Discovery Agents](https://allenai.org/blog/evaluating-scientific-discovery-agents) 强调科学代理不能只按最终答案评分，需要跟踪中间假设、实验和证据使用。
- LamaLab 的 [The Appearance Behind the Epistemic](https://lamalaborg.substack.com/p/the-appearance-behind-the-epistemic) 讨论一项超过 25,000 次 agent run 的研究：许多轨迹在语言上很“科学”，但证据并未真正约束结论。对应的主论文是 [AI scientists produce results without reasoning scientifically](https://arxiv.org/abs/2604.18805)。这与我们的动机高度一致：结果成功不能证明内部完成了科学推理。
- belief-state geometry 作者在 [Alignment Forum 技术解读](https://www.alignmentforum.org/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their) 中解释了为何 next-token prediction 可迫使模型表示 predictive sufficient statistics；这为“posterior 可能自然出现在 residual stream”提供理论直觉。
- 一项独立 [LessWrong 复现实验](https://www.lesswrong.com/posts/vzav5kfbRCDQyEB8v/toy-transformers-may-represent-belief-state-geometry) 发现已经失去预测价值的 belief 信息仍可能残留，提示“能解码”不代表模型保存的是最小充分 belief，更不代表它被行动使用。
- [How to use and interpret activation patching](https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching) 强调 corruption 选择会改变可得结论，并区分探索性定位与确认性因果测试。
- [Why LLMs Aren't Scientists Yet](https://www.lesswrong.com/posts/y7TpjDtKFcJSGzunm/why-llms-aren-t-scientists-yet) 总结一个多代理科学流程中常见的训练记忆偏置、实现漂移、上下文退化和过度宣称成功。它不是正式证据，但说明“科学语言外观”和“可靠认识过程”之间的差距已成为社区显著议题。
- [World Models for Scientific Discovery](https://richardcsuwandi.github.io/blog/2026/wm-discovery/) 把预测世界模型、机制模型、主动探究和 Model Discovery Agent 放到同一研究图景中，说明这个交叉方向正在快速形成共同语言。

社区信号的总体含义：大家已经不满足于“agent 跑通任务”，而是在追问证据是否真正改变了内部状态和后续行动。因此我们的机制闭环有时代性，但也会受到非常严格的因果证据审查。

---

## 9. 主要发表风险

### 风险 1：被判为 HOT-3 + BED-LLM 的拼接

审稿人可能认为 HOT-3 已证明 belief→action，BED-LLM 已证明 belief/EIG→question，本项目只是换成因果图微世界。

**应对：**强调并实证 posterior 的精确量化、顺序更新、matched posterior-policy flip、EIG action 和 causal mediation；任何一个都不能只停在定性图示。

### 风险 2：微世界过于简单，出现捷径

**应对：**最优实验必须随 posterior、噪声、成本而改变；做变量重命名、图同构、选项置换、叙事跨域、反启发式 split。

### 风险 3：微世界太复杂，模型行为接近随机

**应对：**设计分层任务：2 候选/3 变量作为机制定位层，3–6 候选/4–6 变量作为外推层。先确保有可研究的行为信号，再增加丰富度。

### 风险 4：probe 准确但 patching 不工作

这本身也是有价值结果：模型可能显式包含 posterior 信息，却由启发式行动头选择实验。论文应预先定义竞争机制，而不是只允许“存在认识论世界模型”一个成功故事。

### 风险 5：40B 机制实验成本失控

**应对：**小模型发现、大模型定点验证；使用统一 token 位置、少量层窗、低秩方向和成对样本，避免全组合扫描。

---

## 10. 最终立项建议

### 可以继续，但必须改名、改主张、改实验中心

建议暂定题目：

> **Many Worlds, One Experiment: Testing the Causal Bridge from Internal Model Beliefs to Information-Seeking Actions in Open-Weight LLMs**

或更保守：

> **Do Internal Beliefs over Causal Mechanisms Guide Experiment Selection in Language Models?**

建议把核心贡献限定为三条：

1. 一个为**posterior-policy 因果识别**而设计的微世界套件，而不是泛化的科学发现 benchmark；
2. 内部激活与精确候选机制 posterior/entropy 的跨表述定量对齐及顺序更新分析；
3. 通过 matched patching、steering、rescue 和 mediation，测试该状态是否因果控制 EIG 实验选择。

### Go / No-Go 门槛

在写完整系统或大规模跑 40B 前，先做一个 2–3 周 feasibility gate：

- 至少一个 7B/14B 模型在反启发式测试上显著优于随机选择；
- oracle posterior/entropy 能跨模板解码，而不是只在同模板内；
- 至少一个 matched pair 中存在方向正确、可重复的 patch-induced experiment flip；
- flip 不能由候选标签、一般 confidence 或输出 token 直接解释。

若前三项都失败，不要硬讲“世界模型存在”；可转为一篇更批判性的结果：**LLM 表面上会做信息性实验，但内部没有稳定的机制 posterior，或 posterior 并未进入策略。** 结合 2026 年“AI scientists produce results without reasoning scientifically”的大背景，这仍可能是有价值的机制性负结果。

---

## 11. 审计边界

本结论基于截至 2026-09-02 的会议/期刊页面、arXiv/预印本、作者项目页、代码仓库、技术博客与社区讨论。学术数据库无法证明“世界上不存在某篇等价工作”，因此严谨措辞应是：

> **在本次检索中未找到同时满足全部六项识别条件的既有工作；但每个子环节均有强近邻，故创新性属于窄交叉缺口，而非无人研究的空白。**

投稿前应再做一次以“causal model posterior / active experiment / activation patching / information gain / belief-guided action”为关键词的滚动更新，尤其关注 2026 年秋季 arXiv 和 NeurIPS/EMNLP/ICLR 2027 投稿。
