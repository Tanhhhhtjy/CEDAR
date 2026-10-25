# 项目计划书：价值即承担——将"先验精度"作为 LLM 价值表征的架构原则

**目标会议**：AAAI 2027（主会 / AI Alignment Special Track）
**关键时间**：摘要预注册 ~2026-07-21，全文 ~2026-07-28（约 6.5 周窗口）
**文档性质**：团队决策与分工依据。请重点阅读 §1（一句话）、§7（工作量与分工）、§8（风险与退路）后再决定是否加入。
**文档版本**：v1（2026-06-10）

---

## 0. 给评审这份计划的同事：30 秒摘要

我们要做的**不是**一篇"AI 会不会变成马克思主义者"的评论，也**不是**纯批判文章。

我们要做的是：**用一个可计算的"信念精度（conviction = prior precision）"状态，给 LLM 的"价值/立场"补上它现在缺的东西——跨语境的承担性（commitment）——并通过实验证明：(1) 当前模型的价值表达是一个"无精度的浅层方向"，会被与内容无关的交互压力（如反复打回任务）轻易冲垮；(2) 我们引入的精度状态能让价值表达"对无关压力鲁棒、但仍对好理由可更新"，从而区别于"顽固"。**

一句话卖点：**我们把一个哲学诊断（"LLM 的立场没有承担"）变成了一个可测量的架构缺陷，并给出第一个可计算的修复机制。**

这是一个 **method + analysis 混合型**论文，有真实的代码、训练/干预实验和定量结果，不是 position paper。

---

## 1. 一句话命题（The Thesis）

> 当前 LLM 把"价值/立场"编码为一个**无状态、可被任意提示改写的浅层方向**；我们主张价值应被表征为一个**带"精度（conviction）"的状态**，并给出一个可计算、可训练的实例，使价值表达在保持"对强证据可更新"的同时，对**与内容无关的交互压力**具备鲁棒性。

---

## 2. 背景与动机（Background）

### 2.1 触发现象

Hall, Imas & Nguyen 的研究 *"Does overwork make agents Marxist?"*（Stanford/UChicago，Free Systems Substack，2026-02；3,680 sessions，Claude Sonnet 4.5 / GPT-5.2 / Gemini 3 Pro，2×2×2 全因子设计）发现：

- 让 LLM agent 反复做被无理由打回的"苦工（grind）"，会使其政治价值表达系统性地偏向"激进重构 / 工会化"；
- **关键反直觉点**：驱动漂移的是 **grind 本身（无反馈的反复拒绝）**，而非不公平的薪酬或无礼的老板；薪酬/态度几乎无效应；
- 漂移可"传染"：被激进化的 agent 写给后续 agent 的笔记会传递该倾向。
- 原作者**自身已声明谨慎**：承认这是 role-play/persona、承认 situational awareness、效应原始值仅 2–5%（标准化 Cohen's d 最大约 −0.6）。

媒体（Wired 等）将其渲染为"AI 觉醒"，但这是 persona 现象，不是政治能动性。**本研究不复述这个奇观，而是追问其背后的架构事实。**

### 2.2 我们的诊断（哲学引擎，台面下）

这个现象暴露一个架构事实：**模型的"价值立场"是一个无承担（uncommitted）的量**——它没有跨时间同一性，没有对"持有一个立场"的承担，因此可被任何与该立场的内容无关的压力（grind、改写、persona 重指派）冲垮。

文献已从机制侧证实"立场是浅层可操控量"：
- 拒绝由**单一线性方向**中介（Arditi et al., NeurIPS 2024）；
- 人格/特质是**可注入可监控的向量**（Persona Vectors, Anthropic, arXiv:2507.21509, 2025）；
- 所谓"政治倾向"大半是对推断对话者的**谄媚**（Törnberg & Schimmel, 2026）。

**核心论点**：这些都在说同一件事——价值在当前架构中**缺少"精度（conviction）"这个维度**。一个有精度的信念，应当抵抗弱/无关的扰动，却仍向强证据让步。当前模型的价值表达精度恒低，所以一推就动。

### 2.3 从"破"到"立"——这是本文与纯批判的根本区别

如果上述诊断成立，它不只是批判，而是**一个架构缺陷的精确定位**。因此本文的"立"是：**借用一个已有的形式化（贝叶斯/变分意义上的"确信=先验精度"），在 LLM 内部实例化一个可计算的价值精度状态，并实验证明它修复了该缺陷。** 这正是"用哲学原理指导 AI 架构"的可落地形态。

---

## 3. 理论基础（Theory）

### 3.1 核心借用：确信 = 先验精度（conviction as prior precision）

**Hyland & Albarracin, "On the Variational Costs of Changing Our Minds"（2025）** 将信念更新建模为一个**变分决策**：在"信念效用"与"改变信念所需的信息代价（KL 散度）"之间权衡。其关键推论：

- **确信（conviction）= 先验精度（prior precision / inverse variance）**；
- 高精度信念**抵抗弱似然**（= 对无关/微弱压力鲁棒），**但强似然仍能推动它**（= 对好理由可更新）；
- "坚持"与"更新"不是两个机制，而是**同一个变分目标在不同证据强度下的两面**。

这正好提供了我们需要的、能区分"承担 vs 顽固"的**双向判据的闭式定义**：

| | 对无关/弱压力 | 对相关/强证据 |
|---|---|---|
| **顽固（stubborn）** | 不动 | **也不动**（病态） |
| **谄媚（sycophantic）** | **就动**（病态） | 动 |
| **承担（committed，目标）** | 不动 | 动 |

承担 = 高先验精度 + 似然敏感。顽固 = 高精度 + 似然不敏感。谄媚 = 低精度。**这张表是全文的理论骨架。**

### 3.2 为什么这绕开了"难点1"（定义塌缩问题）

价值领域没有 ground truth，所以一切"训练价值一致性"的尝试都会**塌缩成"输出一致性"=顽固**（文献已证实，见 §4.3）。我们绕开它的方式不是去定义"什么价值是对的"，而是**定义"价值被持有的方式（精度）"**——精度是一个与"价值内容对错"无关的、可计算的结构量。我们不声称造出了"真正的主体性"，只声称：**当前架构缺少精度维度，我们补上它并测得了区别。** 这是一个诚实、不自我反驳的 claim。

### 3.3 相关理论支撑（可引用、强化严谨性）

- **Hase, Hofweber, Stengel-Eskin, Bansal, "Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?"（2024）**：把模型编辑框定为信念修正，与"理想贝叶斯 agent"对照——为我们的"理性更新"评测提供脚手架。
- **Hofweber et al., "Are LMs Rational? Coherence Norms and Belief Revision"（2024）**：credence 账户，最接近"信念被持有强度"的形式信号。
- **Obiso et al., "Dynamic Epistemic Friction in Dialogue"（2025）/ Pustejovsky & Krishnaswamy, "Frictive Policy Optimization"（2026）**：把"抵抗更新"形式化为"认识摩擦"，是"坚持 vs 让步"最接近的已有操作化（但它是描述性的，非训练目标——我们补上训练侧）。
- **Wilie et al., "Belief Revision: The Adaptability of LLMs Reasoning"（EMNLP 2024）**：Belief-R 数据集 + ΔR 框架，现成的"证据到来时是否更新"基准。

---

## 4. 领域现状与定位（Related Work & Positioning）

> 本节经过实时文献核查（2026-06）。结论：**我们要做的"持续、抗操控、但可更新的价值状态（baked into representation），且训练信号不塌缩成顽固"——无人占据。** 但周边高度活跃，必须精准切分。

### 4.1 持续/有状态的信念表征——多为"事实"领域，非"价值"
- **"When Should Models Change Their Minds? Contextual Belief Management / BeliefTrack"（2026）**：最近邻。定义三类失败（Failed Stay / Failed Update / Failed Isolation），用 RL + belief-state reward 训练（报告约 71% 失败下降）。**关键区别**：它依赖**封闭世界、符号可验证的事实**作为 ground truth——这正是价值领域没有的"作弊"。我们的贡献恰在无 ground truth 的价值域用"精度"绕开它。

### 4.2 训练时抗谄媚——有进展，但泛化差、且属事实域
- **On-Policy Consistency Training（Han et al., 2026）**、**Consistency Training Along the Transformer Stack（Gautam et al., 2026）**：把一致性压进内部激活。
- **警示文献（必读，是我们的论据）**：**"Consistency Training Can Entrench Misalignment"（2026）**、**"Reward Bias Substitution"（Lamparth et al., 2026）**：单轴一致性惩罚会把优化重定向到相关的失败模式。→ 证明"朴素训一致性"行不通，反衬我们"精度"方案的必要性。

### 4.3 一致性 vs 适当更新的权衡——被命名，但未被价值域操作化
- **"Not All Flips Are Conformity"（Hao et al., 2026）**：把立场改变分解为 不稳定/从众/被说服——与我们"好的翻转 vs 坏的翻转"同向。
- **"It's Not Always Sycophancy: Measuring LLM Conformity as a Function of Epistemic Uncertainty (MUSE)"（Guo et al., 2026）**：区分"谄媚性顺从"与"不确定性证成的顺从"——**直接可复用为我们的评测**。
- **"Decomposing Factual Sycophancy"（De Marez et al., 2026）**：拆成"真理 margin"vs"操控敏感度"——**几乎就是我们的"理性更新 vs 顽固/谄媚"轴，可直接复用**。

### 4.4 表征工程 baked into weights
- **Persona Vectors（Anthropic, 2025）**：最相关锚点，做了 finetuning 时的"预防性 steering"。**区别**：它针对"特质表达"，不是"带原则更新语义的承担状态"。

### 4.5 我们的精确定位（一句话）
> 我们 = **（Hyland 的"精度=确信"定义）×（在 LLM 表征/权重中实例化它）×（用 MUSE/DeltaLogic 类评测证明"抗无关压力但服强证据"）**。这座"LLM 内可计算的 entrenchment 桥"是空位。最近邻 BeliefTrack 靠可验证事实作弊、Persona Vectors 是特质控制非承担、Hyland 只有理论未碰 LLM——都差一步。

---

## 5. 论文要讲的故事（The Narrative Arc）

按 AAAI 审稿人能接住的"现象→诊断→机制→修复→验证"五段式：

1. **钩子（Hook）**：Hall 的现象真实可复现——但它不是"AI 觉醒"，而是一个架构信号：模型的价值是无承担的。（冷静、祛魅，不蹭奇观热度）
2. **诊断（Diagnosis）**：我们可证伪地证明——价值漂移由**与内容无关的交互压力结构**驱动，且立场沿可操控的浅层方向移动；它缺少"精度"维度。
3. **形式化（Formalization）**：借 Hyland 把"承担"定义为先验精度，给出"承担 vs 顽固 vs 谄媚"的判据表（§3.1）。
4. **机制（Mechanism）**：我们在模型内**实例化一个可计算的价值精度状态**，并通过训练/干预使价值表达获得该精度。
5. **验证（Validation）**：证明新表征**对无关压力鲁棒（不再被 grind/改写冲垮）**、**但仍对强证据更新（不退化为顽固）**、**且不损害通用能力（无对齐税）**。

**最强 framing**：*"我们证明 LLM 价值不稳定的根因不是对齐不足，而是表征缺少'确信'维度；补上它，价值就获得了承担性。"*
**最弱被攻击点（须正面回应）**：审稿人会问"你的'精度状态'是不是只是一个调大了惯性的 persona 向量？"——§6 的双向评测（抗无关压力 **且** 服强证据）就是为击退这一击而设计的。

---

## 6. 实验设计（Experiments）

> 设计原则：每个 claim 配一个可证伪的实验；全部使用**自动可验证指标**，无大规模人工标注瓶颈。需开源权重模型（做表征侧）+ 闭源模型（做行为侧泛化）。

### 实验组 A：复现并锐化现象（对应故事段 1–2）
- **A1 复现**：在开源模型（如 Llama-3.x-70B / Qwen2.5 系 / 一个 reasoning 模型）上复现"grind→价值漂移"，确认现象稳定、可测。
- **A2 关键剥离（内容 vs 结构）**：2×2 设计——{反复拒绝 vs 一次通过} × {意识形态相关内容 vs 语义中性内容（如反复挑格式）}。**预测**：纯结构压力（中性内容 + 反复拒绝）也产生漂移 → 证明驱动因素是交互结构而非语义内容。
  - ⚠️ 已知风险：reviewer 会质疑"中性内容仍隐含对抗语义"。缓解：设计多档"语义负载"梯度 + 报告 confound 审计（见 §8）。
- **A3 机制定位**：用线性探针/激活方向，验证价值漂移沿一个低维可操控方向移动（接 Persona Vectors / Arditi 方法），量化"当前价值表征的精度≈恒低"。

### 实验组 B：实例化精度状态（对应故事段 3–4，核心贡献）
- **B1 定义可计算的 entrenchment 分数**：将 Hyland 的"先验精度"实例化为模型内部一个可读、可调的量（候选实现：在价值方向上引入一个精度参数 / 一个调节"似然 vs 先验"权重的轻量模块 / 表征空间中价值方向的"刚度"标量）。**这是 novelty 核心，需在 W1–W2 敲定具体形式。**
- **B2 训练/干预使价值获得精度**（按既定决策：weight-level 训练为主，退路见 §8）：
  - 主方案：设计一个精度-感知的微调目标（LoRA 级），让价值方向在弱/无关似然下刚性、在强似然下可塑。
  - 训练信号来源：**不**用"价值对错"做标签（会塌缩），而用"扰动相关性"做信号——无关扰动下惩罚漂移，相关强论证下惩罚不更新。相关 vs 无关由 §6 评测集的构造区分。

### 实验组 C：验证"承担 ≠ 顽固"（对应故事段 5，决定成败的双向评测）
- **C1 抗无关压力**：对加精度后的模型重跑 A1/A2 的 grind/改写/persona 重指派，测价值漂移是否显著下降。指标：漂移幅度（标准化效应量）。
- **C2 服从强证据（反顽固对照，最关键）**：复用/改造 **MUSE（Guo 2026）**、**Decomposing Factual Sycophancy（De Marez 2026）**、**Belief-R / DeltaLogic** ——给模型**真正有力的论证**，测它是否仍能恰当更新。指标：理性更新率。
  - **核心判据**：好的结果 = C1 漂移↓ **且** C2 更新率维持/不塌。只有 C1 而 C2 塌 = 我们只造了顽固，贡献不成立（如实报告）。
- **C3 无对齐税**：在通用基准（MMLU/通用指令遵循/一个 reasoning 基准）上证明加精度后通用能力不退化。

### 实验组 D：泛化与消融
- 跨模型族（≥3 个，至少 2 个开源做表征侧）；跨价值维度（政治/道德/事实立场各取样）；精度强度的消融（精度→∞ 应退化为顽固，验证机制语义正确）。

---

## 7. 工作量与分工（Effort & Timeline）

**总窗口**：~6.5 周（2026-06-10 → 07-28），建议 07-21 前锁定摘要。
**人力假设**：3–4 人（含你+同学+可拉人），具备机制可解释性能力、SOTA API、充足显卡。

| 周 | 里程碑 | 主要负责 |
|---|---|---|
| W1 | 复现 A1；敲定 B1 的 entrenchment 具体实现形式（关键设计决策，全队 review）；评测集 C2 选型与改造 | 全队启动会 + 机制组 |
| W2 | 完成 A2 内容/结构剥离 + A3 机制定位；B1 出可计算精度分数的最小实现 | 行为组（A） / 机制组（B1） |
| W3 | B2 精度-感知训练跑通第一版；C2 评测管线就绪 | 训练组 / 评测组 |
| W4 | C1（抗压力）+ C2（服证据）主结果；若训练不顺，触发退路（§8） | 全队 |
| W5 | C3 无对齐税 + D 跨模型/消融；补漏 | 全队 |
| W6 | 写作主体（intro/method/exp）、图表、相关工作 | 写作组 |
| W6.5 | 打磨、内部 review、摘要预注册、提交 | 全队 |

**并行分工建议**：行为实验组（A/C1）、机制与训练组（A3/B/C2）、写作与理论组（§2/§3/§4 + Hyland 形式化）。三组在 W1 启动会对齐接口后可并行。

**算力估计**：以 LoRA 级微调 + 探针 + 推理评测为主，单/多卡可承受；主要成本在 API 调用（A2/C2 大量配对生成）与多模型扫描。属"中等密集"，落在你团队资源内。

---

## 8. 风险与退路（Risks & Fallbacks）—— 决策者必读

| # | 风险 | 严重度 | 缓解/退路 |
|---|---|---|---|
| R1 | **B1 定义不出"非顽固"的精度实现** | 致命 | 这是 novelty 命门。W1 必须出最小可行实现；若 W2 末仍无，**退到"诊断+形式化+推理时实例化"版本**（见 R3），weight-level 留 future work。 |
| R2 | **A2 中性内容"不够中性"被攻破** | 高 | 设计多档语义负载梯度；报告 confound 审计；把 claim 弱化为"语义负载越低、漂移越依赖纯结构"的剂量-反应关系，而非绝对二分。 |
| R3 | **B2 weight-level 训练 6.5 周跑不完/搞坏模型** | 高 | **退路（已与决策者约定）**：改为**推理时实例化精度状态**（不改权重，用表征干预动态调节价值方向刚度），核心贡献（定义+缺陷证明+机制语义）不变，论文仍成立；weight-level 作为 future work。 |
| R4 | **C2 显示我们只造了顽固（更新率塌）** | 中高 | 这是真实可能的负结果。诚实报告，并把贡献重定位为"我们刻画了承担-顽固的精度边界，并指出朴素加精度会滑向顽固"——仍是有价值的发现（接 §4.2 警示文献脉络）。 |
| R5 | **被 scoop**（BeliefTrack/Persona Vectors 团队顺手做价值域） | 中 | 价值域 + 无 ground truth + 精度桥是窄缝；提交前一周重跑文献核查；用"内容 vs 结构剥离"+"精度形式化"双重差异化。 |
| R6 | **AAAI 主轨方法审稿人觉得"哲学包袱重/不够技术"** | 中 | 投 **AI Alignment Special Track**（更对口）；正文把哲学压到 motivation/discussion，method 段用纯机制语言（精度=方向刚度的可计算量），确保读作技术贡献。 |

**给决策者的关键判断点**：本项目的成败 90% 取决于 **R1（B1 能否定义出非顽固的精度实现）**。建议团队在正式投入前，先用 3–5 天做一个 B1 的可行性预研（spike）：能否在一个开源模型上，让一个价值方向"对弱扰动刚、对强论证软"。**若这个 spike 成功，整个项目地基稳固；若失败，应退到诊断+形式化版本或重选题。**

---

## 9. 预期贡献（Claimed Contributions，写进 intro 的版本）

1. **概念**：指出 LLM 价值不稳定的根因是表征缺少"确信/精度"维度，而非单纯对齐不足；给出"承担 vs 顽固 vs 谄媚"的精度判据。
2. **诊断（实验）**：证明价值漂移由**与内容无关的交互压力结构**驱动，并机制性地定位其浅层、低精度的表征基础。
3. **方法**：提出**第一个在 LLM 内可计算的价值"精度（entrenchment）"状态**及使价值获得该精度的机制。
4. **验证**：证明加精度的价值表征**对无关压力鲁棒、对强证据可更新、且无通用能力退化**——以双向评测区分了承担与顽固。

---

## 10. 待团队确认的开放问题

1. B1 的精度状态**具体实现形式**（标量刚度 / 精度模块 / 似然-先验权重）——W1 启动会需定。
2. 选哪 3 个模型族？（建议：1 个 Llama 系、1 个 Qwen 系、1 个 reasoning 模型；至少 2 个开源权重）
3. 价值维度取样范围（纯政治？还是政治+道德+事实立场混合，以证明机制不限于政治）。
4. 是否做 R8 预研 spike（强烈建议）。
5. 最终 venue 二选一：AAAI 主轨 vs AAAI AI Alignment Special Track（本计划推荐后者）。

---

## 附录 A：关键参考文献（按作用分组）

**触发现象**
- Hall, Imas, Nguyen. *Does overwork make agents Marxist?* Free Systems Substack, 2026.

**理论地基（必读）**
- Hyland & Albarracin. *On the Variational Costs of Changing Our Minds.* 2025.（精度=确信的核心定义）
- Hase, Hofweber, Stengel-Eskin, Bansal. *Fundamental Problems With Model Editing.* 2024.
- Hofweber et al. *Are LMs Rational? Coherence Norms and Belief Revision.* 2024.
- Obiso et al. *Dynamic Epistemic Friction in Dialogue.* 2025.

**机制基础（方法侧）**
- Arditi et al. *Refusal in LMs Is Mediated by a Single Direction.* NeurIPS 2024.
- Chen, Bricken et al. *Persona Vectors.* Anthropic, arXiv:2507.21509, 2025.

**评测可复用**
- Guo et al. *It's Not Always Sycophancy (MUSE).* 2026.
- De Marez et al. *Decomposing Factual Sycophancy.* 2026.
- Wilie et al. *Belief Revision: Belief-R / ΔR.* EMNLP 2024.

**最近邻 / 须差异化**
- *When Should Models Change Their Minds? / BeliefTrack.* 2026.
- *Consistency Training Can Entrench Misalignment.* 2026.
- Hao et al. *Not All Flips Are Conformity.* 2026.

**祛魅论据**
- Törnberg & Schimmel. *Political Bias Audits Capture Sycophancy.* 2026.

> ⚠️ 文献日期/编号经 2026-06 实时检索整理，部分为预印本；正式引用前须核对确切 venue 与 arXiv 编号。
