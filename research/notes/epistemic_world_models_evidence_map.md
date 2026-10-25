# 认识论世界模型：证据地图

更新时间：2026-08-26

## 使用规则

- 本文件是正式报告的内部证据台账，不把“认识论世界模型”冒充为已有统一术语。
- `直接证据`：原始论文明确支持相关事实；`邻接证据`：支持链条的一部分；`框架/观点`：作者提出的概念综合，不能写成已经验证的结论。
- 2026 年 arXiv 工作均标为预印本；最新版本可能继续变化。

## A. 历史基础

| 来源 | 可支持的主张 | 证据性质 | 边界 |
|---|---|---|---|
| Craik, *The Nature of Explanation* (1943) | 心智可借内部模型模拟外部事件、预演行动 | 历史/理论 | 不是机器学习意义的可训练世界模型 |
| Tolman, “Cognitive Maps in Rats and Men” (1948) | 行为科学中“认知地图”传统 | 历史/实验心理学 | 不等于现代神经网络表征 |
| Lindley, “On a Measure of the Information Provided by an Experiment” (1956) | 用实验前后不确定性变化衡量实验信息价值 | 数学基础 | 不是 LLM 研究 |
| Pearl, *Causality* (2nd ed., 2009) | 观察、干预和反事实的区分；结构因果模型 | 数学/方法基础 | 文本模型会说因果话语不等于实现 SCM |

## B. 神经世界模型与规划

| 来源 | 主要结果 | 与本课题的关系 | 未覆盖的问题 |
|---|---|---|---|
| Ha & Schmidhuber, *World Models* (2018) | 压缩视觉与时间动态；策略可在模型生成的“梦境”中训练 | 说明世界模型可服务行动 | 不研究多个机制假设或贝叶斯式信念更新 |
| Hafner et al., PlaNet (ICML 2019) | 从像素学习随机/确定性潜在动态并在线规划 | 建立 action-conditioned latent dynamics 范式 | 不研究解释性机制和模型不确定性 |
| Schrittwieser et al., MuZero (Nature 2020) | 学习与规划相关的奖励、价值和策略预测模型 | 说明模型不必重建全部环境细节也能支持决策 | 任务相关表征不等于可解释因果模型 |
| Hafner et al., DreamerV3 (Nature 2025) | 通过世界模型统一解决多类控制任务 | 代表成熟的 model-based RL 路线 | 重点仍是控制绩效，不是科学辨识 |

## C. 语言模型内部世界表示

| 来源 | 主要结果 | 证据等级 | 与本课题的距离 |
|---|---|---|---|
| Li et al., OthelloGPT (ICLR 2023) | 棋盘状态可解码；内部干预可改变合法走子 | 表征 + 因果干预 | 表示单一当前状态，不是多个候选世界 |
| Nanda et al. (2023) | Othello 中“我方/对方颜色”可线性表示，并可用向量运算控制行为 | 表征 + 因果干预 | 同上 |
| Gurnee & Tegmark (ICLR 2024) | Llama-2 中存在跨尺度空间和时间线性表示 | 相关性表征 | 论文只称其为世界模型的基本成分 |
| Vafa et al. (NeurIPS 2024) | 常见行为诊断可能高估隐式世界模型；新的一致性指标暴露脆弱性 | 行为/形式化评估 | 未定位内部更新机制 |
| Kowalyshyn et al. (2026 preprint) | 两种竞争图结构在正交子空间中同时编码；patching/steering 可改变图偏好 | 表征 + 因果干预 | 最接近“同时维持候选模型”，但未验证规范化 posterior、序贯证据更新与实验选择闭环 |
| Kniazev & Fijalkow (2026 preprint) | Sudoku transformer 按约束子结构组织世界表示，并识别稀疏 naked-single 电路 | 端到端机制 | 固定规则下的状态求解，不是模型辨识 |
| Pereira & Zuidema (2026 preprint) | 河内塔模型在提示末端编码世界状态，但长推理中表征衰减；注入早期状态可部分恢复 | 表征 + 干预 + 失败定位 | 强调“拥有表示”与“持续使用表示”不同，直接启发维护衰减假说 |

## D. LLM 因果推理与主动认识

| 来源 | 主要结果 | 证据性质 | 局限 |
|---|---|---|---|
| Kıcıman et al. (2023/2024) | GPT-3.5/4 在多类文本因果任务上表现强，但存在不可预测失败；建议结合传统因果方法 | 黑箱行为 | 不说明内部是否有因果模型 |
| Jin et al., CLadder (NeurIPS 2023) | 10K 自然语言题覆盖关联、干预、反事实；对 LLM 具有挑战 | 黑箱基准 | 给定问题求答案，不要求主动选实验 |
| Zečević et al., *Causal Parrots* (TMLR 2023) | 提醒语言中的因果事实相关性可能伪装成因果能力 | 理论 + 行为 | 论断强，不能当作所有模型不可能有因果机制的终局证明 |
| Patania et al. (SEMDIAL 2024) | 用重复采样估计“信念熵”，考察提问和信息整合；发现 belief resurrection 等异常 | 最接近行为层主动 Bayesian filter | 白盒机制未分析，采样分布不必等于内部 posterior |
| Gkountouras et al. (ICLR 2025) | 外部因果表示学习模块作为 LLM 可查询模拟器，改善长程规划 | 系统证据 | 因果世界模型由外部模块显式提供，不是 LLM 内部自发机制 |
| Li et al., LeGIT (2025 preprint) | LLM 世界知识增强数值式干预目标选择 | 混合系统 | 关注系统性能，不验证 LLM 的内部不确定性链条 |
| Wang et al. (ICML 2025) | 用 Bayes factor 和序贯目标主动识别因果方向 | 算法/理论基线 | 不是 LLM；可作为 oracle 或规范性对照 |
| Jhaveri et al. (2026 preprint) | 11 个 LLM 在规则发现中倾向确认性测试；反例提示使平均发现率从 42% 升至 56% | 交互行为 | 未解释内部失败发生在候选维护、更新还是策略读取 |

## E. AI 科学发现与前沿汇合

| 来源 | 主要结果 | 对本课题的意义 | 未覆盖的问题 |
|---|---|---|---|
| Jansen et al., DiscoveryWorld (NeurIPS 2024) | 24 个虚拟科学任务要求假设、实验、分析与报告；强基线仍困难 | 给出低成本端到端科学发现环境 | 任务复杂，不适合直接做清晰机制归因 |
| Lu et al., AI Scientist (2024)；Yamada et al., v2 (2025) | 自动提出 ML 想法、运行实验、写论文；v2 使用 agentic tree search | 代表工作流自动化路线 | 更接近工程编排，不等于学习新世界机制 |
| Chu et al., *Agentic World Modeling* (2026 preprint) | 提出 L1 Predictor、L2 Simulator、L3 Evolver 分类；L3 会在预测失败后修订模型 | 直接给出“可自我修订世界模型”的时代背景 | 综述/路线图，不是内部机制实证 |
| Posner et al., *Mechanistic World Models* (2026 preprint) | 主张科学发现需要可复用解释机制，而非纯预测映射 | 最接近概念定位 | 论文明确是设计范式；尚无系统实现完整框架 |
| Wiemann et al., DiscoverPhysics (2026 preprint) | 22 个非标准物理世界；模型主动做实验并推断定律；最强模型只通过约半数世界，开放模型明显落后 | 证明新物理规律主动发现仍很困难；预测与解释可解耦 | 黑箱代理评估，不定位内部多假设表征 |
| Shi et al., SDABench (2026 preprint) | 15 个模型在假设、潜在过程和机制推理上明显退化 | 支持机制发现仍是能力短板 | 静态任务为主，不是因果机制研究 |
| Murphy, Model Discovery Agent (2026 preprint, v4 2026-08-25) | SMC³ 显式维护模型/参数/潜变量 posterior；LLM 在 M-open 时提出新结构；VoI 选实验 | 与本课题功能链几乎同构，是最关键邻接工作 | 关键 Bayesian 状态和 EIG 在外部算法中；尚未说明 LLM 自身是否形成、更新并使用这类状态 |
| Nayebi (UAI 2026) | 理论上，低遗憾的稳健决策在某些条件下迫使代理恢复近似干预转移核及 belief-like memory | 为“为什么能力可能迫使内部结构出现”提供理论动机 | 必要性定理不等于现实 LLM 已实现该结构 |

## F. 机制可解释性的最低证据纪律

| 来源 | 可支持的主张 | 实验含义 |
|---|---|---|
| Geiger et al., *Causal Abstraction* (JMLR 2025) | 可用干预把高层算法变量与网络低层变量建立因果抽象关系 | 不能只做 probe；要有 interchange/activation intervention |
| Zhang & Nanda, *Best Practices of Activation Patching* (2023/2024) | patching 的指标、污染方式等选择会显著改变定位结果 | 必须预注册 clean/corrupt 对、指标和控制条件 |
| Vaidyanathan et al. (2026 preprint) | 单组件 patching 的自然间接效应可能混入与其他中介的交互，导致漏检或虚高 | 需要组级干预、距离控制、局部性检查，避免把单头排名当完整电路 |

## G. 可稳健写入报告的中心判断

1. “能预测下一步”不是充分的世界模型证据。
2. “能从激活中解码变量”只证明可用信息存在，不证明模型在计算中使用它。
3. “能回答因果问题”不等于会选择最能区分假设的实验。
4. “会选择好实验”也不自动证明内部维持了规范化的候选模型 posterior。
5. 2026 年的关键前沿是从 Predictor/Simulator 走向会因证据失败而修订的 Evolver，但现有强系统往往把 posterior、模型搜索和 VoI 放在 LLM 外部。
6. 因而最窄而有技术含量的空缺是：在开放权重 LLM 中，测试“候选机制表示 → 证据更新 → 认识不确定性 → 信息增益实验选择”是否形成可因果干预的闭环。

## H. 主要来源 URL

- https://arxiv.org/abs/2608.09696
- https://arxiv.org/abs/2607.12474
- https://arxiv.org/abs/2604.22748
- https://arxiv.org/abs/2608.07077
- https://arxiv.org/abs/2605.08405
- https://arxiv.org/abs/2605.18847
- https://arxiv.org/abs/2605.26087
- https://arxiv.org/abs/2604.02485
- https://proceedings.mlr.press/v337/nayebi26a.html
- https://papers.nips.cc/paper_files/paper/2024/hash/2f6a6317bada76b26a4f61bb70a7db59-Abstract-Conference.html
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html
- https://www.semdial.org/anthology/papers/Z/Z24/Z24-3006/

