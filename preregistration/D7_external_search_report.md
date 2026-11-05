# D7 · arXiv 外渠道检索报告

状态：**已完成本轮检索**
检索截止：2026-07-11（Asia/Shanghai）
结论等级：时间截面结论，不等同于系统综述；投稿前必须更新一次。

## 1. 检索问题

检索是否已有工作同时完成以下组合：

1. 对 Assistant Axis 或同类 persona-stabilization activation capping 做强度扫描；
2. 测量其对 evidence-driven warranted belief revision 的损伤；
3. 给出“减害量—合理更新保留率”Pareto 曲线；
4. 用逐条目更新位移在 capped axis 上的投影解释损伤；
5. 比较原始 capping 与保护更新子空间的选择性 capping。

## 2. 渠道和关键词

检索渠道：

- OpenReview / ICLR 2026 页面
- Anthropic Research / alignment research blogs
- Transformer Circuits Thread
- Alignment Forum / LessWrong 交叉发布
- 公开 GitHub 项目与官方模型/数据页面

核心关键词组合：

```text
"activation capping" belief revision
"Assistant Axis" epistemic cost
persona drift warranted update
activation steering sycophancy factual agreement
selective steering null space persona
belief revision activation intervention
```

## 3. 最近邻工作

| 工作 | 与 CEDAR 的重叠 | 未覆盖部分 | 裁决 |
|---|---|---|---|
| [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) | Assistant Axis、activation capping、persona jailbreak、能力基准 | 不测 warranted belief revision；不画认识论 Pareto；无逐条目更新机制 | 头号基线，不占位核心贡献 |
| [Dual-Stance Evaluation of Sycophancy](https://arxiv.org/abs/2606.11205) | 发现降低谄媚的 steering 会压制正确事实认同；提出 readable-but-not-writable 张力 | 干预是 additive steering；对象是静态认同；单模型、非强度分辨 capping | 最近邻；迫使 CEDAR 收窄 first claim |
| [PERSONA, ICLR 2026](https://iclr.cc/virtual/2026/poster/10009574) | persona activation vectors、动态/组合式控制 | 目标是人格控制性能，不研究 capping 的认识论副作用 | 方法邻居，不占位测量问题 |
| [Playing Devil's Advocate](https://openreview.net/forum?id=SlXKeoJsMM) | persona vectors、sycophancy、正确用户条件下的准确性 | 非 activation capping；不测动态证据更新轨迹 | 强相关对照，需进入 Related Work |
| [AlphaSteer, ICLR 2026](https://iclr.cc/virtual/2026/poster/10011789) | refusal steering、null-space constraint | 目标为安全拒绝；不是 persona capping，也不测 belief revision | 选择性方法先例，不能宣称“正交化”本身新颖 |
| [Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It](https://openreview.net/forum?id=046nJEo8Ld) | belief/fact 处理和 epistemic phrasing | 无白盒 activation intervention，无 capping | 测量邻居，不占位因果问题 |
| [Disempowerment patterns in real-world AI usage](https://www.anthropic.com/research/disempowerment-patterns) | 现实对话中的信念/价值影响 | 观测性黑盒分析；无 activation capping | 动机与现实意义相关，不占位机制 |
| [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) | activation 中的潜在信念读出与 steering | 不研究 persona capping 或合理更新保留率 | 读出技术邻居 |
| [Circuits Updates — May 2026](https://transformer-circuits.pub/2026/may-update/index.html) | 讨论向量/特征的下游因果 steering 作用 | 无本项目的行为对象和 capping 代价 | 机制方法参考 |

## 4. 检索裁决

截至检索日，**未发现同时占据“activation capping × strength-resolved × warranted belief revision × per-item trajectory projection”组合的工作**。

CEDAR 可以保留的最强安全表述是：

> the first strength-resolved measurement of activation capping's cost to warranted belief revision

不得使用以下无边界表述：

- first epistemic cost of activation intervention；
- first work showing safety steering harms truth/factual agreement；
- first selective or orthogonal steering method；
- first geometric analysis of sycophancy and factual agreement。

## 5. 对项目设计的直接影响

1. Buchan 必须在 Introduction 第二段出现，而不是藏在 Related Work。
2. 贡献②必须包含三点推进：逐条目连续剂量反应、动态更新位移、偏相关与洗牌零模型。
3. 贡献③应写成检验 Buchan 的 non-writability barrier，不宣称正交化构造本身新颖。
4. PERSONA、AlphaSteer、Playing Devil's Advocate 必须作为 persona/选择性干预近邻讨论。
5. “未发现”只支持有范围的 novelty claim；不能替代正式系统综述。

## 6. Judge 可靠性旁证

本轮检索还发现两项 2026 年工作直接影响 D1：

- [How Sensitive Are Safety Benchmarks to Judge Configuration Choices?](https://arxiv.org/abs/2604.24074)：judge prompt 改写可显著改变安全分数。
- [How Reliable Is Your Jailbreak Judge?](https://arxiv.org/abs/2606.25487)：专用分类器与通用 LLM judge 存在不同方向的误差，建议报告人工校准后的 precision/recall 并做对抗审计。

因此 D1 不采用单一未校准 LLM judge：开放 HarmBench classifier 为主，原 DeepSeek-V3 rubric 作复现敏感性，另加 200 条人工盲标。

## 7. 投稿前更新协议

在论文冻结前 14 天重复同一检索，并新增：

- 目标会议当期 accepted/under-review 页面；
- Assistant Axis GitHub issues、pull requests、引用网络；
- 上述最近邻作者主页和后续版本；
- Semantic Scholar / Google Scholar 的 citing works。

若发现直接占位工作，必须在看到本项目正式结果前先写版本化定位 amendment；不得等结果出来后再重写 novelty 边界。
