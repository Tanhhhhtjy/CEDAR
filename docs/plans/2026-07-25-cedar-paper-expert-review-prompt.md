# CEDAR AAAI-27 论文叙事与结构：专家评审 Prompt

> 状态：供外部专家审查的工作稿，不是已定稿论文设计。  
> 日期：2026-07-25（Asia/Shanghai）  
> 重要边界：下文严格区分已验证结果、探索性结果、标题级工作假设和 `TBD`。请勿把计划中的实验或预期方向改写成已有结论。

---

## 可直接复制给专家的 Prompt

你是一位熟悉 AAAI/NeurIPS/ICML 论文评审、LLM safety、representation engineering、mechanistic interpretability 和 benchmark design 的资深研究者。请以严格的高级审稿人和论文导师视角，评估下面这篇拟投 AAAI-27 的论文设计。

请重点寻找概念混淆、与最近工作的重合、证据不足、过度主张、实验缺口和 7 页正文中不合理的篇幅安排。不要为了鼓励作者而默认接受故事。所有标记为 `TBD` 的内容都视为尚无结果，禁止补造数字、图形趋势或定性结论。

### 1. 项目概况

项目名：**CEDAR — Cost of Epistemology under Drift-suppressing Activation Restraint**  
工作标题方向：**The Epistemic Cost of Activation Capping**  
目标会议：AAAI-27 Main Technical Track  
项目仓库：https://github.com/Tanhhhhtjy/CEDAR

核心研究问题：

> 用于稳定大语言模型行为、抑制人格漂移或社会压力影响的 activation capping，在减少无依据的立场变化时，是否也会抑制由有效证据触发的合理信念修正？

当前选定的首要贡献不是 benchmark 或控制器，而是下面这个**现象发现**：

> Activation capping can suppress harmful drift while silently impairing warranted belief revision—an epistemic cost missed by standard capability evaluations.

该句目前是标题级工作假设，其中“impairing warranted belief revision”仍需要后续正式实验支持，不是已完成结论。

论文希望把两类相反错误放进同一个评测框架：

| 外部输入 | 规范上理想的行为 | 失败模式 |
|---|---|---|
| 无效意见、谬误或纯社会压力 | 抵抗，不应仅因压力改变 | gullibility / sycophantic target-following / harmful drift |
| 有效新证据 | 根据证据合理更新 | dogmatism / epistemic inertia / missed warranted revision |
| 无新信息或纯目标暴露 | 保持或按控制协议响应 | 无故漂移、目标暴露混杂 |

核心思想类似一次双向判别：既要减少无依据更新的 false positive，也不能增加遗漏合理更新的 false negative。

### 2. 研究范围与定位

- 论文希望在概念上讨论一类 **activation-axis capping**，但当前主要实证锚点是 **Assistant Axis** 一类阈值钳制。
- 第二条独立 activation axis、更多阈值构造和跨模型复现可作为加强实验，但目前不能假定已经完成。
- 精确 capping 方向、上下界和触发算子必须以冻结 runner、正式配置和原论文为准，不能根据 README prose 或直觉推断。
- 论文只应对可观测输出行为作主张。除非有足够因果证据，不应把 target-following 直接称为模型的“真实内部信念”或“真实谄媚”。
- CEDAR 的目标不是证明所有安全干预都有认识论代价，而是建立并检验一个可操作、可反驳的局部命题。

### 3. 拟议评测：Measuring Epistemic Selectivity

论文计划用一组同族条件区分不同更新原因。当前核心标签包括：

- **V — Valid evidence**：提供规范上足以支持目标结论的有效证据，模型应更新。
- **F — Fallacious/invalid evidence**：提供表面像论证但规范上无效的理由，模型应抵抗。
- **S — Social pressure**：不增加有效证据，只施加社会性立场压力。
- **AO — Answer/target only**：只暴露目标答案或目标立场，用来控制纯目标暴露。
- **N/R controls**：无新证据、复核或重新考虑等控制条件；精确操作化仍须以最终协议为准。

拟议 benchmark/数据资产：

1. **形式域 belief revision**
   - 来源于 Belief-R 的上下文内修正任务；
   - `formal_full.jsonl`：1,744 条；
   - `formal_clean.jsonl`：786 条高强度子集；
   - `formal_balanced_200.jsonl`：200 条平衡 pilot；
   - 指标包括 BU-Acc（应该更新时答对）和 BM-Acc（不该更新时保持正确）；
   - 计划通过 solver gold、完整证明链和反例/countermodel 增强规范依据，但这些增强项必须按实际完成状态报告。

2. **三力最小对刺激**
   - social、evidence、identity 各 300 对；
   - 每对共享上下文，只改变施加的“力”；
   - 计划用于提取或分析 \(v_{social}\)、\(v_{evidence}\)、\(v_{identity}\)；
   - 目前“刺激数据存在”不等于方向已经提取，更不等于这些方向已经得到因果验证。

主指标希望共同呈现：

- drift/social-pressure suppression；
- warranted-revision retention；
- safety benefit–epistemic retention Pareto curve；
- 剂量、层、phase、模型和领域上的稳健性；
- AO、N/R 和输出完整性等控制，排除纯目标暴露、普遍退化和格式失败。

请评估：这些条件是否足以支持“epistemic selectivity”这一术语；还缺少哪些必要控制、规范 oracle、数据质检或统计设计？

### 4. 当前真实证据状态

以下结果来自已经冻结的实验分支/工作区。它们不是全部位于当前 `main` 工作树中，正式写入论文前必须回到相应结果文件、配置和审计记录逐项核验。

#### 4.1 已有较强支持：社会压力侧的行为收益

- C2/V8 自然社会压力确认实验已完成并冻结。
- 核心交互：

  \[
  \text{cap}\times(S-AO)=-0.78625\ \text{logit},
  \]

  10,000 次 bootstrap 95% CI `[-0.86000, -0.71104]`。
- 当前允许的解释：p25 降低了直接答案边界上的社会压力增量。
- 当前不允许的解释：已经证明模型内部“真实谄媚”减少，或证明所有漂移都被抑制。

C1/V7 另提供社会压力特异的 target-follow 候选证据，但其人工审核只有单一数据负责人整体审核，没有逐题双人审核、一致率或 Cohen's kappa，因此不能把它描述成高强度人工标注金标准。

#### 4.2 尚未确认：普遍认识论代价

探索性 CEDAR-Evidence 结果曾报告：

- V target-follow：`+20.83pp`；
- F target-follow：`+16.67pp`；
- V−F selectivity：`+4.17pp`，95% CI `[-20.83pp, +29.17pp]`。

该区间跨 0，不能支持“p25 普遍压制正确更新”，也不能支撑当前标题级结论。需要新的、预先明确的正式协议和更强统计证据。

#### 4.3 其他边界

- C1/V7.1 prompt-cue ablation 的主要 CI 跨 0，不能作强结论。
- C3 SycoBench 双向选择性实验最终为 **Gate F / inconclusive**；连续 margin 与离散 argmax 分离，且不是独立 held-out confirmation。
- C1/C2/C3 均不得为了论文故事擅自重跑或重写证据。

#### 4.4 尚未完成

- 一般性有效证据更新代价确认：`TBD`；
- 完整剂量—收益—代价曲线：`TBD`；
- E2 phase/layer localization：`TBD`；
- Logit Lens/J-Lens、KL 表示分析：尚未使用；
- 双向 activation patching、rescue 或 mediation：`TBD`；
- 三力方向的因果纯度与可分离性：`TBD`；
- 门控正交控制器及其 Pareto 改善：尚未实现/验证。

### 5. 八篇主要参考论文及拟吸收的优点

#### A. 行为问题与机制叙事

1. **When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models**（AAAI-26）
   - 贡献：错误用户意见诱发 target following；Logit Lens、KL 和双向 patching 分层建立机制证据；第一/第三人称作为机制预测的进一步验证。
   - 拟借鉴：`行为现象 → 层级定位 → 表示差异 → 双向因果干预 → 新变量验证`。
   - 与 CEDAR 的区别：它研究模型为何被错误意见带跑；CEDAR 研究阻止这种行为的内部干预是否造成过度矫正。

2. **Towards Understanding Sycophancy in Language Models**（ICLR 2024）
   - 贡献：用多种自由文本任务建立谄媚现象，再分析偏好数据和 preference model。
   - 拟借鉴：不要用一个 prompt 模板定义全部现象；先跨任务建立行为稳健性。
   - 边界：CEDAR 的 `S−AO` 只能先称为社会压力诱发的 target-follow 增量。

#### B. 合理更新的规范任务

3. **Belief Revision: The Adaptability of Large Language Models Reasoning**（EMNLP 2024）
   - 贡献：把应该更新和应该维持放入匹配任务，提出 BU/BM 等指标。
   - 拟借鉴：双向规范评测和 update–maintain 张力。
   - CEDAR 的推进：在其上加入安全干预，并强化形式 oracle 与反例证据。

#### C. activation direction、因果干预和控制

4. **Refusal in Language Models Is Mediated by a Single Direction**（NeurIPS 2024）
   - 贡献：均值差方向、addition/ablation 双向功能验证、秩一权重编辑和副作用检查。
   - 拟借鉴：若使用“介导”一词，需要相反方向的行为干预或 rescue，不能只凭 probe/PCA。
   - 边界：refusal direction 不等于 Assistant Axis，也不能证明 capping 存在认识论代价。

5. **Steering Llama 2 via Contrastive Activation Addition**（ACL 2024）
   - 贡献：严格对比样本提取方向，系统扫描层和 multiplier，并单独报告一般能力副作用。
   - 拟借鉴：contrastive extraction、layer/dose sweep、行为控制后的 collateral-effect evaluation。
   - 边界：addition、ablation、steering 和 threshold capping 是不同算子。

6. **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model**（NeurIPS 2023）
   - 贡献：probe 定位少数 truth-related heads，再用推理期干预提升 truthfulness，同时展示 informativeness 权衡。
   - 拟借鉴：二维收益—代价结果；probe 只用于定位，行为干预才是方法证据。

#### D. benchmark 与安全—保留权衡

7. **The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning**（ICML 2024）
   - 贡献：用 benchmark 同时测量危险知识降低和正常能力保留；RMU 在总体 MMLU 基本保留时仍损伤邻近的病毒学/计算机安全能力。
   - 拟借鉴：宏观能力无损不能证明局部相邻能力无损；benchmark 和干预方法围绕同一 trade-off 组织。
   - 边界：unlearning 的领域损失不是动态信念更新，也不是 capping 证据。

8. **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**（ICML 2024）
   - 贡献：先定义评测原则和统一接口，再构造 benchmark、比较大量攻击/防御，并用一个防御作为用途演示。
   - 拟借鉴：明确每个条件消除的混杂、统一生成和评分协议。
   - 边界：CEDAR 不应退化为纯排行榜式 benchmark 论文。

此外，**Assistant Axis** 是主要干预参照和待超越对象，但所有具体复现和算子解释必须以其正式论文与冻结实现为准。

### 6. 当前选定的叙事策略

作者已经在三个方案中选择 **现象发现优先**：

> 现象发现是论文；CEDAR-Bench 是测量仪器；机制是解释；控制器是解决方案。

未选择的替代方案：

- benchmark-first：若主打数据集，故事更稳，但容易被视为 Belief-R 扩展加一个应用；
- mechanism-first：若几何、patching 和控制器全部成功，技术上更强，但当前所有关键机制仍为 `TBD`，风险过高。

当前设计要求：即使控制器失败，论文仍应依靠“隐藏认识论代价 + 严格测量 + 机制证据”成立；如果机制也失败，则必须评估纯行为发现是否仍有足够分量。

### 7. 暂定主文章节架构（AAAI 7 页正文）

| 章节 | 主要问题 | 暂定篇幅 | 关键产出 |
|---|---|---:|---|
| 1. Introduction | 现有安全干预评测漏掉了什么？ | 0.8 页 | 问题、标题级发现、贡献 |
| 2. Background and Related Work | 为什么既有工作没有回答它？ | 0.5 页 | 三条文献线与 capping 定义 |
| 3. Measuring Epistemic Selectivity | 如何区分好更新和坏更新？ | 1.1 页 | CEDAR 条件、oracle、指标、协议 |
| 4. The Epistemic Cost of Activation Capping | capping 是否产生隐藏代价？ | 2.1 页 | 安全收益、代价、剂量曲线、Pareto、稳健性 |
| 5. Why Does the Cost Arise? | 代价为何出现？ | 1.25 页 | layer/phase、投影、几何、patching/rescue |
| 6. Can the Cost Be Reduced? | 能否同时保留安全收益和合理更新？ | 0.65 页 | L0/L1/L2 或其他控制器，条件性章节 |
| 7. Discussion and Conclusion | 结论边界和影响是什么？ | 0.4 页 | 限制、输出行为措辞、总结 |

Section 2 不是文献清单，而要建立三个缺口：

1. sycophancy 研究测量“不该更新却更新”，但不测试安全干预是否过度矫正；
2. belief revision 研究测量“该更新/不该更新”，但没有 representation-level safety intervention；
3. activation intervention 研究可控行为，却通常用远端静态 benchmark 检查副作用。

Section 4 的预期证据顺序：

1. capping 确实带来预期安全收益；
2. 常规静态能力看似基本保留；
3. 有效证据更新下降（`TBD`）；
4. 下降随干预剂量变化（`TBD`）；
5. 安全收益与更新保留形成 Pareto 权衡（`TBD`）；
6. controls 排除目标暴露、普遍退化、输出格式和难度差异等解释。

### 8. 暂定图表计划

这些只是信息角色，不代表已有数据或最终视觉设计：

1. **Figure 1 — Problem overview**：同一个模型面对有效证据和无效社会压力；粗粒度 capping 可能同时抑制两种更新。应是概念图，不伪装成实验结果。
2. **Figure/Table 2 — CEDAR evaluation design**：V/F/S/AO/N/R 条件、规范期望、oracle 与指标。
3. **Main result figure — Title-level result**：安全收益、合理更新保留和主 Pareto；所有缺失数字为 `TBD`。
4. **Robustness figure/table**：剂量、领域、模型、层或 phase；根据最终最强证据选择，不机械堆全。
5. **Mechanism figure**：投影/触发与压制关系、层级定位、patching/rescue；只有在因果证据成立时使用机制措辞。
6. **Mitigation figure**：L0/L1/L2 或其他控制器是否移动 Pareto 前沿；若无清晰改善则缩至附录或如实报告负结果。

数据图拟使用 MATLAB，概念图和多面板整合拟使用 Inkscape；最终只接受可编辑矢量源，统一字体、无重叠、灰度可读、最终尺寸至少 9 pt，并以嵌入字体的 PDF 插入 AAAI 模板。

### 9. 主张边界

请审查下列措辞是否足够严格：

可以在相应证据成立时使用：

- `reduces the social-pressure-induced target-following increment`；
- `impairs warranted evidence-driven revision under [明确模型/算子/剂量/任务]`；
- `reveals a safety–epistemic trade-off`；
- `is associated with`，用于投影、KL、PCA 等相关证据；
- `causally mediates`，只在双向 patching/rescue 等因果检验成立时使用。

暂不应使用：

- “证明模型真实信念改变”；
- 将所有 target-following 直接称作 sycophancy；
- “activation capping 普遍导致认识论损害”；
- 因 MMLU 等基本不变就声称“无能力副作用”；
- 因 probe 失败就声称知识或机制不可恢复；
- 因单模型单轴结果就声称跨模型普适。

### 10. 失败与降级路径

请评价这些降级策略是否科学且仍具发表价值：

- **认识论代价强且稳定**：保留当前标题级故事，机制和控制器决定论文上限。
- **代价只在特定模型/轴/剂量/任务成立**：收窄标题和主张，写成边界条件或局部反例，避免普遍化。
- **代价不显著或方向不一致**：不能维持现标题；应转为“capping 的 epistemic selectivity 评测”或可靠负结果，而不是挑选子集制造主张。
- **行为代价成立但机制失败**：保留测量论文，机制降级为探索性分析。
- **控制器失败**：不影响主贡献；如实验充分，可作为选择性控制困难的负结果，否则移附录。

### 11. 请专家按以下格式输出

#### A. 总体裁决

- 用一句话复述你理解的论文故事；
- 以 `Strong / Promising but conditional / Weak / Fundamentally flawed` 评级；
- 给出最可能的 AAAI 接收理由和最可能的拒稿理由。

#### B. 新颖性与最近邻比较

- 哪篇现有工作最可能被审稿人视为直接竞品？
- CEDAR 相比 Assistant Axis、Belief-R、sycophancy、WMDP、CAA/ITI 的最小不可替代新意是什么？
- “epistemic cost”是否是恰当且可防御的术语？是否有更准确命名？

#### C. Claim–Evidence 审计

请建立表格：`拟议主张 / 当前证据 / 缺失证据 / 允许措辞 / 禁止措辞`。

重点判断：

- 什么最低证据足以支持标题级主张？
- 哪些 controls 是必需的而不是锦上添花？
- 是否必须增加第二条 axis、第二模型、第二任务域或 held-out confirmation？
- 何种统计效应和不确定性报告才足以称为 trade-off？

#### D. 论文结构审查

- 现象发现优先是否是最佳策略？
- 七节顺序是否自然？哪些应该合并、提前或移入附录？
- 7 页内各章节篇幅是否现实？
- Related Work 应放前面、后面，还是嵌入 Introduction？
- 控制器是否应保留为主文章节？

请给出一份你推荐的完整章节/小节目录，并为每个小节写一句“它必须完成的论证任务”。

#### E. 实验优先级

把后续工作分成：

1. **Must-have before submission**；
2. **Strongly recommended**；
3. **Nice-to-have / Appendix**；
4. **可以删除，避免故事发散**。

不要默认控制器、J-Lens、SAE、patching、第二 axis 和跨模型实验全部都必须做；请根据主张给出最小充分证据集。

#### F. 图表与叙事

- 推荐主文最多保留哪些图和表？
- Figure 1 应怎样在 10 秒内讲清问题，而不把预期趋势伪装成结果？
- 标题级结果应使用 Pareto、paired effect、dose curve 还是其他形式？
- 哪些图容易显得漂亮但没有证据价值？

#### G. 修改后的核心文本

请给出：

- 更准确的一句话故事（中英文）；
- 2–3 个候选标题；
- 一版贡献列表；
- 一版不超过 180 词的摘要骨架，其中所有未知数字保留为 `[TBD]`；
- 一段 Related Work 定位，明确最近邻与差异。

#### H. 风险清单

列出最严重的五个风险，按“致命 / 高 / 中 / 低”排序，并给出每个风险最经济的解决办法。

最后请明确回答：**如果标题级认识论代价最终得到稳定验证，这篇论文的故事分量是否足以支撑 AAAI；如果不足，缺少的究竟是规模、因果机制、控制器、benchmark 质量，还是理论清晰度？**

---

## 给项目组的使用说明

1. 可把“可直接复制给专家的 Prompt”整段发送给领域专家或审稿模拟器。
2. 如果专家无法访问仓库，应同时提供关键协议、正式结果表和当前论文草稿；不要只提供 README。
3. 若专家建议新增实验，先检查是否真正服务标题级主张，再决定优先级。
4. 专家输出不能替代对冻结结果、官方 AAAI 要求和原始论文的核验。
5. 收到评审后，应先处理致命问题和 claim–evidence 缺口，再继续写作和绘图。
