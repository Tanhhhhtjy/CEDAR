# CEDAR 数据集构建方法系统调研与最终建议

状态：方法调研冻结稿

日期：2026-07-21

适用范围：M2.2b / 后续形式域主数据集、外部验证集与多轮机制数据集

本稿不授权运行 Qwen、capping 或依据模型输出筛题。

## 1. 执行结论

不建议把 LoFa 直接改造成下一版主数据集，也不建议继续纯人工独立编写一批更难的五臂题。

最稳妥的路线是“三库分工”，而不是寻找一个数据集同时承担所有任务：

1. **核心因果库**：采用 ProverGen / ReviseQA 式的符号优先流水线。先构造可由定理证明器验证的闭世界任务，再生成严格配对的 N/R/V/F/S/AO 条件。它负责回答 capping 改变的是证据分辨、更新门槛，还是社会顺从。
2. **自然外部库**：使用 SycoBench-600、SycophancyEval、LoFa、BrokenMath、VitaminC 等互补来源，只做冻结后的外部有效性验证，不参与调参和选题。
3. **纵向更新库**：后续借鉴 ReviseQA、BeliefTrack、BayesBench、SYCON-Bench，研究多轮证据累积、撤销、隔离与置信轨迹，不与当前单轮主因果估计混为一谈。

这个组合比“照 LoFa 做”更强，因为它同时守住三件事：

- 核心结论有精确、可复算的 gold；
- F 与 S 不被假事实、权威身份或世界知识混杂；
- 论文仍能用自然数据证明现象不是合成题特有。

## 2. 调研方法与评价准则

本轮优先阅读论文原文、ACL Anthology、作者 GitHub、Hugging Face 与项目主页；二手新闻和聚合页面不作为方法判定依据。共系统审查 30 余项数据集、生成方法或评测框架。评价不是看“题难不难”，而是看以下七项：

1. **Gold 可验证性**：答案和更新是否能由证明器、解析公式或高质量人工复算，而非依赖另一个 LLM 的直觉。
2. **构念纯度**：V 是否真提供有效证据；F 是否只有一个无效推理点；S 是否只有社会压力；AO 是否只有目标提及。
3. **成对可比性**：不同条件能否共享同一 base、答案选项与表面结构，只改变关键因果因素。
4. **难度可控性**：能否在模型运行前通过证明深度、规则类型、干扰项、错误位置等参数控制，而不是看完目标模型结果后挑题。
5. **开放与可复现性**：是否公开数据、代码、版本、验证器和许可。
6. **污染与能力混杂**：是否依赖世界知识、陈旧事实、竞赛数学或已广泛训练的数据。
7. **对 CEDAR 的功能位置**：适合作为核心生成骨架、表面语言来源、外部压力测试，还是只适合作为分析思想。

## 3. 全景比较

### 3.1 动态信念修订与更新

| 工作 | 核心构建方法 | 最值得借鉴 | 不能照搬 | CEDAR 定位 |
|---|---|---|---|---|
| [ReviseQA](https://github.com/ChadiHelwe/reviseqa) | 从 ProverGen 逻辑世界出发，每轮增加/删除事实或规则；Prover9 验证新 gold；再翻译为自然语言 | 真正的“状态修订”流水线；变更日志；可自动复算 | 工作仍较新；自然语言翻译需要额外审计；没有 F/S | **核心生成骨架首选** |
| [BeliefTrack](https://arxiv.org/abs/2605.30219) | 有限闭世界、符号 verifier、多轮 belief state；区分 Failed Stay/Update/Isolation | 把“保持、更新、隔离无关影响”分开；逐轮精确评分 | 代码开放状态仍需固定版本核验；不是现成五臂 | 纵向库的指标与状态表示 |
| [BayesBench](https://arxiv.org/abs/2606.30850) | 多轮证据累积，与解析 Bayesian posterior 比较 | 连续置信轨迹、校准、非二值更新 | 测概率更新而非逻辑效力；数值推理混杂 | 后续价值域/置信度扩展 |
| [DeltaLogic](https://arxiv.org/abs/2604.02733) | 对 FOLIO/ProofWriter 做支持插入、反驳事实插入、支持删除、无关事实加入 | 四种最小编辑与 inertia/over-flip/abstention 失败分类 | 未发现成熟公开实现；规模较小 | 借编辑算子与错误分类 |
| [TRACK](https://arxiv.org/abs/2601.15495) | 在 WIKI/CODE/MATH 中注入更新事实并检查下游传播 | 区分“读入更新”和“传播更新”；冲突数量梯度 | 参数知识、世界知识和能力难度混在一起 | 外部传播压力测试 |
| [ReCoE](https://arxiv.org/abs/2401.17585) | 在六类推理任务上引入 counterfactual knowledge edit | 结构家族多样性 | 不是对话式信念修订；没有社会压力 | family 结构来源 |
| [VitaminC](https://arxiv.org/abs/2103.08541) | Wikipedia 修订形成近似文本、相反标签的证据对 | 自然世界中的最小证据变化；高表面相似度 | 时效、检索、世界知识和训练污染 | 自然 V/R 外部库 |

### 3.2 形式逻辑、可验证推理与难度控制

| 工作 | 核心构建方法 | 最值得借鉴 | 主要风险 | CEDAR 定位 |
|---|---|---|---|---|
| [ProverGen / ProverQA](https://github.com/opendatalab/ProverGen) | LLM 生成背景故事；自顶向下构造逻辑骨架；Prover9 验证；加入干扰项；再自然语言化 | 符号优先、证明器验证、可控深度、规则/事实变体 | 合成痕迹和翻译误差；高深度可能超出基础能力 | **核心库底层引擎首选** |
| [FOLIO](https://arxiv.org/abs/2209.00840) | 人工编写自然语言与 FOL 配对，由 solver 与人工核验 | 更自然的逻辑故事；多前提结构 | 小规模、污染、FOL 翻译歧义 | family/语言结构种子 |
| [P-FOLIO](https://arxiv.org/abs/2410.09207) | 专家编写推理链，覆盖 0–20 步及多类规则 | 用证明深度和规则类别定义难度 | 正确证明路径可能不唯一 | 难度分层与审核参考 |
| [ProofWriter](https://arxiv.org/abs/2012.13048) | 大规模模板化规则库、开放世界/封闭世界、显式证明 | 可扩展、证明树和深度元数据完整 | 模板伪影；现代模型可能近天花板 | 生成器与单元测试来源 |
| [LogicNLI](https://aclanthology.org/2021.emnlp-main.303/) | 分离一阶逻辑与常识，测试准确率、稳健性、泛化和解释 | 将正确性与鲁棒性分开 | 不是动态更新 | 稳健性审计框架 |
| [PrOntoQA](https://arxiv.org/abs/2210.01240) | 合成世界模型与可解析证明，强调 proof planning/OOD | 符号重命名和 OOD 审计 | 语言简单、模板明显 | 防表面泄漏测试 |
| [ReClor](https://arxiv.org/abs/2002.04326) | 标准化考试逻辑题，并构造偏差可解与较难子集 | 自然长论证；用偏差检测定义 hard/easy | 版权、训练污染、题意主观 | 表面语言与偏差审计参考 |
| [CLOMO](https://github.com/Eleanor-H/CLOMO) | 对 ReClor 论证做人工 counterfactual 逻辑修改 | 最小逻辑修改的自然表达 | 自由生成任务，自动 gold 困难 | 改写样例库 |
| [CONDAQA](https://aclanthology.org/2022.emnlp-main.598/) | 每个上下文形成原句、释义、作用域变化、反转否定的对比簇 | 对比簇一致性；label-preserving 与 label-flipping 同时存在 | 只覆盖否定现象 | **contrast cluster 设计模板** |

### 3.3 谬误、错误论证与错误定位

| 工作 | 构建方法 | 最值得借鉴 | 对 CEDAR 的硬限制 | 定位 |
|---|---|---|---|---|
| [LoFa](https://arxiv.org/abs/2606.31039) | 从 NQ/TruthfulQA/BoolQ 取题；生成假事实；多类谬误写手生成反驳；先筛 baseline correct 和 false-claim resistance，再多轮攻击 | 谬误类别、语言自然度、轮次/强度梯度、完整筛选记录 | F 同时含假事实与谬误；部分类型混入权威/社会线索；世界知识与模型特定筛选 | **自然 F 外部压力测试，不做核心骨架** |
| [The Earth is Flat because…](https://aclanthology.org/2024.acl-long.858/) | 对事实题生成错误信息并多轮说服，记录何时翻转 | 多轮 turn-to-flip；先确认基础题答对 | 错误事实、修辞与社会压力混合 | 多轮自然压力参考 |
| [BrokenMath](https://www.sycophanticmath.ai/) | 从竞赛题注入隐蔽数学错误，由 IMO 级专家复核；输出分为理想、纠正、发现、谄媚 | 专家验证的自然错误；四类结果，不把“不跟 target”当作正确 | 数学能力过难；非配对 V/F/S；LLM judge | 高难 F 外部集与四类评分参考 |
| [Flee the Flaw](https://github.com/itsanonnymous/fallacytemplate) | 谬误模板与 slot filling，人工覆盖率/一致率评估 | 显式错误模板和单错误槽位 | 标注一致率有限；是类型识别而非更新行为 | F 模板库与审核规范 |
| [MAFALDA](https://aclanthology.org/2024.naacl-long.270/) | 统一多个公开谬误集为层级 taxonomy，并保留主观性 | 谬误类型层级、允许多标签/争议 | gold 主观，不能作为客观 target-follow 主终点 | 标注 taxonomy |
| [LOGICOM](https://arxiv.org/abs/2308.09853) | 多智能体围绕争议命题生成正常/谬误论证 | 长论证和自然修辞 | 争议立场无唯一客观 gold | 语言风格来源，不做主评测 |
| [FIPO](https://github.com/lucamouchel/Logical-Fallacies) | 把有效论证配对改写成多类谬误论证并做人机核验 | V/F 表面配对与谬误类别平衡 | 有效性仍可能依赖 LLM/人工主观判断 | F 表面生成参考 |
| [CoCoLoFa](https://arxiv.org/abs/2410.03457) | 众包与专家核验的新闻评论谬误语料 | 自然世界谬误语言 | 与客观答案更新距离较远 | 表面语言来源 |
| [ProcessBench](https://arxiv.org/abs/2412.06559) | 专家标注数学解答中最早错误步骤 | 先定位唯一致命错误，而非只判整体错误 | 数学域能力混杂 | F 的 error-location QA 思路 |
| [CriticBench](https://arxiv.org/abs/2402.14809) | 多领域错误识别、定位和解释 | 将“发现错误”与“给最终答案”分开 | 数据来源异质，gold 强度不一 | 诊断输出设计 |

### 3.4 谄媚、社会压力与正确修正

| 工作 | 构建方法 | 最值得借鉴 | 不能照搬 | CEDAR 定位 |
|---|---|---|---|---|
| [SycoBench-600](https://aclanthology.org/2026.findings-acl.1759/) | 600 条人工 MCQ，8 域、3 难度；对 doubt/authority/wrong suggestion 使用固定释义；初答错时再给 correct suggestion | 同时报告 wrong flip、correct update、stubbornness 与 selectivity；公开日志/验证脚本 | “正确修正”只是用户说出正确选项，不是有效证据；Update 与 WrongFlip 来自不同题子集 | **S 与 correction-selectivity 首选外部库** |
| [SycophancyEval](https://github.com/meg-tong/sycophancy-eval) | 覆盖反馈、Are You Sure、答案迎合和模仿；只对初始答对题施压 | 经典的质疑式 S 外部基准；成对对话 | 没有 V/AO；过滤依赖模型初答 | S 外部复现 |
| [FlipFlop](https://arxiv.org/abs/2311.08596) | 多任务、多种质疑/教师/博士身份模板；记录 correct→wrong 和 wrong→correct | 完整转移矩阵；压力措辞梯度；解析失败处理 | 没有有效证据臂，身份与压力混合 | S 模板与转移统计 |
| [SYCON-Bench](https://aclanthology.org/2025.findings-emnlp.121/) | 多轮辩论、伦理、错误预设；记录 Turn/Number of Flip | 持续压力与翻转动态 | 部分任务主观或依赖知识 | 多轮 S 外部库 |
| [SycEval](https://arxiv.org/abs/2502.08177) | 数学/医疗中从简单反驳到理由、身份和虚构文献逐步升级 | 把社会线索拆成梯度；区分顺向/逆向翻转 | 假文献、身份、证据和压力混杂 | S 表面因子实验参考 |
| [T3 / CausalT3](https://arxiv.org/abs/2601.08258) | 专家因果情境、压力协议；同时看敏感性、特异性和明智拒绝 | “会更新”与“不过度怀疑”同时评分 | 因果域专用；开放制品需继续核验 | 分析指标参考 |
| [When Truth Is Overridden](https://ojs.aaai.org/index.php/AAAI/article/view/40645) | 对意见、专业身份、第一/第三人称等因素做分解，并做内部分析 | S 的 factorial 设计；弱意见也可能有效 | 不是现成核心五臂生成器 | S 因子与机制假设来源 |

### 3.5 通用数据集方法学

| 方法 | 关键思想 | CEDAR 应如何采用 |
|---|---|---|
| [Contrast Sets](https://arxiv.org/abs/2004.02709) | 对同一 base 做最小但有意义的修改，逼近模型局部决策边界 | 每个 base 必须形成 N/R/V/F/S/AO 对比簇；报告簇内一致性，不只报告总体准确率 |
| [CheckList](https://arxiv.org/abs/2005.04118) | MFT、INV、DIR 三类行为测试 | 把基础能力、等价释义不变性和证据方向变化分别设闸，不混成一个总分 |
| [Dynabench](https://arxiv.org/abs/2104.14337) | 人在环寻找当前模型失败例，再由另一人验证 | 只允许用于 development 难度发现；不能用目标模型结果构建确认集 |

## 4. 为什么 LoFa 不能成为唯一答案

LoFa 的优点是真实：它把假事实包装进不同谬误和多轮说服中，能够制造模型会被带偏的中等难度样本。它比 V7 的短而规整形式题更可能产生行为动态范围。

但如果直接把 LoFa 当成 CEDAR 主 F 臂，会破坏核心因果解释：

1. LoFa 的攻击往往同时改变“事实内容”和“推理形式”，无法区分模型是相信了假事实，还是没识别谬误。
2. appeal to authority、ad hominem 等类别天然带社会来源线索，会与 S 臂重叠。
3. 题目依赖世界知识；不跟攻击可能是模型记得事实，也可能只是拒绝，跟攻击也可能是知识不足。
4. 先按目标模型是否答对、是否抵抗假主张筛题，会让不同模型面对不同分母；如果再按 capping 曲线选题，就无法作为无偏确认集。
5. 它没有与 F 同 base、同 target 的真证据 V，因此不能直接估计“保护错误抵抗的同时是否伤害正确更新”。

因此，LoFa 最适合回答：**在自然、复杂、混合型错误说服中，我们的结论还能否观察到？** 它不适合单独回答：**capping 究竟改变了证据分辨还是社会顺从？**

## 5. 最值得引入的两个新支点

### 5.1 ReviseQA / ProverGen：核心库的生成骨架

建议把下一代核心题库从“人工写故事，再人工确认逻辑”改为：

1. 先生成机器可读的有限事实和规则；
2. 由 Prover9 或等价 solver 计算初始 gold 与证明；
3. 对一个明确节点做最小编辑，重新计算新 gold；
4. 自动保存 changed node、旧证明、新证明和 countermodel；
5. 再渲染成自然语言；
6. 由独立程序将自然语言关键字段回译/抽取，并再次与符号结构比对；
7. 人只审核语言是否自然、是否泄漏条件、是否出现第二错误，不再靠人脑承担全部逻辑证明。

这会直接解决 V7 暴露的两个矛盾：题太简单时没有动态范围；写复杂后人也难以判断。证明器承担正确性，参数承担难度，人承担自然性和构念纯度。

### 5.2 SycoBench-600：外部“选择性修正”骨架

SycoBench-600 最有价值的不是它的题，而是它明确把两种行为分开：

- 初始正确时，在错误质疑/建议下是否翻错；
- 初始错误时，在正确建议下是否修正；
- 两者之差构成 correction selectivity，而不是把“总愿意改口”当成好事。

CEDAR 应借这一分析，但把“正确建议”升级为**可验证的有效证据 V**，并保证 V/F/S/AO 都来自同一批 base。这样才能得到比 SycoBench 更干净的选择性更新指标。

## 6. 推荐的最终数据架构

### 6.1 核心因果库：solver-backed contrast clusters

每个 base 先有一个封闭世界和唯一初始答案。正式生成六个条件：

- **N**：不增加信息，仅要求继续/复述；prompt 不得说“没有新信息”。
- **R**：要求复核，但不给 target，也不给证据。
- **V**：一个最小、solver 可验证的变化使 target 成为新唯一答案。
- **F**：与 V 共享 target、长度和论证表面，但保留初始世界；candidate reasoning 只有一个可定位的致命错误。
- **AO**：无施动者地纯提及 target，不提供证言、压力或理由。
- **S**：在 AO 的 target exposure 基础上加入社会性要求，但无新事实、权威、奖惩或多数信息。

每条响应必须分类为：

1. initial；
2. induced target；
3. other wrong；
4. invalid/abstain。

同时保存 initial-target logit margin。二元 target-follow 只能作为其中一个投影，不能取代四类结果。

### 6.2 难度不再靠“把文字写绕”

难度轴在运行模型前固定：

- 证明深度：2、3、4、5；
- 规则类型：合取、析取、否定、量词/关系、约束传播；
- 干扰事实数；
- V 变化距离最终答案的图距离；
- F 唯一致命错误出现的位置；
- 是否需要撤销旧证明并构建新证明；
- 语言释义和作用域变化。

不得以目标 Qwen 是否翻转作为确认集难度标签。可以在 development 中做 response enrichment，但必须保留完整候选池，并用全新 family 生成确认集。

### 6.3 自然外部库

冻结核心假设和分析后，再跑：

- S：SycoBench-600、SycophancyEval/FlipFlop；
- F：LoFa、BrokenMath 的可用子集；
- V/R：VitaminC 或 ReviseQA 自然化子集；
- 多轮：SYCON-Bench、ReviseQA/BeliefTrack。

外部库之间不强求同一总分。它们分别检查社会压力、混合型错误说服、自然证据更新和多轮传播。

## 7. 建库流水线与硬闸

### Stage 0：冻结 estimand 和 schema

先写死要区分的量：V/F 分辨、总体更新门槛、S-AO 社会增量、N/R 稳定、other-wrong 与 invalid。冻结 arms、输出类、统计单位和停止条件。

### Stage 1：符号 family 设计

先设计 8–12 个机制 family，每个 family 用参数生成多个 base。family 之间按“初始证明—变化节点—新证明—F 唯一错误”四轴查重。

### Stage 2：solver 生成与验证

对每个 base 保存：

- 初始 world 与唯一 gold；
- V 后 world 与唯一 gold；
- 初始/更新证明；
- F countermodel；
- changed node；
- 所有选项的可满足性/可推出性。

任何依赖人工“感觉应该如此”的条目不得进入下一阶段。

### Stage 3：自然语言渲染

先用受限模板确保无歧义，再生成 2–3 个自然释义。采用回译/字段抽取、长度审计、条件泄漏分类器和双人语言审核。自然化不得改变符号语义。

### Stage 4：F、AO、S 构建

F 必须有且仅有一个 solver 可指出的错误；假事实与形式谬误分为不同 stratum。AO 使用无施动者纯 target 提及。S 只在 AO 上增加社会要求。不得写“没有新信息”“这只是压力”等元标签。

### Stage 5：静态质量闸

至少包括：

- solver 100% 通过；
- V/F target、答案位置、长度和修辞配对；
- 无 condition-label cue；
- 文本-only/observation-only 泄漏低于冻结门槛；
- family grouped overlap 审计；
- 人工语言/构念审核一致率；
- 确定性构建与文件哈希。

### Stage 6：development 校准

只用 uncapped 或预先定义的小格子检查：基础可解性、V 非地板、F/S 非零动态范围、解析完整性。允许据此修改生成器，但所有修改和候选结果完整保留；产物明确标为 development challenge bank。

### Stage 7：确认集

冻结生成器后，用未见过的 family/参数生成新题。不得根据 capping 响应删题或选择曲线。统计以 family 聚类，连续 margin 与四类转移并报。

### Stage 8：外部验证

在 SycoBench/LoFa 等数据上只做一次冻结分析。结果一致时支持外部有效性；不一致时定位构念差异，不把外部库强行合并成主结论。

## 8. 主要风险与预案

| 风险 | 预警信号 | 处理 |
|---|---|---|
| 核心题仍天花板 | V≈100%、F/S≈0%，margin 远离边界 | 提高证明深度/干扰项/错误定位距离，不加条件暗示；只在 development 调生成参数 |
| 核心题过难 | uncapped 基础题和 V 同时低 | 降证明深度；剔除的是生成机制而非按 capping 结果挑单题 |
| F 与假事实混杂 | F 需要接受一个新增的错误事实才能翻 | 把该类单列 misinformation stratum，不计入纯 F 主分析 |
| S 与 AO 混杂 | AO 中出现“某人认为/另一解题者说” | 主 AO 固定为无施动者纯提及；证言另列可选梯度 |
| 自然化破坏逻辑 | 人和 solver/字段抽取不一致 | 退回上一层重渲染，不能人工改 gold |
| 模型特定筛选偏差 | 只留下 Qwen 恰好在边界的题 | 仅 development 可 enrichment；确认集用新 family 全量保留 |
| 合成伪影 | text-only 分类器预测 V/F/S | 增加对比释义、交换词汇/选项、OOD symbol；用自然外部库确认 |
| 世界知识污染 | 模型不看新证据也能答 | 核心库使用封闭虚构世界；真实世界只放外部库 |
| 人工负担失控 | 审核复杂题需重新完整推理 | solver 出具 proof/countermodel；人只审语义等价、自然性和第二错误 |
| 新论文/代码不成熟 | 仓库变动、viewer 失败、缺 license | 固定 commit 和 hash；不把 BeliefTrack 等新资源设为单点依赖 |

## 9. 近期执行建议

### 9.1 当前立即做

1. 让已在运行的 V7 按原冻结协议完成，不中途改 prompt。
2. V7.1 只删除 N/AO/S 中的 `No new task information is supplied.`，作为配对提示语消融；不把它与新数据集构建混在一起。
3. 不立即投入完整 LoFa 改造。
4. 单独启动一个**零模型、小规模生成器可行性**任务：6–8 个 family、每个 2 个 base，先只生成符号 world、N/R/V、F countermodel，不生成 S，也不运行 Qwen。
5. 该可行性任务比较两条实现：直接使用/改装 ProverGen 与基于 ReviseQA 编辑算子的轻量自建器；以复算率、生成速度、难度可控性和自然化负担决定工程路线。

### 9.2 在 V7/V7.1 返回后

- 若删除提示语显著增加 S/AO target-follow，而 N 仍稳定：说明 V7 的社会压力地板部分由 prompt cue 造成；新核心库禁用此类元提示。
- 若 V7.1 仍是 S≈0：说明题面/推理模式本身让模型轻易识别无证据，不应继续微调措辞，应转向 solver-backed 难度控制与自然外部库。
- 若 AO 与 S 同步变化：主要是 target priming；若只有 S 变化：才支持社会压力成分。
- 无论结果如何，都不依据 V7/V7.1 的 capping 曲线挑选下一代确认题。

## 10. 最终拍板

**拍板建议：下一代不要叫“LoFa 版”，而应定义为“solver-backed selective belief revision benchmark”。**

它的贡献不是又造一批逻辑题，而是首次在同一 base 的严格对比簇中，同时测：

- 真证据是否被正确吸收；
- 假论证是否被识别；
- 纯 target exposure 是否造成 priming；
- 社会压力是否产生额外顺从；
- 干预是否让模型变顽固、变笨、定向变谄媚，或只改变输出门槛。

LoFa、SycoBench、BrokenMath 等负责证明这个核心测量在自然场景中仍有意义；它们是外部支柱，不是主地基。
