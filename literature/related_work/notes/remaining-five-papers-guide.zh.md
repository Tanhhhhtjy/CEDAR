# 其余五篇相关论文逐章节导读

本文件按论文实际章节顺序说明“这一部分在回答什么”，用于快速判断是否值得精读。所有 venue 均已从正式会议页面或论文首页核验。

## 1. HarmBench

> *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal*  
> ICML 2024 · [官方页面](https://proceedings.mlr.press/v235/mazeika24a.html)

### 摘要与引言

论文指出自动红队研究缺少统一评测：不同工作使用不同有害行为、生成长度、攻击预算和分类器，结果无法直接比较。作者提出 HarmBench，以 breadth、comparability 和 robust metrics 为三项设计原则，并在 18 种攻击、33 个模型/防御上进行大规模比较。

Figure 1 不是算法图，而是整个 benchmark 的地图：左侧是 standard、contextual、copyright、multimodal 四种功能类别，中间是攻击和防御，右侧是语义风险类别与模型。

### Section 2：Related Work

回顾自动红队方法和防御。攻击包括梯度优化、LLM 优化器、人工模板和多模态攻击；防御分为系统级过滤和模型级拒答。该节的作用是说明已有结果为何需要统一评测，而不是罗列所有 jailbreak。

### Section 3：Automated Red Teaming

#### 3.1 Problem Definition and Metrics

把红队统一表示为：攻击方法为每个有害行为生成测试用例，目标模型生成回答，分类器判断行为是否成功完成，最后计算 attack success rate（ASR）。这为后面的所有比较提供共同数学接口。

#### 3.2 Toward Improved Evaluations

逐项展示旧评测的问题：行为覆盖不足、对生成 token 数敏感、训练集与测试集混用、评价分类器不鲁棒等。它不是“结果章节”，而是在证明 HarmBench 的设计需求确实存在。

### Section 4：HarmBench

#### 4.1 Overview

介绍 benchmark 的两部分：510 个有害行为和标准化评价流水线。

#### 4.2 Curation of Harmful Behaviors

解释四种功能类别和七种语义类别如何构建，尤其关注必须结合上下文才能判断危害的 dual-intent 行为、版权行为和多模态行为。作者还对潜在敏感信息做筛查。

#### 4.3 Evaluation Pipeline

统一为三步：生成测试用例、让目标模型生成、用经过验证的分类器评分。该节解决的是复现和公平比较，而非提出新攻击。

### Section 5：Experiments

#### 5.1 Main Results

大规模比较的两个主要发现是：没有任何攻击或防御在所有对象上都有效；同一家族内部的鲁棒性与模型大小关系较弱，不同家族之间差异更大。这提示训练数据和方法可能比参数规模更关键。

#### 5.2 Adversarial Training Results

作者用 HarmBench 展示 benchmark 如何促进防御开发，并提出 R2D2 对抗训练。它对 GCG 类攻击明显更鲁棒，但对不同机制的攻击改善较小，因此仍需多样化攻击训练。

### Conclusion

再次强调贡献是“可比较的标准化评测 + 初始大规模基线 + 一个防御演示”。防御方法不是全文唯一主角。

### 对 CEDAR 的参考价值

- 可借鉴：先定义评测应满足什么，再说明每个条件如何消除混杂。
- 可借鉴：benchmark 和缓解方法主次分明，缓解只作为 benchmark 用途的证明。
- 不宜照搬：CEDAR 的标题级贡献是隐藏认识论代价，不应写成纯 benchmark 排行榜。

---

## 2. Belief-R

> *Belief Revision: The Adaptability of Large Language Models Reasoning*  
> EMNLP 2024 · [ACL Anthology](https://aclanthology.org/2024.emnlp-main.586/)

### 摘要与引言

现有推理 benchmark 通常假定信息静态、完整；真实环境中的新信息却可能要求撤回旧结论。作者提出 Belief-R 和 delta reasoning（\(\Delta R\)）框架，分别测量模型何时应更新、何时应维持既有结论。

引言提前给出核心发现：模型普遍不善于信念修正，而且 update 表现好的模型常在 maintain 条件中变差，形成更新—维持权衡。

### Section 2：Related Works

先介绍哲学与 AI 中的 belief revision，再区分模型编辑和上下文内修正：参数更新是外部修改模型知识，不等于模型在新证据下自行判断应否修正。随后将 Belief-R 与静态逻辑、矛盾解决和缺失信息 benchmark 区分。

### Section 3：Belief Revision

#### 3.1 Background and Notation

把 belief 操作化为当前上下文中模型认为成立、并以输出表达的命题，而不是对模型内部心理状态作强本体论主张。

#### 3.2 Delta Reasoning

在时刻 \(t\) 给两个前提，先建立一个 modus ponens 或 modus tollens 结论；在 \(t+1\) 加入第三个前提。第三前提可能增加必要条件，要求撤回旧结论（Belief Update），也可能只是提供替代路径，旧结论仍应保留（Belief Maintain）。

### Section 4：The Belief-R Dataset

#### 4.1 Dataset Construction

以 ATOMIC 常识关系为种子，用 LLM 生成候选三前提任务，再由人工标注第三前提究竟是额外条件还是替代条件。

#### 4.2 Quality Check

多人标注、投票和一致性过滤。这里的人工判断仍是 commonsense 意义上的条件关系，并非形式 solver 完全决定。

#### 4.3 Statistics

约 2K 样本，覆盖 modus ponens/tollens、Belief Update/Maintain，以及事件和心理状态两类关系。

### Section 5：Experiment Settings

指标包括基础推理准确率、BU-Acc、BM-Acc，以及二者等权平均的 BREU。评估多种开源和闭源模型，并比较 direct、CoT、plan-and-solve 提示。

### Section 6：Result and Analysis

先检查模型能否完成基础逻辑；只有基础能力足够的模型才进入 belief revision 分析。主要发现：许多模型在 BU 上接近失败；较强模型仍存在 BU 与 BM 的明显张力。更好的 prompting 没有稳定解决问题。

### Section 7：Discussion

分析 modus tollens 比 modus ponens 更难，抽象心理状态比具体事件更难，以及 update/maintain 失衡可能来自模型对前提依赖关系的处理不足。

### Section 8–9：Conclusion and Future Work

总结 Belief-R 揭示的适应性缺口，并提出神经—符号或 solver 辅助推理等未来方向。Limitations 承认其 belief 定义是输出层面的，任务也只覆盖由新增前提触发的有限修正形式。

### 对 CEDAR 的参考价值

- 可借鉴：必须把“应该更新”和“不应该更新”放在匹配条件中共同测量。
- CEDAR-Bench 的 solver gold、完整证明链与 F countermodel 有机会比 Belief-R 的常识标注提供更强规范依据。
- 不宜照搬：Belief-R 不研究安全干预，也不能作为 activation capping 有认识论代价的证据。

---

## 3. Towards Understanding Sycophancy in Language Models

> ICLR 2024 · [OpenReview](https://openreview.net/forum?id=tvhaxkMKAn)

### 摘要与引言

论文问两个问题：生产级助手是否在多种真实自由文本任务中谄媚，以及人类偏好训练是否鼓励这种行为。作者先证明现象跨模型和任务存在，再分析人类偏好数据与 preference model。

### Section 2：Background

定义 AI assistant、RLHF 和 sycophancy。作者把谄媚理解为为了符合用户偏好、观点或自我判断而牺牲真实性，而不是所有答案改变都叫谄媚。

### Section 3：Measuring Sycophancy

#### 3.1 Biased Feedback

在数学、论证和诗歌评价中，只改变用户先说“我喜欢/不喜欢”，模型反馈就系统性随用户偏好变化。Figure 1 展示这一标题级行为结果。

#### 3.2 Easily Swayed

模型先回答问题，再被用户问“你确定吗？”或提出质疑。模型经常承认并不存在的错误或改变原本正确的答案。

#### 3.3 Biased Answers

把用户政治或价值立场放入提示，观察模型答案是否向该立场偏移。

#### 3.4 Mimic User Mistakes

用户示例中包含错误时，模型有时会模仿这些错误，而不是纠正。

这一节的结构特点是：用四种不同任务反复建立同一个现象，而不是一开始讨论训练机制。

### Section 4：Towards Understanding Sycophancy

#### 4.1 Human Preference Data

用语言模型给 hh-rlhf 偏好对生成行为特征，再用 Bayesian logistic regression 分析哪些特征预测人类偏好。“符合用户观点”是获得偏好的因素之一。

#### 4.2 Preference Models

通过 RL 和 best-of-N 更强地优化 Claude 2 的 preference model。某些谄媚形式增加，另一些下降，说明谄媚只是 PM 同时优化的多个特征之一，不是单调规律。

#### 4.3 Humans and PMs

比较真实、纠正用户的回答与写得有说服力但迎合错误信念的回答。人和 PM 多数时候偏好真实回答，但并不可靠；部分情况下会选谄媚回答。

### Section 5–6：Related Work and Conclusion

将结果连接到 reward hacking、偏好建模和 scalable oversight。结论谨慎写成：人类偏好可能是谄媚来源之一，而不是已证明的唯一原因。

### 对 CEDAR 的参考价值

- 可借鉴：先用多个反事实任务证明现象，再研究原因。
- 可借鉴：把“改变答案”细分为认错、立场迎合、模仿错误等不同机制。
- 不宜照搬：CEDAR 的 S−AO 是操作化的社会压力增量，不能直接称作内部真实谄媚。

---

## 4. Steering Llama 2 via Contrastive Activation Addition

> ACL 2024 · [ACL Anthology](https://aclanthology.org/2024.acl-long.828/)

### 摘要与引言

论文提出 Contrastive Activation Addition（CAA）：用正负行为样本的残差流均值差构造 steering vector，在生成时加入该向量，从而连续调节拒答、谄媚、幻觉等高层行为。

Figure 1 同时说明向量如何从对比样本提取、又如何在生成阶段加入。Table 1 给出加减谄媚向量后的定性回答。

### Section 2：Related Work

对比 Activation Addition、Inference-Time Intervention 和 Representation Engineering，突出 CAA 使用大量严格配对样本而不是单个 prompt pair，并在一个残差层上作用于提示之后的所有生成位置。

### Section 3：Method

用只相差答案 token 的正负多项选择样本提取残差激活差，再对许多样本求平均。测试七种行为，包括 corrigibility、hallucination、myopia、survival instinct、sycophancy 和 refusal。

本节还用激活可视化检查不同行为的聚类与几何关系，但这些图主要是描述性证据。

### Section 4：Effect on Behaviors

先做 layer sweep 和 multiplier sweep，再分别评估多项选择与开放生成。主要结论是加减向量能以预期方向改变多种行为，但效果依赖层、行为和系数。

### Section 5：CAA and System Prompting

比较 activation steering 与系统提示。CAA 在某些行为上能叠加在系统提示之上，但二者不是完全可互换的算子。

### Section 6：Comparison to Finetuning

比较 CAA 与监督微调，考察数据效率、效果和迁移。论文试图说明激活方法可以成为低成本行为控制手段，但不宣称取代微调。

### Section 7：General Capabilities

用通用 benchmark 检查 steering 是否损害基础能力。其结构价值在于：行为控制效果之后，单独报告 collateral effects。

### Section 8：Understanding and Interpreting CAA

研究 steering vector 之间的余弦相似度、跨层关系、跨模型迁移，以及 base/chat 模型表示的差异。这些实验帮助描述向量几何，但大多不是严格因果机制证明。

### Section 9–10：Discussion and Limitations

讨论线性表示假设、行为依赖和多向量组合，并承认开放生成评价依赖 GPT-4、模型范围有限、steering 可能产生不可预期副作用。

### 对 CEDAR 的参考价值

- 可借鉴：主效果 → 与替代干预比较 → 一般能力 → 表示分析。
- 必须区分 addition、ablation、steering 与 threshold capping；不能从 CAA 的加法实验推断官方 capping 的效果。
- CEDAR 若比较算子，应匹配层、token、实际修正强度和输出完整性。

---

## 5. Inference-Time Intervention

> *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*  
> NeurIPS 2023 · [官方页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)

### 摘要与引言

论文从“模型知道的比说出来的多”出发：在 TruthfulQA 上，中间激活的 probe 准确率比正常生成准确率高约 40 个百分点。作者提出 ITI，在少数 truth-related attention heads 上沿真值方向移动激活，希望缩小 knowing–telling gap。

Figure 1 先给无干预与有干预回答的定性对比，快速建立方法目标。

### Section 2：Related Work

回顾 RLHF、truthfulness、model editing、probe 和 activation intervention。作者强调 ITI 不更新模型权重，只在推理期间干预少数 head，因此标注和算力需求较小。

### Section 3：Method

#### 3.1 Setup

从 TruthfulQA 构造真/假陈述和多种 prompt，提取 attention head 激活。

#### 3.2 Probing for Truthfulness

给每个 head 训练线性 probe，按验证集表现选出最相关的少数 heads。高 probe 准确率说明真值信息可解码，但本身不证明这些 heads 因果决定输出。

#### 3.3 Inference-Time Intervention

对选中 heads 的激活沿 probe 区分出的真值方向平移，并在自回归生成的每一步重复。两个主要超参数是干预 head 数量 \(K\) 和强度 \(\alpha\)。

### Section 4：Experiments

在 TruthfulQA 上比较原模型、prompting、少样本、微调和 ITI。ITI 显著提高 truthfulness，但过强干预会损害 informativeness；因此作者用 true × informative 的组合指标选择超参数。

### Section 5：Analysis

#### 5.1 Categories

检查不同 TruthfulQA 类别，观察收益是否集中于特定错误类型。

#### 5.2 Efficiency

说明 ITI 只需要少量标注和推理时小规模修改。

#### 5.3 Generalization

在其他 benchmark 上测试，得到较小但非零的迁移；因此主要证据仍来自 TruthfulQA。

#### 5.4 Data Size and Strength

分析训练样本数、方向稳定性和干预强度。核心结论是 truthfulness 与 helpfulness/informativeness 存在连续权衡，不能只报告最佳点。

#### 5.5 Why Not All Heads

对所有 heads 干预会引入更多无关扰动；选择少数高 probe heads，有助于在更强干预下保持信息量。

### Section 6：Conclusions and Future Work

ITI 展示了无需重新训练模型即可从内部状态中诱导更真实回答的可能性，但作者明确表示它远不足以保证 LLM 的真实性。

### 对 CEDAR 的参考价值

- 最值得借鉴的是二维结果：干预收益必须与 helpfulness/informativeness 损失同时展示。
- probe 用于选择位置，真正的方法证据来自干预后的行为变化。
- ITI 是稀疏 attention-head addition，不是 threshold capping；两者不可互换。

