# 2 Related Work — v3（加入背景引入与段间承接）

> 相对 v2 的改动集中在三处：①开头补一句训练期 → 推理期的研究背景，让读者知道这条线从哪来；
> ②每小节首句改为承接上节结论，形成 “控制可行 → 选择性只按输入定义 → 用符号认证把理由变成可判定” 的链条；
> ③结尾加一句过渡到 §3。引用与主张范围不变。

---

## 2.1 From Training-Time Alignment to Inference-Time Control / 从训练期对齐到推理期控制

Getting a deployed language model to behave as its developers intend has traditionally been a
training-time problem, addressed by supervised fine-tuning and reinforcement learning from human
feedback. A more recent line of work moves the point of intervention to inference. Representation
engineering established that population-level activation patterns can be read out and manipulated as
first-class objects (Zou et al. 2023), and a now-standard toolkit followed: activation addition and
its contrastive refinement move the residual stream along directions derived from paired prompts or
responses (Turner et al. 2023; Rimsky et al. 2024), inference-time intervention edits the attention
heads that carry truthfulness (Li et al. 2023), and refusal has been traced to a direction whose
addition or ablation switches the behavior on and off (Arditi et al. 2024). What unites these methods
is that they *add*: a fixed vector is applied unconditionally, at every token, for as long as
generation continues. The Assistant Axis departs from that convention (Lu et al. 2026). It models the
default assistant persona and stabilizes it with one-sided threshold capping, an operator that stays
inactive until a state crosses a percentile-calibrated boundary and then rectifies only the shortfall.
The distinction is not cosmetic: addition and conditional rectification differ in when they fire, on
which states, and with what magnitude, so evidence about one does not transfer to the other by default.

> 中文对照：让已部署的语言模型按开发者意图行事，长期以来是一个训练期问题，主要依靠监督微调与人类反馈强化学习。
> 较新的一条路线把干预点移到推理期。表征工程确立了“群体层面的激活模式可作为一等对象被读取和操纵”（Zou et al.
> 2023），并催生了一套如今标准化的工具：activation addition 及其对比式改进沿配对提示或回答导出的方向移动残差流
> （Turner et al. 2023; Rimsky et al. 2024）；inference-time intervention 编辑承载真实性的注意力头（Li et al.
> 2023）；拒答行为被定位到一个方向，加上或消融它即可开关该行为（Arditi et al. 2024）。这些方法的共同点是“加法”：
> 一个固定向量被无条件地、在每个 token 上、贯穿整个生成过程施加。Assistant Axis 偏离了这一惯例（Lu et al. 2026）：
> 它刻画默认助手人格，并以单侧阈值钳制加以稳定——只有当状态越过按分位数校准的边界时才介入，且只补足缺口。这一区别
> 并非修辞：加法与条件整流在“何时触发、作用于哪些状态、施加多大幅度”上都不同，因此关于前者的证据不能默认外推到后者。

## 2.2 Selectivity Has Been Asked about Inputs, Not about Reasons / 既有选择性针对输入，而非理由

Because such operators are applied to a general-purpose model, the field has taken their side effects
seriously — but almost always by asking whether *other inputs* or *other tasks* survive. CAST gates
steering on the input context (Lee et al. 2024); KL-then-steer first suppresses distributional drift on
benign prompts (Stickland et al. 2024); SCANS tempers over-refusal while checking that harmful-query
defense and general capability are retained (Cao et al. 2025); persona vectors extend the same control
surface to traits such as sycophancy (Chen et al. 2025). None of these asks whether an intervention
remains sensitive to the *quality of the reason* a user supplies for changing an answer, and the
sycophancy literature shows why the two questions come apart. Models adjust their judgments to a user's
stated beliefs even against the truth (Sharma et al. 2024); yielding is only partly explained by prior
uncertainty (Guo et al. 2026); and correction selectivity, not resistance, is the quantity that
matters, since an assistant that never changes its answer looks robust precisely by refusing warranted
correction (Sinha 2026). Dual-stance evaluation makes the risk concrete: anti-sycophancy steering also
suppresses agreement with true statements, even though the two behaviors occupy separable activation
geometry (Buchan 2026). A decodable direction and a lower agreement rate are therefore both compatible
with a control operator that is not selective at all.

> 中文对照：由于这些算子被施加在通用模型上，该领域一直重视其副作用——但几乎总是在问“别的输入”或“别的任务”是否
> 完好。CAST 依输入上下文决定是否启用 steering（Lee et al. 2024）；KL-then-steer 先抑制良性输入上的分布漂移
> （Stickland et al. 2024）；SCANS 缓解过度拒答，同时核验有害请求防御与一般能力是否保留（Cao et al. 2025）；
> persona vectors 把同一控制面扩展到谄媚等特质（Chen et al. 2025）。它们都没有问：干预是否仍对用户为改答所提供
> “理由的质量”保持敏感——而谄媚文献恰恰说明这两个问题会分离。模型会为迎合用户陈述的信念而背离真相（Sharma et al.
> 2024）；让步只能被此前的不确定性部分解释（Guo et al. 2026）；真正重要的量是修正选择性而非抵抗力，因为一个从不
> 改答的助手之所以显得稳健，正是由于它连该改的时候也不改（Sinha 2026）。dual-stance evaluation 把风险摆到台面上：
> 反谄媚 steering 同样压低模型对正确陈述的赞同，尽管两类行为在激活几何上可分（Buchan 2026）。因此，“方向可被解码”
> 与“赞同率下降”都可以与一个完全不具选择性的控制算子并存。

## 2.3 Certifying What Counts as a Good Reason / 如何认证“好理由”

Turning that concern into a measurement requires a task in which reason quality is not a matter of
judgment. Belief-revision benchmarks supply the frame: Belief-R formalizes revision under new premises
and reports the characteristic tension between updating when one should and holding firm when one
should not (Wilie et al. 2024). Symbolic tooling supplies the guarantee. ProverGen pairs
language-model generation with automated proving so that natural-language items carry checkable
first-order structure (Qi et al. 2025); LoFa probes robustness to fallacious persuasion while screening
out mere gaps in base knowledge (Shen et al. 2026); and on controlled syllogisms, Valentino et al.
(2026) separate formal validity from content plausibility and condition activation steering on that
very distinction. Work on the internal origins of user influence is methodologically closest to ours —
layerwise logit-lens analysis and activation patching have been used to localize the changes an
incorrect user opinion induces (Wang et al. 2026) — but there the contrast is between two different
prompts, whereas ours is between two runs of an identical prompt that differ only by an internal
intervention. CEDAR turns these ingredients toward intervention auditing: within a cluster, the problem
and the proposed target answer are held fixed, a solver-certified valid reason makes the target
correct, and a length- and target-matched invalid argument carries a single localized defect that
leaves the initial answer correct. The question we can then pose is not whether the model revises, but
whether capping preserves the difference between a reason that warrants revision and one that does not
— a property we call *epistemic selectivity*, and the one we formalize next.

> 中文对照：要把这一关切变成可测量的对象，就需要一个“理由质量”不取决于判断的任务。信念修订 benchmark 提供了框架：
> Belief-R 把新前提下的修订形式化，并报告了“该更新时更新”与“不该更新时维持”之间的典型张力（Wilie et al. 2024）。
> 符号工具提供了保证：ProverGen 将语言模型生成与自动证明结合，使自然语言题目携带可核验的一阶结构（Qi et al.
> 2025）；LoFa 检验模型抵抗谬误说服的能力，同时筛除单纯的知识缺口（Shen et al. 2026）；Valentino et al.（2026）
> 在受控三段论上分离形式有效性与内容可信度，并正是依据这一区分来条件化激活 steering。在方法上与我们最接近的是关于
> 用户影响之内部来源的工作——已有研究用逐层 logit lens 与激活 patching 定位错误用户意见所引发的内部变化（Wang et
> al. 2026）——但那里的对照是两个不同的提示，而我们的对照是同一提示的两次运行，二者只在内部干预上不同。CEDAR 把
> 这些材料转向干预审计：在同一对照簇内，题目与被提议的 target 答案完全固定；经证明器认证的有效理由使 target 成为
> 正确答案；与之在长度和 target 上匹配的无效论证只含一个局部缺陷，因而初始答案仍然正确。于是我们能提出的问题不再
> 是模型会不会改答，而是 capping 能否保留“足以支持改答的理由”与“不足以支持改答的输入”之间的区别——我们称之为
> **认识论选择性（epistemic selectivity）**，并在下一节将其形式化。

---

## 相对 v2 的具体改动

| # | 位置 | v2 | v3 | 理由 |
|---|---|---|---|---|
| 1 | 2.1 首句 | 直接从 “Representation engineering established…” 起 | 先补 “训练期对齐 → 推理期干预” 一句 | 读者需要先知道这条线为什么存在。AAAI 邻域论文（如 SCANS、Persona Vectors）几乎都以“对齐/控制”一句开场，再落到具体机制 |
| 2 | 2.1 小节名 | Activation-Space Control and the Capping Operator | From Training-Time Alignment to Inference-Time Control | 小节名本身承担叙事，不只做分类 |
| 3 | 2.2 首句 | “The field has taken side effects seriously.” | “Because such operators are applied to a general-purpose model, the field has taken their side effects seriously — but almost always by asking whether *other inputs* or *other tasks* survive.” | 承接 2.1 的“通用模型上的算子”，并把本节结论前置，读者一句话就知道这节要证明什么 |
| 4 | 2.2 中段 | “Each of these asks whether… None asks whether…” | 合并为一句 “None of these asks whether…, and the sycophancy literature shows why the two questions come apart.” | 去掉一次重复的对仗，并用 “come apart” 明确逻辑关系 |
| 5 | 2.3 首句 | “Answering the question therefore requires…” | “Turning that concern into a measurement requires…” | “that concern” 直接指回 2.2 末句，链条闭合；同时把动作从“回答问题”改成“变成可测量”，与本文是评测论文一致 |
| 6 | 2.3 末句 | “…a property we call *epistemic selectivity*.” | “…*epistemic selectivity*, and the one we formalize next.” | 显式过渡到 §3，避免 Related Work 与 Method 之间断裂 |
| 7 | 全节 | — | 三个小节首句现在依次是“背景—转折—落地”，可单独连读成摘要 | 便于审稿人跳读 |

## 参考的邻域写法（均为本文已核验引用）

- **先框定问题域再落到方法**：*SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering*（AAAI-25）、*Persona Vectors: Monitoring and Controlling Character Traits in Language Models*。
- **用一句“共同点/共同缺陷”统摄一组工作**，而不是逐篇罗列：*Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs*。
- **在 Related Work 内就把最近邻的对照维度讲清楚**：*Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention*、*When Truth Is Overridden*（AAAI-26）。本文 2.3 对 Wang et al. 的 “prompt-vs-prompt 与 intervention-vs-intervention” 定位即属此类。

## 字数与排版

- 英文正文约 **610 词**（v2 约 560）。AAAI 双栏下约 0.55–0.6 页。
- 若 §2 需压到 **0.45 页**，按此顺序删，均不伤主链条：
  1. 2.1 中 Li et al. 与 Arditi et al. 合并为一个从句（约 −25 词）；
  2. 2.2 中 SCANS 与 persona vectors 合并为一句（约 −25 词）；
  3. 2.3 中 LoFa 一句压成半句（约 −20 词）；
  4. 最后才考虑删 2.3 对 Wang et al. 的定位句——但如果正文放逐层 KL 图，这句必须保留。