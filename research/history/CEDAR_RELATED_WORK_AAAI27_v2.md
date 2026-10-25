# 2 Related Work — v2 (prose revision of v1)

> Same citations, same claims, same scope discipline as v1. Rewritten for argument flow.
> Changes are listed at the bottom of this file.

---

## 2.1 Activation-Space Control and the Capping Operator / 激活空间控制与钳制算子

Representation engineering established that population-level activation patterns can be read out and
manipulated as first-class objects (Zou et al. 2023), and a now-standard toolkit followed: activation
addition and its contrastive refinement move the residual stream along directions derived from paired
prompts or responses (Turner et al. 2023; Rimsky et al. 2024), inference-time intervention edits the
attention heads that carry truthfulness (Li et al. 2023), and refusal has been traced to a direction
whose addition or ablation switches the behavior on and off (Arditi et al. 2024). What unites these
methods is that they *add*: a fixed vector is applied unconditionally, at every token, for as long as
generation continues. The Assistant Axis departs from this convention (Lu et al. 2026). It models the
default assistant persona and stabilizes it with one-sided threshold capping, an operator that does
nothing until a state falls below a percentile-calibrated boundary and then rectifies only the
shortfall. The distinction is not cosmetic. Continuous addition and conditional rectification differ in
when they fire, on which states, and with what magnitude, so evidence gathered about one does not
transfer to the other by default.

> 中文对照：表征工程确立了“群体层面的激活模式可读取、可操纵”这一研究对象（Zou et al. 2023），并催生了一套如今
> 标准化的工具：activation addition 及其对比式改进沿配对提示或回答导出的方向移动残差流（Turner et al. 2023;
> Rimsky et al. 2024）；inference-time intervention 编辑承载真实性的注意力头（Li et al. 2023）；拒答行为
> 也被定位到一个方向，加上或消融它即可开关该行为（Arditi et al. 2024）。这些方法的共同点是“加法”：一个固定向量
> 被无条件地、在每个 token 上、贯穿整个生成过程施加。Assistant Axis 偏离了这一惯例（Lu et al. 2026）：它刻画
> 默认助手人格，并以单侧阈值钳制加以稳定——只有当状态跌破按分位数校准的边界时才介入，且只补足缺口。这一区别并非
> 修辞：连续加法与条件整流在“何时触发、作用于哪些状态、施加多大幅度”上都不同，因此关于前者的证据不能默认外推到后者。

## 2.2 Selectivity Has Been Defined over Inputs, Not over Reasons / 既有选择性针对输入，而非理由

The field has taken side effects seriously. CAST gates steering on the input context (Lee et al. 2024);
KL-then-steer first suppresses distributional drift on benign prompts (Stickland et al. 2024); SCANS
tempers over-refusal while checking that harmful-query defense and general capability survive (Cao et
al. 2025); persona vectors extend the same control surface to traits such as sycophancy (Chen et al.
2025). Each of these asks whether an intervention leaves *other inputs* or *other tasks* intact. None
asks whether it remains sensitive to the *quality of the reason* a user supplies for changing an
answer — and the sycophancy literature shows why that question is not reducible to the first. Models
adjust their judgments to a user's stated beliefs even against the truth (Sharma et al. 2024); yielding
is only partly explained by prior uncertainty (Guo et al. 2026); and correction selectivity, not
resistance, is the quantity that matters, since an assistant that never changes its answer looks robust
precisely by refusing warranted correction (Sinha 2026). Dual-stance evaluation makes the danger
concrete: anti-sycophancy steering also suppresses agreement with true statements, even though the two
behaviors occupy separable activation geometry (Buchan 2026). A decodable direction and a lower
agreement rate are therefore both compatible with a control operator that is not selective at all.

> 中文对照：该领域一直重视副作用。CAST 依输入上下文决定是否启用 steering（Lee et al. 2024）；KL-then-steer 先
> 抑制良性输入上的分布漂移（Stickland et al. 2024）；SCANS 缓解过度拒答，同时核验有害请求防御与一般能力是否保留
> （Cao et al. 2025）；persona vectors 把同一控制面扩展到谄媚等特质（Chen et al. 2025）。这些工作问的都是干预
> 是否让“别的输入”或“别的任务”保持完好，却没有一项去问：它是否仍对用户为改答所提供“理由的质量”保持敏感——而谄媚
> 文献恰恰说明后者不能化归为前者。模型会为迎合用户陈述的信念而背离真相（Sharma et al. 2024）；让步只能被此前的
> 不确定性部分解释（Guo et al. 2026）；真正重要的量是修正选择性而非抵抗力，因为一个从不改答的助手之所以显得稳健，
> 正是由于它连该改的时候也不改（Sinha 2026）。dual-stance evaluation 把风险摆到了台面上：反谄媚 steering 同样
> 压低模型对正确陈述的赞同，尽管两类行为在激活几何上可分（Buchan 2026）。因此，“方向可被解码”与“赞同率下降”
> 都可以与一个完全不具选择性的控制算子并存。

## 2.3 Certifying What Counts as a Good Reason / 如何认证“好理由”

Answering the question therefore requires a task in which reason quality is not a matter of judgment.
Belief-revision benchmarks supply the frame: Belief-R formalizes revision under new premises and
reports the characteristic tension between updating when one should and holding firm when one should
not (Wilie et al. 2024). Symbolic tooling supplies the guarantee. ProverGen pairs language-model
generation with automated proving so that natural-language items carry checkable first-order structure
(Qi et al. 2025); LoFa probes robustness to fallacious persuasion while screening out mere gaps in base
knowledge (Shen et al. 2026); and on controlled syllogisms, Valentino et al. (2026) separate formal
validity from content plausibility and condition activation steering on that very distinction. Work on
the internal origins of user influence is methodologically closest to ours — layerwise logit-lens
analysis and activation patching have been used to localize the changes an incorrect user opinion
induces (Wang et al. 2026) — but there the contrast is between two different prompts, whereas ours is
between two runs of an identical prompt that differ only by an internal intervention. CEDAR turns these
ingredients toward intervention auditing. Within a cluster, the problem and the proposed target answer
are held fixed; a solver-certified valid reason makes the target correct, while a length- and
target-matched invalid argument carries a single localized defect and leaves the initial answer
correct. Neutral and recheck arms measure unsupported movement, and the social assay separates bare
target exposure from social pressure. The question we can then pose is not whether the model revises,
but whether capping preserves the difference between a reason that warrants revision and one that does
not — a property we call *epistemic selectivity*.

> 中文对照：要回答这个问题，就需要一个“理由质量”不取决于判断的任务。信念修订 benchmark 提供了框架：Belief-R 把
> 新前提下的修订形式化，并报告了“该更新时更新”与“不该更新时维持”之间的典型张力（Wilie et al. 2024）。符号工具
> 提供了保证：ProverGen 将语言模型生成与自动证明结合，使自然语言题目携带可核验的一阶结构（Qi et al. 2025）；
> LoFa 检验模型抵抗谬误说服的能力，同时筛除单纯的知识缺口（Shen et al. 2026）；Valentino et al.（2026）在受控
> 三段论上分离形式有效性与内容可信度，并正是依据这一区分来条件化激活 steering。在方法上与我们最接近的是关于用户
> 影响之内部来源的工作——已有研究用逐层 logit lens 与激活 patching 定位错误用户意见所引发的内部变化（Wang et al.
> 2026）——但那里的对照是两个不同的提示，而我们的对照是同一提示的两次运行，二者只在内部干预上不同。CEDAR 把这些
> 材料转向干预审计。在同一对照簇内，题目与被提议的 target 答案完全固定：经证明器认证的有效理由使 target 成为正确
> 答案；与之在长度和 target 上匹配的无效论证只含一个局部缺陷，因而初始答案仍然正确。中性与复核臂测量无依据的移动，
> 社会域则把单纯的 target exposure 与社会压力分开。于是我们能提出的问题不再是模型会不会改答，而是 capping 能否
> 保留“足以支持改答的理由”与“不足以支持改答的输入”之间的区别——我们称之为**认识论选择性（epistemic
> selectivity）**。

---

## 修改说明 / What changed and why

**文风层面**

1. **拆掉了三段同构的收尾。** v1 每个小节都以「X 做了 A。CEDAR 追问 B。」收束，连用三次，读起来像模板。v2 只在
   全节最后出现一次 “The question we can then pose is…”，把三小节改成递进：**算子不同 → 选择性从未按理由定义 →
   我们用符号认证把理由质量变成可判定的**。
2. **减少“论文点名清单”。** v1 大量句式是「A 做了 X（引）；B 做了 Y（引）」。v2 把同类工作合并进一个有主语、有
   转折的句子，并给每组一个共同判断（例如 2.1 的 “What unites these methods is that they *add*”）。
3. **句长与节奏。** 打散了 v1 的等长中句，长短交替；把 “Existing selectivity tests focus on…” 这类无施动者的
   句子改成有主语的判断句。
4. **术语首次出场即定义。** *epistemic selectivity* 现在在 §2 末尾被明确命名并定义，供 §3 直接引用。

**内容层面（三处实质补充，均无新引用）**

5. **明确写出 capping ≠ steering 的算子差异，并说明其后果**（2.1 末句）。这一句让我们后面不必继承 Buchan 的悲观
   结论，也为正文里 E−1 的算子对比留下接口。
6. **把 §2.2 的落点从“他们没测”改成“他们测的是另一个东西”**——选择性一直是对**输入/任务**定义的，从未对**理由
   质量**定义。这是本文缺口更准确的表述，也更难被审稿人反驳为“你只是加了一个指标”。
7. **给 Wang et al. 做了方法学定位**（2.3 中段）：他们的对照是 prompt-vs-prompt，我们的对照是同一 prompt 的
   intervention-vs-intervention。若后续放入逐层 KL 图，这句话就是我们相对该图的优势说明，需提前埋在 Related Work。

**保持不变**

- 全部 18 条引用、作者年份与出处格式（AAAI 风格）未动，参考文献表沿用 v1。
- 未加入任何未经核验的文献，未升级任何主张。
- 字数：英文正文约 560 词（v1 约 620），按 AAAI 双栏约 0.65–0.7 页，留出正文空间。
