# §2 Related Work — 我的缩写版（v6），中英对照

> 基线：Overleaf 当前的压缩版（约 400 词）。我的版本 **约 455 词**，多约 55 词、约 5 行：
> 四处被压掉的推理环节 + 五处段落承接（背景引入、2.1→2.2→2.3 的明写推进）。
> 下方每段末尾用 `▸` 说明我改了什么、为什么。**未推送**，等你确认。

---

## 2 Related Work

Deciding what a deployed model should and should not do has largely been a training-time problem, addressed by updating weights through supervised fine-tuning and preference optimization \citep{ouyang2022training,rafailov2023direct}. Activation interventions offer a lighter alternative: a small, reversible edit to internal states, applied only at inference. That is what makes them attractive to deploy, and also what makes them hard to evaluate, because the edit is specified in representation space while its consequences appear in behavior. In particular, when an operator changes an answer, does it still distinguish good reasons for the change from bad ones? The semantics of a direction do not answer this, discrete answer flips do not reveal it, and reason validity is rarely certified independently of the model being audited. We take these three obstacles in turn.

> **中文：** 决定一个已部署模型该做什么、不该做什么，长期以来主要是一个训练期问题，通过监督微调与偏好优化更新权重来解决 \citep{ouyang2022training,rafailov2023direct}。激活干预提供了一种更轻的替代方案：只在推理时对内部状态做一次小而可逆的修改。这既是它便于部署的原因，也是它难以评测的原因——修改写在表征空间里，后果却出现在行为上。具体地说：当一个算子改变了答案时，它是否仍然区分"因为好理由改答"和"因为坏理由改答"？方向的语义回答不了这个问题，离散的答案翻转显现不出它，而理由是否有效也很少由独立于被审计模型的程序来认证。下文依次处理这三重障碍。

> ▸ **改动 1｜先给背景与动机，再提问，最后才列三重障碍。** 原压缩版是 "Existing evaluations leave three gaps: A, B, C."——直接甩清单，读者既没有情境也没有理由相信。这里补了两级承接：**(a)** 训练期 → 推理期的背景切换，说明激活干预是什么、为什么有人用它；**(b)** 一句张力句 "这既是它便于部署的原因，也是它难以评测的原因——修改写在表征空间里，后果却出现在行为上"，把"好用"和"难评"接到一起，问题于是自然长出来，而不是被宣布出来。最后 "We take these three obstacles in turn" 兼作结构路标，也承接引言末尾的三项贡献。**比压缩版多约 35 词（约 3 行）。**

### From Activation Directions to State-Dependent Control

Interpretability is what makes inference-time control feasible in the first place: low-dimensional directions can read and manipulate truthfulness, refusal, and other behaviors \citep{zou2023representation,turner2023steering,li2023inference,rimsky2024steering,arditi2024refusal}. Addition, ablation, projection, and thresholded rectification nevertheless induce different state-dependent operations, so a direction's label does not fix what an operator built from it does. The Assistant Axis illustrates the gap: \citet{lu2026assistant} use a dominant persona direction to track drift and release a percentile-calibrated one-sided cap that stabilizes Assistant-like behavior. Because the cap removes only above-threshold projection at selected layers and token positions, it can act during prompt processing, deliberation, or answer formation while persona behavior looks unchanged. Persona stability is therefore not evidence of epistemic selectivity. Context-gated and retention-constrained steering improve selectivity over \emph{inputs} \citep{lee2025programming,stickland2024steering}; they answer where an intervention should apply, not why an answer should change within one task.

> **中文：** 可解释性研究正是推理期控制得以成立的前提：低维方向可以读取并操控真实性、拒答等行为 \citep{zou2023representation,turner2023steering,li2023inference,rimsky2024steering,arditi2024refusal}。但加法、消融、投影与阈值整流构成不同的状态依赖算子，因此方向的语义标签并不能决定由它构造的算子做了什么。Assistant Axis 使这一缺口具体化：\citet{lu2026assistant} 用主导 persona 方向追踪漂移，并发布按分位数标定的单侧钳制以稳定 Assistant-like 行为。由于该算子只在选定层与 token 位置上削去超过阈值的投影，它可能已经作用于提示处理、推理或答案形成，而人格行为看起来毫无变化。因此，人格稳定并不构成认识论选择性的证据。依赖上下文的门控与保持约束方法提高的是\emph{输入}层面的选择性 \citep{lee2025programming,stickland2024steering}：它们回答干预应当在"哪里"生效，而不是同一道任务中的答案应当"因为什么理由"改变。

> ▸ **改动 2｜找回两句核心判词。** (a) `Persona stability is therefore not evidence of epistemic selectivity.`——这是整段的结论句，压缩版把它删了，导致段落停在描述而没有裁决。(b) `where vs. why`——压缩版写成 "a complementary property within the same task"，抽象且不痛不痒；原来的 where/why 对比是这篇论文最容易被记住的一句话。
> 同时删掉了压缩版里的 "Their evidence supports persona stabilization; effects on task reasoning and warranted revision remain untested."——它和新的结论句重复，删掉正好补上新增词数。**净增约 5 词。**

### Why Existing Revision Readouts Are Not Identifying

If a direction's label cannot tell us what an operator does, the natural fallback is to judge it by behavior, and sycophancy research supplies the most developed readouts. Models can move toward a user's stated view when it conflicts with the correct answer \citep{sharma2024towards}, and layerwise analysis localizes opinion-conditioned divergence before the final prediction \citep{wang2026truth}; in those contrasts target mention and social framing vary together. SycoBench separates resistance to incorrect suggestions from acceptance of corrections \citep{sinha2026sycobench}. One might still hope that a single interpretable direction suffices: if a behavior is legible along one axis, pushing against that axis should remove it and leave the rest intact. Dual-stance evaluation refutes this, showing that anti-sycophancy steering also suppresses correct factual agreement \citep{buchan2026dual}. A property can thus be readable without being selectively writable through the same direction.

Even granting a behavioral readout, its resolution is limited: accuracy, flip rate, and target following compress a continuous answer distribution into a thresholded outcome. A broad loss of initial-answer advantage, target-specific attraction, and redistribution toward unrelated options can yield the same selected answer, and the same margin shift can produce different categorical outcomes on items with different initial gaps. We call this the \emph{margin--choice dissociation}. Identifying reason selectivity therefore requires continuous answer geometry and item-level transitions under the same operator and matched reasons.

> **中文：** 既然方向的语义标签说不清算子做了什么，自然的退路就是从行为上判断它，而谄媚性研究提供了目前最成熟的行为读数。模型会向用户声明的观点移动，即使该观点与正确答案冲突 \citep{sharma2024towards}；逐层分析还把受意见影响的分歧定位在最终预测之前 \citep{wang2026truth}——但在这些对照中，目标是否被提及与社会措辞是同时变化的。SycoBench 把抵抗错误建议与接受正确修正分开测量 \citep{sinha2026sycobench}。即便如此，人们仍可能指望单一可解释方向就够用：既然某种行为能沿一条轴被读出，沿该轴反向施力就应当消除它而不影响其余。dual-stance 评测否定了这一点：反谄媚 steering 同时压低了正确的事实性同意 \citep{buchan2026dual}。可见，某种性质可以被读取，却未必能沿同一方向被选择性地写入。
>
> 而即便接受行为读数，它的分辨率也有限：准确率、翻转率与 target-following 把连续的答案分布压缩成跨阈值的结果。广泛削弱初始答案优势、专门吸引模型走向目标，以及把概率分散到无关选项，都可能产生同一个最终选择；而同一个 margin 位移，在初始差距不同的题目上又会给出不同的类别结果。我们把这一现象称为 \emph{margin--choice 分离}。因此，识别理由选择性必须在同一算子与严格配对的理由下，同时观察连续答案几何与逐题迁移。

> ▸ **改动 3｜把 Buchan 从"又一条并列证据"改回"反驳一个具体立场"。** 压缩版是 "SycoBench 分开了 A/B，而 dual-stance 表明 C"——三个发现并列，读者不知道它们在反对什么。加上"单轴假说"这个靶子（22 词），Buchan 的结果立刻变成一次反驳而不是一条陈列。这也是你上次自己指出的问题（"简单的单轴解释是从哪来的"）——压缩时又被删回去了。**净增约 20 词。**

### Certifying Reason Quality for Intervention Audits

Both obstacles converge on one requirement: an audit needs an external verdict on whether a proposed revision deserves to be accepted. Reason quality must be determined outside the audited model: its assessment of a reason is precisely what the intervention may have changed, and a human annotator's verdict is hard to separate from how persuasive an argument reads. Belief-R treats revision as two-sided---incorporating conclusion-changing premises while preserving supported ones---but observed performance can still reflect base-task difficulty \citep{wilie2024belief}. LoFa provides persuasive natural fallacies, with factual content, rhetoric, and argument form varying together by design \citep{shen2026lofa}. ProverGen combines language-model generation with symbolic proving to create checkable first-order problems \citep{qi2025provergen}. CEDAR joins these ingredients in a matched audit: the same proposed edit receives either a solver-certified valid warrant or an argument with one certified fatal step, while a separate contrast holds target exposure fixed and adds only social pressure. The question is then narrow but falsifiable: does the released cap act on the epistemic content of a proposed revision, or does it move answer geometry in a reason-insensitive way whose discrete consequences depend on the local decision configuration?

> **中文：** 上述两重障碍指向同一个要求：审计需要一个外部裁定，判断一次候选修订是否值得被接受。而理由质量必须由被审计模型之外的程序来判定：模型对理由的评估恰恰就是干预可能已经改变的对象，而人类标注者的裁决又难以与"论证读起来有多有说服力"分开。Belief-R 把改答视为双向问题——既纳入会改变结论的前提，也保留仍有支持的结论——但其表现仍可能受基础任务难度影响 \citep{wilie2024belief}。LoFa 提供自然而有说服力的谬误论证，并有意让事实内容、修辞与论证形式共同变化 \citep{shen2026lofa}。ProverGen 将语言模型生成与符号证明结合，构造一阶结构可核验的问题 \citep{qi2025provergen}。CEDAR 把这些要素组合成一次严格配对的审计：同一候选编辑要么获得经求解器认证的有效理由，要么获得只含一处经认证致命错误的论证；另一组对照固定目标暴露，只增加社会压力。由此得到的问题范围很窄，但可以被证伪：公开的 capping 究竟是作用于候选修订的认识论内容，还是仅以对理由不敏感的方式移动答案几何、并让离散后果取决于局部决策状态？

> ▸ **改动 4｜补回认证的\*理由\*，删掉一句总结。** 压缩版只说 "Reason quality must also be determined outside the audited model."——为什么必须？这是审稿人一定会问的。补上"循环论证 + 标注者无法与说服力分离"两个小句（约 30 词），这条设计就从"我们比较严谨"变成"这是唯一可行的做法"。
> 同时删掉结尾那句 "This design asks whether the released model-specific caps respond to warrant quality or shift answer geometry across reasons."，换成第四部分正文里那个二择一问句——**同样长度，但它是可证伪的假设陈述，而不是设计说明**，并且直接与第四节标题"稳定的答案几何，分化的改答后果"对上。

---

## 我砍掉了什么（用来腾出词数）

| 删除 | 理由 |
|---|---|
| `Their evidence supports persona stabilization; effects on task reasoning and warranted revision remain untested.` | 与新的结论句 "Persona stability is not evidence of epistemic selectivity" 重复。 |
| `These results establish that willingness to update and representational separability do not guarantee selective control.` | 这句话把两条不同的证据强行合并成一个抽象结论；拆开后各自的靶子更清楚，这句就冗余了。 |
| `CEDAR evaluates a complementary property within the same task: whether the operator remains sensitive to the quality of the reason...` | 被 where/why 那句取代，更具体。 |
| `This design asks whether the released model-specific caps respond to warrant quality or shift answer geometry across reasons.` | 被结尾的二择一问句取代。 |

净变化：**+55 词左右（约 5 行）**，其中约 40 词是这一轮补的四处承接。如果放不下，按此顺序砍：改动 3 的"单轴假说"铺垫（22 词）→ 2.3 的开头承接句（18 词）→ 2.1 的开头从句（9 词）。四处承接里 2.2 的两句最不可省，因为它们是"为什么要换一种证据"的唯一交代。

---

## 这一轮补的承接（改动 5）

| 位置 | 补的句子 | 作用 |
|---|---|---|
| 节首 | "这既是它便于部署的原因，也是它难以评测的原因——修改写在表征空间里，后果却出现在行为上。" | 背景 → 张力 → 问题，不再直接甩清单 |
| 2.1 开头 | "可解释性研究正是推理期控制得以成立的前提：" | 把方向文献接到上一段的"激活干预"上，而不是另起炉灶 |
| 2.2 开头 | "既然方向的语义标签说不清算子做了什么，自然的退路就是从行为上判断它，而谄媚性研究提供了目前最成熟的行为读数。" | 明写 2.1 → 2.2 的推进：语义不行 → 换行为证据 |
| 2.2 第二段开头 | "而即便接受行为读数，它的分辨率也有限：" | 把"读数不成立"与"读数太粗"分成两级，而不是并列两段 |
| 2.3 开头 | "上述两重障碍指向同一个要求：审计需要一个外部裁定……" | 明写 2.1+2.2 → 2.3 的收束，第三节由此长出来 |

---

## 一句话总结

压缩版没有事实错误，问题是**把每一段的裁决句都当成"可省的修辞"删掉了**，留下的是证据列表。我这一版把四句裁决句放回去（人格稳定 ≠ 认识论选择性；where vs. why；单轴假说被反驳；认证是唯一非循环的做法），并用结尾的二择一问句把整节收束成一个假设——代价是约 1.5 行。