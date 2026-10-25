# 2 Background and Related Work — v5 (COMPACT base + 6 patches)

> 基线：v4 COMPACT。我同意 Codex 的推荐——正文用短版。以下是我在其上做的 6 处修改，
> 每处在段末以 `【改动 N】` 标注理由。长版仅保留其结尾那段"锐化问题"，已并入本版。

---

## 2 Background and Related Work

Activation control, sycophancy, and belief revision have largely been studied separately. Their intersection exposes a harder question than whether an intervention changes behavior: does it preserve the difference between good and bad reasons for changing an answer? That property cannot be inferred from the meaning of an activation direction, and answer flips alone do not identify it. We organize prior work around this distinction.

> **中文：** 激活控制、谄媚和信念修订长期以来大体被分别研究。三者的交叉提出了一个比"干预是否改变行为"更难的问题：干预是否仍能区分因好理由而改答和因坏理由而改答？这种性质不能从激活方向的语义中推出，仅观察答案翻转也无法识别它。下面的相关工作围绕这一区别展开。

### 2.1 From Activation Directions to State-Dependent Control

Representation engineering and contrastive activation methods show that low-dimensional directions can read and manipulate behaviors such as truthfulness and refusal (Zou et al. 2023; Turner et al. 2023; Li et al. 2023; Rimsky et al. 2024; Arditi et al. 2024). Yet a direction is only one component of an intervention. Persistent addition, ablation, projection, and thresholded rectification act differently across layers, token positions, and model states. Consequently, the semantic label attached to a direction predicts neither the selectivity nor the collateral effects of the operator built from it.

> **中文：** 表征工程和对比激活方法表明，低维方向可以读取并操控真实性、拒答等行为（Zou et al. 2023；Turner et al. 2023；Li et al. 2023；Rimsky et al. 2024；Arditi et al. 2024）。但方向只是干预的一部分。持续加法、消融、投影和阈值整流会在不同层、token 位置和模型状态上产生不同作用。因此，一个方向的语义标签既不能保证控制器的选择性，也不能预测它的连带影响。

The Assistant Axis makes this gap concrete. Lu et al. (2026) identify a dominant persona direction, use its projection to track persona drift, and release a one-sided cap that keeps activations within an Assistant-like region. Their experiments establish persona stabilization in multi-turn and adversarial persona settings. They do not, however, establish that the cap preserves task reasoning or warranted belief revision. This distinction is essential because the cap is state-dependent: it fires only after a layer-specific boundary is crossed, with a correction determined by the triggering state. It may therefore intervene during prompt processing, deliberation, or answer formation even when persona behavior appears stable. Results for continuous steering cannot determine the effects of this thresholded operator, and persona stability is not evidence of epistemic selectivity.

> **中文：** Assistant Axis 将这一缺口具体化。Lu et al.（2026）识别出一个主导 persona 方向，用其投影追踪人格漂移，并发布了一个单侧 capping 算子，将激活限制在更接近 Assistant 的区域。原文实验在多轮交互和对抗性 persona 场景中证明了人格稳定作用，但没有证明该 capping 会保留任务推理或有根据的信念修订。这一点至关重要，因为 capping 是状态依赖的：只有隐藏状态越过逐层边界时才触发，修正量还取决于触发状态。因此，即使 persona 行为保持稳定，它也可能已经干预了提示处理、思考过程或答案形成。持续 steering 的结果不能预测这种阈值算子的作用，人格稳定也不等于认识论选择性。

Conditional steering and side-effect controls improve *input-level* selectivity: CAST gates steering by context, while KL-then-steer and SCANS constrain changes on benign inputs or general capabilities (Stickland et al. 2024; Lee et al. 2025; Cao et al. 2025). These methods answer where an intervention should apply, not why an answer should change within the same task. Broad accuracy can remain stable while relative answer logits move below the decision threshold. Utility retention can therefore miss a systematic shift in the model's update criterion.

> **中文：** 条件 steering 和副作用控制提高了输入层面的选择性：CAST 根据上下文决定是否施加 steering，KL-then-steer 和 SCANS 则约束对无害输入或一般能力的影响（Stickland et al. 2024；Lee et al. 2025；Cao et al. 2025）。但这些方法回答的是干预应该在"哪里"作用，而不是同一道任务中的答案应该"因为什么理由"改变。整体准确率可以保持稳定，而候选答案之间的相对 logit 已在决策阈值以下系统移动。因此，仅检查效用保持可能漏掉模型更新准则的变化。

> **【改动 1】** `Lee et al. 2024` → **`Lee et al. 2025`**。已核对 arXiv:2409.05907 的元数据：`comment = ICLR 2025, Spotlight`。正文其他位置（§3、reference list）已按 2025 引用，此处原本不一致。**提交前请全稿 grep 一遍 `Lee et al. 2024`。**

### 2.2 Why Existing Revision Readouts Are Not Identifying

Sycophancy studies show that unsupported user opinions can displace answers a model would otherwise give (Sharma et al. 2024). Wang et al. (2026) further locate opinion-conditioned divergence in middle-to-late layers using logit lens and activation patching. However, their prompt contrast changes target mention, speaker relation, and social framing together. It localizes where an opinion influences output, but cannot distinguish evidence-based persuasion from social accommodation or target copying. MUSE and SycoBench-600 improve the behavioral picture by relating uncertainty to yielding and separating resistance to incorrect suggestions from acceptance of corrections (Guo et al. 2026; Sinha et al. 2026). Still, uncertainty is not a certificate of logical validity, and providing the correct option is not equivalent to providing a sufficient reason.

> **中文：** 谄媚研究表明，没有证据支持的用户意见也可能使模型偏离原本答案（Sharma et al. 2024）。Wang et al.（2026）进一步利用 logit lens 和 activation patching，将受意见影响的分歧定位到中后层。然而，其 prompt 对照同时改变了目标答案是否出现、说话者关系和社会框架，因此只能定位用户意见在哪里影响输出，不能区分基于证据的说服、社会迎合和目标复制。MUSE 和 SycoBench-600 通过研究不确定性与让步的关系，以及分别测量错误建议抵抗与正确修正接受，改善了行为分析（Guo et al. 2026；Sinha et al. 2026）。但不确定性并不认证逻辑有效性，直接给出正确选项也不等于提供了充分理由。

Mechanistic findings further weaken a simple one-axis explanation. Skeptical persona vectors can reduce sycophancy, but agreeable vectors are not their mirror image and persona directions need not align with a supervised sycophancy direction (Kelkar et al. 2026). Anti-sycophancy steering can also suppress correct factual agreement despite separable internal representations (Buchan et al. 2026). Thus, a property may be readable without being selectively writable through the same direction. These studies do not predict whether a state-dependent Assistant Axis cap changes evidence discrimination, deliberation, commitment to an existing answer, or the final answer map.

> **中文：** 机制研究进一步削弱了简单的单轴解释。怀疑者 persona 向量可以降低谄媚，但顺从型向量并不是它们的行为镜像，persona 方向也未必与监督学习得到的谄媚方向对齐（Kelkar et al. 2026）。即使内部表征可以区分事实性同意与谄媚性同意，反谄媚 steering 仍可能同时压低正确的事实性同意（Buchan et al. 2026）。因此，某种性质可以被读取，却未必能沿同一方向被选择性写入。这些工作仍无法预测状态依赖的 Assistant Axis capping 改变的是证据分辨、思考过程、对已有答案的承诺，还是最终答案映射。

The problem is compounded by discrete readouts. Accuracy, flip rate, and target following threshold a continuous answer distribution: a broad loss of initial-answer advantage, target-specific attraction, and diffuse redistribution can produce the same flip. The same continuous shift can also look beneficial, harmful, or null at different decision boundaries. Prior work has inspected layerwise logits, but to our knowledge intervention audits have not paired item-level transitions with continuous answer geometry under the same operator and matched reasons. We call this failure mode the *margin–choice dissociation*; it makes reason-selectivity unidentifiable rather than merely unmeasured, and Section 5 shows that it is not a hypothetical concern for the operator we audit.

> **中文：** 离散读数又放大了这一问题。准确率、翻转率和 target-following 都把连续答案分布压缩成跨阈值结果：广泛削弱初始答案优势、专门吸引模型走向目标，以及把概率分散到无关选项，都可能产生同样的翻转。同一个连续位移在不同决策边界上也可能分别表现为有益、有害或无效。既有工作虽然观察过逐层 logit，但据我们所知，尚无干预审计在同一算子与严格配对的理由下同时报告逐题迁移与连续答案几何。我们把这一失效模式称为 **margin–choice 分离**；它使理由选择性变得"不可识别"而非"尚未测量"，而第五节表明它对我们所审计的算子并非假想问题。

> **【改动 2】** 三处。(a) `rarely pair` → `to our knowledge ... have not paired`——`rarely` 是一个无引文的综述性断言，审稿人会要求举证；改成认识论上可辩护的表述。(b) 给 margin–choice dissociation **命名**（斜体首现），它是本文的术语，不能写得像领域共识。(c) 加一句前指第五节——**这是把 Related Work 从"文献综述"变成"我们贡献的伏笔"的关键一句**，否则读者不知道这一段是在铺自己的发现。

### 2.3 Certifying Reason Quality for Intervention Audits

Identifying reason-selectivity therefore requires items in which the validity of a reason is certified independently of both the audited model and the annotator who wrote it. Belief-R formalizes update-versus-retain decisions, whereas LoFa supplies natural, persuasive fallacious arguments (Wilie et al. 2024; Shen et al. 2026). The former can conflate revision with base-task difficulty; the latter intentionally varies factual content, rhetoric, and argument form, which is useful for stress testing but prevents a strictly matched valid/invalid contrast. ProverGen provides checkable logical worlds and proof chains, while K-CAST separates formal validity from content plausibility in conditional steering (Qi et al. 2025; Valentino et al. 2026). None alone supplies a matched audit in which the same proposed answer is supported either by a certified valid warrant or by an argument with one certified fatal step.

> **中文：** 因此，要识别理由选择性，就需要这样的题目：其理由是否有效，由独立于被审计模型、也独立于命题者的第三方认证。Belief-R 形式化了更新与维持的决策，LoFa 则提供自然且有说服力的谬误论证（Wilie et al. 2024；Shen et al. 2026）。前者可能把信念修订能力与基础任务难度混在一起；后者会有意共同改变事实内容、修辞和论证形式，适合压力测试，却无法形成严格配对的有效/无效理由对照。ProverGen 提供可验证的逻辑世界和证明链，K-CAST 则在条件 steering 中区分形式有效性与内容合理性（Qi et al. 2025；Valentino et al. 2026）。这些资源单独来看，都没有提供这样一种配对审计：同一个候选答案，要么由经认证的有效理由支持，要么由只含一个经认证致命错误的论证支持。

> **【改动 3】** 补首句连接。原版从上一节的"离散读数不可识别"直接跳到"Belief-R 形式化了……"，中间缺一步推理。这句同时把 **solver 的科学理由**（独立于被审计模型 *和* 标注者）提前，而不是把求解器只写成质量控制手段——这是 Codex 上一轮批评里最重要的一条，短版没有落实。

CEDAR supplies this instrument for the released, model-specific Assistant Axis cap; we do not claim a universally missing evaluation layer, but a comparison that is not identifiable with existing readouts. Within each solver-certified contrast cluster, it holds the world, proposed edit, target, reviewer identity, speech act, and output contract fixed while changing only warrant validity; a separate contrast holds target exposure fixed while adding social pressure. Four-way outcomes are paired with margins that separate leaving the initial answer from target-specific attraction. This design tests whether capping changes evidence discrimination, the general criterion for revision, or sensitivity to social framing—properties that persona stabilization, axis semantics, and answer flips cannot identify on their own.

> **中文：** CEDAR 为官方发布、模型特定的 Assistant Axis capping 提供了这一工具；我们并不宣称发现了一个普遍缺失的评价层，只主张这项比较用现有读数无法识别。在每个求解器认证的对照簇中，基础世界、候选编辑、目标答案、复核者身份、言语行为和输出约束全部保持不变，只改变理由是否有效；另一组对照固定目标暴露，仅增加社会压力。我们同时报告四类离散结果与连续 margin，以区分"离开初始答案"和"被特定目标吸引"。这一设计能够检验 capping 改变的是证据分辨、一般改答准则，还是社会框架敏感性——这些性质无法由人格稳定、轴的语义或答案翻转单独识别。

> **【改动 4】** 从长版把 `we do not claim a universally missing evaluation layer` 这个 hedge 搬进来（压成一个分句）。短版删掉它其实很危险：本节前面已经连续说了五个"现有工作做不到"，不加这一句，reviewer 会读成 grand claim 并逐条反驳。加上之后，语气从"我们发现了空白"变成"我们把一个比较做成可识别的"，**更难被攻击且更准确**。

This framing leads to a narrower but stronger empirical question: does the released cap act selectively on the epistemic content of a proposed revision, or does it move answer geometry in a reason-insensitive manner whose discrete consequences depend on the local decision configuration? Section 3 turns that question into two orthogonal contrasts and a solver-certified measurement protocol.

> **中文：** 由此得到一个范围更窄但更有力度的实证问题：官方 capping 是否会依据候选修订的认识论内容进行选择性作用，还是仅以对理由真假不敏感的方式移动答案几何、并使离散后果取决于局部决策状态？第三节将这一问题落实为两组正交对照和一套求解器认证的测量协议。

> **【改动 5】** 长版结尾这一段**必须保留**，短版把它删掉是这两版之间唯一实质性的损失。它做了三件短版没做的事：(a) 把整节收束成一个**可证伪的二选一问题**，而不是"我们造了个工具"；(b) 它直接就是我们的主线结论（reason-insensitive 的连续压缩 + 边界依赖的离散后果）的伏笔，读者读到第五节时会有"果然如此"的感觉；(c) 给出通向 §3 的连接句。**没有这一段，第二部分仍然像 benchmark 论文；有了它，第二部分是一个 hypothesis statement。**

---

## 【改动 6】提交前必须核对的事项（不改文字，但会致命）

| # | 项目 | 说明 |
|---|---|---|
| 1 | `Lee et al. 2024` / `2025` | 全稿统一为 **2025**（ICLR 2025 Spotlight，arXiv:2409.05907）。 |
| 2 | ProverGen 作者姓氏 | reference list 里若仍是 `Lai, Y.` 需改为 **`Laili, Y.`**（arXiv:2502.06563 第七作者 Yuanjun **Laili**）。正文 `Qi et al. 2025` 是对的（第一作者 Chengwen Qi）。 |
| 3 | 2026 年条目 | Wang / Guo / Sinha / Kelkar / Buchan / Shen / Valentino / Lu 共 8 条 2026 引用，逐条确认 `references.bib` 中有 entry 且 arXiv 号正确——本节的说服力完全建立在这批新工作上，错一条就是硬伤。 |
| 4 | 前指 `Section 5` | 若结果章不是第 5 节，改成正确编号，或用 `\Cref` 交叉引用。 |
| 5 | 数字口径 | 数据集图上的 `24 families · 48 bases · 288 arms` 与 §3 的 `24 bases = 12 families × {easy, medium}` **对不上**。图必须改成后者（我重画的三版都已改正）。 |

---

## 总评

短版的判断是对的，正文用它。它相对 v3 的真正进步在于：**Related Work 第一次是在论证一个命题（方向 ≠ 控制器；离散读数不可识别），而不是在点名一串方法。** 这正好回应了你说的"读起来像个无聊的审计论文"——病根就在于此前每一段都在介绍别人做了什么，而不是在逼出一个只有我们能回答的问题。

上面 5 处改动里，**改动 2（命名 + 前指第五节）和改动 5（结尾的锐化问题）是量级最大的两处**：它们把这一节从"我们造了个工具"改写成"我们提出并将要检验一个二择一假设"。改动 3、4 是防御性的，成本极低但能挡掉审稿人两类最常见的攻击（"求解器只是质检"、"overclaim"）。
