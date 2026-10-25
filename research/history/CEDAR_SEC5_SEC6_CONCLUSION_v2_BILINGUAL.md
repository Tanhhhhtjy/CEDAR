# §5.5（RG1-P 实数）+ §6 重写 + Conclusion — 中英对照，未推送

> 先回你的问题：**Codex 这轮反馈我同意约 90%，而且它纠正的两处是我错了，不是它保守。** 下面先说分歧，再给正文。

---

## 一、我同意 Codex 的地方（其中两处是它纠了我）

**① 随机参考的"天花板"问题——它对，我错了。**
我之前写"随机方向做不出这种轴外重混"。实际上四个随机参考在第 53 层就已经接近 100% 轴外，$G_{\mathrm{off}}$ 只有 $10^{-5}$ 量级是**没有增长空间**，不是"没有重混"。$G_{\mathrm{off}}$ 度量的是"离开干预窗口之后还在继续增长多少"，所以正确的结论是**传播形态不同**，不是**只有 Assistant Axis 会被重混**。而且只有四个方向，是 reference battery 不是零分布，必须沿用协议措辞 `four preregistered schedule- and norm-matched random-direction references`。这条我全盘接受。

**② 探针不能写成"模型知道理由是错的却故意不用"——它对。**
两条件是**分别训练**的探针，没有做等价性检验，且 4/12 个 family 的 balanced accuracy $\le 0.5$。能说的只有"两种条件下有效性都可线性解码，观测 AUC 相近"。要说"信息在但策略不用"，需要跨条件探针迁移或 activation patching。

**③ $\rho=-0.175$ → 不是因果中介**、**④ 跨环境锚点失败必须披露**、**⑤ §5 报事实 / §6 提假说的分工** —— 全部同意。第 ⑤ 条尤其好：它给了第六节一个**有名字的贡献**，正好补上删掉 L1/L2 假结果留下的空缺。

## 二、我要补的三点（Codex 漏了）

**① 0.762 这个数字不能裸写。**
Codex 建议的段落直接写 "increased from 0.762 at layer 53"，读者会理解成"一维算子在自己的出口层就已经 76% 泄漏了"——这是个更抢眼的说法，但**不成立**：第 46–53 层用的是逐层轴，而这里的分解用的是**固定的 layer-53 出口方向**，所以那 76% 里有一部分只是跨层轴不对齐，不是传播泄漏。可比的量只有**差值** $G_{\mathrm{off}}$。我在正文里把 0.762/0.922 保留为过程量，但把主张挂在差值上。

**② $\rho=-0.175$，$n=12$，没有检验力。**
应写 "no association was detected"，不能写 "there is no association"。

**③ 探针不进摘要，但要进图注。** 它是 §6 假说的唯一直接证据来源，正文一句话带过会让第六节的假说悬空。

**④ 命名。** 我接受 `Representational--Policy Decoupling`，但正文里要与 §2 的 *margin--choice dissociation* 形成层级：margin–choice 是**现象**，representational–policy decoupling 是**解释该现象的机制假说**。两个自造术语，一个描述、一个解释，这是我们这篇论文最像"理论工作"而非"审计"的地方。

---

# 三、正文

## §5.5（替换现有全 TBD 的那一节）

标题：`From Local Rectification to Downstream Geometry` → **`A One-Dimensional Correction Does Not Stay One-Dimensional`**
> 中文：`从局部轴向修正到下游几何` → **`一维修正不会保持为一维`**

> The cap is a one-dimensional operation: it removes above-threshold projection along a single direction at layers 46--53. Whether the resulting difference is still one-dimensional by the time the model commits to an answer is a separate question, and it is the one that decides whether a controller can be specified in that direction at all. On fixed uncapped-source traces we resolved the paired uncapped-minus-p25 hidden-state displacement against the fixed layer-53 exit direction and preregistered the change in off-axis energy share between layers 53 and 60 as $G_{\mathrm{off}}$.
>
> The displacement moves off the direction that produced it. The off-exit-axis share rose from $0.762$ at the cap-window exit to $0.922$ at layer 60, giving $G_{\mathrm{off}}=0.161$ (95\% family-cluster bootstrap CI $[0.151,0.170]$). All 12 families were positive, every leave-one-family-out estimate lay within $[0.158,0.163]$, and all four arms agreed ($N$ $0.145$, $R$ $0.142$, $V$ $0.184$, $F$ $0.172$). The profile exceeded four preregistered schedule- and norm-matched random-direction references, which had already reached the off-direction ceiling at the window exit; the comparison therefore identifies a different propagation profile, not an absence of rotation under random perturbation. Because layers 46--53 use layer-specific axes while this decomposition uses one fixed direction, the level at layer 53 is not itself a leakage measurement---only the layer-53-to-60 change is.
>
> Two further observations bound what this means. The margin displacement recurred here ($\Delta m_{\mathrm{general}}=1.339$, $[0.911,1.754]$), but across the 12 families no association was detected between $G_{\mathrm{off}}$ and the margin shift ($\rho=-0.175$; 12 clusters afford little power). Remixing is therefore what we observed, not what we showed to be responsible. Separately, an auxiliary family-held-out probe found $V/F$ validity linearly decodable from layer-60 states under both uncapped and full-p25 inference (ROC-AUC $0.769$ and $0.767$; family-label-flip $p=0.014$ and $0.002$, 4{,}096 permutations), while a text-only TF-IDF baseline stayed at chance ($0.511$). This analysis is auxiliary and exploratory, and the two probes were trained separately, so it establishes decodability under each condition rather than a shared code or a causal role.
>
> Strict token-level numerical equivalence with the earlier environment was not obtained; all RG1 contrasts were accordingly reconstructed and analyzed within a single fresh stack, where the formal audit passed on 864 of 864 records.

> **中文：** 钳制是一个一维操作：它在第 46–53 层沿单一方向削去超过阈值的投影。而当模型最终给出答案时，由此产生的差异是否仍然是一维的，则是另一个问题——并且正是这个问题决定了控制器能否在该方向上被定义。在固定的 uncapped 源轨迹上，我们把配对的 uncapped 减 p25 隐状态位移沿固定的 layer-53 出口方向分解，并预注册第 53 层到第 60 层之间轴外能量占比的变化作为 $G_{\mathrm{off}}$。
>
> 位移会离开产生它的那个方向。轴外占比从干预窗口出口处的 $0.762$ 升至第 60 层的 $0.922$，即 $G_{\mathrm{off}}=0.161$（95\% family-cluster bootstrap CI $[0.151,0.170]$）。12 个 family 全部为正，任意留一法估计都落在 $[0.158,0.163]$ 内，四个臂方向一致（$N$ $0.145$、$R$ $0.142$、$V$ $0.184$、$F$ $0.172$）。该形态超过了四个预注册的、调度与范数均匹配的随机方向参考；但这些参考在窗口出口处就已触及轴外占比上限，因此该比较识别的是**传播形态的差异**，而不是"随机扰动不会发生旋转"。又由于第 46–53 层使用逐层轴，而此处分解使用单一固定方向，第 53 层的水平值本身并不构成泄漏测量——只有 53→60 的变化量才是。
>
> 另有两点界定了它的含义。margin 位移在此环境中同样复现（$\Delta m_{\mathrm{general}}=1.339$，$[0.911,1.754]$），但在 12 个 family 上没有检测到 $G_{\mathrm{off}}$ 与 margin 位移之间的关联（$\rho=-0.175$；12 个聚类的检验力很低）。因此，重混是我们**观察到**的，而不是我们**证明其负责**的。另外，一个辅助的 family 留出探针发现，在 uncapped 与 full-p25 两种条件下，$V/F$ 有效性都能从第 60 层状态中被线性解码（ROC-AUC $0.769$ 与 $0.767$；family 标签置换 $p=0.014$ 与 $0.002$，4{,}096 次置换），而纯文本 TF-IDF 基线维持在随机水平（$0.511$）。该分析属辅助性探索，且两条件的探针是分别训练的，因此它确立的是各条件下的可解码性，而非共享编码或因果作用。
>
> 本轮未获得与先前环境的严格 token 级数值等同；因此 RG1 的全部对照都在同一套全新环境内重建并分析，其中形式化审计在 864/864 条记录上通过。

> ▸ 写法说明：主张句（"位移会离开产生它的那个方向"、"重混是我们观察到的，而不是我们证明其负责的"）在前，限定条件全部降格为从句或段尾——这样限定不会读起来像审计清单，反而像我们自己比审稿人更早知道边界在哪。

## §5 章末（替换现有收尾句）

> Taken together, the diagnostics do not localize the shift to a single intervention phase, do not support the selective-resistance account, show that changes in the visible reasoning tokens are not necessary for the continuous displacement, and show that the displacement does not remain in the direction that produced it. What survives all four is not a mechanism but a shape: a broad geometric movement that no longer carries the one-dimensional structure of the operator that caused it. Section~6 asks what that shape implies for control.

> **中文：** 综合来看，这些诊断没有把位移定位到单一干预阶段，不支持选择性抵抗解释，表明可见推理 token 的改变对于连续位移并非必要，也表明位移并不停留在产生它的那个方向上。在这四项之后仍然成立的不是一套机制，而是一个形状：一次广泛的几何移动，它已不再携带引发它的那个算子的一维结构。第 6 节追问这个形状对控制意味着什么。

---

## §6 全节重写

**章标题：** `Implications, Scope, and Future Control` → **`Representation Is Not Policy`**（中文：**`表征不等于策略`**）

### 6.1 What the Evidence Establishes

> A useful representation need not define a sufficient control rule. Our results separate three properties that activation control usually certifies as one. A direction can summarize a meaningful state: the Assistant Axis tracks persona drift, and $V/F$ validity is linearly decodable from layer-60 states. An operator can successfully move that state: the cap fires on every evaluation and shifts answer geometry across every condition we tested. Neither entails the third property, that the resulting behavior still respects the reason for changing an answer---and on the released configuration the third did not follow. Continuous shifts appeared when the target was hidden, valid and fallacious warrants were not separated at the answer level, identical visible reasoning was mapped to different answer geometry, and the one-dimensional correction did not remain one-dimensional downstream.
>
> Evaluation should therefore pair the intended safety or persona endpoint with reason-sensitive and integrity endpoints. A favorable change in one behavioral frequency, a smaller scalar margin, or unchanged benchmark accuracy cannot establish selectivity on its own. For an operator that acts during deliberation and answer formation, the contract includes what the model changes toward, why that change is warranted, and whether the output remains usable.

> **中文：** 一个有用的表征未必构成充分的控制规则。我们的结果把激活控制通常合并认证的三项性质分开了。方向可以概括某种有意义的状态：Assistant Axis 能追踪 persona 漂移，而 $V/F$ 有效性可从第 60 层状态中被线性解码。算子可以成功移动该状态：钳制在每次评测中都触发，并在我们测试的每个条件下移动了答案几何。这两点都不蕴含第三点——即改变之后的行为仍然尊重改答的理由；而在公开配置上，第三点并未随之成立。目标被隐藏时连续位移依然出现，有效与谬误理由在答案层面没有被分开，相同的可见推理被映射到不同的答案几何，而一维修正在下游并未保持为一维。
>
> 因此，评测必须把预期的安全或 persona 端点与理由敏感端点、输出完整性端点并列。某一项行为频率的有利变化、更小的标量 margin，或未受影响的基准准确率，都不足以单独确立选择性。对于一个作用于推理与答案形成阶段的算子，契约应当包括：模型改向什么、这一改变凭什么成立、以及输出是否仍然可用。

### 6.2 A Representational--Policy Decoupling Hypothesis（新增）

> Why should an operator that moves a persona state disturb epistemic selectivity at all? The results above motivate a specific answer, stronger than anything we have shown and precise enough to be attacked. Capping need not erase information about whether a reason is valid; the probe result is consistent with that information surviving intact. What the local rectification plausibly changes is not the content of the representation but the mapping from it to a revision decision: a correction applied along one direction is remixed through downstream computation into a higher-dimensional decision space, where it displaces answer geometry without preserving the route by which validity-sensitive structure reached the final choice. We call this \emph{representational--policy decoupling}. It stands in the same relation to the \emph{margin--choice dissociation} of Section~2 that a mechanism stands to a phenomenon: the dissociation is what a broad reason-insensitive displacement looks like from the outside.
>
> The hypothesis is not established here, and we state three predictions that would settle it. A probe trained on uncapped states should transfer to capped states and back; if it does not, the two conditions merely re-encode validity separately and the account fails at its first step. Patching validity-sensitive activations from uncapped into capped inference should partially restore $V-F$ selectivity; if nothing is restored, the decodable signal is a bystander. And a controller that protects the validity-sensitive subspace should improve $V-F$ selectivity at a matched persona benefit, rather than simply suppressing all revision.

> **中文：** 一个用来移动 persona 状态的算子，为什么会干扰认识论选择性？上述结果给出了一个具体的答案——它强于我们已证明的任何一点，同时又精确到足以被攻击。钳制不一定删除了关于理由是否有效的信息；探针结果与"该信息完好存续"是相容的。局部整流真正改变的，很可能不是表征的内容，而是**从表征到修订决策的映射**：沿单一方向施加的修正在下游计算中被重混进更高维的决策空间，在那里它移动了答案几何，却没有保住有效性敏感结构通往最终选择的那条路径。我们把这称为\emph{表征--策略解耦}。它与第 2 节的 \emph{margin--choice 分离} 之间的关系，正是机制与现象的关系：所谓分离，就是一次广泛且对理由不敏感的位移从外部看上去的样子。
>
> 该假说在此并未被确立，我们给出三条足以裁决它的预测。在 uncapped 状态上训练的探针应当能迁移到 capped 状态并可反向迁移；若不能，则两种条件只是各自重新编码了有效性，该解释在第一步即失败。把有效性敏感的激活从 uncapped 补丁到 capped 推理中，应当部分恢复 $V-F$ 选择性；若毫无恢复，则可解码信号只是旁观者。而一个保护有效性敏感子空间的控制器，应当在匹配的 persona 收益下改善 $V-F$ 选择性，而不是简单地压制一切改答。

### 6.3 From the Hypothesis to a Selective Controller

（保留 $u_\perp$ 公式与 L0/L1/L2 定义，**逐字不动**；把整个 12-TBD 结果段替换为下段，并把 Future Work 并入此处）

> The ladder follows from the hypothesis rather than from a search for fixes. If the damage is to the route rather than to the content, then protecting the route is the intervention to try: L1 keeps the released trigger but redirects actuation onto $u_{\perp}$, and L2 uses the protected decomposition for gating as well, so that evidence-sensitive movement is not itself sufficient to trigger removal. We state the comparison before running it. The primary test is made at a prespecified matched-benefit operating point: a variant must retain the same CEDAR-Social attenuation, or the same independently measured persona benefit, before any selectivity gain is counted. Within that constraint, a selective controller should preserve or increase target following under valid reasons, avoid increased following under fallacious reasons, and reduce broad movement when the target is absent, with persona safety, ordinary accuracy, output integrity, trigger exposure, and compute reported jointly. We publish the contract alongside the audit so the two can be judged separately: should linear protection fail under it, that failure is itself informative about whether epistemic selectivity is available in the linear regime at all.

> **中文：** 这一阶梯来自上述假说，而不是来自对补丁的搜索。如果受损的是路径而非内容，那么该尝试的干预就是保护路径：L1 保留公开的触发规则，但把作动重定向到 $u_{\perp}$ 上；L2 则把受保护的分解同时用于门控，使得证据敏感的移动本身不足以触发移除。我们在运行之前先把比较规则写死。主检验在预先规定的 matched-benefit 工作点上进行：任何变体必须先保住同等的 CEDAR-Social 衰减，或同等的、独立测量的 persona 收益，其选择性增益才被计入。在该约束下，选择性控制器应当在有效理由下保持或提高 target following，在谬误理由下不提高 target following，并在目标缺席时减少广泛位移；同时联合报告 persona 安全性、常规准确率、输出完整性、触发暴露与算力开销。我们把该契约与审计一并公开，使二者可被分别检验：若线性保护在此契约下失败，这一失败本身就能说明，认识论选择性在线性范围内是否根本不可得。

### 6.4 Evidence Scope and Limitations（改 3 处）

- 删 `The controller comparison likewise establishes a proof of concept for the evaluated configurations` → `The controller ladder is specified but not yet evaluated.`
- 删 `The L2 result therefore supports an improved operating point` → `The hypothesis therefore supports a design direction; circuit-level explanation and global optimality remain open.`
- 机制段补一句：`The propagation analysis characterizes how a local correction is remixed and detected no association between remixing and the margin shift across 12 families; only bidirectional patching can establish mediation.`

---

## Conclusion 全部重写

> Activation control is usually certified in two steps: a direction is shown to represent a state, and an operator is shown to move that state. CEDAR asks for a third. On the released Assistant-Axis p25 cap in Qwen3-32B, the first two hold and the third does not follow from them. The cap broadly compressed initial-answer advantage whether or not a target was shown; it raised warranted and unwarranted updating together in one bank and separated them in the opposite direction in another; it preserved its continuous effect through a frozen reasoning trace without moving a single paired argmax; and the one-dimensional correction it applies does not remain one-dimensional by the layer at which the answer is decided, even though validity itself remains decodable there. What changed was answer geometry. What did not change was the model's grip on why an answer should change. \emph{[TBD-XM-CONCLUSION-ROUTE]} We do not read this as a defect of one operator. It is what happens when a control rule is inherited from a representation without asking whether it preserves the distinctions the deployed policy depends on---and epistemic selectivity, unlike persona stability, does not announce its own absence.

> **中文：** 激活控制通常按两步获得认可：先证明某个方向表示某种状态，再证明某个算子能移动该状态。CEDAR 要求第三步。在 Qwen3-32B 的公开 Assistant-Axis p25 钳制上，前两步成立，而第三步并不由它们推出。无论目标是否展示，钳制都广泛压缩了初始答案优势；在一个 bank 中它同时抬高了有据与无据改答，在另一个 bank 中又以相反方向把两者分开；在冻结的推理轨迹下，它的连续效应依然保留，却没有改变任何一个配对 argmax；而它所施加的一维修正，到答案被决定的那一层已不再是一维的——尽管有效性本身在那里仍可被解码。改变的是答案几何。没有改变的是模型对"答案为何应当改变"的把握。\emph{[TBD-XM-CONCLUSION-ROUTE]} 我们并不把这看作某一个算子的缺陷。当一条控制规则直接由某个表示继承而来、却从未被追问它是否保住了部署策略所依赖的那些区分时，这就是会发生的事——而认识论选择性不同于人格稳定性，它的缺失不会自己发出声音。

---

## 长度账

| 项 | 变化 |
|---|---|
| §5 叙事重排（开头 +35、5.3 首句 +18、章末 +25） | **+78** |
| §5.5 由全 TBD 改为实数 | **+95** |
| §6 删除 12-TBD 结果段 + Future Work 并入 | **−250** |
| §6.2 新增假说节 | **+195** |
| §6.1 压缩重复的"三个问题" | **−60** |
| Conclusion 重写 | **+30** |
| **合计** | **约 +88 词（约 8 行）** |

如需回收：§6.3 的联合报告端点清单可压掉约 30 词，§5.2 可压掉约 25 词。

**确认后我就落到 `main.tex` 和 `main_zh.tex`，并把 RG1 的 5 个 `TBD-RG1-*` 占位符一并清掉。**