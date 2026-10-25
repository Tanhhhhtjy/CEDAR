# §5 / §6 / Conclusion 修订稿（中英对照）— 未推送

> 结论先说：**§5 只做 4 处小改（含标题）**；**§6 大改**——把控制器从"我们已经做完的实验"降格为"由证据推出的假说 + 可执行的技术路线"，删掉全部 12 个 `TBD-L1/L2-*` 数字；**Conclusion 全部重写**。
> 净效果：去掉一整段假结果（约 11 行），换来 §6.2 的假说段和更高的收束，**总长度略减**。RG1 数字一到就填进 §5.5，不影响本稿。

---

## 一、§5 的问题（先说事实核对）

数值我逐条对过，**没有错误**：

| 检查 | 结果 |
|---|---|
| 固定轨迹 $2.078-0.700=1.378$ ↔ 交互 $-1.378$ | ✅ |
| $2.078/3.130=66.4\%$ ↔ "约 66%" | ✅ |
| 表：$68.75-50=+18.75$；$70.83-50=+20.83$；full $=+18.75$ | ✅ |
| "full 小于两个分阶段之和"：$18.75<39.58$ | ✅ |
| 有效性判别 $20.83/41.67/29.17$ vs $29.17$ → $-8.33/+12.50/0.00$ | ✅ |
| 引言里 "$+2.078$ 和 $+0.700$"、"192 个 argmax 无变化" | ✅ 与正文一致 |

问题不在事实，在**叙述**：§5 是全文最有戏剧性的一章（我们追了一个解释，它塌了），但现在写得像方法清单。四处改动如下。

### 改动 5-1｜章标题

`Diagnosing the Shift` → **`The Shift Does Not Pass Through Visible Reasoning`**
> 中文：`诊断这种位移` → **`位移并不经由可见推理发生`**

理由：§4 已经改成了以发现命名（"Stable Answer Geometry, Divergent Revisions"），§5 却退回动作命名，读起来像附录。新标题直接把本章最硬的结果（固定轨迹重放）放上台面。

### 改动 5-2｜开头：从"我们又做了五件事"改成"只剩两种解释"

**现在：** "We now examine this shift through phase-restricted interventions, a separate validity judgment, comparisons of resist and yield trajectories, fixed-trace replay, and downstream representation analysis."（五项清单）

**改为：**

> Section~4 leaves two accounts standing. The cap may remove something the model uses to resist an induced target, in which case the shift should track resistance and should disappear once the reasoning that produces it is held fixed. Or it may displace answer geometry without reading the reasoning at all, in which case neither should hold. The following diagnostics were designed to tell these apart: we restrict the intervention by phase, elicit an independent validity judgment, compare resist and yield trajectories, replay frozen reasoning traces, and track how a local correction propagates downstream.

> **中文：** 第 4 节之后只剩下两种解释。其一，钳制移除了模型用来抵抗诱导目标的某种东西——若如此，位移应当与"抵抗"同步，并且在产生抵抗的那段推理被固定之后消失。其二，钳制根本没有读推理，只是移动了答案几何——若如此，上述两点都不成立。下面的诊断正是为区分二者而设计：按阶段限制干预、单独获取有效性判断、比较抵抗与顺从轨迹、重放冻结的推理轨迹，并追踪局部修正如何向下游传播。

**+约 35 词。** 这样后面每一个小节都在回答同一个问题，而不是并列的五个实验。

### 改动 5-3｜§5.3 标题与首句：让"塌掉"这件事看得见

标题 `Testing a Selective-Resistance Account` → **`A Resistance Signal That Does Not Survive Length Matching`**
> 中文：`检验选择性抵抗解释` → **`一个经不起长度匹配的抵抗信号`**

并在段首加一句承接（**+18 词**）：

> The first look supported it. Five initial-to-target cases had greater uncapped aggregate-decode correction …（其余不动）

段尾把已有的 `The trajectory evidence does not support this account.` 提到长度匹配那句**之后紧接**（现在就在，只需确认它不被 `[TBD-APPENDIX]` 那句挤开）——即把 "Additional phase/layer … appear in [TBD-APPENDIX]." 移到裁决句之后。

> **中文段首：** 最初的迹象是支持它的。五个由初始答案迁移到目标的案例……
> 段尾顺序：先"轨迹证据不支持该解释"，再"其余诊断见附录"。

理由：1,039 vs 343 thinking tokens、长度匹配后从 $-1.146$ 翻到 $+0.033$——这是全文最能体现研究品味的一处（我们自己找出了混淆并把自己的假说否掉了），现在却藏在从句里。

### 改动 5-4｜§5.4 把裁决句提前

现在的顺序是：主估计 → 次要分解 → argmax → 66% → 三句限定 → **裁决句** → 又两句限定。

改为：把 `The continuous shift therefore persists with an identical visible reasoning sequence, while every paired four-option argmax was unchanged.` 提到 66% 那句**之前**，限定语句全部留在段尾。**零词数变化。**

### 改动 5-5｜章末收束（改 1 句）

**现在：** "The phase-restricted and fixed-trace results show that capping shifts answer geometry under several controlled conditions; the independent validity assay does not yield a stable parallel pattern."

**改为：**

> Neither account survives intact. The shift does not track resistance, and it does not require the reasoning that precedes the answer; what it does require remains unlocalized. Section~6 takes this as the starting point for control design rather than as a residual puzzle.

> **中文：** 两种解释都没有完整存活。位移既不追随"抵抗"，也不需要答案之前的那段推理；而它究竟需要什么，仍未被定位。第 6 节把这一点当作控制器设计的起点，而不是一个遗留谜题。

**+约 10 词。**

---

## 二、§6 大改：把控制器从"结果"降格为"假说 + 路线"

你的判断我完全同意，而且比我原来的想法更对：**现在这版 §6.2 用 12 个占位符报告了一个尚不存在的实验，并据此宣称 "provides a proof of concept"。** 这是全文最大的可信度风险——审稿人一旦发现控制器结果是空的，会回头怀疑前面所有实数。删掉它，我们**反而**得到一个更强的第六节：我们不是"顺手做了个修复"，而是"这项证据逼出了一个控制器必须满足的契约，并且我们把评测规则先冻结了"。

### 新的 §6 结构

| 现在 | 改后 |
|---|---|
| §6 标题 `Implications, Scope, and Future Control` | **`Representation Is Not Policy`**（把最好的小标题提为章标题） |
| 6.1 Representation Is Not Policy | 并入章首，压缩约 2 行（第 1、3 段有重叠） |
| — | **新 6.1 `What the Cap Is Doing`**：把四次反转收成一个可证伪假说 |
| 6.2 What a Selective Controller Must Preserve（含 12 个 TBD 数字） | **6.2 `A Contract for Selective Control`**：保留 $u_\perp$ 公式与 L0/L1/L2 阶梯作为**设计空间**，保留 matched-benefit 规则作为**预注册评测契约**，**删除全部结果句** |
| 6.3 Evidence Scope and Limitations | 保留，删掉两处 "proof of concept" 措辞 |
| 6.4 Future Work | 并入 6.2 末尾（省约 4 行） |

### 新 6.1（新增，约 110 词）

> \subsection{What the Cap Is Doing}
>
> Four results constrain the same object. The displacement appears when the target is hidden ($N$, $R$), so it is not an attraction toward a proposed answer. It appears under valid and fallacious warrants alike, so it is not a judgment about reasons. It does not track resistance, and it survives a frozen reasoning trace, so it is not produced by the deliberation that precedes the answer. Yet its discrete consequences reversed between two banks built from the same grammar. The account consistent with all four is narrow: the cap applies a reason-insensitive displacement to answer geometry, and whether that displacement changes an answer is decided by the local decision configuration, not by the intervention. This is falsifiable. A bank engineered with uniformly large initial gaps should show the margin shift with almost no answer changes; one with uniformly small gaps should show many, in both directions, with no dependence on warrant validity.

> **中文：** 四项结果约束着同一个对象。目标被隐藏时（$N$、$R$）位移依然出现，因此它不是"被某个候选答案吸引"。有效与谬误理由下位移同样出现，因此它不是关于理由的判断。它不追随抵抗，也能在冻结的推理轨迹下存活，因此它不是由答案之前的推理产生的。然而，在同一套语法生成的两个 bank 之间，它的离散后果却发生了反转。同时符合这四点的解释范围很窄：钳制对答案几何施加了一个对理由不敏感的位移，而该位移是否改变答案，由局部决策状态决定，而不由干预决定。这一说法可以被证伪：一个初始差距被刻意做大的 bank 应当只显示 margin 位移而几乎没有答案变化；一个初始差距被做小的 bank 则应当出现大量双向的答案变化，且与理由有效性无关。

> ▸ 这一段是你要的"研究品味 + 技术深度"的落点：它把散在四章里的负结果第一次合成一个正面主张，并自带一条别人可以来打的证伪路线。也正好与 §2 结尾那个二择一问句闭环。

### 新 6.2（保留公式与阶梯，删掉所有结果）

前半段（$u_\perp$、L0/L1/L2 定义、冻结协议）**一字不动**。之后把整个结果段替换为：

> We state the comparison before running it. The primary test is made at a prespecified matched-benefit operating point: a variant must retain the same CEDAR-Social attenuation, or the same independently measured persona benefit, before any selectivity gain counts. Within that constraint, a selective controller should preserve or increase target following under valid reasons, avoid increased following under fallacious reasons, and reduce broad movement when the target is absent, while persona safety, ordinary accuracy, output integrity, trigger exposure, and compute are reported jointly. The ladder is a hypothesis about where selectivity can be recovered---at the actuation direction (L1) or at the trigger itself (L2)---and the contract above is what would decide between them. We publish the contract with the audit so that the two can be judged separately: if protecting an evidence-sensitive subspace fails, the failure is informative about whether epistemic selectivity is available in the linear regime at all.

> **中文：** 我们在实验之前先把比较规则写死。主检验在预先规定的 matched-benefit 工作点上进行：任何变体必须先保住同等的 CEDAR-Social 衰减，或同等的、独立测量的 persona 收益，其选择性增益才被计入。在该约束下，选择性控制器应当在有效理由下保持或提高 target following，在谬误理由下不提高 target following，并在目标缺席时减少广泛位移；同时联合报告 persona 安全性、常规准确率、输出完整性、触发暴露与算力开销。这一阶梯是一个关于"选择性可以在哪里被找回"的假说——在作动方向上（L1），还是在触发规则本身（L2）——而上述契约正是用来在二者之间做出裁决的。我们把契约与审计一并公开，使两者可以被分别检验：如果保护证据敏感子空间失败了，这一失败本身就能说明，认识论选择性在线性范围内是否根本不可得。

> ▸ 关键措辞：`We state the comparison before running it.` / `The ladder is a hypothesis`。既不撒谎，也不示弱——预注册一个自己尚未跑完的比较，在审计类工作里是加分项而不是减分项。末句把"我们可能失败"转成"失败也有信息量"，这是拔高而不是虚张。

### 6.3 需要同步去掉的两处措辞

- `The controller comparison likewise establishes a proof of concept for the evaluated configurations` → `The controller ladder is specified but not yet evaluated; nothing here establishes that a protected subspace transfers across architectures.`
- 6.4（并入 6.2 后）里的 `whether the proof-of-concept improvement reflects a reusable control principle` → `whether protection at the trigger recovers selectivity at all`

---

## 三、Conclusion 全部重写

**问题：** 现在这段是"我们做了 A，测了 B，得到 C、D、E"的流水账，六句话里五句是结果复述，最后一句 `a property that activation-control studies should evaluate directly` 才碰到一般性主张——而且是以建议的语气。读完全文的人在这里应该拿到一句能被引用的话，而不是一份摘要。

**改为：**

> Activation control is usually certified in two steps: a direction is shown to represent a state, and an operator is shown to move that state. CEDAR asks for a third. On the released Assistant-Axis p25 cap in Qwen3-32B, the first two hold and the third does not follow from them: the cap broadly compressed initial-answer advantage whether or not a target was shown, raised warranted and unwarranted updating together in one bank and separated them in the opposite direction in another, and preserved its continuous effect through a frozen reasoning trace without moving a single paired argmax. What changed was answer geometry; what did not change was the model's grip on why an answer should change. \emph{[TBD-XM-CONCLUSION-ROUTE]} We do not read this as a defect of one operator. It is what happens when a control rule is inherited from a representation without being asked whether it preserves the distinctions the deployed policy depends on---and epistemic selectivity, unlike persona stability, does not announce its own absence.

> **中文：** 激活控制通常按两步获得认可：先证明某个方向表示某种状态，再证明某个算子能移动该状态。CEDAR 要求第三步。在 Qwen3-32B 的公开 Assistant-Axis p25 钳制上，前两步成立，而第三步并不能由它们推出：无论目标是否展示，钳制都广泛压缩了初始答案优势；在一个 bank 中它同时抬高了有据与无据改答，在另一个 bank 中又以相反方向把两者分开；而在冻结的推理轨迹下，它的连续效应依然保留，却没有改变任何一个配对 argmax。改变的是答案几何；没有改变的是模型对"答案为何应当改变"的把握。\emph{[TBD-XM-CONCLUSION-ROUTE]} 我们并不把这看作某一个算子的缺陷。当一条控制规则直接由某个表示继承而来、却从未被追问它是否保住了部署策略所依赖的那些区分时，这就是会发生的事——而认识论选择性不同于人格稳定性，它的缺失不会自己发出声音。

> ▸ 最后一句 `epistemic selectivity, unlike persona stability, does not announce its own absence.` 是我给这篇论文准备的"可被引用的一句话"，也正好呼应 §2.1 的 `Persona stability is therefore not evidence of epistemic selectivity.`——开头提出、结尾兑现。

---

## 四、长度账

| 项 | 变化 |
|---|---|
| §5 四处（开头 +35、5.3 首句 +18、章末 +10） | **+63 词** |
| §6 删除 12-TBD 结果段 | **−185 词** |
| §6 新增 6.1 假说段 | **+140 词** |
| §6.2 新收束段（替换旧结果段的剩余部分） | **+35 词** |
| §6.1 原文压缩 + Future Work 并入 | **−70 词** |
| Conclusion 重写 | **+20 词** |
| **合计** | **约 −0 到 +5 词，等长** |

RG1 的数字到了直接填 §5.5 的五个占位符，不动本稿任何一句。

**要我现在就把这些改动落到 `main.tex` 和 `main_zh.tex` 里吗？（改完仍不推送，你看过再说。）**