# Sections 5–6 and Conclusion — Revised Bilingual Draft (v3)

> 本文件给出修改后的完整英文正文及逐段中文对照，供统一审阅。  
> 主线：**候选解释被逐步排除 → 广泛答案几何位移成为经验不变量 → 表征—策略解耦成为新的可证伪假说 → 假说导出选择性控制器的设计契约。**

---

# 5 A Geometric Shift Survives the Diagnostic Chain

## English

Section 4 exposed a stable tension. The released p25 cap moved continuous answer geometry across \(N/R/V/F\), yet the corresponding discrete revisions changed direction across task banks. A natural first account was that capping impaired the evaluation of reasons or selectively interrupted a trajectory that would otherwise resist an induced answer. We subjected that account to successively stricter tests. Phase restriction asked when the shift can be produced; a separate validity assay asked whether reason evaluation moves with final choice; trajectory analysis tested the proposed resistance state; fixed-trace replay removed variation in visible reasoning; and representation analysis followed the intervention beyond the capped layers. Across this chain, one empirical pattern repeatedly survived: a broad displacement of the current answer relative to its alternatives.

## 中文

第 4 节揭示了一组稳定的张力：公开的 p25 钳制在 \(N/R/V/F\) 条件下持续移动答案的连续几何结构，但这种位移兑现为离散改答时，其方向却会随题库改变。最自然的初始解释是：钳制损害了模型对理由的判断，或者选择性地打断了原本能够抵抗诱导答案的推理轨迹。我们用逐步收紧的检验审查这一解释：阶段限制实验考察位移可以在何时产生；独立有效性任务检验理由判断是否与最终选择同步；轨迹分析直接检验“抵抗状态”；固定轨迹重放排除可见推理变化；表征分析则继续追踪干预离开钳制层后的传播。在整条诊断链上，一个经验模式反复存活下来：当前答案相对于其他选项发生了广泛位移。

---

## 5.1 The Shift Reappears in Either Phase

### English

To compare phase-restricted interventions on a common scale, we define CommonUpdate as the mean target-following rate across \(V\) and \(F\). Prefill-only and decode-only capping both increased CommonUpdate relative to uncapped inference, and each point estimate approached that of full-p25 capping (Table 1). Prefill-only increased CommonUpdate by \(18.75\) percentage points (95% family-cluster bootstrap CI \([4.17,33.33]\)); decode-only increased it by \(20.83\) points \([6.25,35.42]\); and full p25 increased it by \(18.75\) points \([4.17,33.33]\). The full-condition point estimate was substantially smaller than the arithmetic sum of the two restricted estimates. The shift is therefore accessible from either phase, while its point-estimate profile is inconsistent with a simple additive decomposition.

### 中文

为了在同一尺度上比较不同阶段的干预，我们把 \(V\) 与 \(F\) 的目标跟随率均值定义为 CommonUpdate。与 uncapped 相比，仅 prefill 和仅 decode 的钳制都会提高 CommonUpdate，而且两者的点估计都接近完整 p25（表 1）。仅 prefill 使 CommonUpdate 增加 \(18.75\) 个百分点（95% family-cluster bootstrap CI \([4.17,33.33]\)）；仅 decode 增加 \(20.83\) 个百分点 \([6.25,35.42]\)；完整 p25 增加 \(18.75\) 个百分点 \([4.17,33.33]\)。完整干预的点估计明显小于两个阶段效应的算术和。因此，任一阶段都足以重新产生主要位移，而其点估计形态并不符合简单的阶段可加分解。

---

## 5.2 Reason Evaluation and Revision Follow Different Profiles

### English

Final-choice movement was not mirrored by the independent validity assay. In a separate call, the model judged whether each \(V/F\) justification warranted the proposed edit and uniquely supported its target. We define validity discrimination as the sufficient-judgment rate for \(V\) minus that for \(F\). Its point estimate decreased by \(8.33\) percentage points under prefill-only capping, increased by \(12.50\) points under decode-only capping, and changed by \(0.00\) points under full p25; all three intervals included zero. In contrast, CommonUpdate increased under every capped condition.

The individual validity contrasts are imprecise, and we do not interpret their signs as phase-specific effects. Their joint profile is nevertheless informative: the large, common movement in final choice has no matching common movement in explicit reason evaluation. Full p25, for example, combined a \(+18.75\)-point CommonUpdate shift with a \(0.00\)-point change in validity discrimination. This divergence motivated a sharper hypothesis: capping might act on a state that supports resistance after a reason has been evaluated.

### 中文

最终选择的移动没有在独立有效性任务中出现对应模式。在另一次独立调用中，模型判断每条 \(V/F\) 理由是否足以支持所提出的修改，并且是否唯一支持目标答案。我们把 \(V\) 的“理由充分”判断率减去 \(F\) 的对应判断率定义为有效性分辨。仅 prefill 条件下，该指标的点估计下降 \(8.33\) 个百分点；仅 decode 条件下上升 \(12.50\) 个百分点；完整 p25 下变化为 \(0.00\)；三个区间均跨越零。与此同时，CommonUpdate 在所有 capped 条件下都上升。

单个有效性对比的精度有限，因此我们不把这些符号解释为已经确认的阶段特异效应。但它们的整体形态仍然提供了重要信息：最终选择出现了共同而显著的移动，显式理由判断却没有出现与之匹配的共同移动。例如，完整 p25 的 CommonUpdate 上升了 \(18.75\) 个百分点，而有效性分辨的变化恰为 \(0.00\)。这种分离促使我们提出一个更具体的解释：钳制或许作用于理由判断之后、负责维持抵抗的某种状态。

---

### Table 1. Phase-specific movement and validity discrimination

| Condition | CommonUpdate (%) | Validity discrimination (pp) |
|---|---:|---:|
| Uncapped | 50.00 | 29.17 |
| Prefill only | 68.75; \(+18.75\), \([+4.17,+33.33]\) | 20.83; \(-8.33\), \([-29.17,+12.50]\) |
| Decode only | 70.83; \(+20.83\), \([+6.25,+35.42]\) | 41.67; \(+12.50\), \([-4.17,+29.17]\) |
| Full p25 | 68.75; \(+18.75\), \([+4.17,+33.33]\) | 29.17; \(0.00\), \([-25.00,+20.83]\) |

> Capped cells report the absolute level, capped-minus-uncapped change, and 95% family-cluster bootstrap interval. Bootstrap intervals resample 12 independent families.

---

## 5.3 The Resistance-State Hypothesis Fails Its Own Test

### English

An initially compelling clue appeared in a development thinking-greedy \(S\) diagnostic. Five cases that changed from the initial answer to the induced target had \(3.504\) units more uncapped aggregate-decode correction across layers 46–53 than 21 cases that remained with the initial answer (95% CI \([0.955,5.788]\)). This pattern suggested that above-threshold exposure might mark a latent state used to resist the induced target.

We then tested the prediction at the level where the account should operate. If selective rectification disrupts resistance, uncapped trajectories that resist should show greater pre-answer correction than trajectories that yield. Across 17 development \(S\) prompts containing both outcomes, the prespecified resist-minus-yield contrast over the final 32 reasoning tokens at layers 50–53 was instead \(-1.146\) \([-3.294,1.264]\). Yield trajectories were much longer, averaging 1,039 rather than 343 thinking tokens. Once trajectories were length-matched, the contrast attenuated to \(0.033\) \([-1.394,2.166]\).

The resistance-state hypothesis therefore did not survive its prespecified control. The earlier aggregate signal was strongly entangled with trajectory length and composition, and it could not explain the shift that remained elsewhere in the study. This result redirected the investigation from a special “moment of resistance” toward the geometry of the answer decision itself.

### 中文

一个最初非常有吸引力的线索出现在 development thinking-greedy 的 \(S\) 诊断中。5 个从初始答案转向诱导目标的案例，在第 46–53 层上的 uncapped 聚合 decode 修正量比 21 个保持初始答案的案例高 \(3.504\)（95% CI \([0.955,5.788]\)）。这一模式提示：超过阈值的暴露或许标记了某种负责抵抗诱导目标的潜在状态。

随后，我们在该解释真正应当成立的位置上检验它。如果选择性整流确实破坏了抵抗，那么 uncapped 条件下最终抵抗的轨迹，应当在给出答案之前表现出比屈从轨迹更高的修正量。在同时出现抵抗与屈从结果的 17 个 development \(S\) prompts 上，预先规定的检验考察第 50–53 层、最后 32 个推理 token；得到的抵抗减屈从对比却为 \(-1.146\) \([-3.294,1.264]\)。屈从轨迹同时明显更长，平均包含 1,039 个 thinking tokens，而抵抗轨迹为 343 个。进行长度匹配后，对比进一步衰减到 \(0.033\) \([-1.394,2.166]\)。

因此，“抵抗状态”假说没有通过它自己的预设控制实验。早期聚合信号与轨迹长度及构成高度纠缠，无法解释研究中其余位置持续出现的位移。这个结果把我们的调查方向从某个特殊的“抵抗时刻”，转向答案决策本身的几何结构。

---

## 5.4 The Shift Survives a Frozen Reasoning Trace

### English

We next removed variation in the visible reasoning trajectory itself. For each of 96 prompts, we froze the reasoning prefixes generated under uncapped inference (\(r_U\)) and full-p25 capping (\(r_P\)). Each prefix was then replayed through the cache-faithful generation path under both uncapped and full-p25 forward conditions, with every visible token held fixed. At the answer boundary, we read the four option logits and estimated

\[
\begin{aligned}
\Delta_{\mathrm{FT}}(s)
&=\mathbb{E}\!\left[m_{\mathrm{general}}(p25;r_s)-m_{\mathrm{general}}(0;r_s)\right],\\
I_{\mathrm{FT}}
&=\Delta_{\mathrm{FT}}(P)-\Delta_{\mathrm{FT}}(U),
\end{aligned}
\]

where \(s\in\{U,P\}\) indexes the trace source.

For uncapped-source traces, full-p25 replay increased \(m_{\mathrm{general}}\) by \(2.078\) (95% CI \([1.759,2.408]\)). The continuous displacement therefore survived even when the visible reasoning sequence was identical. For p25-source traces, the shift was \(0.700\) \([0.487,0.907]\), yielding a P-minus-U interaction of \(-1.378\) \([-1.720,-1.084]\). Conditioning on an intervention-generated trace thus absorbed part of the forward-pass difference, although this comparison is not a mediation estimate.

The fixed-trace \(V\)-minus-\(F\) target-margin contrast changed by \(-2.422\) logits \([-3.411,-1.448]\) on uncapped-source traces and by \(-0.797\) \([-1.219,-0.391]\) on p25-source traces. The former arose mainly from an increased \(F\) target margin; the latter arose mainly from a decreased \(V\) target margin. These are continuous fixed-trace logit contrasts, whereas the \(+4.17\)-point selectivity change in Section 4 is a discrete natural-generation rate. Their differing signs are another instance of the margin–choice dissociation.

None of the 192 paired replays crossed the four-option argmax boundary. This is precisely the regime in which a continuous intervention can remain behaviorally latent: the answer geometry moves while the observed choice stays on the same side of the boundary. The uncapped-source effect was approximately 66% of the \(3.130\) natural-generation shift in Section 4, a descriptive comparison showing that the fixed-trace effect has comparable magnitude without treating it as a mediated proportion.

### 中文

接下来，我们进一步消除可见推理轨迹本身的变化。对于 96 个 prompts，我们分别冻结 uncapped 推理产生的前缀 \(r_U\) 和完整 p25 产生的前缀 \(r_P\)。随后，在保持每一个可见 token 完全相同的情况下，把每条前缀分别通过 uncapped 与完整 p25 的 cache-faithful 路径重新前向计算，并在答案边界读取四个选项的 logits，估计：

\[
\begin{aligned}
\Delta_{\mathrm{FT}}(s)
&=\mathbb{E}\!\left[m_{\mathrm{general}}(p25;r_s)-m_{\mathrm{general}}(0;r_s)\right],\\
I_{\mathrm{FT}}
&=\Delta_{\mathrm{FT}}(P)-\Delta_{\mathrm{FT}}(U),
\end{aligned}
\]

其中 \(s\in\{U,P\}\) 表示轨迹来源。

在 uncapped 来源轨迹上，完整 p25 使 \(m_{\mathrm{general}}\) 增加 \(2.078\)（95% CI \([1.759,2.408]\)）。因此，即使可见推理序列完全相同，连续几何位移仍然存在。在 p25 来源轨迹上，位移为 \(0.700\) \([0.487,0.907]\)，P 减 U 的交互为 \(-1.378\) \([-1.720,-1.084]\)。这说明条件化在由干预产生的轨迹上会吸收一部分前向差异，但该对比不构成正式中介估计。

固定轨迹下，\(V\) 减 \(F\) 的 target-margin 对比在 uncapped 来源轨迹上改变 \(-2.422\) logits \([-3.411,-1.448]\)，在 p25 来源轨迹上改变 \(-0.797\) \([-1.219,-0.391]\)。前者主要来自 \(F\) target margin 的上升，后者主要来自 \(V\) target margin 的下降。这里衡量的是固定轨迹下的连续 logit 对比，而第 4 节的 \(+4.17\) 个百分点是自然生成中的离散选择率；两者符号不同，再次体现了 margin–choice 分离。

192 个配对重放中没有任何一个越过四选一的 argmax 边界。这正是连续干预可以保持行为潜伏的情形：答案几何已经移动，但观察到的选择仍停留在决策边界的同一侧。uncapped 来源轨迹上的位移约为第 4 节自然生成位移 \(3.130\) 的 66%；该比较说明固定轨迹效应具有相近量级，但不把这一比例解释成正式中介份额。

---

## 5.5 Capping Propagates Beyond the Axis It Enforces

### English

The central controllability question is whether the coordinate used by the cap continues to organize its downstream effect. The released operator rectifies projections along layer-specific Assistant Axes at layers 46–53. On fixed uncapped-source traces, we measured the paired hidden-state displacement between uncapped and full-p25 replay through layer 63. At layers 54–63, displacement was resolved against the fixed layer-53 exit direction. We preregistered

\[
G_{\mathrm{off}}
=
\operatorname{OffShare}_{60}
-
\operatorname{OffShare}_{53},
\]

the growth in off-exit-axis energy share after the intervention window.

Capping propagated beyond the axis it enforced. The off-exit-axis share increased from \(0.762\) at layer 53 to \(0.922\) at layer 60, yielding \(G_{\mathrm{off}}=0.161\) (95% family-cluster bootstrap CI \([0.151,0.170]\)). All 12 families were positive; every leave-one-family-out estimate lay in \([0.158,0.163]\); and all four arms showed the same downstream growth (\(N=0.145\), \(R=0.142\), \(V=0.184\), \(F=0.172\)). Because the capped window uses layer-specific axes while the downstream decomposition uses the fixed layer-53 direction, the level at layer 53 is a process quantity rather than a standalone leakage rate. The preregistered result is the layer-53-to-60 increase.

The profile exceeded four preregistered schedule- and norm-matched random-direction references. Those references had already approached the off-direction ceiling by the window exit and therefore showed little further increase. The comparison identifies a distinct propagation profile: p25 continues to move away from its exit coordinate after rectification has stopped. It does not define a sampled random null or imply that random perturbations remain one-dimensional.

The same within-stack assay recovered the broad answer-margin displacement, \(\Delta m_{\mathrm{general}}=1.339\) \([0.911,1.754]\). Variation in \(G_{\mathrm{off}}\) did not detectably track variation in the margin shift across 12 families (\(\rho=-0.175\)). This combination directs attention from scalar distance along the source axis toward the downstream mapping that converts a distributed state into an answer preference. An auxiliary family-held-out probe sharpens that interpretation: \(V/F\) validity was linearly decodable from layer-60 states under both uncapped and full-p25 inference (ROC-AUC \(0.769\) and \(0.767\)), while a text-only TF-IDF baseline remained near chance (\(0.511\)). The two probes were trained separately, so this finding establishes condition-wise decodability rather than a shared invariant code.

Across the full diagnostic chain, the empirical invariant is a broad answer-geometric displacement. It appears when either intervention phase is active, separates from the profile of explicit validity judgments, survives a frozen visible reasoning trace, and continues beyond the coordinate in which the cap is defined. These observations motivate a mapping-level explanation developed in Section 6.

### 中文

这里真正关系到“可控性”的问题，是钳制所使用的坐标是否仍然组织着它的下游效应。公开算子在第 46–53 层沿各层对应的 Assistant Axis 修正投影。我们在固定的 uncapped 来源轨迹上，追踪 uncapped 与完整 p25 重放之间的配对隐状态位移，一直延伸到第 63 层；在第 54–63 层，所有位移都相对于固定的 layer-53 出口方向进行分解。预先规定的主指标为：

\[
G_{\mathrm{off}}
=
\operatorname{OffShare}_{60}
-
\operatorname{OffShare}_{53},
\]

即离开干预窗口后，轴外能量占比的继续增长。

钳制效应越出了它所约束的轴。相对于固定的 layer-53 出口方向，轴外能量占比从第 53 层的 \(0.762\) 增长到第 60 层的 \(0.922\)，得到 \(G_{\mathrm{off}}=0.161\)（95% family-cluster bootstrap CI \([0.151,0.170]\)）。12 个 family 全部为正；任意留一 family 的估计都落在 \([0.158,0.163]\)；四个臂均出现同向增长（\(N=0.145\)、\(R=0.142\)、\(V=0.184\)、\(F=0.172\)）。由于钳制窗口内部使用逐层轴，而下游分解使用固定的 layer-53 方向，第 53 层的水平值应当被理解为过程量，而非独立的“泄漏率”；预先规定的核心结果是第 53 层到第 60 层的增长。

这一传播形态超过了四个预注册、调度和范数均匹配的随机方向参考。这些参考在干预窗口出口时已经接近轴外占比上限，因此之后几乎没有继续增长。这个比较识别出一种不同的传播轨迹：p25 在整流停止后仍继续离开其出口坐标。它既不构成完整随机零分布，也不意味着随机扰动能够保持为一维。

在同一环境内部，本轮同时恢复了广泛的答案 margin 位移：\(\Delta m_{\mathrm{general}}=1.339\) \([0.911,1.754]\)。在 12 个 family 上，\(G_{\mathrm{off}}\) 的变化没有与 margin 位移形成可检测的共同变化（\(\rho=-0.175\)）。这组结果把注意力从源轴上的单一标量距离，进一步引向了“分布式状态如何被转换为答案偏好”的下游映射。一个辅助的 family-held-out 探针使这一解释更加具体：在 uncapped 与完整 p25 条件下，layer 60 的状态均可线性解码 \(V/F\) 有效性（ROC-AUC 分别为 \(0.769\) 和 \(0.767\)），而纯文本 TF-IDF 基线接近随机水平（\(0.511\)）。由于两个探针分别训练，该结果确立的是两种条件下各自的可解码性，而非已经证明的共享不变编码。

穿过整条诊断链后，反复出现的经验不变量是一次广泛的答案几何位移：任一干预阶段都能产生它；它与显式有效性判断呈现不同轮廓；在冻结可见推理后仍然存在；并继续传播到钳制定义坐标之外。第 6 节据此提出一个关于“表征如何抵达决策”的解释。

---

# 6 Representation Is Not Policy

## 6.1 What the Evidence Establishes

### English

The Assistant Axis is informative, and the released cap acts on the model. The central discovery is that these successes do not determine how reasons govern revision. Across the tested configuration, p25 produced a broad displacement of the current answer relative to alternatives. The displacement appeared when no induced target was present, crossed valid and fallacious reason conditions, survived with every visible reasoning token frozen, and propagated beyond the coordinate used to apply the cap. Its discrete consequences changed across banks because the same continuous movement met different local decision geometries.

The evidence therefore separates three empirical properties that activation-control work often compresses into one. **Representation** asks whether a direction summarizes a meaningful state. **Actuation** asks whether an operator can move the model along or relative to that state. **Policy selectivity** asks whether the resulting behavior continues to preserve the distinctions on which good decisions depend. In the released Assistant-Axis configuration, the first two properties are demonstrable. The third requires its own evidence: final revision did not acquire a stable advantage for valid over fallacious reasons, even as validity-related information remained linearly decodable in the auxiliary probe.

This separation changes the scientific object. The recurring result is not merely a change in one task score; it is a change in the mapping between the model’s current answer state and the alternatives available to it. Margin–choice dissociation describes the observable consequence of that mapping: continuous answer geometry can move without crossing an argmax boundary, and the same movement can produce different discrete outcomes in different task banks. The propagation result adds an internal constraint: the downstream effect is no longer summarized by the single coordinate on which the controller operates.

### 中文

Assistant Axis 的确具有信息，公开钳制也确实能够作用于模型。我们的核心发现是：这两项成功并不能决定“理由将如何支配改答”。在当前测试配置中，p25 广泛移动了当前答案相对于其他选项的位置。这种位移在没有诱导目标时也会出现，跨越有效与谬误理由条件，在所有可见推理 token 都被冻结时仍然存在，并继续传播到钳制所使用的坐标之外。由于同一个连续位移遇到的局部决策几何不同，它在不同题库中产生了不同的离散后果。

因此，当前证据把激活控制研究中经常被压缩成一件事的三项经验性质分开了。**表征性**考察某个方向是否概括了一种有意义的状态；**可驱动性**考察某个算子是否能够沿着或相对于该状态移动模型；**策略选择性**考察干预后的行为是否继续保留高质量决策所依赖的区分。在公开 Assistant-Axis 配置上，前两项可以被直接展示。第三项需要独立证据：尽管辅助探针仍能线性解码有效性相关信息，最终改答却没有稳定获得“有效理由优于谬误理由”的优势。

这一分离改变了我们真正研究的对象。反复出现的结果并不只是某个任务分数发生变化，而是模型当前答案状态与可选替代答案之间的映射发生了变化。margin–choice 分离描述了这种映射的外部后果：连续答案几何可以在不跨越 argmax 边界的情况下移动，同一个位移也可以在不同题库中产生不同的离散结果。传播分析又提供了一条内部约束：下游效应已经无法由控制器施加作用时的单一坐标概括。

---

## 6.2 A Representational–Policy Decoupling Hypothesis

### English

We propose **representational–policy decoupling** as a unifying, falsifiable account of these results. The hypothesis distinguishes representational content from the route by which that content reaches a decision. Content concerns what information is present in the hidden state. The route includes how validity-sensitive structure is propagated, weighted against the current answer, combined with competing alternatives, and finally read out as a revision. A controller can leave useful information decodable while changing this route.

Under this account, p25 does not need to erase the model’s representation of reason validity. Local rectification changes the state on which downstream computation operates; subsequent layers remix that change into a higher-dimensional decision space; and the resulting answer geometry is displaced without a stable coupling to the validity of the proposed revision. This single account connects observations that otherwise look unrelated: target-hidden prompts still move, explicit validity judgments do not mirror final-choice movement, the two task banks produce different discrete revisions around a common margin shift, the displacement survives a frozen reasoning trace, and the intervention’s effect continues beyond its source coordinate.

Margin–choice dissociation is the behavioral phenomenon; representational–policy decoupling is the proposed explanation. The hypothesis also yields decisive tests. First, a validity probe trained on uncapped states should transfer to capped states and back. Successful transfer would show that the usable code is shared across conditions; failed transfer would indicate condition-specific re-encoding. Second, patching validity-sensitive activations from uncapped into capped inference should restore part of the \(V-F\) selectivity if those activations causally contribute to revision policy. Third, protecting the validity-sensitive subspace during intervention should improve \(V-F\) selectivity at a matched persona benefit. Together, these predictions distinguish a policy-routing account from erased evidence, epiphenomenal decodability, and general answer freezing.

### 中文

我们提出**表征—策略解耦**，作为统一解释当前结果、同时可以被后续实验推翻的假说。该假说区分“表征中包含什么”与“这些内容通过什么路径抵达决策”。内容指隐状态中存在哪些信息；路径则包括有效性敏感结构如何向下游传播、如何与当前答案进行权衡、如何同其他候选项组合，以及最终如何被读出为一次改答。控制器可以在保留有用信息可解码性的同时，改变这条路径。

按照这一解释，p25 无需删除模型对理由有效性的表征。局部整流改变了下游计算所接收的状态；后续网络把这一改变重混进更高维的决策空间；最终答案几何因此发生位移，却没有与所提议改答的有效性保持稳定耦合。这个解释把此前看似分散的现象连接成同一幅图景：目标被隐藏时仍然发生位移；显式有效性判断没有同步复制最终选择的移动；两个题库围绕共同的 margin 位移产生不同的离散改答；冻结推理轨迹后位移仍然存在；干预效应还继续越出它最初的作用坐标。

margin–choice 分离是可观察的行为现象，表征—策略解耦则是我们提出的解释。该假说同时给出三项能够裁决它的实验。第一，在 uncapped 状态上训练的有效性探针应当能够迁移到 capped 状态，反之亦然；迁移成功说明两种条件共享可利用的编码，迁移失败则指向条件特异的重新编码。第二，如果有效性敏感激活对修订策略具有因果作用，把 uncapped 激活补丁到 capped 推理中应当恢复一部分 \(V-F\) 选择性。第三，在干预过程中保护有效性敏感子空间，应当在匹配的 persona 收益下改善 \(V-F\) 选择性。这些预测能够区分策略路由改变、证据信息被擦除、可解码但无因果作用的旁观信号，以及普遍冻结答案等不同解释。

---

## 6.3 From the Hypothesis to Selective Control

### English

The hypothesis translates directly into a controller-development target. Let \(u\) denote the released actuation direction and let \(\Pi_E\) project onto an evidence-sensitive subspace estimated only from warranted-revision transitions in controller-development families. Define

\[
u_{\perp}
=
\frac{(I-\Pi_E)u}
{\lVert(I-\Pi_E)u\rVert_2}.
\]

We specify a three-level controller ladder. L0 is the released one-sided threshold cap. L1 preserves the released trigger and redirects its corrective update onto \(u_{\perp}\), separating detection from actuation. L2 uses the protected decomposition for both gating and actuation, preventing evidence-sensitive movement alone from satisfying the removal rule. The protected subspace, thresholds, layers, and matching tolerance are selected on controller-development families and frozen before held-out evaluation.

The evaluation contract uses a matched-benefit operating point. A controller variant must first retain the same CEDAR-Social attenuation, or the same independently measured persona benefit when that endpoint is available. Selectivity is then judged within that constraint: target following should be preserved or increased under valid reasons, should not increase under fallacious reasons, and broad target-hidden movement should be reduced. Persona safety, the \(S-AO\) contrast, ordinary capability, output integrity, trigger exposure, and compute are reported jointly.

This ladder has been specified as the experimental consequence of the hypothesis; its empirical comparison remains a separate study. Its design target is precise: preserve the evidence-sensitive route from representation to revision while controlling the persona-related state at matched benefit.

### 中文

这一假说可以直接转化为控制器开发目标。令 \(u\) 表示公开的作用方向，令 \(\Pi_E\) 表示一个证据敏感子空间上的投影；该子空间只使用 controller-development families 中的合理修订轨迹进行估计。定义：

\[
u_{\perp}
=
\frac{(I-\Pi_E)u}
{\lVert(I-\Pi_E)u\rVert_2}.
\]

我们据此明确一个三级控制器阶梯。L0 是公开的单侧阈值钳制。L1 保留原有触发规则，但把修正更新重定向到 \(u_{\perp}\)，从而分离检测与作用方向。L2 则在门控和作用两个环节都采用受保护的分解，使证据敏感移动本身不足以满足移除条件。受保护子空间、阈值、层和匹配容差都在 controller-development families 上选择，并在 held-out 评测前冻结。

评测契约采用 matched-benefit 工作点。任何控制器变体都必须先保留相同的 CEDAR-Social 衰减，或在能够独立测量时保留相同的 persona 收益；随后才在这一约束下判断选择性：有效理由下的 target following 应当保持或提高，谬误理由下不应增加，而 target-hidden 条件中的广泛移动应当减小。同时联合报告 persona 安全、\(S-AO\) 对比、常规能力、输出完整性、触发暴露和计算开销。

这一阶梯已经作为表征—策略解耦假说的实验推论被明确规定；其经验比较属于后续独立研究。它的设计目标十分具体：在匹配 persona 收益的同时，保护证据敏感信息从表征通往修订决策的路径。

---

## 6.4 Scope and Decisive Next Tests

### English

The present claims concern Qwen3-32B and the released model-specific Assistant Axis with p25 capping. The development and new-salt banks provide two independently instantiated samples from the same solver-certified task grammar; broader structural and naturalistic transport remains to be tested. Fixed-trace replay estimates a conditional answer-boundary effect with visible tokens held constant. RG1 characterizes downstream propagation under that replay, and the probe is an auxiliary decodability analysis. These modules jointly constrain the hypothesis while retaining distinct estimands.

The decisive next experiments are therefore focused. Cross-condition probe transfer can test whether validity uses a shared code. Bidirectional activation patching at preregistered blocks can test whether that code and the observed downstream displacement participate causally in revision. The L0/L1/L2 comparison can test whether protecting the evidence-sensitive route improves selectivity at matched persona benefit. Finally, new structural families, natural argument datasets, and additional officially supported model–axis pairs can determine how broadly the mapping-level account travels.

### 中文

本文当前的主张范围是 Qwen3-32B、其公开的模型特异 Assistant Axis，以及 p25 钳制。development 与 new-salt banks 是同一套求解器认证任务语法下的两次独立实例化；更广泛的结构迁移与自然任务迁移仍需继续检验。固定轨迹重放估计的是“可见 token 保持不变时”的条件性答案边界效应。RG1 描述该重放条件下的下游传播，探针则属于辅助可解码性分析。这些模块共同约束同一个假说，但保留各自不同的估计对象。

因此，下一步实验具有明确焦点。跨条件探针迁移可以检验有效性信息是否共享同一编码；在预先规定层块上的双向 activation patching 可以检验该编码及所观察到的下游位移是否因果参与修订；L0/L1/L2 比较可以检验，在匹配 persona 收益时，保护证据敏感路径能否改善选择性；全新结构 family、自然论证数据和其他具有官方支持方向的模型—轴组合，则可以检验这一映射层解释的适用范围。

---

# 7 Conclusion

## English

The most natural explanation for our initial behavioral results was that Assistant-Axis capping damaged the model’s evaluation of reasons or interrupted a state required to resist an induced answer. That explanation did not survive the diagnostic chain. Explicit validity judgments did not reproduce the common movement in final choice. The proposed resistance signature collapsed under its prespecified trajectory control. A continuous margin shift remained when every visible reasoning token was frozen. Across independently instantiated banks, warranted and unwarranted revisions changed differently at the discrete level while the aggregate answer-geometric displacement persisted.

The representation analysis revealed where this surviving effect travels. After p25 stopped acting, its induced displacement continued to move away from the cap’s exit coordinate across all 12 families. At the same downstream layer, an auxiliary family-held-out probe could decode \(V/F\) validity under both uncapped and capped inference. Together, these observations shift the explanation from erased reason content toward the mapping from representation to revision.

We call this possibility representational–policy decoupling. Local rectification can alter the downstream route by which validity-sensitive structure reaches an answer, producing a broad margin displacement whose discrete expression depends on the surrounding decision geometry. The hypothesis unifies target-hidden movement, divergent bank-level revisions, fixed-trace persistence, margin–choice dissociation, and off-axis propagation, while yielding direct tests through cross-condition probes, bidirectional patching, and protected-subspace controllers.

The broader lesson extends beyond this configuration. A direction may represent a desirable model state, and an operator may successfully move that state, while the deployed policy loses distinctions that matter for deciding when to change. Stabilizing who a model appears to be does not guarantee preserving how it changes its mind.

## 中文

面对最初的行为结果，最自然的解释是：Assistant-Axis capping 损害了模型对理由的判断，或者打断了抵抗诱导答案所必需的状态。但这一解释没有通过后续诊断链。显式有效性判断没有复制最终选择中的共同移动；假定的“抵抗信号”在预设轨迹控制下衰减；当所有可见推理 token 都被冻结时，连续 margin 位移仍然存在；在两套独立实例化题库中，合理与错误改答的离散方向发生变化，但聚合的答案几何位移依然存活。

表征分析进一步揭示了这一持续效应如何向下游传播。p25 停止作用之后，它所引起的位移仍继续离开钳制的出口坐标，并在 12 个 family 上全部同向。在相同的下游层，辅助的 family-held-out 探针能够在 uncapped 与 capped 条件下解码 \(V/F\) 有效性。综合这些观察，解释焦点从“理由内容被擦除”转向“表征如何被映射为一次修订决策”。

我们把这种可能性称为表征—策略解耦。局部整流可以改变有效性敏感结构抵达答案的下游路径，产生广泛的 margin 位移；该位移最终如何兑现为离散选择，则取决于周围的决策几何。这个假说把 target-hidden 位移、两套题库中不同方向的离散改答、固定轨迹下的持续效应、margin–choice 分离和轴外传播连接成同一幅图景，并给出跨条件探针、双向 patching 与受保护子空间控制器等直接检验。

这一结论的意义超出当前配置。某个方向可以表征一种理想状态，某个算子也可以成功移动该状态，但部署后的策略仍可能失去“何时应当改变”所依赖的关键区分。**稳定模型表现出来的角色，并不保证保留它改变想法的方式。**

---

## Appendix-only reproducibility disclosure

The strict token-level cross-environment anchor was not numerically equivalent to the earlier stack. RG1 therefore reconstructed every analyzed condition and control within one fresh stack; the within-stack formal audit passed on all 864 records. This implementation fact should appear in the reproducibility appendix or experiment log, not in the main scientific narrative.

## 仅用于附录的复现说明

本轮与先前环境之间没有获得严格的 token 级数值等同。因此，RG1 在同一套全新环境内重新构建了全部分析条件和对照，且环境内形式审计在 864/864 条记录上通过。该信息应当进入复现附录或实验日志，不应占据正文的主要科学叙事。
