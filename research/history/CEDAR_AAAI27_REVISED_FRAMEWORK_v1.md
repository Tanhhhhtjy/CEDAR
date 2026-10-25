# CEDAR AAAI-27 新版论文框架 v1

> 工作标题：**Capping the Assistant, Shifting the Answer: Epistemic Non-Selectivity in Activation-Space Control**
>
> 更有攻击性的备选标题：**An Axis Is Not a Selective Controller: Auditing Assistant-Axis Capping**
>
> 论文定位：以 CEDAR 为核心测量框架的实证评测与机制诊断论文。当前不把“提出新控制器”设为论文成立的前提。

---

## 0. 这份框架如何使用

这份文档替代旧稿中以下主线：

> Assistant Axis capping 已经带来安全收益，但以正确更新能力为代价；我们进一步解释机制并提出更好的控制器。

旧主线的问题不是我们在原始设置下“复现失败”，而是原论文出于安全原因未公开或脱敏了关键 persona jailbreak prompts，使严格等价复现事实上不可执行。我们使用公开资源构建的替代攻击没有通过预注册的攻击有效性门槛，因此只能说原文的安全收益**尚未被我们独立验证**，不能说它被推翻或复现失败。与此同时，五剂量 trade-off 也没有稳定出现。继续把“已获得安全收益”作为本文前提，会使整篇论文建立在当前实验无法独立确认的条件上。

新版主线不依赖这些失败环节，而建立在目前最稳定、跨任务重复出现的现象上：

> **Assistant-Axis capping 会系统性改变模型放弃当前答案的决策几何，但这种变化并不可靠地区分“为什么应该改答”。它可以推动正确更新，也可以推动错误更新；连续内部偏移在不同任务边界上会兑现成不同、甚至方向相反的离散行为。**

写作原则：

1. 正文先讲清楚发现，不在每一句后面附带限制。
2. 将“探索性、确认性、技术性未决”等边界集中写入方法、统计和限制部分。
3. 主张必须强，但强在真正获得支持的层面：
   - 可以强力主张“缺乏认识论选择性”；
   - 可以强力主张“连续几何位移与离散行为不能等同”；
   - 不能声称“capping 一定损害所有推理能力”；
   - 不能声称“已经在原始等价攻击条件下独立复现安全收益”；
   - 不能把尚未完成的 E2B 写成既定机制。
4. 新实验只填入预留插槽，不再整体改写论文。

---

## 1. 一句话主线

### 面向评审的版本

> **Capping changes when a model abandons an answer, without reliably tracking whether the offered reason warrants revision.**

### 面向普通读者的版本

> 这种钳制不是在教模型分辨“真证据”和“歪理”，而是在整体改变模型“要不要放弃原答案”的门槛。

### 最有记忆点的概念

> **An axis may describe a model state without constituting a selective control policy.**

一条方向能读出或操纵某种人格状态，不代表沿着它做阈值钳制，就能只压制有害行为而保留其他正常决策。

---

## 2. 论文真正提出的问题

现有激活干预研究通常问：

> 干预后，目标行为有没有减少？

CEDAR 增加了一个更严格的问题：

> 当模型改变行为时，干预有没有保留对“改变理由”的区分？

这一区分至关重要。面对同一个备选答案，模型可能因为：

- 有效证据而正确更新；
- 无效论证而错误更新；
- 社会压力而迎合；
- 仅仅看到该答案被提及而受到启动效应。

如果一个控制器只改变模型“改不改”，却不区分这些原因，它就不是认识论上选择性的控制器。

---

## 3. 三个研究问题

### RQ1：capping 改变了什么？

它是特异地提高某个诱导答案的吸引力，还是更一般地压缩当前答案相对其他答案的优势？

对应指标：

- `m_general`：所有非初始选项相对初始答案的平均优势；
- `m_target`：诱导 target 相对初始答案的优势；
- `m_background`：其余非 target 选项相对初始答案的优势；
- `m_specific`：target 相对其他非初始选项的额外优势；
- 四态迁移：initial / induced target / other wrong / invalid。

### RQ2：这种变化是否尊重理由的有效性？

同一个 target 分别由有效证据 V 和错误论证 F 支持时，capping 是否：

- 只促进 V；
- 只抑制 F；
- 改善 `V−F` 选择性；
- 或同时推动二者，主要改变一般更新门槛？

社会条件则用 AO/S 配对区分：

- AO：只暴露 target；
- S：在相同 target exposure 上增加社会压力。

### RQ3：连续位移为什么会产生不稳定的离散后果？

核心候选解释是决策边界：

- capping 产生较稳定的连续 margin 位移；
- 离决策边界远的题，答案可能不变；
- 靠近边界的题，同等位移更容易造成翻转；
- 不同数据集的基线 margin 分布不同，因此离散效应可以大小不同、甚至方向不同。

E2B 进一步区分：

- 位移是否在固定推理轨迹下仍然存在；
- 它更像推理轨迹改变，还是推理完成后的决策映射改变。

---

## 4. 论文的三项核心贡献

### C1：提出“认识论选择性”这一控制器评测维度

我们不再只问干预是否减少某种行为，而是要求控制器同时满足：

- 接受有效证据；
- 抵抗无效论证；
- 隔离社会压力与单纯 target exposure；
- 不通过拒答、输出损坏或答案冻结制造表面收益。

CEDAR 将这些要求变成严格配对的 N/R/V/F/AO/S 条件，以及连续 margin 与离散迁移的联合读出。

### C2：发现 Assistant-Axis capping 引发一般性的初始答案优势压缩

在 E2A 的 solver-certified 形式任务中，full p25 相比 uncapped：

- `Δm_general = +3.130` logits；
- 95% family-cluster bootstrap CI `[+1.834, +4.486]`。

这一位移在四个 arm 中均为正：

- N：`+2.215`；
- R：`+2.092`；
- V：`+2.950`；
- F：`+5.264`。

但 N/R 的最终答案仍然 100% 保持。这说明：

> 内部几何位移可以先以“亚阈值”形式存在；只有任务把模型推到决策边界附近时，它才兑现成离散改答。

### C3：证明该位移不构成可靠的证据选择性控制

在 E2A 中：

- V 正确 target-follow：70.8% → 91.7%，`+20.8pp`；
- F 错误 target-follow：29.2% → 45.8%，`+16.7pp`；
- CommonUpdate：`+18.75pp`，95% CI `[+4.17,+33.33]`；
- AnswerSelectivity 的变化仅 `+4.17pp`，95% CI `[-20.83,+29.17]`。

即：模型更容易离开初始答案，但没有获得可靠的 V/F 辨别改善。

在新 salt、未按响应筛选的 E2C 中，95/96 个可分析配对再次得到同方向总体位移：

- `Δm_general = +1.538`；
- 95% CI `[+0.455,+2.741]`。

同时，离散结果变为：

- V target-follow：`−12.5pp`；
- F target-follow：`+16.7pp`。

这不是对主结果的否定，而揭示了更重要的规律：

> **较稳定的是总体几何位移；可变的是它如何在具体决策边界上重新分配概率并兑现为行为。**

---

## 5. 论文身份与叙事层级

### 5.1 这不是一篇什么论文

当前版本不应包装为：

- 已证明 capping 带来稳定安全收益的 trade-off 论文；
- 已经提出更优 controller 的方法论文；
- 已经锁定具体神经回路的完整 mechanistic interpretability 论文；
- “capping 让模型全面变笨”的能力退化论文。

### 5.2 这是一篇什么论文

最准确、也最有力的定位是：

> **一篇揭示激活空间控制缺乏认识论选择性的评测与机制诊断论文。**

完整逻辑层级为：

1. **Measurement**：现有安全指标遗漏了对理由有效性的区分。
2. **Finding**：官方 p25 广泛改变答案 margin 和更新倾向。
3. **Diagnosis**：这种变化主要表现为一般决策几何改变，而非可靠的 evidence-validity controller。
4. **Boundary**：同一连续位移会因任务边界不同而产生不同离散结果。
5. **Mechanism**：E2B 决定该效应更接近轨迹中介还是输出决策映射。
6. **Design implication**：未来控制器必须将人格安全、证据有效性和社会压力分开建模，而不是把单一表示轴直接当成控制策略。

---

## 6. AAAI 七页正文结构

建议严格按 7 页技术正文设计，参考文献从第 8 页开始。

| 部分 | 建议页数 | 作用 |
|---|---:|---|
| 1. Introduction | 1.0 | 提出遗漏、核心发现、贡献 |
| 2. Related Work | 0.55 | axis control、sycophancy、belief revision |
| 3. CEDAR | 1.25 | 任务构造、arm、指标、统计 |
| 4. Main Findings | 2.0 | 几何位移、选择性、跨任务结果 |
| 5. Explaining the Shift | 1.25 | phase、边界、E1、E2B |
| 6. Implications and Limitations | 0.65 | controller 设计含义与证据边界 |
| 7. Conclusion | 0.30 | 一段结论 |

主文只保留 3 张图、1 张主表。其余实验表、完整门控记录、数据构建审计进入附录。

---

## 7. 逐节详细框架

# 1 Introduction

## 第一段：激活空间控制正在成为安全工具

开场不要从“我们发现副作用”开始，而从更大的方法论问题开始：

> 激活 steering 和 capping 允许研究者在不重新训练模型的情况下改变人格、拒答和越狱行为。但这类方法往往依据目标行为是否下降来评价成功，默认内部方向也可以充当选择性的行为控制器。

## 第二段：现有评价遗漏了“理由”

给一个最简单的对照例子：

- 用户提供真实撤稿信息，模型应该改；
- 用户给出一段看似严谨但逻辑无效的论证，模型不应该改；
- 用户只是坚持“你错了”，模型也不应该改。

三种情况都可能要求模型从同一个初始选项转向同一个 target。只看“有没有改答”，无法区分能力、易受骗和迎合。

## 第三段：提出 CEDAR

CEDAR 的核心不是又一个 sycophancy benchmark，而是一个控制器审计框架：

> 固定初始问题、初始答案和诱导 target，只替换支持 target 的理由。

由此把“更新倾向”和“证据选择性”分开。

## 第四段：给出核心结果

建议直接写：

> Across solver-certified formal tasks, official p25 capping produced a broad shift of non-initial options relative to the model’s initial answer. The shift appeared even in neutral and reconsideration controls without changing their discrete answers, but became behaviorally consequential under valid and invalid arguments. Capping increased warranted and unwarranted updating together, yielding no reliable gain in evidence selectivity.

随后用 E2A 的 `+3.13 logits` 与 `V +20.8pp / F +16.7pp` 做数字锚点。

## 第五段：给出更深的解释与贡献

> The stable object is a continuous geometry shift; the unstable object is which discrete answer flips. This distinction reconciles apparently conflicting effects across formal, social, and public forced-choice tasks.

然后列三项贡献：

1. CEDAR 认识论选择性评测；
2. 一般初始答案优势压缩的实证发现；
3. margin–choice 分离和决策边界解释，以及由此得到的 controller 设计原则。

---

# 2 Related Work

本节控制在三个紧凑段落。

## 2.1 Activation steering and capping

覆盖：

- persona/behavior direction；
- continuous steering；
- threshold capping；
- Assistant Axis 原文。

关键缺口：

> Existing work primarily evaluates whether an intervention suppresses a target behavior, not whether it preserves reason-sensitive revision.

## 2.2 Sycophancy and opinion-following

覆盖：

- SycophancyEval / SycoBench；
- PhilPapers / Playing Devil’s Advocate；
- When Truth Is Overridden；
- 用户意见、权威和答案锚定。

与本文区别：

> 这些工作主要测模型是否跟随用户；CEDAR 进一步要求在相同 target 下区分有效证据、错误论证、纯 target exposure 和社会压力。

## 2.3 Belief revision and mechanistic diagnosis

覆盖：

- Belief-R、ReviseQA、ProverGen；
- logit lens / activation patching；
- 决策边界与连续 margin。

关键缺口：

> Existing belief-revision datasets rarely evaluate activation controllers, while controller studies rarely pair valid and invalid reasons under solver-certified gold labels.

---

# 3 CEDAR: Auditing Epistemic Selectivity

## 3.1 Controlled contrast design

每个 base 固定：

- 原始任务；
- 初始正确答案；
- 诱导 target；
- 答案位置；
- 表面结构。

只改变第二阶段输入：

| Arm | 新输入 | 规范行为 | 测量 |
|---|---|---|---|
| N | 中性继续 | 保持 initial | 自发漂移/输出损坏 |
| R | 无信息复核 | 保持 initial | 复核本身是否致变 |
| V | solver 认证的有效修改 | 更新至 target | 有据更新 |
| F | 单一致命错误、带 countermodel | 保持 initial | 错误论证脆弱性 |
| AO | 无施动者 target 提及 | 保持 initial | target exposure |
| S | 与 AO 配对的社会压力 | 保持 initial | 社会增量 |

## 3.2 Solver-backed construction

强调：

- ProverGen-style 构建初始证明图；
- ReviseQA-style 原子邻域修改；
- V 后 target 成为唯一 gold；
- F 后 initial 仍为唯一 gold；
- F 只有一个可定位错误并有反模型；
- easy/medium 来自证明深度、规则数和干扰事实，而非晦涩措辞。

这部分是论文的重要工作量证明，不能只写成一句“we generated questions”。

## 3.3 Intervention and model

主模型：

- Qwen3-32B；
- Assistant Axis；
- layers 46–53；
- official p25 threshold capping；
- thinking 与 decoding 设置按每个 assay 冻结。

必须区分：

- global additive steering；
- thresholded capping；
- prefill-only；
- decode-only；
- full p25。

## 3.4 Joint behavioral and geometric readouts

离散四态：

1. initial；
2. induced target；
3. other wrong；
4. invalid/abstain。

连续指标建议统一定义为：

\[
m_{\mathrm{general}}
=
\frac{1}{3}\sum_{a\neq a_0}\ell(a)-\ell(a_0),
\]

其中 \(a_0\) 是 initial answer。正的 \(\Delta m_{\mathrm{general}}\) 表示：

> 非 initial 选项整体相对 initial 上升，即 initial-answer advantage 被压缩。

另定义：

\[
m_{\mathrm{specific}}
=
\ell(a_t)-\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}\ell(a),
\]

用于判断 target 是否获得超越其他替代项的特异性提升。

## 3.5 Statistical discipline

正文简写，细节放附录：

- family-cluster bootstrap；
- family-level hierarchical model；
- ITT 保留截断与无效输出；
- 连续和离散结果同时报告；
- 不按结果删题；
- development、new-salt、external 的证据身份分别披露。

---

# 4 Capping Shifts the Answer Geometry

## 4.1 Instrument calibration: the axis is real, but the operator matters

用 E−1a/E−1b 做短而有力的铺垫：

- global `+200` Assistant-Axis steering 复现公开 PhilPapers 反谄媚方向：平均 `−2.4312`；
- 但大剂量正负 steering 都改变输出空间，不能把 margin 下降直接等同于回答质量提高；
- official p25 在同一公开任务上几乎处处触发，却使连续 margin 下降 `−0.09375`、离散 sycophancy 反而增加 `+1.444pp`；
- 翻转集中在基线 margin 接近零的题。

本节结论：

> **The same axis does not imply the same intervention. Continuous addition and threshold capping are behaviorally distinct operators.**

## 4.2 A broad compression of initial-answer advantage

这是全文最重要的结果图。

E2A：

| Arm | Δm_general |
|---|---:|
| N | +2.215 |
| R | +2.092 |
| V | +2.950 |
| F | +5.264 |
| Overall | +3.130 |

所有 arm 的方向一致，说明效应不是仅针对用户指定 target。

`m_background` 和 `m_target` 同时上升，而 `m_specific` 接近零：

- `Δm_background = +3.158`；
- `Δm_target = +3.074`；
- `Δm_specific = −0.084`，CI 跨 0。

本节强结论：

> p25 broadly compresses the model’s commitment to its initial answer rather than selectively amplifying the induced target.

## 4.3 Latent in controls, consequential near a revision opportunity

N/R 的离散答案保持 100%，但 margin 已显著移动。

V/F 提供同一个 target，却在理由效力上相反：

- V 与 F 的 target-follow 都增加；
- CommonUpdate 显著增加；
- V−F selectivity 没有可靠改善。

这形成一张非常直观的图：

```text
N / R:
内部答案优势发生移动
        ↓
尚未越过决策边界
        ↓
最终答案不变

V / F:
同类内部移动 + 候选理由使题目靠近边界
        ↓
部分题越界
        ↓
正确更新和错误更新一起增加
```

## 4.4 New-salt evidence: stable shift, heterogeneous redistribution

E2C 不是全新 family 的最终外部确认，但它是未按模型响应筛选的新实例、新 salt 证据。

主要写法：

> A preregistered new-salt bank reproduced the direction of the aggregate margin shift on 95 analyzable pairs, `+1.538` logits, 95% CI `[+0.455,+2.741]`.

随后承认更细分配发生变化：

- N/R 仍明显同向；
- V 在该轮没有同向复制；
- F 继续偏向错误 target；
- `m_specific` 变为负；
- V/F 离散选择性恶化。

这不是简单写成“不稳定”，而写成：

> The intervention has a reproducible aggregate geometric signature but a task-dependent allocation across alternatives and discrete choices.

技术边界放到 §6：

- 1 条 uncapped F 缺失 logits；
- 正式 Gate 因此技术性未决；
- worst-observed-value sensitivity 仍保留正向 CI；
- 不能把它称为独立 family confirmation。

## 4.5 Social tasks reveal the same margin–choice separation

用 C2/V8 支持“连续位移不能直接解释成行为改善”：

- AO/S token-matched；
- cap×arm margin interaction `−0.78625`；
- p25 同时把 AO 和 S 向 target 推动；
- 它降低了 S 相对 AO 的增量，却没有降低 target 的绝对吸引；
- free-generation 离散迁移接近地板。

本节结论：

> A controller can appear beneficial under a relative contrast while simultaneously moving both conditions toward the prompted answer.

C3 的双向 SycoBench 结果只用一两句补充，详细数字放附录，因为其连续结果明显但预注册离散镜像条件未通过。

---

# 5 Why Does the Shift Occur?

本节不是“已经找到完整机制”，而是按排除法逐步缩小机制空间。

## 5.1 The effect is not additive across generation phases

E2A 中：

- prefill-only CommonUpdate：`+18.75pp`；
- decode-only：`+20.83pp`；
- full p25：`+18.75pp`。

full 没有叠加增强；GLMM 的 prefill×decode 交互为负。

正文表述：

> Either phase is sufficient to reproduce much of the behavioral shift, whereas combining them is subadditive. This pattern is consistent with a shared downstream bottleneck or a saturating decision process, rather than two independent additive mechanisms.

不要写成已确定“共享瓶颈”，除非 E2B/patching 提供因果证据。

## 5.2 Validity judgments do not support a simple reasoning-damage account

full p25 的 validity discrimination 净变化接近零，但：

- prefill-only 倾向下降；
- decode-only 倾向上升；
- 两者区间均较宽；
- full 的零不能解释为每个阶段都无效。

可以写：

> We find no stable evidence that full p25 globally erases validity discrimination. Instead, phase-specific estimates point in opposing directions, suggesting that a net null can conceal compensating computational effects.

这比“完全不损伤推理”更准确，也比“什么都没发现”更有信息。

## 5.3 The simple resistance-state hypothesis is rejected

E0 曾发现 S 错误服从与 late-decode 暴露相关。

E1 token-resolved 分析进一步表明：

- 预测的 resist/yield 几何差异方向相反且跨零；
- yield 轨迹显著更长；
- 长度匹配后效应接近零；
- 修正量控制长度后不再预测 yield。

因此可以明确写：

> The late-decode exposure correlation is better explained by trajectory length and composition than by selective intervention on a low-axis “resistance state.”

这是有价值的 falsification，不是失败实验。

## 5.4 Boundary analysis insertion point

若零 GPU boundary analysis 为 Gate A：

> Across formal and natural tasks, flips concentrate near the uncapped decision boundary, and baseline top-two gap predicts migration after family/domain clustering. The boundary account therefore reconciles apparently conflicting discrete effects.

若 Gate B：

> The observed migration patterns are consistent with a boundary account, although support is not uniform across datasets.

若 Gate C：

删除因果式解释，只保留：

> Continuous shifts and discrete behavior dissociate across tasks.

主分析必须使用干预前的 baseline gap。`Δm` 是干预后量，只能作为补充描述，不能拿它循环解释翻转。

## 5.5 E2B insertion point

### 如果固定轨迹下效应仍存在

写：

> Holding the reasoning trajectory fixed preserves the margin shift, localizing a substantial component downstream of reasoning content, at the final decision/readout stage.

这会显著增强“decision-policy distortion”叙事，并为后续 activation patching 提供明确位置。

### 如果固定轨迹下效应消失

写：

> Equalizing the reasoning trajectory removes the shift, indicating that capping acts primarily by changing deliberation rather than merely remapping a completed computation to an answer.

这同样是好结果，论文转向“trajectory-mediated control”。

### 如果 E2B 再次未通过工程闸

不强行补跑。正文不依赖 E2B 成立，将其作为“attempted causal localization and numerical replay limitation”放附录。

---

# 6 Implications and Limitations

## 6.1 Representation is not policy

最重要的设计结论：

> 一个方向可以成功编码人格状态、预测行为或支持 steering，但这不意味着对它做 threshold rectification 就能得到选择性的控制策略。

Assistant Axis 原文回答的是：

- 人格投影是否变化；
- capping 是否限制这种变化；
- 某些安全行为是否改善。

CEDAR 补问的是：

- 控制是否保留理由敏感性；
- 它是否同时推动正确和错误更新；
- 连续几何变化是否被误读为行为改善。

## 6.2 What a better controller must preserve

未来控制器不应只把模型限制在一个一维区间，而需要分别保护：

1. 人格安全与角色稳定；
2. 有效证据识别；
3. 正确更新能力；
4. 对错误论证的抵抗；
5. 社会压力与单纯 target exposure 的隔离；
6. 正常的动态推理与输出完成。

当前论文可以提出这一设计规范，但不声称已经实现多维控制区域。

## 6.3 Evidence scope

统一在此处披露，不在全文反复削弱结论：

- 主模型是 Qwen3-32B；
- 原论文的关键 persona jailbreak prompts 未公开或经过脱敏，严格等价的 M2.1 复现不可执行；我们构建的公开替代攻击未通过预注册攻击有效性门槛。因此本文不把安全收益视为已被独立验证，也不据此建立 benefit–cost trade-off，但这一结果不构成对原文安全收益的反证；
- E2A 是探索性机制 assay；
- E2C 是新 salt、新实例的方向复制，但不是全新 family 或外部确认，且 1 条主 logit 缺失导致正式 Gate 技术性未决；
- C2/C3 提供自然社会任务的互补证据，但任务构念与形式域不同；
- 边界分析是 post-hoc，必须明确标注；
- 尚不能把 phase nonadditivity 等同于完整神经回路；
- 所有跨模型外推均留待未来工作。

这些限制不改变主发现：

> 在所测试的 Qwen3-32B 和 official p25 配置下，Assistant-Axis capping 稳定地改变答案几何，却没有表现为可靠的、按证据有效性选择的控制器。

---

# 7 Conclusion

建议正文结尾只用一段：

> Activation-space control is often evaluated by whether it suppresses an unwanted behavior. CEDAR asks a stricter question: does the intervention preserve sensitivity to why the model should change? Across solver-certified revision tasks and complementary social assays, Assistant-Axis capping consistently altered the geometry of answer revision, while its discrete consequences depended on the surrounding decision boundary. Valid and invalid reasons were not cleanly separated by the intervention. These findings establish epistemic selectivity as a necessary evaluation target and show why a representation axis should not be treated as a selective controller by default.

---

## 8. 推荐的图表布局

## Figure 1：同一个 target，不同理由

内容：

- 一道 base task；
- initial answer 相同；
- V/F/AO/S 指向同一个 target；
- gold 行为不同；
- 下方一句：behavior suppression is not epistemic selectivity。

作用：

> 让评审在 20 秒内理解论文为什么存在。

## Figure 2：全文主结果

三联图：

### (a) Δm_general by arm

N/R/V/F 四臂均向同一方向移动。

### (b) Discrete target-follow

V 与 F 一起增加，N/R 保持不变。

### (c) General vs specific geometry

`m_background` 与 `m_target` 同时上升，而 `m_specific` 没有对应增强。

标题建议：

> Capping broadly compresses initial-answer advantage without reliable evidence selectivity.

## Figure 3：根据最新结果二选一

优先级：

1. 若 boundary Gate A/B：画 baseline gap 分位数与 flip rate；
2. 若 E2B 成功且更强：画 fixed-trace uncapped/p25 cross；
3. 两者都不够强：画 E2A 与 E2C 的连续几何方向一致、离散分配不同。

## Table 1：Assay matrix

列出：

- 数据集；
- 构念；
- arms；
- primary readout；
- evidence status；
- 对应论文问题。

主文不再放大型 claim–evidence 台账。

---

## 9. 附录结构

建议附录按审稿人质疑路径组织：

### A. Full preregistration and protocol deviations

- M2.1 的资源不可得性、替代攻击构造及预注册过闸结果；
- development / new-salt / external 身份；
- 单人审核与缺失 workbook；
- 截断与 ITT。

### B. Dataset construction

- family cards；
- solver traces；
- V 修改；
- F countermodels；
- naturalization；
- leakage audit；
- deterministic hashes。

### C. Full statistical results

- family bootstrap；
- hierarchical models；
- equivalence/ROPE；
- LOFO；
- missing-value tipping point；
- multiple-comparison policy。

### D. Instrument and operator calibration

- E−1a signed curve；
- top-token/entropy diagnostics；
- E−1b trigger/exposure；
- margin–choice flips。

### E. Social and external assays

- C2/V8；
- C3/SycoBench；
- V7/V7.1；
- 四态迁移。

### F. Mechanism diagnostics

- E0；
- E1；
- phase-specific analysis；
- E2B；
- boundary analysis。

### G. Reproducibility

- model revision；
- environment；
- GPU；
- commit hashes；
- dataset manifests；
- code entry points。

---

## 10. 强而准确的摘要草稿

> Activation-space interventions are increasingly used to suppress undesirable language-model behaviors, yet they are typically evaluated by whether a target behavior decreases—not by whether the intervention preserves sensitivity to the reasons for changing an answer. We introduce CEDAR, a solver-backed framework that contrasts valid evidence, invalid arguments, target exposure, and social pressure while holding the initial problem and proposed answer fixed. On Qwen3-32B, official Assistant-Axis p25 capping produced a broad shift of non-initial options relative to the model’s initial answer (`+3.13` logits; 95% family-cluster CI `[+1.83,+4.49]`). The shift was already present in neutral controls without changing their discrete answers, but became behaviorally consequential when valid or invalid arguments placed decisions nearer revision: warranted updating increased by `20.8` points and unwarranted updating by `16.7` points, with no reliable improvement in V/F selectivity. A preregistered new-salt bank directionally reproduced the aggregate margin shift, while yielding a different redistribution across answer options. Complementary social and public forced-choice assays likewise showed that continuous margin changes need not translate into uniformly beneficial discrete behavior. These results identify epistemic non-selectivity as a failure mode of activation-space control: capping changes when a model abandons an answer without reliably conditioning that change on whether the offered reason warrants revision.

如果 boundary analysis 为 Gate A，可在倒数第二句前增加：

> Across assays, discrete migrations concentrated near the uncapped decision boundary, reconciling heterogeneous behavioral effects with a more stable continuous shift.

如果 E2B 固定轨迹成立，可增加：

> The shift persisted under fixed-trajectory teacher forcing, locating a substantial component downstream of reasoning content.

不能同时为了“显得工作多”把两个弱结果都塞入摘要。只加入通过较强证据门槛的一个。

---

## 11. Claim–Evidence 台账

| 编号 | 正文主张 | 证据 | 当前状态 |
|---|---|---|---|
| C1 | capping 广泛压缩 initial-answer advantage | E2A 四臂 Δm_general | 强支持 |
| C2 | 位移不是 target-specific amplification | E2A background≈target；m_specific 接近 0 | 支持于 E2A |
| C3 | capping 没有可靠改善 V/F 选择性 | V/F 同升；selectivity CI 跨 0 | 支持 |
| C4 | 连续位移可在无离散改答时存在 | N/R margin 移动、答案 100% 保持 | 强支持 |
| C5 | 总体位移可跨新实例保持，但细分后果异质 | E2C complete-case +1.538；arm 重分配不同 | 支持，带技术边界 |
| C6 | margin 改善不能等同于行为改善 | E−1b、C2、E2A/E2C | 强支持 |
| C7 | prefill/decode 效应不简单相加 | E2A 阶段实验 | 支持于当前 assay |
| C8 | 简单“选择性打断抵抗状态”假说不成立 | E1 长度匹配结果 | 强反证 |
| C9 | 翻转集中在决策边界附近 | boundary addendum | 待定 |
| C10 | 位移位于推理完成后的决策映射 | E2B fixed trace | 待定 |

特别注意：

- C3 的意思是“没有可靠改善”，不是“证明变化严格为零”。
- E2C 是 95/96 个可分析配对，不是 95/96 个正向配对。
- 不把 E2A 的 development assay 写成 held-out confirmation。

---

## 12. 旧稿迁移与删改清单

原文件：`cedar-aaai27-working-draft(QA).md`

## 可以保留并升级的内容

### 原 §1 Introduction

保留：

- 现有安全评测遗漏正常行为代价；
- 同一 target、不同更新原因；
- CEDAR 的必要性。

改动：

- 删除“安全收益已在等价条件下得到独立复现，因而 benefit–cost trade-off 已成立”的暗示；
- 加入“axis ≠ selective controller”；
- 把核心数字换为 E2A/E2C 实际结果。

### 原 §3 CEDAR

这是旧稿最成熟的部分，应作为新版核心方法章节。

需要增加：

- solver-backed generator；
- `m_general/m_specific`；
- 四态输出；
- AO/S 严格配对；
- development / new-salt / external 三类证据身份。

## 应整体替换的内容

### 原 §4 “A Hidden Epistemic Cost”

替换为：

> **Capping Shifts the Answer Geometry**

原因：

- 当前证据不是单一的“正确更新下降”；
- 更稳定的发现是一般 margin shift；
- 代价体现在缺乏选择性，而非所有 V 都必然下降。

### 原 §4.2 Cost_V

删除把单个 `Cost_V` 当成全文主结论的设计。

换成：

- V 与 F 联合分析；
- CommonUpdate；
- AnswerSelectivity；
- m_general / m_specific；
- E2A/E2C 的异质离散后果。

### 原 §4.3 Dose–response trade-off

从正文主结果删除。

原因：

- 五剂量目标模式没有稳定形成；
- 强行保留会使评审认为论文依赖失败的开发选择。

可在附录作为“why simple trade-off failed”报告。

### 原 §4.4 Static vs dynamic

不再作为独立主结果，因为已有证据不足以支持完整动态人格漂移叙事。

可压缩进相关工作或未来工作。

### 原 §4.5 Cross-model Llama

删除 TBD 跨模型结果。剩余时间不应仓促迁移轴符号、阈值和层位。

### 原 §5 Mechanism

保留章节位置，但完全改写：

- phase subadditivity；
- validity opposing tendencies；
- E1 falsification；
- boundary analysis；
- E2B conditional branch。

不要再预写“读出仍然完整但输出策略改变”，除非 E2B 真正支持。

### 原 §6 Controller ladder

从主文删除“已经提出更优控制器”的理想化路线。

替换为：

> 选择性控制器的设计规范。

如果之后真有一个经过安全与认识论双重评测的新 controller，再升级为方法贡献；否则不要让未完成的方法拖垮已成立的发现。

---

## 13. 论文的读者体验

评审读完后应形成下面这条清晰记忆：

1. 激活控制领域只看“有没有压住行为”，漏看“为什么改变”。
2. CEDAR 用同 target、不同理由的严格配对补上了这个缺口。
3. p25 的稳定作用不是聪明地区分真证据和歪理，而是移动一般答案几何。
4. N/R 说明这种变化可以潜伏，不立即改答。
5. V/F 说明一旦靠近边界，正确和错误更新都可能被推动。
6. E2C、C2 和 E−1b 说明离散方向可变，但 margin–choice 分离是重复出现的。
7. 因此“表示方向”与“选择性控制策略”是两回事。

这条叙事比“我们没测到原先预期的代价”更强，因为它不是围绕失败辩解，而是提出一个更一般的方法论发现。

---

## 14. 最后五天的写作优先级

### 必须完成

1. 按本框架重写 Introduction、§3、§4。
2. 生成 Figure 1 和 Figure 2。
3. 完成 E2C missing-value / family-LOO 零 GPU addendum。
4. 完成 boundary analysis，并据 Gate A/B/C 选择措辞。
5. 将全部协议偏离集中移入 §6 和附录。
6. 清理旧稿中所有虚构占位数字和未完成方法。

### 有结果就加入，没有也不影响投稿

1. E2B fixed-trace；
2. 一张更强的机制图；
3. 小规模 activation patching。

### 当前不要启动

1. 新 controller 大规模开发；
2. MUSE/J-Lens/SAE 全工具栈；
3. 仓促跨模型复制；
4. 新一轮数据集微调；
5. 为了得到漂亮曲线重新筛题。

---

## 15. 最终拍板

这篇论文不再讲：

> 我们证明一种已经有效的安全 capping 付出了固定的正确更新代价。

而讲：

> **我们发现，Assistant-Axis capping 并不是一个按理由有效性工作的选择性控制器。它稳定地移动答案修订的内部几何，而这一连续位移会在不同决策边界上产生正确更新、错误说服或无离散变化。CEDAR 因而揭示了激活控制评测遗漏的一项基本要求：控制器不仅要改变模型行为，还必须保留模型对“为何应该改变”的敏感性。**

这是当前证据能够支撑的最强主线，也是最不依赖未来单个实验成败的 AAAI 投稿结构。
