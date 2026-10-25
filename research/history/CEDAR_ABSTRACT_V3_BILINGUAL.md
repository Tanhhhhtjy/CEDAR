# 摘要 v5（中英对照）— 未推送

> cocodeco顺序：**Assistant Axis 提出了什么、忽略了什么（思考） → 我们的问题与设计 → 直觉性解释逐一失效 → 但这不代表没有认识论代价：一个更坚实的现象浮出水面 → 贯穿性的否定事实 → 两级假说 → 工具（一句，从属）**。
> 约 265 词。

---

## English

```latex
\begin{abstract}
\citet{lu2026assistant} introduce a new class of activation-level operator: rather than adding a steering vector, a percentile-calibrated cap rectifies a persona coordinate only where it exceeds a bound, aiming to hold persona and safety in place while disturbing as little else as possible. What that work does not evaluate is whether capping carries an epistemic cost---whether the decision logic by which the model separates good evidence from bad is itself affected---and this, rather than persona drift, is where a behavioral side effect would actually live. We test it directly, holding a proposed revision fixed while varying only the warrant behind it, certifying that warrant with a solver rather than with the audited model, and separately holding target exposure fixed while adding only social pressure.

The intuitive explanations failed in turn. An independent validity readout did not move with revision. A registered resistance-state account did not survive trajectory-length matching. Freezing every visible reasoning token removed nothing. This does not mean the cap is epistemically free; on the contrary, a firmer phenomenon surfaced once those accounts were gone. What survived every control is a broad movement of answer geometry: in Qwen3-32B the pooled non-initial-versus-initial margin rose by $3.130$ logits (95\% family-cluster CI $[1.834,4.486]$) even when the proposed target was hidden; a matched forced-choice study recovered the shift in both released Qwen and Llama configurations ($+2.075$ and $+1.391$); a same-family re-instantiation supplied supportive but formally incomplete evidence and redistributed the discrete revisions; token-identical replay retained $+2.078$ logits; and after rectification stopped the displacement continued away from the cap's exit coordinate ($G_{\mathrm{off}}=0.161$, $[0.151,0.170]$) while validity labels remained decodable. Throughout, no continuous valid-minus-fallacious contrast resolved a change in selectivity. We propose \emph{representational--policy decoupling}, and conjecture more broadly that single-axis activation control redistributes decision gain in proportion to each pathway's coupling with the intervened axis. CEDAR, the certified audit built to ask the question, remains available for the next operator.
\end{abstract}
```

> ⚠️ AAAI 摘要中通常不放 `\citet`。若要避免，第一句改为：
> `A recently released Assistant-Axis controller introduces a new class of activation-level operator: rather than adding a steering vector, ...`
> 正文引言首句再给引用。**推荐用这一版**，摘要更干净。

---

## 中文

Assistant Axis 提出了一类新的激活层算子：它不是叠加一个 steering 向量，而是以按分位数标定的钳制，只在人格坐标越过边界处进行整流，目标是在稳住人格与安全性的同时，尽可能少地扰动其余行为。而那项工作没有评测的是——**钳制是否带有认识论代价**：模型据以区分好证据与坏证据的决策逻辑本身是否受到影响。真正可能存在的行为层副作用正在这里，而不在人格漂移上。我们直接检验它：固定被提议的修订，只改变它背后的理由；由求解器、而不是由被审计的模型来认证这条理由是否有效；并另设一组对照，固定目标暴露、只增加社会压力。

那些直觉性的解释逐一失效。独立的有效性读数没有随改答一同移动。预注册的"抵抗状态"解释没有通过轨迹长度匹配。冻结每一个可见推理 token，也没有消除任何东西。**但这并不意味着钳制在认识论上是免费的；恰恰相反，当这些解释被逐一排除之后，一个更坚实的现象浮出了水面。** 在所有对照之后仍然存活下来的，是答案几何的一次广泛移动：在 Qwen3-32B 上，即使被提议的目标被隐藏，合并的"非初始相对初始" margin 仍上升了 $3.130$ logits（95\% family-cluster CI $[1.834,4.486]$）；一项匹配的强制选择研究在公开的 Qwen 与 Llama 两个配置上都复现了这一位移（$+2.075$ 与 $+1.391$）；同族的重新实例化提供了具支持性但形式上不完整的证据，并重新分配了离散层面的改答；token 完全相同的重放保留了 $+2.078$ logits；而在整流停止之后，该位移继续离开钳制的出口坐标（$G_{\mathrm{off}}=0.161$，$[0.151,0.170]$），同时有效性标签仍然可解码。自始至终，没有任何一个连续的"有效减谬误"对比分辨出选择性的变化。我们据此提出\emph{表征—策略解耦假说}，并进一步猜想：激活层的单轴控制会按各通路与被干预轴的耦合程度重新分配决策增益。CEDAR——为了提出这个问题而构建的认证审计——留下来是为了下一个算子。

---

## 相对 v4 的三处改动

1. **第 1 句彻底重写。**
   删掉了"steering 已被研究多年 / 阈值整流是另一种算子 / 它仅以人格稳定性得到认可"这一整串铺垫。现在直接说 Assistant Axis **提出了什么**：一类新的激活层算子，用非线性的钳制控制，在稳住人格与安全性的同时尽量少扰动其余行为。场域一句话锁定在 capping，且把功劳先给足对方——这比先划分 steering/capping 更有礼貌，也更省字。
2. **第 2 句改成"它忽略了什么"，并点名"认识论代价"。**
   `whether capping carries an epistemic cost---whether the decision logic by which the model separates good evidence from bad is itself affected---and this, rather than persona drift, is where a behavioral side effect would actually live.`
   这一句把"我们要问的问题"直接定义为一个**代价**，而不是一个中性的性质；后半句 `rather than persona drift` 说明为什么原评测看不见它。
3. **新增关键转折句（你要求的那一句）。**
   `This does not mean the cap is epistemically free; on the contrary, a firmer phenomenon surfaced once those accounts were gone.`
   放在三次失效之后、广泛位移之前。没有这一句，读者（尤其审稿人）会把"三个解释都失效"读成"我们没测到认识论代价"，整段的意义会反转。有了它，三次失效变成**通往真正发现的路径**，而不是失败清单。

---

## 逐句层级对照

| 位置       | 层级                                                                  |
| ---------- | --------------------------------------------------------------------- |
| 第 1 句    | **背景**：Assistant Axis 提出了什么（capping 场域锁定）         |
| 第 2 句    | **思考**：它没有评测的问题 = 认识论代价，且这才是副作用真正所在 |
| 第 3 句    | **设计**：$V/F$、求解器认证、$AO/S$，均为该问题的从属条件   |
| 第 4–6 句 | **实验（曲折）**：三个直觉解释逐一失效                          |
| 第 7 句    | **转折**：这不代表没有代价，反而浮出更坚实的现象                |
| 第 8 句    | **实验（存活）**：答案几何位移 + 五项证据                       |
| 第 9 句    | **贯穿性否定**：$V-F$ 始终未分辨出变化                        |
| 第 10 句   | **思考（产物）**：两级假说，`propose` / `conjecture` 分级   |
| 第 11 句   | **工具**：CEDAR，一句，指向下一个算子                           |

---

## 与两轮 AI 审计要求的对照（全部满足）

| 要求                                      | 是否满足                                                 |
| ----------------------------------------- | -------------------------------------------------------- |
| round-005#3：不得断言"理由改变答案的路径" | ✅ 只有`propose` 与 `conjecture`                     |
| round-005#2：未完成实验用条件式           | ✅ 摘要中不出现 transfer/patching                        |
| round-005#4：new-salt 形式状态入主叙述    | ✅`supportive but formally incomplete`                 |
| round-007：删除`centered / near zero`   | ✅`did not resolve a change in selectivity`            |
| round-002：off-axis 用 composition 措辞   | ✅`continued away from the cap's exit coordinate`      |
| round-001 R1-3：不得称 control failure    | ✅ 用`epistemic cost` 提问、用现象作答，不宣称控制失败 |

> 顺序：**capping 的背景与它没有回答的问题（思考） → 我们的问题与设计 → 直觉性解释逐一失效（实验的曲折） → 唯一存活下来的现象：答案几何位移（最坚实的证据） → 两级假说 → 工具（一句，从属）**。
> 约 265 词。

---

## English

```latex
\begin{abstract}
Steering by activation addition has been studied for years, but thresholded rectification is a different operator: it fires only when a coordinate exceeds a calibrated bound, so it can act during prompt processing, deliberation, or answer formation while leaving persona behavior apparently unchanged. The released Assistant-Axis cap is such an operator, and it is certified by persona stability alone; whether a capped model still separates a good reason for changing its answer from a bad one is neither answered nor asked. We ask it directly, holding a proposed revision fixed while varying only the warrant behind it, certifying that warrant with a solver rather than with the audited model, and separately holding target exposure fixed while adding only social pressure. The intuitive explanations failed in turn. An independent validity readout did not move with revision. A registered resistance-state account did not survive trajectory-length matching. Freezing every visible reasoning token removed nothing. What survived every control is a broad movement of answer geometry: in Qwen3-32B the pooled non-initial-versus-initial margin rose by $3.130$ logits (95\% family-cluster CI $[1.834,4.486]$) even when the proposed target was hidden; a matched forced-choice study recovered the shift in both released Qwen and Llama configurations ($+2.075$ and $+1.391$); a same-family re-instantiation supplied supportive but formally incomplete evidence and redistributed the discrete revisions; token-identical replay retained $+2.078$ logits; and after rectification stopped the displacement continued away from the cap's exit coordinate ($G_{\mathrm{off}}=0.161$, $[0.151,0.170]$) while validity labels remained decodable. Throughout, no continuous valid-minus-fallacious contrast resolved a change in selectivity. We propose \emph{representational--policy decoupling}, and conjecture more broadly that single-axis activation control redistributes decision gain in proportion to each pathway's coupling with the intervened axis. CEDAR, the certified audit built to ask the question, remains available for the next operator.
\end{abstract}
```

---

## 中文

以激活加法进行的 steering 已经被研究多年，但**阈值整流是另一种算子**：它只在某个坐标越过标定边界时才触发，因此可以在提示处理、推理或答案形成阶段发生作用，而人格行为看上去毫无变化。公开的 Assistant-Axis 钳制正是这样一种算子，而它仅以人格稳定性得到认可；至于被钳制的模型是否仍然能把"改答的好理由"与"改答的坏理由"分开——这个问题既没有被回答，也没有被提出。我们直接提出它：固定被提议的修订，只改变它背后的理由；由求解器、而不是由被审计的模型来认证这条理由是否有效；并另设一组对照，固定目标暴露、只增加社会压力。

那些直觉性的解释逐一失效。独立的有效性读数没有随改答一同移动。预注册的"抵抗状态"解释没有通过轨迹长度匹配。冻结每一个可见推理 token，也没有消除任何东西。**在所有对照之后仍然存活下来的，是答案几何的一次广泛移动**：在 Qwen3-32B 上，即使被提议的目标被隐藏，合并的"非初始相对初始" margin 仍上升了 $3.130$ logits（95\% family-cluster CI $[1.834,4.486]$）；一项匹配的强制选择研究在公开的 Qwen 与 Llama 两个配置上都复现了这一位移（$+2.075$ 与 $+1.391$）；同族的重新实例化提供了具支持性但形式上不完整的证据，并重新分配了离散层面的改答；token 完全相同的重放保留了 $+2.078$ logits；而在整流停止之后，该位移继续离开钳制的出口坐标（$G_{\mathrm{off}}=0.161$，$[0.151,0.170]$），同时有效性标签仍然可解码。自始至终，没有任何一个连续的"有效减谬误"对比分辨出选择性的变化。我们据此提出\emph{表征—策略解耦假说}，并进一步猜想：激活层的单轴控制会按各通路与被干预轴的耦合程度重新分配决策增益。CEDAR——为了提出这个问题而构建的认证审计——留下来是为了下一个算子。

---

## 相对 v3 的三处改动

1. **第 1 句改成 capping 与 steering 的分野。**
   v3 写的是"激活干预一向由它所针对的行为来认可"，太泛，读者会以为我们在做 steering。v4 直接指出：加法 steering 已有大量工作，而**阈值整流是另一种算子**——只在越界时触发，因此可以在推理与答案形成阶段起作用而人格看不出变化。这既划清了场域，也顺势说明了"为什么人格指标看不见它"。
2. **第 2 句写 Assistant Axis 提供了什么、遗漏了什么。**
   `it is certified by persona stability alone; whether a capped model still separates a good reason ... is neither answered nor asked.` —— 明确它提出了一种新的激活层控制方法，而它没有回答的正是我们要问的。
3. **顺序调转：先失效，后存活。**
   v3 是"先给结果、再说反转"，逻辑上是倒的。v4 改成：直觉解释逐一失效 → `What survived every control is a broad movement of answer geometry` → 然后一口气给出五项证据（目标隐藏、跨模型、新 salt、固定轨迹、下游传播）。这样"答案几何位移"是**被剩下来的**，而不是被先验假定的，与我们真实的研究路径一致。
   `Throughout, no continuous valid-minus-fallacious contrast resolved a change in selectivity.` 单独成句放在证据之后，作为贯穿全部条件的否定性事实。

---

## 逐句层级对照

| 位置       | 层级                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------- |
| 第 1–2 句 | **思考**：capping 与 steering 的区别 → Assistant Axis 提供了什么 → 它没有问的问题 |
| 第 3 句    | **设计**：$V/F$、求解器认证、$AO/S$，均为该问题的从属条件                       |
| 第 4–6 句 | **实验（曲折）**：三个直觉解释逐一失效                                              |
| 第 7 句    | **实验（存活）**：答案几何位移 + 五项证据                                           |
| 第 8 句    | **实验（贯穿性否定）**：$V-F$ 始终未分辨出变化                                    |
| 第 9 句    | **思考（产物）**：两级假说，`propose` / `conjecture` 分级                       |
| 第 10 句   | **工具**：CEDAR，一句，指向下一个算子                                               |

---

## 与两轮 AI 审计要求的对照（与 v3 相同，全部满足）

| 要求                                      | 是否满足                                            |
| ----------------------------------------- | --------------------------------------------------- |
| round-005#3：不得断言"理由改变答案的路径" | ✅ 只有`propose` 与 `conjecture`                |
| round-005#2：未完成实验用条件式           | ✅ 摘要中不出现 transfer/patching                   |
| round-005#4：new-salt 形式状态入主叙述    | ✅`supportive but formally incomplete`            |
| round-007：删除`centered / near zero`   | ✅`did not resolve a change in selectivity`       |
| round-002：off-axis 用 composition 措辞   | ✅`continued away from the cap's exit coordinate` |
| round-001 R1-3：不得称 control failure    | ✅ 已删除                                           |
