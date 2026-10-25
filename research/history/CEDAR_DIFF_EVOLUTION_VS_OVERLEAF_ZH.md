# evolution 最终稿（7/7/7 达标版） vs 当前 Overleaf — 中文对照差异报告

比较对象
- **A = evolution/evolution/main.tex** —— AI 审稿流水线跑完 Medium/High/Xhigh 三阶段、三位评审同时给 7 分的那个 commit（`323e7d0`）。
- **B = CEDAR_overleaf_AAAI27/main.tex** —— 我们现在 Overleaf 上的版本。

共 **33 处实质差异**（已排除纯空白差异）。可以分成三类：

| 类别 | 处数 | 性质 |
|---|---:|---|
| **甲：表述被"大胆化"** | 9 | B 更有攻击性、更像发现型论文。**建议保留 B。** |
| **乙：评审要求的限定被删掉了** | 6 | B 把三轮评审专门要求补上的限定又删了。**⚠️ 建议按 A 恢复。** |
| **丙：`Appendix` → `Supplementary` 等术语统一** | 18 | 无风险，保留 B。 |

---

## ⚠️ 乙类：必须回补的 6 处（这是本报告的重点）

这 6 处不是风格问题。它们正是 round-001 / 003 / 004 三轮评审点名要求加上的限定，B 版把它们删掉了，等于把已经修好的问题又打开了。每处都只是一到两句话，改动成本极低。

### 乙-1｜new-salt 的"缺一个配对 + 预注册门槛未满足"被删（HUNK 15）**最严重**

**A（达标版）**
> The same-family new salt produced a **complete-case** increase of $1.538$ logits ($[0.455,2.741]$)... **One planned continuous pair lacked logits, leaving the prespecified completeness rule unmet; worst-observed sensitivity retained the positive direction.**

**B（现在）**
> The same broad quantity---the average non-initial advantage over $a_0$---rose by $1.538$ logits ($[0.455,2.741]$) under a new salt.

**问题**：R1-2（round-001）、R3-2（round-001）、R1（round-002）三次要求"主结果句必须直接说明缺一个 primary continuous pair，并区分 complete-case 估计、敏感性与正式状态"。B 版把这句话整段删了，现在读起来像一个已完成的确认性结果。
**改法**：在该句后加回一句（约 22 词）：
> One planned continuous pair lacked logits, so the prespecified completeness rule was not met; worst-case sensitivity retained the positive direction.

### 乙-2｜Social 的"仅限固定自由生成子集"限定被删（HUNK 16）

**A**：`In the fixed free-generation subset, no paired $S$ answer changed.`
**B**：`No paired $S$ answer changed.`

**问题**：round-003 R2-4 明确指出——附录的 direct scoring 里**存在** $S$ 臂的 initial→target 迁移。删掉"在固定自由生成子集中"这个限定之后，正文与附录直接矛盾。这是评审最容易抓的一类硬伤。
**改法**：把 `In the fixed free-generation subset,` 加回去。**成本：5 个词。**

### 乙-3｜推断总体（inferential population）声明被删（HUNK 12）

**A**：`...bootstrapped over 12 structural families; **their inferential population is the equally weighted sampled family design.** Social effects are...`
**B**：删掉了中间那句。

**问题**：round-002 R2-2 与 round-003 R1-2 都要求"明确 inferential population"，因为只有 12 个独立聚类。删掉之后，12-cluster 的批评没有任何防守。
**改法**：整句加回。**成本：11 个词。**

### 乙-4｜诊断表里"有效性读数"一行被改成了过强断言（HUNK 22 / 23）

| | 行标题 | 结论列 |
|---|---|---|
| **A** | Independent validity readout | The point estimates diverged; **validity uncertainty leaves preservation versus partial loss unresolved.** |
| **B** | Damaged reason evaluation | Revision moved without matching validity loss; **validity remained readable.** |

**问题**：validity discrimination 的区间是 $[-25.00, +20.83]$ pp，**包含与 CommonUpdate 同量级的损失**。R1-1、R2-1、R3-1（round-001 三人**一致**）都指出：在跨条件探针迁移与因果 patching 完成前，不得声称"排除了 validity 被削弱"。B 版的写法正是被三人否掉的那一种。
**改法**：结论列改回 A 的措辞，或折中为
> Revision moved while the validity estimate stayed near zero with a wide interval; preservation versus partial loss remains unresolved.

### 乙-5｜"随机参考没有可比变化" → "接近零"（HUNK 26）

**A**：`matched random references showed no comparable change.`
**B**：`matched random references remained near zero.`

**问题**：两个版本其实都不够准，但 B 更容易误导。四个随机参考在**干预窗口出口处就已经接近轴外占比上限**，所以后续"几乎没有增长"是天花板效应，不是"随机扰动不会重混"。
**改法**（比 A、B 都好，约 18 词）：
> the matched random references had already reached the off-direction ceiling at the window exit, so the comparison identifies a different propagation profile rather than an absence of rotation.

### 乙-6｜"rotates away" → "continues away"（HUNK 25 / 30）

**A**：`Downstream Displacement **Rotates** Away from the Exit Coordinate`，表中写 `off-exit-axis share increased`
**B**：`The Cap's Coordinate Does Not Contain Its Downstream Effect`，表中写 `share grew`

**问题**：round-002 R1-2 专门指出，$G_{\mathrm{off}}$ 是**占比**（composition）的变化，占比上升 ≠ 轴外**绝对范数**增长。A 用 "composition/rotates" 是对的，B 的 "does not contain its effect" 更强但没有绝对范数支撑。
**改法**：小节标题可以保留 B（更好读），但正文里至少保留一次 "composition"/"share" 的措辞，不要写成绝对量增长。

---

## 甲类：B 更好的 9 处（建议保留，不要回退）

### 甲-1｜标题
- **A**：`CEDAR: Assistant-Axis Capping Shifts Answer Geometry under Token-Identical Reasoning Traces`
- **B**：`The Cap Moves Answer Geometry, Not the Reasoning Trace: Representational--Policy Decoupling under Assistant-Axis Capping`
→ B 把发现放在前面、把自造术语放在副标题，是发现型论文的写法。保留 B。

### 甲-2｜摘要开头
- **A**：「激活干预通常通过目标行为与静态能力评测。CEDAR 审计一项互补性质……」——先说方法。
- **B**：「激活干预通常只看目标行为是否下降。**我们揭示了公开 Assistant-Axis p25 钳制的一个隐藏效应**：它反复移动答案几何，却没有让有效理由相对谬误理由取得稳定优势。」——先说发现。
→ 保留 B。**但要注意**：B 的摘要里也把 new-salt 的 gate 限定删了（同乙-1），需要补一个从句。

### 甲-3｜贡献 (2)(3)
- **A (2)**：「我们建立了一个反复出现的广泛答案几何位移……」（陈述式）
- **B (2)**：「我们揭示了……并**排除了**目标吸引、有效性判断被抹除、选择性抵抗状态、可见推理改变这四种解释。」
→ B 的"逐一排除"写法更有力。**但 "erased validity judgment" 这一项与乙-4 是同一个过强断言**，建议改成 `a validity judgment that moved with revision`（即：我们排除的是"有效性判断与改答同步移动"，而不是"有效性被抹除"）。**这是一个词组的修改，但把三位评审的 HIGH 级意见挡掉了。**

### 甲-4｜第五节开场（HUNK 18）
- **A**：「我们分离目标暴露、干预阶段、独立有效性读数、轨迹构成与可见 token 变化，然后追踪位移。」——实验菜单。
- **B**：「我们**先检验了那些直觉性的解释**——有效性判断受损、抵抗被打断、可见推理被改变——**然后追踪存活下来的位移**。」——假说链。
→ 保留 B，这正是你要的"曲折探索路径"。

### 甲-5｜§6.1 假说段（HUNK 29）
- **A**：「诊断模式促使我们区分可用信息与其下游使用。」
- **B**：「这些实验蕴含着**比常规能力损失更深的一种分离**……为解释这一组合，我们提出……」
→ 保留 B。

### 甲-6｜§6.2 控制器（HUNK 31）
- **A**：「该解释给控制器设计两个目标。」
- **B**：「该解释**改变了控制器的设计目标**。更好的钳制不应只是把模型留在人格坐标附近；它应当保住证据敏感的移动。」
→ 保留 B。

### 甲-7｜结论（HUNK 33）
- **A**：两段式结果复述 + 未来测试。
- **B**：「CEDAR 揭示了一个隐藏的认识论效应……**稳定模型表现出来的角色，并不保证保留它改变想法的方式。**」
→ 保留 B。最后一句是全文最好的一句。

### 甲-8/9｜§4.3 与诊断表行标题改成问句式（HUNK 15 标题、21/24/27）
- A：`Target-hidden movement` / `Token-identical replay` / `Matched-model`（记录了做了什么）
- B：`Target-specific attraction` / `Changed visible reasoning` / `Does the pattern...`（记录了排除了什么假说）
→ 保留 B，表格从"实验清单"变成了"假说排除表"。

---

## 丙类：18 处术语统一

`Appendix Section~X` → `Supplementary Section~X`，全篇一致。无风险。
⚠️ 但要检查 **附录文件 `appendix.tex` 的标题是否同步**——round-002 R3-3 提过附录还在用旧标题。

---

## 最后：六小时内的建议动作

按性价比排序，全部是文字改动，无需实验：

| 优先级 | 动作 | 成本 |
|---|---|---|
| **P0** | 乙-2 Social 加回 `In the fixed free-generation subset,` | 5 词，消除正文↔附录矛盾 |
| **P0** | 乙-4 诊断表"有效性"行改回未决措辞 | 1 行，挡掉三位评审一致的 HIGH 意见 |
| **P0** | 甲-3 贡献 (2) 的 `erased validity judgment` → `a validity judgment that moved with revision` | 1 个词组 |
| **P1** | 乙-1 new-salt 补回 gate 未满足（正文 + 摘要各一处） | 约 22 + 12 词 |
| **P1** | 乙-3 补回 inferential population 一句 | 11 词 |
| **P2** | 乙-5 随机参考的天花板限定 | 18 词 |
| **P2** | 乙-6 保留一次 composition/share 措辞 | 2 个词 |
| **P3** | 检查 `appendix.tex` 标题与 `main.tex` 是否一致；检查是否还有硬编码的 `Section 6` | — |

甲类**全部保留**，一个都不要回退。