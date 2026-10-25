# Figure 2（暂定编号）单栏小图：CEDAR-Evidence 条目的制造与认证

> 位置：§3.2 CEDAR-Evidence 首页，单栏浮动（`figure`，非 `figure*`）。
> 尺寸：宽 3.3 in（AAAI 单栏满宽），高 2.3–2.6 in。
> 它与 Figure 1 的分工：**Figure 1 说“四个臂各自应该发生什么”（语义）；本图说“这些 gold 从哪来、V/F 凭什么算匹配”（来源与认证）。**

---

## 1. 这张图唯一要让审稿人相信的事

> V 和 F 是同一道题的镜像，除了一步被机器认证为致命错误的推理之外没有任何差别；而且所有 gold 都是求解器判定的，不是模型或人判的。

因此图的重心不是“我们如何生成题目”（那是 ProverGen 的图要做的事），而是**对称 + 认证**。生成流程只需交代到能让读者理解对称性从哪里来为止。

---

## 2. 结构（自上而下五层）

```text
┌──────────────────────────────────────────────┐
│ ① Horn world W + query Q                     │
│    ┌────────────────────────────────┐        │
│    │ p(x) ∧ q(x) → r(x) ;  ¬s(b) …  │        │  ← 3–4 行等宽字体
│    └────────────────────────────────┘        │
│                unique  a₀                    │
└──────────────────────────────────────────────┘
                     │  one atomic edit  δ
                     ▼
┌──────────────────────────────────────────────┐
│ ② edited world W⊕δ            unique  aₜ     │
└──────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
╔═════════════════╗       ╔═════════════════╗
║  V              ║       ║  F              ║     ← 两框等宽等高，严格镜像
║ justification   ║       ║ same edit δ,    ║
║ proves δ        ║       ║ one certified   ║
║                 ║       ║ fatal step      ║
║  gold = aₜ      ║       ║  gold = a₀      ║
╚═════════════════╝       ╚═════════════════╝
   └────────── shared wrapper ──────────┘          ← 一条贯穿两框的浅灰底带
      reviewer identity · speech act ·
      proposed edit · output contract
                     │
                     ▼
┌──────────────────────────────────────────────┐
│ ③ certification                              │
│   Python fixed point  ⟺  Prover9             │  ← 等宽字体，像“机器检查条”
│   object-level answer  +  meta-level warrant │
│   Mace4 countermodel (F) · exactly one fatal │
│   step · no second unsupported step          │
└──────────────────────────────────────────────┘
                     │
                     ▼
  12 families × {easy, medium} = 24 bases × {N,R,V,F}
  + one independently salted bank, analyzed separately
```

### 各层的取舍理由

| 层 | 必须画 | 可以省 |
|---|---|---|
| ① | 世界 + 唯一 a₀ | 完整规则集，只给 3–4 行示意 |
| ② | “一次原子编辑”这个箭头标签 | 编辑是如何被确定性选出的（放正文/附录） |
| ③ | **V/F 镜像 + 共享 wrapper 底带** | 具体 justification 文本（放附录样例） |
| ④ | 三个求解器与它们各自负责什么 | 版本号、超时、命令行（放附录） |
| ⑤ | 规模与第二个 salted bank | 六类 error class 的细分（放附录表） |

N 与 R 不进图，只在 caption 里一句：`N and R reuse the same base with no new grounds.` 单栏放不下六个臂，而且它们不参与“认证”这条主线。

---

## 3. 视觉规则（这几条决定图的说服力）

1. **对称性即论证。** V 框与 F 框必须像素级等宽等高、同字号、同行数、同边框粗细。唯一允许的差别是右下角的徽标：V 用 `✓ proof`，F 用 `✗ countermodel`。如果两框看起来不一样大，读者就会怀疑它们的题干也不一样。
2. **共享 wrapper 画成“底带”而不是文字说明。** 一条横跨两框下缘的浅灰色条，写上被固定的四项。这比在正文里写一句 “we match reviewer identity, speech act…” 有效十倍。
3. **认证条用等宽字体**（`\ttfamily` 或图中 monospace），与上方的自然语言框形成质感差异，暗示“这一层是机器做的”。
4. **箭头只有一个方向、不加装饰。** 单栏图里任何弯箭头都会显得拥挤。
5. **灰度可读。** V/F 若用色，选 teal / orange-red，但**必须**同时用 `✓ / ✗` 徽标与实线/虚线边框区分；AAAI 明确要求 “decipherable without using color for distinction”，且正文字体不得用色。
6. **字号 ≥ 8 pt**（正文 10 pt，图内不要小于 8 pt）；等宽片段可 7.5 pt 但不要更小。
7. 导出 PDF 矢量，字体嵌入，无 Type 3。

---

## 4. 参考哪些论文的图

| 参考对象 | 借鉴什么 | 不要照抄什么 |
|---|---|---|
| **ProverGen**（arXiv 2502.06563，Fig. 1 “An overview of our ProverGen framework”） | 用一个贯穿始终的微型实例把各阶段串起来：背景故事 → 逻辑骨架（Prover9 定真值、加干扰项）→ 自然语言翻译。这种“一个例子走完全程”的做法让读者不用读正文就能理解产物长什么样。 | 它是横向三大框、信息量很大的**满宽**图，且重心在“生成”。我们单栏放不下，重心也不同。 |
| **LoFa**（arXiv 2606.31039，Fig. 4 “Overview of the multi-agent pipeline for constructing the LoFa dataset with iterative quality assurance”） | 把**质量保证**画成图里的独立元件（它用 annotator agent + 回流箭头），而不是藏在正文。我们的对应物就是那条 certification 条。 | 它的 QA 是 LLM annotator 迭代；我们的是求解器判定，**这正是我们的差异点**，不要画成 agent 循环，要画成一次性机器验证。 |
| **LoFa** Fig. 3（“From Ambiguous Opinion to Measurable Error”） | “为什么必须有客观 ground truth”的概念对比图——如果审稿人质疑我们为何不用价值域题目，这类图是标准答法。 | 我们已有 Figure 1 承担概念说明，不要再画第二张概念图。 |
| **CAA**（arXiv 2312.06681） | 对比对（正/负）并排、严格镜像的画法。 | 它是 steering 方法图，不涉及认证。 |
| **Belief-R**（EMNLP 2024） | “同一 base + 两种 follow-up”的最小对呈现方式。 | 它没有求解器认证层。 |

**一句话总结差异化定位**：ProverGen 的图回答 “how were items generated”，LoFa 的图回答 “how were items quality-assured by agents”，我们的图回答 **“how is the V/F difference certified to be exactly one fatal step, with everything else held fixed”**——这是前两者都没有的，也正是本文方法贡献的核心。

---

## 5. Caption 草稿（自包含，AAAI 风格）

> **Figure 2: Construction and certification of a CEDAR-Evidence cluster.** A finite function-free Horn
> world with explicit negation and a four-option query yields a unique initial answer $a_0$; a single
> deterministically selected atomic edit yields a unique alternative $a_t$. The valid arm ($V$) supplies a
> justification that proves the edit warranted, while the matched fallacious arm ($F$) proposes the same
> edit through exactly one certified fatal step, leaving $a_0$ correct. The two arms share reviewer
> identity, speech act, proposed edit, wrapper skeleton, and output contract (shaded band). Python
> fixed-point inference and Prover9 must agree on both the object-level answer and the meta-level
> warrant, and every $F$ instance carries a Mace4 countermodel. $N$ and $R$ reuse the same base with no
> new grounds. Full certificates and worked examples appear in [TBD-APPENDIX].

（LoFa 的 caption 是自包含整句，ProverGen 的是极简短语。AAAI 审稿人跳读多，建议采用 LoFa 风格。）

---

## 6. 如果 0.3 页放不下

按此顺序降级：

1. 删 ⑤ 规模行 → 移入正文（正文已有该句）；
2. 删 ① 的符号片段，只留 “Horn world + 4-option query → unique $a_0$”；
3. 认证条从三行压成两行（合并 “exactly one fatal step / no second unsupported step”）；
4. **最后才考虑删共享 wrapper 底带——但删了这张图就失去了存在意义**，此时应整图移入附录，正文只留 Figure 1。