# 尺度收缩发现的落位方案（改动清单，尚未执行）

> 定案：**数据全部进第五节，理论全部进第六节。**
> 第六节的最高理论贡献 = 表征—策略解耦（伞形概念）+ 它的第一个被测到的实例（暴露定量的收缩）+ 唯一仍未测的那一步（为什么累计修正会映射成 logit 尺度）。
> 版面：新增约 22 行，删除约 26 行，**净减约 4 行**。

---

## 一、先回答你的两个问题

**1. 第五节结论"反复出现的经验模式是一次广泛的答案几何位移"——是被推翻还是被加深？**

**现象被加深，措辞被证伪。**

- 位移本身在五个 assay、两个模型上全部复现，而且现在有了闭式描述 $\Delta m_{\mathrm{general}} = (a-1)\,m^{\mathrm{unc}}_{\mathrm{general}}$。这比原来强。
- 但"广泛（broad）/跨候选项空间的重分配"这个描述是错的：它不是重分配，是**选项 logits 的整体尺度收缩**。$\Delta m_{\mathrm{specific}}\approx 0$ 也不再是一个独立发现，而是收缩假设的**必然推论**（并且它通过了检验）。
- 最被加深的是"对理由不敏感"：原来是从一串零结果**推断**出来的，现在是从算子形式**推导**出来的——仿射收缩在数学上不可能优待 $V$。

**2. 表征—策略解耦是否与新机制互斥、可否被替代？**

**不互斥，也不该被替代——它们是两个层级。** 解耦说的是"变的是路径不是内容"；新机制说的是"这条路径具体怎么变的"。新结果恰好把解耦的两半都测出来了：

| 解耦假说的断言 | 现在的证据 |
|---|---|
| 内容被保留 | 探针 AUC 0.769 / 0.767，纯文本基线 0.511 |
| 路径被改变 | 答案头上的仿射收缩 $a=0.44\sim0.91$，仿射部分解释 83–102% 的 margin 位移 |
| 改变与理由无关 | 固定轨迹四臂同一收缩（0.896–0.908）；自然生成中 $a_V-a_F$、$\varepsilon_V-\varepsilon_F$ 区间全部跨零 |

**但新机制在三点上严格更强**：它是被测量的而不是被假设的；它是定量且可预测的；它已经通过了一次证伪检验（no-op 给出 $a=1.0000$）。

**同时它纠正了解耦的一处措辞过头。** 原文暗示"策略不再使用这条信息"。数据不支持——未钳制时模型明显在用（$V$ 跟随 70.8% vs $F$ 29.2%）。正确说法是：**钳制叠加了一个与理由无关的项，模型原有的理由敏感性必须与它竞争。** 这个版本更准确，也更好防守。

---

## 二、第五节改动

### 2.1 新增 §5.6（约 22 行）— 全节唯一的新增内容

标题建议：**`The Displacement Is a Contraction, Not a Redistribution`**
中文：**`这次位移是收缩，而不是重分配`**

放在 §5.5（RG1 传播）之后、章末总结之前。内容顺序：

1. **提出竞争解释**（2 行）
   > A single alternative would explain every result so far: if capping merely contracted the option-logit scale, a positive $\Delta m_{\mathrm{general}}$ would follow mechanically from the initial answer's advantage, with no redistribution among alternatives. Target concealment and fixed traces do not exclude it, so we tested it directly.

2. **方法与恒等式**（3 行）
   逐条目拟合 $\ell^{\mathrm{p25}} = a\,\ell^{\mathrm{unc}} + b + \varepsilon$；截距在 margin 差分中抵消，故 $\Delta m = (a-1)m^{\mathrm{unc}} + \varepsilon$ 项。

3. **结果：收缩成立**（4 行）
   五个 assay 的 $a$ 与"仿射解释比例"，合并成一句 + 一个小表或直接嵌入正文：
   固定轨迹 $0.900$（100.5%）、Qwen FC $0.442$（101.6%）、Llama $0.523$（99.6%）、new salt $0.863$（90.3%）、Social $AO/S$ $0.910/0.889$（83.2%/91.0%）。
   控制：no-op 重复给出 $a = 1.0000$；四个调度与范数匹配的随机方向中有两个也收缩（$0.934$、$0.953$），因此收缩不是 Assistant Axis 独有，但 p25 最强。

4. **关键结果：剂量由暴露决定，不由有效性决定**（8 行，本节的重点）
   - 固定轨迹下四臂收缩相同：$N\,0.902$、$R\,0.901$、$V\,0.908$、$F\,0.896$。
   - 自然生成下按"有没有理由"分化，区间排除零：$a_V-a_R = -0.212\ [-0.342,-0.098]$，$a_F-a_N = -0.311\ [-0.419,-0.206]$；非仿射残差 $\varepsilon_V-\varepsilon_R = +1.46\ [+0.76,+2.32]$，$\varepsilon_F-\varepsilon_R = +2.26\ [+1.43,+3.06]$。
   - 但按"理由对不对"完全不分化：$a_V-a_F = +0.114\ [-0.098,+0.310]$，$\varepsilon_V-\varepsilon_F = -0.796\ [-2.18,+0.78]$。
   - 成因是暴露量：$V/F$ 的可触发位置约为 $N/R$ 的 1.7 倍，累计投影修正量 $+26.7$k / $+30.0$k（区间排除零），而**触发率几乎相同（0.556–0.586）**；跨条目 $\mathrm{corr}(\text{累计修正},a) = -0.463$。
   - 自防守一句：$V$ 与 $F$ 之间暴露已近似匹配（12,122 vs 11,447 位置；修正量差 $+3{,}250$，区间跨零），因此 $V-F$ 的零结果不是被长度掩盖的。

5. **顺带解释两个此前的悬案**（3 行）
   - E2B 的 $P$ 减 $U$ 交互：$P$ 源轨迹已被收缩过一次（$a$ 为 $0.940$ vs $U$ 源 $0.862$），剩余空间更小。
   - 为什么 argmax 几乎不动：未钳制的初始领先中位数 $18.58$ logits，10% 收缩只给出约 $1.9$ logits；$\mathrm{corr}(|\text{初始差距}|,\Delta m)=0.576$。

6. **收束一句**（2 行）
   > The operator's dose is set by how much text the model produced under it, not by whether that text was a good reason.

### 2.2 §5 章末总结句改写

现在：
> Table 1 collects the chain: … all leave the broad displacement standing. Section 6 turns this pattern into a mechanistic hypothesis.

改为：
> The chain leaves one object standing, and the last test identifies it: the cap contracts the option-logit scale by an amount set by its own exposure, and nothing in that chain reads the validity of the reason it just processed. Section 6 asks what kind of control failure this is.

### 2.3 §4.2 措辞改写（省约 6 行）

- 标题 `Broad Compression of Initial-Answer Advantage` → **`The Cap Weakens the Current Answer's Advantage`**（去掉"broad/跨候选项空间"的暗示）
- 正文里 `compressing initial-answer advantage across the alternative-option space` → `weakening the current answer's advantage over its alternatives`
- $\Delta m_{\mathrm{specific}}\approx 0$ 的解释句改成一句：目标特异性接近零，与 §5.6 的收缩解释一致，而不是一个独立的否定结果。**这里省下的篇幅正好抵一部分新增。**

### 2.4 §4.4 删除，降级为一句推论（省约 10 行）

`Answer Geometry Moves Below the Choice Boundary` 整节的内容（Social 的 $-0.786$、无 $S$ 改答）现在是收缩账户的直接推论。做法：

- 把 Social 的两个数字并入 §4.3 结尾一句；
- 删掉小节标题与其余论述；
- "为什么连续效应不跨越边界"的解释挪到 §5.6 第 5 点（初始差距 18.58 vs 位移 1.9）。

### 2.5 *margin--choice dissociation* 降级（省约 10 行）

现在它被收缩账户**推导**出来了，不再需要作为一个自造术语隆重登场。

- §2 删掉 `We call this the \emph{margin--choice dissociation}.` 及其铺垫（约 25 词）
- §4.1 标题 `Calibration Reveals a Margin--Choice Dissociation` → **`Calibration: Continuous and Discrete Readouts Disagree`**
- 保留现象描述与那两个反向移动的数字，只是不再命名。
- **理由**：一篇 8 页论文最多扛两个自造术语。留下的两个是 *representational--policy decoupling*（伞形）与 *exposure-dosed contraction*（实例）。

---

## 三、第六节改动

### 3.1 章标题

现在：`From Displacement to Representational--Policy Decoupling`
改为：**`Representation Is Not Policy: A Dose Set by Exposure`**
或更短：**`Representation Is Not Policy`**（推荐，把机制放到 6.2 标题里）

### 3.2 §6.1 `The Hypothesis` → 重写为两段（不加新数字）

第一段：伞形概念，保持不变但**修掉那处过头的措辞**。

- 删：暗示"策略不再使用有效性信息"的表述
- 加：
  > Ordinary revision does use validity: uncapped, the model follows valid warrants far more often than fallacious ones. What the cap adds is a term that this sensitivity must compete with---one whose size is set by the intervention rather than by the reason.

第二段：把机制作为解耦的**第一个被测实例**引出。

> Section 5 gives this account a measured form. The route that changes is the effective scale at which option logits are read out, and its dose is the cap's own cumulative above-threshold correction---which grows with how much the model wrote, not with whether what it wrote was valid. We call this \emph{exposure-dosed contraction}. It is one instance of representational--policy decoupling, not the whole of it: content survives (validity stays readable), while the mapping to a decision is rescaled by a quantity that carries no epistemic information.

### 3.3 新的 §6.2 = 唯一保留的前沿推测（约 6 行）

标题：**`The Unmeasured Step`** / `尚未测量的那一步`

这是全文唯一诚实标注为"推测"的地方，也正好是你要的最高理论贡献：

> One step in this account is measured only at its two ends. We observe that cumulative rectification predicts the contraction ($r=-0.46$ across items), but not why a bounded, layer-local edit should act like a global change of readout scale. The natural conjecture is that removing above-threshold projection reduces residual-stream magnitude in directions the unembedding reads with high gain, so that a persona-motivated bound on one coordinate becomes an unintended temperature control at the answer head. If this holds, any thresholded one-sided operator---not only the Assistant Axis---inherits a reason-blind confidence term proportional to its own trigger exposure. Norm-matched dose curves, unembedding-gain decompositions, and length-matched target-hidden controls would decide it.

**为什么这一段值得占版面**：它把结论从"这个公开钳制有个副作用"提升到"**这一类算子在结构上都会带一个与理由无关、与话语长度成正比的置信度项**"，并且给出了三个可执行的判定实验。

### 3.4 §6.3 `Toward Selective Control` 改写（不增行）

现在写的是"分离 detection 与 actuation"。改为由机制推出的两条具体要求：

> The account names two design targets. First, gate on cumulative correction rather than on per-token projection, so that a longer answer does not silently receive a larger intervention. Second, protect the non-affine residual---the only component of the cap's effect that varied with the content of the argument---rather than a direction chosen for what it represents.

### 3.5 §6 `Scope.` 段补一句

> The contraction account is measured on recorded option logits at four positions per item; a full-vocabulary fit and a length-matched target-hidden control remain the two direct ways to break it.

---

## 四、其余牵连改动

| 位置 | 改动 |
|---|---|
| 摘要 | "broad compression / redistribution" → "contracts the option-logit scale by an amount set by its own trigger exposure"；删掉 margin–choice 的命名 |
| §1 贡献 (ii) | 由"记录了广泛压缩与异质离散后果"改为"发现该位移是一次暴露定量的尺度收缩，其剂量与理由有效性无关" |
| §1 贡献 (iii) | 加半句：并自行提出且检验了尺度收缩这一竞争解释 |
| 标题 | 现在有了更硬的落点，建议向"剂量由话语长度决定、与理由无关"靠拢，而不是"理由不再支配修订" |
| Figure 3 | Panel A 换成收缩证据：左半是五个 assay 的 $a$（forest），右半是 observed vs affine-predicted 的散点（$r=0.987$）。Panel B（轴外传播）、Panel C（探针）不动 |
| 附录 | 收缩分析的完整规格：拟合、bootstrap、五个 assay 的分臂表、暴露诊断、no-op 与随机参考控制 |

---

## 五、版面账

| 项 | 行数 |
|---|---:|
| 新增 §5.6 | $+22$ |
| §6.1 重写（净增） | $+4$ |
| 新增 §6.2 未测那一步 | $+6$ |
| §4.2 措辞压缩 | $-6$ |
| §4.4 删除并降级 | $-10$ |
| margin--choice 降级（§2 + §4.1） | $-10$ |
| §6.3 改写（等长） | $0$ |
| §5 章末改写（等长） | $0$ |
| **合计** | **$-4$ 行** |

不增页。若还需回收，下一个可砍的是 §5.1（分阶段钳制）——它的结论"任一阶段都能复现"现在也可以由暴露账户解释（两个阶段都产生修正），可压缩约 6 行。

---

## 六、执行顺序建议

1. 先写 §5.6（正文最长的一块，也是其余一切的依据）
2. 再改 §5 章末与 §4.2 / §4.4 / §4.1 标题（删减，腾出版面）
3. 再改 §6.1 / 新 §6.2 / §6.3
4. 最后摘要、贡献、标题、Figure 3、附录
5. 中文镜像 `main_zh.tex` 同步
6. 编译检查页数与交叉引用（注意 §4.4 删除后是否有 `\ref` 指向它）