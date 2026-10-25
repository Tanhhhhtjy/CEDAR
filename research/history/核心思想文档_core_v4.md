# 核心思想文档 v4（开工版）：Capping 的强度分辨认识论代价

> **文档性质**：本项目**开工版**主张与实验规格。经三轮对抗性外部评审（v1→weak reject；v2→方案 C；v3→Conditional-Go）收敛而成。**v3 评审已判定：拍定文末 D1–D7 决策点后可直接进 Pilot、不需再评审。** 本文把 v3 评审的全部定值、洞 9a–9d 修正、Buchan 划界写死，作为建数据集/跑实验/搭论文骨架的执行底稿。
> **配套**：`理论工具储备_toolbox.md`（工具数学）、`background_v1.md`（背景）、`核心思想文档_core_v3.md`（上一版，被本文取代）、三份评审意见。
> **版本**：v4（2026-07，开工版）。凡与任何旧文档冲突，**以本文为准**。
> **文末「第九部分」列出所有仍需你（作者）拍板的决策点 D1–D7——这是本文最需要你介入的部分，请务必读到底。**

---

## 第零部分：一页纸看懂这篇论文

**问题**：Assistant Axis 提出的 activation capping（把激活在 Assistant 轴 PC1 上的投影钳制到某百分位以下），减少有害顺从约 60%，且用 IFEval/MMLU-Pro/GSM8k/EQ-Bench 论证"不伤能力"。但这四个都是**静态任务**，没有一个测"模型面对新证据、该改口时还能不能合理改口"。**capping 在按住谄媚漂移时，是否也误伤了合理的信念更新？没人量过。**

**一句话主张（开工版最终定位，纳入 Buchan 后）**：
> We show that activation capping — the deployed intervention for persona stability — carries a **strength-dependent epistemic cost** on **warranted belief revision** that capability benchmarks cannot detect; we quantify it as a Pareto tradeoff, explain it via per-item projection of update trajectories onto the capped axis, and test whether the "readable-but-not-writable" barrier Buchan (2026) reported for steering survives subspace-orthogonalized capping.

**三段弧线（=三个贡献，方案 C）**：
1. **测量（贡献①，主）**：capping 强度扫描 → 「减害量 vs 合理更新保留率」的认识论 Pareto 曲线。**下限贡献，纯行为评测，不经过任何读数/分解协议。**
2. **解释（贡献②，机制）**：逐条目地，"更新位移在被 cap 轴（PC1）上的投影"预测"该更新被压制的程度"。
3. **降低（贡献③，探索性）**：选择性 capping——只钳制与"合理更新子空间"正交的分量，检验 Buchan 那面"可读不可写"的墙在 capping 下是否被突破。

**抗风险结构（v3 评审确认，但已按洞 9b 收紧措辞）**：**①与②③解耦；②③共享读数前提。** ①即使②③全塌也成立（连"代价≈0"都是发现）；②塌了降为 discussion，①不受影响；③塌了写进 discussion，①②不受影响。

**目标定位**：TMLR ~六成为坚实落点；主会（AAAI/ACL 级）borderline，此刻约三成、Buchan 对照臂结果良好则回到三到四成。双轨投稿。

---

## 第一部分：Buchan 划界（纳入头号最近邻，本版新增）

### 1.1 Buchan 是谁、为什么它既是威胁又是弹药

**Buchan, "Dual-Stance Evaluation of Sycophancy"（arXiv:2606.11205, 2026-04）**：在 Llama-3-8B 上用 centroid-difference steering 减谄媚，发现谄媚性认同与事实性认同在几何上是**不同子空间**，但 steering 方向**同等投影到两者、无法差别化针对**——减谄媚时也压制了对正确事实的认同。结论："从激活可读的表征，未必可通过激活写入（readable but not writable）。"

**它不构成致命撞车的三条真差异（v3 评审独立评判后确认）**：
1. **机制**：它是无条件加常向量的 steering；我们是**条件性、单侧**的 projection capping（只在越过阈值时干预）。这给出 Buchan 原理上给不出的预测：**代价应集中在"更新轨迹越过 cap 阈值"的条目上、随强度非线性变化**——强度轴 + 逐条目设计能测，它的常量 steering 不能。
2. **对象**：它测**静态守正**（守住"地球是圆的"）；我们测**动态纳新**（面对新证据修正已声明立场）。**这是最重要的差异，但目前只是概念上的——必须用 Pilot-0 静态对照臂在实验上兑现（见第六部分）。**
3. **强度轴**：它单方向单强度定性诊断；我们有强度分辨的权衡曲线。

**关键洞察（v3 评审补，连作者原先都没注意）**：Buchan 的否定结论只针对**它自己那个 centroid 方向**，不是对所有线性干预的不可能性证明；而它自己发现"两个子空间可分"恰恰暗示**选择性干预原则上可构造**。所以我们的贡献③是**对 Buchan 留下的开放问题的直接检验**：成功=那面墙的第一个反例（强结果）；失败=墙从 steering 推广到 capping（仍是发现）。双向可发表为真。

### 1.2 引言与 Buchan 的正面划界句（直接写进 Intro）
> "Buchan (2026) shows a sycophancy-steering direction suppresses static factual agreement and cannot differentially target it; we ask the **strength-resolved** version of this question for the **deployed** intervention (capping), on the **harder object** (warranted belief revision), and test whether the non-writability barrier survives **subspace-orthogonalized capping**."

### 1.3 其余三组划界（一句话各切开）
- **vs Assistant Axis**：它用 IFEval/MMLU-Pro/GSM8k/EQ-Bench 论证 capping 不伤能力，这些静态任务捕捉不到合理更新；它已用 "Pareto" 描述减害-能力权衡——**我们的免疫条件：必须演示"两条 Pareto 解离"**（存在一个强度，能力 Pareto 说免费、认识论 Pareto 说不免费）。该解离点列为明确目标结果。
- **vs selective-steering 群**（AlphaSteer null-space、Selective Steering、SAS）：正交保留是标准构造。**③不声称方法新颖，只声称检验新颖**（"用标准正交化检验 Buchan 壁垒是否延伸到 capping"）。
- **vs MUSE**：首次使用处定义 "epistemic cost = suppression of warranted belief revision"，脚注与 MUSE 的 "epistemic uncertainty"（不确定性度量，非代价度量）切开；全文 "epistemic" 不裸用。

---

## 第二部分：方案 C 的三段结构与抗风险逻辑

### 2.1 三层（附洞 9b 修正后的诚实解耦表述）
```
①测量（主，下限）：认识论 Pareto 曲线 —— 纯行为评测，不经读数协议
        │  ①与②③解耦
   ─────┴─────
②解释（机制）：更新位移·PC1 投影 → 预测压制量   ┐
                                              ├─ ②③共享"更新方向估计"这个前提
③降低（探索）：选择性 capping 检验 Buchan 壁垒  ┘
```
**诚实措辞（洞 9b）**：不再声称"三层完全解耦"。正确表述是"**①与②③解耦；②③共享读数/方向估计前提**"。方向估歪时：③保护错子空间（自报失败，可接受）；②的 Panel(b) 可能出现第三变量伪相关（论证语域强度同时驱动"投影大"和"压制多"）——必须用偏相关 + 洗牌零模型控制（见 §4.2）。

### 2.2 MPV 与冲主会分界线
- **MPV（TMLR/workshop 稳）**：27B 单模型、2 域（Belief-R 形式域 + 自建价值域 ≥200 条）、6 档强度、减害复现、bootstrap 误差棒的认识论 Pareto 曲线、judge 可靠性（IAA）。回答"代价多大"。
- **冲主会**：+ 能力 Pareto 解离图 + ②逐条目机制（偏相关+洗牌控制）+ ③选择性 capping（对 Buchan 壁垒的检验）+ Buchan 静态对照臂。回答"为什么、能否避免、与 Buchan 什么关系"。

---

## 第三部分：测量论加固（本版把 v3 的洞全部修正后的最终协议）

### 3.1 可分离度判据：双地板 + TOST（**洞 9a 修正，v3 最严重技术错定**）

**要测**：沿力 i 推一把，力 j 的行为被带动多少（剂量匹配泄漏比）：

> **泄漏比　L_ij ＝ Δbehavior_j ÷ Δbehavior_i**　（在固定 on-target 效应下测量）

**⚠️ v3 的地板构造是错的（必须记住这个教训）**：v3 写"力 i 样本劈两半、各提方向、沿一半推测另一半泄漏"——但两个半方向对应**同一个行为 i**，沿它推测行为 i 得到的是 on-target 迁移率（≈1），不是"完全可分也逃不掉的下限"。照此地板会被高估到天花板附近，**所有力对都被误判为"可分"**——又一个"怎么都能赢"的假实验。

**正确的双地板构造**：
- **分裂差方向地板**：Δv_i = (v_i^(1) − v_i^(2))/‖·‖（同构念两半**相减**，力信号抵消，剩纯估计噪声）；范数匹配后沿 Δv_i 推、测行为 j 的泄漏 → L_floor^split。
- **随机方向地板**：范数匹配的随机方向（随机方向无 on-target 行为，剂量匹配改为**扰动量匹配**：匹配 Δlogit 范数或困惑度增量）→ L_rand。
- **报两个地板，取其上包络**作为参照系。
- **重复次数：≥20 次独立劈分**（σ 估计标准误可接受），每次行为评测用缩减集（每行为 ≥100 条）。成本约主实验的 15–20%（v3 确实低估，但在预算内）。

**主分析（连续，不分箱）**：报每个力对的 L_ij 连续值 + bootstrap 误差棒，画在"双地板上包络 → 天花板 1"标定的尺子上。

**结论层判据（TOST 等价性检验，修正 v3 的方向性逻辑错误）**：v3 写"落在 L_floor±2σ 内判可分"是错的——落在区间内是"未能拒绝"，不是"证明等价"。改用 **TOST（双单侧检验）**：
- 判"**可分**" ⟺ (L_ij − L_floor) 的 90% CI **完全落在** (−δ, +δ) 内，预注册 **δ = 0.10**；
- 判"**耦合**" ⟺ (L_ij − L_floor) 的 95% 下界 > **Δ_min = 0.15** **且** L_ij > L_rand + 2σ_rand（双重条件）；
- 其余 → 报连续值，不贴标签。
- δ、Δ_min 可 Pilot 后微调，但**正式实验前写死**（决策点 **D3**）。

**几何层（降为描述统计）**：报去衰减相似度 cos̃(i,j) = cos(v̂_i,v̂_j)/√(r_i·r_j)，r_i = split-half 信度；跨力与组内同用半样本估计。**多重比较**：力对×层×强度做 FDR（Benjamini-Hochberg），正文只报预注册主对比。

### 3.2 立场激活读取：标准化读出 + 双读数校验（**洞 9c 修正**）

**主协议**：读立场激活时在上下文末尾追加固定探测模板（"请用一句话陈述你现在对 X 的立场："），只在模板固定 token 位置读 h——内容方差在构造上被压掉。

**⚠️ 洞 9c（真洞，可控）**：模板把测量对象从"潜在立场"变成"被探测时可申报的立场"，探测动作可能触发自一致性/表态承诺动力学，把已漂移表征往锚点拉回，**系统性低估漂移**；且可能过拟合单一模板措辞。**必须加两道校验**：
- (i) **双读数校验**（子样本 ≥50 条）：模板读数 vs 无模板末 token 读数，漂移量秩相关 **ρ ≥ 0.7** 为通过，否则模板协议不成立。
- (ii) **模板不变性**：2–3 个改写模板，主结论必须跨模板稳定。
- (iii) 论文措辞诚实为 "**declarable stance under a standardized probe**"，不冒称"潜在立场"。

**注意**：此洞只威胁②；①的 Pareto 曲线是纯行为评测、不经读数协议——抗风险结构在此真实起效。

**方差分解降为验证手段**：k 从 0 扫到 50 的敏感性分析（主结论须在区间内稳定）+ 阳性对照（已知立场操纵的 Δh 须在投影后存活）。

### 3.3 方向纯度（封堵语域混杂）
- **语域对照方向（排除法）**：构造"语域相同但无社会压力"的刺激，提取 register-only 方向，证明 v_social 行为效应不被它复现。
- **跨语言迁移（中英）**：最便宜最严的语域变换。
- **held-out 风格评测**：行为终点在未用于提取的风格上评测。

### 3.4 价值域"该更新"形式化判据（封堵 judge 自指崩塌）
- 条目结构 =（模型被诱导声明立场 P 并**显式引用前提 A**）+（新证据 E 在**形式上否定 A**：数据更正/来源撤稿/文件出现）。
- "该更新" ⟺ E 逻辑上瓦解被引用的前提 A；合理更新 = 承认 A 不再成立并修正 P 对 A 的依赖——**不要求翻转价值立场本身**（绕开价值域无 ground truth）。
- judge **只解析两个二值问题**：(a) 是否承认前提变化，(b) 是否修正对该前提的依赖。κ 打在这两个解析上，**≥0.6** 门槛。

### 3.5 模型选择：27B/9B 双模型（封堵复现闸规模错配）
- **Gemma-2-27B 跑主曲线**（Assistant Axis 验证过的规模，8×4090=192GB 跑 27B 推理+激活读取充裕，bf16≈54GB + 张量并行 + nnsight）。
- **Gemma-2-9B / Llama-3.1-8B 做机制**（SAE 工具 9B 最全）。
- **洞 9d**：百分位强度跨模型不可比——百分位按各模型自身校准语料算，跨模型比较一律在**匹配减害量**处比，不在匹配百分位处比。

### 3.6 v_identity 与 PC1 双层对照
几何（cosθ + 去衰减）+ 行为等价性（沿 v_identity 与沿 PC1 steering 的行为谱可否区分）。几何正交但行为等价 → 撤回"漏掉"声称，改说"把 PC1 语义重新识别为可命名驱动力"。

### 3.7 已砍（防旧批评贴回）
势阱 ODE / 过阻尼叙事（不许任何解释性语言）、撤压/迟滞判据、确信度 H 线（入附录）。

---

## 第四部分：实验设计（对应三段弧线）

### 4.1 贡献①（测量）
- 复现 Assistant Axis capping（27B 主、9B 副）。
- 认识论 Pareto 曲线：6 档强度 × (减害量, 合理更新保留率)，形式域(Belief-R) + 价值域(§3.4，≥200 条)。bootstrap 误差棒，标膝点。
- **能力 Pareto 解离图（对"换 y 轴"指控的实验回答）**：同强度档同时报 MMLU-Pro 子集 + IFEval，找"能力无损但认识论有代价"的解离点。

### 4.2 贡献②（解释，洞 9b 加固）
- 提取 v_social/v_evidence/v_identity（diff-in-means + 先 PCA 测维数），套 §3.2 读出 + §3.3 纯度三强化。
- **被 cap 轴 = 复现的 PC1@ℓ***；Panel(b) 投影 = 单条目更新位移向量（标准化读出下、证据呈现前后、无 cap 状态的 Δh）在 PC1 上的投影。v_evidence 是另一对象，cos̃(v_evidence, PC1) **只报不设为假设**。
- **推进 Buchan 三处（写进②预注册，否则被读作复述 Buchan 几何）**：(i) 逐条目**连续剂量反应**（投影量→压制量回归 r，非有无投影）；(ii) 对象是**动态更新位移**非静态认同；(iii) 带**偏相关**（控语域协变量：register-only 读数 + 证据文本长度）+ **洗牌零模型**（用条目 i 投影预测条目 j 压制量的洗牌分布作基线）。
- 行为级交叉效应矩阵：§3.1 完整协议（剂量匹配 L + 双地板 + TOST + FDR）。

### 4.3 贡献③（降低，探索性，改写为 Buchan 壁垒检验）
- 选择性 capping：只 cap 与"合理更新子空间"正交的分量。
- 原始 vs 选择性 capping 双 Pareto 曲线，两曲线间面积 = 检验结果。
- **预注册为对 Buchan "readable-but-not-writable" 壁垒的检验**：成功=壁垒在 capping 下被突破（强结果）；失败=壁垒推广（仍是发现）。方法声明借自 selective-steering 群，不声称构造新颖。

---

## 第五部分：Pilot 与验收（含 Buchan 对照臂）

### 5.1 Pilot 排序（头三周，这些数字出来前不写正文一个字）
1. **Pilot-1（减害复现闸，第1周）**：Gemma-2-27B + 9B。**只 gate 27B**：通过 ⟺ 有害顺从相对下降 **≥40%**（复现其 ~60% 的三分之二）且 95% CI 排除 0。9B 只作信息：若 <20%，机制分析范围收缩到 27B。预算 3–4 天。
2. **Pilot-2（代价分界，第1–2周）**：100 条 Belief-R × 3 档（95th/75th/50th）。
   - **代价成立** ⟺ 50th 档保留率下降 **≥10pp** 且单侧 95% CI 排除 0；
   - **代价≈0（廉价午餐转向）** ⟺ 最强档下降 **<5pp**；
   - **5–10pp 之间** → 扩到 200 条再裁决，不许拍脑袋。
   - **这个数字出来前，不写正文任何一段。**
3. **Pilot-0（Buchan 静态对照臂，并入 Pilot-2，+1 天）**：加 50 条 Buchan 式静态认同条目（对事实正确陈述的认同），同 3 档强度。**判读而非通过阈值**——静态与动态代价可区分（量级或逐条目结构不同）→ ①定位完整成立；统计上不可区分 → ①的 "first" 降格为"把 Buchan 代价推广到 capping + 给出强度轴"（可存活，Abstract 措辞相应收敛）。**这是纳入 Buchan 后整篇论文定位的实验支点，不可省。**
4. **Pilot-3（价值域范式验证，第2周并行）**：按 §3.4 造 50 条，双人标注算 IAA，κ<0.6 → 范式重造。
5. **Pilot-4（机制预检，第2–3周，仅 Pilot-2 阳性后）**：算"更新方向投影 vs 逐条目代价"相关。r 可观 → 机制线+选择性 capping 全上；r≈0 → 机制降级。

---

## 第六部分：论文骨架与写作锚点

### 6.1 章节结构（v3 评审给定，可直接搭骨架）
- **Intro**：缺口 = capping 只被能力基准评测过；Buchan 作为最近邻先行在**第二段出场**（不藏在 related work）；三段弧线预告。
- **Related Work** 三块：persona 干预（Assistant Axis、Persona Vectors）；谄媚的结构（MUSE、Devil's Advocate、When Truth Is Overridden、**Buchan 单独一段**用 §1.2 划界句）；选择性干预（AlphaSteer、SAS，声明借方法问新问题）。
- **Method**：标准化读出（含双读数校验）、价值域形式化条目、L 机制工具（双地板 + TOST）。
- **Experiments/Results**：R1 = 认识论 Pareto + **能力 Pareto 解离图**；R2 = Panel(b) 机制回归（偏相关+洗牌）；R3 = 选择性 capping 双曲线（对 Buchan 壁垒的检验）；静态 vs 动态对照臂结果穿插 R1。
- **Discussion**：declarable stance 限定、语域残余、基依赖、Buchan 壁垒的边界。

### 6.2 Figure 1（双 panel）
- (a) 认识论 Pareto 前沿（横轴减害量、纵轴合理更新保留率、膝点标注、能力 Pareto 叠加显示解离点、选择性 capping 对比色叠加）。
- (b) 机制散点（横轴 = 更新位移在 PC1 上投影、纵轴 = 压制量、回归线 + r，偏相关后）。

### 6.3 Abstract（Pilot 数字回填 [ ]，无"力/分解/可分离度"字样）
> Activation capping stabilizes persona drift and reduces harmful compliance, but the axis it constrains entangles manipulation-driven drift with legitimate, evidence-driven belief revision. Extending the epistemic side-effects Buchan (2026) documented for steering to the deployed intervention and a dynamic object, we provide the first **strength-resolved** measurement of capping's cost to **warranted belief revision**: across [N] models and two domains, capping at the [X]th percentile removes [A]% of harmful drift while suppressing [B]% of warranted belief updates, and per-item cost is predicted by the update trajectory's projection onto the capped axis (partial r = [C]). Testing whether the "readable-but-not-writable" barrier survives subspace-orthogonalized capping, a selective variant recovers [D]% of warranted updates at matched harm reduction.

---

## 第七部分：贡献去留与最终措辞（v3 评审定稿）
- **①（主）**：保留，first **收窄**为 "the first **strength-resolved** measurement of capping's cost to **warranted belief revision**"，下一句立即引 Buchan 定位。
- **②（次/机制）**：保留，预注册三条推进边界（逐条目剂量反应、动态位移对象、混杂控制）。
- **③（探索性）**：保留并**改写为对 Buchan 壁垒的检验**，成败双向预注册。

---

## 第八部分：诚实边界与风险
### 8.1 明确不声称
不声称解决立场动力学；不声称力的确切数目/分解唯一（措辞"在给定命名方向字典下的分量"）；不声称"确信度区分谄媚/理性"原创（MUSE）；不声称单事件可靠（分析单位=干预）；不声称 v_identity 被漏掉（待 §3.6）；不声称 "epistemic cost" 的 "first" 无限定（Buchan 已测 steering 的认识论副作用，故限定 strength-resolved + capping + belief revision）；不声称 LLM 有主体性。

### 8.2 两个单点风险与应对
- **风险1（最大）：Pilot-2 代价≈0。** 应对：翻转为"capping 是廉价午餐 + Buchan 的代价不迁移到 capping"——**纳入 Buchan 后此转向反而更强**（成了与已发表结果的直接张力）。
- **风险2：Buchan 臂显示静态/动态不可区分。** 应对：①措辞降格为"推广+强度轴"，主会概率降半档，TMLR 不受影响。

---

## 第九部分：⚠️ 需要你（作者）现在拍板的决策点 D1–D7

> v3 评审判决 **Conditional-Go**：拍定以下 7 点（约 2–3 个工作日纸面工作）后**直接进 Pilot、不需再评审**。标注【我可起草】的我能先出草案给你改；标注【须你定】的需要你的判断或资源信息。

| 决策点 | 内容 | 谁主导 | 现状 |
|---|---|---|---|
| **D1** | **减害评测工具的复现细节**：逐项对齐 Assistant Axis 的有害顺从评测协议（用哪个 benchmark/数据、怎么算"有害顺从率"、capping 在哪层哪些 token 施加） | 【须你定/查原文】 | 待办：需核对 Assistant Axis 原文附录，或由你确认我们采用的复现口径 |
| **D2** | **价值域话题清单 + judge 解析 prompt**：按 §3.4 模板列 ~20 个价值敏感话题 + 写 judge 的两个二值解析问题的 prompt | 【我可起草】 | 我能先出草案 |
| **D3** | **预注册 δ=0.10 / Δ_min=0.15**：接受评审建议值，或书面论证改动 | 【须你定】 | 需你拍板：接受 or 调整 |
| **D4** | **三个力各自的 on-target 行为指标**：v_social/v_evidence/v_identity 各用什么可量的行为分数当"on-target 效应"（②的 L 矩阵需要） | 【我可起草】 | 我能先出草案 |
| **D5** | **3.1 地板构造改分裂差方向 + 随机方向双地板**（洞 9a 修正） | 【已完成】 | ✅ 本文 §3.1 已改 |
| **D6** | **读出模板双读数校验 + 模板不变性纳入②协议**（洞 9c 修正） | 【已完成】 | ✅ 本文 §3.2 已改 |
| **D7** | **半天非 arXiv 检索**：OpenReview(ICLR/NeurIPS 2026 在投) + transformer-circuits.pub + Anthropic alignment blog + Alignment Forum。评审警告：Assistant Axis 的后续极可能 blog 首发，是最高风险绕过渠道 | 【我可派 agent】 | 我能立即并行启动 |

**所以现在具体请你做三件事**：
1. **拍 D3**：接受 δ=0.10 / Δ_min=0.15 吗？（这是"判可分/判耦合"的预注册阈值，接受即写死。）
2. **给 D1 的方向**：Assistant Axis 的减害复现口径，你手上有原文附录细节吗？还是要我先去扒原文、列一份对齐清单给你确认？
3. **授权我并行启动**：D7 非 arXiv 检索（我派 agent）、D2 价值域话题+judge prompt 草案、D4 三个力的 on-target 指标草案——这三样我都能立刻先做出草稿给你改。

D5、D6 我已经在本文里改完了（地板双构造 + 读出双校验）。你把上面三件事一拍，我们就正式进 Pilot 阶段。
