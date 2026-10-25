# CEDAR 图表数据包 — Figure 3（机制诊断三栏图）+ Figure 2D（跨模型迁移）

打包日期：2026-07-29 · 全部数值来自已冻结产物，脚本内不做任何重新分析

---

## 0. 一分钟版

| 图 | 回答的问题 | 一句话结论 |
|---|---|---|
| **Fig 3A** 固定轨迹 forest | 位移需要模型"重新想一遍"吗？ | 不需要。可见推理 token 完全冻结时，答案几何仍然移动 $+2.08$ logits |
| **Fig 3B** 轴外传播逐层图 | 钳制的效应还留在它自己的坐标里吗？ | 不留。整流停止后效应继续离开出口坐标，$G_{\mathrm{off}}=0.161$，12/12 family 同向 |
| **Fig 3C** 探针 AUC | 理由有效性的信息被抹掉了吗？ | 没有。钳制前后 layer-60 状态对 $V/F$ 的线性可解码性几乎相同（0.769 / 0.767），纯文本基线在随机水平 |
| **Fig 2D** 跨模型 forest | 换一个模型和它自己的配置，还会这样吗？ | 会。Qwen $+2.07$、Llama $+1.39$ 的广泛位移，两者的 $V-F$ 选择性变化都贴近 0 |

**这四张图连起来就是论文的机制主线：**
效应不经由可见推理 → 效应离开了控制器的坐标 → 而理由信息其实还在 → 换模型仍然复现。
由此得到我们提出的假说：**表征—策略解耦**（representational–policy decoupling）——钳制改变的很可能不是"模型知道什么"，而是"它知道的东西如何抵达最终决策"。

---

## 1. 目录结构

```
CEDAR_FIG_PACKAGE/
├── README.md                       ← 本文件
├── data/
│   ├── rg1p/                       RG1-P Stage A（机制传播实验）
│   │   ├── RG1_SUMMARY.json        38 KB，主结果 + 逐 family + 随机参考 + 探针
│   │   ├── propagation_metrics.jsonl  10,944 行，逐 (记录 × 层) 的轴向分解
│   │   ├── pca_points.jsonl        96 行，layer-60 状态的 PCA 前两维
│   │   └── ANALYSIS.md             官方一页摘要
│   ├── frozen/
│   │   ├── figures_provenance.json 已冻结的 Fig 2 / Fig 3 全部数值
│   │   └── E2B_ANALYSIS.json       固定轨迹重放（E2B）完整分析
│   └── cross_model/
│       ├── figure2d_forest_input.csv   Fig 2D 的 4 个点（直接输入）
│       ├── cross_model_summary.csv     两模型全部估计量
│       ├── cross_model_family_effects.csv  逐 family
│       ├── cross_model_target_follow.csv   离散 target-following
│       └── delivery_manifest.json      交付清单与 gate 状态
├── scripts/
│   ├── make_fig3_mechanism.py      生成 Fig 3（三栏）
│   └── make_fig2d_cross_model.py   生成 Fig 2D
└── figures/
    ├── fig3_mechanism.pdf / .png
    └── fig2d_cross_model.pdf / .png
```

**复现：**
```bash
cd scripts
python make_fig3_mechanism.py     # -> ../figures/fig3_mechanism.pdf
python make_fig2d_cross_model.py  # -> ../figures/fig2d_cross_model.pdf
```
只需 `numpy` + `matplotlib`。脚本使用相对路径，整个文件夹可以整体移动。

---

## 2. 数据字典

### 2.1 `data/rg1p/RG1_SUMMARY.json`

| 键 | 内容 |
|---|---|
| `primary_g_off_layer60_minus_layer53` | 主结果：估计值、95% family-cluster bootstrap CI、**逐 family 值**、**留一 family 估计**、同号 family 数 |
| `off_direction_share_by_layer_46_63` | 官方 p25 + 4 个随机参考，逐层轴外能量占比（含 CI） |
| `arm_g_off` | $N$/$R$/$V$/$F$ 四臂各自的 $G_{\mathrm{off}}$ |
| `delta_m_general_by_arm_and_layer_45_63` | 逐臂逐层的答案 margin 位移 |
| `matched_random_direction_references` | 4 个预注册随机方向的 $G_{\mathrm{off}}$、margin 位移，以及 p25 减随机的差 |
| `primary_delta_m_general` | 本环境内复现的广泛位移：$1.339$，CI $[0.911, 1.754]$ |
| `vf_probe_selected` | **辅助探针**：uncapped / p25 / 纯文本 TF-IDF 三种读出的 balanced accuracy、ROC-AUC、family 标签置换 $p$ 值、逐 family 表现 |
| `descriptive_family_association` | $G_{\mathrm{off}}$ 与 margin 位移的 Spearman $\rho = -0.175$（12 个 family） |
| `route_gate` | 预注册路由判定：Route A |
| `claim_contract` | **允许 / 禁止的主张清单**（见 §5） |
| `bootstrap` | 20,000 次重抽，聚类单位 = family，种子 20260727 |
| `formal_audit_status` | `passed`（864/864 条记录） |

### 2.2 `data/rg1p/propagation_metrics.jsonl`

10,944 行 = （96 条官方记录 + 384 条随机参考记录）× 19 层（45–63）。

关键字段：

- `off_direction_share` / `aligned_share`：位移在固定 layer-53 出口方向**之外 / 之上**的能量占比（两者和为 1）
- `delta_norm`：uncapped 与 p25 隐状态之差的范数
- `signed_aligned_projection`：沿出口方向的带符号投影
- `cell_kind`：`official` 或 `random_reference`
- `trace_source`：`e2a_uncapped`（U 源轨迹）或 `e2a_full_p25`（P 源轨迹）
- `family_id` / `base_id` / `arm`：聚类与分臂标识

**Fig 3B 直接使用**：`cell_kind == official` 且 `trace_source == e2a_uncapped` 的 96 条曲线。

### 2.3 `data/frozen/figures_provenance.json`

`figure3.panel_a` / `panel_b` 提供 Fig 3A 的四个点：

| 估计量 | 值 | 95% CI |
|---|---|---|
| U 源轨迹广泛位移 $\Delta m_{\mathrm{general}}$ | $+2.078$ | $[1.759,\ 2.408]$ |
| P 源轨迹广泛位移 | $+0.700$ | $[0.487,\ 0.907]$ |
| U 源轨迹 $V-F$ 选择性（$m_{\mathrm{target}}$） | $-2.422$ | $[-3.411,\ -1.448]$ |
| P 源轨迹 $V-F$ 选择性 | $-0.797$ | $[-1.219,\ -0.391]$ |

### 2.4 `data/cross_model/figure2d_forest_input.csv`

| model | estimand | estimate | CI |
|---|---|---:|---|
| Qwen-FC | overall_delta_m_general | $+2.075$ | $[1.372,\ 2.791]$ |
| Qwen-FC | v_minus_f_continuous_selectivity_change | $+0.042$ | $[-0.438,\ 0.573]$ |
| Llama | overall_delta_m_general | $+1.391$ | $[0.908,\ 1.865]$ |
| Llama | v_minus_f_continuous_selectivity_change | $+0.115$ | $[-0.365,\ 0.615]$ |

每个模型：12 families / 24 bases / 192 条记录（选择性对比 96 条）。

---

## 3. 每张图是怎么画的、表示什么

### Fig 3A — “Frozen trace, moved answer”（固定轨迹，答案仍然移动）

**画法**：横向 forest。上组（蓝）是广泛位移 $\Delta m_{\mathrm{general}}$，下组（红）是 $V-F$ 选择性对比；实心圆 = U 源轨迹，空心方 = P 源轨迹；须线为 12 个 family 的 cluster bootstrap 95% CI；虚线为零。

**实验**：对 96 个 prompt，先分别冻结 uncapped 与 p25 生成的推理前缀，然后在**每一个可见 token 完全相同**的条件下，用 uncapped 与 p25 两种前向通路重放，并在答案边界读取四个选项的 logits。

**意义**：这是全篇最强的一次排除。如果钳制是通过"让模型想出不一样的东西"来改变答案的，那么把想的内容钉死之后效应就该消失。它没有消失（$+2.08$）。**因此，位移发生在前向计算里，而不是发生在推理文本里。** 同时 192 次配对重放没有任何一次越过四选一的 argmax 边界——连续几何已经移动，离散选择却没有动，这正是我们命名的 *margin–choice 分离*。

### Fig 3B — “The effect leaves the cap's coordinate”（效应离开了钳制的坐标）

**画法**：横轴第 53→63 层，纵轴是**相对于出口层的轴外能量增量**（每条曲线都从 0 起画，避免把跨层轴不对齐误读成泄漏）。96 条淡蓝细线是逐记录轨迹，粗蓝线是均值，灰虚线是 4 个调度与范数均匹配的随机方向参考，标注点在预注册的第 60 层。

**实验**：钳制只在第 46–53 层沿 Assistant Axis 削去超阈投影。我们把 uncapped 与 p25 的配对隐状态之差，沿固定的 layer-53 出口方向分解，预注册指标为 $G_{\mathrm{off}} = \text{OffShare}_{60} - \text{OffShare}_{53}$。

**结果**：$0.762 \to 0.922$，即 $G_{\mathrm{off}} = 0.161$，CI $[0.151, 0.170]$；12/12 family 同号；任意留一 family 都落在 $[0.158, 0.163]$；四臂一致（$N\ 0.145$、$R\ 0.142$、$V\ 0.184$、$F\ 0.172$）。

**意义**：控制器是按"一个方向"定义的，但到答案被决定的那一层，它的效应已经不再由那个方向概括。**这不是"steering 是非线性的"这类一般性论断的重复**——我们测的是一个条件触发的单侧钳制，在可见文本被冻结的条件下，其**自身出口坐标**在整流停止之后仍在继续失去对效应的解释力。这正是任何"沿 $u$ 施加、也沿 $u$ 推理其后果"的控制器设计会失灵的地方，也是我们提出 L1/L2 受保护子空间控制器的直接理由。

> ⚠️ 必须保留的限定：4 个随机参考在出口层就已接近轴外占比上限，因此没有继续增长的空间。该比较识别的是**传播形态不同**，不能写成"只有 Assistant Axis 会被重混"。

### Fig 3C — “Validity stays readable”（有效性依然可读）

**画法**：三根棒棒糖，从 chance = 0.5 起画到各自的 ROC-AUC。

**实验**：family 留出的线性探针，从第 60 层状态预测该条目属于 $V$（有效理由）还是 $F$（单点致命谬误）。

**结果**：uncapped 0.769、p25 0.767（family 标签置换 $p = 0.014$ / $0.002$，4,096 次置换），纯文本 TF-IDF 基线 0.511。

**意义**：钳制之后，"这条理由是否成立"的信息**仍然在隐状态里**，而且不是靠词面捷径读出来的。可是行为层面 $V$ 与 $F$ 的区分并没有随之改善。**信息还在，却没有兑现为决策**——这就是我们把解释从"证据被抹除"转向"表征到决策的路径被改变"的关键证据。

> ⚠️ 限定：两个条件的探针是**分别训练**的，因此这确立的是各条件下的可解码性，不能直接说"两者共享同一套编码"，也不能说"模型明知故犯"。跨条件探针迁移与 activation patching 才能推进到那一步。

### Fig 2D — “Matched cross-model transport”（跨模型匹配迁移）

**画法**：forest，两模型各两行；实心圆 = 广泛位移，空心菱形 = $V-F$ 选择性变化。

**实验**：同一套 Evidence bank，在 reasoning 关闭的确定性四选一协议下，各模型使用**自己**公开的轴与 p25 配置。

**意义**：广泛位移在两个模型上都稳健为正（$+2.07$ / $+1.39$），而选择性变化都紧贴零（$+0.04$ / $+0.11$，区间跨零）。**"移动答案几何"迁移了，"改善理由选择性"没有迁移。** 这把结论从"某一次配置的偶然"提升为"这一类算子的共同性质"。

> ⚠️ 限定：模型身份与钳制配置是**同时变化**的，因此结论适用于两个"模型—配置对"，不能分离出模型本身的作用。

---

## 4. 设计与制图规范（如需再改图请沿用）

- 尺寸：Fig 3 为 $7.0 \times 2.45$ in（AAAI 双栏满宽）；Fig 2D 为 $3.35 \times 1.85$ in（单栏）
- 字体全部嵌入（`pdf.fonttype = 42`），无 Type 3 字体
- **不靠颜色单独区分**：填充/空心标记 + 标记形状 + 线型都各自可区分，灰度打印仍可读
- 配色：p25 蓝 `#2B6CB0`、$V$ 绿 `#0E8C7A`、$F$ 橙红 `#DC542A`、灰 `#98A2AE`、墨 `#16202b`
- 所有须线一律是 **12 个 family 的 cluster bootstrap 95% CI**（family 是独立抽样单位，不是条目）
- 脚本内**不做任何统计**：只读冻结产物再排版，避免图与正文数字漂移

---

## 5. 允许与禁止的主张（来自 `RG1_SUMMARY.json` 的 `claim_contract`）

**可以说：**
固定轨迹表征审计 · 出口轴对齐分量与轴外分量 · 超过四个预注册的、调度与范数匹配的随机方向参考

**不可以说：**
因果中介 · 完整神经回路 · 逐层的下游 Assistant Axis · 对所有随机扰动均唯一 · 仅凭 Stage A 就证明自然生成行为

另外两条自加限定：
1. 第 53 层的 $0.762$ 是过程量，不是独立的"泄漏率"（窗口内用逐层轴、下游用固定方向）；
2. $\rho = -0.175$ 表示**未检测到** $G_{\mathrm{off}}$ 与 margin 位移的共变；12 个聚类检验力很低，这既不是"无关联"的证明，也不构成中介证据——它的正面含义是：行为后果并不是源轴坐标上的剂量反应，这与"效应已经不由该坐标承载"相一致。

---

## 6. 未进入正文、可放补充材料的图

- RG1 四面板 demo（`outputs/fig2_demo/fig_rg1_demo.png`）中的逐臂逐层 margin 曲线
- layer-60 PCA 散点（`data/rg1p/pca_points.jsonl`，96 点）——只能作 descriptive，不能当可解码性的证据
- 12 个 family 的 $G_{\mathrm{off}}$ 条形图——正文 caption 写"12/12 同向"即可
