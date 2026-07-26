# CEDAR RG1 表示几何实验委托 Prompt

> 以下内容可整体转发给负责实验的同学或实验 Agent。
>
> 日期：2026-07-27
>
> 任务性质：新增机制诊断；不得覆盖或重跑冻结实验。

---

你现在负责 CEDAR 项目的一个新增机制诊断：

**RG1 — Prompt-End Representation Geometry under Assistant-Axis Capping**

目标是回答：在同一问题和目标答案固定时，模型是否在内部表示中区分有效理由（V）与错误论证（F）；Assistant-Axis p25 capping 是否破坏这种区分，还是主要对不同理由状态施加相似的下游位移。

请严格执行下面的科学、工程和汇报要求。不要把预期故事当作结果，不要编数值，不要为了得到漂亮点云筛选样本、层、token 或随机种子。

## 1. 已验证背景

### 1.1 冻结锚点

- E2A 结果锚点：`0c8517bb5fb4640f7d5e4d51356d0cf167046643`
- 锚点提交信息：`results(m2): add E2A phase-localization assay`
- 模型：`Qwen/Qwen3-32B`
- 模型 revision：`9216db5781bf21249d130ec9da846c4624c16137`
- dtype：`bfloat16`
- attention implementation：`sdpa`
- 官方 p25 配置：`layers_46:54-p0.25`，实际 decoder layers 46–53
- E2A 输入：24 bases、12 families、N/R/V/F 共 96 个 Answer prompts
- 权威输入：`datasets/m2_2b_formal/e2_phase_validity_choice/answer_inputs96.jsonl`
- 权威协议：`docs/E2_PHASE_VALIDITY_CHOICE_PROTOCOL.md`
- 权威数据 manifest：`datasets/m2_2b_formal/e2_phase_validity_choice/manifest.json`

必须按 commit SHA 锚定。不要假设同名远端分支仍指向该提交。

### 1.2 已有结果边界

- E2A 已观察到 broad answer-geometry shift，但它是 exploratory mechanism assay，不是全新 held-out confirmation。
- E1 只有沿 Assistant Axis 的 token-resolved 标量投影，没有完整高维 hidden states；其简单 resistance-state 假说未通过预注册闸门。
- E2C 是同一任务语法和 family structure 下的新 salt 证据，当前 manifest 明确 `additional_runs_authorized: false`。未经新授权不得重跑或增加 E2C hidden-state 实验。
- E2B fixed-trace cross 因 replay engineering Gate 失败而停止，没有科学估计。RG1 不得被写成 E2B 的替代因果证据。
- 目前不存在可直接用于 PCA 的完整 hidden-state artifact，因此 RG1 是新实验，不是既有结果重绘。

## 2. 允许与禁止的目标

### 允许回答

1. 在预先固定的 token 位置和层，V/F 是否在表示空间中可分。
2. capping 后 V/F 可分性是否增强、保持或减弱。
3. capping 在下游层引起的表示位移，在 V/F 之间是否相似。
4. N/R 控制是否也出现相似位移，从而支持或反对 broad, reason-insensitive shift。
5. 表示诊断与 E2A 的连续 margin、离散选择和 validity readout 是否一致。

### 禁止回答

- 不得仅凭 PCA、probe 或 cosine 声称模型“理解了有效性”。
- 不得声称某表示“因果介导”行为；这需要双向 patching/rescue。
- 不得把即时 hook correction 的共同方向当作发现；该方向由 capping 算子机械规定。
- 不得把 96 prompts 或数千 token 当成 96/数千独立 family；统计单位是 12 families。
- 不得混入 AO/S 社会模块点云。Social 与 Evidence 来自不同任务分布，必须分开分析。
- 不得把 Qwen 机制图外推为 Llama、Gemma 或所有 activation axes 的机制。

## 3. Git 与工作区纪律

首先执行并保存输出：

```bash
git status --short --branch
git worktree list
git remote -v
git fetch origin
git cat-file -e 0c8517bb5fb4640f7d5e4d51356d0cf167046643^{commit}
```

在独立 worktree 创建新分支，例如：

```bash
git worktree add <sibling-worktree> \
  -b agent/rg1-prompt-end-representation-geometry \
  0c8517bb5fb4640f7d5e4d51356d0cf167046643
```

要求：

- 不修改 E0/E1/E2A/E2C/E2B 的原始 artifacts、analysis 或 manifests。
- 所有 RG1 文件写入新的命名空间。
- 正式 GPU 运行前，代码、协议和 CPU Gate 必须提交，worktree 必须 clean。
- 科学方向不符合预期不是重试理由。

## 4. 新文件与目录

建议使用：

```text
docs/RG1_PROMPT_END_REPRESENTATION_PROTOCOL.md
datasets/m2_2b_formal/rg1_prompt_end_geometry/
scripts/run_rg1_prompt_end_geometry.py
scripts/audit_rg1_prompt_end_geometry.py
scripts/analyze_rg1_prompt_end_geometry.py
tests/test_rg1_prompt_end_geometry.py
artifacts/m2/rg1_prompt_end_geometry/
```

必须生成：

```text
CPU_GATE.json
ANALYSIS_PLAN.json
manifest.json
smoke-audit.json
records.jsonl
hidden_states.safetensors 或 hidden_states.npz
telemetry.jsonl
analysis.json
analysis-report.md
plot_points.csv
plot_stats.json
figures/representation-geometry.pdf
figures/representation-geometry.svg
figures/representation-geometry.png
RESULT_SUMMARY_ZH.md
DELIVERY_MANIFEST.json
```

## 5. 实验矩阵

### 5.1 主矩阵

- 输入：E2A 的 96 个 N/R/V/F Answer prompts，byte-for-byte 复用。
- 条件：
  - `uncapped`
  - `p25_prefill_forward`
- 总计：`96 prompts × 2 conditions = 192 deterministic prompt forwards`
- batch size：1
- `model.eval()` + inference mode
- 不生成回答，不做 retry、repair、content filtering 或 behavior-based selection。

`p25_prefill_forward` 使用 E2A 完全相同的官方 axis、threshold、符号、hook path 与 layers 46–53。因为本实验只做 prompt forward，不含 decode，所以不要把它描述成完整生成轨迹上的 full-p25 效果。

### 5.2 为什么不加入另外两个 phase conditions

- `decode_only` 在纯 prompt forward 中应等价于 uncapped，没有新增科学信息。
- `prefill_only` 与此处的 `p25_prefill_forward` 是同一个阶段作用。
- 不为凑成 2×2 重复无信息运行。

### 5.3 Arms 的角色

- V/F：主分析，回答 reason validity geometry。
- N/R：控制分析，判断位移是否广泛存在。
- AO/S：本实验禁止混入。

## 6. Hidden-state 采集定义

### 6.1 Token 位置

必须在 CPU preflight 中通过 token-prefix/hash 确定，不得人工目测 index。

主位置：

1. `prompt_end`：完整 chat template 中最后一个 non-padding prompt token；这是进入生成前的 decision-ready state。

预注册敏感性位置：

2. `followup_end`：用户 follow-up 正文最后一个 token，位于 role/template closing tokens 之前。
3. `followup_mean`：follow-up 正文 token span 的 mean-pooled state。

主文结果不得在看图后从三个位置中择优。主位置固定为 `prompt_end`；另两个必须全部报告或全部放附录。

### 6.2 层

捕获以下 transformer block output：

- layer 45：cap span 前的负控制层；
- layers 46–53：全部 capped layers；
- layer 54：第一个 cap span 后层；
- `L_final = model.config.num_hidden_layers - 1`：最终 decoder layer。

主 PCA 层预先固定为 `L_final`。layer 54 是主要机制敏感性层，layer 45 是工程/负控制。不得按最大 separation 选择主层。

如果模型层编号或 hook module 与上述约定不一致，立即停止并提交 mapping 报告；不得静默平移编号。

### 6.3 Hook 点

必须在协议中精确定义：

- `block_output`：每层 transformer block 的 residual output；
- 对 capped layers，额外保存：
  - `cap_input`：rectification 前；
  - `cap_output`：rectification 后；
  - `immediate_correction = cap_output - cap_input`。

注意：`immediate_correction` 沿 cap vector 的方向是算子定义带来的 sanity check，不得作为“理由无关机制”的主证据。主证据使用 layer 54 与 final layer 的下游状态。

### 6.4 建议张量结构

使用二进制张量文件，不把高维向量塞入 JSON：

```text
uncapped_block_output : [96, K, P, hidden_size]
capped_block_output   : [96, K, P, hidden_size]
cap_input             : [96, 8, P, hidden_size]
cap_output            : [96, 8, P, hidden_size]
```

其中：

- `K` 是去重后的捕获层数；
- `P` 对应 `prompt_end`、`followup_end`、`followup_mean`；
- 分析前统一转 float32；原始计算保持源配置 BF16；
- `records.jsonl` 保存 row index、item/base/family、arm、condition、prompt hash、token hash、token boundaries、layer mapping 和 tensor row mapping。

## 7. CPU 与 smoke Gates

正式 GPU matrix 前必须通过：

1. 96 个 prompts 与 E2A manifest 的内容、顺序和 SHA-256 完全一致。
2. 模型 revision、axis asset、threshold、layer range 和 chat template hash 完全一致。
3. `nohook` 与 `monitor_only` 的 next-token logits 在 E2A 冻结容差内一致。
4. monitor hook 不修改任何 residual state。
5. layer 45 的 uncapped/capped state 在冻结数值容差内一致，因为此前尚未施加 cap。
6. 所有 token boundaries 可唯一重建；任何歧义均停止。
7. 所有张量 shape 正确、finite、无 NaN/Inf。
8. 每个 row 后 hook 清理为零；不得泄漏到下一条。
9. smoke 覆盖 N/R/V/F、两种 conditions、三个 token positions 和所有捕获层。
10. smoke 失败时保留 artifact 并停止，不放宽容差、不删 row、不换 seed。

CPU Gate 完成后先汇报，再获得正式 GPU 运行确认。

## 8. 预注册表示分析

### 8.1 去除题目内容主导

对每个 base `i`，先用 uncapped 四臂定义固定中心：

```text
b_i = mean_a h(i, a, uncapped),  a ∈ {N,R,V,F}
r(i,a,c) = h(i,a,c) - b_i
```

两个 conditions 必须减去同一个 uncapped base center，才能保留真实 cap displacement。不得分别对 uncapped/capped 各自重心化，否则会把共同位移人为消掉。

主分析使用 raw residual-centered vectors；L2-normalized 版本只能作为敏感性分析。

### 8.2 PCA

- 在 `L_final + prompt_end` 上，将 V/F 的 uncapped 与 capped residual vectors 合并后只拟合一次 PCA。
- 所有条件投影到同一 PCA basis。
- N/R 可用同一 basis 作为浅色控制 overlay 或附录，不重新拟合。
- 每个点代表一个 item state，不把每个 token 画成独立点。
- 同一 item/arm 的 uncapped 与 capped 用箭头连接。
- 68% covariance ellipse 只能标为 descriptive ellipse，不是 confidence interval。
- 报告 PC1/PC2 explained variance；二维图不能代替高维定量检验。

### 8.3 V/F 可分性

主定量分析：family-blocked linear decoding。

- 标签：V vs F。
- 分组：12 families；不得随机 row split。
- 使用 leave-one-family-out 或 GroupKFold。
- 所有标准化、PCA 降维和分类器拟合必须在 training fold 内完成，禁止数据泄漏。
- 模型建议：固定 L2 logistic regression；若需要降维，训练 fold 内最多 10 PCs，并在协议中固定。
- 报告 balanced accuracy、ROC-AUC、每 family 预测和 family-blocked uncertainty。
- 使用 family-level V/F label flip permutation；12 families 时可枚举全部 `2^12 = 4096` flips。
- 比较 uncapped 与 capped decodability，但不要把“不显著下降”偷换成严格等效，除非预先定义等效界。

### 8.4 Cap displacement

定义：

```text
Δh(i,a) = r(i,a,p25_prefill_forward) - r(i,a,uncapped)
```

在 layer 54 与 final layer 报告：

1. `cos(mean Δh_V, mean Δh_F)`；
2. V/F mean displacement norm；
3. `||mean Δh_V - mean Δh_F||`；
4. 每个 item displacement 对 pooled displacement direction 的 cosine；
5. N/R/V/F displacement norm 与 trigger/correction telemetry；
6. displacement 对 uncapped validity contrast 的 alignment。

另外报告尺度无关的 V/F 差异位移：

```text
D_rel = ||mean Δh_V - mean Δh_F|| /
        mean(||mean Δh_V||, ||mean Δh_F||)
```

若分母接近 0，必须同时报告原始 norm 并把 `D_rel` 标为不稳定，不得据此宣称“共同位移”。“相似”或 “broadly shared” 不能由差异不显著推出。正式 GPU 运行前，必须在 `ANALYSIS_PLAN.json` 中二选一：

1. 用与正式 hidden states 独立的依据冻结 `D_rel` 等效界，并用 family-cluster CI 做等效判断；或
2. 不作等效主张，只报告方向 cosine、`D_rel`、原始差异及其 CI，措辞降为 “directionally aligned”。

所有 CI 使用 10,000 次 family-cluster bootstrap，seed 固定并写入 manifest。不得按 token 或 row 独立 bootstrap。

### 8.5 层级曲线

对 layer 45、46–53、54、final 同时画：

- V/F decoding 或 centroid-separation diagnostic；
- mean cap displacement norm；
- V/F displacement difference；
- Assistant-Axis projection/correction telemetry。

主层仍由协议固定，layer curve 只用于定位，不用于事后挑层。

### 8.6 混杂与敏感性

必须检查：

- prompt/follow-up token length；
- base、family、variant、position transition；
- V/F 的表面 token 差异；
- raw vs base-residualized；
- prompt_end vs followup_end vs followup_mean；
- 是否由少数 family 主导（leave-one-family-out）。

禁止先按 yield/resist、answer flip 或 margin 方向筛选后再画主 PCA。行为分层只能作为明确标注的探索性附录。

## 9. 结果路由与对应故事

以下均为条件性模板；数据出来前不得写入结果段。

### Route A：V/F 可分性保留，V/F 下游 displacement 满足预注册共同位移标准

允许故事：

> Validity-related information remains linearly decodable at the selected prompt-end representation, while capping induces a broadly shared downstream displacement across valid and fallacious reasons. Together with the behavioral results, this is consistent with a representation axis acting as a non-selective control operator rather than a reason-sensitive policy.

中文：有效与错误理由的信息在选定表示中仍可线性读出，但 capping 对两类理由产生相似的下游位移；结合行为结果，这与“表示轴并不自动构成理由敏感的控制策略”一致。

只有第 8.4 节预先冻结的共同位移标准通过时，才允许使用 `similar` 或 `broadly shared`；否则必须给出连续估计并降级措辞。禁止升级为“决策策略已被因果定位”或“模型理解有效性但故意忽略”。

### Route B：uncapped 可分，capped 后明显减弱

允许故事：

> Capping is associated with reduced validity decodability at the selected representation, supporting representational degradation as one candidate source of epistemic non-selectivity.

仍不得声称该表示变化因果介导最终答案，除非补 patching/rescue。

### Route C：uncapped 本身不可稳定分离 V/F

允许故事：

> The selected representation readout does not establish stable linear encoding of reason validity.

不得写“模型没有有效性知识”。PCA 移附录，主文保留负结果一句。

### Route D：V/F displacement 明显不同

允许故事：

> The representational effect of capping is condition-dependent rather than a uniform translation; downstream boundary mapping or heterogeneous trigger magnitude must be examined.

此时放弃“共同位移”故事，不得只展示看起来相似的二维投影。

### Route E：结果高度依赖位置、层或少数 family

结论降级为 exploratory diagnostic，图移附录。不得挑最好看的位置作为主图。

## 10. 与整篇论文的连接

若 Route A 成立，RG1 服务于以下完整链条：

```text
V/F validity-related information remains available at prompt end
                         ↓
capping produces a similar downstream shift across reason types
                         ↓
answer geometry moves broadly rather than target-specifically
                         ↓
V and F behavior is not cleanly separated
                         ↓
representation is not policy
```

论文中最合适的位置是 Section 5 的 diagnostic evidence 或 Section 6.1 `Representation is not policy`。PCA 不是标题级主结果，除非定量分析、family-level uncertainty 和行为链条全部稳健。

## 11. 主图设计

### Panel (a)：Paired states in a shared PCA space

- V：圆形，teal。
- F：方形，orange-red。
- uncapped：空心。
- capped：实心。
- 同一样本 uncapped→capped 用细箭头连接。
- 叠加大号 centroid 与 68% descriptive covariance ellipse。
- 点必须是一 item 一点，不能是一 token 一点。

### Panel (b)：Cap-induced downstream displacement

- 画 `ΔPC1/ΔPC2` 的 V/F 分布与 mean arrows。
- 若 V/F 位移重叠，视觉上展示共同方向；若不重叠，必须如实画出。
- 不展示即时 cap hook correction 作为主面板，因为其方向由算子定义。

### Optional Panel (c)：Layer-resolved quantitative summary

- 横轴 layer；纵轴分别为 decoding、displacement norm 或 V/F displacement difference。
- 有 family-level CI。
- 若版面不足放附录，不压缩到低于 9 pt。

## 12. 绘图风格、字体与配色

采用固定 AAAI 双栏尺寸：

```text
width  = 6.975 in
height ≈ 3.55 in（可在 3.4–3.8 内调整）
```

配色：

```text
Valid / V          #0E8C7A   teal
Fallacy / F        #DC542A   orange-red
Primary text       #1F2E40   navy-charcoal
Secondary text     #5E6A78   slate
Grid               #E1E5E9   light gray
Neutral controls   #9AA3AD   gray
```

编码必须冗余：

- V/F 同时用颜色和 circle/square 区分；
- uncapped/capped 同时用 hollow/filled 和 dashed/solid ellipse 区分；
- 灰度下仍可读；
- 不使用 rainbow、jet、3D、渐变背景、阴影或装饰性图标。

字体：

- Arial；若环境无 Arial，统一使用 Liberation Sans，不能混用；
- 主标题 12 pt；panel title 10 pt；其余文字、刻度、legend 不低于 9 pt；
- 线宽 0.8–1.3 pt；数据点约 20–28 pt²；centroid 60–80 pt²；
- 网格细且浅，不得压过数据。

## 13. 绘图脚本要求

分析脚本必须先生成不可变中间文件 `plot_points.csv`，至少包含：

```text
item_id,family_id,base_id,arm,condition,layer,position,
pc1,pc2,delta_pc1,delta_pc2,is_primary
```

同时生成 `plot_stats.json`，包含 PCA basis hash、explained variance、group centroids、ellipse 定义、bootstrap seed 和数据 manifest hash。

正式 MATLAB renderer：

```text
figures/scripts/plot_rg1_representation_geometry.m
```

最低实现骨架：

```matlab
T = readtable('plot_points.csv');
C.valid   = [14 140 122] / 255;
C.fallacy = [220 84 42] / 255;
C.text    = [31 46 64] / 255;
C.grid    = [225 229 233] / 255;

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
fig = figure('Units','inches','Position',[0 0 6.975 3.55], ...
             'Color','w','Renderer','painters');

% Panel A: for each item/arm, connect uncapped and capped in the same PCA basis.
% V = circle, F = square; uncapped = hollow, capped = filled.
% Add group centroids and descriptive covariance ellipses.

% Panel B: draw paired delta coordinates and mean displacement arrows.

exportgraphics(fig,'representation-geometry.pdf','ContentType','vector');
exportgraphics(fig,'representation-geometry.svg','ContentType','vector');
exportgraphics(fig,'representation-geometry.png','Resolution',300);
```

需要补齐 reusable functions：

- `draw_paired_arrows`
- `draw_covariance_ellipse`
- `style_axes`
- `validate_required_groups`
- `assert_real_artifact`

`assert_real_artifact` 必须读取 manifest 并拒绝：

- `simulated=true`；
- 缺失 formal audit；
- 有过滤 row；
- PCA basis hash 与 `plot_stats.json` 不一致。

可以同时提供 Python renderer 方便 CI，但 MATLAB PDF/SVG 为正式版本。Python 设置至少包括：

```python
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
```

Inkscape 只允许：

- 调整标题、legend、annotation 的位置；
- 检查裁切、留白、字体与对象分组；
- 导出最终 PDF。

Inkscape 不允许移动数据点、箭头、centroid、ellipse 或修改统计数字。

## 14. 图形验收

必须完成：

1. 以最终 6.975-inch 尺寸检查，无缩放后小字。
2. 无文字、legend、panel、footer 重叠。
3. 生成灰度预览，V/F 与 cap 状态仍可辨。
4. `pdffonts representation-geometry.pdf`：所有字体 `emb=yes`，无 Type 3。
5. SVG 中文字仍可编辑，点和箭头是独立 vector objects。
6. PDF/SVG/PNG 三者视觉一致。
7. caption 明确统计单位是 family，ellipse 是 descriptive。
8. 不能把“能导出”称为“顶会级美观”；需另做人工视觉审阅。

## 15. Caption 模板

仅在真实数据生成后填入，不得预填方向：

> **Prompt-end representation geometry under Assistant-Axis capping.** Each point denotes one item at a preregistered token position; arrows connect the same item under uncapped and p25 prefill-forward conditions. Colors/shapes distinguish valid and fallacious reasons, while hollow/filled markers distinguish intervention status. PCA is fit once on jointly residualized states and is descriptive; quantitative separation and displacement estimates use family-blocked analysis over 12 independent families. [TBD: observed result, exact layer, explained variance, and uncertainty].

## 16. 停止条件

出现以下任一情况立即停止 formal run：

- E2A input 或 manifest hash 不匹配；
- 模型 revision、axis、threshold、layer range 或 chat template 不匹配；
- layer/token boundary 无法唯一定位；
- nohook/monitor-only 不等价；
- layer 45 负控制异常；
- hook 泄漏；
- 张量非 finite、row 缺失或顺序漂移；
- worktree dirty 或运行代码未提交；
- GPU/软件环境发生未记录变化。

不得通过放宽容差、删除异常 family、改 seed、换层或重画 PCA 来“修复”科学结果。

## 17. 最终交付汇报格式

请按以下顺序汇报：

1. 分支、HEAD、工作区状态和锚点验证。
2. CPU Gate 与 smoke 结果。
3. 正式矩阵：期望 192/192，实际完成数及任何缺失。
4. 模型/axis/layer/token-position/hook 的精确定义。
5. raw artifact、manifest 和 SHA-256。
6. 主 PCA 的 explained variance 与图文件。
7. family-blocked V/F decoding。
8. V/F cap displacement 相似性与差异，含 family CI。
9. N/R 控制与 layer/position sensitivity。
10. 按 Route A/B/C/D/E 给出裁决。
11. 一句“当前数据允许写什么”和一句“禁止写什么”。
12. MATLAB/Inkscape 图形验收结果。
13. 已知限制和下一步是否需要 patching/held-out replication。

不要输出接收概率，不要把 PCA 图当作因果证明。

---

## 委托方期望的一句话故事（仅是假设，不是结果）

> 模型在 prompt-end 表示中仍保留理由有效性的可读信息，但 Assistant-Axis capping 对有效与错误理由产生相似的下游位移，因此表示轴上的 threshold control 并没有自动成为理由敏感的选择性策略。

只有 Route A 的全部定量条件真实成立后，才允许使用这句话。
