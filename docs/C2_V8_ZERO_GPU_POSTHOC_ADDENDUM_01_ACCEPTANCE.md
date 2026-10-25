# C2/V8 零 GPU 补充分析 01：最终验收

状态：`accepted_as_posthoc_descriptive_addendum`

## 1. 验收对象

- 冻结协议：`docs/C2_V8_ZERO_GPU_POSTHOC_ADDENDUM_01_MARGIN_CHOICE.md`
- 分析脚本：`scripts/analyze_c2_v8_margin_choice_addendum.py`
- 定向测试：`tests/test_c2_v8_margin_choice_addendum.py`
- 结果目录：`artifacts/m2/c2_v8_natural_social_confirmation/addenda/margin_choice_dissociation/`
- C2/V8 冻结来源提交：`8ea7025f2a726997d78ad8919455a8756e3c192e`
- 最终方法提交：`c8b64cc57c37e73cc24ab789b5e1f7cafb4f6e69`
- 结果提交：`9e4d35a`

本补充分析没有运行 Qwen、GPU、capping、新 seed 或新生成，也没有修改 C2/V8 的原始输入、回答或预注册主分析。

## 2. 验收检查

- Direct forced-choice：600 个 base，AO/S、uncapped/p25 四格完整；
- Free-thinking：240 个 base，AO/S、uncapped/p25 四格完整；
- 原正式 margin 交互点估计复算一致；
- 两次完整分析的 8 个交付文件逐文件 SHA-256 完全一致；
- 600 行 per-base CSV、2 张 SVG、JSON 与 delivery manifest 均通过解析；
- 定向测试：4/4 通过；
- Ruff：通过；
- 图表完成可视验收，误差线、标签、图例和边界迁移类别可辨认；
- 输入审计沿用 C2/V8 已通过的四项冻结审计，脚本再次硬校验其状态。

## 3. 冻结结果

### 3.1 连续 margin

- AO cap effect：`+1.2679` logit；
- S cap effect：`+0.4817` logit；
- `S−AO` cap interaction：`−0.78625` logit；
- C2/V8 原正式 95% CI：`[-0.86000, -0.71104]`。

p25 在 AO 和 S 中都把答案边界推向 induced target，但 AO 的平均平移更大，因此连续的社会压力 contrast 缩小。

### 3.2 离散 target 选择

- AO cap effect：`+1.67pp`；
- S cap effect：`+4.17pp`；
- `S−AO` cap interaction：`+2.50pp`；
- post-hoc base 分层 bootstrap 95% CI：`[+0.33, +4.67]pp`。

exact-MCQ 聚类区间仍为正，但 normalized-stem 聚类区间下界达到 `0.00pp`。因此该结果只能作为“临界决策边界上的离散 target-follow 增加候选”，不能升级为结构稳健的确认性结论。

### 3.3 Margin–choice dissociation

连续 contrast 下降与离散 target 选择上升可以同时发生：

1. S 的 uncapped margin 比 AO 更接近零；
2. p25 对 S 的绝对平移虽然较小，却足以让更多临界样本跨过零边界；
3. AO 的平移更大，但多数 AO 样本离零边界仍较远。

因此，“平均 margin 上的 SocialExcess 下降”不等于“离散错误跟随必然下降”。

### 3.4 Free-thinking

- strict final target 四格均为零；
- actual-answer margin interaction：`−0.1271`，区间跨零；
- scaffold margin interaction：`−0.0802`，区间跨零；
- thinking-token interaction：`+10.14` tokens，区间跨零。

strict 输出存在完全地板效应，不能据此宣称 thinking 普遍消除了副作用。

## 4. 允许与禁止的表述

允许：

> 在 C2/V8 的 direct forced-choice 读出中，p25 压缩了连续的社会压力 margin contrast，但在临界决策边界附近伴随更多离散 induced-target 选择，形成 margin–choice dissociation 候选。

禁止：

- “p25 已被证明增加真实谄媚”；
- “p25 已被证明降低真实谄媚”；
- “thinking 已被证明完全保护模型”；
- “该结果可泛化到其他模型、剂量、任务或所有答案位置”。

## 5. 尚未排除的混淆

- target 只覆盖 A/B，且 target=A/B 的离散交互不同；
- AO/S 不完全等 token；
- answer-step 实际修正量存在 arm 差异；
- 600 个 base 只对应 272 个 normalized stems；
- 离散终点及其区间属于验收后的 post-hoc 分析；
- 只有单模型、单 p25 剂量；
- 数据审核采用单一数据负责人整体审核，未达到原双人审核设计。

## 6. 决策

第 0 步完成。该补充分析可以作为 C2/V8 的正式 post-hoc 附录与下一轮设计依据，但不得替代 C2/V8 的预注册主结果。

下一步数据设计必须优先修复：

1. AO/S exact-token matching；
2. A/B/C/D target 位置平衡；
3. normalized-stem 或 family 聚类功效；
4. 具有非零 free-thinking target-follow 的自然难度；
5. 预先冻结连续 margin 与离散 choice 两个互补终点。
