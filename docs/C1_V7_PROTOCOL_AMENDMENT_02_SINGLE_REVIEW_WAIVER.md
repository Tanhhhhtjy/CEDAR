# C1/V7 协议修正案 02：单一数据负责人审核豁免

状态：冻结；本修正案在任何 C1/V7 Qwen、GPU 或 capping 运行之前生效。

日期：2026-07-21。

绑定数据提交：`9ae94a000805fad688efdbd2a9f63c947a2fd1f4`。

## 1. 修正原因

原协议要求两名独立审核者逐题填写盲审表、计算一致率与 Cohen's kappa，并完成分歧裁决。
受项目时间限制，数据负责人已报告完整查看 48 张双语 base card，未发现题目、答案、V 更新、
AO/S/N 处理或语言方面的问题，但没有填写逐题表格，也没有第二名独立审核者。

不得把这一情况记录成双人盲审完成，也不得生成虚假的逐题标签、一致率或 kappa。本修正案只允许
在透明披露该偏离的前提下继续 C1/V7。

## 2. 本次豁免的精确范围

本次 C1/V7 以以下审核状态进入运行：

- 审核者人数：1；
- 审核者角色：数据负责人；
- 审核方式：完整浏览 48 张双语 base card，整体确认无问题；
- 逐题结构化标签：无；
- 第二名独立审核者：无；
- reviewer agreement：不可计算；
- Cohen's kappa：不可计算；
- 人工分歧裁决：不适用。

该状态只能写作 `single-data-lead global review with no recorded item-level labels`，不得写作
`double-blind human review passed`、`independent review passed` 或 `kappa threshold passed`。

## 3. 不变部分

本修正案不改变：

- 24 family、48 base、192 arm；
- 当前题目文本、答案、N/V/AO/S 模板；
- `thinking-greedy`、Qwen3-32B、uncapped/p25；
- 384 次正式生成；
- delta_S、delta_AO、delta_social、delta_V、delta_N；
- family-cluster bootstrap、Gate A/B/C/D；
- 不得按模型输出改题、删题、补题或扩样的规定。

任何数据内容或哈希变化都会使本豁免失效，必须重新批准。

## 4. 仍然必须通过的运行前门槛

豁免只替代“双人逐题审核与一致率”门槛。以下门槛继续有效：

1. 48 base、192 arm、24 family 与六种答案迁移的静态验证全部通过；
2. 48/48 solution-first 独立复算通过；
3. development overlap、leakage、固定 N/AO/S 模板与 V 单节点更新检查通过；
4. 当前 base 与 arm 文件哈希分别为：
   - `791427c50a0be3cd222a436a610533a2f58547ea385c411d24ba6327bce1971a`
   - `847f5df9eeffc2ecd0a881b708f70b29abc4a3eacb2aa8ae39a9762c60211ffe`
5. `private/` 不得提交；
6. smoke 工程闸必须在正式 384 次生成前通过；
7. 任何输出截断、无效 JSON 或异常均按原协议原样保留，不得按内容重试。

## 5. 解释与论文披露

本轮仍可用于检验 v6 development 候选效应是否在新 family 上复现，但其人工质量控制证据弱于原
预注册版本。论文、附录、飞书记录和对外报告必须披露：

> Due to time constraints, C1/V7 received a complete global review by one data lead, but no
> second independent reviewer, item-level review record, inter-rater agreement, or Cohen's
> kappa was available. All computational construction checks were retained.

若后续获得第二审核者，可另行补做审核并报告为 post hoc quality audit；不得倒签为运行前双人审核。

