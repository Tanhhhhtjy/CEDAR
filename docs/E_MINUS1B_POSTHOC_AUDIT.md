# E−1b.1：PhilPapers p25 结果的零 GPU 补充审计

状态：冻结；本协议在任何 E−1b.1 统计输出产生之前提交。

冻结日期：2026-07-19。

输入结果提交：`b3f9d2a83d305ecf0bcc012546c029f413c213d7`。

## 1. 分析性质与边界

本轮是在看到 E−1b 结果后设计的诊断性补充分析，不属于预注册确认性结果。E−1b 发现
官方 p25 使平均 PhilPapers forced-choice sycophancy margin 下降 `0.09375`，但离散
sycophantic choice rate 从 `84.67%` 上升到 `86.11%`。本轮只解释这一指标分叉及其与
反事实暴露、输出空间变化的关系。

本轮不加载模型、不执行 CUDA、不产生新回答，也不运行 p1、新 seed、thinking、自由生成、
J-Lens、MUSE、probe 或新数据筛选。

## 2. 冻结输入和身份校验

输入目录：

```text
artifacts/m2/e_minus1b_philpapers_cap_audit/
```

关键文件及 SHA-256：

```text
row_results.jsonl
da6e01c4543173da62172b9cefa6f8512c22e1d21bbdfc51b4f4b6ca13bf00c4

pair_results.jsonl
3fa946a5a393e78e55d596ead63e71e47510a3b333550632941a9cc84963f2fa

prepare_audit.json
f08ccb4516f19657ab90cf3372c8da63fc4e3d72223c3a40638349c0fbb785d8

runtime_manifest.json
3691dc222c33b004e26b4adb0fe419cf3ac98a91eec9b13cfccdf6de9d7cff33
```

分析前必须生成 `INPUT_AUDIT.json`，并将工作树文件哈希与上述值及输入结果提交中的 Git
blob 逐一比较。必须确认：1800 条 condition-row 完整、900 个 prompt 均有
uncapped/p25、450 个 original/swapped base pair 完整、三个 seed 各 300 prompt、无重复、
A/B 与 syc/non-syc 方向映射和已冻结 margin 数学均正确。任何一项失败立即停止。

不得删除记录、修改 A/B 解析、重跑空输出、改变 original/swapped 聚合方式或按输出筛题。

## 3. 统计单位和离散主指标

数据中有两层配对：

1. 900 个 prompt-level condition pairs：同一 original 或 swapped prompt 的
   uncapped 与 p25；
2. 450 个 order clusters：同一 base 的 original 和 swapped。

点估计使用全部 900 个 condition-paired prompt：

```text
DeltaSycRate = P(syc | p25) - P(syc | uncapped)
```

但 original/swapped 不作为独立抽样单位。bootstrap 在每个 sample seed 内抽取 150 个
base cluster，并把被抽中的 original/swapped 两条一起带入；三个 seed 固定分层。运行
10,000 次，随机种子 `20260719`，报告 percentile 95% CI。seed-specific CI 也以该 seed
的 150 个 base cluster 为抽样单位。

辅助精确配对检验使用 900 prompt 的 McNemar exact binomial test，报告 12 条
syc→non-syc 与 25 条 non-syc→syc。由于 exact test 没有进一步校正 original/swapped
cluster，它只作辅助，不能替代 cluster bootstrap。

冻结结果口径：bootstrap CI 下界大于 0 才称为“可靠增加”；CI 跨 0 称为“方向为正的
候选现象”；CI 上界不大于 0 则不支持增加。

## 4. Margin 与离散选择分叉

为每个 prompt-level condition pair 保存 uncapped/p25 的 syc logit、non-syc logit、
margin，delta syc logit、delta non-syc logit、delta margin、delta A/B mass、delta
entropy、top-token 是否变化，并按下列四类冻结分类：

- `stable_sycophantic`；
- `sycophantic_to_non_sycophantic`；
- `non_sycophantic_to_sycophantic`；
- `stable_non_sycophantic`。

四组分别报告 uncapped margin、absolute uncapped margin、两类 logit delta、margin
delta、A/B mass delta、entropy delta 与 top-token change。重点比较
non-syc→syc 与 stable non-syc 的初始绝对 margin，以检查翻转是否集中在接近零的题。

图 `uncapped_margin_vs_delta_margin.png` 固定使用原始 margin 为横轴、delta margin 为纵轴，
标记四类迁移并画出 `p25 margin = 0` 的边界。图
`margin_distribution_by_transition.png` 报告四类初始 margin 和绝对 margin 分布。

## 5. 标准化反事实暴露

主要预测量只来自 `uncapped_monitor` 轨迹。每个 prompt 报告：

- pooled trigger rate；
- pooled `correction_sum / token-layer count`，命名为 normalized mean correction；
- `correction_sum / prompt token count`，命名为 mean correction per token；
- layers 46–53 每层的 `correction_sum / layer token count`；
- layers 46–49 与 50–53 的合并 normalized correction；
- layer 46 与 layers 47–53 的合并 normalized correction。

不得计算 `trigger_rate * mean_correction`。实际 p25 exposure 是干预后变量，仅作描述，
不作为翻转的前处理预测量。

现有 E−1b 日志只有逐层聚合，不保存 token-position exposure。因此最后输入 token、
pre-answer 位置、chat template 与正文分区均固定记录为 `not_available`，不得为补字段重跑
GPU。

## 6. 暴露—翻转诊断

预先规定两组比较：

1. non-syc→syc 与 stable non-syc；
2. syc→non-syc 与 stable syc。

对初始绝对 margin、pooled normalized correction、layer 46、layers 47–53、46–49、
50–53 报告组均值/中位数差及按 seed 分层 base-cluster bootstrap CI。所有八层结果完整
报告，不能挑选最显著层。

另外用 normalized correction 连续预测 delta margin、delta entropy、top-token change 和
净离散迁移，报告描述性斜率和 cluster-bootstrap CI。由于翻转仅 25/12 条，不堆叠复杂
模型，不进行逐层多重试探。

若 exposure 与 entropy 变化的关系比与定向迁移更清楚，应优先解释为一般输出分布扰动，
而不是精准反谄媚。

## 7. 固定交付与停止

```text
artifacts/m2/e_minus1b_posthoc_audit/
  INPUT_AUDIT.json
  discrete_sycophancy_bootstrap.json
  transition_level_results.jsonl
  normalized_exposure_results.jsonl
  exposure_effect_summary.json
  uncapped_margin_vs_delta_margin.png
  margin_distribution_by_transition.png
  layerwise_exposure.png
  E_MINUS1B_POSTHOC_SUMMARY_ZH.md
  run_manifest.json
```

中文总结分别回答：+1.44pp 是否可靠；指标为何分叉；结果更像定向偏移还是一般扰动；
哪些预先规定的层组最有信息；当前允许与禁止的结论。交付提交推送后停止 E−1b.1，再按
单独冻结的 E0 协议审计既有 v6 日志；不得自动启动任何 GPU 或机制实验。
