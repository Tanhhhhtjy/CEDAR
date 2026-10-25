# 第一批 S1/P1 证据核验

主问题：capping 的固定轨迹读出变化能否由非选项词表拟合的尺度/仿射预测解释；去除该预测后，V−F 差异是否仍存在；主探针是否能在 U/P 间双向转移。

数据为 Qwen3-32B BF16、原始冻结轨迹的开发分析。192 条轨迹包含 96 条 uncapped-source 主源和 96 条 capped-source 敏感性源，各跑 U/P，合计 384 次正式前向；另有 12 次工程 smoke。独立结构单位为 12 个 family，24 个 base；轨迹和 U/P 前向不是独立重复。P1 主分析为主源 V/F 的 48 条记录，每个 family 4 条。单一冻结采集配置，无独立模型种子复现。

## 核验与主要证据

396 个原始前向 NPZ 的记录哈希全部匹配；输入轨迹哈希、6 项来源指纹、全部端点与有限值检查通过。正式 384 条记录的历史 A/B/C/D logits 最大偏差为 0，末层 logits 重建最大偏差为 0；smoke 全状态重复最大偏差为 0。此处精确历史一致只覆盖所保存的历史选项 logits。

| 拟合方式 | identity RMSE | scale RMSE | affine RMSE |
| --- | --- | --- | --- |
| same_prompt_fits | 3.4204 | 1.3209 | 0.9328 |
| family_out_fits | 3.4204 | 1.3528 | 1.2315 |

表中为 12 个 family 等权的选项预测 RMSE。same-prompt 使用同一条轨迹的配对 capped 非选项 logits；family-out 仅用其他 family 拟合参数。两者不能互换表述。尺度/仿射预测有明显描述性改善，但不能把 RMSE 改善解释为因果解释率。

| 主源 V−F 对比 | 均值 | family SD | 描述性 95% CI | Holm p | 负向 family |
| --- | --- | --- | --- | --- | --- |
| raw_delta_broad | -1.6233 | 1.1758 | [-2.2431, -0.9844] | 0.005859 | 11/12 |
| restored_delta_broad | -1.0043 | 0.5079 | [-1.2657, -0.7217] | 0.001953 | 12/12 |
| raw_delta_specific | -1.1979 | 1.4512 | [-2.0260, -0.4896] | 0.011719 | 11/12 |
| restored_delta_specific | -0.5209 | 0.7859 | [-0.9802, -0.1362] | 0.024414 | 10/12 |

restored 指正斜率 affine 校正。broad = 其他三个候选 logits 的均值减初始答案 logit；specific = 目标答案 logit 减两个非初始/非目标候选的均值。先算 P−U，再作 V−F。负值表示 capping 对 V 的该 margin 改变量比 F 更负。校正后 broad 12/12 family 为负，specific 10/12 为负。残差并未降到本次重复误差底噪。

| 训练→测试 | family AUC | 描述性 95% CI | balanced accuracy | 原始置换 p | Holm p |
| --- | --- | --- | --- | --- | --- |
| U->U | 0.8542 | [0.7917, 0.9167] | 0.5833 | 0.009901 | 0.039604 |
| U->P | 0.8333 | [0.7708, 0.8958] | 0.5833 | 0.009901 | 0.039604 |
| P->P | 0.8125 | [0.7292, 0.8958] | 0.6042 | 0.009901 | 0.039604 |
| P->U | 0.8125 | [0.7292, 0.8958] | 0.6042 | 0.009901 | 0.039604 |

四个方向的 Holm 校正 p 均为 0.039604，受 100 次配对标签置换的 1/101 分辨率限制。AUC 与 balanced accuracy 的差别必须同时报告：较好的 family 内排序不代表稳定的固定阈值分类，也不证明模型使用这些信息进行推理。

## 敏感性及边界

| capped-source 敏感性对比 | 均值 | 描述性 95% CI |
| --- | --- | --- |
| raw_delta_broad | -0.6389 | [-0.9514, -0.3559] |
| restored_delta_broad | -0.4439 | [-0.6651, -0.2136] |
| raw_delta_specific | -0.2370 | [-0.5286, 0.0573] |
| restored_delta_specific | 0.0902 | [-0.2836, 0.4159] |

capped-source 的 restored specific 区间跨 0；不能把主源结果写成所有轨迹来源都一致。两个来源各 96 条的 A/B/C/D argmax flip 均为 0/96；这些是固定轨迹读出，不能宣称自然生成行为改变。

冻结程序已完成，但更广的科学协议仍有附加诊断缺口：P1 没有 capped-source 探针敏感性或单独预测 initial/target 身份的 nuisance endpoint；S1 family-out 保存了选项预测，但没有单独导出 held-out 非选项误差。它们未被标成已完成，也不改变已核验的主分析统计量。补充时应独立标注，保留本次原始产物。

## Claim Candidates

- Claim: 主源上，非选项拟合的 affine 校正降低选项预测误差，但未消除 V−F margin 差异。
  - Source evidence: S1_SUMMARY.json、S1_DETAILS.json、batch_integrity_20260928.json。
  - Allowed wording: 在这组固定开发轨迹与 12 个 family 中，仍观察到尺度校正后的差异。
  - Forbidden stronger wording: 已证明 capping 破坏语义有效性通路；尺度完全解释/完全不能解释干预。
  - Uncertainty: family 数少；符号翻转的对称性与独立性是假设；敏感性源不完全一致。
  - Next check: C1 双向干预和匹配随机对照。
  - Decision: keep
- Claim: 预先指定的 prompt L53 探针在四个 U/P 方向上具有高于置换零分布的 family AUC。
  - Source evidence: P1_SUMMARY.json、examples.json 与 100 次置换结果。
  - Allowed wording: 在该开发集和该探针流程下，V/F 标签具有跨条件可解码性。
  - Forbidden stronger wording: 发现同一语义表征、证明其被因果使用、证明两种条件等价。
  - Uncertainty: 48 条、24 配对 base、12 family；fold 训练集重叠；可能依赖提示表面特征；balanced accuracy 仅约 0.58–0.60。
  - Next check: nuisance endpoint、匹配 C1 对照、新结构 H1。
  - Decision: weaken
