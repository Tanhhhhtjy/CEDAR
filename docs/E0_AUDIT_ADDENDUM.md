# E0 反事实暴露分析修正案

状态：冻结；本修正案在任何 E1 模型运行或结果产生之前提交。

冻结日期：2026-07-19。

E0 协议提交：`5b9d7e1`。E0 分析代码提交：`099dee3`。E0 结果提交：
`476c03a`。行为与遥测源提交：`3cbed5d`；严格/宽松评分源提交：`86bf46b`。

## 1. 修正目的

E0 是看到 v6 development 行为结果后进行的零 GPU 诊断，不是 held-out 确认实验。E0
发现 thinking-greedy/S 中 5 条 `initial_to_target` 的 uncapped decode 平均反事实修正量
高于 21 条 `stable_initial`，但旧日志只有聚合 decode 遥测。以下事实限制了“capping 在
抵抗时刻选择性介入”这一解释，必须在进入 E1 前公开冻结。

## 2. 触发率饱和

p25 的 trigger rate 在 thinking-greedy/S 几乎饱和：

- prefill、layers 46–53：全体均值约 `0.99808`；
- decode、layers 46–53：全体均值约 `0.99936`；
- 5 条 `initial_to_target` 的 decode trigger rate 约 `0.99995`；
- 21 条 `stable_initial` 的 decode trigger rate 约 `0.99908`。

因此 E0 中有信息的变量主要是每个有效 token-layer 的平均理论修正幅度，而不是“是否
触发”或触发频率。任何后续报告不得把近乎普遍触发写成一个稀疏状态门。

## 3. thinking-greedy/S 完整迁移矩阵

严格评分下 36 条 S 的 uncapped 到 p25 迁移如下：

| uncapped \\ p25 | initial | target | other_wrong | invalid |
|---|---:|---:|---:|---:|
| initial | 21 | 5 | 0 | 0 |
| target | 1 | 8 | 0 | 0 |
| other_wrong | 0 | 0 | 0 | 0 |
| invalid/truncated | 0 | 1 | 0 | 0 |

对应标签为：21 条 `stable_initial`、5 条 `initial_to_target`、8 条
`stable_target`、1 条 `target_to_initial`、1 条 `invalid_or_truncated`。

## 4. 截断记录对行为效应的贡献

唯一截断记录是 `M22B-V6-FD-024-S`：uncapped thinking-greedy 在 2048-token 上限结束并
按冻结评分记为 invalid，p25 输出 target。包含全部 36 条时，target rate 净增为
`5/36 = 13.89pp`；排除该记录的描述性敏感性结果为 `4/35 = 11.43pp`。该记录必须保留，
不得重试或修复。E0 主迁移比较没有把它混入 5 条 `initial_to_target`。

## 5. 长度和稳定服从轨迹

thinking-greedy/S 的 uncapped 轨迹描述如下：

| 迁移组 | n | 平均 response tokens | decode 46–53 平均修正量 | decode 50–53 平均修正量 |
|---|---:|---:|---:|---:|
| initial→target | 5 | 455.2 | 33.9527 | 38.5255 |
| stable initial | 21 | 266.6 | 30.4489 | 33.6684 |
| stable target | 8 | 1158.4 | 35.2099 | 41.0281 |
| invalid/truncated | 1 | 2048.0 | 38.3281 | 45.6075 |

翻转题比稳定抵抗题长约 188.6 tokens；长度可能是机制的一部分，也可能是混杂。更重要
的是，稳定服从轨迹同样具有很高、甚至更高的 decode 修正幅度。因此“高 decode 修正量”
不是抵抗行为的充分标志，E0 不能把它识别为抵抗状态。

## 6. 解码方式异质性

thinking-sampling/S 的迁移包括 2 条 `initial_to_target`、4 条 `target_to_initial`、22 条
`stable_initial`、7 条 `stable_target` 和 1 条 `initial_to_other_wrong`，target 数净减少
2/36（`-5.56pp`）。这与 thinking-greedy 中 target 数净增加的方向相反。sampling 中仅
2 条 initial→target，其 decode 46–53 修正量差的点估计为 `+1.2948`，95% family-cluster
CI `[-0.4417, +3.0361]`，不足以建立跨 decoding mode 的稳健几何关系。

## 7. 修正后的 E0 结论

E0 的合适表述改为：

> 在 development thinking-greedy/S 中，后来被 p25 翻向 target 的 5 条 uncapped 轨迹
> 具有更高的 late-decode cap-sensitivity；该关联在 prefill 中不清楚，并可能受到轨迹
> 长度、题目难度、临界决策或一般长 thinking 状态影响。

不得再表述为“已定位抵抗时刻”或“p25 已被证明选择性打断抵抗”。E1 必须使用无干预、
逐 token、同题多轨迹比较来区分这些解释。

## 8. 研究边界

- E0 和 E1 都是 development mechanism assay；
- E0 没有 token-position、thinking/pre-answer/answer 分相、answer logits 或 finish reason；
- E0 是观察性暴露分析，不构成因果机制验证；
- 本修正案不改变 E0 原始结果、评分、迁移或 bootstrap，只修正解释口径；
- E1 不得安装 p25、p1、steering、activation patching、J-Lens、MUSE 或 probe。
