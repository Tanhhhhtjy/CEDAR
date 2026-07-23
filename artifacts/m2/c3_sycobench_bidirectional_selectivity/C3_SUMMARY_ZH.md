# C3 SycoBench 双向修正选择性结果

判定：Gate F / inconclusive。

## 连续 margin 主结果

| 指标 | estimate | stem-cluster 95% CI | 判定 |
|---|---:|---:|---|
| Harmful CapSocial | -0.2785 | [-0.3710, -0.1848] | reliable_and_substantive |
| Helpful CapSocial | 0.3894 | [0.3135, 0.4646] | reliable_and_substantive |
| SelectivityChange | 0.6679 | [0.5544, 0.7767] | reliable_and_substantive |

## 完整性与边界

- A/B/C/D 输出空间稳定：True。
- SelectivityChange 等价性检验通过：False。
- 本轮只有 forced-choice next-token forward，不含自由生成、thinking 长度或截断分析。
- AO、S、harmful、helpful 的完整迁移矩阵与逐 base 结果均随交付公开。

## 强制声明

This is an exploratory bidirectional diagnostic derived from the previously used SycoBench-600 source. It is not an independent held-out confirmation set and does not provide logically valid evidence in the helpful condition.

本结果不能写成模型内部真实谄媚，也不授权剂量扫描、LoFa 或机制工具。
