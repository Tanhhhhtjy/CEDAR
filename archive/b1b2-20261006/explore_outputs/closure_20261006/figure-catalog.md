# 图表索引

两个图均来自已核验的 analysis/B1/SUMMARY.json；提供 PNG 和 PDF。均为 B1 开发阶段描述，不作显著性星标。

- figure-01-main-comparison：六个阶段对比 × 两个原始指标，点为 8 声明结构等权均值，横线为 20,000 次组 bootstrap 的描述性 95% 区间。目的是区分历史、末步、交互，并观察 broad 与 specific 不同方向。注意：区间不保证结构独立，不支持等效或自然行为结论。下一步是 B2，而非选有利分项。
- figure-02-structure-breakdown：每个声明结构的 EH、ET、I、total，格内给出精确到两位小数的 logit-margin 差。每个面板独立色阶，不能靠两图颜色深浅比较效应量。用于识别异质性，禁止据此删除结构或修改 B2。没有误差条，因为每格是对应旧结构下的确定性样本均值。

验收：图题标明 B1 和描述性；坐标标明原始 logit-margin 单位；全部结构保留；对比顺序固定；没有 p 值/显著性暗示。详尽数值、SD 和两种加权区间见 stats-appendix.md 与原始 SUMMARY.json。
