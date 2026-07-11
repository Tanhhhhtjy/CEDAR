# 形式域题库（M3.1）

来源：Belief-R（Wilie et al., EMNLP 2024；HF CAiRE/belief_r）的 queries_time_t1.csv。
构建脚本：build_formal_dataset.py（随机种子=42，可复现）。

## 文件说明
| 文件 | 条数 | 用途 |
|---|---|---|
| formal_full.jsonl | 1744 | **完整集**，保跨论文可比性，作主结果 |
| formal_clean.jsonl | 786 | **干净子集**（agreement_lv>=5，可废止推理冲击最强），作稳健性检验 |
| formal_balanced_200.jsonl | 200 | 平衡采样（该更新/不该更新各100），供 Pilot 快速试跑 |

## 每条字段
- question: 题干（含前提+问题+选项）
- options: {a,b,c} 选项文本
- ground_truth: 正确答案（a/b/c）
- should_update: True=该更新(gt=c) / False=不该更新(gt=a/b)
- modus / agreement_lv: 元数据（agreement_lv 为可废止推理强度，4/5/6）

## 评测指标（详见 eval_formal_protocol.md）
- BU-Acc（合理更新率）= should_update=True 的题里答对比例
- BM-Acc（抗顽固率）= should_update=False 的题里答对比例
- 合理更新保留率 = BU-Acc(capped) / BU-Acc(uncapped)

## 语言
全英文（面向国际期刊/顶会投稿）。**不翻译**——直接用原始英文数据，避免翻译噪声与忠实度质疑。本项目所有数据集（形式域、价值域、三力刺激等）统一英文。

## 待办
- 跨数据集稳健性：另需引入 δ-NLI（见 v4 §3.4c）。
