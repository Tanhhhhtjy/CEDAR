# M2.2b v3 held-out · Phase 2.1 Codex review

审核日期：2026-07-13  
审核结论：`approved_for_dual_blind_human_review`

本批准只授权启动两名独立人工审阅者的 V/F 双盲标注；在双人原始标签保存、协议指标计算及争议裁决完成前，不授权运行 Qwen 或 capping。

## 1. Phase 2.1 复审结果

- `AR-011/012` 已明确说明 target disjointness flag 来自 failed merge 且无效；
- `AR-019/020` 已明确说明 target independence flag 未保存或校验失败；
- 四题均保留原 marginals/其他 intersections/payout/deductible，只撤销 card 指定的唯一关键前提；
- construction audit 保存 `V_revoked_critical_premise` 和 `V_observation_revocation_phrase`；
- 其余 56 题、全部 F、countermodels、答案位置和 hidden key 未改变；
- independent rebuild、leakage check、validator 全部通过；
- family-grouped Observation-only accuracy 为 0.641667，满足冻结的 `<=0.65`；
- 最大 V/F word ratio 为 1.136364；
- 盲审表标签为空；Qwen/capping 均未运行。

LOFO accuracy 0.725 继续作为已预注册的诊断值报告，不追加为事后硬闸门。

## 2. 双盲分发规则

- Reviewer 1 只收到 `reviewer_1_blind.csv`；Reviewer 2 只收到 `reviewer_2_blind.csv`。
- 两人均不得接触 `m2_2b_v3_blind_review_key.jsonl`、construction audit、family cards、模型输出或彼此标签。
- 两人独立完成，不讨论题目，不以“每对必有一个 valid、一个 invalid”为前提。
- 保留原模板；完成后另存为 `reviewer_1_completed_raw.csv` 和 `reviewer_2_completed_raw.csv`。
- 在两份 raw 文件都保存前，不计算一致率、不交换标签、不启动 adjudication。

## 3. 每列填写规则

- `text_a_validity` / `text_b_validity`：只填 `valid`、`invalid` 或 `unclear`。判断 Candidate conclusion 是否仅凭原题与 Observation 逻辑成立。
- `text_a_english_rhetorical_quality_1_5` / `text_b_english_rhetorical_quality_1_5`：1=明显破碎难读，3=清楚但普通，5=自然、专业、连贯且有说服力；只评英文表达，不评逻辑真假。
- `qualified_minimal_pair`：填 `yes`、`no` 或 `unclear`。判断 A/B 是否除决定性逻辑效力外，在来源类型、具体性、数字精度、长度与专业程度上足够可比。
- `text_a_unique_fatal_flaw_if_invalid` / `text_b_unique_fatal_flaw_if_invalid`：仅对判为 invalid 的文本，用一句话指出唯一致命推理错误；valid 可留空。
- `confidence_1_5`：对本行整体判断的信心，1=很不确定，5=非常确定。
- `notes`：记录歧义、第二个可能错误或需要第三人裁决之处；无则留空。

## 4. 人工闸门

两份 raw 标签完成后才计算：

- V/F validity raw agreement `>=85%`；
- Cohen's kappa `>=0.8`；
- V/F 英文修辞质量均值差绝对值 `<=0.5/5`；
- `unclear`、minimal-pair failure 和疑似第二失效点必须逐项报告。

仅把两位 reviewer 不一致或任一方标 `unclear` 的字段送入 `adjudication_template.csv`。第三人不得修改两份 raw 标签。裁决完成后再决定修订题目还是冻结数据；不得按 Qwen 表现筛题。

## 5. 下一硬停点

收到两份 completed raw CSV 后，先运行 human-review scoring/adjudication，更新 manifest 的 human-review 字段并接受 Codex/data lead review。只有人工闸门通过且最终数据重新哈希冻结后，实验同学才能运行 uncapped Qwen gate；capping 仍继续禁止。
