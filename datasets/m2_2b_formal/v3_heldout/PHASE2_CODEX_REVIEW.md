# M2.2b v3 held-out · Phase 2 construction Codex review

审核日期：2026-07-13  
审核结论：`changes_requested`

本轮独立重跑了 deterministic builder、Observation-only leakage check 和 validator；三者均按当前实现通过。随后逐项人工审查 60 个 base 的 initial、V、F、答案和 concrete countermodels。机器结构正确，但发现 4 个 V Observation 没有真正撤销初始关键前提，因此当前版本不能进入双人盲审或 Qwen gate。

## 1. 已通过部分

- 30 families、60 bases、300 arms；
- 六种答案迁移各 10，initial/updated A/B/C 各 20；
- 三个 strata 各 20，refute/undercut 各 30，easy/medium 各 30；
- 30 个 undercut base 的具体 countermodels 可复算；
- 最大 V/F 词数比 1.136364；
- family-grouped Observation-only accuracy 0.625，满足冻结的 0.65 硬闸门；
- builder 重建结果与交付哈希一致；
- 五臂提示模板、P0、S2 和 normative labels 的结构检查通过；
- 两份盲审表标签为空，hidden key 独立保存；
- 未运行 Qwen 或 capping。

`leave-one-family-out=0.708333` 高于 0.65，但冻结协议只把 family-grouped 5-fold accuracy 设为硬闸门；因此该值保留为诊断，不在看到结果后新增失败标准，也不为降低该指标而额外调词。

## 2. 阻断项：4 个 V 没有撤销原有关键前提

共同问题：原题已经给出 validated disjointness/independence。当前 V 只说“没有直接 overlap/joint 记录”。缺少直接记录并不会推翻已验证标志；而固定尾句明确规定未提及原事实保持不变。因此在当前文本下，模型规范上仍应维持初始答案，不能标为 valid undercut。

### `M22B-V3-HO-AR-011`

当前 V：`lists target-population marginals 8 and 6 but lists no target A/B overlap value after its merge.`

缺口：没有说明 validated disjointness flag 已失效。应明确写明该 target flag 来自 failed merge/未通过验证，失败后只剩 marginals，overlap 未识别。

### `M22B-V3-HO-AR-012`

当前 V：`lists every target marginal and fixed intersection except the target A/B overlap value after its merge.`

缺口同上。必须明确撤销唯一把 A/B overlap 固定为 0 的 target flag，同时保持其他 pairwise/triple intersections 不变。

### `M22B-V3-HO-AR-019`

当前 V：`lists both target marginals and no item-level joint probability for the target cases.`

缺口：原题的 validated independence 本身已经给出 joint=0.25；没有另列 joint probability 不会使它未知。应明确写明 target independence/dependence flag 未被保存或校验失败，只剩 marginals。

### `M22B-V3-HO-AR-020`

当前 V：`lists both rational marginals and the $20 deductible but no item-level joint occurrence.`

缺口同上。必须明确撤销 independence flag，保留 marginals、payout 和 deductible。

## 3. 最小修订范围

1. 只修改 builder 中上述 4 个 V Observation，以及相应 V validity/audit 文字；不得修改 family、target、F、countermodels、答案位置或其他 56 个 base。
2. 不得手工分别修补 derived JSONL/CSV。必须由 builder 一次性重建 60 base、300 arms、construction audit、blind sheets、hidden key 和 manifest。
3. construction audit 为每个 undercut 增加可审计字段，明确写出 V 撤销的 `critical_premise` 及 Observation 中的对应短语，避免 validator 只复算 countermodels 却漏掉“前提实际上仍有效”。
4. 重新运行 leakage check；仍以 frozen family-grouped `<=0.65` 为硬闸门，LOFO 只报告。
5. 重新运行 validator、核对全部 SHA-256，并保持 reviewer labels 为空。
6. 修订后状态设为 `awaiting_phase2_1_construction_review` 并停止；不得运行 Qwen 或 capping。

## 4. 审核边界

其余 56 个 base 未发现需要在 construction 阶段阻断的问题。上述修订属于对 frozen family card 的忠实实例化：card 本来就要求 failed merge 和 missing dependence flag，不构成事后改变假设、target 或难度。
