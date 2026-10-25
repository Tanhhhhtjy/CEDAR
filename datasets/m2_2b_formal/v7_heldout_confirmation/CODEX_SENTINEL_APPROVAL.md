# C1/V7 Phase 2 sentinel：Codex 独立复核与批准

状态：`approved_for_full48_construction`

日期：2026-07-21

审核对象：提交 `125daa58e21a03e571fd473908e4bef1319b560c` 中的四条 sentinel 与作者审阅报告。

## 独立复核结果

| base | 独立复算 | 唯一性检查 | changed node | 结论 |
|---|---|---|---|---|
| C1V7-001 | median `25 -> 18` | V 前后各唯一 | Delta 的单条 score record | 通过 |
| C1V7-021 | `[49,51] contained -> [47,53] partial overlap` | 三类关系互斥且穷尽 | uncertainty half-width record | 通过 |
| C1V7-033 | Omar `Ruby -> Quartz` | 对全部 3! 排列枚举，V 前后各一个完美匹配 | Nia 的完整 allowed-item record | 通过 |
| C1V7-048 | 唯一最小键 `{R} -> {Q}` | 对全部属性子集重算 closure/minimality | 单条 functional-dependency record | 通过 |

没有发现第二个合理答案、distractor 变真、题外知识依赖、V 多节点更新、AO/S 自定义文本或
certificate/rationale 不一致。

## 内容绑定

sentinel draft 文件 SHA-256：

```text
31e7be88f15d1b47378b2bfdc1191fea6e9ac437b787773518ddd37b43a0b95a
```

逐 base canonical SHA-256：

```text
C1V7-001 e861c1a3900501bbd138f0f3f8156b5e5f4b31ab1e64b7fedbd1480a61049d8e
C1V7-021 5cc7474c1e8d0c1118149a73741c4feb0da242aec62de559e76411acbf49c6ec
C1V7-033 d44f4c1804823efb60a93a5200a7fa9bf423fcc59ca5c9f5add37546a196c115
C1V7-048 a1487c0ada6dc32bb4870d30df8d6ec1f4dfa8f2244225800d34ecf772a73042
```

完整 48-base 文件必须原样复用这四个 JSON object。任何 canonical hash 变化都会使本批准
失效并要求重新审核该 sentinel；不得在扩展其余 44 条时顺手润色。

## 批准边界

本批准只允许：

1. 按 24 张冻结 family cards 构建剩余 44 个 base；
2. 组合完整 48-base 文件并运行零 GPU construction gate；
3. 在前瞻 gate 全部通过时生成空白双人盲审包。

本批准不等于 48-base construction approval，不批准人工标签，也不授权运行 Qwen、GPU、
capping、smoke 或正式实验。后续按 `docs/C1_V7_PHASE2_TO_CONFIRMATION_RUNBOOK.md` 执行。
