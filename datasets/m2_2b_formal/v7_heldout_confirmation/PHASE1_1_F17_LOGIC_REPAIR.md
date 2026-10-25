# C1/V7 Phase 1.1：F17 逻辑矛盾修复与重新批准

状态：`approved_minimal_logic_repair`

日期：2026-07-21

范围：仅 `C1V7-F17 unique_bipartite_assignment`。没有生成 sentinel、base、arm 或模型输出；
没有运行 Qwen、GPU 或 capping。

## 1. 原冻结卡为什么不可能实例化

原卡同时要求：

1. 原图有且只有一个完美匹配 `M`；
2. V 只增加或删除一条允许边；
3. V 后有且只有一个不同的完美匹配 `M'`。

三种情况均矛盾：

- **增加一条边**：旧匹配 `M` 的所有边仍在，因此 `M` 仍有效。若出现不同的 `M'`，更新后
  至少有两个完美匹配，不可能只有 `M'`。
- **删除不属于 M 的边**：`M` 完全不受影响，不能迁移到不同的唯一匹配。
- **删除属于 M 的边**：若删除后存在 `M'`，则 `M'` 不使用被删除边；它在删除前也已经
  存在，与“原图只有 M”矛盾。

因此问题不在参数选择，而在原 family card 的迁移合同本身。继续硬写只会伪造不可能的题。

## 2. 最小修订

把 V changed node 从：

```text
add or remove one allowed edge
```

修订为：

```text
replace one agent's complete allowed-item record while every other allowance row remains fixed
```

一个 agent 的完整 eligibility record 是一个原子数据记录，所以仍满足“V 只修改一个明确记录
节点”。该记录替换可以同时撤销旧边并声明新边；其他 agent 的记录、item universe 和匹配规则
全部不变。

## 3. 可实现性见证

三个 agent 为 A、B、C，三个 item 为 1、2、3。

原始允许集合：

```text
A -> {1}
B -> {1, 2}
C -> {2, 3}
```

唯一完美匹配为：

```text
A-1, B-2, C-3
```

V 只把 A 的完整记录替换为：

```text
A -> {3}
```

B、C 两行保持不变。更新后的唯一完美匹配为：

```text
A-3, B-1, C-2
```

所以修订后的 family 可以合法实例化，同时仍要求全局一一匹配和全局重算，不退化成局部查表。

## 4. 重新批准与绑定

同步修改：

- `family_cards24.jsonl` 的 F17 `v_changed_node`、`v_update_mechanism`、
  `migration_structure`、`nonisomorphism_rationale`；
- `development_overlap_audit24.jsonl` 的 F17 rationale 与 canonical card hash；
- `DEVELOPMENT_OVERLAP_MATRIX.md` 的 F17 人类可读说明；
- `PHASE1_FAMILY_REVIEW.md` 与 Claude sentinel 护栏。

新 F17 canonical SHA-256：

```text
3f3a1366ecfb7be74dbfe43a581007ae855ab8b9905ecb5aeca54b25c1576e8b
```

重新批准理由：

- initial key fact 未变：初始 allowance matrix 有唯一完美匹配；
- answer-generation mechanism 未变：通过全局一一约束求唯一匹配；
- V 仍只改变一个明确的 eligibility record node；
- 与最近 development family 的 partition/reachability 机制仍不同；
- family 数量、stratum、variant 数量、N/V/AO/S 模板与统计口径均未改变。

F17 允许重新进入 4-sentinel 构建；若实例化再次改成单边增删，必须硬失败并退回修题。
