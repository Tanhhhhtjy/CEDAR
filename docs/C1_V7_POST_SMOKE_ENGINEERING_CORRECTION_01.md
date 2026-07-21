# C1/V7 post-smoke 工程修正 01：p25 cap-stats 审计路径

状态：冻结的只读审计器修正；发生在三组 smoke 生成完成后、正式 384 次生成之前。

日期：2026-07-21。

修正前提交：`38925fe812ba8304d590608f2f55c514c6dbdf47`。

```text
pre-correction auditor SHA-256:
3924a508cadae85cdfc0ee7682ca9cbde4e2e4b2471a32f2b3025f4230fe44af

corrected auditor SHA-256:
73f63178586ffd1050564da4d279bc267df7852c1c0197e9766522978bea81ec
```

## 1. 触发原因

三组 smoke 均完整生成并退出成功。首次运行 `audit_c1_v7_heldout.py --stage smoke` 时，唯一
失败项为 `p25_stats_complete_layers_46_53`；其余检查均通过，包括：

- no-hook 与 monitor-only 输出逐字一致；
- A/B/C logits 最大差为 0；
- prompt hash 一致；
- monitor 遥测存在；
- hook 全部清除；
- thinking integrity 通过。

诊断发现 runner 沿用仓库既有 `stats_to_dict` 合同，将逐层统计保存为：

```text
cap_stats["by_layer"]["46" ... "53"]
```

但首次冻结的 auditor 错误地在 `cap_stats` 顶层寻找 `"46" ... "53"`，因此把完整遥测误判为
缺层。四条 p25 sentinel 的 `by_layer` 实际均精确包含 46--53，所有层的
`total_positions > 0` 且 `projection_correction_sum >= 0`。

## 2. 修正范围

只修改只读 smoke auditor，使其从 `cap_stats["by_layer"]` 校验八层，并新增以下单元测试：

1. 实际 nested `by_layer` schema 可以通过；
2. 缺少任意一层仍然失败。

不修改：

- 24 family、48 base、192 arm 或任何 hash；
- runner、hook、monitor、capper、模型或生成配置；
- 已生成的 20 条 smoke 轨迹及 checkpoint；
- Gate、统计 estimand 或正式运行规模。

不会为了本修正重跑 smoke 生成，只重新运行只读 auditor。

## 3. 已查看信息披露

修正前只查看了工程闸输出、cap-stats 的字段结构与逐层非负 correction/position 计数。没有计算
或查看聚合 `delta_S`、`delta_AO`、迁移率、Gate A/B/C/D 或正式结果。本修正不由行为方向
触发。
