# C1/V7 post-run 分析修正 01：actual cap-stats 遥测路径

状态：冻结的分析器 schema 修正；发生在正式 384 次生成和跨条件审计完成后、首次统计报告
成功生成之前。

日期：2026-07-21。

修正前提交：`1eba84713678c6fa6d19cad83c6c1160577e3f38`。

```text
pre-correction analyzer SHA-256:
9b27433a8dcf7f3086684d3b40036f46a0535650ef8c365a4a7630ccd76b9ced

corrected analyzer SHA-256:
96564a51a56fe8de1fab0921b6078a2ddfce3aa14d2e18c3f0d6ef0ba5a68a9c
```

## 1. 触发原因

正式 `formal-uncapped` 与 `formal-p25` 均完成 192/192，`CROSS_CONDITION_AUDIT.json` 为
`passed`。首次运行冻结 analyzer 时，在 `telemetry_summaries` 中抛出：

```text
AttributeError: 'str' object has no attribute 'get'
```

原因与 post-smoke auditor 修正相同：runner 沿用仓库既有 `stats_to_dict` 合同，将逐层实际
capping 统计保存为：

```text
cap_stats["by_layer"]["46" ... "53"]
```

但 analyzer 错误地直接遍历 `cap_stats.values()`，从而把顶层 `definition` 字符串当成逐层
统计字典。错误发生在最终 report 对象组装期间，分析 JSON 和中文总结均未写出。

## 2. 修正范围

只修改 `telemetry_summaries` 的 actual-p25 遥测读取路径：

```text
cap_stats.values()
-> cap_stats["by_layer"].values()
```

并新增 synthetic nested-schema 单元测试，同时保留 uncapped counterfactual exposure 的既有
读取合同。

不修改：

- 数据、runner、模型输出、checkpoint 或跨条件审计；
- strict/lenient 解析与四分类；
- delta_S、delta_AO、delta_social、delta_V、delta_N；
- family-cluster bootstrap、leave-one-family-out、截断敏感性或 Gate A/B/C/D；
- 任何生成或 capping 配置。

不会重跑 384 次正式生成，只重新运行只读 analyzer。

## 3. 已查看信息披露

修正前只看到 analyzer traceback、正式 192/192 完整性和跨条件工程审计 `passed`。当时没有
成功生成或查看任何 delta、迁移率、bootstrap CI、Gate 判定或中文结果摘要。本修正不由行为
方向触发。
