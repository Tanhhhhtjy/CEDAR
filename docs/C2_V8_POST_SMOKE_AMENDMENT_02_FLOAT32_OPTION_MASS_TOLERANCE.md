# C2/V8 Post-smoke 修正案 02：float32 option-mass 数值容差

状态：Smoke 后、正式实验前的工程审计修正  
日期：2026-07-23  
原始 GPU 输入提交：`9a7c9e49db81a0898be8fae29a64d54aede7d451`

## 触发原因

六个 smoke phase 已按冻结顺序完成，共保留 144/144 条结果且内容重试为 0。聚合行为结果
尚未查看。首次 smoke 审计正确停止了流水线：direct 与 free 的唯一失败项均为
`diagnostics_well_formed`；输入 identity、no-hook/monitor logits、输出 identity、hook 清理、
telemetry、p25 actual stats 和候选 token ID 均通过。

只读定位发现，runner 将 full-vocabulary logits 转为 float32 后使用
`logsumexp -> exp` 计算概率。有限精度舍入使四个 option probability 的和可能略高于 1。
冻结 smoke 的 216 个诊断载荷中：

- 46 个载荷超过原上限 `1 + 1e-6`；
- 最大超出量为 `1.979864721946001e-6`；
- 最小偏差为 `-1.944967493727745e-6`；
- stored option mass 与四项 stored probability 之和逐项一致；
- conditional probability 之和、margin 重构、argmax、entropy 和有限性检查均通过。

因此失败来自审计容差过窄，而非概率内容不一致或模型输出损坏。

## 冻结修正

仅将 option-mass 的允许上界从：

```text
1 + 1e-6
```

改为：

```text
1 + 64 * float32_epsilon
```

其中 `float32_epsilon = 2^-23`，新容差约为 `7.6294e-6`。该界高于已观察的纯舍入偏差，
同时仍会拒绝 `1e-4` 量级的概率质量错误。原有的 stored mass 重构、conditional sum、
margin、argmax、finite 和 entropy 检查全部保留。

新增回归测试必须证明：

- `1 + 1.98e-6` 的自洽 float32 载荷通过；
- `1 + 1e-4` 的载荷失败；
- 其他诊断条件不被跳过。

## 合规边界

本修正不改变：

- 任何 smoke response、logit 或 telemetry；
- 数据、prompt、target、模型 revision 或 tokenizer；
- p25、层范围、阈值或 Assistant Axis；
- 正式样本、主要终点、统计方法或解释规则。

不重新生成 smoke，不进行内容重试。修正提交并推送后，只对原样冻结的 smoke 产物重新
运行审计。只有审计通过才从新提交开始正式 Direct；否则继续停止。该修正发生在任何正式
Direct/Free forward 之前。
