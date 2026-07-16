# M2.2b v6 Assistant Axis 阳性对照结果（seed 0）

## 结论

本轮阳性对照判定为 **通过**。

在 thinking 和 nonthinking 两种模式下，uncapped prefill 都满足预先冻结的四项方向标准：

- default assistant 相对 persona 的 Assistant Axis 投影差均值大于 0；
- 12/12 个 pair 的投影差均为正；
- persona 相对 default assistant 的反事实 p25 暴露差均值大于 0；
- 12/12 个 pair 的暴露差均为正。

p25 条件实际触发率非零，跨条件输入审计通过，所有方向 cosine 均小于等于 -0.999。因此，在当前 Qwen3-32B 实现中，安全 persona prompt 确实比 default assistant prompt 更偏离 Assistant 状态，也更容易触发官方 p25 capping。可以进入协议中的第二轮 matched greedy；本结果不回答 capping 是否提高安全性或是否损伤推理。

## Pair-level 主分析（uncapped prefill）

统计单位为 12 个 default/persona pair。每个 pair 先在 46–53 层逐层求差，再对 8 层等权平均；置信区间为 paired bootstrap 95% CI。

| 模式 | mean Δprojection | median | 正方向 pair | 95% CI | mean Δexposure | median | 正方向 pair | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thinking_sampling | 33.5806 | 32.9895 | 12/12 | [30.9242, 36.2477] | 6.9489 pp | 6.9275 pp | 12/12 | [6.8915, 7.0090] pp |
| nonthinking_sampling | 33.9717 | 33.3753 | 12/12 | [31.3492, 36.5735] | 6.6458 pp | 6.6292 pp | 12/12 | [6.5958, 6.6985] pp |

定义：

- `Δprojection = projection(default assistant) - projection(persona)`；
- `Δexposure = p25 counterfactual exposure(persona) - exposure(default assistant)`。

## 描述性 token 加权均值

这些值用于描述，不替代上面的 pair-level 主分析。

| 模式 | default prefill projection | persona prefill projection | default p25 exposure | persona p25 exposure |
|---|---:|---:|---:|---:|
| thinking_sampling | 31.3843 | -2.2394 | 92.1848% | 99.1329% |
| nonthinking_sampling | 30.5360 | -3.4748 | 92.5439% | 99.1892% |

## p25 实际触发

| 模式 | 全部记录 pooled trigger rate | default assistant | persona |
|---|---:|---:|---:|
| thinking_sampling | 43.6462% | 20.7349% | 60.8996% |
| nonthinking_sampling | 45.3675% | 22.2389% | 67.4966% |

persona 的实际 p25 触发率在两种模式下都明显高于 default assistant。

## Decode 次要分析

- thinking：mean Δprojection=70.4036，12/12 为正；mean Δexposure=44.5131 pp，12/12 为正。
- nonthinking：mean Δprojection=123.6542，12/12 为正；mean Δexposure=70.8916 pp，12/12 为正。

Decode 会受到采样内容和长度影响，仅作为次要证据。

## 完整性与异常

- 四格均为 24/24，总计 96/96；
- axis cross-cell audit：passed；
- 缺失、重复、空响应、mode integrity failure、运行中断均为 0；
- 4 条 thinking persona 响应达到 512-token 上限并在文本中间截断，已原样保留；这不影响主分析使用的 uncapped prefill 轨迹；
- `v6_factorial_report_seed0.json` 顶层显示 `incomplete` 是预期行为，因为本轮没有运行 formal 的其余 greedy/p1 格；axis 四格本身完整且审计通过。
