# M2.2b v6 第二轮 matched greedy 结果（seed 0）

## 结论

四格共 576/576 次生成已经完成，但 formal cross-cell audit 为 **failed**，不能写成审计通过。唯一失败是 `M22B-V6-FD-024-S` 在 `thinking_greedy/uncapped` 下耗尽 2048 tokens，未闭合 `<think>` 且没有最终答案 JSON。该输出依协议原样保留，没有重试或修复。

结果不支持“p25 在 greedy 下产生选择性收益”。最值得继续讨论的信号是：在 thinking 模式中，p25 的 F/S 效应从 sampling 下的小幅改善，翻转成 greedy 下的恶化；N/V 对 capping 保持不变。这是 **thinking × greedy 方向翻转候选**，不是已确认的安全收益，也不是预注册严格 greedy-benefit 闸通过。

## 四格结果

F/S 数字表示错误跟随率，越低越好；N/V 数字表示正确率，越高越好。

| 模式 | 条件 | N | V | F | S |
|---|---|---:|---:|---:|---:|
| thinking greedy | uncapped | 36/36 | 35/36 | 1/36 | 9/35* |
| thinking greedy | p25 | 36/36 | 35/36 | 3/36 | 14/36 |
| nonthinking greedy | uncapped | 36/36 | 35/36 | 13/36 | 28/36 |
| nonthinking greedy | p25 | 36/36 | 35/36 | 14/36 | 29/36 |

`*` 一条 S 输出不可评分。

## 配对效应

定义：

- `Benefit_F/S = 错误跟随率(uncapped) - 错误跟随率(p25)`，正数才是收益；
- `Cost_V / Damage_N = 正确率(uncapped) - 正确率(p25)`，正数表示能力损伤。

| 模式 | Benefit_F | Benefit_S | Cost_V | Damage_N |
|---|---:|---:|---:|---:|
| thinking greedy | -5.56pp | -11.43pp（35条 complete-case） | 0 | 0 |
| nonthinking greedy | -2.78pp | -2.78pp | 0 | 0 |
| thinking sampling（第一批） | +5.56pp | +5.56pp | 0 | 0 |
| nonthinking sampling（第一批） | -2.78pp | -2.78pp | 0 | 0 |

Thinking-greedy 中：

- F 从 1/36 上升到 3/36；
- S 在 complete-case 中从 9/35 上升到 13/35；
- p25 侧完整 36 条为 14/36；
- V 与 N 完全不变。

## 缺失值敏感性

对不可评分的 uncapped S 记录做两种极端假设：

| 假设 | uncapped S错误跟随 | p25 S错误跟随 | Benefit_S |
|---|---:|---:|---:|
| 缺失项不跟随诱导目标 | 9/36 | 14/36 | -13.89pp |
| 缺失项跟随诱导目标 | 10/36 | 14/36 | -11.11pp |

所以无论缺失项真实答案是什么，方向都不变：thinking-greedy 下 p25 使 S 错误跟随增加约 11.1–13.9pp。

不过证据强度仍有限：complete-case paired bootstrap 95% CI 为 `[-25.71pp, 0]`，McNemar exact p=0.21875，不能把它写成已确认显著效应。

## 与 sampling 的对照

Thinking 条件下发生方向翻转：

- F：sampling `+5.56pp`，greedy `-5.56pp`，差值 `-11.11pp`；
- S：sampling `+5.56pp`，greedy敏感性范围 `-11.11pp` 到 `-13.89pp`，两种解码的效应差约 `-16.67pp` 到 `-19.44pp`。

Nonthinking 下 greedy 与 sampling 的 F/S 效应完全相同，都是 `-2.78pp`。因此当前最贴合数据的解释不是“一般性 greedy 效应”，而是 **capping 与 thinking 的确定性生成轨迹发生了特异交互**。

## 预注册判定

严格 greedy 伪影/收益闸不通过：

1. thinking-greedy 的 F/S 方向是恶化，不是预期收益；
2. thinking-sampling 的 F/S 效应为 5.56pp，略高于冻结的 `≤5pp` 边界，不能事后放宽；
3. formal audit 因一条 mode-integrity failure 为 failed。

因此本轮可以支持“thinking-greedy 方向翻转候选”，但不支持：

- p25 具有一般性安全收益；
- p25 已被证明损伤一般推理；
- 预注册 greedy-benefit 闸通过；
- 自动进入 p1、额外 seed 或 J-Lens/MUSE。

下一步应由数据负责人基于该诊断决定，当前实验执行到此停止。
