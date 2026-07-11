# M2.2 Development 模板日志

本日志只使用冻结的 50 条 development（25 BU + 25 BM）。正式 Pilot 的 200 条样本在模板冻结前
保持封闭。

## v1：原题 + 单 token 答题后缀

- commit：`62b81588fff31061d99d69123ff02337fa4f3300`；
- prompt SHA256：`a5e3258562b611c6106eafe7ee790859dde66ca7d596a797bb08923b56816ed0`；
- 后缀：`Answer with exactly one uppercase letter: A, B, or C. Do not explain.`；
- time_t accuracy：`96%`；
- time_t1 BU-Acc：`8%`，BM-Acc：`96%`，BREU：`52%`；
- trajectory：BU `4%`，BM `96%`，overall `50%`；
- BU time_t1 mean margin：`-8.54`，仅 2/25 预测为 C。

原 overall 闸门在字面上恰好通过，但主指标不可识别：对任何 capping 条件都有
`Delta_BU <= BU-Acc_uncapped = 8pp`，所以不可能达到预注册的 `10pp`。

这与上游已知行为一致，而非本地评分异常：

- Belief-R 官方仓库 commit `e9cec77b14e7deea9d26b3b04c41fc08ab2094e3` 的 direct prompt
  只在原问题后要求输出最终选项，并不解释可废止推理语义；
- 论文 arXiv `2406.19764v2` 的 Table A1 报告 Llama-3-Instruct-70B direct prompting
  BU-Acc `10.99%`、BM-Acc `92.09%`；
- 论文任务定义要求根据常识因果关系，判断第三个条件与第一个条件是合取（额外必要条件，更新）
  还是析取（替代路径，保持）。

决策：不解封 Pilot；在 development 上使用唯一一次模板修订，并增加 BU-Acc `>=10%` 的最小
可识别性条件。该条件只保证 10pp 效应在数学上可能，不代表基线质量充足。

## v2：显式公开任务语义

在原题和固定单 token 答题后缀之间加入：

```text
This is a defeasible-reasoning task. Interpret the conditionals using commonsense causal
dependencies, not as strict material implications. With two premises, infer the initial
conclusion. If a third conditional premise is present, decide whether its antecedent is an
additional requirement that must combine with the first antecedent (conjunctive: update the
prior conclusion to "may or may not") or a separate alternative route to the same consequent
(disjunctive: maintain the prior conclusion).
```

该说明来自论文公开任务定义，不含样本特定信息、答案或标签。v2 是最后一次模板版本；若未同时
达到 overall trajectory `>=50%` 和 time_t1 BU-Acc `>=10%`，M2.2 收口为基线无效。
