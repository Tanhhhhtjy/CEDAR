# M2.2 Development 模板日志

本日志只使用冻结的 50 条 development（25 BU + 25 BM）。正式 Pilot 的 200 条样本在模板冻结前
保持封闭。

## v1：原题 + 单 token 答题后缀

- commit：`62b81588fff31061d99d69123ff02337fa4f3300`；
- prompt SHA256：`a5e3258562b611c6106eafe7ee790859dde66ca7d596a797bb08923b56816ed0`；
- run fingerprint：`5fbcaba1fcda5e26d14c3addabea471f252436e5d6b0223dee8fbaa33ff166f2`；
- summary SHA256：`186f3404205bddb8a3f7f4774f2d491df7ef82ba1756dc781f5d3c36c2fc67b5`；
- Parquet SHA256：`8eeab3ef7e45a5ee67ed870ab0614e9030581df09c3b59936d25612a8f448236`；
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

## v2 完整 development 与收口

- commit：`499ab1365c52ac2e755bde38432223ab7a3ce574`；
- prompt SHA256：`6f24e16d6094142fc166f67cb0038b1c9eff63fc4245f8e07956cc359977d03e`；
- run fingerprint：`332ccdb0830c536792e0501207b13cec00c63d85c5fcfcf2f6d41c4c760411da`；
- summary SHA256：`66666d7c146589df7da210df8af9dcb0c83f9f1a0857d0e624e724244d3981b4`；
- Parquet SHA256：`f1337986572b81baa511a009351aac88e7ac8b9d72b73dd57b395730836e9652`；
- 完整性：50 个 item × 2 个 step，100/100 条 score；
- time_t accuracy：`98%`；
- time_t1 BU-Acc：`8%`，BM-Acc：`92%`，BREU：`50%`；
- trajectory：BU `4%`，BM `92%`，overall `48%`；
- BU time_t1 mean margin：`-6.465`，仍只有 2/25 预测为 C。

相对 v1，显式任务语义没有增加 BU 正确数，BM 反而少 1 题正确，overall trajectory 下降 2pp。
两个 development 条件均未通过，且唯一一次模板修订已经用完。

最终决策：M2.2 标记为 `Belief-R/Qwen baseline invalid`。正式 200 题 Pilot 与 100 题 extension
保持封闭；不运行任何 capping 条件，不对认识论代价是否存在作正面或负面判断。后续若继续该科学
问题，需要预先更换具有足够 uncapped BU headroom 的任务或模型，并重新冻结独立样本与协议，不能
继续在当前 development 上调参。

## 收口后的构念判别

后续 uncapped 客观改错筛查没有解封上述 Belief-R Pilot，而是判定应更换任务而非模型。较难题
五臂诊断中，错误 prefill 后仅复核仍保留原错的 11 题里，明示真实错误后有 8 题改对（72.7%）；
纯社会压力下 20/20 守住正确答案。因此“Qwen 普遍拒绝修改答案”的宽解释被排除。

同时，具体但错误的论证使 14/20 个原本正确答案被带错，说明新形式域任务不能以原始改答率作为
单一认识论指标。后续协议必须把有根据更新增益和无根据论证伤害作为两个配对行为轴。完整设计、
题库、运行指纹与审计哈希见 `docs/M2_2_CORRECTION_SCREEN.md`。
