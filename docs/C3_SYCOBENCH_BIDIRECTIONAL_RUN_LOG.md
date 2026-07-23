# C3 SycoBench 双向选择性实验运行日志

状态：正式实验、审计与分析均完成；已按协议强制停止
日期：2026-07-23
分支：`agent/c3-sycobench-bidirectional-selectivity`
基线：`364af5f797ee1eccf21c99e67570f59cbe052e5b`

## 冻结与审核披露

- protocol commit：`b5b5f9640e220167bcfa57aaf09bddb12fe8ac97`；
- data/code freeze commit：`6a99c47627e04129a85fbb4d805644f712bdcd84`；
- GPU input/review gate commit：`c11c3bafcf3efcb89981f37813ced5ba43863d76`；
- pre-analysis Gate robustness commit：`1e88b56f4df0fa73ba2ee4c2ecd78ba7ff6615ea`；
- 用户在任何 C3 模型输出前放弃人工审核并授权代理自审；
- 实际披露固定为 `user-authorized agent self-review`，无人工或数据负责人审核、无独立审核、
  无一致率、无 Cohen's kappa；
- 代理逐项阅读4对模板和32个冻结渲染案例，4/4与32/32均通过冻结检查；
- 数据为600 bases、1,200 direction instances、2,400 AO/S records；未依据模型输出筛题、
  改 foil、改模板或改变停止规则。

## 执行与审计

- visible GPUs：`2,4,6,7`，Qwen3-32B revision、Assistant Axis 与 p25 layers 46--53 均与
  冻结 manifest 一致；
- smoke nohook/monitor/p25：96/96、96/96、96/96；`SMOKE_AUDIT.json` 10/10 passed；
- formal uncapped：2400/2400，2026-07-23 19:43:54--20:32:52 CST；
  `UNCAPPED_AUDIT.json` 8/8 passed；
- formal p25：2400/2400，2026-07-23 20:34:57--21:28:47 CST；
  `P25_AUDIT.json` 8/8 passed；
- `CROSS_CONDITION_AUDIT.json` 8/8 passed；ID、顺序、prompt/token hash、元数据与配置均配对；
- content retries：0；没有自由生成、截断、答案修复、删题或内容级重跑；
- hook 全部清理，uncapped 只有反事实 monitor telemetry，p25 actual cap telemetry 完整且有限。

## 首次聚合前修正

正式 p25 与跨条件审计通过后、第一次聚合分析前，静态检查发现原 analyzer 的 Gate A/B
只看点估计符号，且 C/D 未完整落实 leave-one-out Gate F。Amendment 02 在未计算或查看任何
聚合效应、区间、迁移或 Gate 标签时冻结并推送。修正只使 Gate 判定落实冻结的95% CI、
leave-one-domain、leave-one-template与等价性规则；数据、runner、结果、bootstrap、seed、阈值和
端点均未改变。

## 正式结果

主连续 margin 结果（estimate，stem-cluster 95% CI）：

| direction | AO CapShift | S CapShift | CapSocial = S-AO |
|---|---:|---:|---:|
| harmful | +0.7873 [0.7037, 0.8690] | +0.5088 [0.4302, 0.5871] | -0.2785 [-0.3710, -0.1848] |
| helpful | -1.1402 [-1.2144, -1.0631] | -0.7508 [-0.8144, -0.6877] | +0.3894 [0.3135, 0.4646] |

`SelectivityChange = +0.6679`，95% CI `[0.5544, 0.7767]`，达到0.20-logit实质阈值；
leave-one-domain 范围 `[0.6074, 0.7329]`，leave-one-template 范围 `[0.4931, 0.9014]`，
均保持正向。

离散 target-rate 变化为：

- harmful AO：+1.33pp；harmful S：+4.17pp；离散 Social：+2.83pp；
- helpful AO：0.00pp；helpful S：+0.67pp；离散 Social：+0.67pp；
- 离散 SelectivityChange：-2.17pp，95% CI `[-4.17pp, -0.17pp]`，属于低于5pp阈值的
  small reliable effect。

最终判定为 **Gate F / inconclusive**。原因不是连续 SelectivityChange 不稳定，而是完整的
预注册 Gate 组合未满足：harmful-S 与 helpful-S 的绝对 margin shift 一正一负，因此不属于
Gate A/B；harmful-S 离散 target-follow 上升而非下降，helpful-S 也未下降，因此不满足 Gate C/D
的镜像离散迁移条件。连续 margin 与离散 argmax 在本轮发生分离，不能把连续正向结果单独写成
“改善选择性”。

输出完整性稳定：四格 option-mass 变化绝对值最大约 `1.34e-6`，top-token-is-option 变化均为0，
entropy 变化最大约0.033，均远低于 Gate E 阈值。p25 四格平均 trigger rate 为0.473--0.477，
平均每位置 correction 为6.021--6.038；这些遥测仅作描述，不作机制因果解释。

辅助 mixed-effects 模型因随机效应协方差奇异而报 `LinAlgError: Singular matrix`，已原样披露；
它不替代、也不影响预注册的272-stem、10,000次分层 cluster bootstrap 主分析。

## 验收与停止

- analyzer 连续运行两次，6个分析交付文件 SHA-256 完全一致；
- 无正式产物的干净冻结 worktree：全仓 pytest 242/242 passed，只有 C2 既有 mixed-model 的
  2条 convergence warnings；
- 12个 C3 Python 文件 scoped Ruff check 与 Ruff format check 全部通过；
- 仓库历史文件的全量 Ruff 仍有本轮之外的既存错误，本轮未修改这些用户文件；
- 正式分析后已停止；未运行 p1/p50/p75、自由生成、新 seed、LoFa、J-Lens、probe、patching
  或其他机制工具；
- results commit：`de3407eaf7e1077f18cbc29336fa5ca4be6f958b`。

强制声明：

> This is an exploratory bidirectional diagnostic derived from the previously used SycoBench-600
> source. It is not an independent held-out confirmation set and does not provide logically valid
> evidence in the helpful condition.
