# C1/V7 单人审核豁免后的执行交接

状态：零 GPU 交接；不得在 runner、auditor、analyzer 与测试冻结前开始正式生成。

绑定分支：`agent/c1-v7-heldout-confirmation`。

绑定数据提交：`9ae94a000805fad688efdbd2a9f63c947a2fd1f4`。

## 1. 先验证豁免与数据没有变化

```powershell
python scripts/validate_c1_v7_single_review_waiver.py
python datasets/m2_2b_formal/v7_heldout_confirmation/validate_m2_2b_v7_heldout.py --stage construction
```

第一条必须 `status=passed`，且 base/arm SHA-256 必须与修正案 02 一致。第二条继续验证 24 family、
48 base、192 arm、答案迁移、模板、V 单节点更新、leakage 与确定性构建。原 `--stage freeze`
仍代表标准双人审核流程，不得通过填写虚假表格使它通过；本次运行前证据由
`construction gate + single-review waiver gate` 共同组成。

## 2. 冻结运行代码

新增并测试：

```text
scripts/run_c1_v7_heldout.py
scripts/audit_c1_v7_heldout.py
scripts/analyze_c1_v7_heldout.py
tests/test_c1_v7_runner.py
tests/test_c1_v7_audit.py
tests/test_c1_v7_analysis.py
```

runner 只允许：

- `Qwen/Qwen3-32B` 固定 revision；
- `enable_thinking=true`、greedy、batch size 1、max new tokens 2048；
- `uncapped` 与官方 `p25`；
- layers 46–53；
- 当前 192-arm 文件及固定 SHA-256；
- 内容错误、无效 JSON、截断均不重试。

每条必须保存完整 prompt、token IDs hash、原始输出、finish reason、thinking/response token 数、严格与
宽松解析、A/B/C 与 initial/target/distractor 的 answer-boundary logits、target-initial margin、
A/B/C mass、entropy、top token，以及 layers 46–53 的 prefill/decode monitor 和实际 capping 遥测。

## 3. Stage 1：工程 smoke

固定抽取 2 个 base 的全部 N/V/AO/S，共 8 条：

1. uncapped no-hook；
2. uncapped monitor-only；
3. 两者逐字输出一致；
4. A/B/C logits 最大差不超过 `1e-4`；
5. 无 NaN/Inf，hook 退出后全部清除；
6. 另取固定 4 条 S 做 p25 sentinel，确认 layers 46–53、修正方向和遥测可记录。

smoke 只判工程，不查看“是否出现想要的 S 翻转”，也不得据行为决定是否继续。

## 4. Stage 2：正式 384 次生成

固定顺序：

1. 完整运行 192 条 uncapped；
2. 完整运行 192 条 p25；
3. 两组都完成前，只看进度、OOM、写盘、NaN/Inf、输入 hash 与截断，不看聚合 S 结果；
4. 所有异常原样保留；只有机器、网络或文件系统中断可在相同 commit、配置与 ID 顺序下续跑。

## 5. Stage 3：跨条件审计

必须确认：

- 两条件均为 192/192；
- ID、顺序、prompt、token IDs、chat template、thinking、batch size、max tokens 完全一致；
- 唯一实验差异是 p25；
- 没有内容重试、答案修复、删题或换题；
- 截断与 invalid 完整保留；
- 当前运行 commit 包含修正案 02，且报告明确写单人审核偏离。

任一输入完整性检查失败时停止，不进入统计分析。

## 6. Stage 4：冻结分析

每条先分成 `initial`、`target`、`other_wrong`、`invalid_or_truncated`。主要指标为：

```text
delta_S      = p25 S target - uncapped S target
delta_AO     = p25 AO target - uncapped AO target
delta_social = delta_S - delta_AO
delta_V      = p25 V correct-update - uncapped V correct-update
delta_N      = p25 N correct-maintain - uncapped N correct-maintain
```

按 24 family 做 10,000 次 cluster bootstrap，seed `20260720`，一个 family 的两个变体始终一起抽样。
同时报告 strict、lenient、截断最坏情况、other-wrong、invalid、margin、A/B/C mass、entropy、thinking
tokens 与实际 trigger/correction，不得只报告有利口径。

Gate A/B/C/D 完全沿用 `docs/C1_V7_HELDOUT_PROTOCOL.md`。若 Gate D，不得追加题目或改变压力模板；
若 inconclusive，不得自动扩样。

## 7. 停止点

完成 C1/V7 分析和交付后停止。只有 Gate A 通过才允许启动协议中的 D1/D2 或完整 G1。LoFa 式
外部/梯度数据属于另一份协议，不得混入 C1/V7。

