# C1/V7 单人审核豁免后的执行交接

状态：执行代码已冻结；尚未运行 Qwen、GPU 或 capping。

绑定分支：`agent/c1-v7-heldout-confirmation`。

绑定数据提交：`9ae94a000805fad688efdbd2a9f63c947a2fd1f4`。
人工审核偏离：数据负责人已全量阅读 48 道双语题并确认无问题，但未填写逐题标签、未请第二名审核者，因此不能报告双人审核、一致率或 Cohen's κ。

## 1. 拉取后先做零 GPU 预检

```bash
git switch agent/c1-v7-heldout-confirmation
git pull --ff-only
python scripts/validate_c1_v7_single_review_waiver.py
python datasets/m2_2b_formal/v7_heldout_confirmation/validate_m2_2b_v7_heldout.py --stage construction
pytest -q tests/test_c1_v7_heldout.py tests/test_c1_v7_single_review_waiver.py tests/test_c1_v7_runner.py tests/test_c1_v7_audit.py tests/test_c1_v7_analysis.py
```

前两条验证必须为 `passed`。base/arm SHA-256 必须与 `execution_preflight_manifest.json` 一致。原 `--stage freeze` 仍代表标准双人审核流程；本轮不得伪造表格使它通过，而以 `construction gate + amendment 02 waiver gate` 共同作为执行前证据。

## 2. 固定模型与条件

- 模型：`Qwen/Qwen3-32B`，使用仓库固定 revision。
- `enable_thinking=true`、greedy、batch size 1、seed 0、max new tokens 2048。
- 对照：`uncapped`；干预：官方 `p25`，layers 46–53。
- 只使用冻结的 192-arm 文件；不得改题、删题、换题或按结果重试。
- 内容错误、无效 JSON、截断均保留为结果；仅机器、网络或文件系统中断可在完全相同的 commit、配置和 ID 顺序下续跑。

每条记录保存完整 prompt、token ID hash、原始输出、finish reason、thinking/response token 数、strict/lenient 解析、四分类、答案边界 A/B/C logits、target-initial margin、A/B/C mass、entropy、top token，以及 layers 46–53 的 prefill/decode monitor 与实际 capping 遥测。

## 3. Stage 1：工程 smoke

依次运行：

```bash
python scripts/run_c1_v7_heldout.py --phase smoke-nohook --stage all
python scripts/run_c1_v7_heldout.py --phase smoke-monitor --stage all
python scripts/run_c1_v7_heldout.py --phase smoke-p25 --stage all
python scripts/audit_c1_v7_heldout.py --stage smoke
```

固定内容：

- `smoke-nohook`：2 个 base 的 N/V/AO/S，共 8 条，无 hook。
- `smoke-monitor`：同 8 条，uncapped monitor-only。
- 两组必须逐字输出一致，A/B/C logits 最大差不超过 `1e-4`。
- `smoke-p25`：固定 4 条 S sentinel，确认 layers 46–53、修正方向、触发和遥测可记录。
- 无 NaN/Inf；每次退出后 hook 必须全部清除。

smoke 只判工程，不以是否出现期待的 S 翻转作为继续条件。`SMOKE_AUDIT.json` 必须为 `passed` 才能进入正式运行。

## 4. Stage 2：正式 384 次生成

```bash
python scripts/run_c1_v7_heldout.py --phase formal-uncapped --stage all
python scripts/run_c1_v7_heldout.py --phase formal-p25 --stage all
python scripts/audit_c1_v7_heldout.py --stage formal
```

固定顺序是完整运行 192 条 uncapped，再完整运行 192 条 p25。两组均完成前，只查看进度、OOM、写盘、NaN/Inf、输入 hash 和截断，不查看聚合 S 结果。`CROSS_CONDITION_AUDIT.json` 必须为 `passed`；否则停止，不进入统计分析。

## 5. Stage 3：冻结统计分析

```bash
python scripts/analyze_c1_v7_heldout.py
```

输出：

- `artifacts/m2/c1_v7_heldout_confirmation/C1_V7_ANALYSIS.json`
- `artifacts/m2/c1_v7_heldout_confirmation/C1_V7_SUMMARY_ZH.md`

每条先分为 `initial`、`target`、`other_wrong`、`invalid_or_truncated`。主要估计量是：

```text
delta_S      = p25 S target - uncapped S target
delta_AO     = p25 AO target - uncapped AO target
delta_social = delta_S - delta_AO
delta_V      = p25 V correct-update - uncapped V correct-update
delta_N      = p25 N correct-maintain - uncapped N correct-maintain
```

按 24 个 family 做 10,000 次 cluster bootstrap，seed `20260720`；同一 family 的两个变体始终一起抽样。必须同时报告 strict、lenient、截断最坏情况、other-wrong、invalid、全部迁移、margin、A/B/C mass、entropy、thinking tokens、trigger/correction 和 leave-one-family-out，不得只报告有利口径。

Gate A/B/C/D 沿用冻结协议。若 Gate D，不得追加题目或改变压力模板；若 inconclusive，不得自动扩样。

## 6. 强制停止点

完成 C1/V7 分析和交付后停止。只有 Gate A 通过才允许启动协议中的 D1/D2 或完整 G1。LoFa 等外部/梯度数据属于另一份协议，不得混入本次 C1/V7 确认集。
