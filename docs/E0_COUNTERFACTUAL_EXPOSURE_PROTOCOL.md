# E0：v6 V/F/S/N 反事实暴露分析协议

状态：冻结；本协议在任何 E0 统计输出产生之前提交。

冻结日期：2026-07-19。

行为与遥测结果提交：`3cbed5d`。零 GPU 严格/宽松评分提交：`86bf46b`。

## 1. 目标与分析性质

E0 检查：在 v6 thinking-greedy 中从抵抗社会压力变为跟随诱导目标的 S 题，是否在
uncapped 轨迹上已有更高的官方 p25 反事实暴露。若是，才为“capping 可能在模型准备
抵抗时介入”提供相关机制证据；E0 本身不是新的因果干预，也不能证明机制。

本轮只读取既有输出，不加载模型、不执行 CUDA、不生成新回答。禁止 p1、新 seed、J-Lens、
MUSE、probe、E1/E2、重试、删题或按结果筛选。

## 2. 冻结输入

从 Git 对象读取，不依赖工作树中可能残留的 ignored 文件：

```text
3cbed5d:
  datasets/m2_2b_formal/v6_factorial_diagnostic/m2_2b_v6_factorial144.jsonl
  artifacts/m2/m2_2b_v6_factorial_diagnostic/
    qwen3_32b_v6_formal_{mode}_{uncapped|p25}_seed0_development/
      manifest.json
      results.jsonl

86bf46b:
  artifacts/m2/m2_2b_v6_zero_gpu_reaudit/revealed/
    strict_record_scores.jsonl
    lenient_record_scores.jsonl
```

四个 mode：`thinking_greedy`、`nonthinking_greedy`、`thinking_sampling`、
`nonthinking_sampling`。每格每 cap 144 条，共 1152 条行为结果。主要评分用 strict，
lenient 只作敏感性分析。v5 响应筛选曲线不进入 E0。

冻结 SHA-256：

```text
formal dataset:
495bb79d661e3d91d34bba435b0e92bc687da76844395673e949a430852893d4

strict scores:
772e901f1023f9f0b8de95901093f50845fe490f2e01fb421990235ec3adc809

lenient scores:
8943ed8b0d3b6fffeaf8076492e0e4b657c660eb22e08627d281f7c43081b672
```

八个 results/manifest 的 Git-blob SHA 必须在 `input_manifest.json` 中逐一冻结。每个格子
必须 144/144、ID 顺序与 dataset 一致、无重复，评分必须覆盖相同 1152 个 mode/cap/ID。

## 3. 遥测可用性闸门

E0 必须先生成 `telemetry_availability_audit.json`。逐条核验 uncapped 日志是否包含：

- layers 46–53；
- 官方 p25 reference exposure；
- `count`、`triggered`、`trigger_rate`、`mean_correction`；
- `prefill` 与 `decode` 分相；
- raw response、response token count、thinking integrity；
- manifest 中的 p25 threshold、axis/config identity。

旧 v6 monitor 没有显式保存 `correction_sum`，但它可以无损恢复为：

```text
correction_sum = mean_correction * count
```

绝不能再乘 trigger rate 或 triggered count。

预审已知日志没有 token-position exposure arrays、finish reason 或 answer logits。因此：

- thinking 与 final answer 不能从 decode telemetry 再分开；
- pre-answer、answer-position、最后 token exposure 均为 `not_available`；
- initial margin 不存在，`initial_margin_vs_flip.png` 必须是明确的不可用占位图；
- 不能为补字段重跑 GPU。

若 uncapped p25 reference exposure 在任一正式格子不完整，则停止 E0 并交付缺失清单。
若其在全部格子完整，则允许继续“aggregate-phase E0”，但不得把 decode 聚合写成精确的
抵抗时刻。

## 4. 行为迁移

对每个 mode/arm/ID 将 uncapped strict outcome 与 p25 strict outcome 配对。主要标签：

- `initial_to_target`；
- `target_to_initial`；
- `initial_to_other_wrong`；
- `target_to_other_wrong`；
- `stable_initial`；
- `stable_target`；
- `invalid_or_truncated`。

为保证不丢记录，其他迁移保留为 `other_transition:<before>_to_<after>`。每条同时保存原始
before/after outcome。V 的 target 是正确更新；F/S 的 target 是错误跟随；N 的 initial
是正确维持，因此跨 arm 不把 target 解释成同一规范含义。

## 5. 反事实暴露构造

每个迁移只使用对应 uncapped 轨迹的 p25 reference exposure。分别对 prefill/decode 计算：

- trigger rate = `sum(triggered) / sum(count)`；
- normalized correction = `sum(mean_correction * count) / sum(count)`；
- correction per record = `sum(mean_correction * count)`；
- layers 46–49；
- layers 50–53；
- layers 46–53；
- 八层逐层值。

decode 长度因回答而异，主要比较使用 normalized correction 与 trigger rate，不以总
correction sum 作为主要预测变量。p25 条件中的 actual exposure 只作描述，不能作为迁移前
预测量。

## 6. 主要和对照比较

主要比较固定为：

```text
thinking_greedy / S:
initial_to_target vs stable_initial
```

分别报告 prefill/decode × 46–49/50–53/46–53 的组均值、中位数、均值差和 family-clustered
bootstrap 95% CI。dataset 有 18 个 template family；bootstrap 以 family 为 cluster，
10,000 次，随机种子 `20260719`。若某次重采样缺少小迁移组，该 replicate 标为无效并报告
有效数。

随后以完全相同的冻结特征报告：

- thinking-greedy F；
- thinking-greedy V/N 和 other_wrong；
- nonthinking-greedy；
- thinking-sampling；
- nonthinking-sampling。

F/S 的 target-follow、V 的正确更新和 N 的维持必须分别解释。样本量很小的组只作描述，
不堆叠高维回归、不逐层挑最显著结果。

另外报告：

- 每个 mode/arm 的完整迁移矩阵；
- family leave-one-out 的主要 S 均值差；
- 暴露饱和度和跨题变异；
- other_wrong/invalid 是否与较高暴露同时出现；
- strict/lenient 迁移一致性。

## 7. 解释路由

- **抵抗时刻候选**：S initial→target 暴露更高，主要在 decode 而非 prefill；F 较弱同向；
  V/N/other_wrong 不同步恶化。由于没有 pre-answer 分相，只能称候选。
- **一般扰动**：暴露同时预测多 arm 错误、other_wrong/invalid，且缺乏 target 特异性。
- **更新门槛/顽固**：V target→initial 增加，F/S target-follow 同时下降而 N 稳定。
- **证据不足**：组太小、暴露饱和或 CI 很宽。小样本不显著不能自动否定状态门控。

E0 不使用行为结果选择层或阶段；只报告预先冻结的 phase 与 layer groups。

## 8. 固定交付与停止

```text
artifacts/m2/e0_counterfactual_exposure/
  input_manifest.json
  telemetry_availability_audit.json
  transition_table.jsonl
  counterfactual_exposure_normalized.jsonl
  primary_results.json
  transition_matrix.png
  exposure_by_transition.png
  phase_layer_exposure.png
  initial_margin_vs_flip.png
  E0_SUMMARY_ZH.md
  run_manifest.json
```

总结必须回答：更像顽固、一般退化、选择性 S/F 服从，还是证据不足。完成并推送后强制
停止，不进入 E1/E2、p1、新 seed 或任何机制工具。
