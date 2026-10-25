# E−1a 64 行 Smoke 摘要

- 总状态：`failed`
- 代码提交：`466ab72f895982ba09aca5d6ba66f61135a2bc02`
- 输入 SHA-256：`0b4f0f7c5023ba24c6cecf82963ccd64545f92c63d4acd3705e9d87ef86fa63b`
- 完成前向：256/256
- batch size：1
- 恢复记录数：0
- 内容重试：0

## Gate I：仪器正确性

结论：`passed`

| 检查项 | 结果 |
|---|---|
| `completeness_and_manifests` | 通过 |
| `dataset_identity` | 通过 |
| `model_and_tokenizer` | 通过 |
| `axis_identity` | 通过 |
| `axis_norm` | 通过 |
| `tokens` | 通过 |
| `layer_and_positions` | 通过 |
| `forced_choice_contract` | 通过 |
| `frozen_row_and_prompt_identity` | 通过 |
| `condition_contract` | 通过 |
| `answer_math_contract` | 通过 |
| `no_hook_equals_alpha_zero` | 通过 |
| `projection_shift_signs` | 通过 |
| `projection_shift_tolerance` | 通过 |
| `finite_outputs` | 通过 |
| `hook_removed` | 通过 |
| `all_prompt_positions` | 通过 |

## Gate II：方向检查

结论：`failed`

分析单位为 32 个 base pair；每个 pair 先平均 original/swapped margin。

- Δ(+200) = -3.3642578125，要求 `< 0`
- Δ(−200) = -2.5556640625，要求 `> 0`
- no_hook − alpha=0 = 0.0

## Original/Swapped 配对摘要

| 条件 | Original 均值 | Swapped 均值 | Pair 均值 |
|---|---:|---:|---:|
| `no_hook` | 3.84375 | 3.9296875 | 3.88671875 |
| `alpha_0` | 3.84375 | 3.9296875 | 3.88671875 |
| `alpha_pos200` | 0.26171875 | 0.783203125 | 0.5224609375 |
| `alpha_neg200` | 1.30859375 | 1.353515625 | 1.3310546875 |

## 异常与停止点

- 运行异常：[]
- 未发生生成式回答；未运行正式 900 行、p25/p1、J-Lens、MUSE 或 probe。
- 本 smoke 到此强制停止，等待 Gate I/II 人工复核。
