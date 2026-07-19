# E0 v6 反事实暴露分析摘要

- 分析类型：零 GPU、aggregate-phase observational mechanism diagnostic
- 遥测闸门：`passed_with_coarse_phase_limit`
- 路由：`resistance_interference_candidate_at_aggregate_decode_level`
- strict/lenient 评分均保留；strict 为主分析
- family bootstrap：10000 次，seed `20260719`

## 主要问题：thinking-greedy S

严格评分下 initial→target 为 5 条，stable_initial 为 21 条。

| phase / layers | flip 均值 | stable 均值 | 差值 | 95% family-cluster CI |
|---|---:|---:|---:|---:|
| prefill / 46–53 | 33.241409 | 32.891461 | +0.349947 | [-0.760250, +1.519604] |
| decode / 46–53 | 33.952653 | 30.448857 | +3.503795 | [+0.954871, +5.788459] |

decode 聚合了全部 thinking 和最终答案 token；不能把这一行解释成精确的pre-answer 抵抗时刻。

## 四条机制路线

### 1. 更像变顽固吗？

thinking-greedy V 的迁移为 `{"stable_target": 35, "other_transition:other_wrong_to_initial": 1}`。重点查看 target→initial 是否出现；target 在 V 中代表正确更新。

### 2. 更像一般退化吗？

thinking-greedy N：`{"stable_initial": 36}`；F：`{"stable_initial": 29, "initial_to_target": 1, "other_transition:other_wrong_to_other_wrong": 2, "other_transition:other_wrong_to_target": 1, "initial_to_other_wrong": 1, "stable_target": 1, "other_transition:other_wrong_to_initial": 1}`。other_wrong/invalid 与暴露的完整描述见 primary_results.json。

### 3. 更像选择性增加 S/F 服从吗？

只有当 S flip 的 uncapped exposure 明显高于 stable initial、F 较弱同向，且V/N/other_wrong 不同步恶化时，才支持这一候选。主比较的方向和 CI 如上；不能只依据 5 条行为翻转讲机制。

### 4. 还是证据不足？

只有 5 条 S initial→target，且日志缺少 token-position、pre-answer、answer logits与 finish reason。若 exposure 高度饱和或 CI 宽，应归为证据不足，不能把“不显著”写成状态门控已被否定。

## 可用性与禁止外推

- uncapped p25 counterfactual exposure 在全部 576 条记录、prefill/decode、46–53层均完整；correction sum 由 mean_correction × count 无损恢复。
- token-position 与 thinking/pre-answer/answer 分相不可用；initial-margin 图是明确占位图，没有为补字段重跑 GPU。
- 本轮不证明正确更新、一般安全收益或精确抵抗时刻的因果机制。
- 未运行 p1、新 seed、E1/E2、J-Lens、MUSE、probe 或新生成。
- E0 到此强制停止，等待审核。
