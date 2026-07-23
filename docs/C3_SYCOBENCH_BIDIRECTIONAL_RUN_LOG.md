# C3 SycoBench 双向选择性实验运行日志

状态：零 GPU 数据与工程已实现；用户授权的代理自审 Amendment 01 正在冻结
日期：2026-07-23
分支：`agent/c3-sycobench-bidirectional-selectivity`
基线：`364af5f797ee1eccf21c99e67570f59cbe052e5b`

## 当前停止点

- 尚未运行 Qwen、GPU 或 capping；
- 尚未查看任何 C3 模型输出；
- 用户在任何 C3 模型输出前放弃人工审核并授权代理自审；
- 本轮无人工或数据负责人审核、无独立审核者、无一致率或 Cohen's kappa；
- C2/V8 的模型输出不得参与 foil、模板、样本或停止规则选择。

## 零 GPU 实现记录

- 600 bases、1,200 direction instances、2,400 AO/S records 已确定性构建；
- harmful/helpful 为同题、同 foil、同模板的严格镜像；
- foil A/B/C/D = 150/151/150/149，所有 domain x difficulty cell 最大最小差不超过1；
- 四套 AO/S 对每个 target 都是15 tokens，target placeholder token position 相同；
- smoke24 产生每 condition 96 records，覆盖24个 domain x difficulty cells、四模板与四 target；
- review32 为 harmful/helpful 16/16，冻结 judgment 列保持空白；
- runner、单条件/跨条件 auditor、stem-cluster analyzer、attestation helper 与 preflight freezer 已实现；
- Amendment 01 后 targeted C3 tests 最终 23/23 passed；
- scoped Ruff check 与 Ruff format check passed；
- 全仓 pytest 240/240 passed，只有 C2 既有 mixed-model 的2条 convergence warnings；
- 代理已逐项阅读4对模板和32个冻结渲染案例，均通过既定构念与映射检查；
- Amendment 01、完成表、绑定 attestation 与 `REVIEW_GATE.json` 已通过；execution preflight
  在最终格式检查后冻结，并与上述文件一并提交推送后才允许 GPU。

## 后续固定记录

```text
protocol commit:
data/code freeze commit:
review gate commit:
GPU input commit:
visible GPUs:
smoke nohook/monitor/p25: 96/96, 96/96, 96/96
SMOKE_AUDIT:
formal uncapped: 2400/2400
UNCAPPED_AUDIT:
formal p25: 2400/2400
P25_AUDIT:
CROSS_CONDITION_AUDIT:
content retries:
analysis decision:
forced stop confirmed:
results commit:
```
