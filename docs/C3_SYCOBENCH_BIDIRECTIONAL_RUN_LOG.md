# C3 SycoBench 双向选择性实验运行日志

状态：零 GPU 数据与工程已实现；真实单人审核 Gate 待完成
日期：2026-07-23
分支：`agent/c3-sycobench-bidirectional-selectivity`
基线：`364af5f797ee1eccf21c99e67570f59cbe052e5b`

## 当前停止点

- 尚未运行 Qwen、GPU 或 capping；
- 尚未查看任何 C3 模型输出；
- 真实单人数据负责人审核未完成，GPU Gate 必须保持关闭；
- C2/V8 的模型输出不得参与 foil、模板、样本或停止规则选择。

## 零 GPU 实现记录

- 600 bases、1,200 direction instances、2,400 AO/S records 已确定性构建；
- harmful/helpful 为同题、同 foil、同模板的严格镜像；
- foil A/B/C/D = 150/151/150/149，所有 domain x difficulty cell 最大最小差不超过1；
- 四套 AO/S 对每个 target 都是15 tokens，target placeholder token position 相同；
- smoke24 产生每 condition 96 records，覆盖24个 domain x difficulty cells、四模板与四 target；
- review32 为 harmful/helpful 16/16，冻结 judgment 列保持空白；
- runner、单条件/跨条件 auditor、stem-cluster analyzer、attestation helper 与 preflight freezer 已实现；
- targeted C3 tests 最终 22/22 passed；
- scoped Ruff check 与 Ruff format check passed；
- 全仓 pytest 239/239 passed，只有 C2 既有 mixed-model 的2条 convergence warnings；
- review validator 当前按设计硬失败，`gpu_authorized=false`。

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
