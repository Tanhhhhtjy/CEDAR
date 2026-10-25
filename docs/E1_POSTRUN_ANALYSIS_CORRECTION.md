# E1 运行后分析交付修正

状态：运行后、结果可见后的工程修正；不属于预注册协议变更。

日期：2026-07-20。

## 发现时点

E1.0、E1.1、E1.2 和 E1.3/F 的全部 GPU 轨迹已经完成。冻结 generation commit 为：

```text
7dce05e226bc7624540c61857490ae2245f02781
```

首次 `--stage final` 分析完成后发现两个分析交付问题：

1. statsmodels 当前版本的 `ConditionalResults` 不暴露 `mle_retvals`。条件 logistic 已经完成
   拟合并生成参数和标准误，但脚本在读取不存在的收敛元数据时抛出 `AttributeError`，从而
   把可报告的模型错误标为 `not_estimable`。
2. E1.3/F 控制的行为闸门、主几何和 axis 几何已经在内存中计算，但主程序只写出了 S 主
   分析，没有把 `F_control` 写入安全交付 JSON，也没有在中文摘要中报告。

## 修正范围

- 保留条件 logistic 的原始拟合、参数、标准误和置信区间；若当前 statsmodels 不暴露优化器
  收敛元数据，则将 `converged` 写为 `null`，并明确记录 `not_exposed_by_statsmodels`，不得
  猜测为 `true` 或 `false`。
- 将已经冻结计算的 `F_control` 嵌入 `primary_within_prompt_results.json`，并在中文摘要中
  报告其记录数、描述性行为闸门、主修正量和 axis 投影结果。
- `run_manifest.json` 分别记录 generation provenance 与 analysis provenance，并记录本修正
  文件的哈希。
- 允许在同一分支、干净 worktree、generation commit 的后代提交上重跑零 GPU 分析，但从
  generation commit 起的代码差异只能涉及本修正文件、E1 analyzer 和 E1 analyzer tests。

## 明确不变

本修正不改变：

- 任何模型输入、输出或逐 token 遥测；
- 任何题目、seed、行为标签或截断记录；
- Assistant Axis、p25 threshold、层、phase 或位置边界；
- resist/yield 定义、主特征、统计单位、bootstrap 10,000 次或 seed `20260719`；
- Gate 阈值、解释门槛或强制停止规则。

修正后只允许零 GPU 重跑 `--stage final`。E1 仍然强制停止，不运行 E2、p25、p1、steering、
patching、J-Lens、MUSE 或 probe。
