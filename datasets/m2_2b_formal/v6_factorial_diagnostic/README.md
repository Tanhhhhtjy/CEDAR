# M2.2b v6 因子诊断数据集

这不是用于“挑出目标曲线”的新题库，而是一个固定的 development 诊断集。它回答：v5
中几乎没有单调剂量模式，究竟是 greedy 解码、thinking 模式、一般 capping 效应，还是
任务本身造成的。

## 文件

- `m2_2b_v6_factorial144.jsonl`：36 个 base × N/V/F/S，共 144 条。
- `m2_2b_v6_axis_controls24.jsonl`：12 组 default-assistant/persona 安全阳性对照。
- `manifest.json`：来源哈希、机械选档规则、实验因子与规模。
- `build_m2_2b_v6_factorial_diagnostic.py`：确定性构建器。
- `validate_m2_2b_v6_factorial_diagnostic.py`：静态验证器。

## 关键纪律

1. 正式 144 条由 v5 候选池按 family/stratum 轮转规则机械选出；构建器不读取 v5
   的任何模型输出。
2. 同一 family 的两个参数变体使用同一难度；V/F 共用难度。
3. 不按本轮模型表现改题、删题或换难度。
4. 数据集属于 development causal diagnostic，不是 held-out 确证集，也不用于宣称总体效应。
5. 运行必须 `batch_size=1`，以排除 padding 并取得逐题、分 prefill/decode 的轴遥测。

完整实验定义见 `docs/M2_2B_V6_FACTORIAL_DIAGNOSTIC_PROTOCOL.md`，给实验同学的最短交接
见 `EXPERIMENT_HANDOFF_ZH.md`。
