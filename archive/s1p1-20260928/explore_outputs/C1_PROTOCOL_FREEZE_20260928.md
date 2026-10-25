# CEDAR C1 双向成分移植：冻结方案（待服务器预检）

状态：`frozen_pending_remote_preflight`。本文档只冻结新增 C1 的问题、分层、对照和输出字段；目前没有 C1 GPU 结果，不能把历史 V7 脚本或 S1/P1 结果当作 C1。

## 目标

检验 P1 中可留出 family 的 V/F 方向是否具有局部因果作用：把同一条目在相反 capping 条件下的方向分量移植到当前状态，是否会在留出 family 上特异地改变 V−F 选择性？主结论必须比较真实方向与匹配随机方向，并同时报告 S1 固定规则校正后的结果。

## 固定数据与划分

- 使用已核验的 192 条冻结轨迹和原始 Qwen3-32B BF16、原 p25 配置；不生成新推理文本，不改 token、精度、attention 实现或指标。
- 主集使用 `e2a_uncapped` source 的 48 条 V/F 行（12 个 family、24 个 base），每个外层 family 留出；`full_p25` source 只作为预先声明的来源敏感性层。
- 每个外层 family 的方向、nuisance 回归、标准化和任何随机方向均只用其余 11 个 family 拟合或生成。测试 family 不用于挑层、方向、阈值、剂量或最佳结果。
- 主位点固定为 prompt 边界、layer 53 表示，即原采集协议中 layer-54 输入（已经过 layer-53 cap）的最后 prompt 位置。只在该一次 prompt prefill 调用修改该位置；后续 decode 位置不修改。

## 干预定义

设 `h_T` 为目标 capping 条件的主位点状态，`h_D` 为同一条目、相反条件的 donor 状态，`d_f` 为仅由训练 family 学得的 unit validity direction。对每个标量剂量 `alpha`：

```text
delta = alpha * <h_D - h_T, d_f> * d_f
h_T' = h_T + delta
```

`U→P` 表示 donor=U、target=P；`P→U` 表示 donor=P、target=U。`alpha=1` 是主移植；`alpha=0` 是 no-op；预先固定的有符号剂量为 `{-1, -0.5, 0.5, 1}`，只作方向响应曲线，不能在观察结果后删点。

## 对照

每个外层 family 固定生成 8 个等维随机方向，并按该 cell 的真实 `||delta||_2` 匹配实际扰动范数。随机方向的 seed、生成规则和配对索引写入 manifest；不挑选有利方向。每个 cell 必须包含：

1. no-op（零改动）；
2. 真实 `d_f` 单方向分量移植；
3. 8 个等维、等实际范数随机方向；
4. 完整局部状态替换 `h_T' = h_D` 作为标记性的阳性参考。

完整局部状态替换不是完整 KV cache 替换，也不单独证明语义恢复。若真实方向和随机方向只产生相同的整体尺度变化，C1 对该位点和剂量的特异机制判据为阴性。

## 主要终点与报告

主终点是外层 family 等权的 V−F selectivity 改变量，分别在 raw 和冻结的 S1 scale-corrected 规则下计算；比较真实方向与随机方向的差值及 family 级分布。同步保存 N/R 一般扰动、选项 logits/margin、argmax、实际投影变化、实际 `||delta||_2`、cap telemetry、运行环境和失败 cell。

主推断只使用 12 个 family 作为独立结构单位。bootstrap/置换区间作为描述性证据；不能把 forwards、token 或 cell 数量当独立样本。若真实方向在校正后相对随机方向呈预先预测的方向和剂量响应，才支持“该位置的该分量具有局部因果参与”；阴性结果只否定本位点/本方向/本剂量，不能推广为模型完全不用理由信息。

## 运行门槛

服务器启动前必须通过：权重与 cap SHA-256、冻结轨迹和 supporting archive hash、P1 方向及元数据存在、S1 校正参数已冻结、无本批残留进程/锁、物理 GPU 1/2/4/6 每卡至少 22 GiB 可用。先做一个小型工程 smoke（单 family、成对 V/F、U/P 双向、no-op/真实/一个随机/完整局部替换），验证 hook 只命中一次 prompt 边界且输出有限；smoke 不计入科学结果。完整运行必须断点写入逐 cell JSON/NPZ，并保留失败记录。

## 与历史结果的边界

历史 C1/V7 held-out 行为确认脚本使用另一套 192 条 N/R/V/F 协议，不能未经审查直接运行，也不能把它的输出标作本 C1。S1/P1 是观察性尺度和跨条件可解码性证据；本方案才是新增的局部因果干预。
