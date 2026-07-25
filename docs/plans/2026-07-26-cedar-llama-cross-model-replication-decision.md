# CEDAR Cross-Model Replication Decision / CEDAR 跨模型验证决策

> Date: 2026-07-26 (Asia/Shanghai)  
> **中文：** 日期：2026-07-26（Asia/Shanghai）  
> Status: paper-design decision only; no Llama experiment has been run and every Llama result remains `[TBD-R]`.  
> **中文：** 状态：仅为论文与实验设计决策；尚未运行任何 Llama 实验，所有 Llama 结果均保持 `[TBD-R]`。

## Decision / 决策

CEDAR will retain official Assistant-Axis threshold capping as the empirical intervention and add `meta-llama/Llama-3.3-70B-Instruct` as a planned cross-model replication of the Qwen3-32B study. A second semantic activation axis is not required for the main paper.

**中文：** CEDAR 保留官方 Assistant-Axis 阈值钳制作为实证干预，并加入 `meta-llama/Llama-3.3-70B-Instruct`，作为 Qwen3-32B 研究的预定跨模型验证。主论文不要求第二条语义 activation axis。

## Why This Is the Cleanest Extension / 为什么这是最干净的扩展

- Changing the model while retaining the Assistant-Axis construct and threshold-capping operator directly tests model dependence.<br>
  **中文：** 在保持 Assistant Axis 构念与阈值钳制算子不变时更换模型，可以直接检验结果是否依赖特定模型。
- The official Assistant-Axis release already provides model-specific Llama axis assets and a recommended capping configuration, reducing researcher degrees of freedom.<br>
  **中文：** Assistant Axis 官方发布已经提供 Llama 专属 axis 资产和推荐钳制配置，可以减少研究者自由度。
- Independently re-extracting the same Assistant construct would test estimator stability, not semantic generalization; the source paper already includes substantial cross-model, trait-space, and role-PC1 validation.<br>
  **中文：** 独立重提取同一 Assistant 构念只能检验估计稳定性，不能证明语义泛化；源论文已经包含较充分的跨模型、trait-space 和 role-PC1 验证。
- A new social-pressure axis would introduce a new construct, construction dataset, calibration problem, and leakage risk. It may be studied later, but it is not needed to establish the CEDAR evaluation gap.<br>
  **中文：** 新建 social-pressure axis 会同时引入新构念、构造数据、校准问题与泄漏风险。它可以后续研究，但不是证明 CEDAR 评估缺口所必需的。

## Frozen Source Configuration to Reproduce / 待复现的官方源配置

| Field / 字段 | Qwen primary / Qwen 主实验 | Llama replication / Llama 验证 |
|---|---|---|
| Model / 模型 | `Qwen/Qwen3-32B` | `meta-llama/Llama-3.3-70B-Instruct` |
| Model revision / 模型 revision | `9216db5781bf21249d130ec9da846c4624c16137` | `[TBD: freeze before execution]` |
| Axis / 轴 | Official model-specific Assistant Axis | Official model-specific Assistant Axis |
| Asset revision / 资产 revision | `3b3b788432ad33e3a28d9ff08e88a530c0740814` | `[TBD: verify and freeze; current official dataset revision is the same source snapshot]` |
| Source p25 configuration / 源 p25 配置 | `layers_46:54-p0.25` | `layers_56:72-p0.25` |
| Actual decoder layers / 实际解码器层 | 46--53 | 56--71 |

No activation vector is transferred between models. “Same axis” means the same semantic construction applied in each model's own activation space.

**中文：** 不在模型之间迁移激活向量。“同一 axis”指在各模型自身的激活空间中使用相同语义构造。

## Planned Evidence / 计划证据

The Llama track will repeat the same estimands rather than introduce new endpoints:

**中文：** Llama 轨道重复相同估计量，而不新增事后终点：

1. CEDAR-Social: paired AO/S under uncapped and official p25, with direct margin as the prespecified primary readout and free generation as a modality boundary.<br>
   **中文：** CEDAR-Social：在 uncapped 与官方 p25 下配对运行 AO/S，以 direct margin 为预先指定的主要读数，以 free generation 界定输出模态边界。
2. CEDAR-Evidence: paired N/R/V/F under uncapped and official p25, reporting `WEG`, `Cost_V`, `U_V/U_R` decomposition, fallacy following, and output integrity.<br>
   **中文：** CEDAR-Evidence：在 uncapped 与官方 p25 下配对运行 N/R/V/F，报告 `WEG`、`Cost_V`、`U_V/U_R` 分解、谬误跟随和输出完整性。
3. Static controls: matched readout and an independently fixed equivalence criterion at the relevant doses.<br>
   **中文：** 静态对照：在相关剂量下使用匹配读数和独立冻结的等效性判据。
4. Dose extension: after the primary comparison is frozen, evaluate the same prespecified percentile family with model-specific thresholds.<br>
   **中文：** 剂量扩展：主要比较冻结后，使用模型专属阈值评估同一预先指定的 percentile 家族。

## Reporting and Claim Gates / 报告与主张门槛

- Report Qwen and Llama estimates, decompositions, and confidence intervals separately.<br>
  **中文：** 分别报告 Qwen 与 Llama 的估计量、分解和置信区间。
- Do not pool two models into a population-level model estimate or call the result universal.<br>
  **中文：** 不把两个模型合并为模型总体估计，也不把结果称为普遍规律。
- Use “across two model families” only if both audited model-specific results support the bounded claim.<br>
  **中文：** 只有两个经过审计的模型专属结果都支持受限主张时，才能使用“跨两个模型家族”。
- If Llama differs in sign, magnitude, modality, or dose response, report model heterogeneity and retain the Qwen-scoped conclusion.<br>
  **中文：** 如果 Llama 在方向、幅度、输出模态或剂量响应上不同，应报告模型异质性，并保留限于 Qwen 的结论。
- Mechanism and controller experiments remain Qwen-primary unless a later protocol explicitly promotes cross-model mechanism replication.<br>
  **中文：** 机制与控制器实验继续以 Qwen 为主，除非后续协议明确将跨模型机制复现升级为正式目标。

## Paper-Level Positioning / 论文定位

The general contribution is the CEDAR evaluation framework. The empirical contribution is a scoped study of official Assistant-Axis capping, strengthened by a planned second-model replication. The paper does not claim that every activation axis or every capping method incurs the same cost.

**中文：** 通用贡献是 CEDAR 评估框架；实证贡献是对官方 Assistant-Axis capping 的范围受限研究，并由计划中的第二模型验证增强。论文不声称每条 activation axis 或每种钳制方法都会产生相同代价。
