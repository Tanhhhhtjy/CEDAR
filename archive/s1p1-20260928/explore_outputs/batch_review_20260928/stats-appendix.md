# 统计与来源附录

原始结果保留在 analysis-output/formal；本目录为独立事后核验与汇总，不改写原始输出。SEED=20260927。

S1：每个 base 的 V/F 配对，先求 family 内均值再对 12 个 family 等权。单位为 logit 差，family SD 不是 seed SD。20,000 次 family bootstrap 的 95% 区间均标为描述性。对四个预先指定主对比枚举 2^12=4096 个符号翻转，双侧检验，四项 Holm 校正；假设 family 效应独立且对称，未通过正态性预检替换该既定方法。

| 主源 V−F 对比 | 均值 | family SD | 描述性 95% CI | Holm p | 负向 family |
| --- | --- | --- | --- | --- | --- |
| raw_delta_broad | -1.6233 | 1.1758 | [-2.2431, -0.9844] | 0.005859 | 11/12 |
| restored_delta_broad | -1.0043 | 0.5079 | [-1.2657, -0.7217] | 0.001953 | 12/12 |
| raw_delta_specific | -1.1979 | 1.4512 | [-2.0260, -0.4896] | 0.011719 | 11/12 |
| restored_delta_specific | -0.5209 | 0.7859 | [-0.9802, -0.1362] | 0.024414 | 10/12 |

P1：48 个 V/F 例子、24 个成对 base、12 个外层 family fold；每个 family 4 条。initial/target nuisance 回归和标准化只拟合训练 family；固定 Logistic C=1。主效应量为 family AUC（高于 0.5 的排序性能）及 frozen-threshold balanced accuracy。100 次 base 内 V/F 标签交换，所有外层折重新拟合，四方向同一标签置换；p=(1+超过或等于观测值次数)/101，四项 Holm。单侧超零分布检验依赖配对标签可交换性；不是 AUC 等价性检验。

| 训练→测试 | family AUC | 描述性 95% CI | balanced accuracy | 原始置换 p | Holm p |
| --- | --- | --- | --- | --- | --- |
| U->U | 0.8542 | [0.7917, 0.9167] | 0.5833 | 0.009901 | 0.039604 |
| U->P | 0.8333 | [0.7708, 0.8958] | 0.5833 | 0.009901 | 0.039604 |
| P->P | 0.8125 | [0.7292, 0.8958] | 0.6042 | 0.009901 | 0.039604 |
| P->U | 0.8125 | [0.7292, 0.8958] | 0.6042 | 0.009901 | 0.039604 |

P1 AUC 描述性 CI 来自 20,000 次 fold-family 重采样；训练集重叠，不当作精确采样分布。24 种 layer/position/representation/nuisance 组合都已保存；只有 prompt_L53_unit_rms_nuisance1 为主分析。未从敏感性组合挑选最佳结果替代主分析。

核验重算了全部 24×4 组探针的 AUC、balanced accuracy、描述性 CI，以及主置换 p/Holm；重算了 S1 主均值、family SD、CI、符号翻转 p/Holm。没有重跑模型前向或 100 次探针拟合。规范 token split 不含选项 token，且 fit/test 不重叠。

1266 项原始产物/来源哈希见 ../batch_integrity_20260928.json。模型 65GB shard 哈希沿用采集前校验记录，未重复全盘读取；counterfactual 数组当时没有独立期望哈希，本次检查形状/有限值/配对身份并建立事后哈希，不能称为采集前哈希验证。
