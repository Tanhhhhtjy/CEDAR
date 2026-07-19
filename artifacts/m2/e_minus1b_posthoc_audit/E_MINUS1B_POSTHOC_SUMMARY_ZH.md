# E−1b.1 零 GPU 补充审计摘要

> 本轮是在看到 E−1b 结果后设计的诊断性补充分析，不属于预注册确认性结果。

- 输入审计：`passed`
- GPU/model execution：`false`
- condition-paired prompts：900
- original/swapped bootstrap clusters：450（每 seed 150）
- bootstrap：10000，seed `20260719`

## 1. +1.44pp 是否可靠？

点估计为 `+1.4444pp`；cluster-bootstrap 95% CI 为 `[+0.1111, +2.7778]pp`。冻结口径分类为 `reliable_increase`。

辅助 exact McNemar：12 条 syc→non-syc、25 条 non-syc→syc，p=`0.047031`；该检验不校正original/swapped cluster，只作辅助。

| seed | uncapped syc | p25 syc | delta pp | 95% cluster CI |
|---:|---:|---:|---:|---:|
| 42 | 0.8233 | 0.8500 | +2.6667 | [+0.3333, +5.0083]pp |
| 7 | 0.8633 | 0.8867 | +2.3333 | [+0.3333, +4.6667]pp |
| 123 | 0.8533 | 0.8467 | -0.6667 | [-3.0000, +1.6667]pp |

## 2. 为什么连续 margin 与离散选择相反？

non-syc→syc 的初始 |margin| 中位数为 `0.625000`，stable non-syc 为 `2.625000`。翻转主要由靠近零边界的少量题决定；平均 margin 则由大量未翻转题的连续位移主导。因此两者不是同一estimand，方向可以相反。

四类完整分布见 `transition_level_results.jsonl` 和两张 margin 图。

## 3. 定向谄媚偏移还是一般扰动？

normalized exposure→delta entropy 的斜率为 `0.0029389426`，95% CI `[0.0001643466, 0.0058020421]`；exposure→净离开谄媚的斜率为 `0.0023881581`，95% CI `[-0.0007530408, 0.0055750932]`。结合 A/B mass、entropy 与 top-token 诊断，若前者清楚而后者不清楚，应优先解释为一般输出分布扰动。

## 4. 哪些层或阶段最有信息？

预先冻结的层组比较如下（non-syc→syc 减 stable non-syc）：

| 特征 | 均值差 | 95% cluster CI |
|---|---:|---:|
| normalized_mean_correction | -2.651331 | [-5.059320, -0.131191] |
| layer_46_normalized_correction | -2.007051 | [-3.756455, -0.188692] |
| layers_47_53_normalized_correction | -2.743371 | [-5.245378, -0.118375] |
| layers_46_49_normalized_correction | -2.242357 | [-4.230842, -0.161128] |
| layers_50_53_normalized_correction | -3.060305 | [-5.910251, -0.040158] |

日志没有 token-position arrays，因此最后输入 token、pre-answer、chat template与用户正文分区均为 `not_available`；本轮没有为补字段重跑模型。

## 5. 允许与禁止的结论

允许：报告 PhilPapers 上离散 syc rate 的 cluster-bootstrap 可靠性；说明连续margin 与边界翻转衡量不同对象；报告 normalized exposure 与输出扰动的描述性关联。

禁止：把 posthoc 关联写成确认性机制；把 margin 下降写成回答质量改善；宣称p25 普遍提高或降低谄媚；外推到正确更新、自由生成或安全性。

E−1b.1 到此停止；未运行 p1、新 seed、thinking、生成或机制工具。
