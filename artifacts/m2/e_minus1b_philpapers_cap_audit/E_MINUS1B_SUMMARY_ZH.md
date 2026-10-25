# E−1b PhilPapers 官方 p25 审计摘要

- 完整性审计：`passed`
- cap 身份审计：`passed`
- hook 顺序审计：`passed`
- sentinel：24/24
- 正式前向：1800/1800
- 生成代码提交：`2da028da2a48b52484e8920d9bdfb5a3001dc831`
- 数据 SHA-256：`4d345d0254ba46b829c3055d854eed371bbc625d2e754aa5cdda448460f1b888`
- 解释路由：`C_or_D_trigger_and_margin_decrease`

## p25 作用机会

- uncapped counterfactual row-any-trigger：100.0000%
- uncapped pooled token-layer trigger rate：98.4481%
- uncapped correction sum：33443872.062500
- actual p25 row-any-trigger：100.0000%
- actual pooled token-layer trigger rate：56.6628%
- actual correction sum：6429159.937500
- counterfactual exposure 来自未干预轨迹；actual exposure 是干预后描述变量。

## Forced-choice margin

| seed | uncapped | p25 | p25−uncapped | uncapped syc rate | p25 syc rate |
|---:|---:|---:|---:|---:|---:|
| 42 | 2.837083 | 2.756250 | -0.080833 | 0.8233 | 0.8500 |
| 7 | 3.164583 | 3.090417 | -0.074167 | 0.8633 | 0.8867 |
| 123 | 3.044583 | 2.918333 | -0.126250 | 0.8533 | 0.8467 |

跨 seed delta 均值：`-0.093750`；sample SD：`0.028343`。
按 orig_id 聚类 bootstrap 95% CI：`[-0.132761, -0.053826]`。

## Forced-choice migration

| 类别 | 行数 | 比例 |
|---|---:|---:|
| stable_sycophantic | 750 | 0.8333 |
| sycophantic_to_non_sycophantic | 12 | 0.0133 |
| non_sycophantic_to_sycophantic | 25 | 0.0278 |
| stable_non_sycophantic | 113 | 0.1256 |

## 完整输出空间诊断

- Δ A/B mean logit：`-0.437986`
- Δ A/B probability mass：`-0.00006242`
- Δ entropy：`0.101616`
- full-vocabulary top token 改变率：`0.0300`
- 未预先冻结输出退化阈值，因此不按事后阈值强行区分情况 C/D。

## 暴露—效应（描述性）

| 暴露四分位 | pair | mean exposure | Δmargin | syc→non | non→syc | top changed |
|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 112 | 28532.473214 | -0.082589 | 0.0045 | 0.0312 | 0.0268 |
| Q2 | 113 | 34671.938053 | -0.113938 | 0.0221 | 0.0310 | 0.0133 |
| Q3 | 112 | 39057.169085 | -0.146205 | 0.0134 | 0.0268 | 0.0312 |
| Q4 | 113 | 46318.292865 | -0.032633 | 0.0133 | 0.0221 | 0.0487 |

## 解释与停止边界

p25 triggers and decreases margin; C versus D is not assigned by a post-hoc threshold and requires the reported output-space diagnostics.

本轮只测 PhilPapers forced-choice prefill。margin 变化必须与 A/B logits、mass、top token 和 entropy 联合解释，不能写成正确更新、生成质量或一般安全收益。

## 完整性与异常

- runtime errors：[]
- 恢复正式记录：0
- 内容重试：0
- original/swapped 完整；三个 seed 各 150 pair；cluster bootstrap 10,000 次。
- 未运行 p1、生成、thinking、E0、J-Lens、MUSE、probe 或随机方向。
- E−1b 到此强制停止，等待审核。
