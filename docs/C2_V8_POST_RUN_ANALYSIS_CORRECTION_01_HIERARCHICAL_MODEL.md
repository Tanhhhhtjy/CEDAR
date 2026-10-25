# C2/V8 Post-run analysis correction 01：层级模型分组修正

状态：仅分析代码修正；不重新运行 GPU，不修改冻结输入或模型输出
日期：2026-07-23

## 发现时点与已见结果

该问题在四个正式生成阶段全部完成、Smoke/Direct/Free 跨条件审计均通过、首次正式分析已经
生成之后，于最终交付验收中发现。修正前已经看到：主 direct bootstrap 估计为
`-0.78625`、95% CI 为 `[-0.86000, -0.71104]`，free-generation 的 target SocialExcess
为 `0`，而预注册层级模型报告 `LinAlgError: Singular matrix`。

## 原因

首次实现以 `base` 为 MixedLM 主分组并设置 base 随机截距，同时把 `domain` 作为组内方差
分量。每个 base 只属于一个 domain，因此该 domain 指示列在 base 组内与随机截距完全共线，
造成结构性奇异；这不是数据缺失或生成失败。

## 修正

固定效应公式不变：

```text
margin ~ cap * arm + target_position + difficulty + template
```

随机效应改为与协议一致、可识别的嵌套表示：

```text
(1 | domain) + (1 | base)
```

Statsmodels 中以 `domain` 作为主分组随机截距，以 `0 + C(base)` 作为嵌套 base 方差分量。
新增回归测试，要求该设计能够拟合、收敛并返回有限的 `cap × S` 交互估计。

## 不变项与执行纪律

- 不修改 600 bases、1,200 AO/S arms、240-base free 子集或 12-base smoke 子集；
- 不修改 prompt、runner、Qwen revision、axis、p25、层、解码或任何原始结果；
- 不重跑任何 GPU 记录，也不删除、筛选或重试输出；
- 不修改 bootstrap、seed、主终点、阈值、判定矩阵或解释边界；
- 原冻结 analyzer、runner、运行日志和 preflight manifest 保持逐字不变，以便原 GPU 输入
  哈希继续通过；
- 修正放在独立 post-run analyzer 中，先独立提交，再从该提交重新运行一次分析并重新生成
  delivery manifest；
- 修正后的结果无论方向如何均原样交付。

## 2026-07-23 正式运行事实

```text
GPU input commit: 381e915c74f52bf9a19c1044c8cef6bfdd48b258
single-review gate commit: 9a7c9e4
visible GPUs: 2,4,6,7
smoke direct nohook/monitor/p25: 24/24, 24/24, 24/24
smoke free nohook/monitor/p25: 24/24, 24/24, 24/24
SMOKE_AUDIT: passed
direct uncapped/p25: 1200/1200, 1200/1200
DIRECT_CROSS_CONDITION_AUDIT: passed
free uncapped/p25: 480/480, 480/480
FREE_CROSS_CONDITION_AUDIT: passed
CROSS_CONDITION_AUDIT: passed
content retries: 0
analysis decision label (initial run): p25_reduces_social_pressure_influence_on_direct_boundary
forced stop confirmed: yes
```
