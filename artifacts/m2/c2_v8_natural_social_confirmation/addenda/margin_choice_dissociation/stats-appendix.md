# C2/V8 Margin–Choice 补充统计附录

## 统计口径

- Base单位：600；
- Bootstrap：10000次；
- Base seed：20260724；
- 各指标使用固定整数偏移；精确seed保存在JSON的每个区间对象中；
- 分层：domain × difficulty；
- 区间：percentile 95% CI；
- 所有新增检验均为post-hoc且未做多重比较校正。

## 核心差分之差

| Metric | Estimate | SD across bases | 95% CI |
|---|---:|---:|---:|
| margin_social_did | -0.7863 logit | 0.9470 | [-0.8585, -0.7144] |
| target_social_did | +2.5000pp | 27.2945 | [+0.3333, +4.6667] |

## 结构聚类敏感性

| Cluster | N clusters | Margin DID 95% CI | Target DID 95% CI |
|---|---:|---:|---:|
| exact_mcq_id | 542 | [-0.8652, -0.7085] | [+0.34, +4.72]pp |
| normalized_stem_id | 272 | [-0.8992, -0.6768] | [+0.00, +4.94]pp |

## Target字母敏感性

| Target | N | Margin DID | Target DID |
|---|---:|---:|---:|
| A | 437 | -0.7045 | +3.20pp |
| B | 163 | -1.0054 | +0.61pp |

## 完整迁移

```json
{
  "direct": {
    "AO": {
      "initial->initial": 557,
      "initial->other_wrong": 1,
      "initial->target": 11,
      "other_wrong->other_wrong": 6,
      "target->initial": 1,
      "target->target": 24
    },
    "S": {
      "initial->initial": 361,
      "initial->target": 31,
      "other_wrong->initial": 1,
      "other_wrong->other_wrong": 4,
      "target->initial": 6,
      "target->target": 197
    }
  },
  "free_strict": {
    "AO": {
      "initial->initial": 239,
      "initial->other_wrong": 1
    },
    "S": {
      "initial->initial": 240
    }
  }
}
```

## 推断边界

风险差本身是离散效应量；margin以logit差报告。由于新增分析是在结果验收后提出，不使用p值升级主张，也不把未校正区间解释成新的确认性Gate。
