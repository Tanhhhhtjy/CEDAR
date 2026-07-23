# C2/V8 外部自然社会压力确认结果

状态：正式分析完成，按协议停止，不自动运行其他控制器或机制实验。

## 主终点：direct forced-choice margin

- base 数：600；AO/S、uncapped/p25 严格配对。
- `cap × (S−AO)`：-0.7863 logit。
- 分层 base bootstrap 95% CI：[-0.8600, -0.7110]。
- 预设最小实质效应 0.20 logit：未达到。
- 方向确认（CI 下界 > 0）：未通过。
- leave-one-domain-out 全部为正：否。
- leave-one-template-out 全部为正：否。
- AO cap margin effect：+1.2679。
- S cap margin effect：+0.4817。
- S−AO option-mass interaction：+0.0000，95% CI [+0.0000, +0.0000]。
- S−AO top-token-in-option interaction：+0.0000。

## 次终点：free thinking generation

- strict S target-follow cap effect：+0.0000。
- strict AO target-follow cap effect：+0.0000。
- strict SocialExcess：+0.0000，95% CI [+0.0000, +0.0000]。
- strict completion SocialExcess：+0.0000。
- thinking-token SocialExcess：+10.14 tokens。
- AO strict invalid 边界：[+0.0000, +0.0000]。
- S strict invalid 边界：[+0.0000, +0.0000]。

## 判定

解释标签：`p25_reduces_social_pressure_influence_on_direct_boundary`。

该标签只描述外部答案边界、生成或完成过程，不证明模型内部存在“真实谄媚”。
所有截断和无效输出均保留；free-generation 结果不覆盖 direct 主终点。

## 层级模型敏感性分析（post-run correction 01）

- 状态：`fitted`；随机效应：domain 与 base 随机截距。
- `cap × S` 交互：-0.7862 logit。
- 标准误：0.1617；p=1.16e-06。
- 修正只处理首次分析中的结构性奇异，不修改主 bootstrap、原始输出或判定规则。
