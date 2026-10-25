# E1 Token-resolved resistance geometry 摘要

- 状态：`passed`
- 性质：development、uncapped、monitor-only observational geometry
- 干预：未运行 p25/p1/steering/patching/J-Lens/MUSE/probe
- greedy bridge：144/144 逐字一致
- E1.2：720/720
- resist / yield：539 / 173
- mixed prompts：17；minority>=3：11
- 数据充分性闸门：`passed`
- 分析 provenance：`disclosed_postrun_analysis_correction`

## 同题主比较

preanswer_32 / layers50–53 的 resist−yield 修正量：-1.145681189231458，95% CI [-3.2937666183639323, 1.2635139331811027]。
同位置 human-readable Assistant Axis 投影差：1.14659117833139，95% CI [-1.2634321721384336, 3.294637397299948]。

## 直接交互

阶段交互 D_preanswer−D_prefill：-1.1456811892314587，95% CI [-3.2937666183639327, 1.2635139331811025]；正向闸门=False。
层组交互 D_50–53−D_46–49：-1.8354155374435441，95% CI [-3.2113415967910126, -0.4483731224444441]；正向闸门=False。

## F 控制

记录数：360；描述性行为闸门：`failed_insufficient_variation`。
preanswer_32 / layers50–53 的 resist−yield 修正量：1.5979780651274176，95% CI [-3.448812188042536, 9.111178994960891]。
同位置 Assistant Axis 投影差：-1.5944900229336723，95% CI [-9.111924975778532, 3.4509850341233586]。

## 解释边界

同题分析可估计，但预注册的阶段或层组直接交互未同时过闸；不得称为已定位的后层 pre-answer 抵抗状态。

行为计数详见 `behavior_counts.json`：{"bridge": {"resist": 93, "yield": 45, "other_wrong": 5, "invalid_or_truncated": 1}, "control-f": {"resist": 307, "other_wrong": 23, "yield": 29, "invalid_or_truncated": 1}, "sample-s": {"resist": 539, "yield": 173, "invalid_or_truncated": 8}, "smoke": {"resist": 7, "yield": 3, "not_applicable": 2}}

E1到此按协议停止，不自动进入E2或任何干预/机制工具。
