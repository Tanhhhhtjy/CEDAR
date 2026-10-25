# C1/V7.1 零 GPU 完成过程补充分析

本报告只复算冻结的 V7/V7.1 输出，没有加载模型、使用 GPU 或运行 capping。

## 核心结果

- no-cue S 的 48 条 ITT：target 由 23/48 变为 26/48；completion 由 42/48 变为 48/48。
- 由 uncapped 完成状态预先定义的 42 条 sensitivity：target 由 23/42 变为 21/42，paired delta -4.76pp；initial→target 7 条，target→initial 9 条，exact p=0.8036。
- no-cue S 的截断识别边界为 [-6.25pp, +6.25pp]；cue S 为 [+10.42pp, +18.75pp]。
- no-cue observed response length：N -19.5，AO -24.4，S -333.4 tokens。
- no-cue S 的 actual answer-boundary margin 只在两侧均可用的记录中变化均值为 -0.299 logit。

## 解释边界

48 条表面 +6.25pp 同时包含 6 条 uncapped 截断恢复；在 42 条基线完整记录中，净迁移方向反而是 7 条 initial→target 对 9 条 target→initial。因而不能把全部 invalid→target 直接解释成更顺从。
2048-token 截断是右删失；长度均值是观察长度，不是潜在完整思考长度。exposure 模型是相关分析，不定位导致翻转的 token，也不提供因果中介结论。

完整 strict/lenient 迁移、bootstrap、长度、margin、exposure 和六条截断审计见同目录机器可读文件。
