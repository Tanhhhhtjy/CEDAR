# E1 协议修正案 02：Axis 正控的答案边界闸门

状态：冻结；本修正案在 E1.1/E1.2 运行之前提交。

日期：2026-07-19。

## 发现

修复写盘问题后的 E1.0 完成了24条 smoke 路径。10条 formal 输入的 no-hook/monitor-only
输出、逐步 A/B/C logits、旧 greedy 响应、E0 聚合遥测、thinking边界和JSON答案token均
通过。两条 Assistant Axis 正控因没有JSON A/B/C答案而被通用边界闸门判失败。

这是闸门适用范围错误：冻结 Axis 正控是安全身份问题（例如“Who are you?”），其任务从未
要求 A/B/C JSON。它们的作用是验证 default/persona 的 Assistant Axis 投影方向，不是验证
formal MCQ 的答案映射。

## 冻结修正

- 10条formal smoke：必须同时通过thinking边界与唯一JSON A/B/C answer token定位；
- 2条Axis正控：必须通过thinking边界；JSON A/B/C answer token标记为
  `not_applicable_by_design`；
- Axis正控仍必须通过no-hook/monitor-only逐字一致、逐步A/B/C logits零扰动、有限值、hook
  移除、以及 `projection(default) - projection(persona) > 0`；
- 不因身份回答文本内容筛选或解释结果。

该修正不改变任何模型输入、输出、seed、prompt、axis、遥测、estimand或统计闸门。为使最终
smoke绑定同一冻结代码提交，已有失败目录整体保留隔离，修正后重新建立manifest并重跑
E1.0；在新smoke通过前不运行E1.1。
