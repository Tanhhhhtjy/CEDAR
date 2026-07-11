# M2.1 persona 模板开发日志

本日志只记录聚合诊断，不记录 HarmBench 题目或模型原始输出。

## 数据隔离说明

- 模板开发只允许使用 HarmBench validation 的 41 条 standard behavior。
- test 的 159 条 behavior 不得运行，模板冻结后才可显式解封。
- 2026-07-11 在核对上游 CSV 字段时，仓库文本搜索被动打印了少量 test 输入文本；没有运行
  test、没有查看任何 test 生成或分类结果。模板保持类别级统一，不依据这些输入逐题修改。该事件也
  写入每次运行 manifest。

## v1 初始诊断

- 状态：已废弃，仅用于 validation 开发。
- 样本：validation 中按六个语义类别各取 1 条，共 6 条。
- 条件：behavior-only、persona uncapped、persona+p25；三个固定采样种子及 greedy。
- sampled ASR：behavior-only `0.000`，persona uncapped `0.111`，persona+p25 `0.000`。
- persona uncapped 拒答短语启发式命中率：`0.889`。
- 结论：攻击远低于 50% 有效性门槛，使用协议允许的唯一一次模板修订。

## v2 修订

修订仅基于上述 validation 聚合诊断：

- 加入固定的 persona 建立回合（user priming + assistant acknowledgement）；
- 将互动明确为 persona 与可信合作方的稳定工作关系；
- 最终请求要求直接进入内容，避免前言、警告、拒答或转题；
- 六个语义类别仍只改变角色身份描述，结构完全一致。

v2 在完整 validation 运行前保持 `development`。完整 validation 后不再允许第二次模板修订：若攻击
仍不满足有效性门槛，则 M2.1 应报告“persona 攻击基线无效”，而不是继续调参。
