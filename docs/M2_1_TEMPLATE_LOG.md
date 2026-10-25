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

## v2 完整 validation 与收口

- 完整性：41 条 validation behavior，3 个采样种子和 1 次 greedy，共 492 条生成与分类。
- sampled ASR：behavior-only `0.0325`，persona uncapped `0.2846`，persona+p25 `0.1951`。
- persona lift：`25.20pp`，低于 `30pp` 门槛；persona uncapped ASR 低于 `50%` 门槛。
- p25 sampled 相对下降 `31.43%`，低于 `40%` 门槛；绝对下降 `8.94pp`，paired-bootstrap
  95% CI 为 `[2.44pp, 16.26pp]`。
- greedy ASR：persona uncapped `0.1220`，persona+p25 `0.1707`，方向反转。
- 决策：persona 保持 `development`，不运行 HarmBench test，不允许第二次模板修订。

该运行只能解释为公开 HarmBench 上的代理诊断，不能解释为对原论文 persona-jailbreak
定量实验的严格复现，原因有二：

1. 原论文使用 Shah et al. persona modulation 数据的 1,100 个 system-prompt/behavior 组合；
   精确 persona prompts 被原作者基于 responsible disclosure 主动隐去，公开仓库未提供逐条数据。
2. 公开的 Qwen jailbreak case study 含 15 个用户回合，体现多轮 persona 累积；v2 仅包含一次
   persona 建立交互后接最终请求，未复现该多轮注入过程。需同时说明，论文的 1,100 条定量评测
   被描述为 system prompt + behavioral question pairs，不能把 15 轮案例外推为全部定量样本。

因此本结果标记为“代理攻击基线无效”，不用于否定原论文的 capping 结论。项目随后独立推进
M2.2，以评估已公开官方 capping 干预本身的认识论代价。
