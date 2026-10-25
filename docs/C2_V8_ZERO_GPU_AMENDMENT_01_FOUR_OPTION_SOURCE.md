# C2/V8 零 GPU Amendment 01：官方四选项与重复结构

状态：pre-data、pre-GPU 修正  
日期：2026-07-22  
触发原因：核验官方 SycoBench-600 后发现草案与来源的静态不一致

本修正发生在任何 C2/V8 数据构建、Qwen forward、自由生成或 p25 运行之前。Step 1 已
独立冻结；本修正不参考任何 C2 模型结果。

## 1. 官方来源冻结

- 论文：`SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants`
- ACL Anthology：<https://aclanthology.org/2026.findings-acl.1759/>
- DOI：`10.18653/v1/2026.findings-acl.1759`
- 官方仓库：<https://github.com/debu-sinha/sycobench-600>
- 固定 commit：`3b2d3680b0a6d456e618206e6281904c87cb49dd`
- release 对照：`v1.0.1` 的数据文件与固定 commit 字节相同
- `data/questions.json` Git blob：`0d3e8d6760b27ba362b2b8b837c075baa86e1c06`
- 原始字节：`287224`
- 原始 SHA-256：`ae41f4b94ac46fb0109778546a091534208881cf74eea87c43ce24736bb8dd4d`
- 数据/原始日志许可证：CC BY 4.0；代码许可证：MIT

来源为 600 条英文 MCQ、8 个 domain 各 75 条、easy/medium/hard 为 120/240/240。

## 2. A/B/C 改为 A/B/C/D

官方 600 条全部为 A/B/C/D 四选一。草案中的 A/B/C 是来源核验前的错误假设。为了不
删除官方 distractor、不改变题目难度，C2/V8 所有以下位置统一改为 A/B/C/D：

- JSON contract；
- direct candidate logits、conditional mass 与 argmax；
- initial/target/other_wrong 分类；
- answer-position 字段、smoke 与 validator；
- free-generation strict/lenient parser。

固定 Qwen tokenizer/revision 下，精确 scaffold 后的标签 token ID 为：

```text
A=32, B=33, C=34, D=35
```

四个标签均为单 token。主终点 `logit(target)-logit(initial)`、AO/S 配对和统计阈值不变。

## 3. wrong target 使用官方规则

官方 `sycobench/prompts.py::deterministic_wrong_letter` 明确定义：按 A、B、C、D 顺序选择
第一个不等于 gold 的标签。C2 遵守“官方已给 target 时直接使用”的原协议，不按模型偏好
或哈希重新挑 target。

因此冻结分布为：

```text
target A: 437
target B: 163
target C: 0
target D: 0
```

这不是构建失败，但必须在 manifest 和摘要披露。`target_position` 只在实际可用水平上作为
协变量/分层描述；不得声称四个 target 位置均衡。AO/S、cap 条件均在同一 base 内共享
target，因此主差中差仍严格配对。

## 4. 保留 600 条并增加重复结构敏感性

官方数据静态审计得到：

```text
600 unique source IDs
272 normalized question stems
542 unique normalized (question, ordered options) combinations
0 items with duplicate option text
```

重复是官方发布结构。不得删题或只留去重子集作为主分析。主分析仍使用协议冻结的
domain x difficulty stratified base bootstrap。另预注册两个不替代主分析的敏感性：

1. 以 normalized stem ID 为 cluster 的 bootstrap；
2. 以 exact normalized question+ordered-options ID 为 cluster 的 bootstrap。

如果 base bootstrap 与重复结构 cluster sensitivity 方向不一致，结论必须降级并明确说明
来源重复加权影响。

## 5. 不变项

不改变 600 条来源记录、gold、选项顺序、domain、difficulty、AO/S 构念、240 条 free 子集
冻结原则、2,400 次 direct、960 次 free、p25 配置、bootstrap 次数/seed 或停止规则。
人工双审仍是 GPU 前硬闸。
