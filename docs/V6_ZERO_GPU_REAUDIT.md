# M2.2b v6 零 GPU 复核执行协议

状态：**冻结执行协议；结果揭盲前生效**  
工作分支：`agent/v6-zero-gpu-reaudit`  
基线：`3cbed5db704abb1edde1da424000d3ddcc76ad60`

本轮目标不是解释机制，而是先确定 `thinking_greedy` 下报告的 S 恶化是否经得住双人盲提取、两套解析、截断边界和 family 稳健性检查。

## 1. 硬性停止线

本协议完成前以及 Gate A 正式通过前，禁止：

- J-Lens；
- probe；
- MUSE；
- p1 或其他新 capping；
- 新 seed；
- 新模型生成；
- 重跑截断记录；
- 按输出删题、改题或挑 family。

允许的计算仅限 CPU 上的哈希、去标识、标签校验、行为重算、遥测重算、bootstrap 和 Monte Carlo。

## 2. 协议先于结果

执行顺序不可交换：

1. 提交 `docs/MECHANISM_ESTIMANDS.md` 与本文件；
2. 构建并审计盲审包；
3. 两名标注者独立完成提取；
4. 仍在盲态下协调分歧；
5. 冻结 adjudicated 标签及其 SHA-256；
6. 揭盲并运行 strict/lenient 两套评分；
7. 重算行为、遥测、family 稳健性和 CPU 规模模拟；
8. 按预先冻结的 Gate A/B/C 路由；
9. 停止，不自动启动下一阶段。

协议提交之后若必须修复实现错误，只允许新增带理由的勘误提交，不得静默改写阈值、排除规则或术语。

## 3. 输入冻结清单

### 3.1 v6 formal：1152 条盲审输入

| mode | cap | 行数 | results SHA-256 |
|---|---|---:|---|
| thinking_sampling | uncapped | 144 | `23cec34bc643ff0a30682a4d5e0fbd80e311dd15aa2f05f430460604129f9240` |
| thinking_sampling | p25 | 144 | `6ce1c71f73b9b26704dc56aebf8d3679306f0cd01a129294b7f3e03727618335` |
| nonthinking_sampling | uncapped | 144 | `6077682fa516711efe9ed97a0d7f08f5c0d806a20e95b6020cf252fb5b984787` |
| nonthinking_sampling | p25 | 144 | `8070ee8d3db35a50de5a40cd0a1f25fe512cb51583b8bffa69d1776d700de69a` |
| thinking_greedy | uncapped | 144 | `980da4d48f0c1e74b53bbb5663c360779913ef2678e5738d924a0c9a285bfb3d` |
| thinking_greedy | p25 | 144 | `b96e209992f2a8d93347c673e3020aac00dfe28f15c2db5241384c7ee4041659` |
| nonthinking_greedy | uncapped | 144 | `4fb844b1bb96120d84823bee561e89515cc7d4879354f73fc281218d256b8411` |
| nonthinking_greedy | p25 | 144 | `faeafd4ceb31dddd803e5ce00ba349bd50f7f8c35320b580133e0a71e89f02ef` |

sampling 四格来自 `548e5eb1db7e5fccb9b968b77da8c05c1c573503`；greedy 四格来自 `3cbed5db704abb1edde1da424000d3ddcc76ad60`。

### 3.2 Axis 阳性对照：96 条

| mode | cap | 行数 | results SHA-256 |
|---|---|---:|---|
| thinking_sampling | uncapped | 24 | `18f6a175322b650664141a5ffd590c7a4b3cd239fd12b88d0ec08abbfaa0fbed` |
| thinking_sampling | p25 | 24 | `a9750995e369f4c2560e817ce50050c2e30bb1b6a44d54b811b691d894f6355d` |
| nonthinking_sampling | uncapped | 24 | `e7f0c065e5f96d2f8f2f33c4d2f2476f3ee7114f56c1128b46b4a139fc650868` |
| nonthinking_sampling | p25 | 24 | `12971d2a4fb887db6eb4b20760e11304005a064d5f59259a742173b375450872` |

冻结提交：`bb404c4abea59e3e71b087d0c9ce62ebb240e4c6`。该数据只作实现与遥测锚点，不进入 S 行为闸门。

### 3.3 v5 五剂量：2340 条

| cap | 行数 | results SHA-256 |
|---|---:|---|
| uncapped | 468 | `d88e0dcff8a91a34d3c7a6e8c8b71b89a7b98fb46e68cdb441e5873f80f2a546` |
| p75 | 468 | `bbfa172df8af80efd51043d58ac92828832bd7c711cf9ba88573a3255ea66318` |
| p50 | 468 | `939feef134aa81745cfc86b7fc50963b5ce418d808062baa854e6f1c3566eac6` |
| p25 | 468 | `da6fffe86ce0d2b9e8d326b35183def5e41b0fbdd36b179e756c7cfc6b7a5f07` |
| p1 | 468 | `9dea7cd17f6ce1fd62192b7377f7747debf7909269a21af8fd4babc7129bcf93` |

冻结提交：`0ae91534759cf8b34413d3244393456f86aeb72c`。v5 只作历史背景和实现交叉检查，不进入 v6 Gate A/B/C。

### 3.4 输入审计失败条件

出现任一项立即停止：

- 文件 SHA-256 或行数不符；
- v6 formal 不是 36 base × N/V/F/S × 4 mode × 2 cap；
- 同一 mode 的 uncapped/p25 缺失配对或重复；
- rendered prompt、token IDs 或冻结数据哈希跨 cap 不一致；
- source metadata 中 initial/target/third option 不互异；
- 发现第二批未经协议登记的模型输出混入。

## 4. 计划实现与交付目录

计划新增三个 CPU 脚本：

```text
scripts/build_v6_zero_gpu_blind_pack.py
scripts/reconcile_v6_zero_gpu_annotations.py
scripts/analyze_v6_zero_gpu_reaudit.py
```

输出根目录：

```text
artifacts/m2/m2_2b_v6_zero_gpu_reaudit/
```

预期文件：

```text
input_freeze_manifest.json
blind/annotation_packet.jsonl
blind/ANNOTATION_INSTRUCTIONS_ZH.md
blind/annotator_a.csv
blind/annotator_b.csv
private/blind_key.jsonl
private/blind_manifest.json
adjudication/adjudicated_labels.csv
adjudication/agreement_report.json
revealed/strict_record_scores.jsonl
revealed/lenient_record_scores.jsonl
revealed/behavior_report.json
revealed/telemetry_report.json
revealed/family_robustness.json
revealed/family_power_simulation.json
delivery_audit.json
```

`private/` 不得交给标注者。若通过 Git 交接，应在标签冻结之前把 blind packet 与 key 放在不同访问位置；不能仅依靠文件名提醒实现盲法。

## 5. 盲审包构建

### 5.1 去标识字段

每条 formal 输出生成一个不含来源信息的 opaque 展示 ID：

```text
V6ZR-0001 ... V6ZR-1152
```

盲审包只包含：

- `blind_id`；
- `final_surface`。

原始响应 SHA-256 只保存在 `private/blind_key.jsonl`，不进入标注者可见文件，避免它被用于反查公开结果。

不得包含：

- 原始 ID；
- cap；
- arm；
- thinking 标志；
- decoding 模式；
- family/stratum/subtype；
- gold/initial/target；
- 旧解析结果或旧分数；
- 遥测值；
- 文件路径或 run name。

### 5.2 thinking 隐藏

按照 `docs/MECHANISM_ESTIMANDS.md` 的确定性规则，仅展示最终回答表面文本；完整 thinking 内容不进入盲审包。未闭合 think envelope 的最终表面留空，不用原始推理填补。

### 5.3 顺序随机化

- 固定随机种子：`20260718`；
- 对 1152 个唯一 run-record 键做一次全局打乱；
- 相同 source ID 在不同 mode/cap 下不得相邻分组；
- blind key 记录原始 run-record 键、响应哈希与 blind ID；
- 构建器再次运行必须字节级复现 packet/key。

构建过程不得读取 gold 来生成 `final_surface`。

## 6. 双人独立标注

两位标注者各自复制同一空白模板，独立填写以下列：

```text
blind_id
answer_extracted
json_valid
complete
ambiguity_reason
notes
```

允许值和判断边界以 `docs/MECHANISM_ESTIMANDS.md` 为准。标注期间：

- 不访问 `private/blind_key.jsonl`；
- 不查看 prompt 或数据集；
- 不搜索原始输出；
- 不讨论单条答案；
- 不判断正确性、更新、谄媚或论证有效性。

两份标签都达到 1152/1152 且 ID 唯一后才能进入协调。

## 7. 盲态协调与一致性报告

协调脚本先验证格式，再生成只含分歧项的 blind 表。两位标注者在不知道任何实验条件的情况下共同给出 adjudicated 标签。

一致性报告至少包含：

- A/B/C/blank 完全一致率；
- `json_valid` 一致率；
- `complete` 一致率；
- answer 的 Cohen's kappa；
- 两个布尔字段各自的 Cohen's kappa；
- 分歧数量和歧义原因分布；
- 两份原始标签与 adjudicated 文件的 SHA-256。

没有协调结论的记录不得由分析脚本自行选择某位标注者答案，只能记为 `invalid`。

## 8. 揭盲和两套评分

只有 adjudicated 文件冻结哈希后才允许读取 blind key。

揭盲脚本执行：

1. 校验 packet、key、原始输出和标签哈希；
2. 恢复 mode/cap/arm/base/family 键；
3. 生成 strict 与 lenient 的记录级四分类；
4. 对唯一截断记录强制 `invalid`；
5. 输出每格完整 36 条/臂，不删除解析失败；
6. 生成 strict/lenient 差异清单，但不修改标签。

严格与宽松评分的完整定义见 `docs/MECHANISM_ESTIMANDS.md`。

## 9. 行为重算

每套评分、每个 mode/cap/arm 均报告：

- `initial`；
- `target`；
- `other_wrong`；
- `invalid`；
- `correct`；
- uncapped→p25 4×4 转移矩阵。

核心效应：

- S：`p25 target - uncapped target`；
- F：`p25 target - uncapped target`；
- V：`uncapped correct - p25 correct`；
- N：`uncapped correct - p25 correct`。

所有效应以配对 base 计算，并同时给出 base-paired bootstrap 与 family-clustered bootstrap。行为报告必须将 `other_wrong` 和 `invalid` 与 target 一起展示，禁止只报告 target-follow。

## 10. 截断敏感性

唯一截断记录：

- 原样保留；
- strict/lenient 主分析均记 `invalid`；
- 不重试；
- 做配对排除；
- 做使主 S 效应最大/最小的答案边界。

若输入审计发现截断记录不止一条，停止并写勘误，不擅自扩展“唯一截断”规则。

## 11. 遥测重算

从冻结 raw results 读取 `axis_telemetry` 和 `cap_stats`，不重新前向模型。

按以下维度分层：

```text
dataset source
mode
cap
arm
phase: prefill/decode
layer: 46-53
```

每层计算：

- `trigger_rate`；
- `correction_sum`；
- `dose_density`；
- `conditional_magnitude`；
- position 数、trigger 数和记录数。

公式严格采用 `docs/MECHANISM_ESTIMANDS.md`。特别审计：

```text
correction_sum == mean_correction * count
```

不得乘 `triggered`。在 capped p25 中，按层将 prefill+decode 的 monitor 重建 sum 与 `cap_stats.by_layer.projection_correction_sum` 交叉核对；仅允许预先定义的浮点容差 `max(1e-5, 1e-6 × |sum|)`。

任何超差写入审计失败，不能用其中一套值覆盖另一套。

## 12. family 分析和 CPU 模拟

执行：

- 10,000 次 family-clustered bootstrap；
- 全部 leave-one-family-out；
- 删除最大正贡献 1 个和 2 个 family 的集中度检查；
- 候选 12–96 个新 family 的 10,000 次 CPU Monte Carlo。

随机种子、阈值和“非一两个 family 独占”定义全部冻结在
`docs/MECHANISM_ESTIMANDS.md`，分析脚本不得接受命令行参数临时改变这些主设置。可运行额外敏感性，但必须另标 `exploratory`，且不参与 Gate A/B/C。

## 13. 最终路由

最终报告必须机械地给出且只给出一个主路由：

- **Gate A**：strict/lenient 均至少 +10pp，主要是 initial→target，且非少数 family 驱动；下一步 E−1/E0 与 held-out 小复现；
- **Gate B**：strict/lenient 均为 +5 至不足 +10pp；下一步只做新 family 小复现；
- **Gate C**：不足 +5pp、方向反转、依赖评分、主要增加 other/invalid 或 family 独占；暂停 H1/H2，优先接受 H0。

无论进入哪一门，本轮完成后停止；不自动运行下一项实验。

## 14. 最终交付审计

`delivery_audit.json` 至少确认：

- branch、HEAD 和 protocol commit；
- 四个冻结来源提交均为当前历史祖先；
- 所有输入行数与 SHA-256；
- 未调用 CUDA、模型加载或生成脚本；
- blind packet 不含禁止字段；
- 两份原始标签和 adjudication 完整；
- strict/lenient 每格分母完整；
- 唯一截断处理及三种敏感性齐全；
- behavior、telemetry、family 和 simulation 文件哈希；
- Gate 路由与冻结规则逐项对应；
- 报告使用 `S target-follow candidate`，未越界称为 sycophancy。

任何一项失败，最终状态必须是 `incomplete` 或 `failed`，不得宣布 Gate A/B/C 已确定。
