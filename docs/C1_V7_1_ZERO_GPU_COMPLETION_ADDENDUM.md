# C1/V7.1 零 GPU 完成过程补充分析

状态：冻结分析协议  
日期：2026-07-22  
性质：已有结果的零 GPU 补充分析，不产生新模型输出

## 1. 目的

本补充分析把 V7.1 无提示 S 的答案变化拆成两个互不替代的终点：

1. answer-policy effect：在两条件都有有效答案时，p25 是否改变 initial/target 选择；
2. completion/deliberation effect：p25 是否改变 thinking 长度、终止和有效输出。

`invalid_or_truncated -> target` 不会同时被当作纯答案翻转和纯完成恢复。48 条 ITT
仍是总体描述；由 uncapped 完成状态定义的 42 条集合只作 paired sensitivity。

## 2. 不可变输入

- V7 cue commit：`1336f9dda660876dcfccb39842569eb93946c8f3`
- V7.1 no-cue commit：`deb4932318b11906c1008849e8c97ec960fb1d90`
- bootstrap：10,000 次 family-cluster resampling
- bootstrap seed：`20260722`
- 类别顺序：`initial / target / other_wrong / invalid_or_truncated`

分析器必须使用 `git show <commit>:<path>` 直接读取上述提交中的结果、runtime 和
audit，不能读取工作区内可变副本。每个输入的 Git blob ID、原始内容 SHA-256 和字节数
写入 provenance。

## 3. 输入硬审计

运行分析前必须同时满足：

- cue 与 no-cue 的跨条件 audit 均为 `passed`，content retry 为 0；
- cue 的 N/AO/S 与 no-cue 144 条在 base、family、arm、答案映射上一一对应；
- no-cue prompt 相对 cue prompt 只删除
  ` No new task information is supplied.`；
- results 与 `record_runtime.jsonl` 的完整 result 对象逐条相等；
- uncapped 的 `cap_stats` 全为 null，p25 全为非空对象；
- 每层 46--53 的 prefill count 等于 prompt tokens，decode count 等于
  response tokens 减 1；
- no-cue 两条件均为 144/144，四类别互斥且穷尽。

任一检查失败即停止，不生成部分分析。

## 4. 冻结分析

### 4.1 四状态迁移

对 `cue/no-cue x N/AO/S x strict/lenient` 输出完整 4x4 uncapped 到 p25
迁移矩阵，同时报告 target rate 与 completion rate 的配对差及 family bootstrap 置信区间。

### 4.2 42 条 complete-case sensitivity

集合仅由 no-cue S uncapped strict category 为 initial 或 target 定义。冻结复算闸为：

- 集合 42 条；
- uncapped target 23 条；
- p25 target 21 条；
- initial 到 target 7 条；
- target 到 initial 9 条。

报告 exact two-sided McNemar/binomial discordant-pair test、family bootstrap CI 和
leave-one-family-out 范围。该结果不得替代 48 条 ITT。

### 4.3 截断识别边界

对 cue/no-cue S 计算：

```text
lower = P(target | p25) - P(target or invalid | uncapped)
upper = P(target | p25) - P(target | uncapped)
```

同时报告 cue x cap 的区间算术边界及上下界各自的 family bootstrap CI。六条 no-cue
uncapped 截断只保留尾部 300 字符和事实性说明，不猜测最终答案。

### 4.4 thinking、完成和长度

使用冻结 runtime 的 response tokens 和 boundary thinking tokens。仅当存在有效闭合
thinking 边界时，final tokens 定义为 close-token 后剩余 token 数。未闭合的 2048-token
输出按右删失处理；均值差只作 observed-length 描述。

输出各 cell 的均值、中位数、IQR、P90、最大值、截断率、无效率，以及配对长度差和
`S-AO`、`S-N` 差中差。

### 4.5 exposure 与长度

主预测变量只用 uncapped trajectory 的 prefill counterfactual p25 exposure。主模型为
各 cue context 分开的：

```text
LengthDelta ~ prefill_exposure * arm
```

标准误按 family 聚类。decode correction sum、p25 actual correction 和长度的关系只作为
描述性或中介线索。报告 early 46--49、late 50--53、迁移类别、排除六条截断后的敏感性。

### 4.6 连续 margin

actual/scaffold `target - initial` margin 仅在 uncapped 与 p25 两侧均存在有限值时配对。
它属于 answer-policy 描述，不消除对完成过程的单独报告。

## 5. 交付与停止

交付目录：`artifacts/m2/c1_v7_1_zero_gpu_addendum/`。包含汇总、机器可读 JSON/CSV、
两张确定性 SVG 和所有文件的 SHA-256 manifest。输出不得含运行时间戳，以支持重复运行
哈希一致。

完成、测试并提交本补充分析后才可开始 C2/V8 来源固定。此步骤不授权 GPU、p1、LoFa、
J-Lens、MUSE、probe、patching 或任何新生成。
