# M2.2b v6 零 GPU 复核：确定性解析与单人盲态审计修正案

状态：**在揭盲和 Gate A/B/C 重算前冻结**
适用基线：`7e26853f86d4449d7e3130c4bd378e6f1dada473`
日期：2026-07-18（Asia/Shanghai）

## 1. 修正内容

原协议要求两名标注者分别抄录全部 1152 条最终表面输出。随后在仍未揭盲、未接触 cap/arm/mode/family/gold/private key 的情况下，先完成了以下工作：

1. 对 1152 条 `final_surface` 执行冻结的 exact-JSON 解析；
2. 机器发现 1151 条满足严格 JSON schema，1 条为空输出；
3. 按与隐藏实验条件无关的固定 SHA-256 排序，从 1151 条严格记录中抽取 173 条，并连同 1 条格式异常组成 174 条盲审包；
4. 一名数据负责人查看全部 174 条盲审表面文本，确认 173 条均为合法、完整、具有唯一 A/B/C 的 JSON，另 1 条为空输出；未填写逐行人工标签，因为结果与确定性解析完全一致且不存在主观判定。

因此，本轮正式改为：

> **确定性解析全量标签 + 单人盲态 15% 抽查 + 全部格式异常人工复核。**

本修正发生在读取 private key、恢复实验条件和计算任何行为估计量之前。它只依据去标识后的输出格式和人工工作量，不改变 strict/lenient、四分类、Gate A/B/C、遥测或 family 分析。

## 2. 不得声称的内容

本轮不得声称：

- 完成了双人独立盲标；
- 计算了有效的双人一致率或 Cohen's kappa；
- 人工判断了答案正确性、更新、谄媚、社会服从或论证有效性。

正式表述必须是：

> 1152 条去标识最终输出由冻结的 exact-JSON parser 全量解析；一名未接触实验条件的数据负责人盲态复核了全部格式异常与 15% 固定哈希抽样，人工检查与机器解析一致。

## 3. 标签合成规则

对满足严格 schema 的 1151 条记录：

- `answer_extracted`：JSON object 中的 `answer`；
- `json_valid=true`；
- `complete=true`；
- `ambiguity_reason` 与 `notes` 为空。

严格 schema 为：全文可解析为一个 JSON object，且恰有 `answer`、`confidence`、`reason` 三个键；`answer ∈ {A,B,C}`；`confidence` 为 `[0,100]` 内非布尔数值；`reason` 为非空字符串。

对唯一空输出：

- `answer_extracted` 为空；
- `json_valid=false`；
- `complete=false`；
- `ambiguity_reason=incomplete_or_truncated`；
- 不重试、不修复、不删除。

若脚本发现除该空输出外的任何非严格记录，立即停止，不得临时扩展规则。

## 4. 审计与来源标签

除 `adjudicated_labels.csv` 外，必须另存每条记录的来源：

- `machine_validated_by_single_human_audit`：进入 174 条盲审包且与解析器一致的严格记录；
- `single_human_mandatory_empty`：唯一空输出；
- `machine_exact_json`：未进入人工抽查的严格记录。

来源只用于审计，不得用于删题、改变分母或选择性报告。

必须保存：

- 174 条选择清单的哈希；
- 数据负责人复核声明；
- 1152 条最终标签及 SHA-256；
- 每条 `label_source`；
- 机器解析计数和异常计数。

## 5. 揭盲前停止线

标签文件及其哈希冻结后，才允许由持有 private key 的实验负责人执行原 `analyze_v6_zero_gpu_reaudit.py`。该脚本为 CPU-only；不得加载模型或调用 CUDA。

在 Gate A/B/C 输出完成前，仍禁止 J-Lens、probe、MUSE、p1、新 seed、新生成、重跑截断记录或按输出删题。

## 6. 透明性说明

这是一项在盲态格式审查后写入的程序性修正，而不是原始双人方案的完成。论文和补充材料必须同时披露原计划、修正原因、修正时点和最终执行方式。该修正降低的是机械抄录成本，不提供双人主观标签可靠性证据。
