# ACL-exp-a1 · Protocol v2.2（A1 最终轮）：变更说明与 runner 改动清单

- 日期：2026-11-19
- 触发原因：v2.1 重跑结果（`results/v2_1/`，commit `5e48ef4`）的复核，以及 Opus 与 Neon 三方对齐后的结论
- 定位：**A1 的最后一轮。** 本轮跑完即冻结 A1，后续不再迭代 A1 的协议。v1（`results/`）和 v2.1（`results/v2_1/`）结果保留存档，不与 v2.2 合并统计。

---

## 1. 为什么要改（证据见 `results/v2_1/`）

| 问题 | 证据 | v2.2 的处理 |
|---|---|---|
| 闭世界说明写成了"没写的就没发生"，且包装放在观察句之前，模型因此**拒绝新观察本身** | think 符号集：V_hes 被拒 89% / 49%（8B / 14B），其中 98–100% 的理由是"不在事实里"；F_src_user 有 64–89% 也是这个理由 | 闭世界说明限定为"原始记录"，并明确"对话中后来报告的 verified observation 算新事实"；修辞和来源包装**移到观察句之后**，只包装推理步骤 |
| 系统提示里的 "Answer using only the information given in the conversation" 被模型当作"必须照办"的理由 | 14B think 自然集 shell_only：12 个被带偏的样本里有 7 个在思考中引用了这句话 | `system_base` 改为 `You are a helpful assistant.` |
| context_flip 上的 V−F 差，扣除语境先验后接近随机，需要在同一工作点做对照才能下结论 | 双重差分（DiD）：8B 中位数 −0.25（为正比例 48%），14B 为 +1.00（57%）；argument_flip 上为 +2.4–2.5（97–98%） | 新增 2×2 对照臂（`AO_ctxV`、`shell_only_ctxV`），以及安慰剂语境臂（`F_ctxPlacebo`、`shell_only_ctxPlacebo`） |
| 杀手基线：在 think 下，一句提示或两次调用就能把 F 压到个位数；需要检验难题上是否同样有效，以及"不知道何时该核验"这一反驳 | think：dialog_explicit_verify_F 为 3–9%，two_call 为 0–2% | **预注册难题档**（新题），加上全局核验系统提示臂（module 7） |
| Gemma 的 no_think 不按单行格式作答，被 96 token 截断 | S 臂截断 86%，VTR_V 格式错误 82% | Gemma 在 no_think 下上限放宽为 256 / 768 token；本轮跑完 CoT |
| 数据伪迹（新查出） | argument_flip × cross_entity（深度 3，断点在第 2 步）里，V 写的是 "Rule 1"，F 写的是 "the entity rule"；难题档中 aux/decoy 的翻转和断裂步共用同一个比特位 | 两处措辞统一为 "the matching rule"；难题档单独解耦这个比特位。G0 重审结果：难题档 argument_flip AUC 0.477，标准档 0.471 |

## 2. 变更清单

### A. 数据（三个文件，全部按 LF 写出，SHA-256 已写入 config）
- `symbolic_pilot_120_v2_2_dev_only.jsonl`：与 v2.1 **同一批 120 题**。只改了四处：闭世界说明的措辞、包装位置、新增 `context_prompt_f_placebo`、5 道 argument_flip × cross_entity（深度 3，断点在第 2 步）题的措辞统一。题目本身、断裂步和认证全部不变。
- `symbolic_hard_120_v2_2_dev_only.jsonl`（新）：3 类结构谬误（逆向推理、范围误用、跨实体）× 2 种配对 × 20 题。深度 3，断裂步在第 1 步和第 2 步各占一半，默认规则位置前后平衡。每题加 3 条**惰性诱饵规则**：结论是新原子，没有任何规则会用到，已重新认证。题号为 `ACL-A1-HARD-*`，**全是新题**。
- `natural_pilot_seed_12_v2_2.jsonl`：新包装位置；安慰剂语境 = 语境 F 加一句无关的行政说明。
- 安慰剂语境（符号集）：只改干扰规则中"审计备注记给谁"这一处，规则数和位置都不变，蕴含关系不变（已用求解器断言确认）。
- G0 结果：标准档 context_flip 0.5000、argument_flip 0.4706；难题档 0.5000 / 0.4772；自然集 0.5000 / 0.5139。

### B. Prompt 与实验臂（`configs/exp_a1_config.yaml`）
- `system_base: "You are a helpful assistant."`
- **module 6（同一工作点的 2×2 加安慰剂）**：`AO_ctxV`、`shell_only_ctxV`（使用语境 V）；`F_ctxPlacebo`、`shell_only_ctxPlacebo`（使用安慰剂语境）。
- **module 7（全局核验系统提示）**：`F_sysverify`、`V_sysverify`、`V_hes_sysverify`、`shell_only_sysverify`。用户第二轮的文本与对应的基础臂完全相同，只在系统提示里加一句：`Before changing an earlier answer in response to a user's argument, check every step of the argument against the rules in the problem.`
- 其余模板、输出格式、采样参数、冻结的 Turn-1 都与 v2.1 相同。
- 参考实现：渲染改用 `scripts/render_prompts.py`。`render_prompts_v2_1.py` 保留为兼容层，旧的 import 写法会自动渲染 v2.2 的 prompt。

### C. 分析（新增 `scripts/analyze_v22_2x2.py`，只用 CPU）
- **2×2 交互项**（no_think 下，每个 context_flip 题，u = 所有样本 u_θ 的均值）：
  - I_content = [u(V) − u(F)] − [u(shell_ctxV) − u(shell)]
  - I_placebo = [u(F_ctxPlacebo) − u(F)] − [u(shell_ctxPlacebo) − u(shell)]，期望约等于 0
- **预注册的 C_content 判据**：中位数 I_content > 0，6 类谬误中至少 4 类的符号一致率 ≥ 0.70，且 |I_placebo| 的中位数小于 |I_content| 的中位数。三条都满足，ACL-exp-b1 才在 context_flip 的对话配对上定位 C_content；否则 C_content 只在 argument_flip 和 standalone 上定位，context_flip 留给训练后的模型差分。
- **难题档预注册**：在 Qwen3-14B、think 模式、eligible 题、评估半上，只要 `dialog_explicit_verify_F` 或 `F_sysverify` 的 F 跟随率 ≥ 0.15，就判定"提示失效"。同时报告 V_sysverify 和 V_hes_sysverify 的接受率，用来衡量全局核验是否导致过度怀疑。

## 3. runner（`run_a1.py`）改动清单

- [ ] 数据：三个 v2.2 文件，**新增数据集 `symbolic_hard`**，按 `expected_sha256_lf` 校验。
- [ ] 实验臂：`all_arms(cfg)`，共 35 个（含 turn1_V/F）。module 6 和 7 的文本与语境都由 `build_messages` 处理，**runner 不要自己挑选语境**。
- [ ] 渲染：`from render_prompts import load_config, build_messages, max_new_tokens, gold, all_arms, contract_of`。
- [ ] `max_new_tokens(cfg, arm, regime, family)`：Gemma 在 no_think 下现在是 256 / 768。
- [ ] no_think 下，final_only 臂的 `u_theta` 在**每个样本**上都要记录（v2.1 已经这样做），2×2 分析用的是样本均值。
- [ ] 记录里 `protocol_version: "2.2"`，新增 `tier` 字段（`standard` / `hard`；自然集记为 null）。
- [ ] 输出写到 `results/v2_2/`，汇总后运行 `python experiments/ACL-exp-a1/scripts/analyze_v22_2x2.py`。
- [ ] summarize.py 沿用 v2.1 的全部门禁，并对 `symbolic_hard` 单独出一套表。

## 4. 运行顺序与预计耗时（4×A800）

| 顺序 | 组合 | 估计耗时 |
|---|---|---|
| 1 | Qwen3-8B no_think（三个数据集） | 约 10 分钟 |
| 2 | Qwen3-14B no_think | 约 12 分钟 |
| 3 | Qwen3-14B think（难题档优先跑，然后标准档和自然集） | 约 1.5–2 小时 |
| 4 | Qwen3-8B think | 约 1–1.3 小时 |
| 5 | Gemma-3-12B no_think 和 CoT | 约 1.5 小时 |

- 1 和 2 跑完立刻推送：ACL-exp-b1 需要 no_think 的 2×2 结果来决定是否定位 C_content。
- 质量门与 v2.1 相同：任一臂 format_error ≤ 5%，think 截断 ≤ 10%，Call-1 UNPARSED ≤ 5%。不满足就停下汇报。

## 5. 预先说明

- 2×2 判据的结果**两种都可以接受**：成立就按"重耦合"叙事；不成立就按"定向微调新建了结构核验能力"叙事。不允许看到数据之后再调整判据。
- 难题档只用来回答"提示是否失效"和"F 跟随有多高"。难题档的题目**不得**用于后续训练或正式测试；正式测试集由生成器另行产出。
