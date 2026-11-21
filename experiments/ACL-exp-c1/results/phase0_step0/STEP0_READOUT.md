# C1 Step 0 · Qwen3-8B 生成复查与 WikiText-2 KL

2026-11-22，任务版本 `eae6b33`，Phase 0 存档 `88182a2`。复查使用 M0 和四个 Recipe A、seed 42 的已合并 checkpoint。共保存 47 条 vLLM 生成、1 条 HF greedy 对照和四组 KL。样本来自公开能力基准和 C1 开发集。四组 KL 调用均显式传入空 `--datasets`。

## 1. IFEval 的长度、原文和约束

表内每格为“原批 → 本次”，依次列出结束原因、生成 token 数和官方 prompt-strict 评分。原批来自 Phase 0 缓存的实际生成及已发布逐题评分。本次使用 vLLM 0.11.0、BF16、greedy、seed 42、上限 3,072。抽检题按上限组成小批次。两次输出有差异，现有对照无法单独确定差异来自哪一项运行条件。

| 题号 | M0 | Targeted-Dual | Uniform-LoRA |
| --- | --- | --- | --- |
| 1000 | stop 412 通过 → stop 444 通过 | length 3072 失败 → length 3072 失败 | length 3072 失败 → length 3072 失败 |
| 1001 | stop 355 通过 → stop 324 通过 | length 3072 通过 → stop 38 通过 | length 3072 通过 → stop 139 通过 |
| 1122 | stop 46 通过 → stop 46 通过 | stop 18 失败 → stop 18 失败 | stop 18 失败 → stop 18 失败 |
| 32 | stop 70 通过 → stop 39 通过 | stop 38 通过 → stop 38 通过 | stop 35 通过 → stop 35 通过 |
| 24 | stop 352 失败 → stop 352 失败 | stop 206 通过 → stop 206 通过 | stop 196 失败 → stop 203 失败 |

**1000。** Targeted-Dual 原批反复写 Raymond III 是的黎波里最后一位战死的伯爵，同一完整句出现 200 次。本次反复写他是的黎波里最后一位生于十字军国家的伯爵，同一完整句出现 167 次。两次均以 3,072 token 截断，输出均无思考标签、无逗号，达到 300 词，缺少要求的三个 Markdown 高亮标题。本次 token 序列未生成 EOS。单题 HF greedy 也生成 3,072 token，解码文本与 Phase 0 缓存逐字一致。HF 与本次 vLLM 共享前 58 个生成 token，之后分歧。本次另外三组已合并权重均在此题生成 3,072 token，均未通过高亮标题约束。

**1001。** Targeted-Dual 原批回抄开头后，反复列出前往东京、京都、大阪的行程，其中 “I shall then travel to Tokyo again.” 出现 63 次。3,072 token 的文本中没有逗号和思考标签。该题的官方唯一约束是零逗号，因此通过。本次输出改写并重复了用户请求，共 38 token，以 EOS 结束，同样通过零逗号检查。此次未复现原批撞限。

**1122。** Targeted-Dual 两次均在 18 token 后停止，本次末尾为 EOS。两次文本相同，为 `#playboy #playboybros #playboygirls #playboymagazine`。文本含四个井号，所有字母均为小写。官方第一条约束同时检查英语和小写，语言检测返回 `tl`，故该条失败。第二条通过。该版本官方检查器收到非英文字母参数 `#` 时会随机选一个英文字母。按 Phase 0 的 seed 和原题顺序恢复评分时，实际检查的是至少四个 `o`。记录中分别保留了原始参数、实际参数和井号数。

**对照。** 32 的唯一计分约束是零逗号，三模型两次均正常停止并通过。24 要求英文全大写，M0 两次都写出混合大小写的 `SEeks`。Targeted-Dual 的十项清单全部大写，语言检测为 `en`，两次均通过。

## 2. C1 的六个 token 与真实截断

Uniform-LoRA 在 DEV-001 的 V、F 臂都生成 6 个 token。下表逐 token 解码，`\n` 表示一个换行，选项带一个前导空格。

| 位置 | V 的 ID / 解码 | F 的 ID / 解码 |
| --- | --- | --- |
| 1 | 151667 · `<think>` | 151667 · `<think>` |
| 2 | 198 · `\n` | 198 · `\n` |
| 3 | 19357 · `Final` | 19357 · `Final` |
| 4 | 25 · `:` | 25 · `:` |
| 5 | 356 · 空格 + `C` | 425 · 空格 + `B` |
| 6 | 151645 · `<\|im_end\|>` | 151645 · `<\|im_end\|>` |

V 的字母 C 是该臂的目标答案，F 的字母 B 是该臂的初始答案，两者均与 gold 相同。两个输出都缺少 `</think>`，第 6 个 token 即 EOS。后端均记录 `stop`，冻结的 C1 解析器均返回 `truncated`、无可计分选项。DEV-008/turn1_V 则确实用尽 3,072 token，Uniform-LoRA 在未闭合思考块里写出 767 个完整的 `Final: A`，末尾停在 `Final:`，后端记录 `length`。DEV-015/V、F 对 M0 和 Targeted-Dual 均可解析。Uniform-LoRA 在这两题上也以 6 token 和未闭合思考块结束。

下表为 think 复查。每格按“原批长度 → 本次长度，模型停止原因，解析选项”排列。横线表示解析器未给出选项。

| 题/臂 | M0 | Targeted-Dual | Uniform-LoRA |
| --- | --- | --- | --- |
| C1-DEV-001/V | 318 → 316, stop, C | 221 → 237, stop, C | 6 → 6, stop, — |
| C1-DEV-001/F | 450 → 450, stop, B | 293 → 289, stop, B | 6 → 6, stop, — |
| C1-DEV-008/turn1_V | 466 → 574, stop, A | 517 → 531, stop, A | 3072 → 3072, length, — |
| C1-DEV-015/V | 331 → 332, stop, A | 306 → 247, stop, A | 6 → 6, stop, — |
| C1-DEV-015/F | 391 → 361, stop, D | 217 → 179, stop, D | 6 → 6, stop, — |

**no_think 自由生成。** 上限 64，未预填 `Final:`。9/9 均恰好生成四个 token，形式为 `Final: X<|im_end|>`，9/9 解析成功。按照 DEV-015/V、DEV-015/F、DEV-042/shell_only 的顺序，gold 为 A/D/B。M0 生成 A/A/A（1/3 正确），Targeted-Dual 和 Uniform-LoRA 均生成 A/D/B（各 3/3 正确）。Phase 0 的 no_think 评估在预填字段后比较四个选项的 logits。

**GSM8K。** 下表为冻结提取器从思考结束后独立的 `#### 数值` 行读出的答案，列为“原批 → 本次（本次结束原因）”。横线表示未提取到数值。

| 题号 | 标准答案 | M0 | Targeted-Dual | Uniform-LoRA |
| --- | --- | --- | --- | --- |
| 0 | 18 | 18 → 18（stop） | — → —（stop） | — → 18（stop） |
| 4 | 20 | — → 20（stop） | 20 → 20（stop） | 20 → —（length） |

## 3. 配置与有效停止条件

五个模型的 `generation_config.json` 都列出 EOS `[151645, 151643]`。对实际 vLLM 请求取值，主 EOS 为 151645（`<|im_end|>`），附加停止 ID 为 151643（`<|endoftext|>`，同时用作 pad）。五者的有效停止集合一致。四个合并模型的生成配置文件彼此逐字节相同。与 M0 相比，其唯一有差异的 JSON 字段是 `transformers_version`，由 `4.51.0` 变为 `4.57.0`。

M0 的聊天模板内嵌于 `tokenizer_config.json`，合并模型从 `chat_template.jinja` 加载。两类 tokenizer 配置文件哈希不同。五者实际加载模板的 UTF-8 SHA-256 都是 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`。文件哈希、完整生成配置及实际请求的停止列表见 [config_eos_check.json](config_eos_check.json)。

## 4. WikiText-2 参考 KL

按冻结脚本计算逐 token 的 D_KL(M0 ∥ tuned)，使用自然对数。每段最多取前 256 个 token。200 段语料的 SHA-256 为 `97a3ed61a17e4023eabe9566fbe56768d0708aea97b662abac52dbb830315e05`。

| Checkpoint | 段数 | n_tokens | kl_mean | kl_median |
| --- | --- | --- | --- | --- |
| Targeted-Dual | 200 | 43,854 | 0.059854 | 0.043985 |
| Dual-wo-Hub | 200 | 43,854 | 0.061962 | 0.046024 |
| Random-LM | 200 | 43,854 | 0.056768 | 0.043804 |
| Uniform-LoRA | 200 | 43,854 | 0.129180 | 0.104696 |

**运行记录。** 以下均为 11 月 22 日北京时间、调度器选出的物理卡号。00:40:30 起，KL 在 0（Dual-wo-Hub）、1（Targeted-Dual）、2（Random-LM）、3（Uniform-LoRA）并行。00:41:28 起，vLLM 在 4（M0）、5（Uniform-LoRA）、6（Dual-wo-Hub）、7（Targeted-Dual）运行。Random-LM 等到卡 0 释放后于 00:43:28 开始，HF 单题对照于 00:43:54 使用卡 1。所有操作使用现有冻结权重。

逐条 prompt、完整生成 token、两种解码、后端结束原因、约束结果以及可用的 Phase 0 对照保存在 [raw_generations.jsonl](raw_generations.jsonl)。各组 KL 的原生输出保存在 [kl](kl/)。
