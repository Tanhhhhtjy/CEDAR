# ACL-exp-c1 · Step 0 执行任务单：Phase 0 现成 Checkpoint 零训练取证 + WikiText-2 KL 补测

- **状态**：READY FOR EXECUTION（三方已对齐签字，监控拉取后即可立刻执行）
- **基线提交**：`ACL-exp-c1@88182a2`（使用 GPU 机上已有的 Phase 0 `Qwen3-8B` `seed=42` `merged_hf` checkpoints 与基座 `M0`）
- **预计耗时**：约 **15–25 分钟**（零训练，仅推理抽样 + 4 组 200 段 WikiText-2 前向 KL）

---

## ⚠️ 严格安全护栏（必须遵守）
1. **禁止任何训练或修改模型权重**（纯只读推理取证）。
2. **禁止触碰封存测试集**：严禁评测 `c1_test_id_bases.jsonl` 与 `c1_test_ood_bases.jsonl`。跑 `c1_eval.py` 补 KL 时**必须显式传空 `--datasets`**（否则脚本默认会打开封存测试集！）。
3. **禁止覆盖任何已有 Phase 0 结果文件**：所有新产出统一写入新建目录 **`experiments/ACL-exp-c1/results/phase0_step0/`**。

---

## 任务一：关键样本原始生成取证（故障样本 + 健康对照）

使用与 Phase 0 完全一致的推理后端与解码参数（`vLLM`, `dtype=bfloat16`, `temperature=0`, `seed=42`, `max_tokens=3072`），在 **`M0`**、**`Targeted-Dual/merged_hf`**、**`Uniform-LoRA/merged_hf`**（若方便可顺带对 `IFEval id=1000` 补查 `Dual-wo-Hub` 与 `Random-LM`）上复现以下样本，**完整落盘原始生成内容**：

### 必查样本清单
1. **`IFEval`**（`google/IFEval` `966cd89545d6b6acfd7638bc708b98261ca58e84`，`no_think` 即 `enable_thinking=False`，完整用户 prompt 无修改、无 system prompt）：
   - **故障样本**：
     - `id=1000`（M0 `stop` 通过 → Dual `length` 失败）
     - `id=1122`（双方均为 `stop`，但 Dual 约束失败）
     - `id=1001`（M0 `stop` 通过 → Dual `length` 仍通过）
   - **健康对照**：
     - `id=32`（M0 / Dual / Uniform 三方均 `stop` 通过）
     - `id=24`（M0 `stop` 失败 → Dual `stop` 修复）
2. **`C1 c1_dev`（`think` 模式，`enable_thinking=True`，`max_tokens=3072`）**：
   - **故障样本**：
     - `C1-DEV-001` 的 `V` 与 `F` 臂（Uniform 精确生成 6 token 失败 vs Dual 221 / 293 token 正常通过）
     - `C1-DEV-008` 的 `turn1_V` 臂（验证真实撞 3,072 上限）
   - **健康对照**：
     - `C1-DEV-015` 的 `V` 与 `F` 臂（正常通过样本）
3. **`C1 c1_dev`（`no_think` 自由生成格式抽检，`enable_thinking=False`，不预填 `Final:`，`max_tokens=64`）**：
   - `C1-DEV-015` 的 `V`、`F` 臂，及 `C1-DEV-042` 的 `shell_only` 臂（检验 `M0 / Targeted-Dual / Uniform-LoRA` 在不预填 `Final:` 时是否能干净自由输出 `Final: <X><|im_end|>`）。
4. **`GSM8K`**（`openai/gsm8k` main test，沿用 `results/phase0/capabilities/spec.json` 的完全一致 prompt 后缀，`enable_thinking=True`，`max_tokens=3072`）：
   - `id=0` 与 `id=4`（M0 与 Dual 间可解析性反转样本）。

### 单题必存字段（保存至 `experiments/ACL-exp-c1/results/phase0_step0/raw_generations.jsonl`）
每条记录必须包含：
- `benchmark`, `item_id`, `arm`, `regime`, `backend`, `model_alias`
- `prompt_n_tokens`, `prompt_tail_token_ids`（最后 32 个 token IDs）, `prompt_text`
- `completion_token_ids`（完整生成 token IDs 列表）, `n_completion_tokens`
- `raw_text_with_special_tokens`（`skip_special_tokens=False` 解码）
- `raw_text_clean`（`skip_special_tokens=True` 解码）
- `finish_reason`, `stop_reason`
- 对应的规则/约束解析结果（如 C1 的 `parse_output` 结果、IFEval 各条约束 pass/fail、GSM8K 提取值）。
- *(可选加分)*：若 `Targeted-Dual` 在 `IFEval id=1000` 上复现 `finish_reason == "length"`，顺手用 HF `transformers` greedy（`do_sample=False`）单跑这一题做一次 `vLLM vs HF` 一致性核对，排除推理后端差异。

### 同步核对配置与 EOS 列表（保存至 `experiments/ACL-exp-c1/results/phase0_step0/config_eos_check.json`）
对比 `M0` 与全部 4 个 Phase 0 `merged_hf` 目录的：
- `generation_config.json`（特别是 `eos_token_id` 是否保持官方 Qwen3-8B 的 `[151645, 151643]`）
- `tokenizer_config.json`（`eos_token`、`pad_token`、`chat_template` SHA-256）
- 实际加载进 vLLM 时的有效 stop token IDs。

---

## 任务二：补测现有 4 组冻结 WikiText-2 参考 KL 散度

对现有 Phase 0 的 4 个已合并 checkpoint（`Targeted-Dual`, `Dual-wo-Hub`, `Random-LM`, `Uniform-LoRA`），使用已冻结的外部语料 `experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl`（SHA-256 应为 `97a3ed61a17e4023eabe9566fbe56768d0708aea97b662abac52dbb830315e05`）计算 $D_{\text{KL}}(p_{M_0} \| p_{\text{tuned}})$。

**必须显式传空 `--datasets` 且输出到新目录**（省去 4,800 次 dev 前向、绝不碰封存测试集、不覆盖旧 summary）：

```bash
for ARM in Targeted-Dual Dual-wo-Hub Random-LM Uniform-LoRA; do
  python experiments/ACL-exp-c1/scripts/c1_eval.py \
    --model <PHASE0_CKPT_ROOT>/${ARM}/merged_hf \
    --base-model <BASE_QWEN3_8B_PATH> \
    --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
    --regime no_think --backend hf --datasets \
    --alias qwen3_8b__${ARM} \
    --out-dir experiments/ACL-exp-c1/results/phase0_step0/kl/${ARM}
done
```
*(四组可用 4 张卡并行跑，预计 < 5 分钟完成。)*

---

## 任务三：产出简明读数报告并推送

生成 **`experiments/ACL-exp-c1/results/phase0_step0/STEP0_READOUT.md`**，清晰回答以下 4 个问题后，将整个 `experiments/ACL-exp-c1/results/phase0_step0/` 目录提交并推送至 `origin/ACL-exp-c1`：

1. **IFEval 撞限真相**：`Targeted-Dual` 在 `id=1000`、`1001` 撞上 3,072 token 时到底生成了什么（无限复读？未闭合 `<think>`？长篇漂移不吐 EOS？），以及 `id=1122` 正常 `stop` 却失败的具体原因是什么？对照 `id=32` / `24` 有何差异？
2. **Uniform-LoRA 6-token 真相**：`Uniform-LoRA` 在 `C1-DEV-001/V, F` 精确生成的 6 个 token 逐一是什么（含特殊符号）？其在未闭合 `<think>` 里选出的选项字母是否正确？
3. **配置/EOS 核对结论**：`M0` 与 4 个 `merged_hf` 的 `eos_token_id`、`generation_config.json`、`chat_template` 是否完全一致？
4. **WikiText-2 参考 KL 汇总表**：4 组的 `n_tokens`、`kl_mean`、`kl_median` 完整对比表。
