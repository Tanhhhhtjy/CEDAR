# ACL-exp-c1 · Step 2B-Preflight 任务单（2 卡 10 分钟实卡预飞验收，正式 8 卡开跑前必过硬门禁）

- **状态**：**`READY TO EXECUTE`（立即执行；仅跑本预飞验收，暂不执行 `STEP2B_FINAL_SHOWDOWN_PLAN.md` 的 8 卡训练！）**
- **目的**：严格落实起飞前实卡验收，在消耗 8 卡算力前，用 2 张 A800 在 ~10 分钟内实测验证 3 件决定性事项：
  1. **新版 2×2 零捷径主终点 `c1_step2b_confirm_ce`（1,728 题）上的 M0 真实 `no_think` 双侧 Turn-1 合格题数**是否达到 `≥ 800`（并冻结 `M0/no_think/eligible_ids.json` 三键合并写入与 `M0/think/eligible_ids.json`）；
  2. **vLLM 0.11.0 原生 `--enable-lora` 跨后端保真校验（v2：100% 4 选项 `prompt_logprobs` 覆盖）**在 Step 2B 实际用到的全部三种秩 **`max_lora_rank = 128 / 32 / 8`** 上是否全部 `passed=True`；
  3. **版本化能力评测脚本 `scripts/c1_eval_capabilities.py`** 在 `--peft-lora-dir`（原生 LoRA）下跑通 `IFEval` + `GSM8K` 各 20 题小样并产出与 `results/phase0/capabilities/spec.json` 一致的 `summary.json` / `items.jsonl`。
- **结果目录**：全部写入 **`experiments/ACL-exp-c1/results/step2b_preflight/`**（不改动任何历史结果目录；严禁打开或评测 `c1_test_id` / `c1_test_ood`！）。

---

## 一、Step 0：CPU 全量自检（~15 秒，全部通过才允许开卡）

```bash
python experiments/ACL-exp-c1/scripts/c1_export_peft_lora.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_check_vllm_lora_parity.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_step2_stats.py --selftest
python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
```

---

## 二、2 卡并行实卡验收（约 8–12 分钟）

### GPU 0：M0 基线评测 + 冻结双侧 Turn-1 合格分母（检验 `c1_step2b_confirm_ce` 合格题 `≥ 800`）
1. **M0 HF `no_think` on `c1_step2b_confirm_ce`**（8 个核心臂，约 6–8 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU0> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --alias qwen3_8b__M0 --regime no_think --backend hf \
     --datasets c1_step2b_confirm_ce \
     --arms turn1_F turn1_V AO shell_only F V V_hes V_auth \
     --eligible-ids-out experiments/ACL-exp-c1/results/step2b_preflight/M0/no_think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2b_preflight/M0/no_think
   ```
   - 完成后立刻重命名 `eval_summary_no_think.json` 为 `eval_summary_no_think_confirm_ce.json`（保留 `records_no_think_c1_step2b_confirm_ce.jsonl`）。
2. **M0 HF `no_think` on `c1_dev c1_proxy_template_v2`**（全 15 臂，**必须加 `--merge-eligible-ids-out`**，约 2 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU0> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --alias qwen3_8b__M0 --regime no_think --backend hf \
     --datasets c1_dev c1_proxy_template_v2 \
     --eligible-ids-out experiments/ACL-exp-c1/results/step2b_preflight/M0/no_think/eligible_ids.json \
     --merge-eligible-ids-out \
     --out-dir experiments/ACL-exp-c1/results/step2b_preflight/M0/no_think
   ```
   - 断言 `M0/no_think/eligible_ids.json` 同时包含 `["c1_dev", "c1_proxy_template_v2", "c1_step2b_confirm_ce"]` 三个键，并打印每个键的合格题数。
3. **M0 vLLM `think` on `c1_dev c1_proxy_template_v2`**（全 15 臂，严禁 `--backend hf`，约 5 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU0> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --alias qwen3_8b__M0 --regime think --backend vllm \
     --max-new-tokens 3072 --datasets c1_dev c1_proxy_template_v2 \
     --eligible-ids-out experiments/ACL-exp-c1/results/step2b_preflight/M0/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2b_preflight/M0/think
   ```

### GPU 1：覆盖全部三种秩（`max_r=128 / 32 / 8`）的 vLLM 原生 LoRA 保真验收 + `IFEval/GSM8K` 小样
1. **复用已有 Step 2A checkpoint 验证 `max_lora_rank=128` 与 `max_lora_rank=8`**：
   - `r128`：使用已有 `s2a_dual_w17` 的 `adapter_state_step0295.pt`（`c1_components_frozen.json`, `Targeted-Dual`, `R_export=112 -> max_r=128`）；
   - `r8`：使用已有 `s2a_unif_w10` 的 `adapter_state_step0295.pt`（`c1_components_frozen.json`, `Uniform-LoRA`, `R_export=4 -> max_r=8`）；
   - `r32`：快速训练一个 10 步 `Targeted-Dual-Tight` 烟测权重（`c1_components_step2_tight.json`, `--epochs 1 --max-steps 10 --lr 3e-5 --accept-weight 1.7 --no-merge-save`, ~30 秒，`R_export=24 -> max_r=32`），专门验证 `max_lora_rank=32` 真实加载与保真！
2. **依次导出 PEFT LoRA 并跑 `c1_check_vllm_lora_parity.py`（`--dataset c1_dev --limit 40`）**：
   - 输出 3 份 SHA 绑定保真报告（每份必须 `passed=True` 且 `u_coverage_rate=1.0`）：
     - `experiments/ACL-exp-c1/results/step2b_preflight/parity_r128_dual.json`
     - `experiments/ACL-exp-c1/results/step2b_preflight/parity_r32_tight_dual.json`
     - `experiments/ACL-exp-c1/results/step2b_preflight/parity_r8_unif.json`
   - 若任一秩失败，立即记录完整报错与指标（不静默跳过），停止后续小样或尝试定位 vLLM 0.11.0 原因。
3. **原生 LoRA 能力评测小样（`--limit 20`，~2 分钟）**：
   - 用 `r128` 的导出目录 `<TMP_PEFT_R128>` 调用 `scripts/c1_eval_capabilities.py`，指向服务器上已冻结的 `data/ifeval.jsonl` 与 `data/gsm8k.jsonl`（以及官方 `evaluation_lib` 所在目录 `--ifeval-lib-dir`）：
     ```bash
     CUDA_VISIBLE_DEVICES=<GPU1> python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py \
       --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
       --peft-lora-dir <TMP_PEFT_R128> \
       --ifeval-jsonl <CAP_DIR>/data/ifeval.jsonl \
       --gsm8k-jsonl <CAP_DIR>/data/gsm8k.jsonl \
       --ifeval-lib-dir <IFEVAL_LIB_DIR> \
       --alias preflight_r128_smoke20 --limit 20 \
       --out-dir experiments/ACL-exp-c1/results/step2b_preflight/cap_smoke20_r128
     ```
   - 核对 `cap_smoke20_r128/summary.json` 同时包含 `ifeval`（20 题）与 `gsm8k`（20 题）且 `peft_adapter_model_sha256` 非空。

---

## 三、交付物与推送要求

完成后汇总写入 `experiments/ACL-exp-c1/results/step2b_preflight/preflight_summary.json`（含三秩 `parity` 通过状态、M0 三个数据集的合格题数、`cap_smoke20` 摘要和服务器 `<CAP_DIR>` 路径记录），将 `results/step2b_preflight/`（不含大型临时权重）提交并推送至 `origin/ACL-exp-c1`！
