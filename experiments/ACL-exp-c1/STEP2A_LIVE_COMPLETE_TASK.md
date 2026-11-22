# ACL-exp-c1 · Step 2A-live 补齐任务单（2 卡并行，~10–12 分钟，零训练）

- **状态**：**READY FOR EXECUTION**（监控拉取后立刻执行本任务单；**零训练、绝不执行 Step 2B、绝不触碰封存测试集 `c1_test_id / c1_test_ood`**，推完 `STEP2A_LIVE_READOUT.md` 后保持空闲等待 Step 2B 正式任务单）
- **基线提交**：`ACL-exp-c1@f5a631a`（复用 Step 2A 存盘的 `Qwen3-8B` `adapter_state_step0295.pt` 权重）
- **核心目标（与 Astra 复核 100% 对齐）**：
  1. 补齐缺失的两个 **纯 live `γ=1.0 @ 295` 全三集 `no_think` + 200 段 KL** 对照：`unif_w10_s295_g100_live` 与 `dual_w17_s295_g100_live`（与已有 `dual_w10_s295_g100` (live) 组成绝对同权重、同步数、同后端的干净对照）；
  2. 使用 `scripts/c1_diag_weight_rounding.py` 在真实 8B 权重上直接测量 `q_proj / o_proj / gate_proj / down_proj` 的真实 `bf16` 单次/二次舍入及 `fp16` 归零率与余弦相似度，并在**完全相同的 200 段 token（max_len=256，token 等权）**上输出 `KL(M0→live)`、`KL(M0→merged_bf16_single)`、`KL(live→merged_bf16_single)` 三联表。

---

## ⚠️ 严格安全护栏
1. **零训练、绝不触碰 14B、绝不触碰封存测试集**：`c1_eval.py` 必须显式传 `--datasets c1_dev c1_proxy_ood c1_proxy_template_ood`。
2. **独立产物目录**：统一写入 **`experiments/ACL-exp-c1/results/step2a_live_complete/`**，严禁改动已有结果目录。
3. **启动前 CPU 自检（~10 秒）**：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_diag_weight_rounding.py --selftest-tiny
   ```

---

## 一、2 卡并行执行清单（总耗时约 10–12 分钟）

### GPU 0（`unif_w10_s295` 纯 live 评测 + 真实权重舍入与 200 段 KL 三联）
- 校验 SHA-256 = `c5795252db8a76ae6354e7a06b9297b83c3035f1e1cddaa32175cf92ef1b3d30`（`/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/s2a_unif_w10/adapter_state_step0295.pt`）
1. `no_think` + 200 段 KL（~8 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --model-key qwen3_8b --arm-name Uniform-LoRA \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/s2a_unif_w10/adapter_state_step0295.pt \
     --adapter-scale 1.0 --alias qwen3_8b__unif_w10_s295_g100_live \
     --regime no_think --backend hf --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2a_live_complete/unif_w10_s295_g100_live/no_think
   ```
2. 真实权重舍入 + 统一 200 段 KL 三联（~2 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_diag_weight_rounding.py \
     --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Uniform-LoRA \
     --adapter-state /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/s2a_unif_w10/adapter_state_step0295.pt \
     --expected-adapter-sha256 c5795252db8a76ae6354e7a06b9297b83c3035f1e1cddaa32175cf92ef1b3d30 \
     --out-json experiments/ACL-exp-c1/results/step2a_live_complete/weight_rounding_unif_w10_s295.json
   ```

### GPU 1（`dual_w17_s295` 纯 live 评测 + `dual_w17_s295` & `dual_w10_s295` 真实权重舍入与 200 段 KL 三联）
- 校验 `dual_w17_s295` SHA-256 = `882ab0fb2264fb91d2ff72021ce36a51d1fb8cab414bf27bc301bccd399854b6`
- 校验 `dual_w10_s295` SHA-256 = `e201a03005778fc77389f4eac706f4b89ec235e2456cf6619bc639af4cc00214`
1. `dual_w17_s295_g100_live` `no_think` + 200 段 KL（~8 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=1 python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --model-key qwen3_8b --arm-name Targeted-Dual \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/s2a_dual_w17/adapter_state_step0295.pt \
     --adapter-scale 1.0 --alias qwen3_8b__dual_w17_s295_g100_live \
     --regime no_think --backend hf --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2a_live_complete/dual_w17_s295_g100_live/no_think
   ```
2. 真实权重舍入 + 统一 200 段 KL 三联（各 ~2 分钟）：
   - `s2a_dual_w17/adapter_state_step0295.pt` → `results/step2a_live_complete/weight_rounding_dual_w17_s295.json`
   - `s2a_dual_w10/adapter_state_step0295.pt` → `results/step2a_live_complete/weight_rounding_dual_w10_s295.json`

---

## 二、汇总与推送

生成 **`experiments/ACL-exp-c1/results/step2a_live_complete/STEP2A_LIVE_READOUT.md`**，包含两张表并推送至 `origin/ACL-exp-c1`：
1. **纯 Live `γ=1.0 @ 295` 三方公平对比表**（`dual_w10_s295` vs `unif_w10_s295` vs `dual_w17_s295`，全为纯 live，并列各自的 `merged` 参照，展示 3 套集上的 `V/F 双对`、`CF`、`AF`、`V / V_hes / V_auth`、`u(V)-u(F)` 与 `KL(M0→live)`）。
2. **真实 8B 权重舍入与统一 200 段 KL 三联表**（按 `q_proj / o_proj / gate_proj / down_proj / overall` 列出 `dw_rms`、`bf16_single`、`bf16_double`、`fp16_single` 的精确归零率与余弦相似度，以及统一 200 段 token 等权 `KL(M0→live)`、`KL(M0→merged_bf16_single)`、`KL(live→merged_bf16_single)`）。
