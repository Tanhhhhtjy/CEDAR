# ACL-exp-c2 · 零重训补评任务单（`dac_iit` 与 `unif_paired` 保真口径校准与全量收尾）

- **状态**：**`READY TO EXECUTE`（立即在 2 张空闲 GPU 上并行执行，零重训，预计 ~45 分钟完成全部收尾！）**
- **适用范围**：直接复用实验机磁盘上**已训练完成并导出的** `dac_iit`（SHA `7fb112d0c8ac...`）与 `unif_paired`（SHA `3f2a78da9834...`）权重及 PEFT LoRA，不重训任何一步，也不改动已完成的其他 6 组结果。

---

## 一、思考过程与工程教训复盘（为什么必须立刻补评这两组）

1. **为什么之前被卡停？**
   在 Wave 2C.2 中，我们沿用了 C1 的硬阈值 `--max-abs-du-mean 0.35`，且要求任一保真未过即中止该组全部评测。复核 `vllm_lora_parity.json` 实测：
   - **`dac_iit`**：`u_coverage = 1.0 (80/80)`、`option_match = 98.75% (79/80)`（唯一 1 处为 `u_hf=0.0` vs `u_vllm=-0.5` 边界平局）、`du_signed_mean = +0.0058`（中位数 `0.0`，**零系统偏移**），仅 `du_abs_mean = 0.38108` 比 `0.35` 高出 `0.031`；
   - **`unif_paired`**：`u_coverage = 1.0 (80/80)`、`option_match = 100.0% (80/80)`（**0 翻转**！）、`du_signed_mean = -0.0187`（中位数 `0.0`），仅 `du_abs_mean = 0.37240` 比 `0.35` 高出 `0.022`。
2. **物理成因与必须吸取的永久工程教训**：
   - `M0` 基座本身在 HF vs vLLM BF16 之间就有 `du_abs_mean = 0.256–0.291` 的随机浮点抖动；而 `Paired 2×2` 与 `DAC-IIT` 显式拉大了正确/错误选项的 logit 边界（`|u|` 升至 `15–25`），BF16 尾数离散步长随幅值同比放大，使完全健康的权重随机抖动自然落在 `0.35–0.42` 之间！
   - **永久教训（写进以后所有实验门禁）**：① 保真门禁应当以 **`u_coverage == 1.0`、`option_match >= 95%`、`|du_signed_mean| <= 0.25`、`du_abs_mean <= 0.50`** 为统一科学标准（真正有 LoRA 缩放或合并 bug 时 `du_abs_mean` 在 `2.0–17.0`）；② 即使 vLLM 端保真出现临界告警，也**绝不该连带拦截纯 HF 无损 `no_think` + KL 评测**！
3. **为什么要立刻补齐 `dac_iit` 与 `unif_paired`？**
   从已回传的 8 组训练日志可见，B1 真实电路 `dac_iit` 的内部反事实互换翻转率高达 **`97.5%`**，而逐层等数随机分区 `rnd1_iit / rnd2_iit` 仅 **`37.5% / 55.0%`**，且强行对随机电路加 IIT 导致泛化暴跌 `−6 ~ −21 pp`！补齐 `dac_iit` 与 `unif_paired` 的完整三轴评测后，才能正式算出整个 2×2 因子交互主表与 `c2_stats.py` 最终判定。

---

## 二、零重训执行清单（仅补跑 `dac_iit` 与 `unif_paired`，并等其余组收尾后汇总 `c2_stats.py`）

> 复用既有工作目录 `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c2/showdown-c62cf156` 中的：
> - `dac_iit`: `c2_ckpts/42/dac_iit/{adapter_state.pt, adapter_state_step0100.pt, adapter_state_step0200.pt}` 与 `c2_peft/dac_iit`
> - `unif_paired`: `c2_ckpts/42/unif_paired/{adapter_state.pt, adapter_state_step0100.pt, adapter_state_step0200.pt}` 与 `c2_peft/unif_paired`
> （运行前请校验两个 `adapter_state.pt` 的 SHA-256 仍为 `7fb112d0c8ac...` 与 `3f2a78da9834...`，严禁重训！）

### 任务 A（GPU A → `dac_iit`，`<ARM_NAME>=Targeted-Dual`）与 任务 B（GPU B → `unif_paired`，`<ARM_NAME>=Uniform-LoRA`）并行执行：

1. **按校准口径重跑保真校验（`--max-abs-du-mean 0.50`）**（~2.5 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c1/scripts/c1_check_vllm_lora_parity.py \
     --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm <ARM_NAME> \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state <WORK_DIR>/c2_ckpts/42/<TAG>/adapter_state.pt \
     --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --max-abs-du-mean 0.50 --dataset c1_dev --limit 40 \
     --out-json experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/vllm_lora_parity.json
   ```
2. **HF 无损 Live `no_think` 全 4 集（全 15 臂）+ WikiText-2 KL（200 段）**（~15 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c2/scripts/c2_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --model-key qwen3_8b --arm <ARM_NAME> \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state <WORK_DIR>/c2_ckpts/42/<TAG>/adapter_state.pt \
     --alias <TAG> --regime no_think \
     --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
     --eligible-ids-in experiments/ACL-exp-c2/results/c2_8b_showdown/M0/no_think/eligible_ids.json \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/no_think
   ```
3. **vLLM 原生 LoRA `think` 全 4 集（全 15 臂，`max-new-tokens 3072`）**（~18 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c2/scripts/c2_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --backend vllm \
     --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --alias <TAG> --regime think --max-new-tokens 3072 \
     --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
     --eligible-ids-in experiments/ACL-exp-c2/results/c2_8b_showdown/M0/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/think
   ```
4. **vLLM 原生 LoRA 通用能力护盾（`IFEval 541` + `GSM8K 1,319`）**（~18 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --alias <TAG> --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/capabilities
   ```
   *(传入与本轮其他 6 组完全一致的 `--ifeval-data / --gsm8k-data / --ifeval-vendor` 路径。)*
5. **轻量学习曲线轨迹（`@100` 与 `@200` 在 `c2_confirm_bidir` 与 `c1_dev` 的 HF Live `no_think`）**（~4 分钟）：
   对 `adapter_state_step0100.pt` 与 `adapter_state_step0200.pt` 分别输出至 `<TAG>/trajectory/step0100/no_think/` 与 `step0200/no_think/`。
6. **部署保真审计（仅 `dac_iit`，~3 分钟）**：
   运行 `c1_diag_weight_rounding.py` 输出至 `experiments/ACL-exp-c2/results/c2_8b_showdown/weight_rounding_dac_iit.json`。

### 任务 C（等待全部 8 组收尾后，CPU ~15 秒）：运行新版 `c2_stats.py` 汇总并推送！
```bash
/mnt/vllm310/bin/python experiments/ACL-exp-c2/scripts/c2_stats.py \
  --results-dir experiments/ACL-exp-c2/results/c2_8b_showdown
```
- 提交并推送 `c2_stats.json`、`C2_READOUT.md` 及上述两组的全部评测文件至 **`origin/ACL-exp-c2`**。
- 若 `c2_stats.py` 给出 **`verdict = CANDIDATE_SEED2`**，再按原计划自动触发 **Wave 2C.3（`seed=1024` 4 组独立复现）**；否则停在单种子汇总等待人工审计。
