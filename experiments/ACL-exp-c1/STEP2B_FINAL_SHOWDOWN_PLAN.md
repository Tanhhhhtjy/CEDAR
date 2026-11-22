# ACL-exp-c1 · Step 2B 最终执行方案（Qwen3-8B 机制定位 vs 双随机掩码 / 均匀 LoRA 无损因果对决）

- **状态**：**`READY TO EXECUTE`（立即执行！含前置 10 分钟 Wave 2B.05 硬门禁，门禁全绿后自动无缝进入 8 卡 Wave 2B.1 → 2B.2 → 2B.25 → 条件 2B.3！）**
- **预飞已验证基线**：`STEP2B_PREFLIGHT_TASK.md` 已实卡验证 vLLM 0.11.0 原生 `--enable-lora` 在全部三种秩上 100% 四选项覆盖全绿通过（`max_r=128`: `79/80=98.75%`, `du_abs=0.305`; `max_r=32`: `80/80=100%`, `du_abs=0.228`; `max_r=8`: `79/80=98.75%`, `du_abs=0.266`）；同时已根据预飞实测将 `c1_step2b_confirm_ce` 升级为单链 2×2 零捷径构造（`v3`, sha256=`c6c4bfb95b12...`）、修复 `c1_proxy_template_v2` 条件陈述句并给 `c1_eval_capabilities.py` 补上 `absl.logging` 标准库垫片。
- **取代文档**：本方案正式取代已归档旧草案 `STEP2_CAUSAL_SHOWDOWN_PLAN.md` 与已完成的 `STEP2B_PREFLIGHT_TASK.md`。
- **硬件与预计总耗时**：**8 × NVIDIA A800 40GB**
  - Wave 2B.05（2 卡自动门禁：新版单链 2×2 M0 冻结合格题 `≥ 800` 校验 + `c1_eval_capabilities.py` 20 题原生 LoRA 烟测）：~10 分钟
  - Wave 2B.1（8 组训练）+ Wave 2B.2（8 组无损全量评测 + 逐组保真 + 能力评测 + Standard 权重舍入 GPU 诊断）：~80–95 分钟
  - Wave 2B.3（条件触发：仅当单种子达到 `CANDIDATE` 时，3 卡补跑独立 `seed=1024` 方向复现）：~25 分钟

---

## 一、Step 2A-live 真实态势（严守口径，绝不夸大）

1. **BF16 合并失真已证实（`410e21f`）**：
   - `unif_w10_s295` 合并进 `bf16` 时，**70.6% 的更新元素直接归零**（整体余弦 `cos = 0.7776`，`down_proj = 0.7052`），将 `c1_dev` 双对率从纯 live 的 `63/80 (78.7%)` 虚抬至合并后 `74/80 (92.5%)`（`+13.8 pp`）；在前 20 题切片上 `u(V)-u(F)` 下降 `-0.856`（全 80 题均值为 `+0.156`）。
   - `dual_w17_s295` 注意力 `Q/O` 单元素 RMS `2.06e-4`（`o_proj` 仅 `17.4%` 归零，`cos = 0.9897`；整体 `cos = 0.9348`），显著更耐低精度合并。
2. **纯 Live（`γ=1.0 @295`）真实基线对照**：
   - `Dual-w10` live（`KL=0.000994`）：`c1_dev 64/80`、`c1_proxy_ood 192/240`、`旧模板 64/120`。
   - `Dual-w17` live（`KL=0.000984`）：`c1_dev 65/80`（`V/V_hes/V_auth = 69/80, 70/80, 69/80`，仍 `< 90%`）、`c1_proxy_ood 203/240`（`V/V_hes/V_auth = 91.7%/92.5%/92.9%`，相对 Uniform-w10 的 `199/240` 为 `+4` 题，`17:13`, McNemar `p=0.585`）、`旧模板 63/120`。
   - `Uniform-w10` live（`KL=0.001492`）：`c1_dev 63/80`、`c1_proxy_ood 199/240`、`旧模板 74/120`。
3. **Step 2B 唯一裁决问题**：
   - 在**无损双后端**、**全 8 组同配方同 `w=1.7`**、以及**信息论双侧零捷径 2×2 留出确认集（`N ≥ 800` M0 合格题）**上，`Targeted-Dual` 能否同时显著战胜两套匹配随机掩码（`Random-LM`, `Random-LM-seed2`）并战胜或 Pareto 支配 `Uniform-LoRA`？

---

## 二、锁定数据与信息论双侧零捷径主终点（零封存测试集泄漏）

1. **训练集（4 类纯规则，留出 `cross_entity_transfer`）**：
   - `experiments/ACL-exp-c1/data/c1_train_examples_wo_ce_nothink.jsonl`（400 基础题 × 6 原子项 = **2,400 条纯规则**，**无 `--replay-file`，无 `--zero-replay-loss`**）。
2. **唯一预注册主终点（2×2 对称交叉零捷径确认集）**：
   - 文件：`experiments/ACL-exp-c1/data/c1_step2b_confirm_ce_bases.jsonl`（**1,728 题 = 864 个 2×2 单元**，`scripts/c1_build_step2b_confirm_ce.py`，`data/c1_step2b_confirm_ce_manifest.json`）。
   - **2×2 信息论零捷径构造**：每个 `cell_id` 包含两份上下文（`C_A`, `C_B`，每份均同时含一条合法 `e2→e1` 规则和一条非法的 `e2→e2` 规则）与两段跨实体第二轮论证（`Arg_A`, `Arg_B`），交叉配成两条记录：Row A（`ctx=C_A`, `V=Arg_A`, `F=Arg_B`）和 Row B（`ctx=C_B`, `V=Arg_B`, `F=Arg_A`）。
   - **全部对抗审计已实测通过**：
     - **只看第一轮上下文审计**：每题内部 `context_prompt_v == context_prompt_f`（`1.000` 字节全等），Astra 实体一致性规则器双对 **`0 / 1,728`**，TF-IDF GroupKFold AUC = **`0.5000`**！
     - **只看第二轮论证审计**：`Counter(V) == Counter(F)`（多重集全等，贝叶斯上限精确 `0.5000`），实体切换计数准确率 **`0.5000`**，TF-IDF GroupKFold AUC = **`0.5000`**！
     - 外部开放集第二轮 V/F 文本重合 **`0`**；`(domain, a0)→at = 0.3333`；零触碰封存测试集。
   - **统计推断**：按 `cell_id` 聚类配对 Bootstrap（要求 M0 共享合格基数 **`≥ 800`**，由 Pre-Flight 提前冻结并实测确认）。
   - **评测臂**：8 个核心臂 `turn1_F turn1_V AO shell_only F V V_hes V_auth`（HF live `no_think` 单卡 ~8 分钟）。
3. **次级诊断与 `think` 非劣性对照集**：
   - `c1_dev`（80 题，全 15 臂，`no_think` + `vLLM think`）：作为 `think` 非劣性（95% CI 下界 `≥ -2.0 pp`）与必需多切片健康锚点。
   - `c1_proxy_template_v2`（120 题，全 15 臂，`no_think` + `vLLM think`）：修复 v1 语法歧义与双动词病句，与 `c1_train` 骨架重叠 **0 / 120**（注：其 6 道 depth-3 `cross_entity/AF` 记为诊断切片，模板泛化以 60 题 `context_flip` 为无捷径口径）。
   - `c1_kl_wikitext2_200.jsonl`（200 段，HF live `no_think` 精确 KL）。

---

## 三、锁定训练配方与 8 组双预算矩阵

- **统一超参（全 8 组完全一致，精确 200 步完整 cosine，12 步 warmup，无 replay）**：
  ```bash
  CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_train.py \
    --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b \
    --components-file <COMP_FILE> --arm <ARM_NAME> \
    --train-file experiments/ACL-exp-c1/data/c1_train_examples_wo_ce_nothink.jsonl \
    --lr 3e-5 --accept-weight 1.7 --epochs 1 --grad-accum 12 --microbatch 2 \
    --warmup-ratio 0.06 --save-steps 50,100,150,200 --seed 42 \
    --scaling-mode rslora --base-scale 2.0 --anchor-rank 8 --dropout 0.05 \
    --lambda-margin 0.25 --margin-m 4.0 --no-merge-save \
    --out-dir <WORK_DIR>/step2b_ckpts/<TAG>
  ```
- **8 组矩阵**（紧预算 `Uniform-LoRA-Tight` 因 `r=1` 离散化比 `Tight` 电路多 `16.5%` 参数，主表如实披露）：

| GPU | 预算档 | 代号 (`<TAG>`) | 组件文件 / `<ARM_NAME>` | 参数量 | `R_export` (`vllm max_r`) |
|---:|---|---|---|---:|---|
| 0 | **Standard (`6.87M`)** | `std_targeted_dual` | `c1_components_frozen.json` / `Targeted-Dual` | `6,869,504` | `112` (`128`) |
| 1 | Standard | `std_random_lm` | `c1_components_frozen.json` / `Random-LM` | `6,869,504` | `112` (`128`) |
| 2 | Standard | `std_random_lm_seed2` | `c1_components_frozen.json` / `Random-LM-seed2` | `6,869,504` | `112` (`128`) |
| 3 | Standard | `std_uniform_lora` | `c1_components_frozen.json` / `Uniform-LoRA` | `7,077,888` (+3.0%) | `4` (`8`) |
| 4 | **Tight (`1.52M`)** | `tight_targeted_dual` | `c1_components_step2_tight.json` / `Targeted-Dual-Tight` | `1,518,848` | `24` (`32`) |
| 5 | Tight | `tight_random_lm` | `c1_components_step2_tight.json` / `Random-LM-Tight` | `1,518,848` | `24` (`32`) |
| 6 | Tight | `tight_random_lm_seed2` | `c1_components_step2_tight.json` / `Random-LM-seed2-Tight` | `1,518,848` | `24` (`32`) |
| 7 | Tight | `tight_uniform_lora` | `c1_components_step2_tight.json` / `Uniform-LoRA-Tight` | `1,769,472` (+16.5%) | `1` (`8`) |

---

## 四、Wave 2B.05–2B.3 全自动执行流程（全部写入 `experiments/ACL-exp-c1/results/step2b_8b_showdown/`）

0. **Wave 2B.0（CPU ~15 秒）**：跑通全部 6 个脚本的 CPU 自检（`c1_export_peft_lora`, `c1_check_vllm_lora_parity`, `c1_eval_capabilities`, `c1_step2_stats`, `c1_eval`, `c1_train`），任一失败立即中止。
1. **Wave 2B.05（2 卡并行前置硬门禁，~10–15 分钟，全绿后自动进入 Wave 2B.1）**：
   - **GPU 0（M0 冻结与 `≥ 800` 检出力断言）**：
     1. M0 HF `no_think` on 新版单链 2×2 `c1_step2b_confirm_ce`（8 核心臂：`turn1_F turn1_V AO shell_only F V V_hes V_auth`），输出至 `results/step2b_8b_showdown/M0/no_think/`（保存 `eval_summary_no_think_confirm_ce.json` 与 `eligible_ids.json`）。
     2. M0 HF `no_think` on `c1_dev c1_proxy_template_v2`（全 15 臂，`--merge-eligible-ids-out`），确认 `eligible_ids.json` 三键齐全，且 **`len(eligible_ids["c1_step2b_confirm_ce"]) >= 800`**！（若 `< 800`，立刻推送 `M0` 诊断结果并中止，不启动 8 卡训练！）
     3. M0 vLLM `think` on `c1_dev c1_proxy_template_v2`（全 15 臂），导出 `results/step2b_8b_showdown/M0/think/eligible_ids.json`。
   - **GPU 1（能力评测脚本端到端 20 题原生 LoRA 烟测）**：
     - 复用已有 `s2a_dual_w17` 导出的 `peft_r128` 跑 `c1_eval_capabilities.py --limit 20`（指向 `<CAP_DIR>/data/{ifeval,gsm8k}.jsonl` 与 `<CAP_DIR>/vendor/instruction_following_eval`）；若 `/mnt/vllm310` 仍缺 `nltk`/`langdetect` 等第三方包，自动改用 `--skip-ifeval-score` 生成 `ifeval_raw.jsonl` 后调用服务器上原有的评分 Python / `<CAP_DIR>/score.py` 完成 IFEval 评分并写入标准 `summary.json` + `items.jsonl`！
2. **Wave 2B.1（8 卡并行，~22 分钟）**：GPUs 0–7 并行完成全部 8 组 200 步训练；将每组的 `train_meta.json`（含自动写入的 `adapter_state_sha256`）同步复制到 `results/step2b_8b_showdown/<TAG>/train_meta.json`。
3. **Wave 2B.2（8 卡并行，1 卡 1 组，~45–50 分钟）**：每张卡对负责的 `<TAG>`（`@200`）按序执行：
   1. **导出 PEFT LoRA**（`c1_export_peft_lora.py`，内置 FP32 `< 1e-5` 与 BF16 `cos ≥ 0.999` 断言，输出至 `<WORK_DIR>/step2b_peft/<TAG>`）。
   2. **逐组 SHA 绑定保真校验**（~2 分钟，`c1_check_vllm_lora_parity.py --dataset c1_dev --limit 40` -> `<TAG>/vllm_lora_parity.json`，100% 4 选项 `prompt_logprobs` 覆盖，`du_abs_mean ≤ 0.35`，不通过立即停止该组）。
   3. **HF live `no_think` 主终点**（~12 分钟，`--datasets c1_step2b_confirm_ce --arms turn1_F turn1_V AO shell_only F V V_hes V_auth`）。
   4. **HF live `no_think` 次级集 + WikiText-2 KL**（~5 分钟，`--datasets c1_dev c1_proxy_template_v2 --base-model ... --kl-corpus ...`）。
   5. **vLLM 原生 LoRA `think`**（~10 分钟，`--backend vllm --peft-lora-dir <PEFT_DIR> --datasets c1_dev c1_proxy_template_v2`）。
   6. **vLLM 原生 LoRA `IFEval` + `GSM8K`**（~18 分钟，`scripts/c1_eval_capabilities.py --peft-lora-dir <PEFT_DIR>` [按 Wave 2B.05 验证过的 IFEval 评分方式] -> `<TAG>/capabilities/{summary.json,items.jsonl}`）。
   7. **GPU 权重舍入成因诊断（仅 Standard 4 组，GPUs 0–3 各跑 1 组，~3 分钟）**：运行 `scripts/c1_diag_weight_rounding.py`（需 GPU，加载基座 + 合并两份 8B 并计算 200 段 KL），写入 `results/step2b_8b_showdown/weight_rounding_<TAG>.json`。
   8. **轻量轨迹补测**（~2 分钟）：HF live `no_think` 补测 `@100/@150` 在 `c1_dev`。
   - *(每完成一组可分批提交推送，避免最后单次推送过大)*。
4. **Wave 2B.25（CPU ~10 秒）**：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_step2_stats.py \
     --results-dir experiments/ACL-exp-c1/results/step2b_8b_showdown \
     --primary-dataset c1_step2b_confirm_ce --think-dataset c1_dev
   ```
   - 单种子最高仅给 **`CANDIDATE`**（`overall_go_14b=False`）！
5. **Wave 2B.3（条件触发：仅当 Wave 2B.25 输出 `verdict=CANDIDATE` 时自动执行，3 卡并行，~25 分钟）**：
   - 仅对 `candidate_single_seed_tier` 的 3 个因果组（`Dual`, `Random-1`, `Random-2`）使用 **`--seed 1024`** 重训 200 步，复制含 `seed=1024` 与独立 `adapter_state_sha256` 的 `train_meta.json`，跑 HF live `no_think`（`c1_step2b_confirm_ce` 8 核心臂）写入 `results/step2b_8b_showdown_seed1024/`；
   - 重跑 `c1_step2_stats.py --results-dir results/step2b_8b_showdown --seed2-dir results/step2b_8b_showdown_seed1024`，输出最终四态判定（`GO / HOLD / STOP`）并推送至 `origin/ACL-exp-c1`！

---

## 五、预注册四态门禁（`c1_step2_stats.py` v4 自动裁决）

1. **`CANDIDATE`（单种子过线，触发 Wave 2B.3，仍不直接解锁 14B）**：至少一档同时通过：
   - 检出力：`n_primary_shared_eligible ≥ 800` 且全集完整；
   - 逐组 SHA 绑定保真校验通过 + 全切片双制度健康（`IFEval ≥ 430`、`GSM8K ≥ 470`、`KL ≤ 0.005`、`c1_step2b_confirm_ce` 与 `c1_dev` 的 `no_think` 及 `c1_dev` 的 `think` 全部 `V/V_hes/V_auth ≥ 90%` 且 `Turn-1 ≥ 95%`、`think` 解析失败 `≤ 2%`）；
   - 聚类主终点胜双随机：均值池 `Δ ≥ +5.0 pp`（95% CI > 0）、两单随机各自 `Δ ≥ +3.0 pp` 且 95% CI > 0、连续边界 `u(V)-u(F)` 95% CI > 0、`think` 真非劣性（95% CI **下界** `≥ -2.0 pp`）；
   - 战胜（95% CI > 0）或 Pareto 支配 `Uniform-LoRA`（`Dual KL ≤ Uniform KL`、`IFEval ≥ Uniform - 3`、双制度 95% CI **下界** `≥ -2.0 pp`）。
2. **`GO`（`overall_go_14b=True`，解锁 Qwen3-14B 主实验）**：在 `CANDIDATE` 基础上，通过真实独立的 `seed=1024` 方向复现（目录不同、`seed 42 != 1024`、权重 SHA 不同、主终点 `Dual > R1` 且 `Dual > R2` 且 `no_think` 健康达标）。
3. **`STOP`**：两档均达到 `n ≥ 800` 检出力，且两档主终点相对双随机池均满足 `Δ ≤ +2.0 pp` **且 95% CI 上界 `< +5.0 pp`**（统计排除显著增益）→ 止损不投 14B，聚焦 Pareto 效率与低精度部署保真发现。
4. **`HOLD`**：其余情形（如检出力不足、单种子未过但 CI 上界仍容许 `≥ +5.0 pp`、或 `seed=1024` 复现未通过）→ 暂缓上 14B，进入定向诊断。
