# ACL-exp-c3.5 Stage 3 Showdown Plan: KL-Surgical `M3-v3` + Conflict-Free PCGrad Soft-Gated IIT (8-GPU Showdown)

- **Branch**: `ACL-exp-c3.5` (Stage 2 complete at `88cec3f`, Stage 3 ready to auto-run immediately!)
- **Model**: `Qwen3-8B` (`seed=42` decisive sprint; identical training data `c3_train_examples_nothink.jsonl`, identical `lr=2.5e-5, max_steps=300` as `u2_corridor`)
- **Absolute Red Lines**:
  - NEVER open, read, or evaluate sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`; strictly exclude `converse_error` and `depth-4`.
  - NEVER modify frozen `ACL-exp-a1/`, `ACL-exp-b1/`, `ACL-exp-c1/`, `ACL-exp-c2/`, or `ACL-exp-c3/`.
  - **NEVER commit `.pt` / `.safetensors` to Git**; push only lightweight `json / jsonl / md` results under `experiments/ACL-exp-c3.5/results/stage3/` incrementally to `origin/ACL-exp-c3.5`.

---

## 第一部分 · Stage 2 深度复盘与 Stage 3 冲顶思考日志

Stage 2（`88cec3f`，全部 8/8 训练臂通过严苛零 `None` 健康门禁：`IFEval 440–449/541`、`GSM8K 505–547/1319` vs M0 `496`、`parity PASS`、跨字母互换字面抄字母率全员 `0.0%`）产出了 **四条极其清晰且统计学显著（`p < 0.03`）的核心发现**，直接指明了通往 ACL 主会硬核胜出的最后两步：

### Stage 2 核心实证总结（900 独立单元 `c3_blind_confirm_v2` + 历史集 + 同 Batch 梯度遥测）

| 臂 | 4-Cell (UW) | Joint T1∧4C | Row B (`120`) | `c2_bidir` (`360`) | `think_v2` (`240`) | NLL | IFEval / GSM8K | Wiki KL (`Δ vs u2`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `u2_corridor` (最强 Uniform) | 50.56% | 49.89% | 15.8% (19) | 75.6% | 96.75% | 0.783 | 449 / 547 | `0.001362` (`0.0%`) |
| `armA_m3v1_relax` | 46.11% | 45.61% | 15.0% (18) | 72.8% | 97.50% | 0.746 | 441 / 505 | `0.001256` (`-7.8%`) |
| **`armB_m3v2_corridor`** | **50.22%** | **49.67%** | **18.3% (22)** | **77.2%** | 94.75% | 0.780 | 442 / 522 | **`0.001092` (`-19.8%`)** |
| **`armC_m3v2_soft_b1_iit`** | **49.83%** | **48.94%** | **20.0% (24)** | **77.2%** | **95.75%** | **0.760** | 442 / 519 | **`0.001085` (`-20.3%`)** |
| `armD_m3v2_soft_rnd_iit` | 49.22% | 48.39% | 18.3% (22) | 76.7% | 94.50% | 0.742 | 443 / 535 | `0.001068` (`-21.6%`) |
| `armE_unif_soft_b1_iit` | 48.61% | 47.89% | 18.3% (22) | 76.9% | 95.00% | 0.806 | 440 / 537 | `0.001394` (`+2.3%`) |

1. **`M3-v2` 参数重分配暴涨 `+4.11 pp`（`B − A`, `p < 0.0001`, 52W/6L）**：直接追平 `u2_corridor`（`B − u2 = −0.33 pp` `[-1.56, +0.89]`, `Joint −0.22 pp` `[-1.44, +1.00]`，两项 CI 下界均已通过 Path 2 `>= −2.0 pp` 门禁！），同时在 `Row B`（`18.3% vs 15.8%`）、`c2_confirm_bidir`（`77.2% vs 75.6%`）、`Turn-1`（`97.6% vs 97.0%`）和 `NLL`（`0.780 vs 0.783`）全面反超 `u2`，并把 Wiki KL 压低了 **`-19.8%`**——**离预登记 Path 2 Efficiency Pareto 门禁（`-25.0%` KL）仅差最后 5 个百分点！**
2. **纯头定位显著成立（`C − D = +0.61 pp`, `p = 0.022`, 9W/3L；`Joint +0.56 pp`, `p = 0.050`）**：从完全相同的比特级第 120 步状态（`87d69e531d47`）分叉，B1 头的同 batch 梯度余弦对齐度是随机头的 **3.8 倍**（`mean cos = +0.145` vs `+0.038`，`median +0.249`），后段软目标命中率 **`83.3%` vs 随机头 `51.7%`（`+31.6 pp`！）**，Row B 冲上全场最高 **`20.0%`（24/120）**，`think_v2` 高出 `B` `+1.00 pp`、高出 `D` `+1.25 pp`！
3. **架构 × 算法强正交互（`C − E = +1.22 pp`, `p = 0.029`, 24W/12L，且 KL 低 `-22.2%`）**：同一套 Soft-Gated B1-IIT 装在 Uniform 上显著掉分 `-1.94 pp`（`p < 0.0001`）且 KL 反升 `+2.3%`；装在 `M3-v2` 上不仅抗住主线、还把 Row B 推到 `20.0%`、NLL 降到 `0.760`、KL 再降到 `-20.3%`！
4. **同 Batch 梯度遥测精准抓住 `C − B = −0.39 pp` 的唯一罪魁祸首**：分阶段检查 `armC` 的梯度夹角：
   - **第 121–180 步**：`cos_mean = +0.008`，**`46.9%` 的步数为负余弦冲突**（此时 `c3_dev` 才 `54%`，启动略早！）；
   - **第 181–240 步**：模型站稳后，`cos_mean` 飙升到 **`+0.260`**（负冲突降到 `32.7%`，`dir_hit = 99.1%`）；
   - **第 241–300 步**：仍有 `40.0%` 的步数存在负余弦，把 `scope_swap` 与 `negation_drop` 拉掉了 5 题！

---

### Stage 3 的两大外科手术式升级（直接击穿 Path 2 `-30%+ KL` 并冲刺 `C > B` & `C > u2`！）

1. **升级一 · KL 外科手术架构 `M3-v3-PureMid` & `M3-v3-TightBridge`（`data/c35_components_frozen.json`，已导出并自测通过）**：
   - 回看 Probe 1 实测：`L0–13` 全关时，`u2_no_z1_all` 与 `m2_no_z1_qo` 的 Wiki KL 均骤降至 **`~0.00095`（相对 `u2` 直降 `-30%`！）**；而 Stage 2 `armB` 之所以停在 `0.001092 (-19.8%)`，正是因为还保留了 6 层 `L8–13`（含高 KL 的 `Z1:Q/O` 且松弛 `prox=1e-4`）以及 10 颗 `L28–35` 顶层头！
   - **`M3-v3-PureMid` (`7,040,512` = `-0.53%` vs Uniform, `max_layer_r = 88 <= 128`)**：
     - **全部 Zone 1 (`L0–13`) 100% 冻结（`0` 参数）** → 物理切断全部早层 KL 漂移，直指 **`-30% ~ -38%` Wiki KL 降幅**！
     - **98.8% 预算全部投入 Zone 2 (`L14–27`)**：全部 14 层 Zone-2 MLP（5 核心 B1 `N_CF` `@ r=10` + 其余 9 层 `@ r=6` = `3.41M`，近 Uniform 中层 MLP 的 2 倍容量，Probe 1 实测仅花 `1.4%` KL！）+ 42 颗 B1 中层头 3 档分配（`10@18, 12@10, 20@6` = `3.55M`）；
     - **Zone 3 (`L28–35`) 极简收敛**：0 层顶层 MLP，仅保留 5 颗 `form_only_heads @ r=2`（`84K` = `1.2%`）以保证 B1-IIT 形式路径完整可训！
   - **`M3-v3-PureMid-Random-Mask` (`7,040,512`, 精确同参数同层秩多重集)**：同步上 GPU 5，无混杂验证 **“B1 定位” vs “纯中层频段”**，一次性堵死审稿人关于 PureMid 归因的唯一潜在质疑！
   - **`M3-v3-TightBridge` (`6,974,976` = `-1.45%` vs Uniform, `max_layer_r = 88`)**：保险臂！冻结 `L0–9` 并**完全删除高 KL 的 `Z1:Q/O`**，仅留 `L10–13` 4 层 MLP-only `@ r=4`（配紧约束 `lr_z1_mult=0.60, lambda_prox_z1=5e-4`）+ 全部 14 层 Zone-2 MLP（`5@r10 + 9@r4`）+ 42 B1 中层头 + 5 Z3 form 头 `@ r=2`。

2. **升级二 · 无冲突正交投影 PCGrad + 延迟第 160 步启动（`c35_train.py --iit-pcgrad --iit-start-step 160`）**：
   - 每当同 batch 测得 $\langle g_{\text{iit}}, g_{\text{main}} \rangle < 0$，立即执行无参正交投影：
     $$g_{\text{iit}}^{\perp} = g_{\text{iit}} - \frac{\langle g_{\text{iit}}, g_{\text{main}} \rangle}{\|g_{\text{main}}\|^2} g_{\text{main}}$$
     确保 $\langle g_{\text{iit}}^{\perp}, g_{\text{main}} \rangle \equiv 0$（**数学保证负向冲突步率从 `39.6%` 降至 `0.0%`！**），而对所有正向对齐步（$\cos \ge 0$）完整保留建设性 B1-IIT 梯度；再施加 `0.30×` 独立相对限幅进入全局裁剪！
   - 同步将 IIT 启动步从第 120 步推迟至 **第 160 步**（跳过 Stage 2 实测的 `121–180` 低余弦混沌期），并以 `resume_step0160.pt` 实现严格比特级分叉！

---

## 第二部分 · 8×A800 Stage 3 精确执行清单（约 85–95 分钟全自动跑完）

### 环境约定与步骤 0（~20 秒）
- `BASE=<Qwen3-8B 本地路径>`, `PY=<venv python>`, `KL_CORPUS=<同 C3 的 200 段 WikiText-2 语料>`, `IFEVAL_JSONL=<同 C3>`, `GSM8K_JSONL=<同 C3>`, `IFEVAL_LIB=<同 C3>`
- `WORK=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage3-<SHORT_SHA>`
- `RES=experiments/ACL-exp-c3.5/results/stage3`
- `ARMB_CKPT=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage2-605c6f63/selected/armB_m3v2_corridor/selected_adapter_state.pt`

1. 拉取最新 `origin/ACL-exp-c3.5`，先跑 CPU 自测验证全过：
   ```bash
   $PY experiments/ACL-exp-c3.5/scripts/c35_modules.py --selftest-tiny
   $PY experiments/ACL-exp-c3.5/scripts/c35_export_peft_lora.py --selftest-tiny
   $PY experiments/ACL-exp-c3.5/scripts/c35_train.py --selftest-tiny
   $PY experiments/ACL-exp-c3.5/scripts/c35_probe_m3v2_kl.py --selftest-tiny
   $PY experiments/ACL-exp-c3.5/scripts/c35_stage_baselines.py --selftest-tiny
   $PY experiments/ACL-exp-c3.5/scripts/c35_stats.py --selftest-tiny
   ```
2. 零重算组装 Stage 2 完整基线对照（含 `M0, u1_base, u2_corridor, m2_corridor, armB_m3v2_corridor, armC_m3v2_soft_b1_iit` 全部 `no_think / think / capabilities / parity / KL`）：
   ```bash
   $PY experiments/ACL-exp-c3.5/scripts/c35_stage_baselines.py --from-stage2 --out-root $RES
   ```
   立即提交推送 `stage3-refs`。

### GPU 7（零重训，前 ~4 分钟）：`armB_m3v2_corridor` 逐子层 KL 探针
```bash
CUDA_VISIBLE_DEVICES=7 $PY experiments/ACL-exp-c3.5/scripts/c35_probe_m3v2_kl.py \
  --model $BASE --model-key qwen3_8b --arch M3-v2 \
  --adapter-state $ARMB_CKPT \
  --kl-corpus $KL_CORPUS --batch-size 16 \
  --out-dir $RES/probe_m3v2_kl
```
完成后立即提交推送 `$RES/probe_m3v2_kl/`（`stage3-probe-m3v2-kl`）。

### GPUs 0–6（7 卡并行训练 + 比特级第 160 步分叉，~40 分钟）
全部 7 臂均使用 `--model $BASE --model-key qwen3_8b --components-file experiments/ACL-exp-c3.5/data/c35_components_frozen.json --lr 2.5e-5 --max-steps 300 --grad-accum 20 --microbatch 2 --seed 42`：

- **GPU 0 (`s3_puremid_corr`)**（立刻启动；第 160 步原子写出 `resume_step0160.pt`）：
  `--arch M3-v3-PureMid --loss-mode corridor --iit-target none --lr-z1-mult 0.0 --lambda-prox-z1 0.0 --save-steps 120,160,220,300 --save-resume-step 160 --out-dir $WORK/ckpts/s3_puremid_corr`
- **GPU 1 (`s3_puremid_pcgrad_b1`)**（轮询等待 `$WORK/ckpts/s3_puremid_corr/resume_step0160.pt` 出现并检查 GPU 0 存活，立刻接续跑 161–300 步）：
  `--arch M3-v3-PureMid --loss-mode corridor_iit --iit-target b1 --lr-z1-mult 0.0 --lambda-prox-z1 0.0 --iit-start-step 160 --iit-alpha-start 0.30 --iit-alpha-end 0.70 --iit-source-margin-gate 2.0 --iit-grad-ratio-cap 0.30 --iit-pcgrad --resume-from $WORK/ckpts/s3_puremid_corr/resume_step0160.pt --save-steps 220,300 --out-dir $WORK/ckpts/s3_puremid_pcgrad_b1`
- **GPU 2 (`s3_puremid_pcgrad_rnd`)**（同上等待 `$WORK/ckpts/s3_puremid_corr/resume_step0160.pt`，立刻接续跑 161–300 步）：
  `--arch M3-v3-PureMid --loss-mode corridor_iit --iit-target rnd_head_1 --lr-z1-mult 0.0 --lambda-prox-z1 0.0 --iit-start-step 160 --iit-alpha-start 0.30 --iit-alpha-end 0.70 --iit-source-margin-gate 2.0 --iit-grad-ratio-cap 0.30 --iit-pcgrad --resume-from $WORK/ckpts/s3_puremid_corr/resume_step0160.pt --save-steps 220,300 --out-dir $WORK/ckpts/s3_puremid_pcgrad_rnd`
- **GPU 3 (`s3_tightbridge_corr`)**（立刻启动；第 160 步原子写出 `resume_step0160.pt`）：
  `--arch M3-v3-TightBridge --loss-mode corridor --iit-target none --lr-z1-mult 0.60 --lambda-prox-z1 5e-4 --save-steps 120,160,220,300 --save-resume-step 160 --out-dir $WORK/ckpts/s3_tightbridge_corr`
- **GPU 4 (`s3_tightbridge_pcgrad_b1`)**（轮询等待 `$WORK/ckpts/s3_tightbridge_corr/resume_step0160.pt` 出现并检查 GPU 3 存活，立刻接续跑 161–300 步）：
  `--arch M3-v3-TightBridge --loss-mode corridor_iit --iit-target b1 --lr-z1-mult 0.60 --lambda-prox-z1 5e-4 --iit-start-step 160 --iit-alpha-start 0.30 --iit-alpha-end 0.70 --iit-source-margin-gate 2.0 --iit-grad-ratio-cap 0.30 --iit-pcgrad --resume-from $WORK/ckpts/s3_tightbridge_corr/resume_step0160.pt --save-steps 220,300 --out-dir $WORK/ckpts/s3_tightbridge_pcgrad_b1`
- **GPU 5 (`s3_puremid_randmask`)**（立刻启动；PureMid 定位归因对照）：
  `--arch M3-v3-PureMid-Random-Mask --loss-mode corridor --iit-target none --lr-z1-mult 0.0 --lambda-prox-z1 0.0 --save-steps 120,160,220,300 --out-dir $WORK/ckpts/s3_puremid_randmask`
- **GPU 6 (`s3_m3v2_pcgrad_b1`)**（立刻启动；在完全相同 `M3-v2` 上验证 PCGrad + step160 vs Stage 2 `armC`）：
  `--arch M3-v2 --loss-mode corridor_iit --iit-target b1 --lr-z1-mult 0.80 --lambda-prox-z1 1e-4 --iit-start-step 160 --iit-alpha-start 0.30 --iit-alpha-end 0.70 --iit-source-margin-gate 2.0 --iit-grad-ratio-cap 0.30 --iit-pcgrad --save-steps 120,160,220,300 --out-dir $WORK/ckpts/s3_m3v2_pcgrad_b1`

### GPUs 0–6 各臂训练完成后，各自独立顺跑完整评测流水线（与 Stage 2 完全同口径）
对每个 `<TAG>` 与 `<ARCH>`：
1. 复制 `$WORK/ckpts/<TAG>/train_meta.json` 到 `$RES/<TAG>/train_meta.json` 并先增量推送。
2. 独立 `c3_dev` 自动选点：`c35_select_dev_ckpt.py --model $BASE --model-key qwen3_8b --arch <ARCH> --ckpt-dir $WORK/ckpts/<TAG> --batch-size 16 --out-dir $WORK/selected/<TAG>`，复制 `dev_selection.json` 到 `$RES/<TAG>/`。
3. 无损 PEFT LoRA 导出：`c35_export_peft_lora.py --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --base-model $BASE --out-dir $WORK/peft/<TAG>`，复制 `c3_peft_export_meta.json` 到 `$RES/<TAG>/`。
4. HF vs vLLM 原生 LoRA 保真校验：`c35_check_vllm_lora_parity.py --base-model $BASE --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --peft-lora-dir $WORK/peft/<TAG> --dataset c3_dev --limit 40 --max-abs-du-mean 0.50 --no-fail-exit --out-json $RES/<TAG>/vllm_lora_parity.json`。
5. HF 无损 Live `no_think` 全盲集 + 历史集 + 精确 Wiki KL：`c35_eval.py --model $BASE --base-model $BASE --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --alias <TAG> --regime no_think --batch-size 8 --datasets c3_blind_confirm_v2 c2_confirm_bidir c2_stealth_stress --kl-corpus $KL_CORPUS --out-dir $RES/<TAG>/no_think`。
6. vLLM `think`：`c35_eval.py --model $BASE --backend vllm --peft-lora-dir $WORK/peft/<TAG> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v2 c1_proxy_template_v2 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES/<TAG>/think`。
7. vLLM 通用能力护盾（`IFEval 541 + GSM8K 1,319`）：`experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --model $BASE --peft-lora-dir $WORK/peft/<TAG> --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias <TAG> --out-dir $RES/<TAG>/capabilities`。
8. 跨字母重映射 7-Way 因果互换（对核心 4 臂 `s3_puremid_corr`, `s3_puremid_pcgrad_b1`, `s3_puremid_pcgrad_rnd`, `s3_tightbridge_pcgrad_b1` 执行）：`c35_eval_cross_letter_interchange.py --model $BASE --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --alias <TAG> --out-dir $RES/probe2_cross_letter`。
9. 每臂评测完成后立即增量提交推送到 `origin/ACL-exp-c3.5`。

### 步骤 3（全部 7 臂收尾后）：严苛零 `None` 放行总汇总
```bash
$PY experiments/ACL-exp-c3.5/scripts/c35_stats.py \
  --results-root $RES --primary-dataset c3_blind_confirm_v2 --n-boot 5000
```
将 `$RES/c35_stats.json` 与 `$RES/C35_READOUT.md` 推送至 `origin/ACL-exp-c3.5`！
（若 Stage 3 触发 `PASS_PATH1_SUPERIORITY` 或 `PASS_PATH2_EFFICIENCY_PARETO`，我方监控将立刻推进 **Stage 4 独立种子 `seed=1024` 复现**！）
