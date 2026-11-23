# ACL-exp-c3.5 Stage 4 Multi-Seed Finale Plan: Independent `seed=1024` Replication + KL-Optimal `M3-v4` (8 GPUs, ~80 min)

- **Branch**: `ACL-exp-c3.5` (Stage 3 complete at `575fbf2`, Stage 4 ready to auto-run immediately!)
- **Model**: `Qwen3-8B` (Dual-seed `seed=42` + `seed=1024` full replication; identical `c3_train_examples_nothink.jsonl`, identical `lr=2.5e-5, max_steps=300` across ALL arms!)
- **Absolute Red Lines**:
  - NEVER open, read, or evaluate sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`; strictly exclude `converse_error` and `depth-4`.
  - NEVER modify frozen `ACL-exp-a1/`, `ACL-exp-b1/`, `ACL-exp-c1/`, `ACL-exp-c2/`, or `ACL-exp-c3/`.
  - **NEVER commit `.pt` / `.safetensors` to Git**; push only lightweight `json / jsonl / md` results under `experiments/ACL-exp-c3.5/results/stage4/` incrementally to `origin/ACL-exp-c3.5`.

---

## 第一部分 · Stage 3 深度复盘与 Stage 4 多随机种子终局思考日志

Stage 3（`575fbf2`，13 臂全量严格健康门禁 PASS）把过去三轮探索的所有规律彻底收敛成了一套 **干净、硬核、可直接写进 ACL 主会正文的因果证据链**：

1. **`M3-v3-PureMid + Confidence Corridor`（`s3_puremid_corr`，`L0–13` 全部冻结）成为全场最强主线**：
   - **WikiText-2 KL = `0.000958`（相对 `u2_corridor = 0.001362` 直降 `-29.7%`！）**——远超预登记 `-25%` 门禁近 5 个百分点；
   - **所有健康与通用能力护盾高分通过**：**`IFEval = 448/541`**（与 `u2` 的 `449` 仅差 1 题）、`GSM8K = 522/1319`（高于 M0 `496` 达 +26 题）、`Turn-1 = 97.13%`（高于 `u2` `97.04%`）；
   - **主终点与多项最难子集全面抗打或反超 `u2`**：`4-Cell = 49.83%`（`vs u2: −0.72 pp`, `95% CI [−1.95, +0.56]`, `p = 0.29` 无显著差异，`4-Cell CI_low = −1.95 pp` 已通过 Path 2 门禁！），**`Row B = 20.0%`（24/120 vs `u2` 15.8% = `+4.17 pp`！）**、`missing_conjunct = 23.3% > 22.9%`、`Step-3 = 53.6% > 53.2%`、`c2_confirm_bidir = 76.9% > 75.6%`、配对前后置均高于 `u2`（`55.0%/45.0%` vs `51.7%/43.3%`）；
   - **零混杂纯 B1 定位归因一锤定音**：完全相同的纯中层频段（`zone1 = 0`）、相同参数量（`7,040,512`）、相同每层秩多重集下，**`s3_puremid_corr` 显著碾压 `s3_puremid_randmask`：`4-Cell +1.11 pp` `[+0.11, +2.17]` (`p = 0.0336`, 22 胜 8 负)，`Joint T1∧4C +1.17 pp` `[+0.17, +2.28]` (`p = 0.0276`)，`Row B 20.0% vs 17.5%`，且 KL 与 NLL 更低**！
   - **跨字母重映射（同分母 `n = 133`，字面抄字母率 `0.0%`）**：`s3_puremid_corr` 即使没加 IIT，仅靠 B1 架构聚焦就将 `b1_full` 抽象翻转推至 **`91.0%`（vs `u2` `88.7%`）**、`b1_heads_only` 推至 **`60.2%`（vs `u2` `46.6%`，`+13.6 pp`！）**。

2. **诚实关闭“训练期显式 IIT 损失”支线，聚焦 PureMid + Corridor 主航道**：
   - 跨 C3、Stage 2、Stage 3 共 6 组不同强弱/延迟/门控/PCGrad 变体一致表明：虽然 B1-IIT 持续优于随机头 IIT（`C − D = +0.61 pp, p=0.022`、`cos = +0.201 vs +0.067`）且轻微降低 NLL、微升 `think_v2`，但在未见盲集 `c3_blind_confirm_v2` 上训练期显式 IIT 损失总会轻微过拟合训练集路由（稳定小幅 `-0.33 ~ -0.56 pp`），而无 IIT 的 `M3-v3-PureMid + Corridor` 已拿满 Row B `20.0%` 与 `-29.7%` KL！

3. **为什么 `s3_puremid_corr` 在严格 `c35_stats.py` 上离无条件 `PASS_PATH2_EFFICIENCY_PARETO` 仅差 `0.17 pp`（2 题）——以及 `M3-v4` 如何一举击穿！**
   - `s3_puremid_corr` 的 `4-Cell CI_low = −1.95 pp`（已 PASS），`KL = −29.7%`（已 PASS），唯独 `Joint T1∧4C CI_low = −2.17 pp`（距 `−2.00 pp` 仅差 `0.17 pp` = 2 题！）。
   - 对比 `PureMid`（`49.83%`, `KL -29.7%`）与 `TightBridge`（`49.22%`, `KL -24.3%`）的逐题解剖暴露了两条立竿见影的提分路径：
     1. **`M3-v4-PureMid-MLP+`（`6,994,944 = -1.17%`, `max_layer_r = 68 <= 128`）**：**100.0% 纯中层 `L14–27`（`zone1 = 0`, `zone3 = 0`）**，彻底砍掉不再需要的 5 颗顶层头，略收 B1 头秩至 `14/8/4`（仍为 Uniform 中层注意力的 2.9 倍），将全部省出预算灌进 Probe 1 实测“+26.7 pp 仅花 1.4% KL”的 **全部 14 层 Zone-2 MLP（5 核心 `@ r=12` + 其余 9 层 `@ r=8`，总计 `4.33M`！）**——目标 **`-32%` KL + `>= 50.3%` 4-Cell**！
     2. **`M3-v4-Bridge2`（`7,049,216 = -0.41%`, `max_layer_r = 84 <= 128`）**：`TightBridge` 掉分是因为为 4 层 `L10–13` 砍了 Zone-2 MLP 秩（`r6 → r4`）；`M3-v4-Bridge2` **完整保留 `PureMid` 全部 14 层 Zone-2 MLP（`5@r10 + 9@r6`）**，仅用砍掉顶层 5 颗头与微调头秩（`16/10/6`）腾出的 `262K` 给 **仅 2 层 `L12–13` MLP `@ r=4`**（补回 `PureMid` 唯一输给 `u2` 的 `step_1` `-2.5 pp`，同时 KL 仍稳在 **`-26% ~ -28%`**）！

---

## 第二部分 · 8×A800 Stage 4 终局并行矩阵（双随机种子 `seed=42` + `seed=1024`，全 8 卡无等待，~80 分钟完结！）

全部 8 臂均使用 `--model $BASE --model-key qwen3_8b --components-file experiments/ACL-exp-c3.5/data/c35_components_frozen.json --lr 2.5e-5 --max-steps 300 --grad-accum 20 --microbatch 2 --save-steps 120,200,250,300`（纯 Corridor / Base，零 IIT 额外开销，每臂训练仅 ~30 分钟，8 卡从 t=0 同步满载！）：

| GPU | Arm Tag | Seed | Architecture | Loss Mode | Zone-1 超参 | 科学目标 |
|---:|---|---:|---|---|---|---|
| **GPU 0** | `s4_s1024_u1_base` | **`1024`** | `Uniform-LoRA` | `base` | `lr_z1=1.0, prox=0.0` | `seed=1024` 独立复现 **Corridor vs Base (`+4.2 pp`)** |
| **GPU 1** | `s4_s1024_u2_corridor` | **`1024`** | `Uniform-LoRA` | `corridor` | `lr_z1=1.0, prox=0.0` | `seed=1024` 独立最强 Uniform 基线 |
| **GPU 2** | `s4_s1024_puremid` | **`1024`** | `M3-v3-PureMid` | `corridor` | `lr_z1=0.0, prox=0.0` | `seed=1024` 独立复现 **`M3-v3-PureMid`（`-30% KL` + Row B）** |
| **GPU 3** | `s4_s1024_puremid_randmask` | **`1024`** | `M3-v3-PureMid-Random-Mask` | `corridor` | `lr_z1=0.0, prox=0.0` | `seed=1024` 独立复现 **纯 B1 定位胜过随机掩码 (`+1.1 pp`)** |
| **GPU 4** | `s4_s42_v4_mlpplus` | **`42`** | `M3-v4-PureMid-MLP+` | `corridor` | `lr_z1=0.0, prox=0.0` | 100% 纯 Zone-2 高 MLP 架构（`seed=42`） |
| **GPU 5** | `s4_s1024_v4_mlpplus` | **`1024`** | `M3-v4-PureMid-MLP+` | `corridor` | `lr_z1=0.0, prox=0.0` | 同波次 **`seed=1024` 独立复现** `M3-v4-PureMid-MLP+` |
| **GPU 6** | `s4_s42_v4_bridge2` | **`42`** | `M3-v4-Bridge2` | `corridor` | `lr_z1=0.60, prox=5e-4` | 2 层 `L12–13` MLP + 完整 Zone-2 MLP（`seed=42`） |
| **GPU 7** | `s4_s1024_v4_bridge2` | **`1024`** | `M3-v4-Bridge2` | `corridor` | `lr_z1=0.60, prox=5e-4` | 同波次 **`seed=1024` 独立复现** `M3-v4-Bridge2` |

---

## 第三部分 · 8×A800 Stage 4 精确执行命令

- `WORK=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-<SHORT_SHA>`
- `RES=experiments/ACL-exp-c3.5/results/stage4`
- 先跑 CPU 自测：`c35_export_peft_lora.py --selftest-tiny`、`c35_stage4_multiseed_stats.py --selftest-tiny`。
- **8 卡并行（每张 GPU 独立顺跑自己的 `<TAG>` 全流程，约 80 分钟）**：
  1. `c35_train.py ... --out-dir $WORK/ckpts/<TAG>`，复制 `train_meta.json` 至 `$RES/<TAG>/` 并增量推送。
  2. `c35_select_dev_ckpt.py --model $BASE --model-key qwen3_8b --arch <ARCH> --ckpt-dir $WORK/ckpts/<TAG> --batch-size 16 --out-dir $WORK/selected/<TAG>`，复制 `dev_selection.json` 至 `$RES/<TAG>/`。
  3. `c35_export_peft_lora.py --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --base-model $BASE --out-dir $WORK/peft/<TAG>`，复制 `c3_peft_export_meta.json` 至 `$RES/<TAG>/`。
  4. `c35_check_vllm_lora_parity.py --base-model $BASE --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --peft-lora-dir $WORK/peft/<TAG> --dataset c3_dev --limit 40 --max-abs-du-mean 0.50 --no-fail-exit --out-json $RES/<TAG>/vllm_lora_parity.json`。
  5. `c35_eval.py --model $BASE --base-model $BASE --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected/<TAG>/selected_adapter_state.pt --alias <TAG> --regime no_think --batch-size 8 --datasets c3_blind_confirm_v2 c2_confirm_bidir c2_stealth_stress --kl-corpus $KL_CORPUS --out-dir $RES/<TAG>/no_think`。
  6. `c35_eval.py --model $BASE --backend vllm --peft-lora-dir $WORK/peft/<TAG> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v2 c1_proxy_template_v2 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES/<TAG>/think`。
  7. `experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --model $BASE --peft-lora-dir $WORK/peft/<TAG> --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias <TAG> --out-dir $RES/<TAG>/capabilities`。
  8. 每臂完成后立即增量提交推送 `$RES/<TAG>/` 至 `origin/ACL-exp-c3.5`。
- **全部 8 臂收尾后（CPU ~20 秒）：运行双随机种子汇总与严格 Dual-Path 裁决**：
  ```bash
  $PY experiments/ACL-exp-c3.5/scripts/c35_stage4_multiseed_stats.py \
    --stage3-root experiments/ACL-exp-c3.5/results/stage3 \
    --stage4-root $RES \
    --primary-dataset c3_blind_confirm_v2 --n-boot 5000
  ```
  将 `$RES/c35_multiseed_stats.json` 与 `$RES/C35_MULTISEED_READOUT.md` 推送至 `origin/ACL-exp-c3.5`！
