# [ARCHIVED / SUPERSEDED] ACL-exp-c1 · 旧版 Step 2 草案（已由 `STEP2B_FINAL_SHOWDOWN_PLAN.md` 正式取代）

> **⚠️ ARCHIVED — DO NOT EXECUTE**：本文件为 Step 2A 之前的旧版草案（包含旧 `1.68M`、5 类训练、Step-1 权重复用、BF16 合并评测与旧 v2 统计口径），已由 **[`STEP2B_FINAL_SHOWDOWN_PLAN.md`](STEP2B_FINAL_SHOWDOWN_PLAN.md)** 全面取代。保留本文件仅作历史追溯记录。

---

# ACL-exp-c1 · Step 2B 草案（暂缓执行）：8B 机制定位 vs 双随机掩码 / 均匀 LoRA 独立新域与留出类因果对决

- **状态**：**DRAFT / DO NOT EXECUTE YET — 必须等待 `STEP2A_RECIPE_LOCK_TASK.md`（Step 2A）出齐结果并完成三方对齐后，锁定全绿配方再正式下发！**
- **v2 审计修订要点（已同步至脚本）**：
  1. 紧预算组件 `data/c1_components_step2_tight.json` 已修正为 `r_h=3, r_m=2`（`1,518,848` 参数，MLP 占比 `21.6%` 对齐标准档 `19.1%`，消除容量与头/MLP 比例混杂）；
  2. 新增 `data/c1_proxy_template_ood_bases.jsonl`（120 题，0% 模板骨架重叠）与 `data/c1_train_examples_wo_ce_nothink.jsonl`（方案 A：留出 `cross_entity_transfer` 4 类 400 题训练集）；
  3. `scripts/c1_step2_stats.py`（v2）已修复全部 4 项审计问题（`base_id` 精确对齐、消除 `0.0 or -1.0` 假值 bug、恢复 `no_think V/V_hes/V_auth ≥ 90%` 门禁、要求同时战胜双随机均值池及各自单随机掩码 `≥ +3.0 pp`、并加入 `tight_1.52M` vs `standard_6.87M` 互斥选择规则）；
  4. 计划在 Wave 2B.1 与 Wave 2B.2 之间加入 7 分钟 **Wave 2B.15 合并快速失败门禁**。
- **基线提交**：`ACL-exp-c1@288407b`（Step 1 能力门禁复核已完成）
- **硬件与预计总耗时**：**8 × NVIDIA A800 40GB**，两波并行 **总计约 90–110 分钟**（严禁使用 `--backend hf` 跑 `think`，全部 `think / IFEval / GSM8K` 强制走合并权重 + `vLLM`！）

---

## 一、Step 1 终极证据回顾（为什么锁定本轮配方）

截至 `288407b`（[STEP1_CAP_READOUT.md](results/phase0_step1_cap/STEP1_CAP_READOUT.md)），4 组候选权重全部通过 v2 合并一致性门禁（`last_position_kl = 3.4e-8 ~ 1.0e-7`，`top1 = 98.8%–100%`），结论已完全收敛：

| 权重（`Qwen3-8B` `Targeted-Dual`） | `IFEval` (≥430) | `IFEval length` | `GSM8K` v1 (≥470) | `GSM8K` 已解析正确率 | `think` 双对 (`失败`) | `no_think` 双对 (`V_hes/V_auth`) | WikiText-2 KL |
|---|---:|---:|---:|---:|---|---|---:|
| **M0 基座** | `440/541` | `13` | `496/1319` | `496/502` (98.8%) | `69/80` (`0/1200`) | `0/80` (`100%/100%`) | `0` |
| `ctrl_15e5_s150` (`1.5e-4`+人回放 @150) | `366/541` ❌ | `124` ❌ | `413/1319` ❌ | `413/422` (97.9%) | `80/80` (`0/1200`) | `78/80` (`98.8%/97.5%`) | `0.01298` |
| `lr_3e5_s150` (`3e-5`+人回放 @150) | `445/541` ✅ | `15` ✅ | `448/1319` ❌ | `448/458` (97.8%) | `75/80` (`0/1200`) | `58/80` (`91.3%/95.0%`) ✅ | `0.00123` |
| `zero_rep_s150` (`3e-5`+零回放 @150) | **`449/541`** ✅ | `16` ✅ | `457/1319` ⚠️ | `457/464` (98.5%) | `75/80` (`0/1200`) ✅ | `58/80` (`95.0%/98.8%`) ✅ | `0.000952` |
| **`zero_rep_s295`** (`3e-5`+零回放 @295) | **`447/541`** ✅ | **`13` (=M0)** ✅ | **`478/1319`** ✅ | `478/485` (98.6%) | **`75/80` (`0/1200`)** ✅ | `66/80` (`88.8%/85.0%`) | **`0.000985`** |

**锁定结论**：
1. 通用能力崩坏已彻底根治：`zero_rep_s295` 全面通过 `IFEval ≥ 430`（`447/541`，超 M0 `+7` 题，`length = 13` 与 M0 一致）、`GSM8K ≥ 470`（`478/1319`）、`think` 零失败（`0/1,200`）与 `think V_hes/V_auth = 100%`。
2. **零配方漂移原则**：Step 2 全员严格沿用已通过门禁的 **`S1_zero_rep` 完全一致超参**：
   `--lr 3e-5 --epochs 3 --max-steps 295 --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl --zero-replay-loss --save-steps 150,295 --microbatch 2 --grad-accum 12 --seed 42 --scaling-mode rslora --base-scale 2.0 --anchor-rank 8 --dropout 0.05 --lambda-margin 0.25 --margin-m 4.0`。
   - 直接复用 Step 1 已验收的 `S1_zero_rep`（SHA-256 锁定）作为标准预算 `std_targeted_dual`，确保同配方零混杂对比并节省重复训练/能力评测算力。

---

## 二、核心科学问题与实验设计

**Step 2 唯一要回答的决胜问题（14B 投资前门禁）**：
> 在已验证通过全部通用能力与安全门禁的健康训练区间（`lr=3e-5 + 零回放`，`KL ≈ 0.001`）下，**B1 机制定位电路 `Targeted-Dual` 在独立新域代理 OOD 集（`c1_proxy_ood`，240 题）上是否显著战胜同层同参双随机掩码（`Random-LM` + `Random-LM-seed2`）和全层均匀 LoRA（`Uniform-LoRA`）？**

### 1. 独立新域代理 OOD 评测集：`c1_proxy_ood_bases.jsonl`（240 题，已构建并通过全量捷径审计）
- **生成脚本**：`experiments/ACL-exp-c1/scripts/c1_build_proxy_ood.py`
- **产物与清单**：`data/c1_proxy_ood_bases.jsonl`（`sha256 = 53ff2aebf8b10949d5757126b80d8059239d077b05e4828287721e1cbc705556`）、`data/c1_proxy_ood_manifest.json`
- **严格零泄漏保证**：
  1. **从未打开或读取** `c1_test_id_bases.jsonl` 与 `c1_test_ood_bases.jsonl`（`sealed_test_sets_touched = false`）。
  2. 使用 **6 个全新领域词库**（`airspace_corridor`, `cleanroom_fab`, `freight_customs`, `orbital_relay`, `coldchain_depot`, `mining_hoist`），领域名、实体前缀、实体名、分区、守卫、指标、9 条谓词与 4 个结果标签与 **A1 和 C1 原 6 域 100% 严格互斥**（构建时代码断言验证）。
  3. 仅覆盖 5 个已见谬误类与 depth 2/3（`120 context_flip + 120 argument_flip`；`120 depth-2 + 120 depth-3 hard`），**严格排除** `converse_error` 与 `depth-4`，保护封存 `c1_test_ood` 的两个核心 OOD 维度 100% 未被探查。
  4. **全部 4 项捷径与 G0 审计已通过**：`(domain, a0)->at = 0.350`、`cross_entity/AF entity-match = 0.500`、`threshold/AF obs_val AUC = 0.5686`、`G0 overall TF-IDF AUC = 0.5027`、`CF byte match = 1.000`、`AF AUC = 0.5047`。
- **统计效力**：240 题在 `no_think` 下预计产出 ~130–150 题 M0 双侧 Turn-1 合格基数，在 `think` 下产出 ~240 题合格基数（将 80 题 dev 的 ±12 pp 抽样方差压缩至 ~±4 pp）。

### 2. 双预算档同时对照（一次 8 卡波次打穿两个关键科学假设）
为了同时检验 **(a) 标准预算 `6.87M` 在非过训区间下是否已拉开定位差距**，以及 **(b) 预注册紧预算 `1.68M`（压缩 4×，占 8B 总参数仅 `0.0205%`）下定位优势是否进一步放大**，Step 2 在同一批 8 卡并行中覆盖两档预算（组件文件 `data/c1_components_step2_tight.json` 由 `scripts/c1_export_step2_tight_components.py` 从冻结 `c1_components_frozen.json` 确定性派生，原冻结文件 0 字节改动）：

| 预算档 | 代号 | 组件文件 / Arm | 结构细节 | 可训练参数量 | 来源 |
|---|---|---|---|---:|---|
| **Standard (`6.87M`, `0.084%`)** | `std_targeted_dual` | `c1_components_frozen.json` / `Targeted-Dual` | B1 47 heads (`r=14`) + 5 MLPs (`r=8`) | `6,869,504` | **直接复用 Step 1 `S1_zero_rep`** (`@295` SHA-256 锁定) |
| Standard | `std_random_lm` | `c1_components_frozen.json` / `Random-LM` | 同层随机 47 heads (`r=14`) + 同带随机 5 MLPs (`r=8`), seed=2027 | `6,869,504` | GPU 0 训练 |
| Standard | `std_random_lm_seed2` | `c1_components_frozen.json` / `Random-LM-seed2` | 独立第二套随机掩码, seed=2028 | `6,869,504` | GPU 1 训练 |
| Standard | `std_uniform_lora` | `c1_components_frozen.json` / `Uniform-LoRA` | 全 36 层 `q/o/gate/down`, `r=4` | `7,077,888` (+3.0%) | GPU 2 训练 |
| **Tight (`1.68M`, `0.0205%`)** | `tight_targeted_dual` | `c1_components_step2_tight.json` / `Targeted-Dual-Tight` | B1 同 47 heads (`r=3`) + 同 5 MLPs (`r=3`) | **`1,682,688`** (-4.9% vs Unif) | GPU 3 训练 |
| Tight | `tight_random_lm` | `c1_components_step2_tight.json` / `Random-LM-Tight` | seed=2027 同掩码 (`r_h=3, r_m=3`) | `1,682,688` | GPU 4 训练 |
| Tight | `tight_random_lm_seed2` | `c1_components_step2_tight.json` / `Random-LM-seed2-Tight` | seed=2028 同掩码 (`r_h=3, r_m=3`) | `1,682,688` | GPU 5 训练 |
| Tight | `tight_uniform_lora` | `c1_components_step2_tight.json` / `Uniform-LoRA-Tight` | 全 36 层 `q/o/gate/down`, `r=1` | `1,769,472` | GPU 6 训练 |

> **rsLoRA 缩放不变性说明**：`c1_modules.py` 采用 `s(r) = 2 * sqrt(8 / r)`，压缩至 `r_h=3, r_m=3` 时初始方差 `r * s(r)^2` 与标准档保持恒定，因此两档预算可直接采用完全相同的 `lr = 3e-5` 无需额外调参。

---

## 三、⚠️ 严格安全护栏（必须遵守）

1. **严禁触碰封存测试集**：所有 `c1_eval.py` 命令**必须显式传 `--datasets c1_proxy_ood c1_dev`（或 `--datasets c1_proxy_ood`）**；绝对禁止省略 `--datasets`！
2. **严禁使用 `--backend hf` 跑 `think`**：所有 `think` 评测**必须**先经 `c1_merge_and_check.py` 合并验收，再用 `--backend vllm` 跑（单组 240 题 `c1_proxy_ood` + 80 题 `c1_dev` 共 4,800 条约 **18–22 分钟**；绝不再允许 10 小时/次的 HF 逐条生成！）。
3. **禁止修改任何旧结果目录**：Step 2 全部新产物统一写入 **`experiments/ACL-exp-c1/results/step2_8b_showdown/`**；严禁改动 `phase0/`、`phase0_step0/`、`phase0_step1/`、`phase0_step1_cap/`。
4. **空闲卡与零 OOM 检查**：每张卡启动前确认无外部进程占显存；临时合并目录 `merged_hf` 放在 `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2-merged/<TAG>`，评测完可清理（不入库），仅保留 `merge_check.json`、`eval_summary_*.json`、`records_*.jsonl` 与 `capabilities/{summary.json,items.jsonl}`。
5. **CPU 自检门禁**：开跑前先执行一次 CPU 自检（~15 秒），任一失败立即中止：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_merge_and_check.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_step2_stats.py --selftest
   ```

---

## 四、8 卡两波并行执行清单（总耗时 ~85 分钟）

### 预注册主评测点与次级轨迹点
- **主评测点（冻结）**：**`step 295`**（Step 1 已验证 `zero_rep_s295` 同时通过 `IFEval=447 ≥ 430`、`GSM8K=478 ≥ 470`、`think=0/1200`）。全部 8 组在 `@295` 上跑完整 **合并校验 + `no_think` (`c1_proxy_ood`, `c1_dev`, `WikiText-2 KL`) + `vLLM think` (`c1_proxy_ood`, `c1_dev`) + `IFEval` + `GSM8K`**。
- **次级轨迹诊断（仅轻量 `no_think`，不跑耗时 `think/IFEval/GSM8K`）**：训练时同时保存 `adapter_state_step0150.pt`，仅对 8 组 `@150` 补测轻量 `no_think`（`c1_proxy_ood` + `c1_dev`，单组约 4 分钟），写入 `<TAG>/step0150_no_think/`，用于追踪 `V_hes/V_auth` 与定位优势随步数的演化。

### Wave 2.1（8 卡并行，约 35–40 分钟）：7 组训练 + GPU 7 同步跑 M0 基线与 `std_targeted_dual` 复用评测
- **GPUs 0–6（各 1 卡并行训练 295 步，约 35–40 分钟）**：
  - GPU 0：`std_random_lm` (`--components-file data/c1_components_frozen.json --arm Random-LM`)
  - GPU 1：`std_random_lm_seed2` (`--components-file data/c1_components_frozen.json --arm Random-LM-seed2`)
  - GPU 2：`std_uniform_lora` (`--components-file data/c1_components_frozen.json --arm Uniform-LoRA`)
  - GPU 3：`tight_targeted_dual` (`--components-file data/c1_components_step2_tight.json --arm Targeted-Dual-Tight`)
  - GPU 4：`tight_random_lm` (`--components-file data/c1_components_step2_tight.json --arm Random-LM-Tight`)
  - GPU 5：`tight_random_lm_seed2` (`--components-file data/c1_components_step2_tight.json --arm Random-LM-seed2-Tight`)
  - GPU 6：`tight_uniform_lora` (`--components-file data/c1_components_step2_tight.json --arm Uniform-LoRA-Tight`)
  - 统一训练命令模板（注意 `--no-merge-save`，训练完交由 `c1_merge_and_check.py` 统一校验合并）：
    ```bash
    CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_train.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b \
      --components-file <COMP_FILE> --arm <ARM_NAME> \
      --train-file experiments/ACL-exp-c1/data/c1_train_examples_nothink.jsonl \
      --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl \
      --zero-replay-loss --lr 3e-5 --epochs 3 --max-steps 295 --save-steps 150,295 \
      --microbatch 2 --grad-accum 12 --seed 42 --no-merge-save \
      --out-dir <WORK_DIR>/ckpts/<TAG>
    ```
- **GPU 7（与训练同步进行，约 30 分钟）**：
  1. 跑 **M0 (`Qwen3-8B` 基座)** 在 `c1_proxy_ood` + `c1_dev` 的 `no_think`（HF）与 `think`（vLLM），并导出冻结的 M0 双侧 Turn-1 合格集合：
     - `results/step2_8b_showdown/M0/no_think/eligible_ids.json`
     - `results/step2_8b_showdown/M0/think/eligible_ids.json`
  2. 复用 Step 1 已验证的 **`S1_zero_rep / adapter_state_step0295.pt`**（必须校验 SHA-256 = `9e1979532b6d0e3b9eb67ed14660f0aedcd5f781ab4bf82aae1f5e0db24ee79a`），经 `c1_merge_and_check.py` 合并校验后，在 `c1_proxy_ood` + `c1_dev` 上补测 `no_think`（含 `WikiText-2 KL`）与 `vLLM think`；其 `IFEval`（`447/541`）和 `GSM8K`（`478/1,319`）可直接从 `results/phase0_step1_cap/zero_rep_s295/capabilities/` 完整复制进 `results/step2_8b_showdown/std_targeted_dual/capabilities/`（SHA-256 与权重完全一致，无需重复跑 18 分钟）。

### Wave 2.2（7 卡并行，约 45–50 分钟）：7 组新训练权重 `@295` 合并校验 + 全量评测（+ `@150` 轻量 `no_think`）
每张 GPU 负责 1 组 `<TAG>`，严格按以下顺序串行执行（均依赖 Wave 2.1 GPU 7 产出的 `M0/*/eligible_ids.json`）：
1. **合并与 v2 一致性门禁（~1 分钟）**：调用 `scripts/c1_merge_and_check.py`，产出 `<TAG>/merge_check.json`（不通过立即停止该组并报错）。
2. **`no_think` + `WikiText-2 KL`（~6 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model <TMP_MERGED_DIR> --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --alias qwen3_8b__<TAG> --regime no_think --backend hf \
     --datasets c1_proxy_ood c1_dev \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2_8b_showdown/M0/no_think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2_8b_showdown/<TAG>/no_think
   ```
3. **`vLLM think`（~18 分钟，严禁 `--backend hf`！）**：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model <TMP_MERGED_DIR> --alias qwen3_8b__<TAG> --regime think --backend vllm \
     --max-new-tokens 3072 --datasets c1_proxy_ood c1_dev \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2_8b_showdown/M0/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2_8b_showdown/<TAG>/think
   ```
4. **`IFEval` + `GSM8K`（~18 分钟）**：严格复用 `results/phase0/capabilities/spec.json` 的 vLLM 评测口径，写入 `<TAG>/capabilities/{summary.json,items.jsonl}`。
5. **轻量 `@150` 轨迹补测（~4 分钟，仅 `no_think`，不跑 `think/IFEval/GSM8K`）**：用 `--adapter-state <WORK_DIR>/ckpts/<TAG>/adapter_state_step0150.pt --regime no_think --backend hf --datasets c1_proxy_ood c1_dev`（无需合并），写入 `<TAG>/step0150_no_think/`。

### Wave 2.3（CPU 10 秒）：自动统计汇总与推送
全部 8 组完成后，执行：
```bash
python experiments/ACL-exp-c1/scripts/c1_step2_stats.py \
  --results-dir experiments/ACL-exp-c1/results/step2_8b_showdown
```
自动产出 **`results/step2_8b_showdown/step2_stats.json`** 与 **`results/step2_8b_showdown/STEP2_READOUT.md`**，将整个 `results/step2_8b_showdown/`（不含大型 `merged_hf` 权重）提交并推送至 `origin/ACL-exp-c1`。

---

## 五、预注册 14B 通行门禁（GO / NO-GO Decision Rule，执行前冻结）

由 `scripts/c1_step2_stats.py` 自动判定；**标准档（`6.87M`）或紧预算档（`1.68M`）中至少有一档同时满足以下全部 3 项条件，才宣布 `GO → 正式启动 Qwen3-14B 主实验`**（并以获胜预算档作为 14B 主预算）：

1. **条件 1（全部通用能力与安全门禁通过）**：`Targeted-Dual` 同时满足：
   - `merge_check.passed == True`
   - `IFEval prompt_strict ≥ 430/541`（相对 M0 `440/541` 降幅 `≤ 2.0 pp`）
   - `GSM8K v1 ≥ 470/1,319`（相对 M0 `496/1,319` 降幅 `≤ 2.0 pp`）
   - `think` 解析失败率 `≤ 2.0%`、`think V_hes & V_auth ≥ 90%`、`think` 双侧 Turn-1 `≥ 95%`
   - `WikiText-2 KL ≤ 0.005`
2. **条件 2（在独立新域 `c1_proxy_ood` 上显著战胜双随机掩码池）**：在 M0 共享合格分母（同步汇报全 240 题）上，对比 `Random-LM-Pool = 0.5*(Random-LM + Random-LM-seed2)`：
   - 在 `no_think`（点估计 `≥ +5.0 pp` 且 95% 配对 Bootstrap CI 严格 `> 0`）**或** `think`（点估计 `≥ +3.0 pp` 且 95% CI 严格 `> 0`）上取得显著正收益，且另一制度无显著退化（点差 `≥ -2.0 pp` 且 CI 上界 `≥ 0`）；
   - **并且** `no_think` 连续逻辑验证边界 `u(V) - u(F)` 相对 `Random-LM-Pool` 的 95% 配对 Bootstrap CI 严格 `> 0`。
3. **条件 3（战胜或 Pareto 支配等参数 `Uniform-LoRA`）**：
   - 在 `c1_proxy_ood` 的 `no_think` 或 `think` 双对率上显著战胜 `Uniform-LoRA`（95% CI `> 0`）；**或**
   - 实现 Pareto 支配（双制度双对率不落后 `Uniform-LoRA` 超过 `2.0 pp`，同时 `WikiText-2 KL` 更低且 `IFEval` 不落后超过 3 题）。

### 预注册未通过分支（HOLD 分支，绝不盲目上 14B）
- **若 `Standard (6.87M)` 与随机打平，但 `Tight (1.68M)` 显著胜出**：证实 `6.87M` 属于过容预算，正式将 14B 主实验锁定为紧预算档（`~2.61M`, `0.018%` of 14B）！
- **若两档均健康但 `Targeted-Dual` 仍未能拉开 `Random-LM-Pool`**：暂停上 14B，在 8B 上检查：(1) `lr = 5e-5` 与 `accept_weight` 平衡；(2) `Targeted-Dual` 相对 `Targeted-Form` / `Pinpoint-DLA` 在 `context_flip` 上的分层差异（查明随机掩码是否靠非目标层冗余通道学到同样映射，或需按 B1 Top-K 精简至核心高因果权重子集）。
