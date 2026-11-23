# ACL-exp-c3.5 Stage 2 Showdown Plan: Probe-1-Guided `M3-v2` + Soft-Gated IIT (5-Arm 8B Showdown)

- **Branch**: `ACL-exp-c3.5` (Stage 1 complete at `b275221`, Stage 2 ready to execute)
- **Model**: `Qwen3-8B` (single-seed `seed=42` decisive sprint; 14B remains paused pending strict Stage 2 Dual-Path PASS + `seed=1024` replication)
- **Absolute Red Lines**:
  - NEVER open, read, or evaluate sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`; strictly exclude `converse_error` and `depth-4`.
  - NEVER modify frozen `ACL-exp-a1/`, `ACL-exp-b1/`, `ACL-exp-c1/`, `ACL-exp-c2/`, or `ACL-exp-c3/`.

---

## 第一部分 · 科学思考与推演日志（Stage 1 实证如何彻底重写我们的架构与算法）

在 C3（`96f2859`）收尾审计后，我们与 Astra 共识把 C3.5 拆成两步：**先用零重训的 Stage 1 探针把所有物理事实和盲集生成器问题查清，再把算力精准押注到 Stage 2 的 5 个关键新臂**。Stage 1（`b275221`）刚刚全部跑完，带来了 **三条颠覆性但彼此严丝合缝的实证发现**：

### 发现一 · Probe 1（子投影消融 + 精确 WikiText-2 KL）直接推翻了 `M3-v1` 的底层/顶层假设！

在 C3 结束时，我们最初猜测：`M3-v1` 在顶层浪费预算导致高 KL，底层 Q/O 是扛 Row B 的主力。**Stage 1 Probe 1（`results/stage1/probe1_subproj/SUBPROJECTION_PROBE.md`）彻底推翻了这个直觉**：

| 模型 & 消融配置 | 关闭的 `(Zone, Proj)` | `c3_dev` 4-Cell (`180`) | `c3_dev` Row B (`30`) | `c2` Row B (`36`) | `c3_dev` Step-3 (`40`) | Wiki KL | `ΔKL vs full` |
|---|---|---:|---:|---:|---:|---:|---:|
| `u2_full` | `none` | **70.0%** (126) | 5/30 | **15/36** (41.7%) | 75.0% | `0.001381` | `0.0%` |
| `u2_no_z1_qo` | `Z1:attn` | 64.4% (-5.6) | 5/30 (0) | **14/36 (-1)** | 60.0% (-15.0) | `0.001354` | `-2.0%` |
| `u2_no_z1_mlp` | `Z1:mlp` | 60.0% (-10.0) | 5/30 (0) | **5/36 (-10!)** | **52.5% (-22.5!)** | `0.001253` | `-9.3%` |
| `u2_no_z1_all` | `Z1:attn + mlp` | 57.2% (-12.8) | 3/30 (-2) | 3/36 (-12) | 50.0% (-25.0) | `0.000947` | **`-31.4%`** |
| `u2_no_z2_qo` | `Z2:attn` | 46.1% (-23.9) | 2/30 (-3) | 5/36 (-10) | 35.0% (-40.0) | `0.001336` | `-3.3%` |
| `u2_no_z2_mlp` | `Z2:mlp` | **43.3% (-26.7!)** | **2/30 (-3)** | **2/36 (-13!)** | **32.5% (-42.5!)** | `0.001361` | **`-1.4%`** |
| `u2_no_z3_mlp` | `Z3:mlp` | **70.0% (0.0!)** | **6/30 (+1!)** | 14/36 (-1) | 72.5% (-2.5) | `0.001370` | `-0.8%` |
| `u2_no_z3_all` | `Z3:attn + mlp` | **70.0% (0.0!)** | **6/30 (+1!)** | 14/36 (-1) | 72.5% (-2.5) | `0.001375` | `-0.4%` |
| `m2_full` | `none` | **63.3%** (114) | 4/30 | 10/36 (27.8%) | 57.5% | `0.001261` | `0.0%` |
| `m2_no_z3_mlp` | `Z3:mlp` (`[29,32,35]`) | 62.8% (-0.5) | **5/30 (+1!)** | 10/36 (0) | 57.5% (0.0) | `0.001247` | `-1.1%` |
| `m2_no_z1_qo` | `Z1:attn` (`L0–13`) | 62.8% (-0.5!) | 4/30 (0) | 8/36 (-2) | 57.5% (0.0!) | **`0.000956`** | **`-24.2%`** |
| `m2_no_z2_mlp` | `Z2:mlp` (5 B1 MLPs) | **32.8% (-30.5!)** | **0/30 (-4!)** | **1/36 (-9!)** | **20.0% (-37.5!)** | `0.001217` | **`-3.5%`** |

**我们的深层推演**：
1. **Zone 3（顶层 `L28–35`）MLPs 是彻底的零收益废预算**：关掉 `u2` 全顶区掉 `0.0 pp` 且 Row B 还升 1 题；关掉 `m2` 的 `[29,32,35]` 顶层 MLP，Row B 和 Step-3 零损失甚至微升！说明 `M3-v1` 把 **22.2%（`1.56M`）参数** 投在顶层纯属浪费。
2. **Zone 1（底层 `L0–13`）存在剧烈的“早层 KL 代价 vs 晚层 MLP 绑定收益”分化**：`m2` 关掉 `L0–13` Q/O 立刻砍掉 **24.2% 的 Wiki KL**（`u2` 全关 Zone 1 砍掉 **31.4% KL**），因为早层 `L0–7` 处理的是通用词法/句法表示，任何微调都在推高 WikiText-2 KL；然而真正驱动 `u2` 上 Row B（`+10/36`）与 Step-3（`+22.5 pp`）的是 **`Z1:MLP`**（`M3-v1` 里为 0！）。
3. **Zone 2（中层 `L14–27`）MLP 是全网性价比最逆天的推理引擎**：在 `u2` 上扛起 **`+26.7 pp` dev、`+13/36` Row B、`+42.5 pp` Step-3**，却**只花 `1.4%` 的 Wiki KL**！而 `M3-v1` 只开了 5 层 Zone-2 MLP，漏掉了其余 9 层！

**因此我们锁定了 `M3-v2`（100% 由 B1 张量自动导出，见 `data/c35_components_frozen.json`）**：
- **早期 `L0–7` 完全冻结（`0` 参数）**：物理隔绝通用词法/句法层漂移，直指 **KL 降低 ≥25%**；
- **晚底层桥接 `L8–13`（6 层）**：`MLP @ r=4`（`786K`）+ `Dense Q/O @ r=2`（`197K`）= `983K`（13.8%），既补齐 `Z1:MLP` 的 Row B/Step-3 绑定，又不碰高 KL 的 `L0–7`；
- **全中层核心 `L14–27`（14 层，占总预算 81.5% = `5.80M`！）**：
  - **全部 14 层 Zone-2 MLP 全部打开**：5 层 B1 `N_CF` 核心 MLP `[18,20,22,23,27]` `@ r=8` + 其余 9 层 Zone-2 MLP `@ r=4`（总计 `2.49M`，比 `Uniform-LoRA` 中层 MLP 多 35% 容量，且几乎零 KL！）；
  - **42 颗 B1 中层头 3 档分配**（`10@16, 10@10, 22@6`，冲突头钳位到最低档 `r=6`，总计 `3.31M` = `Uniform` 中层注意力的 3.6 倍！）；
- **顶层 `L28–35` 极简收敛**：**全部删除 3 层顶层 MLP**（`0` 参数），10 颗 B1/DLA 顶层头统一钳位至最小档 `r=4`（`338K`，仅 4.7%，保留顶层 `form_only_heads` 供 B1-IIT 可微调）；
- **精确参数对齐**：`qwen3_8b` 总计 **`7,122,944`**（相对 `Uniform-LoRA` `7,077,888` 仅 **`+0.64%`**，单层最大 rank `80 <= 128`）；`qwen3_14b` 总计 **`10,421,760`**（相对 `10,485,760` 仅 **`-0.61%`**，`max_layer_r = 96 <= 128`）。

---

### 发现二 · Probe 2（跨字母重映射 7-Way 互换）：彻底证明 B1 传递的是抽象逻辑有效性而非字面字母！

在 `cross_letter_remapped`（源题与目标题选项字母完全不相交 `{a0, at} ∩ {o1, o2} = ∅`）的 **严格同分母（`n = 126` 对）** 比较下（`results/stage1/probe2_cross_letter/`）：

| 干预组件 (`cross_letter_remapped`, 同分母 `n=126`) | `u1_base` | `u2_corridor` | `u3_b1_iit` | `m2_corridor` | 字面抄字母率 (`literal_copy`) |
|---|---:|---:|---:|---:|---:|
| **`b1_full`** (B1 头 + 5 B1 MLP) | 88.9% (112) | 88.9% (112) | **93.7% (118)** | 88.9% (112) | **0.0%** |
| `b1_mlp_only` (仅 5 B1 MLP) | 84.1% (106) | 84.1% (106) | **89.7% (113)** | 83.3% (105) | **0.0%** |
| **`b1_heads_only`** (仅 B1 头，无 MLP!) | 50.0% (63) | 48.4% (61) | **62.7% (79)** | **68.3% (86)** | **0.0%** |
| `b1_heads_rnd_mlp` (B1 头 + 随机 MLP) | 46.0% (58) | 49.2% (62) | 59.5% (75) | **63.5% (80)** | **0.0%** |
| `rnd_heads_b1_mlp` (随机头 + 5 B1 MLP) | 77.8% (98) | 78.6% (99) | 88.1% (111) | 81.0% (102) | **0.0%** |
| **`rnd_full`** (随机头 + 随机 MLP) | 4.0% (5) | 4.0% (5) | **0.8% (1)** | **0.0% (0)** | **0.0%** |

**核心结论**：
1. **所有模型、所有组件上字面抄字母率均为精确 `0.0%`**，同时跨字母抽象逻辑翻转率高达 **88.9%–93.7%**（全随机仅 **0.0%–4.0%**），证明 B1 环路编码的是**与字母表解耦的抽象逻辑真假决策变量**。
2. **B1 注意力头自身携带强因果决策信号**：即使从未见过 B1 的 `u1/u2`，加上 B1 头比 `rnd_heads_b1_mlp` 仍净增 **`+10.3 ~ +11.1 pp`**；而 `m2_corridor`（`68.3%`，`+19.9 pp`）与 `u3_b1_iit`（`62.7%`，`+14.3 pp`）大幅把抽象因果路由收束进 B1 头，同时把 `rnd_full` 压到 **`0.0%–0.8%`**！

---

### 发现三 · 修复后的盲集 `c3_blind_confirm_v2` 揭示了自信走廊的真实大增益（`+4.17 pp`）！

修复 CE 干扰句遗漏（`240/240`）与 `s1 ∧ s2` 独立双前提 DAG 后，在 `900` 个独立生成单元（`1,080` 题）上（`results/stage1/baselines_v2/C35_READOUT.md`）：

| 基线臂 | 4-Cell (UW) | **Joint T1∧4C (UW)** | `DC_orig / DC_rev` | Row B (4C / Joint) | Step 1/2/3 | Turn-1 | 4-Opt NLL |
|---|---:|---:|---|---:|---|---:|---:|
| `M0` | 0.00% | 0.00% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0 / 0.0 / 0.0% | 32.9% | 13.784 |
| `u1_base` | 46.39% | 45.61% | 47.7% / 55.6% | 15.0% / 15.0% | 38.2 / 48.6 / 50.0% | 96.9% | 0.894 |
| **`u2_corridor`** | **50.56%** | **49.89%** | **51.6% / 57.3%** | **15.8% / 15.8%** | **43.2 / 52.5 / 53.2%** | 97.0% | 0.783 |
| `m2_corridor` | 45.61% | 44.94% | 47.7% / 52.8% | 10.8% / 10.8% | 37.1 / 50.4 / 47.1% | **97.6%** | **0.764** |

- **`u2_corridor − u1_base = +4.17 pp` `[+2.94, +5.50]` (`p < 0.0001`, 49 胜 3 负)**，`Joint T1∧4C = +4.28 pp` `[+3.00, +5.67]`！这证明在干净数据上，**置信走廊+语气不变性在 Uniform 强基线上也稳定暴涨 +4.17 pp**。
- **`m2_corridor`** 已经在 Turn-1（`97.6% > 97.0%`）、4-Opt 校准 NLL（`0.764 < 0.783`）与 Wiki KL（`0.001261 < 0.001381`）上优于 `u2_corridor`，唯独因为缺了 `Z1:MLP` 与 9 层 `Z2:MLP` 而在 `Step 1/3` 与 `Row B` 落后 `-4.94 pp`。这正好是 **`M3-v2`** 精准发力的位置！

---

## 第二部分 · Stage 2 精准 5 臂实验矩阵与因果拆解设计

复用 `c3_train_examples_nothink.jsonl` 与已有的 `M0 / u1_base / u2_corridor / m2_corridor` 对照，仅训练 **5 个精准新臂（`seed=42`, `300` 步）**：

| 新臂标签 | 架构 | 损失与 IIT 配方 | Zone-1 超参 | 第 120 步状态来源 | 核心因果对比与回答的问题 |
|---|---|---|---|---|---|
| **`armA_m3v1_relax`** | `M3-v1` (7.03M) | Corridor | `lr_z1=0.80, prox=1e-4` | 从 0 跑 300 步 | **`A − m2`**：纯放宽底层束缚的净增益 |
| **`armB_m3v2_corridor`** | **`M3-v2`** (7.12M) | Corridor | `lr_z1=0.80, prox=1e-4` | 从 0 跑 300 步（第 120 步原子存 `resume_step0120.pt`） | **`B − A`**：纯 Probe-1 参数重分配净增益；**`B − u2`**：`M3-v2` 对比最强 Uniform（含 KL 降幅） |
| **`armC_m3v2_soft_b1_iit`** | **`M3-v2`** (7.12M) | Corridor + **Soft-Gated B1-IIT** | `lr_z1=0.80, prox=1e-4` | **比特级复用 `B` 第 120 步状态、AdamW 动量与样本顺序** | **`C − B`**：纯 Soft-Gated B1-IIT 净增益；**`C − u2`**：旗舰对比最强 Uniform |
| **`armD_m3v2_soft_rnd_iit`** | **`M3-v2`** (7.12M) | Corridor + **Soft-Gated `rnd_head_1` IIT** (固定 5 B1 MLP) | `lr_z1=0.80, prox=1e-4` | **比特级复用 `B` 第 120 步状态、AdamW 动量与样本顺序** | **`C − D`**：零架构/零轨迹混杂的**纯 B1 头定位净增益** |
| **`armE_unif_soft_b1_iit`** | `Uniform-LoRA` (7.08M) | Corridor + **Soft-Gated B1-IIT** | `lr_z1=1.00, prox=0.0` | 从 0 跑 300 步（前 120 步纯 Corridor 与 `u2` 同序） | **`E − u2`**：验证柔和 IIT 是否治愈 C3 硬 IIT 掉分（`-2.66 pp`）；**`C − E`**：架构对决 |

### Soft-Gated IIT 的 5 项核心算法升级（已全部通过 `c35_train.py --selftest-tiny` 0 误差校验）
1. **延迟启动 + 第 120 步比特级分叉**：前 120 步纯跑走廊损失建立基础正确率，第 120 步后才启动 IIT；`B/C/D` 共享完全相同的 `resume_step0120.pt`（含 `adapter_state` 与 `AdamW` 一二阶矩），彻底消除前 120 步轨迹混杂。
2. **真实 4 选源端 Margin 门控（`m_gate = 2.0`）**：Verification IIT 仅在源题真实四选 margin `logit(gold) - max_{c != gold} logit(c) >= 2.0` 时触发（Form IIT 仅在目标题 `>= 2.0` 时触发），并逐子群（`all`, 5 谬误类, `ce_row_A`, `ce_row_B`, `step_1/2/3`）记录精确触发率。
3. **软激活插值配对软方向目标**：第 121→300 步线性抬升 `α: 0.30 → 0.70`（`h_tgt ← (1-α) h_tgt + α h_src`），目标设定为数学一致的 `target_u = (1-α) * clamp(u_tgt_orig, -10, 10) + α * (y_src * 3.5)`，彻底杜绝弱插值却逼模型 100% 硬翻转的矛盾梯度。
4. **同 Batch 独立梯度夹角遥测**：每一步先反传主任务并快照 `g_main`，清零后再在同一 batch 上反传 IIT，记录精确 `norm_main`、`norm_iit_raw`、全局及 Zone-2 余弦相似度 `cos(g_main, g_iit)`，直接判定旧硬 IIT 是“量级过大”还是“方向冲突”。
5. **全局裁剪前 `0.30×` 相对范数独立限幅**：强制 `||g_iit_applied|| <= 0.30 * ||g_main||` 后再合并 `g_main + g_iit_applied` 进入全局 `max_grad_norm = 2.0` 裁剪，绝不让 IIT 挤掉主任务更新量！

---

## 第三部分 · 8×A800 Stage 2 完整执行流程（约 85–95 分钟全自动跑完）

- **GPU 0–4**：并行执行 5 个新臂（`A, B, C, D, E`）的训练 → `c3_dev` 自动选点 → 无损 PEFT LoRA 导出 → HF/vLLM 保真校验 → `no_think` 全盲集 (`c3_blind_confirm_v2 + c2_confirm_bidir + c2_stealth_stress` + WikiText-2 KL) → vLLM `think` (`c3_blind_think_v2 + c1_proxy_template_v2`) → `IFEval + GSM8K` 能力护盾 → 跨字母重映射 7-Way 因果互换！
  - 注：GPU 2 (`armC`) 与 GPU 3 (`armD`) 自动等待 GPU 1 (`armB`) 在第 ~12 分钟原子写出 `resume_step0120.pt` 后立刻接续跑 121–300 步（带父进程存活性检测，绝不死等）。
- **GPU 5–6（在训练期间同步并行，零额外耗时）**：把 Stage 1 的 `M0, u1_base, u2_corridor, m2_corridor` 完整组装进 `results/stage2/`（复用 C3 已测的 `reference_kl`, `capabilities/summary.json`, `vllm_lora_parity.json`），并仅补跑 `u1_base, u2_corridor, m2_corridor` 在 `c3_blind_think_v2` 上的快速 vLLM `think`（每臂 ~2 分钟），保证全部 9 臂在 `c35_stats.py` 严苛零 `None` 放行模式下拥有 100% 完整指标！
- **CPU 汇总**：运行 `c35_stats.py --results-root experiments/ACL-exp-c3.5/results/stage2 --primary-dataset c3_blind_confirm_v2 --n-boot 5000`（开启完整严格健康门禁与双路径裁决！），只提交轻量 `json/jsonl/md` 推回 `origin/ACL-exp-c3.5`！

---

## 第四部分 · 严格双路径成功门禁（Stage 2 裁决标准）

由 `c35_stats.py` 强制零 `None` 放行判定：
1. **统一健康底线（全部必过）**：`IFEval >= 430/541`、`GSM8K >= 470/1319`、`Wiki KL <= 0.005`、`think parse_fail <= 2.0%`、`S / shell_only / P0 / I0 >= 95%`、`DC_rev >= DC_orig - 10 pp`、`vllm_lora_parity == PASS`。
2. **路径一（效果胜出 `PASS_PATH1_SUPERIORITY`）**：`4-Cell (UW)` 相对最强健康 Uniform **`>= +3.0 pp`** 且 900 单元簇 95% CI 下界 **`> 0`**，`Joint T1∧4C` CI 下界 **`>= -2.0 pp`**，且若含 IIT 必须同时胜过无 IIT（`C > B`）与同轨迹随机头 IIT（`C > D`）。
3. **路径二（效率 / Pareto 支配 `PASS_PATH2_EFFICIENCY_PARETO`）**：`4-Cell` 与 `Joint T1∧4C` 相对最强健康 Uniform 的 95% CI 下界均 **`>= -2.0 pp`**，且 **WikiText-2 KL 至少降低 25%**（`KL <= 0.75 * KL_U_best`）。
4. **止损与后续**：单种子 `seed=42` 完成后先推送至 `origin/ACL-exp-c3.5` 并交回审计，**严禁未经确认擅自开 `seed=1024`、`14B` 或触碰密封测试集**！
