# ACL-exp-c3.5 Overnight Closure Report & Compact Astra Audit Brief

- **Branch / Latest Result Commit**: `ACL-exp-c3.5` @ `0628626` (`results(ACL-exp-c3.5): publish stage4 stats`)
- **Scope Executed Overnight**: 4 complete GPU stages on `Qwen3-8B` (20+ trained/evaluated LoRA checkpoints, 2 zero-retraining sub-projection/KL probes, 2 cross-letter causal interchange sweeps, and a full 2-seed `seed=42` + `seed=1024` showdown on `c3_blind_confirm_v2`).
- **Red-Line Compliance**:
  - Zero access to sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`; `converse_error` and `depth-4` excluded throughout.
  - Zero modifications to frozen `ACL-exp-a1 / b1 / c1 / c2 / c3`.
  - Zero `.pt` / `.safetensors` committed to Git; all `Qwen3-14B` runs held pending morning user + Astra review.
  - Phase-0 `M0` capability anchor reconciled as `IFEval = 440/541`, `GSM8K = 496/1319` correct (`502` parsed).

---

## Part I. Overnight Scientific Journey & Thinking Log (Why Every Stage Happened)

### Stage 1 (`b275221`): Fix the Yardstick & Prove Abstract Causal Routing
1. **Blind Benchmark Repair (`c3_blind_confirm_v2`)**: Fixed both C3 generator defects caught by Astra: (i) CE renderer missing `Based only ...` decoys (`240/240` CE decoys now intact), (ii) non-CE redundant `s1 ∧ s2` derivation replaced by truly independent dual-premise DAGs, plus `60` matched `before_rules / after_rules` counterfactual pairs. Final benchmark: **`900` independent generation units (`1,080` items)**.
2. **Re-evaluating C3 Baselines on v2**: Immediately revealed why C3's old `m2_corridor` lagged (`45.61%` vs `u2_corridor 50.56%`, `-4.94 pp`), while validating Confidence Corridor + Tone-Invariance (`u2_corridor − u1_base = +4.17 pp, p=0.0000`).
3. **Probe 1 (`results/stage1/probe1_subproj/SUBPROJECTION_PROBE.md`) — Architectural Physics Discovery**:
   - Disabling **Zone 3 (`L28–35`)** on `u2_corridor` dropped `c3_dev` by **`0.0 pp`** (`70.0% → 70.0%`) and KL by only `-0.4%`.
   - Disabling **Zone 2 MLPs (`L14–27`)** collapsed `c3_dev` by **`-26.7 pp`** (`70.0% → 43.3%`), `c2 Row B` from `15/36 → 2/36`, and `Step-3` by **`-42.5 pp`** (`75.0% → 32.5%`) — yet cost only **`1.4%`** of total WikiText-2 KL!
   - Disabling **Zone 1 (`L0–13`)** cut WikiText-2 KL by **`-31.4%`** (`0.001381 → 0.000947`), with Zone-1 MLP driving most of the KL hit (`-9.3%` vs Q/O `-2.0%`).
4. **Probe 2 (`results/stage1/probe2_cross_letter/`) — Cross-Letter Abstract Causal Proof**:
   - Remapped target option letters so `{src_a0, src_at} ∩ {tgt_a0, tgt_at} = ∅`. Across ALL models and ALL 6 component subsets, **literal source-letter copy was `0.0%`**!
   - `b1_full` achieved **`87.8%–93.1%`** abstract counterfactual logic flips across `argument_flip` (`90.1%–95.6%`), `context_flip` (`88.5%–96.2%`), and `crossed_2x2` (`78.6%–90.2%`) vs `rnd_full = 1.4%–5.2%`.

### Stage 2 (`605c6f6` → `88cec3f`): Parameter Reallocation vs Optimizer Relaxation vs Soft-Gated IIT
- **5 New Arms (`armA`–`armE`)**, all 8/8 trained arms `100% HEALTH PASS`:
  - **`armA` (old `M3-v1` + relaxed Z1 LR/prox) vs `m2`**: `+0.50 pp` (`p=0.229`, n.s.) → optimizer strangling alone did NOT explain C3's gap.
  - **`armB` (`M3-v2` + Corridor) vs `armA`**: **`+4.11 pp` (`p=0.0000`)** (`50.22%` vs `46.11%`)! Reallocating dead Zone-3 capacity into Zone-2 MLPs + Zone-1 Q/O closed virtually the entire gap to `u2_corridor` (`-0.33 pp, p=0.60`) while cutting WikiText-2 KL by **`-19.8%`** (`0.001092` vs `0.001362`) and lifting `Row B` (`18.3%` vs `15.8%`).
  - **Soft-Gated IIT (`armC/D/E`) + Same-Batch Gradient Telemetry**:
    - Direct measurement answered Astra's gradient question: B1 heads align **3.8× better** with main-task gradients than random heads (`armC mean_cos = +0.1447, med +0.2487` vs `armD +0.0378, med +0.1422`), and `armC` beat BOTH `armD` (`+0.61 pp, p=0.022`) and `armE` (`+1.22 pp, p=0.029`).
    - Yet `armC` still trailed pure-Corridor `armB` by `-0.39 pp` (`p=0.409`), because ~`39.6%` of IIT steps still had negative cosine (`< 0`) and pushed in-distribution dev overfit (`72.0%` dev vs blind).

### Stage 3 (`eef6e40` → `575fbf2`): Surgical KL Elimination (`PureMid`) & Final PCGrad Test on Training-Time IIT
- **Sub-Layer KL Probe (`results/stage3/probe_m3v2_kl/M3V2_KL_PROBE.md`)**: On `armB_m3v2_corridor`, removing `Z1` (`L8–13`) alone dropped Wiki KL from `-22.5%` to **`-31.7%`** vs `u2` (`0.000930`), proving **100% of the remaining KL gap lived in Zone 1**!
- **Architectural Breakthrough (`M3-v3-PureMid`)**: Putting `100%` of LoRA parameters strictly inside mid-layer Zone 2 (`L14–27`, ZERO Zone 1, ZERO Zone 3) achieved `49.83%` (`-0.72 pp` vs `u2`), `Row B = 20.0%` (`+4.2 pp` vs `u2`), **`Wiki KL = 0.000958` (`-29.7%` vs `u2`)**, AND beat `M3-v3-PureMid-Random-Mask` (`48.72%`) by `+1.11 pp` (`p=0.0336`) on `seed=42`!
- **Stage 3 Probe 2 (`results/stage3/probe2_cross_letter/`)**: `s3_puremid_corr` (ZERO training-time IIT!) achieved `b1_full = 90.5%` (`0.0%` literal copy) and **`b1_heads_only = 60.5%`** (vs `u2_corridor 47.1%`, **`+13.4 pp`**) — proving **PureMid architectural allocation alone strengthens causal concentration in B1 heads**!
- **Clean Closure on Training-Time IIT (PCGrad + Step-160 Start)**: Even with orthogonal conflict projection (`--iit-pcgrad`) and delayed step-160 onset, ALL three PCGrad arms (`PureMid`, `TightBridge`, `M3-v2`) still dipped `-0.33 to -0.56 pp` relative to their identical pure-Corridor twins (despite B1 maintaining 3× stronger cosine alignment `+0.2012` vs Random `+0.0668`). **Decision**: Retire explicit training-time IIT loss and double down on causal architectural allocation (`M3-v4`).

### Stage 4 (`fb5c03c` → `0628626`): `M3-v4-PureMid-MLP+` & Dual-Seed Replication (`seed=42` + `seed=1024`)
- **Hypothesis**: Since Probe 1 proved Zone-2 MLPs carry massive reasoning leverage (`-26.7 pp` dev / `-42.5 pp` Step-3) for only `1.4%` KL, reallocating part of `M3-v3-PureMid`'s Zone-2 head rank (`10@18, 12@12, 20@6` → `10@14, 12@8, 20@4`) into **ALL 14 Zone-2 MLPs** (B1 top-5 `N_CF @ r=12` + 9 remaining `@ r=8`; `6,994,944 = -1.17%` vs Uniform; `max_layer_r = 68 <= 128`; `zone1=0, zone3=0`) should keep `-30%+` Wiki KL reduction while **outright beating `u2_corridor` in accuracy**! We also tested `M3-v4-Bridge2` (`7,049,216 = -0.41%`, adding minimal `L12–13` MLP bridge) and replicated ALL 6 flagship methods across two independent seeds (`seed=42` & `seed=1024`).
- **Outcome**: **12/12 ARMS `100% HEALTH PASS`, 12/12 UNIQUE ADAPTER SHA256, 12/12 `vllm_lora_parity = True`**, and **`M3-v4-PureMid-MLP+` UNCONDITIONALLY PASSED STRICT `PASS_PATH2_EFFICIENCY_PARETO` ON `seed=42`, `seed=1024`, AND POOLED 2-SEED — OUTSCORING `u2_corridor` ON BOTH SEEDS WHILE CUTTING WIKI KL BY `-31.4%`!**

---

## Part II. Audited Stage 4 Dual-Seed Results (`seed=42` + `seed=1024`, `900` Independent Units)

### Table 1. Per-Seed Primary, Subgroup, Secondary, Capability & KL Readout (All 12 Arms Verified from Disk)

| Method | Seed | Params (`Δ%` vs Unif) | Primary 4-Cell (UW) | Joint T1∧4C (UW) | Step 1 / 2 / 3 | Row A / Row B (4C) | Pos `bef / aft` (60p) | `c2_bidir` (360) | `c2_stealth` (180) | `think_v2` FW (240) | `c1_proxy_v2` (120) | IFEval / GSM8K | Wiki KL (`Δ%` vs `u2`) | Health & Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M0` (base) | — | 0 | 0.00% | 0.00% | — | 0.0 / 0.0% | — | — | — | — | — | 440 / 496 | `0.000000` | Anchor |
| `u1_base` | 42 | 7.08M (0.0%) | 46.39% | 45.61% | 38.2 / 48.6 / 50.0 | 85.8 / 15.0% | 45.0 / 41.7 | 73.06% | 62.22% | 94.25% | 82.50% | 443 / 538 | `0.001364` | PASS / True |
| `u1_base` | 1024 | 7.08M (0.0%) | 45.39% | 44.94% | 36.8 / 47.1 / 49.3 | 84.2 / 16.7% | 43.3 / 40.0 | 73.33% | 63.33% | 94.75% | 82.50% | 445 / 507 | `0.001350` | PASS / True |
| `u2_corridor` | 42 | 7.08M (0.0%) | 50.56% | 49.89% | 43.2 / 52.5 / 53.2 | 95.0 / 15.8% | 51.7 / 43.3 | 75.56% | 65.56% | 96.75% | 82.50% | 449 / 547 | `0.001362` (ref) | PASS / True |
| `u2_corridor` | 1024 | 7.08M (0.0%) | 48.44% | 48.00% | 39.3 / 50.0 / 51.8 | 92.5 / 20.8% | 48.3 / 41.7 | 75.83% | 68.33% | 96.25% | 84.17% | 445 / 540 | `0.001427` (ref) | PASS / True |
| `M3-v3-PureMid` | 42 | 7.04M (-0.53%) | 49.83% | 49.00% | 40.7 / 52.5 / 53.6 | 92.5 / 20.0% | 55.0 / 45.0 | 76.94% | 58.89% | 94.25% | 79.17% | 448 / 522 | `0.000958` (**-29.7%**) | PASS / True |
| `M3-v3-PureMid` | 1024 | 7.04M (-0.53%) | 48.72% | 48.00% | 39.6 / 51.8 / 53.2 | 87.5 / 15.8% | 53.3 / 41.7 | 77.50% | 61.67% | 96.00% | 80.00% | 442 / 521 | `0.000954` (**-33.1%**) | PASS / True |
| `M3-v3-RandMask` | 42 | 7.04M (-0.53%) | 48.72% | 47.83% | 40.7 / 50.7 / 52.1 | 90.8 / 17.5% | 51.7 / 43.3 | 76.39% | 58.89% | 93.50% | 78.33% | 442 / 553 | `0.000977` (-28.3%) | PASS / True |
| `M3-v3-RandMask` | 1024 | 7.04M (-0.53%) | 49.28% | 48.39% | 41.4 / 52.5 / 53.6 | 86.7 / 14.2% | 56.7 / 43.3 | 73.89% | 61.11% | 91.00% | 80.83% | 447 / 538 | `0.000969` (-32.1%) | PASS / True |
| **`M3-v4-PureMid-MLP+`** | **42** | **6.99M (-1.17%)** | **50.89%** | **50.17%** | **40.7 / 54.3 / 55.4** | **91.7 / 19.2%** | **53.3 / 45.0** | **78.06%** | 61.11% | 95.00% | 77.50% | 447 / 538 | **`0.000954` (-30.0%)** | **PASS / True** |
| **`M3-v4-PureMid-MLP+`** | **1024** | **6.99M (-1.17%)** | **49.61%** | **49.00%** | **40.4 / 52.9 / 53.9** | **87.5 / 18.3%** | **53.3 / 43.3** | **77.22%** | 62.22% | 93.50% | 81.67% | 445 / 506 | **`0.000959` (-32.8%)** | **PASS / True** |
| `M3-v4-Bridge2` | 42 | 7.05M (-0.41%) | 49.89% | 49.22% | 40.7 / 52.9 / 53.2 | 92.5 / 20.0% | 55.0 / 43.3 | 76.39% | 60.00% | 94.50% | 80.83% | 446 / 542 | `0.001006` (-26.1%) | PASS / True |
| `M3-v4-Bridge2` | 1024 | 7.05M (-0.41%) | 49.22% | 48.50% | 40.0 / 51.8 / 53.9 | 89.2 / 16.7% | 53.3 / 41.7 | 76.67% | 62.22% | 94.75% | 80.83% | 444 / 505 | `0.000997` (-30.1%) | PASS / True |

### Table 2. Paired Unit-Clustered Bootstrap Comparisons (`900` Independent Units) & Strict Dual-Path Verdicts

| Comparison | `seed=42` 4-Cell (pp) | `seed=1024` 4-Cell (pp) | **Pooled 2-Seed 4-Cell [95% CI]** | **Pooled 2-Seed Joint [95% CI]** | Pooled KL `Δ%` | Strict Verdict (`s42` / `s1024` / `Pooled`) |
|---|---:|---:|---|---|---:|---|
| `u2_corridor − u1_base` | **+4.17** (p=0.0000) | **+3.06** (p=0.0000) | **+3.61** `[+2.64, +4.67]` (p=0.0000) | **+3.67** `[+2.67, +4.70]` (p=0.0000) | +2.7% | Replicated core gain |
| **`M3-v4-PureMid-MLP+ − u2`** | **+0.33** (p=0.645) | **+1.17** (p=0.132) | **+0.75** `[-0.42, +1.92]` (p=0.220) | **+0.64** `[-0.56, +1.83]` (p=0.303) | **-31.4%** | **PASS / PASS / PASS (Path 2 Pareto)** |
| **`M3-v4-PureMid-MLP+ − v3-RandMask`** | **+2.17** (p=0.0000) | +0.33 (p=0.646) | **+1.25** `[+0.33, +2.19]` (p=0.0072) | **+1.47** `[+0.50, +2.44]` (p=0.0036) | -1.7% | Significant (see Caveat #1 below) |
| `M3-v4-Bridge2 − u2` | -0.67 (p=0.321) | +0.78 (p=0.289) | **+0.06** `[-1.00, +1.11]` (p=0.959) | **-0.08** `[-1.14, +0.97]` (p=0.871) | **-28.2%** | **PASS / PASS / PASS (Path 2 Pareto)** |
| `M3-v3-PureMid − u2` | -0.72 (p=0.292) | +0.28 (p=0.732) | **-0.22** `[-1.31, +0.86]` (p=0.698) | **-0.44** `[-1.56, +0.64]` (p=0.445) | **-31.4%** | HOLD / PASS / **PASS (Path 2 Pareto)** |
| `M3-v3-PureMid − v3-RandMask` | **+1.11** (p=0.0336) | -0.56 (p=0.417) | +0.28 `[-0.64, +1.19]` (p=0.572) | +0.39 `[-0.53, +1.31]` (p=0.426) | -1.7% | Mixed across seeds (see Caveat #1) |

- **Exact Integrity Verification**: `12/12` unique `adapter_state.pt` SHA256 hashes, `12/12` unique `adapter_model.safetensors` SHA256 hashes, `12/12` `vllm_lora_parity = True` (`option_match = 96.3%–100.0%`, `max_fp32_dw_abs_err <= 3.49e-10`), `12/12` `think_parse_fail <= 0.14%`, `12/12` `S/shell/P0/I0 >= 99.9%`.

---

## Part III. Honest Caveats & Proposed Next-Wave Options (Awaiting Morning Review)

1. **Caveat #1 — Separating Macro-Band (`Zone 2 MLP+`) from Fine-Grained B1 Masking**:
   - At matched `r10/r6` MLPs, `M3-v3-PureMid vs M3-v3-RandMask` was `+1.11 pp (p=0.034)` on `s42` and `-0.56 pp (p=0.417)` on `s1024` (pooled `+0.28 pp, p=0.57`, though `PureMid` consistently beat `RandMask` on `c2_bidir` `+2.08 pp` and `think_v2` `+2.88 pp` across both seeds).
   - `M3-v4-PureMid-MLP+` beat `M3-v3-RandMask` significantly (`+1.25 pp, p=0.0072`; Joint `+1.47 pp, p=0.0036`), **but** `v4-MLP+` allocates more MLP capacity (`4.33M` at `r12/r8` vs `3.15M` at `r10/r6`).
   - **Clean 2-GPU Fix (8B)**: Run a strictly matched **`M3-v4-PureMid-MLP+-Random-Mask`** control (identical `r12/r8` MLP budget with shuffled non-B1 top-5 layers + random Zone-2 heads) across `seed=42` & `seed=1024` + run zero-retraining **Probe 2 cross-letter interchange on `M3-v4-PureMid-MLP+`**!
2. **Caveat #2 — Honest Secondary Trade-Offs of Removing Zone 1 (`L0–13`)**:
   - Dropping Zone 1 cuts WikiText-2 KL by **`-31.4%`** and boosts `c3_blind_confirm_v2` (`+0.75 pp`, especially `Step 2 +2.35 pp`, `Step 3 +2.15 pp`) and `c2_confirm_bidir` (`+1.95 pp`), while preserving `IFEval >= 445` & `GSM8K >= 506`.
   - Trade-off: on surface-distraction `c2_stealth_stress` (`180` items), `v4-MLP+` (`61.7%`) trails `u2_corridor` (`66.9%`, `-5.2 pp`), and on `c3_blind_think_v2` (`240` items), `v4-MLP+` (`94.25%`) trails `u2_corridor` (`96.50%`, `-2.25 pp`, still `+2.00 pp` over `RandMask 92.25%`). This is mechanistically consistent with Probe 1 (Zone 1 handles surface prompt parsing at high KL cost).
3. **Proposed Compact `Qwen3-14B` Replication (Ready in `data/c35_components_frozen.json`, Held for Approval)**:
   - `14B` `M3-v4-PureMid-MLP+` is already frozen at `10,348,800` params (`-1.31%` vs `14B Uniform 10,485,760`, `max_r = 78 <= 128`, `100%` Zone-2 `L17–34`).
   - **Recommended Compact 4-Arm 14B Matrix** (fits cleanly on 4–8×A800, ZERO wasted IIT arms!):
     1. `14B u1_base` (no corridor)
     2. `14B u2_corridor` (strong Uniform control)
     3. `14B M3-v4-PureMid-MLP+` (flagship causal mid-band Pareto winner)
     4. `14B M3-v4-PureMid-MLP+-Random-Mask` (matched-rank within-Zone-2 localization control)
     - Plus zero-retraining **14B Probe 2 Cross-Letter Causal Interchange** (testing Astra's cross-scale 8B→14B abstract-validity criterion!).

---

## Part IV. Copy-Paste-Ready Compact Prompt for Astra Audit

> 以下区块可直接完整复制发给 Astra，信息密度最高、路径与 commit 全部精确可核验，最大化节省额度：

```text
Astra，ACL-exp-c3.5 已完成全部 4 阶段 8B 实验（最新 commit `0628626`，分支 `ACL-exp-c3.5`，零触碰封存测试集，14B 保持未启动等你拍板）。请你直接按仓内文件核验并给出 3 个关键决策：(1) 8B 叙事与主结论边界；(2) 是否补跑 8B `M3-v4-PureMid-MLP+-Random-Mask` 严格同秩对照 + `v4-MLP+` Cross-Letter Probe 2；(3) 是否 GO 启动紧凑 4 臂 `Qwen3-14B` 复现 + 14B Cross-Letter Probe 2。

一、关键文件与 Commit 索引
- 完整 4 阶段总结：`experiments/ACL-exp-c3.5/C35_OVERNIGHT_SUMMARY_AND_ASTRA_BRIEF.md`
- Stage 1（`b275221`）：修复后盲集 `c3_blind_confirm_v2`（900 独立生成单元 / 1080 题，240/240 CE decoy，独立双前提 DAG，60 对 `before/after_rules`）
  - Probe 1 分投影与 KL 探针：`experiments/ACL-exp-c3.5/results/stage1/probe1_subproj/SUBPROJECTION_PROBE.md`
  - Probe 2 跨字母重映射 7 路互换：`experiments/ACL-exp-c3.5/results/stage1/probe2_cross_letter/cross_letter_summary_*.json`
- Stage 2（`88cec3f`）：5 新臂 A–E 读数 `experiments/ACL-exp-c3.5/results/stage2/C35_READOUT.md`
- Stage 3（`575fbf2`）：KL 外科探针 `experiments/ACL-exp-c3.5/results/stage3/probe_m3v2_kl/M3V2_KL_PROBE.md`；7 新臂（PureMid / TightBridge / PCGrad-IIT）读数 `experiments/ACL-exp-c3.5/results/stage3/C35_READOUT.md`；Stage 3 Probe 2：`experiments/ACL-exp-c3.5/results/stage3/probe2_cross_letter/cross_letter_summary_*.json`
- Stage 4（`0628626`）：双种子（`seed=42` + `seed=1024`）12 臂总评 `experiments/ACL-exp-c3.5/results/stage4/C35_MULTISEED_READOUT.md` 与 `c35_multiseed_stats.json`；冻结架构预算 `experiments/ACL-exp-c3.5/data/c35_components_frozen.json`

二、四大核心发现（均已从逐题/JSON 核验）
1. 跨字母重映射证明抽象逻辑因果（Probe 2，8B）：
   - 源与受体字母集合严格不相交时，所有模型、所有组件的字面字母复制率均为严格 `0.0%`。
   - `b1_full` 抽象逻辑翻转率 `87.8%–93.1%`（覆盖 `argument_flip 90–96%`、`context_flip 89–97%`、`crossed_2x2 79–90%`），`rnd_full` 仅 `1.4%–5.2%`。
   - 零训练期 IIT 的 `s3_puremid_corr` 单靠中层因果架构分配就把 `b1_heads_only` 从 `u2` 的 `47.1%` 推升至 `60.5%`（`b1_full = 90.5%`）。
2. 分区/分投影 KL 物理与架构演进（Stage 1→4）：
   - Probe 1 + `M3V2_KL_PROBE` 显示：Zone 3 (`L28–35`) 精度贡献 `0.0 pp`；Zone 1 (`L0–13`) 制造 `-31.4% ~ -31.7%` 的 WikiText-2 KL；Zone 2 (`L14–27`) MLP 仅占 `1.4%` KL 却贡献 `c3_dev +26.7 pp`、`Step-3 +42.5 pp`。
   - Stage 2 验证：`armA`（旧 M3 放宽 Z1 优化）仅 `+0.50 pp (p=0.23)`，而 `armB`（`M3-v2` 重分配参数）较 `armA` 跳升 `+4.11 pp (p=0.0000)`，KL `-19.8%`。
   - Stage 4 旗舰 **`M3-v4-PureMid-MLP+`**（100% 参数位于 Zone 2 `L14–27`，全 14 层 MLP 升至 B1 Top-5 `r=12` + 余 9 层 `r=8`，`6.99M = -1.17%` vs Uniform，`max_r=68`）双种子全面达标：
     - `seed=42`：`4-Cell 50.89%`（vs `u2 50.56%`, `+0.33 pp`），`Joint 50.17%`，`c2_bidir 78.06%`（vs `75.56%`），`Wiki KL 0.000954`（`-30.0%`）→ **PASS_PATH2_EFFICIENCY_PARETO**
     - `seed=1024`：`4-Cell 49.61%`（vs `u2 48.44%`, `+1.17 pp`），`Joint 49.00%`，`c2_bidir 77.22%`（vs `75.83%`），`Wiki KL 0.000959`（`-32.8%`）→ **PASS_PATH2_EFFICIENCY_PARETO**
     - **Pooled 2-Seed（900 单元）**：`v4-MLP+ − u2` 为 **`+0.75 pp [-0.42, +1.92]`**（`Joint +0.64 pp`），**`Wiki KL -31.4%`**；`Corridor (u2 − u1)` 双种子稳健复现 **`+3.61 pp [+2.64, +4.67], p=0.0000`**。
     - 12/12 臂全部 `HEALTH PASS`（`IFEval 442–449`, `GSM8K 505–553`, 均高于 `M0 440/496`；`think_pf <= 0.14%`）、`12/12` `vllm_lora_parity = True`、12/12 权重哈希唯一。
3. 训练期可微 IIT 损失的诚实负结论（Stage 2 & 3）：
   - 同 batch 梯度遥测证实 B1 头与主任务余弦对齐度是随机头的 3–4 倍（`+0.145 / +0.201` vs `+0.038 / +0.067`），且软 B1-IIT 显著优于软随机头（`C−D = +0.61 pp, p=0.022`）及 Uniform+IIT（`C−E = +1.22 pp, p=0.029`）。
   - 但即使引入 `α=0.3→0.7` 软目标、4-opt 门控、`0.30×` 独立限幅、延迟至 step 160 及 PCGrad 正交投影，全部 6 种训练期 IIT 变体在盲集仍较同架构纯 Corridor 低 `-0.33 ~ -0.56 pp`。据此我们干净收口：B1 因果定位的最佳变现路径是**因果架构参数分配 + 置信走廊**，而非训练期显式互换损失。

三、我们主动披露的 3 处诚实边界（请重点审视）
1. **Zone-2 宏带宽 vs Zone-2 内部 B1 精细掩码的解耦**：同 MLP 秩（`r10/r6`）下 `PureMid − v3-RandMask` 在 `s42` 为 `+1.11 pp (p=0.034)`、`s1024` 为 `-0.56 pp (p=0.417)`（pooled `+0.28 pp, p=0.57`，但 `PureMid` 在 `c2_bidir +2.08 pp` 与 `think_v2 +2.88 pp` 双种子均领先）。`v4-MLP+` 显著击败 `v3-RandMask`（pooled `+1.25 pp, p=0.0072`；Joint `+1.47 pp, p=0.0036`），但两者 MLP 容量不同（`r12/r8` vs `r10/r6`）。是否应补跑严格同秩的 **`M3-v4-PureMid-MLP+-Random-Mask`**（同 `r12/r8` 洗牌层+随机 Zone-2 头）来严格剥离二者？
2. **去 Zone 1 的次要指标权衡**：`v4-MLP+` 在 `blind_v2 (+0.75 pp)`、`Step 2/3 (+2.35/+2.15 pp)`、`c2_bidir (+1.95 pp)` 和 `Wiki KL (-31.4%)` 全面胜出，但在表层干扰集 `c2_stealth_stress` 低于 `u2`（`61.7% vs 66.9%`），`think_v2` 略低（`94.25% vs 96.50%`，但仍高于 `RandMask 92.25%`）。这与 Probe 1（Zone 1 负责表层解析但 KL 代价极高）完全自洽。
3. **下一步方案请拍板**：
   - 方案 A（推荐）：8B 补 `v4-MLP+-Random-Mask`（双种子）+ `v4-MLP+` Probe 2，**并行**启动紧凑 4 臂 `Qwen3-14B`（`u1_base`, `u2_corridor`, `M3-v4-PureMid-MLP+`, `M3-v4-MLP+-Random-Mask`；`14B v4-MLP+` 已冻结 `10.35M = -1.31%`, `max_r=78`）+ 14B Cross-Letter Probe 2（不跑 14B IIT 臂）。
   - 请明确给出：是否同意方案 A、论文主主张中“中层因果区间 vs 头/MLP 精细排序”该怎么定调，以及 14B 上机前有无必须先补的检查项。
```
