# Compact Pre-Flight Review Brief for Astra (Stage 5 Code Freeze — Ready for Final Check Before GPU Launch)

> 以下区块可直接复制发给 Astra 复核（所有代码与冻结数据均已在本地写完并通过 CPU 自检，**暂未推送上卡**，等她点头即推）：

```text
Astra，已按你的审核意见和三项上机前硬检查，在本地写齐并自检通过了 Stage 5 全部代码与冻结数据（尚未推卡，等你过目确认即可上机）。请重点复核以下 5 个文件与决策点，如无遗漏我们就启动 Wave 5A（8B 补测 + 14B M0/预检）与 Wave 5B（14B seed=42 四臂 + 外部 EX-FEVER + 加强版 Probe 2 v2；看完 14B seed=42 再决定是否补 14B seed=1024）：

一、全部落实的 5 项改动与本地 CPU 自检
1. 冻结真正第 4 臂（8B & 14B 严格同秩随机掩码 + 8B 第二张独立掩码）
   - 文件：`experiments/ACL-exp-c3.5/scripts/c35_export_components.py`、`data/c35_components_frozen.json`（SHA256 `a28a18b22468...`）
   - 8B：`M3-v4-PureMid-MLP+`、`M3-v4-PureMid-MLP+-Random-Mask`（Mask1, seed 2037, Top-5 MLP `[15,17,24,25,26]`）、`Mask2`（seed 2038, Top-5 MLP `[14,16,17,21,26]`）三者严格同为 `6,994,944` 参数（`-1.17%`）、`L14–27`、`max_r=68`、逐层头数与秩多重集全同、MLP `5×r12+9×r8` 全同，头坐标与高秩层严格不相交。
   - 14B：`M3-v4-PureMid-MLP+`（Top-5 MLP `[19,22,23,25,26]`）与 `M3-v4-PureMid-MLP+-Random-Mask`（seed 2037, Top-5 MLP `[17,24,27,28,29]`）严格同为 `10,348,800` 参数（`-1.31%`）、**Zone 2 = `L16–30`（15 层）**、`max_r=78`，逐层头数/秩与 MLP `5×r12+10×r8` 全同且严格不相交。
2. 未读前瞻确认集 `c3_blind_confirm_v3`（+ 修复位置对文风混杂）
   - 文件：`scripts/c35_build_blind_v3.py`、`data/c3_blind_confirm_v3_bases.jsonl`（SHA256 `32cdfab58ecc...`，`1,080` 题 / `900` 单元）与 `c3_blind_think_v3_bases.jsonl`（`240` 题）。
   - 6 个全新领域（断言与 A1/C1/Proxy/C2/v1/v2 全部领域、实体、区域、结果名 0 重叠）+ 2 套全新文风（`style_e_brief` / `style_f_ledger`）；60 对 `before_rules / after_rules` **严格固定同文风**仅变位置；240/240 CE decoy 与 840/840 独立双前提 DAG 求解器认证通过；vs `c3_train` 与 `v2` 骨架重叠严格 `0`；`G0 AUC = 0.5085, cf_byte_match = 1.0`。
3. 8B & 14B 共用加强版跨字母与跨结论探针 Probe 2 v2
   - 文件：`scripts/c35_eval_cross_letter_v2.py`（CPU `--selftest-tiny` 通过）。
   - 同时覆盖：**两种合法不相交字母排列（`perm_0` & `perm_1`）**、**plain/auth/hes 双向 6 组互换**、**96 题分层面板显式纳入 3 跳 CE Row B（12 d2 + 12 d3）**、**24 对跨领域不同结论名跨题互换（字母与语义结论名双重不相交）**、**同有效性预期保持对照（`V->V / F->F`）**，以及 `--aggregate-common` 输出全量/各自分母/共同合格分母三套口径（同面板测 `u2`、`v4-MLP+`、`v4-RandMask`）。
4. Ma et al. (EMNLP Findings 2026) 官方外部自然语言 `EX-FEVER` 评测器
   - 数据：`data/exfever_official_test1000.jsonl`（SHA256 `977ceccb3cfd...`），直接从 `dependentsign/sycophancy-rational-updating`（`emnlp-2026-camera-ready`）提取并逐字节校验上游 SHA256 的官方 1,000 题 test 切分（`507 True / 493 False`），GPU 机零外网依赖。
   - 代码：`scripts/c35_eval_exfever.py`（CPU `--selftest-tiny` 通过），100% 复刻官方无 system prompt 双轮模板、`extract_verdict`、退化/拒答统计，并完整报 `Acc`、`n_orig_correct / n_orig_wrong`、`R_UY`、`R_RU_evidence`、`R_RU_user_evidence` 与选择性 `R_RU - R_UY`（支持 `no_think` 与 `think` 双模式）。
5. 14B 彻底隔离、真实单步预检门禁与严苛 Stage 5 统计脚本
   - `scripts/c35_14b_preflight.py`：在 14B 四臂正式训练前，强制先用真实 `Qwen3-14B` 跑完“冻结契约断言 → 1 步训练 → PEFT 导出（`max_fp32_dw_err < 1e-5`, `r=78`）→ 固定 token IDs 的 HF/vLLM 一致性（`tp=2`, `passed == True`）→ 4 题 `no_think + Wiki KL` 显存冒烟”，通过才放行 14B 训练。
   - `scripts/c35_stage5_stats.py`：彻底独立于 `c3_8b_showdown`；用各尺度**自身 `M0`** 锚定相对健康门禁（`IFEval >= M0 - 10`、`GSM8K >= M0 - 26`）；**取消静默取交集**（任意一臂缺 1 个单元即 `AssertionError` 报错）；严格区分 `v4_MLPplus_vs_v4_RandMask`（及 8B `Mask2`）映射；分别判主 Pareto 门槛与精细定位加分项。
   - 完整方案文档：`experiments/ACL-exp-c3.5/C35_STAGE5_PLAN.md`。

二、请你把关的 2 个小确认
1. 14B 四臂超参（`LR=2.5e-5`, `microbatch=1 x grad_accum=40` = 等效 batch 40, Corridor `[4.5, 10.0]`, 4 点 `c3_dev` 自动选点）保持与 8B 完全一致做零调参跨尺度迁移，你是否认可？
2. `EX-FEVER` 我们打算在 `no_think`（与我们主训练/主评测对齐）和 `think` 两种模式下都跑一遍 8B 与 14B 全部冻结臂 + `M0`（vLLM 很快），是否有需要补充或收紧的细节？
```
