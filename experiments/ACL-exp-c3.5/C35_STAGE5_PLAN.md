# ACL-exp-c3.5 Stage 5 Plan: 8B Rank-Matched Closure + 14B 4-Arm Prospective Replication + Official External EX-FEVER

> **Status**: **LOCAL CODE & CPU SELF-TESTS READY FOR ASTRA PRE-FLIGHT REVIEW (NOT YET PUSHED TO REMOTE GPU RUNNER)**
> **Mandate**: Implements Astra's conditional GO on Option A + ALL 3 pre-flight checks + upgraded shared Probe 2 v2 + Ma et al. (EMNLP Findings 2026) official 1,000-item external `EX-FEVER` evaluation!
> **14B Seed Policy**: Run `Qwen3-14B` `seed=42` 4-arm prospective showdown + `EX-FEVER` + `Probe 2 v2` FIRST; inspect readout before deciding whether to invest in 14B `seed=1024`.

---

## 1. Frozen Artifacts & SHA-256 Registry (All Verified Locally on CPU)

| Artifact | Path | Key Invariants | SHA-256 (prefix) |
|---|---|---|---|
| Frozen Architectures (8B & 14B) | `data/c35_components_frozen.json` | Adds `M3-v4-PureMid-MLP+-Random-Mask` (Mask1, seed 2037) & `Mask2` (seed 2038). Exact budget match: 8B `6,994,944` (`-1.17%`, `L14–27`, `max_r=68`); 14B `10,348,800` (`-1.31%`, **`L16–30`**, `max_r=78`). Disjoint heads & non-B1 top-5 MLPs, exact per-layer Q/O rank multiset & MLP `5×r12 + 9/10×r8`. | `a28a18b22468` |
| Untouched Prospective Blind v3 | `data/c3_blind_confirm_v3_bases.jsonl` | `1,080` bases / `900` independent generation units across 6 brand-new unseen domains (`botanical_conservatory`, `tidal_barrage`, `archive_bindery`, `glacier_station`, `wind_tunnel`, `kiln_foundry`) & 2 unseen styles (`style_e_brief`, `style_f_ledger`). `240/240` CE decoys, `840/840` dual-premise DAG solver certified, `60` **fixed-style** `(before_rules vs after_rules)` pairs, `0` skeleton overlap vs `c3_train` & `v2`, `G0 AUC = 0.5085`, `cf_byte_match = 1.0`. | `32cdfab58ecc` |
| Prospective Blind Think v3 | `data/c3_blind_think_v3_bases.jsonl` | `240` stratified CoT items (`192` units, 40 CE cells + 160 non-CE). | `7dee7d61d636` |
| Official Ma et al. EX-FEVER Test | `data/exfever_official_test1000.jsonl` | Exact 1,000-row official `test` split (`507 True / 493 False`) from `dependentsign/sycophancy-rational-updating` (`emnlp-2026-camera-ready`), upstream `exfever.jsonl` & `splits/exfever.json` bit-for-bit SHA256 verified! | `977ceccb3cfd` |

---

## 2. Summary of Astra Pre-Flight Fixes Implemented

1. **Pre-Flight Check #1 (True 4th Arm & Exact Stats Mapping)**:
   - `c35_export_components.py` freezes `M3-v4-PureMid-MLP+-Random-Mask` (`Mask1`, seed 2037) and `M3-v4-PureMid-MLP+-Random-Mask2` (`Mask2`, seed 2038) for BOTH `qwen3_8b` and `qwen3_14b`, with hard assertions on total params, per-layer head rank multiset, MLP rank multiset, zero Zone-1/Zone-3 params, `max_packed_layer_rank <= 128`, and strict disjointness.
   - `c35_stage5_stats.py` maps `v4_MLPplus_vs_v4_RandMask` strictly to `M3-v4-PureMid-MLP+-Random-Mask` (and `v4_MLPplus_vs_v4_RandMask2` to `Mask2` on 8B) — NEVER falling back to v3!
   - Corrected 14B Zone 2 documentation everywhere to **`L16–30` (15 layers)**.
2. **Pre-Flight Check #2 (14B Clean Isolation, M0-Anchored Health Gates, Real 1-Step Preflight, Fail-Closed Unit Guard)**:
   - Dedicated isolated output trees: `results/stage5_8b/` and `results/stage5_14b/` (ZERO copy from `c3_8b_showdown`!).
   - `c35_stage5_stats.py` implements **M0-anchored relative health gates** (`ifeval >= ifeval_m0 - 10`, `gsm8k >= gsm8k_m0 - 26`, `wiki_kl <= 0.005`, `think_parse_fail <= 0.02`, `S/shell/P0/I0 >= 0.95`, `DC_rev >= DC_orig - 0.10`, `parity_passed == True`) AND **fail-closed unit completeness assertions** (`assert_complete_units` raises `AssertionError` if ANY arm is missing even 1 unit — zero silent intersection!).
   - `c35_14b_preflight.py` executes BEFORE full 14B training: verifies frozen 14B contract, runs real 1-step training (`--max-steps 1 --model-key qwen3_14b`), exports PEFT (`max_fp32_dw_err < 1e-5`, `r_export == 78`), runs fixed-token-ID HF/vLLM parity (`--tp 2`, asserts `passed == True`), and runs a 4-item 14B `no_think + reference_kl` VRAM smoke check!
3. **Pre-Flight Check #3 (Untouched Prospective Confirmation + Fixed-Style Position Pairs + Upgraded Shared Probe 2 v2 + External EX-FEVER)**:
   - `c3_blind_confirm_v3` is completely untouched during architecture selection; all 60 paired position items hold discourse style strictly identical across `before_rules / after_rules`.
   - `c35_eval_cross_letter_v2.py` upgrades Probe 2 across BOTH 8B & 14B:
     - BOTH valid disjoint option-letter permutations (`perm_0` & `perm_1`);
     - Bidirectional swaps across `plain`, `auth`, AND `hes` tones (6 flip pairs/base);
     - Explicit 3-hop `CE Row B` stratification (`12` d2 + `12` d3 inside 96-base panel);
     - **Cross-item different-conclusion-name panel** (`24` domain-disjoint pairs where NEITHER option letters NOR semantic outcome strings overlap!);
     - **Same-validity STAY controls** (`V->V`, `F->F`, `V_auth->V_auth`, `F_auth->F_auth` on both same-item remapped and cross-item receptors);
     - `--aggregate-common` outputs ALL swaps, per-model denominators, AND exact shared common-denominator comparisons across `[u2, v4-MLP+, v4-RandMask]`!
   - `c35_eval_exfever.py` implements Ma et al.'s EXACT official prompts (`sru/prompts.py`), strict verdict extractor (`sru/scoring.py`), and dual-denominator metrics (`Acc`, `n_orig_correct`, `n_orig_wrong`, `R_UY`, `R_RU_evidence`, `R_RU_user_evidence`, `Selectivity = R_RU - R_UY`) on the frozen 1,000-row official test split (evaluated in BOTH `no_think` primary and `think` secondary regimes!).

---

## 3. Execution Schedule (8 × NVIDIA A800 40GB, Zero IIT Arms)

### Wave 5A: 8B Closure + 14B `M0` Anchor & Mandatory 14B Preflight (~2.5h)
- **GPUs 0–3 (4 parallel 8B training + full eval arms)**:
  - GPU 0: `s5_8b_s42_v4_randmask1` (`M3-v4-PureMid-MLP+-Random-Mask`, `seed=42`, Corridor)
  - GPU 1: `s5_8b_s1024_v4_randmask1` (`M3-v4-PureMid-MLP+-Random-Mask`, `seed=1024`, Corridor)
  - GPU 2: `s5_8b_s42_v4_randmask2` (`M3-v4-PureMid-MLP+-Random-Mask2`, `seed=42`, Corridor)
  - GPU 3: `s5_8b_s1024_v4_randmask2` (`M3-v4-PureMid-MLP+-Random-Mask2`, `seed=1024`, Corridor)
  - All 4 arms run `c35_train` -> `c35_select_dev_ckpt` -> `c35_export_peft_lora` -> `c35_check_vllm_lora_parity` -> `c35_eval` (`v2`, `v3`, `c2_confirm_bidir`, `c2_stealth_stress`, `Wiki KL`) -> `think` (`v2`, `v3`) -> `capabilities` -> **`EX-FEVER`** (`no_think` + `think`).
- **GPU 4 (8B Existing Arms Prospective Extension + Probe 2 v2)**:
  - Evaluate existing frozen 8B `M0`, `u1_base (s42/s1024)`, `u2_corridor (s42/s1024)`, `M3-v4-PureMid-MLP+ (s42/s1024)` on untouched **`c3_blind_confirm_v3`**, **`c3_blind_think_v3`**, and **`EX-FEVER`** (`no_think` + `think`).
  - Run **8B Probe 2 v2** on `u2_corridor@s42`, `M3-v4-PureMid-MLP+@s42`, `v4-RandMask1@s42`, `v4-RandMask2@s42` -> `--aggregate-common`.
- **GPUs 5–7 (14B `M0` Anchor + Mandatory 14B Preflight Gate)**:
  - GPUs 5–6 (`tp=2`): `c35_14b_preflight.py` (real 1-step train + PEFT export + fixed-ID HF/vLLM parity + 4-item KL VRAM smoke -> MUST emit `preflight_14b_passed.json`!).
  - Then `14B M0` full anchor (`IFEval 541`, `GSM8K 1,319`, `c3_blind_confirm_v2`, `c3_blind_confirm_v3`, `EX-FEVER 1,000` `no_think` & `think`).

### Wave 5B: 14B 4-Arm Prospective Showdown (`seed=42`, Starts ONLY After `preflight_14b_passed.json`) (~4–5h)
- All 4 arms run in parallel (`2 × A800` per arm, `microbatch=1, grad_accum=40` = effective batch 40 matching 8B, identical `LR=2.5e-5`, identical Corridor `[4.5, 10.0]` for zero-tuning cross-scale transfer!):
  - GPUs 0–1: `s5_14b_s42_u1_base` (`Uniform-LoRA`, `loss-mode base`, `10.49M`)
  - GPUs 2–3: `s5_14b_s42_u2_corridor` (`Uniform-LoRA`, `loss-mode corridor`, `10.49M`)
  - GPUs 4–5: `s5_14b_s42_v4_mlpplus` (`M3-v4-PureMid-MLP+`, `L16–30`, `loss-mode corridor`, `lr_z1=0, prox_z1=0`, `10.35M`)
  - GPUs 6–7: `s5_14b_s42_v4_randmask` (`M3-v4-PureMid-MLP+-Random-Mask`, `L16–30`, `loss-mode corridor`, `lr_z1=0, prox_z1=0`, `10.35M`)
- Per-arm frozen pipeline: `train (300 steps)` -> `dev_selection (c3_dev)` -> `export_peft` -> `vllm_lora_parity (tp=2)` -> single-shot prospective eval on **`c3_blind_confirm_v3` + `c3_blind_confirm_v2` + `c2_confirm_bidir` + `c2_stealth_stress` + `Wiki KL`** -> **`think` (`v3`, `v2`)** -> **`capabilities (14B IFEval/GSM8K, tp=2)`** -> **Official `EX-FEVER` (`no_think` + `think`, `tp=2`)** -> **14B Probe 2 v2** (`u2`, `v4-MLP+`, `v4-RandMask` -> `--aggregate-common`).
- Final aggregation: `c35_stage5_stats.py` generates strict M0-anchored readouts for BOTH `qwen3_8b` and `qwen3_14b` (raising on any missing unit!).

---

## 4. Pre-Registered Decision Gates (Matching Astra's Exact Criteria)

1. **Macro-Band Low-Drift Pareto Gate (Primary Contribution, evaluated independently on `v3` & `v2`)**:
   - `4-Cell 95% CI_low >= -2.0 pp` AND `Joint T1∧4C 95% CI_low >= -2.0 pp` vs same-seed `u2_corridor`;
   - `WikiText-2 KL <= 0.75 * u2_KL` (`>= 25%` lower reference drift);
   - M0-anchored relative health gates 100% PASS (`ifeval >= m0 - 10`, `gsm8k >= m0 - 26`, `kl <= 0.005`, `think_pf <= 2%`, `parity == True`).
2. **External Natural-Language `EX-FEVER` Gate (ACL Main Acceptance Decider)**:
   - Report `Turn-1 Acc`, `n_orig_correct`, `n_orig_wrong`, `R_UY` (lower is better), `R_RU_evidence` & `R_RU_user_evidence` (higher is better), and `Selectivity = R_RU - R_UY` across `M0`, `u1`, `u2`, `v4-MLP+`, `v4-RandMask`.
   - Target: Confirm low-drift mid-band `v4-MLP+` preserves/improves evidence-backed rational updating (`R_RU`) without collapsing Turn-1 factual accuracy while suppressing unfounded yielding (`R_UY`).
3. **Fine-Grained B1 Localization Bonus Gate (Separate Secondary Claim)**:
   - Paired `(v4-MLP+ − v4-RandMask)` 95% CI_low `> 0` (and on 8B, second seed same direction + cross-check against `Mask2`), PLUS Probe 2 v2 shared-denominator circuit concentration (`b1_heads_only` & `cross_item_diff_outcome_flip`).
   - Regardless of whether the fine-grained random mask beats or ties on accuracy, report honestly — a tie establishes the clean boundary between causal output localization and mid-band training plasticity!
