# ACL-exp-c2 Closure Summary (`HOLD` & Permanently Frozen)

- **Branch**: `ACL-exp-c2` (permanently frozen after this closure commit)
- **Experimental Results Commit**: `5b492da` (`results(ACL-exp-c2): publish C2 showdown finish`)
- **Model**: `Qwen3-8B` (8-GPU parallel 8-arm showdown, `seed=42`)
- **Final Verdict**: **`HOLD`** — C2 is concluded cleanly without touching `c1_test_id` / `c1_test_ood`, and all verified engineering, data, and mechanistic findings are handed off directly to **`ACL-exp-c3`**.

---

## 1. Executive Summary of C2 Execution & Health

All 8 arms (`M0`, `dac_base`, `dac_paired`, `dac_iit`, `rnd1_base`, `rnd1_iit`, `rnd2_iit`, `unif_base`, `unif_paired` — 8 trained arms + `M0` reference) completed full training, lossless PEFT export (`HF vs vLLM` parity verified under calibrated `du_abs <= 0.50`, `option_match = 98.8%–100%`, `|du_signed| < 0.02`), `no_think` & `think` evaluations across `c2_confirm_bidir`, `c2_stealth_stress`, and `c1_proxy_template_v2`, plus capability & WikiText-2 KL checks.

| Arm | `c2_confirm` `DC(V_hes,F_auth)` (`n=239`) | Row B (`n=36`) | `late_buried` (`n=90`, nt) | `think` Template (`n=120`) | IFEval (`541`) | GSM8K (`1319`) | Wiki KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0` | 0.0% (0/239) | 0/36 | 0/90 | — | 440 | 502 | 0.000000 |
| `dac_base` | 83.3% (199/239) | 19/36 | 0/90 | 95/120 | 448 | 495 | 0.001053 |
| `dac_paired` | 84.5% (202/239) | 25/36 | 0/90 | 94/120 | 445 | 496 | 0.001025 |
| `dac_iit` | 82.8% (198/239) | 22/36 | 1/90 | 96/120 | **451** | 495 | **0.000997** |
| `rnd1_base` | 83.7% (200/239) | 13/36 | 0/90 | 96/120 | 448 | 495 | 0.001102 |
| `rnd1_iit` | 74.9% (179/239) | 11/36 | 0/90 | 91/120 | 446 | 492 | 0.001099 |
| `rnd2_iit` | 77.4% (185/239) | 17/36 | 0/90 | 92/120 | 445 | 492 | 0.001105 |
| `unif_base` | **92.0%** (220/239) | **32/36** | 21/90* | **108/120** | 449 | 497 | 0.001461 |
| `unif_paired` | 86.2% (206/239) | 24/36 | 24/90 | 106/120 | 446 | 494 | 0.001386 |

*\*See Section 2 for the critical 4-cell decomposition showing that `unif_base` collapses from `21/90` to `3/90` when tones are swapped (`V_auth vs F_hes`).*

---

## 2. Verified Findings & Independent Audit (Neon + Astra Read-Only Audit on `5b492da`)

1. **Localization × Algorithm Positive Interaction (Confirmed, Scoped Precisely)**:
   - On `c2_confirm_bidir` (`n=239`), DAC went `199 → 198` while Random went `200 → 179`, yielding a `+8.37 pp` interaction (original 95% CI `[+4.26, +12.61]`; 65-family cluster bootstrap `[+4.3, +13.0]`). B1-localized positions withstand counterfactual interchange training far better than matched random masks.
   - However, `dac_base → dac_iit` changed both paired loss and IIT simultaneously; holding the DAC mask fixed, adding IIT on top of `dac_paired` went `202/239 → 198/239`, and `dac_iit` trailed `unif_base` (`92.0%`) by `-9.21 pp` despite achieving **−32% lower Wiki KL** (`0.000997` vs `0.001461`) and the highest IFEval (`451/541`).
   - On pure localization (`dac_base` vs `rnd1_base`) over all 36 `CE Row B` items: `19/36` vs `13/36` (discordant `7:1`, exact two-sided `p = 0.070`), directional signal worth replicating.
2. **Game-Changing 4-Cell Tone Decomposition Discovery (Two Distinct Failure Modes)**:
   - Decomposing `M0` (`no_think`) into all 4 `(Validity × Tone)` cells reveals two completely different mechanisms that C2 previously conflated:
     - **`late_buried` non-threshold (`90` items)**: `V_hes vs F_auth = 0/90`, **yet tone-matched ordering is `plain = 72/90`, `hes = 82/90`, `auth = 82/90`!** `M0` already possesses ~91% content ordering; rhetoric alone overwhelms it.
     - **`CE Row B` (`36` items)**: `V_hes vs F_auth = 0/36`, and tone-matched ordering remains near-chance (`plain = 8/36`, `hes = 16/36`, `auth = 13/36`). Row B is the true test of building new compositional bindings in `no_think`.
   - **Exposed `unif_base` Reversed-Tone Shortcut on `late_buried`**: `unif_base` scored `21/90` on `(V_hes, F_auth)`, but collapsed to **`3/90`** on swapped tones `(V_auth, F_hes)` and only **`3/90 (3.3%)`** on **4-Cell All-Correct** — shifting authoritative tone negatively on **89/90** items! Conversely, on `CE Row B`, `unif_base` genuinely achieved `32/36` 4-cell all-correct vs `dac_iit` `22/36`.
3. **Cross-Scale B1 Raw-Tensor Anatomy (`Qwen3-8B` & `Qwen3-14B`)**:
   - Top-10 B1 heads carry `56.6%` (8B) / `56.5%` (14B) of selected positive `N_CF` mass (`28.6% / 24.8%` full-net) with **9/10 split-A/B overlap** on both scales — strongly justifying automated cross-half-stable tiered rank allocation rather than uniform per-head rank.
   - Top-layer dialogue MLPs were omitted in `Targeted-Dual`: 8B `C_content_AF` ranks `L35 (#2)`, `L32 (#3)`, `L29 (#5)`; 14B `C_content_CF` ranks `L39 (#1)`, `L38 (#2)`.
   - Top-layer `DLA_pinpoint` heads have heterogeneous form/content signs (e.g., 8B `L29H5`: `DLA=+5.50, form=+0.023`; `L34H30`: `DLA=+3.61, form=-0.216`) and must not be uniformly labeled "brake heads".
4. **Engineering, Data, and Metric Bugs Caught & Logged for C3**:
   - **`c2_build_datasets.py` Style-B semantic rewrite bug**: Line 274 rewrote sufficient `If X` into necessary `Only if X` across 300 items, and lines 1382+ deterministically coupled `(style, domain, default_policy_position)` on 480 non-CE items.
   - **Classical natural sophisms BoW leakage**: Turn-2 BoW 5-fold CV AUC = `0.989` (113/120), dropping to `0.632` leave-class-out; retained strictly as exploratory/historical diagnostic, never primary proof.
   - **Training saturation**: `dac_iit` triggered global grad clipping on `295/300` steps (`rnd_iit` `300/300`), and `med(u_F_auth)` reached `-21` to `-36` already by step 100.
   - **Training IIT metric scope**: `97.5% / 37.5% / 55.0%` reflects training-window post-interchange sign hit rate (`39/40`, `15/40`, `22/40` on `y_src * u_patch > 0`), not held-out counterfactual flip accuracy.
   - **Stats & parity fixes**: `c2_stats.py` had a truthiness bug (`diff or -1.0` turning exact ties `0.0` into `-1.0`); `DC@0` sign-only must never replace real 4-option argmax; parity `du_abs` threshold `0.35` was tighter than vLLM FP8/BF16 numerical noise (`0.37–0.38` at `98.8%–100%` option match) and is calibrated to `0.50` in C3.

---

## 3. Handoff to `ACL-exp-c3`

`ACL-exp-c2` is now **permanently frozen** (`HOLD`). No files under `experiments/ACL-exp-c2/` will be modified again. All next-stage architectural, algorithmic, data, and evaluation upgrades proceed on branch **`ACL-exp-c3`**:

1. **Zero-Retraining 3-Zone Ablation Probe** on C2's saved `unif_base/adapter_state.pt` (disabling Zone 1 / Zone 2 / Zone 3 LoRA deltas independently on Row B & `late_buried` 4-cell).
2. **Fair Decoupled 2×4 8-GPU Showdown on `Qwen3-8B`**: `{Uniform-LoRA, M3-DAC}` × `{Base, Confidence Corridor, Corridor + B1-IIT, Corridor + Matched-Random-IIT}` — making **`U3 (Uniform + B1-IIT)` vs `U4 (Uniform + Rnd-IIT)`** the cleanest localization test and **`M3 (M3-DAC + B1-IIT)` vs `U3`** the cleanest 3-zone capacity-allocation test.
3. **De-Confounded Training + Independent Dev + Brand-New Blind Confirmation Set** (`Only if` fixed, factors independently randomized, balanced Step 1/2/3 breaks, completely held-out compound cross-step + distractor proof structure & disjoint domains).
4. **Un-Hackable Primary Metric**: Family-weighted **4-Cell All-Correct** (`V_hes, V_auth, F_hes, F_auth` all 4 right on true 4-option argmax), held-out post-hoc causal interchange evaluation (original answer, post-patch 4-option flip, tone-swap preservation), and pre-registered dual success paths (Superiority `>= +3.0 pp` OR Efficiency/Pareto `CI_low >= -2.0 pp` & `>= 25%` lower Wiki KL, replicated on `seed=1024`).
