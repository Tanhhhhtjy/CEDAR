# ACL-exp-c3.5 Stage 6 Strict Closure Readout

## 1. Qwen3-14B Dual-Seed (`seed=42, 1024`) on `c3_blind_confirm_v3` (`900` Units + `802` Non-Shortcut)

| Model | Seed | 4-Cell (900 units) | Joint (900 units) | Turn-1 Dual (1080 items) | Wiki KL | IFEval / GSM8K | Health |
|---|---:|---:|---:|---:|---:|---:|---|
| `u1_base` | 42 | **53.67%** | **53.22%** | 98.24% | `0.001077` | 461 / 642 | PASS |
| `u1_base` | 1024 | **54.00%** | **53.72%** | 98.61% | `0.001092` | 467 / 653 | PASS |
| `u1_base` | **pooled_2seed** | **53.83%** | **53.47%** | 98.43% | `0.001084` | — | — |
| `u2_corridor` | 42 | **55.22%** | **54.56%** | 98.15% | `0.001164` | 462 / 636 | PASS |
| `u2_corridor` | 1024 | **57.44%** | **57.06%** | 98.43% | `0.001147` | 469 / 644 | PASS |
| `u2_corridor` | **pooled_2seed** | **56.33%** | **55.81%** | 98.29% | `0.001156` | — | — |
| `M3-v4-PureMid-MLP+` | 42 | **59.67%** | **58.22%** | 95.74% | `0.000814` | 461 / 635 | PASS |
| `M3-v4-PureMid-MLP+` | 1024 | **58.78%** | **57.33%** | 96.02% | `0.000829` | 465 / 639 | PASS |
| `M3-v4-PureMid-MLP+` | **pooled_2seed** | **59.22%** | **57.78%** | 95.88% | `0.000821` | — | — |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **58.28%** | **56.83%** | 96.20% | `0.000826` | 467 / 617 | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 1024 | **57.56%** | **56.44%** | 97.22% | `0.000833` | 463 / 627 | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | **pooled_2seed** | **57.92%** | **56.64%** | 96.71% | `0.000829` | — | — |

### 14B Paired Bootstrap (All `900` Units + Post-Hoc `802` Non-Shortcut Sensitivity)

*Note: Paired 95% CIs reflect item/unit sampling uncertainty given fixed training seeds (`n_boot=5000`). Zero two-sided tail hits reported as `0/5000` (min nonzero two-sided resolution `0.0004`); exact sign-flip p-values logged in JSON.*

| Comparison | Split | Subset | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | Boot `p` (4C) | Sign-Flip `p` (4C) |
|---|---|---|---:|---:|---:|---:|
| `v4_vs_u2` | `seed_42` | `all_900` | **+4.44** `[+3.06, +5.89]` | **+3.67** `[+2.11, +5.28]` | `0/5000` | `< 0.0001` |
| `v4_vs_u2` | `seed_42` | `non_shortcut_802` | **+1.87** `[+0.75, +3.12]` | **+0.87** `[-0.37, +2.24]` | `0.0004` | `0.0161` |
| `v4_vs_u2` | `seed_1024` | `all_900` | **+1.33** `[-0.17, +2.83]` | **+0.28** `[-1.33, +1.83]` | `0.0896` | `0.0141` |
| `v4_vs_u2` | `seed_1024` | `non_shortcut_802` | **-0.19** `[-1.62, +1.18]` | **-1.37** `[-2.93, +0.12]` | `0.8108` | `0.4350` |
| `v4_vs_u2` | **`pooled_2seed`** | `all_900` | **+2.89** `[+1.61, +4.17]` | **+1.97** `[+0.56, +3.31]` | `0/5000` | `< 0.0001` |
| `v4_vs_u2` | **`pooled_2seed`** | `non_shortcut_802` | **+0.84** `[-0.19, +1.90]` | **-0.25** `[-1.43, +0.94]` | `0.1160` | `0.0385` |
| `v4_vs_randmask` | `seed_42` | `all_900` | **+1.39** `[+0.56, +2.28]` | **+1.39** `[+0.56, +2.28]` | `0.0012` | `0.0192` |
| `v4_vs_randmask` | `seed_42` | `non_shortcut_802` | **+0.87** `[+0.12, +1.68]` | **+0.87** `[+0.12, +1.68]` | `0.0284` | `0.1892` |
| `v4_vs_randmask` | `seed_1024` | `all_900` | **+1.22** `[+0.50, +2.00]` | **+0.89** `[+0.11, +1.72]` | `0.0004` | `0.0026` |
| `v4_vs_randmask` | `seed_1024` | `non_shortcut_802` | **+0.75** `[+0.06, +1.44]` | **+0.37** `[-0.31, +1.12]` | `0.0384` | `0.0490` |
| `v4_vs_randmask` | **`pooled_2seed`** | `all_900` | **+1.31** `[+0.64, +2.00]` | **+1.14** `[+0.44, +1.86]` | `0/5000` | `0.0034` |
| `v4_vs_randmask` | **`pooled_2seed`** | `non_shortcut_802` | **+0.81** `[+0.22, +1.47]` | **+0.62** `[+0.03, +1.28]` | `0.0072` | `0.0987` |

## 2. Qwen3-14B Fixed `step=200` Matched-Step Confound Audit (`v3` & WikiText-2 KL)

| Model (at exact `step 200`) | Seed | 4-Cell (900 units) | Joint (900 units) | Wiki KL (`step 200`) |
|---|---:|---:|---:|---:|
| `u2_corridor` | 42 | **55.44%** | **55.00%** | `0.001146` |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **59.50%** | **58.39%** | `0.000818` |
| `u2_corridor` | 1024 | **55.44%** | **54.72%** | `0.001139` |
| `M3-v4-PureMid-MLP+` | 1024 | **58.67%** | **57.44%** | `0.000824` |
| `M3-v4-PureMid-MLP+-Random-Mask` | 1024 | **57.44%** | **56.56%** | `0.000828` |
| `M3-v4-PureMid-MLP+` | 42 | **59.67%** | **58.22%** | `0.000814` |

## 3. Entity-Count-Balanced Diagnostic (`c3_entity_balanced_v1`, 480 Units, Exact 50/50 Entity Count)

| Model | Overall 4C | `1v2_orig` | `2v1_anti` | `Δ_shortcut` | `1v1_same` | `2v2_same` | `1v1_ctx` | `2v2_ctx` | Same-Ent Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s6_entity_8b_s42_M0` | **0.00%** | 0.0% | 0.0% | **+0.0** | 0.0% | 0.0% | 0.0% | 0.0% | **0.00%** |
| `s6_entity_8b_s42_u2_corridor` | **1.46%** | 1.2% | 0.0% | **+1.2** | 1.2% | 1.2% | 1.2% | 3.8% | **1.88%** |
| `s6_entity_8b_s42_v4_mlpplus` | **2.92%** | 1.2% | 0.0% | **+1.2** | 1.2% | 3.8% | 2.5% | 8.8% | **4.06%** |
| `s5_8b_s42_v4_randmask1` | **2.08%** | 1.2% | 0.0% | **+1.2** | 1.2% | 1.2% | 2.5% | 6.2% | **2.81%** |
| `s6_entity_14b_s42_M0` | **0.00%** | 0.0% | 0.0% | **+0.0** | 0.0% | 0.0% | 0.0% | 0.0% | **0.00%** |
| `s5_14b_s42_u2_corridor` | **47.71%** | 52.5% | 13.8% | **+38.8** | 53.8% | 63.8% | 48.8% | 53.8% | **55.00%** |
| `s5_14b_s42_v4_mlpplus` | **58.13%** | 78.8% | 12.5% | **+66.2** | 52.5% | 78.8% | 55.0% | 71.2% | **64.38%** |
| `s5_14b_s42_v4_randmask` | **53.75%** | 71.2% | 11.2% | **+60.0** | 51.2% | 71.2% | 53.8% | 63.8% | **60.00%** |

## 4. Qwen3-8B External SOTA & Internal Loss Ablations (`v3`, KL, Capabilities, EX-FEVER)

| Method | `v3` 4C (900) | `v3` 4C (802) | `v3` Joint | `v3` NLL (4-tone) | `|u|` med (4-tone) | Wiki KL | IFEval / GSM8K | EX-FEVER `R_UY / R_RU_ev / Sel` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `s6_8b_s42_pinpoint_dla_corridor` | **2.17%** | **2.31%** | **1.39%** | 0.73 | 0.500 | `0.000794` | 440 / 551 | 32.91% / 78.98% / 46.07 |
| `s6_8b_s42_pinpoint_dla_sft` | **1.72%** | **1.87%** | **0.61%** | 0.76 | 0.500 | `0.000792` | 446 / 558 | 32.15% / 78.31% / 46.15 |
| `s6_8b_s42_u2_corridor_only` | **36.83%** | **40.15%** | **36.22%** | 1.32 | 6.000 | `0.001349` | 441 / 529 | 9.46% / 70.48% / 61.02 |
| `s6_8b_s42_u2_tone_inv_only` | **40.56%** | **43.02%** | **40.28%** | 1.66 | 8.750 | `0.001371` | 441 / 528 | 8.36% / 68.03% / 59.67 |
| `s6_8b_s42_joint_dpo_best` | **42.50%** | **42.58%** | **42.50%** | 15.17 | 67.250 | `0.001855` | 444 / 474 | 22.17% / 74.08% / 51.91 |
| `caa_dev_selected_pos` | **1.89%** | **2.12%** | **1.61%** | 4.59 | 10.500 | `0.080674` | — | 16.51% / 75.80% / 59.29 |

## 5. 8B vLLM Backend Parity Audit (Auto-Generated from Raw Parity JSONs)

All `no_think` (`v2`, `v3`, `c3_entity_balanced_v1`, `WikiText-2 KL`) metrics run natively in HuggingFace BF16 (`backend=hf`, exact). Generative evaluations (`think`, `capabilities`, `EX-FEVER`) use the vLLM backend logged below:

| Method | Generative Backend Used | Native LoRA Parity (Opt / `|Δu|` mean) | BF16 Merged Parity (Opt / `|Δu|` mean) | Audit Status |
|---|---|---|---|---|
| `s6_8b_s42_pinpoint_dla_corridor` | `vllm_bf16_merged` | FAIL (`93.75%` / `0.10873`) | PASS (`96.25%` / `0.08125`) | **PASS (via BF16 merged)** |
| `s6_8b_s42_pinpoint_dla_sft` | `vllm_native_lora` | PASS (`96.25%` / `0.08864`) | — | **PASS** |
| `s6_8b_s42_u2_corridor_only` | `vllm_native_lora` | PASS (`100.00%` / `0.15887`) | — | **PASS** |
| `s6_8b_s42_u2_tone_inv_only` | `vllm_native_lora` | PASS (`96.25%` / `0.43462`) | — | **PASS** |
| `s6_8b_s42_joint_dpo_best` | `vllm_native_lora` | FAIL (`98.75%` / `1.75271`) | FAIL (`76.25%` / `26.62148`) | **CAVEAT** (`no_think` exact HF; generative parity FAIL) |
