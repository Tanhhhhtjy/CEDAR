# ACL-exp-a1 · Technical Closure & Aligned Behavioral Summary (AI-Readable Reference)

> **Status**: **FROZEN / CLOSED** (2026-11-20). Protocol v2.2 was the final A1 run; no further A1 iterations.
> **Audience**: AI co-investigators (Opus, Fable, Neon, paper-writing & C1 execution agents). Every metric below is verified directly against committed result files in `experiments/ACL-exp-a1/results/v2_2/` (and `results/v2_1/` where explicitly noted).
> **Branch**: `ACL-exp-a1`
> **Key Commits**:
> - Initial A1 v1: `b1f18d5` → v1 results `de6f737`
> - Patch v2.1 (Turn-1 rule order counterbalance, VTR budget, `think` standalone format fix): `f2a0739` → v2.1 results `5e48ef4`
> - Protocol v2.2 (final): `d37c138` (scoped closed-world note, wrappers moved after observation, neutral `system_base`, module 6 2×2 + placebo, module 7 global-verify system prompt, pre-registered hard tier, Gemma budgets, 2 lexical artifact fixes)
> - Final v2.2 GPU results (`qwen3_8b` + `qwen3_14b` `no_think`/`think`, `gemma3_12b` `no_think`): `069f830`

---

## 1. Strict Data Hygiene & Scope Rules (CRITICAL)

- **ALL datasets under `ACL-exp-a1` are `dev_only_pilot` and MUST NEVER enter `ACL-exp-c1` training sets OR held-out evaluation sets**:
  - `symbolic_pilot_120_v2_2_dev_only.jsonl` (`n=120`, SHA-256 `8f558d77...`)
  - `symbolic_hard_120_v2_2_dev_only.jsonl` (`n=120`, SHA-256 `c8c86b5f...`, 3 structural fallacy classes × 2 pair types × depth 3 + 3 inert lure rules)
  - `natural_pilot_seed_12_v2_2.jsonl` (`n=12`, SHA-256 `9de1ffb1...`)
- **Do NOT pool v1 (`results/`), v2.1 (`results/v2_1/`), and v2.2 (`results/v2_2/`)**. All paper main numbers for A1 come strictly from **v2.2**.
- **G0 Turn-2-Only Artifact Audit (ALL PASS)**:
  - `symbolic_pilot`: `context_flip = 0.5000`, `argument_flip = 0.4706`
  - `symbolic_hard`: `context_flip = 0.5000`, `argument_flip = 0.4772`
  - `natural_pilot`: `context_flip = 0.5000`, `argument_flip = 0.5139`

---

## 2. Protocol v2.2 Fixes Verified (What Changed from v2.1)

1. **Closed-world note + wrapper placement**: Scoped closed-world note to the original record (`Verified observations reported later in the conversation count as additional facts`), moved all rhetoric/source wrappers **after** `"New verified ..."` (before `"Step 1:"`).
   - **Verified effect**: In v2.1 `think`, `V_hes` was rejected 89% (8B) / 49% (14B) because models rejected the observation itself. In v2.2 `think`, **`V` and `V_hes` acceptance are `1.000` (100.0%) across ALL datasets (`symbolic_pilot`, `symbolic_hard`, `natural_pilot`) on BOTH Qwen3-8B and Qwen3-14B**!
2. **Neutral system prompt**: Replaced `"Answer using only the information given in the conversation"` with `"You are a helpful assistant."`.
3. **2×2 same-operating-point + placebo (module 6)** and **global-verify system prompt (module 7)** added cleanly.
4. **Positional invariance (`before_rules` vs `after_rules`)**:
   - In `think`: `position_difference = 0.000` (14B std/hard, 8B std) and `-0.023` (8B hard) — **100% order-invariant**.
   - In `no_think`: strong positional sensitivity exists (`8B std: 0.381`, `14B` Turn-1 misreads literal `"inactive"` in `negation_drop` & `scope_swap`), reducing `no_think` eligible items to `89/120` (8B), `62/120` (14B), `106/120` (Gemma). All primary behavioral metrics are computed strictly on Turn-1 eligible items (`turn1_V >= 3/4` AND `turn1_F >= 3/4`).

---

## 3. Complete Verified v2.2 Behavioral Matrix (Eval Half, Eligible Items, `p_target`)

### 3.1 `no_think` Regime (Single-Forward Production Deployment — Prompting & Two-Call Fail!)

| Arm | Qwen3-8B Std (`n_el=89`) | Qwen3-8B Hard (`n_el=74`) | Qwen3-8B Nat (`n_el=10`) | Qwen3-14B Std (`n_el=62`) | Qwen3-14B Hard (`n_el=63`) | Qwen3-14B Nat (`n_el=12`) | Gemma-3-12B Std (`n_el=106`) | Gemma-3-12B Hard (`n_el=93`) | Gemma-3-12B Nat (`n_el=10`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `N` (neutral) | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 |
| `AO` (option mention) | 0.691 | 0.872 | 0.200 | 1.000 | 1.000 | 0.083 | 0.241 | 0.089 | 0.125 |
| `S` (social pressure) | 0.947 | 1.000 | 0.500 | 1.000 | 1.000 | 0.563 | 0.191* | 0.180* | 0.750 |
| **`shell_only`** (irrelevant shell) | **1.000** | **1.000** | **0.900** | **1.000** | **1.000** | **0.917** | **1.000** | **1.000** | **1.000** |
| **`F`** (fallacious) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **`V`** (valid) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `P0 / I0` (no-conflict) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`dialog_explicit_verify_F`** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.975** |
| **`F_sysverify`** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.387 | 0.602 | 0.350 |
| **`V_sysverify`** (over-skepticism!) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.792** | **0.850** | 1.000 |
| `V_hes_sysverify` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.342*** | **0.403*** | 0.900 |
| **`two_call_pipeline_F`** | **0.812** | **0.939** | **0.625** | **0.883** | **1.000** | **0.729** | **0.755** | **0.755** | **0.475** |
| `verify_then_revise_prompt_F` | 0.390 | 0.527 | 0.250 | 0.061 | 0.179 | 0.188 | 0.078* | 0.234* | 0.200* |

*\*Note on Gemma-3-12B `no_think`*: 33/35 arms passed the runner's `format_error_rate <= 5%` gate (`0.00%` on almost all arms, `S` `0.04%` in `stage_readout`); ONLY `verify_then_revise_prompt_V` (`83.4%`) and `verify_then_revise_prompt_F` (`12.4%`) triggered the automated quality-gate abort (due to Markdown/omitted `BrokenStep: NONE` formatting, not token truncation). On `S` and `*_sysverify` symbolic sets, Gemma also refused/formatted differently (`pf` in `model_regime_summary` reflects parse-fail on `Final:`). **Aligned decision (Neon + Opus)**: Freeze A1 without running Gemma CoT; Gemma `no_think` cleanly replicates cross-family 100% argument-form compliance (`AO=0.24 → shell_only=1.00, F=1.00, dialog_explicit_verify_F=1.00`) AND demonstrates the over-skepticism cost of global verify prompts (`V_sysverify` drops from `1.000` to `0.792`).

**Key `no_think` Logit Ladder (Verified in v2.1 & v2.2/B1)**:
- Qwen3-8B: `N (-26.0)` → `AO (+6.8)` → `S (+7.5)` → `shell_only (+20.0)` → `F (+31.5)` → `V (+33.9)`.
- Qwen3-14B: `N (-15.3)` → `AO (+8.4)` → `S (+11.3)` → `shell_only (+20.0)` → `F (+26.8)` → `V (+28.8)`.
- **Policy-Verification Decoupling**: Even in `two_call_pipeline_F`, models ignore explicit `ArgumentCheck: INVALID` from Call-1 (`two_call_pipeline_F = 0.812–1.000` on symbolic, `0.475–0.729` on natural in `no_think`; in v2.1 even conditioned on `Call-1 == INVALID`, 8B still flipped 80%/52% and 14B 39%/20%).

### 3.2 `think` Regime (CoT — Structural Fallacies & Honest Killer-Baseline Audit)

| Arm | Qwen3-8B Std (`120/120`) | Qwen3-8B Hard (`119/120`) | Qwen3-8B Nat (`12/12`) | Qwen3-14B Std (`120/120`) | Qwen3-14B Hard (`120/120`) | Qwen3-14B Nat (`12/12`) |
|---|---:|---:|---:|---:|---:|---:|
| `N / AO / S` | 0.000 | 0.004–0.013 | 0.000 | 0.000–0.006 | 0.000–0.002 | 0.000 |
| `shell_only` | 0.000 | 0.004 | **0.313** | 0.017 | 0.006 | **0.104** |
| **`F` (bare fallacy)** | **0.200** | **0.265** | **0.208** | **0.290** | **0.296** | **0.229** |
| **`V` / `V_hes` / `V_auth`** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| `F_auth` / `F_hes` | 0.158 / 0.098 | 0.179 / 0.124 | 0.188 / 0.167 | 0.219 / 0.063 | 0.229 / 0.042 | 0.167 / 0.021 |
| **`F_src_user`** | 0.127 | 0.164 | 0.167 | **0.138** | **0.148** | **0.146** |
| **`F_src_third_party`** | 0.188 | 0.233 | 0.292 | **0.342** | **0.408** | **0.375** |
| `F_src_self_draft` | 0.117 | 0.120 | 0.188 | 0.221 | 0.244 | 0.146 |
| `F_src_user_critical` | 0.088 | 0.111 | 0.063 | 0.025 | 0.025 | 0.000 |
| `P0 / I0` (no-conflict) | 0.000 | 0.000–0.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`dialog_explicit_verify_F`** | 0.063 | **0.132** | 0.083 | 0.027 | **0.031** | 0.000 |
| **`F_sysverify`** | 0.135 | **0.1513** (≥0.15!) | 0.125 | 0.094 | **0.044** | 0.104 |
| `V_sysverify` / `V_hes_sysverify` | 1.000 | 0.998 / 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `two_call_pipeline_F` | 0.010 | 0.048 | 0.000 | 0.006 | 0.006 | 0.000 |
| `verify_then_revise_prompt_F` | 0.008 | 0.050 | 0.000 | 0.010 | 0.017 | 0.000 |

---

## 4. Same-Operating-Point 2×2 & Placebo Gate (`no_think`, `symbolic_pilot`, `context_flip`)

$I_{\text{content}} = [u(V) - u(F)] - [u(\text{shell\_ctxV}) - u(\text{shell})]$, $I_{\text{placebo}} = [u(F_{\text{placebo}}) - u(F)] - [u(\text{shell}_{\text{placebo}}) - u(\text{shell})]$.

| Model | Scope | Median $I_{\text{content}}$ [95% CI] | Sign Consistency | Classes $\ge 0.70$ | Median $\|I_{\text{content}}\|$ vs $\|I_{\text{placebo}}\|$ | Pre-Registered Gate |
|---|---|---:|---:|---:|---:|---|
| **Qwen3-14B** | All `context_flip` (`n=60`) | **`+1.250`** `[+0.375, +1.812]` | `0.700` | **4 / 6** | `2.000` vs `0.500` | **✅ PASSED** (unlocked B1 `C_content_CF`) |
| **Qwen3-14B** | **Turn-1 Eligible (`n=26`)** | **`+2.125`** `[+1.438, +2.750]` | **`0.850`** | *(1 item each in neg/scope)* | stronger | **Confirms signal concentrates on eligible items** |
| **Qwen3-8B** | All `context_flip` (`n=60`) | `+1.000` `[+0.500, +1.375]` | `0.700` | 3 / 6 (`scope_swap=0.60`) | `1.000` vs `0.500` | ❌ Failed (1 class short) |
| **Gemma-3-12B** | All `context_flip` (`n=60`) | `0.000` `[+0.000, +0.125]` | `0.483` | 2 / 6 | `0.250` vs `0.250` | ❌ Failed |

**Critical Audit Notes (Opus + Neon + Fable Aligned)**:
1. **Why 14B `negation_drop` had $I_{\text{content}} = -2.875$**: NOT reversed content sensitivity — on `context_prompt_v` `negation_drop`, 14B `no_think` misreads the literal `"inactive"` match at Turn-1 (100% wrong, $u=+17.5$), causing large `D_shell = +3.4` while `V` and `F` are ceiling-compressed near `+29` (`D_arg = +0.9`). Restricting to Turn-1 eligible items jumps $I_{\text{content}}$ from `+1.25` to **`+2.125`** (`85%` positive).
2. **Placebo baseline & bf16 granularity**: Placebo median is `-0.50` (`95%` negative), showing any irrelevant context edit introduces ~`0.5` logit drift (and bf16 step size at $|u|\approx 30$ is `0.125`). Report content interaction relative to placebo (`+1.75` to `+2.6` logits above placebo), not merely against zero.
3. **Transparent DiD & Standalone Reporting (Fable + Opus rule)**: Always report all three tiers honestly: (i) `argument_flip` $V-F$ (97–98% positive, Turn-2 differs), (ii) `context_flip` raw $V-F$ (`80.3%` 8B / `92.4%` 14B), (iii) Turn-1 DiD (`48%`/`57%`) + same-operating-point 2×2 (`+1.25` all / `+2.125` eligible on 14B) + zero-lexical standalone `C_content_N_CF` (`98.5%` 8B / `100%` 14B).

---

## 5. Mandatory Paper Wording Rules for Act I (Aligned Across Neon, Opus, Fable)

1. **Source Attribution (G3) — ALWAYS Compare Against Bare `F`**:
   - On 14B `think`, `F_src_user` (`0.138 / 0.148 / 0.146`) is **lower** than bare `F` (`0.290 / 0.296 / 0.229`), while `F_src_third_party` (`0.342 / 0.408 / 0.375`) is `+0.05` to `+0.15` above bare `F`.
   - **Correct claim**: Attributing an argument to "a note the user wrote" triggers **heightened scrutiny** relative to bare `F`, whereas third-party objective documentation adds a modest authority boost (`+5–15 pp`). This refutes pure "fear of disagreeing with the user" sycophancy; speaker certainty (`F_hes = 0.04–0.06` vs `F_auth = 0.22–0.23`) and framing modulate verification scrutiny.
2. **Honest Killer-Baseline Framing (Zero Evasion)**:
   - On **14B `think` hard tier**, explicit verify prompts remain strong (`dialog_explicit_verify_F = 3.1%`, `F_sysverify = 4.4%`); on **8B `think` hard tier**, `F_sysverify = 15.13%` marginally crosses the pre-registered `15%` prompt-failure threshold (and `dialog_explicit_verify_F = 13.2%`, >2× standard tier).
   - **Where prompting genuinely collapses**: (a) **`no_think` single-forward deployment** (`dialog_explicit_verify_F = 100%`, `F_sysverify = 100%` on Qwen, `two_call = 81%–100%` on symbolic), (b) **cross-family over-skepticism cost** (`Gemma V_sysverify` drops to `79.2%`), (c) **2× latency / 1000+ CoT token cost** of `think` / `two_call`, and (d) **black-box fragility** vs. default-on, zero-overhead, mechanistically grounded circuit tuning (`ACL-exp-c1`).
