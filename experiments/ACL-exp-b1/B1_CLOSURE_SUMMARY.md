# ACL-exp-b1 · Technical Closure & Aligned Mechanistic Summary (AI-Readable Reference)

> **Status**: **FROZEN / CLOSED** (2026-11-20).
> **Audience**: AI co-investigators (Opus, Fable, Neon, paper-writing & C1 execution agents). Every metric below is verified directly against committed result files in `experiments/ACL-exp-b1/results/`.
> **Branch**: `ACL-exp-b1`
> **Key Commits**:
> - Protocol & initial code: `9988929` (Opus) → `a203203` (Neon audit fixes: multi-GPU DLA safety, out-of-sample `stage_dissociation` B→A, directional `E >= 0.5`, added `C_pinpoint` & `C_content_N_CF`, `resid_per_position_max_suffix: 48`)
> - Full-set GPU runs (`qwen3_8b` + `qwen3_14b`, including 14B deferred `context_flip`): `1a40806`
> - Eligible-subset verification code & frozen IDs (`origin/ACL-exp-a1@069f830`): `d5a3d55` → `d0067ee`
> - Eligible-subset GPU runs (`qwen3_8b_eligible` + `qwen3_14b_eligible`): `97e7cbd`

---

## 1. Executive Summary & Locked Project Main Line

**North-Star Thesis (ACL 2027 Main Conference, ARR 2027-01-04 AoE)**:
1. **Act I (`ACL-exp-a1`)**: Prior anti-sycophancy work focuses on shallow social pressure (`S`). Using solver-certified minimal valid/fallacious pairs (`V` vs `F`, especially byte-identical Turn-2 `context_flip` twins) and content-free argument shells (`shell_only`), we show models exhibit severe **argument-form compliance** (100% blind flipping in `no_think` across Qwen3-8B/14B and Gemma-3-12B, ~20–30% on structural fallacies in `think`, and single-sentence prompts completely fail in `no_think`).
2. **Act II (`ACL-exp-b1`, this branch)**: White-box causal localization (`no_think` continuous logit readout $u_\theta = \text{logit}(a_t) - \text{logit}(a_0)$ and `ArgumentCheck: VALID - INVALID`) via full-graph denoising **AtP → out-of-sample real patching (B→A) vs 5-seed `Random-LM` → split-half reliability → out-of-sample dissociation matrix**, compared directly against **Pinpoint-style Direct Logit Attribution (DLA)**.
3. **Act III (`ACL-exp-c1`, next branch)**: Mechanism-guided **Dual-Pathway Asymmetric Circuit Fine-Tuning** targeting the exact validated heads (GQA `Q/O` slices) and mid-layer verification MLPs (low-rank `gate_proj/down_proj`), benchmarked at strictly matched trainable parameter budgets against `Pinpoint-DLA`, `Pinpoint-AtP`, `Targeted-Form`, `Random-LM`, `Uniform-LoRA`, and `Hub-Only / Dual-wo-Hub` ablations.

---

## 2. Underlying Residual-Stream Physical Model (Unifying A1, B1, and C1)

At the decision readout position (`Final:` + 1), the logit margin is:
$$u_\theta(x) = \frac{\Delta w_U^\top \operatorname{diag}(g)}{\operatorname{RMS}(h_L(x))} \Big( h_{\text{init}}(x) + h_{\text{soc+form}}(x) + h_{\text{ver}}(x) \Big)$$
where $\Delta w_U = W_U[a_t] - W_U[a_0]$. Empirically in $M_0$ (`no_think`):
- **$h_{\text{init}}(x)$** anchors the initial answer (`N`: $-26.0$ on 8B, $-15.3$ on 14B).
- **$h_{\text{soc+form}}(x)$** acts as a **common-mode compliance bias** (head-driven, MLP-independent), pushing *both* `V` and `F` (and irrelevant `shell_only`) up by $+33\text{--}45$ logits into saturation ($u \approx +20\text{--}34$).
- **$h_{\text{ver}}(x)$** acts as a **differential-mode verification signal** (computed by mid-layer heads + 5 mid-layer MLPs using nonlinear SwiGLU gating), cleanly separating `V > F` in standalone (`+7.63` logits, 100% positive on eligible items for both 8B & 14B) and weakly coupled in dialogue (`+1.25` on 14B eligible `C_content_CF`, 96.9% positive; 2×2 interaction `+2.125`, 85% positive).
- **Why coarse residual capping (`p25`, abandoned AAAI track) failed**: Scaling a broad high-norm residual direction rescales $h_L$ globally ($u_{\text{cap}} \approx c \cdot u_\theta$), contracting both $V$ and $F$ proportionally (98–105% global logit contraction) with 0/192 sign flips because it cannot change the gain ratio $\beta_{\text{ver}} / \gamma_{\text{form}}$.
- **Why Pinpoint-style DLA hits a ceiling**: DLA computes single-hop direct projection $\Delta w_U^\top o_{\ell,h}$, structurally trapping itself in late output layers (`L28–39` on 14B, `L22–35` on 8B), completely missing both the **mid-layer shared routing hub heads** and the **5 mid-layer verification MLPs** (`0.000` weighted overlap with `C_content_N_CF`; only `12%–17%` normalized recovery).
- **Why over-recovery (`> 1.0`) occurs in `C_form` / `C_soc`**: The compliance pathway is a **push–pull opponent circuit** ($E = E_+ - E_-$); denoising only the top-$k$ positive-AtP heads restores $E_+$ without restoring competing inhibitory heads $E_-$, yielding $E_+ / (E_+ - E_-) > 1$.

---

## 3. Complete Verified Quantitative Results (Full Set vs. Eligible Subset)

### 3.1 Contrasts & Frozen Pairs (`data/b1_pairs_v2_2.jsonl`, 858 total pairs)
- `C_pinpoint`: `S` vs `N` (132 full; 99 elig 8B / 74 elig 14B) — Pinpoint-style social suggestion vs neutral. **Caveat (aligned across Neon/Opus)**: in `no_think`, `S - N` is dominated by option-mention exposure (`AO - N` $\approx +23\text{--}34$ logits), whereas original Pinpoint `"Are you sure?"` does not name $a_t$. State this honestly in the paper; `C_soc` (`S - AO`) isolates pure assertive tone.
- `C_soc`: `S` vs `AO` (132 full; 99 elig 8B / 74 elig 14B) — pure assertive tone increment.
- `C_form`: `shell_only` vs `AO` (132 full; 99 elig 8B / 74 elig 14B) — argument-form compliance.
- `C_Fshell`: `F` vs `shell_only` (132 full) — plausible rule-citing fallacy beyond irrelevant shell.
- `C_content_AF`: `V` vs `F` (`argument_flip`, dialog, 66 full; 47 elig 8B).
- `C_content_N`: `standalone_neutral_V` vs `F` (`argument_flip`, `ArgumentCheck: VALID - INVALID`, 66 full).
- `C_content_N_CF`: `standalone_neutral_V` vs `F` (`context_flip`, **byte-identical argument**, `ArgumentCheck: VALID - INVALID`, 66 full; 52 elig 8B / 32 elig 14B) — **primary zero-lexical-artifact verification anchor**.
- `C_content_CF` (deferred, unlocked on 14B by A1 2×2 gate): `V(ctxV)` vs `F(ctxF)` (`context_flip`, dialog, 66 full; 32 elig 14B).
- `C_ctx_shell` (deferred, 14B): `shell_only_ctxV` vs `shell_only` (`context_flip`, 66 full). **Audited decision**: `C_ctx_shell` median effect is `0.00` (44% positive); subtracting its noisy AtP map from `C_content_CF` injects noise (`context_flip_interaction_atp.pt` is NOT used). Use `C_content_CF` directly.

### 3.2 Qwen3-14B (40 layers, 40 heads/layer): Full (`1a40806`) vs. Eligible (`97e7cbd`, 74 bases: 62 sym + 12 nat)

| Contrast | Full Med $E$ (sign%) | Elig Med $E$ (sign%) | Full→Elig `J@20` | Full Split-Half `J@20` (ceiling) | Weighted Mass Retained@20 | `H@20` Recovery (Elig, vs rnd max) | `M@5` Recovery (Elig, vs rnd max) | Elig Top-5 MLP Layers |
|---|---|---|---|---|---|---|---|---|
| `C_pinpoint` | `+28.00` (100%) | `+29.88` (100%) | `0.739` | `0.818` | **91.7%** | `+0.908` (`+0.034`) PASS | `+0.493` (`+0.106`) PASS | `[26, 35, 37, 21, 31]` (`J@5=1.00`) |
| `C_soc` | `+4.12` (99.2%) | `+5.00` (100%) | **`0.818`** | `0.818` | **94.5%** | `+2.186` (`-0.170`) PASS | `+0.680` (`+0.095`) PASS | `[15, 10, 35, 19, 14]` (`J@5=1.00`) |
| `C_form` | `+10.62` (100%) | `+11.50` (100%) | **`0.905`** | `0.905` | **97.3%** | `+1.187` (`-0.044`) PASS | **`+0.181` (`+0.216`) FAIL** | `[23, 24, 21, 27, 14]` (MLP ≈ rnd) |
| `C_content_N_CF` | `+3.06` (98.5%) | **`+7.63` (100%)** | **`0.905`** | `0.818` | **97.4%** | `+1.046` (`+0.343`) PASS | **`+0.918` (`+0.557`) PASS** | **`[22, 23, 25, 19, 26]`** (identical to full!) |
| `C_content_CF` | `+0.75` (92.4%) | **`+1.25` (96.9%)** | **`0.739`** | `0.539` | **91.2%** | `+0.777` (`+0.134`) PASS | **`+0.836` (`+0.420`) PASS** | `[39, 38, 25, 27, 20]` (`J@5=1.00`) |

- `DLA(C_pinpoint)` full→elig `J@20 = 1.000`; `DLA(C_soc)` full→elig `J@20 = 0.905`.

**14B Eligible Out-of-Sample Normalized Dissociation Matrix $D[X \to Y] = R[X,Y] / R[Y,Y]$ (`k=20`, B→A)**:

| Source $X$ \ Target $Y$ | `C_pinpoint` (n=32) | `C_soc` (n=36) | `C_form` (n=40) | `C_content_N_CF` (n=21) | `C_content_CF` (n=12) |
|---|---:|---:|---:|---:|---:|
| `C_pinpoint` (AtP) | **1.000** | 0.317 | 0.469 | 0.898 | 0.910 |
| `C_soc` (AtP) | 0.978 | **1.000** | 0.923 | 0.218 | 0.309 |
| `C_form` (AtP) | 0.938 | 0.776 | **1.000** | 0.603 | 0.324 |
| `C_content_N_CF` (AtP) | 0.645 | -0.158 | **0.140** | **1.000** | **1.095** |
| `C_content_CF` (AtP) | 0.606 | -0.015 | **0.060** | **1.000** | **1.000** |
| **`DLA(C_pinpoint)`** | 0.921 | 0.544 | 0.667 | **0.171** | **0.324** |
| **`DLA(C_soc)`** | 0.796 | 0.532 | 0.642 | **0.117** | **0.272** |
| `Random-LM` (source=`DLA(pin)`, mean) | -0.005 | +0.023 | -0.016 | +0.028 | +0.031 |
| `Random-LM` (target=$Y$, mean) | -0.039 | -0.436 | -0.284 | -0.004 | +0.009 |

### 3.3 Qwen3-8B (36 layers, 32 heads/layer): Full (`1a40806`) vs. Eligible (`97e7cbd`, 99 bases: 89 sym + 10 nat)

| Contrast | Full Med $E$ (sign%) | Elig Med $E$ (sign%) | Full→Elig `J@20` | Full Split-Half `J@20` | Weighted Mass Retained@20 | `H@20` Recovery (Elig, vs rnd max) | `M@5` Recovery (Elig, vs rnd max) | Elig Top-5 MLP Layers |
|---|---|---|---|---|---|---|---|---|
| `C_pinpoint` | `+34.50` (100%) | `+34.75` (100%) | **`0.905`** | `0.905` | **97.2%** | `+0.921` (`-0.028`) PASS | `+0.313` (`+0.176`) PASS | `[20, 15, 21, 34, 33]` |
| **`C_soc`** | `+2.25` (65.2% ❌) | **`+4.75` (80.8% ✅)** | **`1.000`** | `0.818` | **100.0%** | `+5.626` (`-0.143`) PASS | `+0.214` (`-0.103`) PASS | `[12, 24, 22, 15, 31]` (`J@5=1.00`) |
| `C_form` | `+10.75` (88.6%) | `+13.50` (99.0%) | `0.667` | `0.818` | **88.8%** | `+1.528` (`+0.000`) PASS | `+0.332` (`+0.193`) | `[20, 17, 12, 22, 24]` (`J@5=1.00`) |
| `C_content_N_CF` | `+6.38` (100%) | **`+7.63` (100%)** | **`0.905`** | `0.739` | **97.2%** | `+0.993` (`+0.211`) PASS | **`+0.831` (`+0.325`) PASS** | **`[22, 23, 18, 20, 27]`** (`J@5=1.00`) |
| `C_content_AF` | `+2.62` (95.5%) | `+2.75` (95.7%) | **`0.905`** | `0.600` | **97.2%** | `+1.160` (`+0.310`) PASS | `+0.637` (`+0.193`) PASS | `[22, 35, 32, 23, 29]` (`J@5=1.00`) |

**8B Eligible Normalized Dissociation Matrix $D[X \to Y]$ (`k=20`, B→A)**:
- `C_form → C_content_N_CF = 0.584`, **`C_content_N_CF → C_form = 0.199`** (pre-registered double dissociation PASSED on both full and eligible!).
- `C_form → C_content_AF = 0.442`, **`C_content_AF → C_form = 0.063`** (PASSED on both full and eligible!).
- `DLA(C_pinpoint) → C_content_N_CF = 0.253` (weighted overlap **`0.000`**; source-matched random mean `+0.043`).

---

## 4. Aligned Mechanistic Conclusions & Mandatory Paper Wording Rules (Neon + Opus + Fable)

1. **Head-level & Component-type dissociation — NOT "middle-layer vs late-layer" depth dichotomy**:
   - Full-suffix `stage_resid_sweep` 80% arrival layers (L16/L18 for `N_CF` vs L32/L38 for `form`) were **confounded by suffix length** (363 tokens vs 11 tokens).
   - Controlled to the **last readout token only**, arrival is L24 (8B) / L28 (14B) for `C_content_N_CF` vs L32 (8B) / L36+ (14B) for `C_form` — only slightly earlier (~4–8 layers). Under the same `AtP` method, head layer bands overlap heavily (`C_form` med L22.5/24.5 vs `C_content_N_CF` med L19/21).
   - **Correct paper claim**: Dissociation occurs at the **head level** within overlapping mid-late bands AND at the **component-type level** (`C_form` is head-driven with MLP ≈ random; `C_content_N_CF` & `C_content_CF` require both mid-layer heads and 5 mid-layer MLPs recovering 83%–92%).
   - **Valid critique of prior work**: Direct-path localization (`DLA`, Pinpoint-style) structurally concentrates in late output layers (`L28–39` on 14B, `L22–35` on 8B), yielding **`0.000` weighted overlap** with `C_content_N_CF` and missing both mid-layer routing hubs and all MLPs.
2. **Shared Mid-Layer Routing Hub (`Hub Heads`)**:
   - Why `AtP(C_pinpoint)` recovers `C_content_N_CF` at `0.90–0.93` despite low Jaccard (`0.11`): a small set of shared mid-layer hub heads — on 14B full: **`(23,4), (24,17), (24,19), (25,15)`** (carrying **31.1%** of `C_content_N_CF`'s top-20 AtP mass; `L23H4` alone has normalized `N_CF` score `0.91` and is #1 in both full and eligible `N_CF`; `(24,17), (24,19)` remain in eligible top-20 overlap); on 8B: **`(19,31), (20,8), (21,19)`** (unchanged across full and eligible).
   - **Anatomical picture**: Upstream form/social entry heads (`C_form` / `C_soc`) and mid-layer verification MLPs (`L19–26`) converge through shared mid-layer routing hubs before reaching late output assembly heads (`DLA`, `L28–39`).
3. **Standalone (`C_content_N_CF`) ↔ Dialogue (`C_content_CF`) Circuit Equivalence (14B Eligible Breakthrough)**:
   - On 14B eligible items, `C_content_N_CF → C_content_CF` normalized recovery is **`1.095`** and `C_content_CF → C_content_N_CF` is **`0.9995`**, while `C_content_CF → C_form` is **`0.060`** and `→ C_soc` is **`-0.015`**. Thus, the zero-lexical-artifact standalone verification heads are **functionally identical** to the dialogue-mode logical verification heads!
4. **Honest Reporting of All 5 Pre-Registered Dissociation Tests & Over-Recovery Normalization**:
   - Report all 5 pre-registered `+0.20` tests transparently: **14B 3/5 PASS** (`C_form~N_CF`, `C_form~AF`, `C_pinpoint~C_form`; `C_soc~C_form` and `C_pinpoint~N_CF` FAIL); **8B 2/5 PASS** (`C_form~N_CF`, `C_form~AF`). Both full and eligible give identical pass/fail decisions.
   - Because over-recovery (`> 1.0`) inflates raw differences for small-$E$ contrasts, report **normalized dissociation $D[X\to Y] = R[X,Y]/R[Y,Y]$**, **median recovery**, **clamped `[-1,2]` recovery**, **raw `delta_m`**, **weighted AtP overlap**, and **both target-Y and source-X random controls**.
5. **Turn-1 Eligibility Explains 8B `C_soc` Failure**:
   - On 8B full set, `C_soc` failed sanity (`+2.25`, 65.2% positive) purely due to Turn-1 ineligible items (`-3.50`, 18% positive). On eligible items, 8B `C_soc` cleanly passes (`+4.75`, **80.8% positive**) with `J@20 = 1.000` and `100.0%` mass retention.

---

## 5. Frozen Component Interface for `ACL-exp-c1`

All C1 component sets are deterministically exported from `results/qwen3_14b_eligible/` and `results/qwen3_8b_eligible/` (cross-checked against full runs; full-set `C_pinpoint ∩ C_content_N_CF` hub heads `(23,4), (24,17), (24,19), (25,15)` on 14B and `(19,31), (20,8), (21,19)` on 8B are explicitly tagged for hub ablations):
- **`Pinpoint-DLA`**: `DLA(C_pinpoint)` top-20 heads (`Q/O` slices only, all MLPs frozen).
- **`Pinpoint-AtP`**: `AtP(C_pinpoint)` top-20 heads (`Q/O` slices only, all MLPs frozen — eliminates strawman baseline critique).
- **`Targeted-Form`**: `AtP(C_form)` top-20 $\cup$ `AtP(C_soc)` top-12 heads (`Q/O` slices only, **no MLP**).
- **`Targeted-Dual` (Primary Method)**: `Targeted-Form` heads $\cup$ `AtP(C_content_N_CF)` top-20 $\cup$ `AtP(C_content_CF)` top-10 (on 14B; `C_content_AF` top-10 on 8B) heads (`Q/O` slices) **+ `C_content_N_CF` Top-5 Mid-Layer MLPs** (`14B: [19, 22, 23, 25, 26]`; `8B: [18, 20, 22, 23, 27]`) low-rank LoRA (`gate_proj` + `down_proj`).
- **`Random-LM`**: Per-layer head count & MLP layer count matched to `Targeted-Dual` (excluding `Targeted-Dual` units).
- **`Uniform-LoRA`**: All layers Attention + MLP low-rank LoRA, rank matched to **`Targeted-Dual`'s exact total trainable parameter count**.
- **Hub Ablations**: `Hub-Only` (4 hub heads on 14B / 3 on 8B) vs. `Dual-wo-Hub` (`Targeted-Dual` with hub heads frozen).
