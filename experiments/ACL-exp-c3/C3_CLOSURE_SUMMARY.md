# ACL-exp-c3 Closure Summary (`HOLD` / Stop Old 14B Full Matrix → Handoff to `ACL-exp-c3.5`)

- **Branch**: `ACL-exp-c3` (permanently frozen after this closure commit)
- **Experimental Results Commit**: `c9def7c` (`results(ACL-exp-c3): publish finish_after_cap_retry`)
- **Model**: `Qwen3-8B` (8-GPU parallel 2×4 showdown, `seed=42`)
- **Final Verdict**: **`HOLD` (Old C3 recipe triggers 14B full-matrix stop; handoff to cleanly scoped 8B `ACL-exp-c3.5`!)**

---

## 1. Verified Execution & Results Summary (`c9def7c`)

All 8 trained arms (`u1_base..u4_rnd_iit`, `m1_base..m4_rnd_iit`) + `M0` + Wave 3.0 zero-retraining zone probe + held-out causal interchange + IFEval/GSM8K/Wiki KL completed cleanly (`8/8` health gates passed, `8/8` vLLM LoRA parity passed with `option_match = 96.25%–98.75%`, `du_abs = 0.177–0.247`).

*(Note on Phase-0 `M0` capability reference: exact Phase-0 GSM8K is **496/1319 correct** and 502 parsed; earlier C2 summary mistakenly listed 502 as correct and is reconciled here.)*

| Arm | `c3_blind` M0-Elig (`n=130`) | `c3_blind` Full (`960` units / `1,080` items) | Row B (`120`) | Turn-1 | `think` (`240`) | IFEval (`541`) | GSM8K (`1319`) | Wiki KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` | `0.00%` | `0.00%` | `0.0%` | `12.0%` | `91.8%` | `440`* | `496`* | `0.000000` |
| `u1_base` | **`81.54%`** | `65.68%` | `32.5%` | `94.8%` | **`98.8%`** | `443` | `538` | `0.001364` |
| `u2_corridor` | `78.46%` | **`66.15%`** | **`35.8%`** | `95.4%` | `97.8%` | **`449`** | **`547`** | `0.001362` |
| `u3_b1_iit` | `77.69%` | `63.49%` | `21.7%` | `87.4%` | `95.2%` | `446` | `535` | `0.001330` |
| `u4_rnd_iit` | `77.69%` | `62.08%` | `19.2%` | `86.9%` | `97.2%` | `440` | `541` | `0.001326` |
| `m1_base` | `70.00%` | `59.90%` | `20.0%` | `93.9%` | `96.8%` | `444` | `521` | `0.001235` |
| `m2_corridor` | `72.31%` | `63.70%` | `26.7%` | **`95.5%`** | `94.8%` | `438` | `529` | `0.001256` |
| `m3_b1_iit` | `73.08%` | `59.74%` | `16.7%` | `91.7%` | `96.2%` | `441` | `514` | `0.001253` |
| `m4_rnd_iit` | `75.38%` | `58.75%` | `16.7%` | `90.7%` | `96.0%` | `445` | `513` | `0.001229` |

*\*Phase-0 M0 capability reference (`experiments/ACL-exp-c1/results/phase0/capabilities/M0/summary.json`).*

---

## 2. Verified Key Findings & Astra Read-Only Audit Corrections

1. **Wave 3.0 Zero-Retraining Zone Probe**: On C2's `unif_base`, disabling Zone 3 (`L28–35`) kept Row B at `31/36 (86.1%)` with zero drop, while disabling Zone 1 (`L0–13`) collapsed Row B to `8/36` and disabling Zone 2 (`L14–27`) collapsed it to `0/36`. Next step (C3.5 Stage 1) must separate `Zone-1 Q/O` vs `Zone-1 MLP` and measure the exact Wiki KL delta of disabling `m2_corridor`'s 3 top MLPs `[29, 32, 35]`.
2. **Corridor + Pure Head Localization Signals**: On full 960 generation units, `m2_corridor − m1_base = +3.80 pp [+2.45, +5.21] (p<0.0001)` and `u3_b1_iit − u4_rnd_iit = +1.41 pp [+0.42, +2.45] (p=0.0076, fixed 5 B1 MLPs)`. However, hard un-gated IIT from Step 1 degraded both architectures (`u3 − u2 = -2.66 pp`, `m3 − m2 = -3.96 pp`) and hurt joint Turn-1 + 4-cell Row B success (`u2: 40/120 → u3: 14/120`).
3. **Held-Out Interchange & Next Mechanistic Challenge**: On `n=60` held-out pairs (`identity = 100%, max_du = 0.0`), B1 patching yielded `94.4%` (`168/178`) counterfactual 4-option flips on `u1_base` vs `78.6%` (`rnd_head + b1_mlp`) vs `9.6%` (`rnd_full`), and `97.1%` (`165/170`) on `u3_b1_iit` (on the shared `152` jointly-correct pairs, `u1` vs `u3` was `149/152` vs `150/152`). C3.5 Stage 1 adds the **Cross-Letter Remapped Interchange Challenge** (disjoint A/B/C/D receptor permutation + `MLP-only / heads-only / B1-heads+rnd-MLP` across BOTH `context_flip` and `argument_flip`) to distinguish abstract validity transfer from literal answer-letter copying!
4. **Bugs Caught for `ACL-exp-c3.5`**:
   - `c3_build_datasets.py` line 866: CE Style-C decoy clause waited for `Question:` while CE prompt began with `Based only ...`, dropping the decoy on all 120 `before_rules` CE items and keeping it on all 120 `after_rules` CE items;
   - `c3_build_datasets.py` line 434: `s1 ∧ s2` synthesis was redundant because `s2` already required `s1` — fixed in `c3_blind_confirm_v2` via two truly independent premise branches (solver-asserting that removing either branch independently flips gold!);
   - `c3_stats.py` line 265: tighten missing-value health/control checks so `None` never silently passes, and add joint `Turn-1 dual-correct ∧ 4-cell all-correct` headline column;
   - `c3_train.py` line 893: log separate main-task vs IIT gradient norms and their cosine angle on the exact same batch BEFORE independent `0.30×` IIT grad-norm capping and global clipping!

---

## 3. Handoff to `ACL-exp-c3.5`

`ACL-exp-c3` is now **permanently frozen**. `ACL-exp-c3.5` executes in two compact stages on `Qwen3-8B` (reusing C3's training data and existing `u1_base / u2_corridor / m2_corridor` checkpoints):
- **Stage 1 (Zero-Retraining Probes + Fixed `c3_blind_confirm_v2`, ~15 min)**: Sub-projection Q/O vs MLP + KL probe, Cross-Letter Remapped 7-way Causal Interchange challenge, and zero-retraining baseline evaluation of `M0, u1_base, u2_corridor, m2_corridor` on `c3_blind_confirm_v2`.
- **Stage 2 (5 Targeted New Arms `A, B, C, D, E`)**: Budget locked strictly by Stage 1 Probe 1 + `c3_dev`, testing Zone-1 optimization (`A`), `M3-v2` budget reallocation (`B`), Soft-Gated B1-IIT (`C`), Soft-Gated Random-Head IIT (`D`), and Uniform + Soft-Gated B1-IIT (`E`).
