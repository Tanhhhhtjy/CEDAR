# ACL-exp-c3.5 Readout (`c3_blind_confirm_v2`, `900` Independent Generation Units / `1,080` Items)

- **Phase-0 M0 Capability Anchor**: `IFEval = 440/541`, `GSM8K = 496/1319 correct` (`502` parsed)
- **Strongest Healthy Uniform Control**: `u2_corridor`

## 1. Main Table (Full Blind v2, Unit-Weighted 4-Cell AND Joint Turn-1 ∧ 4-Cell)

| Arm | 4-Cell (UW) | **Joint T1∧4C (UW)** | `DC_orig / DC_rev` | Row B (4C / Joint) | Turn-1 | NLL | IFEval / GSM8K | Wiki KL | Health |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `M0` | **0.00%** | **0.00%** | 0.0% / 0.0% | 0.0% / 0.0% | 32.9% | 13.784 | — / — | 0.000000 | FAIL(2) |
| `u1_base` | **46.39%** | **45.61%** | 47.7% / 55.6% | 15.0% / 15.0% | 96.9% | 0.894 | — / — | — | FAIL(1) |
| `u2_corridor` | **50.56%** | **49.89%** | 51.6% / 57.3% | 15.8% / 15.8% | 97.0% | 0.783 | — / — | — | FAIL(1) |
| `m2_corridor` | **45.61%** | **44.94%** | 47.7% / 52.8% | 10.8% / 10.8% | 97.6% | 0.764 | — / — | — | FAIL(1) |

## 2. Paired Unit-Clustered Comparisons (`4-Cell` & `Joint T1∧4C`)

| Comparison | 4-Cell Diff (pp) [95% CI] | Joint T1∧4C Diff (pp) [95% CI] |
|---|---|---|
| `u2_vs_u1` | **+4.17** `[2.94, 5.5]` (p=0.0000) | **+4.28** `[3.0, 5.67]` (p=0.0000) |
| `m2_vs_u2` | **-4.94** `[-6.78, -3.28]` (p=0.0000) | **-4.94** `[-6.78, -3.28]` (p=0.0000) |
