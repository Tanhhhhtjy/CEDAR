# ACL-exp-c3 Showdown Readout (`c3_blind_confirm`)

- **Frozen M0 Dual-Eligible Denominator**: `130` bases / `130` independent families
- **Strongest Healthy Same-Data Uniform Control**: `u1_base`
- **Headline Verdict (`m3_b1_iit`)**: **`HOLD`**

## 1. Main 2×4 Table (`no_think`, M0 Dual-Eligible, Family-Weighted 4-Cell All-Correct)

| Arm | 4-Cell (FW) | `DC(V_hes,F_auth)` | `DC(V_auth,F_hes)` | Step 1/2/3 (4-Cell) | Row B (4-Cell) | 4-Opt NLL | IFEval | GSM8K | Wiki KL | Health |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `M0` | **0.00%** | 0.0% | 0.8% | 0.0%/0.0%/0.0% | — | 11.848 | — | — | 0.000000 | FAIL |
| `u1_base` | **81.54%** | 82.3% | 85.4% | 87.1%/76.9%/80.9% | — | 0.361 | 443 | 538 | 0.001364 | PASS |
| `u2_corridor` | **78.46%** | 79.2% | 81.5% | 83.9%/73.1%/76.5% | — | 0.407 | 449 | 547 | 0.001362 | PASS |
| `u3_b1_iit` | **77.69%** | 78.5% | 81.5% | 87.1%/69.2%/77.9% | — | 0.340 | 446 | 535 | 0.001330 | PASS |
| `u4_rnd_iit` | **77.69%** | 79.2% | 80.0% | 83.9%/69.2%/79.4% | — | 0.351 | 440 | 541 | 0.001326 | PASS |
| `m1_base` | **70.00%** | 70.0% | 83.1% | 71.0%/65.4%/72.1% | — | 0.425 | 444 | 521 | 0.001235 | PASS |
| `m2_corridor` | **72.31%** | 74.6% | 76.1% | 74.2%/69.2%/73.5% | — | 0.375 | 438 | 529 | 0.001256 | PASS |
| `m3_b1_iit` | **73.08%** | 77.7% | 80.8% | 77.4%/69.2%/73.5% | — | 0.335 | 441 | 514 | 0.001253 | PASS |
| `m4_rnd_iit` | **75.38%** | 77.7% | 83.9% | 77.4%/69.2%/77.9% | — | 0.281 | 445 | 513 | 0.001229 | PASS |

## 2. Key Paired Family-Clustered Comparisons (Primary 4-Cell Endpoint)

| Comparison | Mean A | Mean B | Diff (pp) | 95% CI (pp) | Two-Sided p | Wins A : B : Tie |
|---|---:|---:|---:|---:|---:|---|
| `u3_b1_iit_vs_u4_rnd_iit` | 77.69% | 77.69% | **+0.00** | [-2.31, +2.31] | 1.0000 | 1 : 1 : 128 |
| `u3_b1_iit_vs_u2_corridor` | 77.69% | 78.46% | **-0.77** | [-3.85, +2.31] | 0.8236 | 2 : 3 : 125 |
| `u2_corridor_vs_u1_base` | 78.46% | 81.54% | **-3.08** | [-6.92, +0.00] | 0.1108 | 1 : 5 : 124 |
| `m3_b1_iit_vs_m4_rnd_iit` | 73.08% | 75.38% | **-2.31** | [-6.15, +1.54] | 0.3348 | 2 : 5 : 123 |
| `m3_b1_iit_vs_m2_corridor` | 73.08% | 72.31% | **+0.77** | [-2.31, +3.85] | 0.8368 | 3 : 2 : 125 |
| `m2_corridor_vs_m1_base` | 72.31% | 70.00% | **+2.31** | [-0.77, +5.38] | 0.2400 | 4 : 1 : 125 |
| `m3_b1_iit_vs_u3_b1_iit` | 73.08% | 77.69% | **-4.62** | [-9.23, +0.00] | 0.0660 | 2 : 8 : 120 |
| `m3_b1_iit_vs_u1_base` | 73.08% | 81.54% | **-8.46** | [-14.62, -3.08] | 0.0032 | 3 : 14 : 113 |
| `m2_corridor_vs_u1_base` | 72.31% | 81.54% | **-9.23** | [-15.38, -3.85] | 0.0016 | 2 : 14 : 114 |

## 3. Dual-Path Pre-Registered Verdicts

- **`m3_b1_iit` vs `u1_base`**: `HOLD` (diff=`-8.46 pp`, 95% CI low=`-14.62 pp`, KL reduction=`8.14%`, health=`True`)
- **`u3_b1_iit` vs `u1_base`**: `HOLD` (diff=`-3.85 pp`, 95% CI low=`-7.69 pp`, KL reduction=`2.49%`, health=`True`)
